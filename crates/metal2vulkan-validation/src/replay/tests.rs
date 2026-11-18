//! The stream format, the address map and the player's order, on a backend that only keeps what it
//! is asked (no device). The runtime backends' tests are device tests (`runtime_lane`).

use super::addresses::AddressMap;
use super::player::{
    Backend, Counters, Counts, IdDiffers, Made, PlayError, Player, PrivateBytes, Relaid, Report,
    WAIT_COMPLETED,
};
use super::stream::{segments, Call, StreamError, StreamReader, StreamWriter};
use super::value::{hex, Value};
use crate::ScratchDir;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

/// A stream `build` writes into a file of `dir`, named `name`.
fn written(
    dir: &ScratchDir,
    name: &str,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> PathBuf {
    let path = dir.path().join(name);
    let mut writer = StreamWriter::new(File::create(&path).unwrap()).unwrap();
    build(&mut writer).unwrap();
    writer.finish().unwrap();
    path
}

fn calls(reader: &mut StreamReader) -> Result<Vec<Call>, StreamError> {
    let mut all = Vec::new();
    while let Some(call) = reader.next_call()? {
        all.push(call);
    }
    Ok(all)
}

fn map(entries: &[(&str, Value)]) -> Value {
    Value::Map(
        entries
            .iter()
            .map(|(k, v)| (k.to_string().into(), v.clone()))
            .collect(),
    )
}

/// Every kind of value (object 0, a property set to nil, among them), a map key and a selector used
/// twice and a blob written twice come back as written; the names and the blob are each written
/// once.
#[test]
fn a_stream_reads_back_what_was_written() {
    let dir = ScratchDir::new("replay-round-trip").unwrap();
    let data = vec![7u8; 100];
    let all = Value::List(vec![
        Value::Nil,
        Value::Bool(true),
        Value::Bool(false),
        Value::Uint(u64::MAX),
        Value::Sint(-3),
        Value::Sint(i64::MIN),
        Value::F32(1.5),
        Value::F64(-0.25),
        Value::Object(9),
        Value::Object(0),
        Value::Address(1 << 46),
        Value::Str("kernel_main".into()),
        Value::Bytes(vec![1, 2, 3]),
        map(&[("width", Value::Uint(4)), ("height", Value::Uint(2))]),
    ]);
    let mut blobs = Vec::new();
    let path = written(&dir, "s.m2vr", |w| {
        for _ in 0..2 {
            let blob = w.blob(&data)?;
            blobs.push(blob);
            let args = [all.clone(), Value::Blob(blob)];
            w.call(
                "setBytes:length:atIndex:",
                3,
                0,
                &args,
                &[("width", Value::Uint(1))],
            )?;
        }
        Ok(())
    });
    let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
    let read = calls(&mut reader).unwrap();
    assert_eq!(read.len(), 2);
    assert_eq!(read[0], read[1]);
    assert_eq!(&*read[0].selector, "setBytes:length:atIndex:");
    assert_eq!((read[0].receiver, read[0].result), (3, 0));
    assert_eq!(read[0].args[0], all);
    assert_eq!(read[0].answer("width"), Some(&Value::Uint(1)));
    assert_eq!(reader.blob(&blobs[1]).unwrap(), data);
    assert_eq!(blobs[0], <[u8; 32]>::from(Sha256::digest(&data)));
    let bytes = std::fs::read(&path).unwrap();
    let count = |needle: &[u8]| bytes.windows(needle.len()).filter(|w| *w == needle).count();
    assert_eq!(
        count(b"setBytes:length:atIndex:"),
        1,
        "a name is written once"
    );
    assert_eq!(count(&data), 1, "a blob is written once");
    assert!(reader.truncated().is_none());
}

/// A record cut short at the end of the last segment ends the stream there, and everything before
/// it reads; in a segment a later one follows, it is an error. A later segment reads the names and
/// blobs of an earlier one.
#[test]
fn segments_read_in_order_and_a_cut_tail_ends_the_stream() {
    let dir = ScratchDir::new("replay-segments").unwrap();
    let mut hash = [0u8; 32];
    let first = written(&dir, "s.0000.m2vr", |w| {
        hash = w.blob(b"library")?;
        w.call("newLibraryWithData:error:", 1, 2, &[Value::Blob(hash)], &[])
    });
    // The second segment, written on as one writer would: its first record (the name the first
    // segment defined already) dropped, so its calls name 1 by the first segment's record.
    let second = written(&dir, "s.0001.m2vr", |w| {
        w.call("newLibraryWithData:error:", 1, 3, &[], &[])?;
        w.call("commit", 4, 0, &[], &[])
    });
    let mut records = std::fs::read(&second).unwrap();
    let name = b"newLibraryWithData:error:";
    assert_eq!(records[16..18], [1, name.len() as u8]);
    records.drain(16..18 + name.len());
    std::fs::write(&second, &records).unwrap();
    let found = segments(&first);
    assert_eq!(found, vec![first.clone(), second.clone()]);
    let mut reader = StreamReader::open(&found, None).unwrap();
    let read = calls(&mut reader).unwrap();
    let selectors: Vec<&str> = read.iter().map(|c| &*c.selector).collect();
    assert_eq!(
        selectors,
        [
            "newLibraryWithData:error:",
            "newLibraryWithData:error:",
            "commit"
        ]
    );
    assert_eq!(reader.blob(&hash).unwrap(), b"library");

    // The last segment cut in its final record: the calls before it read.
    let mut tail = records.clone();
    tail.truncate(tail.len() - 2);
    std::fs::write(&second, &tail).unwrap();
    let mut reader = StreamReader::open(&found, None).unwrap();
    assert_eq!(calls(&mut reader).unwrap().len(), 2);
    assert_eq!(
        reader.truncated().map(|(p, _)| p.clone()),
        Some(second.clone())
    );

    // A cut first segment, with the second after it: an error.
    let mut cut = std::fs::read(&first).unwrap();
    cut.truncate(cut.len() - 1);
    std::fs::write(&first, &cut).unwrap();
    let mut reader = StreamReader::open(&found, None).unwrap();
    assert!(matches!(calls(&mut reader), Err(StreamError::Cut { .. })));
}

/// A record of a kind the format does not name is an error; a header of another version is
/// refused.
#[test]
fn unknown_kinds_and_versions_are_refused() {
    let dir = ScratchDir::new("replay-unknown").unwrap();
    let path = written(&dir, "s.m2vr", |w| w.call("commit", 1, 0, &[], &[]));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(&[9, 0]);
    std::fs::write(&path, &bytes).unwrap();
    let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
    assert!(matches!(
        calls(&mut reader),
        Err(StreamError::UnknownKind { kind: 9, .. })
    ));
    bytes[8] = 1;
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        StreamReader::open(&[path], None),
        Err(StreamError::Header(_))
    ));
}

/// A blob no record holds is read from the blob directory, by its hex hash, and only where its
/// bytes hash to it; one neither holds is an error naming it.
#[test]
fn a_blob_left_out_is_read_from_the_blob_directory() {
    let dir = ScratchDir::new("replay-blobs").unwrap();
    let data = b"a metallib's bytes".to_vec();
    let hash: [u8; 32] = Sha256::digest(&data).into();
    let path = written(&dir, "s.m2vr", |w| {
        w.call("newLibraryWithData:error:", 1, 2, &[Value::Blob(hash)], &[])
    });
    let blobs = dir.path().join("blobs");
    std::fs::create_dir(&blobs).unwrap();
    let mut reader = StreamReader::open(std::slice::from_ref(&path), Some(blobs.clone())).unwrap();
    assert!(matches!(
        reader.blob(&hash),
        Err(StreamError::MissingBlob(_))
    ));
    std::fs::write(blobs.join(hex(&hash)), b"other bytes").unwrap();
    assert!(matches!(reader.blob(&hash), Err(StreamError::WrongBlob(_))));
    std::fs::write(blobs.join(hex(&hash)), &data).unwrap();
    assert_eq!(reader.blob(&hash).unwrap(), data);
}

/// Ranges nest: a word in a buffer inside a block moves with the buffer, one in the block alone
/// with the block, one past every range stays. A removed block leaves its buffers.
#[test]
fn the_address_map_relocates_by_the_innermost_live_range() {
    let block = 1u64 << 46;
    let mut map = AddressMap::default();
    map.insert(1, block, 0x1_0000, 0x10_0000).unwrap();
    map.insert(2, block + 0x100, 0x100, 0x90_0000).unwrap();
    // A buffer of the block's own bounds: the inner one.
    map.insert(3, block, 0x1_0000, 0x10_0000).unwrap();
    assert_eq!(map.relocated(block + 0x10), Some(0x10_0010));
    assert_eq!(map.relocated(block + 0x108), Some(0x90_0008));
    assert_eq!(map.relocated(block + 0x1_0000), None);
    assert!(
        map.insert(2, block + 0x400, 8, 0).is_err(),
        "#2 has a range"
    );
    map.remove(1);
    map.remove(3);
    assert!(!map.holds(1));
    assert_eq!(map.relocated(block + 0x10), None);
    assert_eq!(map.relocated(block + 0x108), Some(0x90_0008));

    // Words at multiples of 8 from the target's start move; a value 4 past one is counted and
    // stays; bytes before the first such word are no word.
    let mut bytes = vec![0u8; 4];
    bytes.extend_from_slice(&(block + 0x100).to_le_bytes());
    bytes.extend_from_slice(&[0u8; 4]);
    bytes.extend_from_slice(&(block + 0x110).to_le_bytes());
    // `bytes` starts 4 bytes into its target: the first address is at the target's offset 8, the
    // second at its offset 20.
    let (moved, misaligned) = map.relocate(&mut bytes, 4);
    assert_eq!((moved, misaligned), (1, 1));
    assert_eq!(&bytes[4..12], &0x90_0000u64.to_le_bytes());
    assert_eq!(
        &bytes[16..24],
        &(block + 0x110).to_le_bytes(),
        "4 past 8 stays"
    );
}

/// A range that meets a live one in part ends that one, whose inner ranges stay: the program put
/// the newer object in that memory while the older lived on, so a word there relocates by the newer
/// one.
#[test]
fn a_range_that_meets_a_live_one_in_part_ends_it() {
    let block = 1u64 << 46;
    let mut map = AddressMap::default();
    map.insert(1, block, 0x1_0000, 0x10_0000).unwrap();
    map.insert(2, block + 0x100, 0x100, 0x90_0000).unwrap();
    assert_eq!(map.insert(3, block + 0x180, 0x100, 0xa0_0000), Ok(1));
    assert!(!map.holds(2));
    assert_eq!(map.relocated(block + 0x188), Some(0xa0_0008));
    assert_eq!(
        map.relocated(block + 0x108),
        Some(0x10_0108),
        "the block's again"
    );
    // One that meets the block itself in part ends the block; its buffer stays.
    assert_eq!(map.insert(4, block - 8, 0x10, 0xb0_0000), Ok(1));
    assert!(!map.holds(1));
    assert_eq!(map.relocated(block - 8), Some(0xb0_0000));
    assert_eq!(map.relocated(block + 0x188), Some(0xa0_0008));
}

/// A backend that keeps what it is asked: the calls, the waits, the bytes written, the objects
/// whose resource IDs it gave. It places blocks and buffers apart from the recorded addresses, and
/// a buffer in a block at its offset. It holds the bytes of the buffers made by length, which
/// writes and buffer copies change and a read gives; it runs no dispatch. It counts submissions,
/// and buffer copies once there was one.
#[derive(Default)]
struct Kept {
    calls: Vec<String>,
    /// Each call's arguments.
    args: Vec<Vec<Value>>,
    waits: Vec<u64>,
    writes: Vec<(u64, u64, Vec<u8>)>,
    bases: HashMap<u64, u64>,
    next: u64,
    commits: u64,
    /// The resource IDs it gives; an object not here has none (its getter raises).
    ids: HashMap<u64, u64>,
    asked: Vec<u64>,
    memory: HashMap<u64, Vec<u8>>,
    copies: u64,
    /// The size and alignment it answers for a heap texture, by the id of its descriptor; a heap
    /// buffer it answers its length at 256 bytes.
    heap_answers: HashMap<u64, (u64, u64)>,
    /// Each heap size it was asked: the placing selector and how many calls made and set the
    /// texture descriptor.
    heap_asked: Vec<(String, usize)>,
    /// Whether it declines the player's snapshots.
    declines_snapshots: bool,
    /// The GPU start and end it gives each command buffer; one not here has none.
    gpu: HashMap<u64, (u64, u64)>,
}

impl Backend for Kept {
    fn call(&mut self, call: &Call) -> Result<Made, PlayError> {
        self.calls.push(call.to_string());
        self.args.push(call.args.clone());
        let mut made = Made::default();
        match &*call.selector {
            "m2v.newBlockWithLength:storageMode:"
            | "newBufferWithLength:options:"
            | "newBufferWithDescriptor:" => {
                self.next += 0x1000_0000;
                made.address = Some(self.next);
            }
            "m2v.newBufferInBlock:offset:length:options:" => {
                let block = call.args[0].as_object().unwrap();
                made.address = Some(self.bases[&block] + call.args[1].as_uint().unwrap());
            }
            "commit" => self.commits += 1,
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:" => {
                let at = |at: usize| call.args[at].as_uint().unwrap() as usize;
                let (from, to) = (call.args[0].as_object(), call.args[2].as_object());
                let bytes = &self.memory[&from.unwrap()][at(1)..at(1) + at(4)];
                let bytes = bytes.to_vec();
                let into = self.memory.get_mut(&to.unwrap()).unwrap();
                into[at(3)..at(3) + at(4)].copy_from_slice(&bytes);
                self.copies += 1;
            }
            "notPlayed" => {
                return Err(PlayError::Refused {
                    selector: call.selector.to_string(),
                    why: "kept refuses it".into(),
                })
            }
            _ => {}
        }
        if let Some(address) = made.address {
            self.bases.insert(call.result, address);
        }
        if &*call.selector == "newBufferWithLength:options:" {
            let length = call.args[0].as_uint().unwrap() as usize;
            self.memory.insert(call.result, vec![0; length]);
        }
        Ok(made)
    }

    fn write(&mut self, target: u64, offset: u64, bytes: &[u8]) -> Result<(), PlayError> {
        if let Some(memory) = self.memory.get_mut(&target) {
            let offset = offset as usize;
            memory[offset..offset + bytes.len()].copy_from_slice(bytes);
        }
        self.writes.push((target, offset, bytes.to_vec()));
        Ok(())
    }

    fn wait(&mut self, command_buffer: u64) -> Result<(), PlayError> {
        self.waits.push(command_buffer);
        Ok(())
    }

    fn gpu_interval(&mut self, command_buffer: u64) -> Option<(u64, u64)> {
        self.gpu.get(&command_buffer).copied()
    }

    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError> {
        let memory = self.memory.get(&object).cloned();
        Ok(memory.unwrap_or_else(|| object.to_le_bytes().to_vec()))
    }

    fn resource_id(&mut self, object: u64) -> Result<Option<u64>, PlayError> {
        self.asked.push(object);
        Ok(self.ids.get(&object).copied())
    }

    fn heap_size_and_align(
        &mut self,
        call: &Call,
        descriptor: &[Call],
    ) -> Result<(u64, u64), PlayError> {
        self.heap_asked
            .push((call.selector.to_string(), descriptor.len()));
        Ok(match descriptor.first() {
            Some(made) => self.heap_answers[&made.result],
            None => (call.args[0].as_uint().unwrap(), 256),
        })
    }

    fn counters(&mut self) -> Counters {
        let mut counters = Counters::from([("submissions", self.commits)]);
        if self.copies > 0 {
            counters.insert("copies", self.copies);
        }
        counters
    }

    fn lines(&self) -> Vec<String> {
        Vec::new()
    }

    fn live(&mut self) -> Counters {
        Counters::from([("calls", self.calls.len() as u64)])
    }

    fn snapshots(&self) -> bool {
        !self.declines_snapshots
    }
}

/// `build`'s stream, played on `backend`.
fn played(
    backend: &mut Kept,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> (Report, Result<(), PlayError>) {
    let dir = ScratchDir::new("replay-player").unwrap();
    let path = written(&dir, "s.m2vr", build);
    let mut reader = StreamReader::open(&[path], None).unwrap();
    Player::new(backend).play(&mut reader)
}

const BLOCK: u64 = 1 << 46;

fn addressed(address: u64, length: u64) -> [(&'static str, Value); 2] {
    [
        ("gpuAddress", Value::Address(address)),
        ("gpuLength", Value::Uint(length)),
    ]
}

/// A buffer whose recorded range meets a live buffer's in part, as a guest's allocator reuses
/// block memory before the older buffer's release, ends the older range; the report counts it.
#[test]
fn a_range_met_in_part_is_counted() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let block = [Value::Uint(0x1_0000), Value::Uint(0)];
        let made = addressed(BLOCK, 0x1_0000);
        w.call("m2v.newBlockWithLength:storageMode:", 1, 2, &block, &made)?;
        for (id, offset) in [(3, 0x100), (4, 0x180)] {
            let inner = [
                Value::Object(2),
                Value::Uint(offset),
                Value::Uint(0x100),
                Value::Uint(0),
            ];
            let at = addressed(BLOCK + offset, 0x100);
            w.call(
                "m2v.newBufferInBlock:offset:length:options:",
                2,
                id,
                &inner,
                &at,
            )?;
        }
        Ok(())
    });
    ended.unwrap();
    assert_eq!(report.parted, 1);
    assert!(report.lines().contains(&"parted 1".to_string()));
}

/// A write into a block and a constant upload carry recorded addresses: the player rewrites them
/// to the backend's, counts them, and leaves an address 4 past a multiple of 8 counted apart; a
/// write into a buffer of no address is not touched.
#[test]
fn uploads_are_relocated_and_counted() {
    let mut kept = Kept::default();
    let pointer = (BLOCK + 0x100 + 8).to_le_bytes();
    let mut misaligned = vec![0u8; 4];
    misaligned.extend_from_slice(&pointer);
    misaligned.extend_from_slice(&[0u8; 4]);
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let block = [Value::Uint(0x1_0000), Value::Uint(0)];
        w.call(
            "m2v.newBlockWithLength:storageMode:",
            1,
            2,
            &block,
            &addressed(BLOCK, 0x1_0000),
        )?;
        let inner = [
            Value::Object(2),
            Value::Uint(0x100),
            Value::Uint(0x100),
            Value::Uint(0),
        ];
        let at = addressed(BLOCK + 0x100, 0x100);
        w.call(
            "m2v.newBufferInBlock:offset:length:options:",
            2,
            3,
            &inner,
            &at,
        )?;
        w.call(
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            1,
            4,
            &[],
            &[],
        )?;
        let pairs = [
            Value::Uint(0),
            Value::Bytes(pointer.to_vec()),
            Value::Uint(16),
            Value::Bytes(misaligned.clone()),
        ];
        w.call("m2v.write", 2, 0, &pairs, &[])?;
        w.call(
            "m2v.write",
            4,
            0,
            &[Value::Uint(0), Value::Bytes(pointer.to_vec())],
            &[],
        )?;
        let bytes = [
            Value::Bytes(pointer.to_vec()),
            Value::Uint(8),
            Value::Uint(0),
        ];
        w.call("setBytes:length:atIndex:", 9, 0, &bytes, &[])
    });
    ended.unwrap();
    let moved = (0x1000_0000u64 + 0x100 + 8).to_le_bytes();
    assert_eq!(kept.writes[0], (2, 0, moved.to_vec()));
    assert_eq!(kept.writes[1], (2, 16, misaligned), "4 past 8 stays");
    assert_eq!(
        kept.writes[2],
        (4, 0, pointer.to_vec()),
        "no address: untouched"
    );
    let constants = kept.args.last().and_then(|args| args.first());
    assert_eq!(constants, Some(&Value::Bytes(moved.to_vec())));
    assert_eq!(report.total.stream["relocated_words"], 2);
    assert_eq!(report.total.stream["misaligned_address_words"], 1);
    assert_eq!(report.total.stream["written_bytes"], 8 + 16 + 8);
    assert_eq!(
        report.total.stream["unrelocated_writes"], 1,
        "the write of no address"
    );
}

/// A copy from the guest's RAM lands each piece at its byte: into a block, an address is
/// relocated at a multiple of 8 and counted apart 4 past one; into a buffer of no address, nothing
/// moves. A copy of another form (one the platform refused, recorded before the recorder left
/// those out) stops the play.
#[test]
fn guest_ram_copies_are_relocated_where_their_buffer_has_an_address() {
    let pointer = (BLOCK + 0x10).to_le_bytes();
    let moved = (0x1000_0000u64 + 0x10).to_le_bytes();
    let piece = |at: u64, data: Value| Value::List(vec![Value::Uint(at), data]);
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let block = [Value::Uint(0x1_0000), Value::Uint(0)];
        let at = addressed(BLOCK, 0x1_0000);
        w.call("m2v.newBlockWithLength:storageMode:", 1, 2, &block, &at)?;
        w.call(
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            1,
            3,
            &[],
            &[],
        )?;
        let data = w.blob(&[pointer.to_vec(), vec![0; 8]].concat())?;
        let pieces = Value::List(vec![piece(8, Value::Blob(data))]);
        w.call(
            "m2v.copyFromGuestRam",
            9,
            0,
            &[Value::Object(2), pieces],
            &[],
        )?;
        let pieces = Value::List(vec![
            piece(0x10, Value::Bytes(pointer.to_vec())),
            piece(0x24, Value::Bytes(pointer.to_vec())),
        ]);
        w.call(
            "m2v.copyFromGuestRamAt",
            9,
            0,
            &[Value::Object(2), pieces],
            &[],
        )?;
        let pieces = Value::List(vec![piece(0, Value::Bytes(pointer.to_vec()))]);
        w.call(
            "m2v.copyFromGuestRam",
            9,
            0,
            &[Value::Object(3), pieces],
            &[],
        )
    });
    ended.unwrap();
    let data = |call: usize, entry: usize| match &kept.args[call][1] {
        Value::List(pieces) => pieces[entry].as_list().unwrap()[1].clone(),
        other => panic!("{other}"),
    };
    let copied = [moved.to_vec(), vec![0; 8]].concat();
    assert_eq!(data(3, 0), Value::Bytes(copied));
    assert_eq!(data(4, 0), Value::Bytes(moved.to_vec()));
    assert_eq!(data(4, 1), Value::Bytes(pointer.to_vec()), "4 past 8 stays");
    assert_eq!(
        data(5, 0),
        Value::Bytes(pointer.to_vec()),
        "no address: untouched"
    );
    assert_eq!(report.total.stream["relocated_words"], 2);
    assert_eq!(report.total.stream["misaligned_address_words"], 1);
    assert_eq!(report.total.stream["written_bytes"], 16 + 16 + 8);
    assert_eq!(
        report.total.stream["unrelocated_writes"], 1,
        "the copy of no address"
    );

    let (_, ended) = played(&mut Kept::default(), |w| {
        let runs = Value::List(vec![Value::List(vec![
            Value::Uint(0x7000),
            Value::Uint(0x7008),
        ])]);
        let args = [runs, Value::Object(2), Value::Uint(0)];
        w.call("m2v.copyFromGuestRam", 9, 0, &args, &[])
    });
    assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");
}

/// The host's plumbing plays as the plain Metal call it stands for: a padded, exportable or
/// dma-buf buffer as `newBufferWithLength:options:`, none where the host made none, and a queue
/// kept apart as `newCommandQueue`. The records of the host's own scheduling reach no backend, and
/// a split or part the host did not cut (result 0) neither commits nor makes a part.
#[test]
fn host_plumbing_plays_as_the_plain_call() {
    let mut kept = Kept::default();
    let (_, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let padded = [Value::Uint(64), Value::Uint(0), Value::Nil];
        w.call("m2v.newBufferOfLengthPaddingUnobserved", 1, 2, &padded, &[])?;
        let exported = [Value::Uint(64), Value::Uint(0), Value::Uint(256)];
        w.call("m2v.newExportableBufferOfLength", 1, 0, &exported, &[])?;
        w.call(
            "m2v.newDmaBufBufferOfLengthPaddingUnobserved",
            1,
            3,
            &exported,
            &[],
        )?;
        w.call("m2v.newCommandQueueUnshared", 1, 4, &[], &[])?;
        w.call("commandBuffer", 4, 5, &[], &[])?;
        w.call(
            "m2v.setCompletionJoinsNext",
            5,
            0,
            &[Value::Bool(true)],
            &[],
        )?;
        w.call("m2v.holdsNoPredication", 5, 0, &[], &[])?;
        w.call("m2v.rebindNoCopy", 2, 0, &[Value::Nil, Value::Nil], &[])?;
        w.call("m2v.split", 5, 0, &[], &[])?;
        w.call("m2v.part", 5, 0, &[], &[])?;
        w.call("commit", 5, 0, &[], &[])
    });
    ended.unwrap();
    assert_eq!(
        kept.calls,
        [
            "#1 = MTLCreateSystemDefaultDevice",
            "#2 = #1 newBufferWithLength:options: 64 0",
            "#3 = #1 newBufferWithLength:options: 64 0",
            "#4 = #1 newCommandQueue",
            "#5 = #4 commandBuffer",
            "#5 commit",
        ]
    );
    assert_eq!(kept.waits, [5]);
}

/// A command buffer that waits on an event the CPU signals later is not waited for until then; a
/// later one on another queue is waited for at its commit. Each group's counters hold its own
/// commits.
#[test]
fn a_command_buffer_held_on_an_event_does_not_hold_the_others() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("newCommandQueue", 1, 3, &[], &[])?;
        w.call("newSharedEvent", 1, 4, &[], &[])?;
        w.call("commandBuffer", 2, 5, &[], &[])?;
        w.call(
            "encodeWaitForEvent:value:",
            5,
            0,
            &[Value::Object(4), Value::Uint(1)],
            &[],
        )?;
        w.call("commit", 5, 0, &[], &[])?;
        w.call("commandBuffer", 2, 6, &[], &[])?;
        w.call("commit", 6, 0, &[], &[])?;
        w.call("commandBuffer", 3, 7, &[], &[])?;
        w.call("commit", 7, 0, &[], &[])?;
        w.call("setSignaledValue:", 4, 0, &[Value::Uint(1)], &[])?;
        w.call(WAIT_COMPLETED, 5, 0, &[], &[])
    });
    ended.unwrap();
    assert_eq!(kept.waits, [7, 5, 6], "#6 waits behind #5 on its queue");
    let groups: Vec<_> = report
        .completed
        .iter()
        .map(|(ids, c)| (ids.clone(), c.executed["submissions"]))
        .collect();
    assert_eq!(groups, [(vec![7], 3), (vec![5, 6], 0)]);
}

/// The CPU waiting for a command buffer that only a later record lets run, a stream that ends with
/// one held, and a selector the backend refuses each stop the play, the calls played counted; a
/// check reads what it names.
#[test]
fn a_play_stops_where_the_stream_or_the_backend_cannot_go_on() {
    let held = |w: &mut StreamWriter<File>| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("newEvent", 1, 3, &[], &[])?;
        w.call("commandBuffer", 2, 4, &[], &[])?;
        w.call(
            "encodeWaitForEvent:value:",
            4,
            0,
            &[Value::Object(3), Value::Uint(2)],
            &[],
        )?;
        w.call("commit", 4, 0, &[], &[])
    };
    let (_, ended) = played(&mut Kept::default(), |w| {
        held(w)?;
        w.call(WAIT_COMPLETED, 4, 0, &[], &[])
    });
    assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");
    let (_, ended) = played(&mut Kept::default(), held);
    assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");

    let (report, ended) = played(&mut Kept::default(), |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newTextureWithDescriptor:", 0, 2, &[map(&[])], &[])?;
        w.call("m2v.check", 0, 0, &[Value::Object(2)], &[])
    });
    ended.unwrap();
    assert_eq!(report.checks.len(), 1);

    let (report, ended) = played(&mut Kept::default(), |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("notPlayed", 1, 0, &[], &[])
    });
    assert!(matches!(ended, Err(PlayError::Refused { .. })), "{ended:?}");
    assert_eq!(
        report.total.stream["calls"], 1,
        "the refused call is the next"
    );
}

/// A present closes a frame of the groups since the last, reading nothing (its texture may be
/// nil); a released device releases what it made, last made first; the host's split commits its
/// command buffer and makes the next part on the same queue.
#[test]
fn frames_releases_and_splits() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("commandBuffer", 2, 3, &[], &[])?;
        w.call("m2v.split", 3, 4, &[], &[])?;
        w.call("commit", 4, 0, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        w.call("release", 1, 0, &[], &[])
    });
    ended.unwrap();
    assert_eq!(kept.waits, [3, 4]);
    let stream = Counters::from([
        ("calls", 4),
        ("commits", 2),
        ("dispatches", 0),
        ("draws", 0),
        ("encoders", 0),
        ("misaligned_address_words", 0),
        ("relocated_words", 0),
        ("written_bytes", 0),
        ("address_arguments", 0),
        ("unrelocated_writes", 0),
    ]);
    let executed = Counters::from([("submissions", 2)]);
    assert_eq!(report.frames, [Counts { stream, executed }]);
    assert!(report.checks.is_empty());
    let released: Vec<&str> = kept
        .calls
        .iter()
        .rev()
        .take(4)
        .map(String::as_str)
        .collect();
    assert_eq!(
        released,
        ["#1 release", "#2 release", "#3 release", "#4 release"]
    );
    assert!(kept.calls.contains(&"#4 = #2 commandBuffer".to_string()));
}

/// What a stream fixes, apart from what the runtime executed: its calls played, its commits, the
/// encoders its command buffers made and the draws and dispatches their calls issued, per group of
/// command buffers waited for and in total. The call that waited for a group counts in the next.
#[test]
fn the_stream_fixes_its_calls_commits_encoders_draws_and_dispatches() {
    let draw = "drawPrimitives:vertexStart:vertexCount:";
    let (report, ended) = played(&mut Kept::default(), |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("commandBuffer", 2, 3, &[], &[])?;
        w.call("computeCommandEncoder", 3, 4, &[], &[])?;
        w.call(
            "dispatchThreadgroups:threadsPerThreadgroup:",
            4,
            0,
            &[],
            &[],
        )?;
        w.call("endEncoding", 4, 0, &[], &[])?;
        w.call(
            "renderCommandEncoderWithDescriptor:",
            3,
            5,
            &[Value::Nil],
            &[],
        )?;
        w.call(draw, 5, 0, &[], &[])?;
        w.call(draw, 5, 0, &[], &[])?;
        w.call("endEncoding", 5, 0, &[], &[])?;
        w.call("commit", 3, 0, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])
    });
    ended.unwrap();
    let fixed = |calls| {
        Counters::from([
            ("calls", calls),
            ("commits", 1),
            ("dispatches", 1),
            ("draws", 2),
            ("encoders", 2),
            ("misaligned_address_words", 0),
            ("relocated_words", 0),
            ("written_bytes", 0),
            ("address_arguments", 0),
            ("unrelocated_writes", 0),
        ])
    };
    let executed = Counters::from([("submissions", 1)]);
    let group = Counts {
        stream: fixed(10),
        executed: executed.clone(),
    };
    assert_eq!(report.completed, [(vec![3], group.clone())]);
    assert_eq!(report.frames, [group]);
    let total = Counts {
        stream: fixed(12),
        executed,
    };
    assert_eq!(report.total, total);
    assert_eq!(
        report.lines()[0],
        "completed #3 stream calls=10 commits=1 dispatches=1 draws=2 encoders=2 \
         misaligned_address_words=0 relocated_words=0 written_bytes=0 executed submissions=1"
    );
}

/// `m2v.commitPart` commits its command buffer where its answer `cut` is "Cut" or absent, and
/// nothing where the host committed nothing ("Fails", `{Whole: reason}`); a command buffer kept
/// whole commits at its own `commit`. Any other answer stops the play.
#[test]
fn a_part_commits_only_where_the_host_cut_it() {
    let mut kept = Kept::default();
    let cut = |cut: &str| [("cut", Value::Str(cut.to_owned()))];
    let whole = [("cut", map(&[("Whole", Value::Str("Committed".to_owned()))]))];
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        for command_buffer in 3..=6 {
            w.call("commandBuffer", 2, command_buffer, &[], &[])?;
        }
        w.call("m2v.commitPart", 3, 0, &[], &cut("Cut"))?;
        w.call("m2v.commitPart", 4, 0, &[], &[])?;
        w.call("m2v.commitPart", 5, 0, &[], &cut("Fails"))?;
        w.call("m2v.commitPart", 6, 0, &[], &whole)?;
        w.call("commit", 6, 0, &[], &[])?;
        w.call("m2v.commitPart", 5, 0, &[], &[("cut", Value::Uint(1))])
    });
    match ended {
        Err(PlayError::Invalid(why)) => assert!(why.contains("m2v.commitPart's cut"), "{why}"),
        other => panic!("the play went on past an unknown cut: {other:?}"),
    }
    assert_eq!(kept.waits, [3, 4, 6]);
    assert_eq!(kept.commits, 3);
    assert_eq!(report.total.stream["commits"], 3);
}

/// A buffer made as a copy of host memory starts with its descriptor's `contents`, a blob or
/// inline bytes: the backend makes it with the contents nil, and the player writes them from its
/// first byte, relocated where it answered a `gpuAddress` (here a pointer into itself) and
/// counted; a buffer of no address is written as it is, and a no-copy buffer's nil contents
/// write nothing.
#[test]
fn a_copied_buffer_starts_with_its_contents() {
    let mut kept = Kept::default();
    let pointer = (BLOCK + 0x40).to_le_bytes();
    let mut long = vec![5u8; 128];
    long[8..16].copy_from_slice(&pointer);
    let descriptor = |length: u64, contents: Value, no_copy: bool| {
        let options = Value::List(Vec::new());
        let fields = [
            ("length", Value::Uint(length)),
            ("options", options),
            ("contents", contents),
            ("noCopy", Value::Bool(no_copy)),
        ];
        [map(&fields)]
    };
    let selector = "newBufferWithDescriptor:";
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let blob = Value::Blob(w.blob(&long)?);
        let at = addressed(BLOCK, 128);
        w.call(selector, 1, 2, &descriptor(128, blob, false), &at)?;
        let short = Value::Bytes(pointer.to_vec());
        w.call(selector, 1, 3, &descriptor(8, short, false), &[])?;
        w.call(selector, 1, 4, &descriptor(8, Value::Nil, true), &[])
    });
    ended.unwrap();
    for args in &kept.args[1..] {
        let contents = args[0]
            .as_map()
            .unwrap()
            .iter()
            .find(|(k, _)| &**k == "contents");
        assert_eq!(contents.map(|(_, v)| v), Some(&Value::Nil));
    }
    let mut moved = long.clone();
    moved[8..16].copy_from_slice(&(0x1000_0000u64 + 0x40).to_le_bytes());
    let untouched = (3, 0, pointer.to_vec());
    assert_eq!(kept.writes, [(2, 0, moved), untouched]);
    assert_eq!(report.total.stream["written_bytes"], 128 + 8);
    assert_eq!(report.total.stream["relocated_words"], 1);
}

/// A `gpuResourceID` record compares the ID the program saw with the runtime's, once per object:
/// an object the runtime gives another ID, or none, is reported; one whose getter raised (no
/// answer) is not asked; one the stream never made stops the play.
#[test]
fn a_resource_id_the_runtime_gives_otherwise_is_reported() {
    let mut kept = Kept {
        ids: HashMap::from([(2, 0x4d), (3, 0x99)]),
        ..Kept::default()
    };
    let id = |id: u64| [("gpuResourceID", Value::Uint(id))];
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        for object in 2..=5 {
            w.call("newSamplerStateWithDescriptor:", 1, object, &[], &[])?;
        }
        w.call("gpuResourceID", 2, 0, &[], &id(0x4d))?;
        w.call("gpuResourceID", 3, 0, &[], &id(0x1_0000_0005))?;
        w.call("gpuResourceID", 3, 0, &[], &id(0x1_0000_0005))?;
        w.call("gpuResourceID", 4, 0, &[], &id(7))?;
        w.call("gpuResourceID", 5, 0, &[], &[])?;
        w.call("gpuResourceID", 9, 0, &[], &id(1))
    });
    match ended {
        Err(PlayError::Invalid(why)) => assert!(why.contains("#9, made by no call"), "{why}"),
        other => panic!("the play went on past an unmade object: {other:?}"),
    }
    assert_eq!(kept.asked, [2, 3, 4]);
    let differs = |object, recorded, played| IdDiffers {
        object,
        recorded,
        played,
    };
    let ids = [differs(3, 0x1_0000_0005, Some(0x99)), differs(4, 7, None)];
    assert_eq!(report.ids, ids);
    let lines = report.lines();
    assert!(
        lines.contains(&"id #3 0x100000005 0x99".to_owned()),
        "{lines:?}"
    );
    assert!(lines.contains(&"id #4 0x7 none".to_owned()), "{lines:?}");
}

/// The first object of the player's own.
const OWN: u64 = (1 << 63) + 1;

/// A dispatch with two `m2v.privateBytes` records, which come after its command buffer's commit
/// and its buffer's release: the player puts both ranges' bytes before in a Shared buffer of its
/// own, ends the encoder, copies them in a blit encoder into the ranges, makes the encoder afresh
/// with every call that set its state, dispatches, and the same with a blit that copies the ranges
/// out, each encoder after a fence the one before updated. At the completion it reads the bytes
/// after back and counts the words that differ from the recording's (here, the dispatch changed
/// nothing). The stream's counts hold none of the player's calls; the report counts the two
/// splits, and the executed counts of the group that held them (its copies among them) are
/// reported apart, in no group, frame or total; a later dispatch plays as it is.
#[test]
fn a_dispatch_plays_between_its_private_bytes() {
    let before: Vec<u8> = (0..24).collect();
    let mut after = before.clone();
    after[0] = 99;
    after[13] = 99;
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        let (head, two) = (w.blob(&before)?, w.blob(&after)?);
        let word = w.blob(&before[..4])?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let private = [Value::Uint(32), Value::Uint(0x20)];
        w.call("newBufferWithLength:options:", 1, 3, &private, &[])?;
        w.call("commandBuffer", 2, 4, &[], &[])?;
        w.call("computeCommandEncoder", 4, 5, &[], &[])?;
        let bound = [Value::Object(3), Value::Uint(0), Value::Uint(0)];
        w.call("setBuffer:offset:atIndex:", 5, 0, &bound, &[])?;
        let used = [map(&[("Buffer", Value::Object(3))]), Value::Uint(3)];
        w.call("useResource:usage:", 5, 0, &used, &[])?;
        let threads = map(&[("width", Value::Uint(1)), ("height", Value::Uint(1))]);
        let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
        w.call(dispatch, 5, 0, &[threads.clone(), threads.clone()], &[])?;
        w.call("memoryBarrierWithScope:", 5, 0, &[Value::Uint(1)], &[])?;
        w.call(dispatch, 5, 0, &[threads.clone(), threads], &[])?;
        w.call("endEncoding", 5, 0, &[], &[])?;
        w.call("commit", 4, 0, &[], &[])?;
        w.call("release", 3, 0, &[], &[])?;
        for (location, length, before, after) in [(8, 24, head, two), (0, 4, word, word)] {
            let range = map(&[
                ("location", Value::Uint(location)),
                ("length", Value::Uint(length)),
            ]);
            let args = [
                Value::Uint(7),
                range,
                Value::Blob(before),
                Value::Blob(after),
            ];
            w.call("m2v.privateBytes", 3, 0, &args, &[])?;
        }
        Ok(())
    });
    ended.unwrap();
    let (staging, fence) = (OWN, OWN + 1);
    let (blit_in, blit_out) = (OWN + 2, OWN + 3);
    let copy = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
    let state = [
        "#5 setBuffer:offset:atIndex: #3 0 0".to_string(),
        "#5 useResource:usage: {Buffer: #3} 3".to_string(),
    ];
    let mut expected = vec![
        format!("#{staging} = #1 newBufferWithLength:options: 96 0"),
        format!("#{fence} = #1 newFence"),
        format!("#5 updateFence: #{fence}"),
        "#5 endEncoding".to_string(),
        format!("#{blit_in} = #4 blitCommandEncoder"),
        format!("#{blit_in} waitForFence: #{fence}"),
        format!("#{blit_in} {copy} #{staging} 0 #3 8 24"),
        format!("#{blit_in} {copy} #{staging} 32 #3 0 4"),
        format!("#{blit_in} updateFence: #{fence}"),
        format!("#{blit_in} endEncoding"),
        format!("#{blit_in} release"),
        "#5 = #4 computeCommandEncoder".to_string(),
        format!("#5 waitForFence: #{fence}"),
    ];
    expected.extend(state.clone());
    expected.extend([
        "#5 dispatchThreadgroups:threadsPerThreadgroup: {width: 1, height: 1} {width: 1, height: 1}"
            .to_string(),
        format!("#5 updateFence: #{fence}"),
        "#5 endEncoding".to_string(),
        format!("#{blit_out} = #4 blitCommandEncoder"),
        format!("#{blit_out} waitForFence: #{fence}"),
        format!("#{blit_out} {copy} #3 8 #{staging} 48 24"),
        format!("#{blit_out} {copy} #3 0 #{staging} 80 4"),
        format!("#{blit_out} updateFence: #{fence}"),
        format!("#{blit_out} endEncoding"),
        format!("#{blit_out} release"),
        "#5 = #4 computeCommandEncoder".to_string(),
        format!("#5 waitForFence: #{fence}"),
    ]);
    expected.extend(state);
    expected.extend([
        "#5 memoryBarrierWithScope: 1".to_string(),
        "#5 dispatchThreadgroups:threadsPerThreadgroup: {width: 1, height: 1} {width: 1, height: 1}"
            .to_string(),
        "#5 endEncoding".to_string(),
        "#4 commit".to_string(),
        format!("#{staging} release"),
        format!("#{fence} release"),
        "#3 release".to_string(),
    ]);
    let at = kept
        .calls
        .iter()
        .position(|c| c.starts_with(&format!("#{staging} =")));
    assert_eq!(&kept.calls[at.unwrap()..], &expected[..]);
    assert_eq!(
        kept.writes,
        [
            (staging, 0, before.clone()),
            (staging, 32, before[..4].to_vec())
        ]
    );
    let compared = |location, length, differing| PrivateBytes {
        buffer: 3,
        dispatch: 7,
        location,
        length,
        differing,
    };
    assert_eq!(report.private, [compared(8, 24, 2), compared(0, 4, 0)]);
    let lines = report.lines();
    assert!(lines.contains(&"private #3 7 8+24 differs 2/6".to_string()));
    assert!(lines.contains(&"private #3 7 0+4 match".to_string()));
    let stream = &report.total.stream;
    assert_eq!(
        (stream["calls"], stream["encoders"], stream["dispatches"]),
        (15, 1, 2)
    );
    assert_eq!(stream["written_bytes"], 28);
    assert_eq!((kept.copies, report.splits), (4, 2));
    let split = Counters::from([("copies", 4), ("submissions", 1)]);
    assert_eq!(report.split_executed, split);
    assert!(lines.contains(&"split 2 executed copies=4 submissions=1".to_string()));
    assert_eq!(report.completed.len(), 1);
    assert!(report.completed[0].1.executed.is_empty());
    assert!(report.frames.is_empty());
    let total = &report.total.executed;
    assert!(total.values().all(|&value| value == 0), "{total:?}");
}

/// A message to nil (receiver 0, no result, no class named) plays nothing and does not stop the
/// play; the stream still counts it as a call.
#[test]
fn a_message_to_nil_does_nothing() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let region = map(&[
            ("origin", map(&[("x", Value::Uint(0))])),
            ("size", map(&[("width", Value::Uint(1))])),
        ]);
        let args = [
            region,
            Value::Uint(0),
            Value::Bytes(vec![0; 4]),
            Value::Uint(4),
        ];
        w.call(
            "replaceRegion:mipmapLevel:withBytes:bytesPerRow:",
            0,
            0,
            &args,
            &[],
        )?;
        Ok(())
    });
    ended.unwrap();
    assert_eq!(kept.calls, ["#1 = MTLCreateSystemDefaultDevice"]);
    assert_eq!(report.total.stream["calls"], 2);
}

/// Each frame's end reports the backend's gauges as they stand then, never their growth over the
/// frame, beside the frame's wall time.
#[test]
fn a_frame_end_reports_the_gauges_as_they_stand() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        w.call("newCommandQueue", 1, 3, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        Ok(())
    });
    ended.unwrap();
    let live: Vec<_> = report.ends.iter().map(|end| end.live["calls"]).collect();
    assert_eq!(live, [2, 3]);
    let lines = report.lines();
    let ends: Vec<_> = lines
        .iter()
        .filter(|line| line.starts_with("end "))
        .collect();
    assert_eq!(ends.len(), 2, "{lines:?}");
    assert!(ends[1].ends_with(" live calls=3"), "{lines:?}");
}

/// A frame's end counts the GPU's busy time as the union of the intervals of the command buffers
/// the frame waited for, where two overlap once, and its line says where the wall time went.
#[test]
fn a_frame_end_counts_overlapping_gpu_intervals_once() {
    let mut kept = Kept {
        gpu: HashMap::from([(3, (100, 300)), (4, (200, 400)), (5, (1000, 1100))]),
        ..Kept::default()
    };
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        for command_buffer in [3, 4, 5] {
            w.call("commandBuffer", 2, command_buffer, &[], &[])?;
            w.call("commit", command_buffer, 0, &[], &[])?;
        }
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        Ok(())
    });
    ended.unwrap();
    let busy: Vec<_> = report.ends.iter().map(|end| end.times.gpu_busy).collect();
    assert_eq!(busy, [400, 0]);
    let end = &report.ends[0];
    assert!(end.times.player(end.wall_nanos) <= end.wall_nanos);
    let lines = report.lines();
    let line = lines.iter().find(|line| line.starts_with("end 0 "));
    let line = line.expect("an end line");
    assert!(line.contains(" times stream="), "{line}");
    assert!(line.contains(" gpu_busy=400 live"), "{line}");
}

/// On a backend that takes no snapshots, or with a player that takes none, a dispatch an
/// `m2v.privateBytes` record names plays as it is, with none of the player's own calls, and the
/// report counts it.
#[test]
fn a_backend_without_snapshots_plays_the_dispatch_as_it_is() {
    for player_declines in [false, true] {
        let bytes: Vec<u8> = (0..4).collect();
        let mut kept = Kept {
            declines_snapshots: !player_declines,
            ..Kept::default()
        };
        let dir = ScratchDir::new("replay-player").unwrap();
        let path = written(&dir, "s.m2vr", |w| {
            let word = w.blob(&bytes)?;
            w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
            w.call("newCommandQueue", 1, 2, &[], &[])?;
            let private = [Value::Uint(4), Value::Uint(0x20)];
            w.call("newBufferWithLength:options:", 1, 3, &private, &[])?;
            w.call("commandBuffer", 2, 4, &[], &[])?;
            w.call("computeCommandEncoder", 4, 5, &[], &[])?;
            let threads = map(&[("width", Value::Uint(1)), ("height", Value::Uint(1))]);
            let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
            w.call(dispatch, 5, 0, &[threads.clone(), threads], &[])?;
            w.call("endEncoding", 5, 0, &[], &[])?;
            w.call("commit", 4, 0, &[], &[])?;
            let range = map(&[("location", Value::Uint(0)), ("length", Value::Uint(4))]);
            let args = [Value::Uint(5), range, Value::Blob(word), Value::Blob(word)];
            w.call("m2v.privateBytes", 3, 0, &args, &[])?;
            Ok(())
        });
        let mut reader = StreamReader::open(&[path], None).unwrap();
        let player = Player::new(&mut kept);
        let player = match player_declines {
            true => player.without_snapshots(),
            false => player,
        };
        let (report, ended) = player.play(&mut reader);
        ended.unwrap();
        assert!(kept.calls.iter().all(|c| !c.contains(&format!("#{OWN}"))));
        assert_eq!((report.unsnapshotted, report.splits), (1, 0));
        assert!(report.private.is_empty());
        assert!(report.lines().contains(&"unsnapshotted 1".to_string()));
    }
}

/// A record must name an earlier call, and that call a compute encoder's dispatch; a blob must hold
/// its range's bytes. Each else stops the play.
#[test]
fn a_private_bytes_record_names_an_earlier_dispatch() {
    let stream = |dispatch: u64, length: u64| {
        move |w: &mut StreamWriter<File>| {
            let bytes = w.blob(&[0; 8])?;
            w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
            w.call("newCommandQueue", 1, 2, &[], &[])?;
            w.call(
                "newBufferWithLength:options:",
                1,
                3,
                &[Value::Uint(8), Value::Uint(0x20)],
                &[],
            )?;
            w.call("commandBuffer", 2, 4, &[], &[])?;
            w.call("computeCommandEncoder", 4, 5, &[], &[])?;
            let threads = map(&[("width", Value::Uint(1))]);
            let args = [threads.clone(), threads];
            w.call("dispatchThreads:threadsPerThreadgroup:", 5, 0, &args, &[])?;
            w.call("endEncoding", 5, 0, &[], &[])?;
            w.call("commit", 4, 0, &[], &[])?;
            let range = map(&[
                ("location", Value::Uint(0)),
                ("length", Value::Uint(length)),
            ]);
            let args = [
                Value::Uint(dispatch),
                range,
                Value::Blob(bytes),
                Value::Blob(bytes),
            ];
            w.call("m2v.privateBytes", 3, 0, &args, &[])
        }
    };
    let (report, ended) = played(&mut Kept::default(), stream(5, 8));
    ended.unwrap();
    assert_eq!(report.private.len(), 1);
    for (dispatch, length) in [(8, 8), (4, 8), (5, 4)] {
        let mut kept = Kept::default();
        let (_, ended) = played(&mut kept, stream(dispatch, length));
        assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");
    }
}

/// The arguments the backend was given for the call that made `object`.
fn args_of(kept: &Kept, object: u64) -> &[Value] {
    let made = format!("#{object} = ");
    let at = kept.calls.iter().position(|call| call.starts_with(&made));
    &kept.args[at.unwrap()]
}

/// A placement heap's resources take the runtime's sizes and alignments, read ahead before the
/// play: where the runtime makes a texture larger than the room the stream left it below a buffer,
/// the buffer moves up past it and the heap grows by as much; a texture made again at its offset
/// shares it. A heap where every answer fits keeps the stream's layout. A texture's size is asked
/// of every call that made and set its descriptor.
#[test]
fn placement_heaps_take_the_runtimes_sizes_and_alignments() {
    let mut kept = Kept::default();
    kept.heap_answers.insert(3, (0xC000, 0x4000));
    kept.heap_answers.insert(9, (0x4000, 0x4000));
    let heap = |size: u64| {
        [map(&[
            ("size", Value::Uint(size)),
            ("heapType", Value::Uint(1)),
            ("options", Value::List(Vec::new())),
        ])]
    };
    let buffer = [Value::Uint(0x4000), Value::Uint(0), Value::Uint(0x8000)];
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newHeapWithDescriptor:", 1, 2, &heap(0x1_0000), &[])?;
        w.call("m2v.textureDescriptor", 0, 3, &[map(&[])], &[])?;
        w.call("setUsage:", 3, 0, &[Value::Uint(1)], &[])?;
        let texture = [Value::Object(3), Value::Uint(0)];
        w.call("newTextureWithDescriptor:offset:", 2, 4, &texture, &[])?;
        w.call("newBufferWithLength:options:offset:", 2, 5, &buffer, &[])?;
        w.call("release", 4, 0, &[], &[])?;
        w.call("newTextureWithDescriptor:offset:", 2, 6, &texture, &[])?;
        w.call("newHeapWithDescriptor:", 1, 7, &heap(0x1_0000), &[])?;
        w.call("m2v.textureDescriptor", 0, 9, &[map(&[])], &[])?;
        let fits = [Value::Object(9), Value::Uint(0)];
        w.call("newTextureWithDescriptor:offset:", 7, 10, &fits, &[])?;
        w.call("newBufferWithLength:options:offset:", 7, 11, &buffer, &[])
    });
    ended.unwrap();
    let size = |object| args_of(&kept, object)[0].as_map().unwrap()[0].1.as_uint();
    let offset = |object, at: usize| args_of(&kept, object)[at].as_uint();
    assert_eq!(size(2), Some(0x1_4000), "grown by what the buffer moved");
    assert_eq!((offset(4, 1), offset(6, 1)), (Some(0), Some(0)));
    assert_eq!(offset(5, 2), Some(0xC000), "past the larger texture");
    assert_eq!(size(7), Some(0x1_0000), "every answer fits");
    assert_eq!((offset(10, 1), offset(11, 2)), (Some(0), Some(0x8000)));
    let asked = |selector: &str, calls| (selector.to_owned(), calls);
    let texture = "newTextureWithDescriptor:offset:";
    let buffer = "newBufferWithLength:options:offset:";
    assert_eq!(
        kept.heap_asked,
        [
            asked(texture, 2),
            asked(buffer, 0),
            asked(texture, 2),
            asked(texture, 1),
            asked(buffer, 0)
        ]
    );
    assert_eq!(
        report.relaid,
        Relaid {
            heaps: 1,
            moved: 1,
            grown: 0x4000
        }
    );
    assert!(report
        .lines()
        .contains(&"relaid 1 moved 1 grown 16384".to_owned()));
}
