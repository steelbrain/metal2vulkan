//! The stream format, the address map and the player's order, on a backend that only keeps what it
//! is asked (no device). The runtime backends' tests are device tests (`runtime_lane`).

use super::addresses::AddressMap;
use super::player::{Backend, Counters, Made, PlayError, Player, Report};
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

/// Every kind of value, a map key and a selector used twice and a blob written twice come back as
/// written; the names and the blob are each written once.
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
    let mut reader = StreamReader::open(&[path.clone()], None).unwrap();
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
    let mut reader = StreamReader::open(&[path.clone()], None).unwrap();
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
    let mut reader = StreamReader::open(&[path.clone()], Some(blobs.clone())).unwrap();
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
/// with the block, one past every range stays. A range that meets another in part is refused; a
/// removed block leaves its buffers.
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
        map.insert(4, block + 0x80, 0x100, 0).is_err(),
        "meets #2 in part"
    );
    assert!(
        map.insert(4, block - 8, 0x10, 0).is_err(),
        "meets #1 in part"
    );
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
    // `bytes` starts 4 bytes into its target: the second address is at the target's offset 16.
    let (moved, misaligned) = map.relocate(&mut bytes, 4);
    assert_eq!((moved, misaligned), (1, 1));
    assert_eq!(
        &bytes[4..12],
        &(block + 0x100).to_le_bytes(),
        "4 past 8 stays"
    );
    assert_eq!(&bytes[16..24], &0x90_0010u64.to_le_bytes());
}

/// A backend that keeps what it is asked: the calls, the waits, the bytes written. It places
/// blocks and buffers apart from the recorded addresses, a buffer in a block at its offset, and
/// makes every object at the resource ID its call answers, `skew` past it.
#[derive(Default)]
struct Kept {
    calls: Vec<String>,
    /// The last call's arguments.
    args: Vec<Value>,
    waits: Vec<u64>,
    writes: Vec<(u64, u64, Vec<u8>)>,
    bases: HashMap<u64, u64>,
    next: u64,
    skew: u64,
    commits: u64,
}

impl Backend for Kept {
    fn call(&mut self, call: &Call) -> Result<Made, PlayError> {
        self.calls.push(call.to_string());
        self.args = call.args.clone();
        let mut made = Made {
            resource_id: call
                .answer("gpuResourceID")
                .and_then(Value::as_uint)
                .map(|id| id + self.skew),
            address: None,
        };
        match &*call.selector {
            "m2v.newBlockWithLength:storageMode:" | "newBufferWithLength:options:" => {
                self.next += 0x1000_0000;
                made.address = Some(self.next);
            }
            "m2v.newBufferInBlock:offset:length:options:" => {
                let block = call.args[0].as_object().unwrap();
                made.address = Some(self.bases[&block] + call.args[1].as_uint().unwrap());
            }
            "commit" => self.commits += 1,
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
        Ok(made)
    }

    fn write(&mut self, target: u64, offset: u64, bytes: &[u8]) -> Result<(), PlayError> {
        self.writes.push((target, offset, bytes.to_vec()));
        Ok(())
    }

    fn wait(&mut self, command_buffer: u64) -> Result<(), PlayError> {
        self.waits.push(command_buffer);
        Ok(())
    }

    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError> {
        Ok(object.to_le_bytes().to_vec())
    }

    fn counters(&mut self) -> Counters {
        Counters::from([("submissions", self.commits)])
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
            1,
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
    assert_eq!(kept.args.first(), Some(&Value::Bytes(moved.to_vec())));
    assert_eq!(report.total["relocated_words"], 2);
    assert_eq!(report.total["misaligned_address_words"], 1);
    assert_eq!(report.total["written_bytes"], 8 + 16 + 8);
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
        w.call("waitUntilCompleted", 5, 0, &[], &[])
    });
    ended.unwrap();
    assert_eq!(kept.waits, [7, 5, 6], "#6 waits behind #5 on its queue");
    let groups: Vec<_> = report
        .completed
        .iter()
        .map(|(ids, c)| (ids.clone(), c["submissions"]))
        .collect();
    assert_eq!(groups, [(vec![7], 3), (vec![5, 6], 0)]);
}

/// The CPU waiting for a command buffer that only a later record lets run, a stream that ends with
/// one held, a resource ID the backend did not give, and a selector the backend refuses each stop
/// the play, and the report keeps what ran.
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
        w.call("waitUntilCompleted", 4, 0, &[], &[])
    });
    assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");
    let (_, ended) = played(&mut Kept::default(), held);
    assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");

    let texture = |w: &mut StreamWriter<File>| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let answer = [("gpuResourceID", Value::Uint(77))];
        w.call("newTextureWithDescriptor:", 1, 2, &[map(&[])], &answer)?;
        w.call("m2v.check", 0, 0, &[Value::Object(2)], &[])
    };
    let (report, ended) = played(&mut Kept::default(), texture);
    ended.unwrap();
    assert_eq!(report.checks.len(), 1);
    let mut skewed = Kept {
        skew: 1,
        ..Kept::default()
    };
    let (report, ended) = played(&mut skewed, texture);
    assert!(
        matches!(
            ended,
            Err(PlayError::ResourceId {
                object: 2,
                recorded: 77,
                made: Some(78)
            })
        ),
        "{ended:?}"
    );
    assert!(report.checks.is_empty());

    let (_, ended) = played(&mut Kept::default(), |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("notPlayed", 1, 0, &[], &[])
    });
    assert!(matches!(ended, Err(PlayError::Refused { .. })), "{ended:?}");
}

/// A present reads the texture shown and closes a frame of the groups since the last; a released
/// device releases what it made, last made first; the host's split commits its command buffer and
/// makes the next part on the same queue.
#[test]
fn frames_releases_and_splits() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("commandBuffer", 2, 3, &[], &[])?;
        w.call("m2v.split", 3, 4, &[], &[])?;
        w.call("commit", 4, 0, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Object(3)], &[])?;
        w.call("release", 1, 0, &[], &[])
    });
    ended.unwrap();
    assert_eq!(kept.waits, [3, 4]);
    assert_eq!(
        report.frames,
        [Counters::from([
            ("misaligned_address_words", 0),
            ("relocated_words", 0),
            ("submissions", 2),
            ("written_bytes", 0),
        ])]
    );
    assert_eq!(report.checks.len(), 1);
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
