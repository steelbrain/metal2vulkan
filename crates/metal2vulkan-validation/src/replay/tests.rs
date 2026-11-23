//! The stream format, the address map and the player's order, on a backend that only keeps what it
//! is asked (no device). The runtime backend's tests are device tests (`replay::rt_replay`).

use super::addresses::AddressMap;
use super::compare::{compare_plays, shown_differences};
use super::covers::SlotStage;
use super::dumps::{compare_dumps, CommandBufferList, Difference, Part};
use super::new_runtime::indices::Indices;
use super::new_runtime::places::Waiting;
use super::new_runtime::second::merge;
use super::new_runtime::unwritten::Unwritten;
use super::new_runtime::{judged_records, Ended};
use super::pixels::{difference, Shape};
use super::player::{
    Backend, Counters, Counts, IdDiffers, Made, PlayError, Player, ReadCopy, Relaid, Report, Shown,
    TextureWrites, WAIT_COMPLETED,
};
use super::stream::{segments, Call, StreamError, StreamReader, StreamWriter};
use super::value::{hex, Value};
use crate::ScratchDir;
use metal2vulkan_runtime as rt;
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

/// A stream's recording range is its sidecar's where it has one, else the row of the table that
/// names its first segment's file name and length, else the whole stream; a sidecar that reads as
/// neither form is an error.
#[test]
fn a_recording_range_comes_from_the_sidecar_then_the_table() {
    use super::range::{recording_range, sidecar, RangeSource, RecordingRange};
    let dir = ScratchDir::new("replay-range").unwrap();
    let first = dir.path().join("s.0000.m2vr");
    let mut writer = StreamWriter::new(File::create(&first).unwrap()).unwrap();
    writer.call("commit", 1, 0, &[], &[]).unwrap();
    writer.finish().unwrap();
    let length = std::fs::metadata(&first).unwrap().len();
    let table = [("x", "s.0000.m2vr", length, RecordingRange::LastFrame(7))];
    let other = [("x", "s.0000.m2vr", length + 1, RecordingRange::LastFrame(7))];
    let found = |table: &[super::range::RangeRow]| recording_range(&first, table).unwrap();
    assert_eq!(
        found(&table),
        (RecordingRange::LastFrame(7), RangeSource::Table("x"))
    );
    assert_eq!(
        found(&other),
        (RecordingRange::Whole, RangeSource::Unstated)
    );
    assert_eq!(sidecar(&first), dir.path().join("s.range"));
    std::fs::write(sidecar(&first), "whole\n").unwrap();
    assert_eq!(found(&table), (RecordingRange::Whole, RangeSource::Sidecar));
    std::fs::write(sidecar(&first), "last-frame 3\n").unwrap();
    assert_eq!(
        found(&table),
        (RecordingRange::LastFrame(3), RangeSource::Sidecar)
    );
    std::fs::write(sidecar(&first), "frames 3\n").unwrap();
    let why = recording_range(&first, &table).unwrap_err();
    assert!(why.ends_with("is neither last-frame F nor whole"), "{why}");
}

/// A play ends at the recording range's last frame, or earlier at `--until-frame`; past the range
/// only where asked, and a whole stream's play ends only where `--until-frame` says.
#[test]
fn a_play_ends_at_the_recording_range_unless_asked_past_it() {
    use super::range::RecordingRange::{LastFrame, Whole};
    assert_eq!(LastFrame(1128).last_frame(false, None), Some(1128));
    assert_eq!(LastFrame(1128).last_frame(false, Some(1200)), Some(1128));
    assert_eq!(LastFrame(1128).last_frame(false, Some(100)), Some(100));
    assert_eq!(LastFrame(1128).last_frame(true, None), None);
    assert_eq!(LastFrame(1128).last_frame(true, Some(1200)), Some(1200));
    assert_eq!(Whole.last_frame(false, None), None);
    assert_eq!(Whole.last_frame(false, Some(5)), Some(5));
}

/// A render pipeline record's tessellation state and call-stack depths: the recorder's pinned
/// record (`recorder::tests::on_vulkan_a_render_pipeline_record_carries_its_tessellation_state`)
/// parses to its values; a record without the keys, as every stream before them, to Metal's
/// defaults; a key the layout does not hold, at the top or in the tessellation map, is refused.
#[test]
fn a_render_pipeline_record_s_tessellation_state_and_depths_are_read() {
    use super::args::Fields;
    use super::translator::{stages_state, StagesState, RENDER_PIPELINE_KEYS};
    use metal2vulkan::metal::{
        TessellationControlPointIndexType, TessellationFactorStepFunction,
        TessellationPartitionMode, Winding,
    };
    use metal2vulkan::pipeline::TessellationDescriptor;
    let call = |fields: Vec<(&str, Value)>| Call {
        selector: "newRenderPipelineStateWithDescriptor:error:".into(),
        receiver: 1,
        result: 2,
        args: vec![Value::Map(
            fields.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        )],
        answers: Vec::new(),
    };
    let read = |call: &Call| -> Result<StagesState, PlayError> {
        stages_state(&Fields::arg(call, 0, &RENDER_PIPELINE_KEYS)?)
    };
    let tessellation = |extra: Option<(&str, Value)>| {
        let mut keys = vec![
            ("partitionMode", Value::Uint(3)),
            ("maxFactor", Value::Uint(32)),
            ("factorScaleEnabled", Value::Bool(true)),
            ("factorFormat", Value::Uint(0)),
            ("controlPointIndexType", Value::Uint(2)),
            ("factorStepFunction", Value::Uint(1)),
            ("outputWinding", Value::Uint(1)),
        ];
        keys.extend(extra);
        Value::Map(keys.into_iter().map(|(k, v)| (k.into(), v)).collect())
    };
    let pinned = call(vec![
        ("tessellation", tessellation(None)),
        ("maxVertexCallStackDepth", Value::Uint(2)),
        ("maxFragmentCallStackDepth", Value::Uint(3)),
    ]);
    let mut expected = TessellationDescriptor::default();
    expected.partition_mode = TessellationPartitionMode::FractionalEven;
    expected.max_tessellation_factor = 32;
    expected.factor_scale_enabled = true;
    expected.control_point_index_type = TessellationControlPointIndexType::UInt32;
    expected.factor_step_function = TessellationFactorStepFunction::PerPatch;
    expected.output_winding_order = Winding::CounterClockwise;
    let state = read(&pinned).unwrap();
    assert_eq!(
        state,
        StagesState {
            tessellation: expected,
            vertex_call_stack_depth: 2,
            fragment_call_stack_depth: 3,
        }
    );
    let before = call(vec![("label", Value::Nil)]);
    assert_eq!(
        read(&before).unwrap(),
        StagesState {
            tessellation: TessellationDescriptor::default(),
            vertex_call_stack_depth: 1,
            fragment_call_stack_depth: 1,
        }
    );
    let inner = call(vec![(
        "tessellation",
        tessellation(Some(("stepRate", Value::Uint(1)))),
    )]);
    assert!(matches!(read(&inner), Err(PlayError::Refused { .. })));
    let outer = call(vec![("maxTileCallStackDepth", Value::Uint(1))]);
    assert!(matches!(read(&outer), Err(PlayError::Refused { .. })));
}

/// The header's flags, the same in every segment, are the stream's; a segment with another bit,
/// or with flags unlike the first segment's, is refused.
#[test]
fn header_flags_are_read_alike_in_every_segment() {
    let dir = ScratchDir::new("replay-flags").unwrap();
    let mark = super::FLAG_ANNOUNCED_BYTES_RECORDED;
    let segment = |name: &str, flags: u32| {
        let path = dir.path().join(name);
        let mut writer = StreamWriter::with_flags(File::create(&path).unwrap(), flags).unwrap();
        writer.call("commit", 1, 0, &[], &[]).unwrap();
        writer.finish().unwrap();
        path
    };
    let first = segment("s.0000.m2vr", mark);
    let second = segment("s.0001.m2vr", mark);
    let reader = StreamReader::open(&[first.clone(), second], None).unwrap();
    assert_eq!(reader.flags(), mark);
    let unmarked = segment("u.m2vr", 0);
    assert_eq!(StreamReader::open(&[unmarked], None).unwrap().flags(), 0);
    let other = segment("s.0001.m2vr", 0);
    assert!(matches!(
        StreamReader::open(&[first.clone(), other], None),
        Err(StreamError::Flags(_, 0))
    ));
    let fixed = mark | super::FLAG_COMPLETION_BYTES_RECORDED;
    let completed = segment("c.m2vr", fixed);
    assert_eq!(StreamReader::open(&[completed], None).unwrap().flags(), 3);
    let unknown = segment("o.m2vr", fixed | 4);
    assert!(matches!(
        StreamReader::open(&[unknown], None),
        Err(StreamError::Flags(_, 7))
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

/// Ranges nest: an address in a buffer inside a block goes with the buffer, one in the block alone
/// with the block, one past every range nowhere. A removed block leaves its buffers.
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

    // The census counts words at multiples of 8 from the target's start apart from values 4 past
    // one; bytes before the first such word are no word.
    let mut bytes = vec![0u8; 4];
    bytes.extend_from_slice(&(block + 0x100).to_le_bytes());
    bytes.extend_from_slice(&[0u8; 4]);
    bytes.extend_from_slice(&(block + 0x110).to_le_bytes());
    // `bytes` starts 4 bytes into its target: the first address is at the target's offset 8, the
    // second at its offset 20.
    assert_eq!(map.census(&bytes, 4), (1, 1));
}

/// Words below the first live range and at or past the last one's end are not counted, and those
/// inside a range are: the span check before each lookup drops only words a lookup would drop.
#[test]
fn words_outside_the_live_ranges_span_are_not_counted() {
    let block = 1u64 << 46;
    let mut map = AddressMap::default();
    map.insert(1, block, 0x100, 0x10_0000).unwrap();
    map.insert(2, block + 0x1000, 0x100, 0x20_0000).unwrap();
    let words = [
        block - 8,
        block,
        block + 0x800,
        block + 0x1008,
        block + 0x1100,
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(map.census(&bytes, 0), (2, 0));
}

/// A range that meets a live one in part ends that one, whose inner ranges stay: the program put
/// the newer object in that memory while the older lived on, so an address there goes by the newer
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
/// buffer copies once there was one, and for each submission the counts `per_commit` names.
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
    /// The documented high words it gives (`Backend::resource_word`); an object not here has none.
    words: HashMap<u64, u32>,
    asked: Vec<u64>,
    memory: HashMap<u64, Vec<u8>>,
    copies: u64,
    /// The size and alignment it answers for a heap texture, by the id of its descriptor; a heap
    /// buffer it answers its length at `buffer_align` bytes (256 where that is 0).
    heap_answers: HashMap<u64, (u64, u64)>,
    buffer_align: u64,
    /// Each heap size it was asked: the placing selector and how many calls made and set the
    /// texture descriptor.
    heap_asked: Vec<(String, usize)>,
    /// The GPU start and end it gives each command buffer; one not here has none.
    gpu: HashMap<u64, (u64, u64)>,
    /// The recording device's `hasUnifiedMemory` the player told it.
    unified: Option<bool>,
    /// The textures it says are Managed; it says every other one is not.
    managed: Vec<u64>,
    /// Whether it skips every draw, and the draws it skipped.
    skips_draws: bool,
    skipped: u64,
    /// The frame and position of each call it was told it plays next.
    located: Vec<(u64, u64)>,
    /// The texture slots each compute pipeline state's kernel writes; one not here answers none.
    kernel_writes: HashMap<u64, Vec<u64>>,
    /// The counts each submission adds to its counters.
    per_commit: Counters,
    /// The buffers it keeps two copies of, and which one its read takes.
    read_copies: HashMap<u64, ReadCopy>,
}

impl Backend for Kept {
    fn call(&mut self, call: &Call) -> Result<Made, PlayError> {
        self.calls.push(call.to_string());
        self.args.push(call.args.clone());
        if self.skips_draws && call.selector.starts_with("draw") {
            self.skipped += 1;
        }
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
        let buffer = [
            "newBufferWithLength:options:",
            "newBufferWithLength:options:offset:",
        ];
        if buffer.contains(&&*call.selector) {
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

    fn read_copy(&self, object: u64) -> Option<ReadCopy> {
        self.read_copies.get(&object).copied()
    }

    fn read_texture_part(&mut self, object: u64, part: Part) -> Result<Vec<u8>, PlayError> {
        let words = [object, part.level, part.slice, part.plane];
        Ok(words.iter().flat_map(|word| word.to_le_bytes()).collect())
    }

    fn resource_id(&mut self, object: u64) -> Result<Option<u64>, PlayError> {
        self.asked.push(object);
        Ok(self.ids.get(&object).copied())
    }

    fn resource_word(&mut self, object: u64) -> Option<u32> {
        self.words.get(&object).copied()
    }

    fn heap_size_and_align(
        &mut self,
        call: &Call,
        descriptor: &[Call],
    ) -> Result<(u64, u64), PlayError> {
        self.heap_asked
            .push((call.selector.to_string(), descriptor.len()));
        let align = match self.buffer_align {
            0 => 256,
            align => align,
        };
        Ok(match descriptor.first() {
            Some(made) => self.heap_answers[&made.result],
            None => (call.args[0].as_uint().unwrap(), align),
        })
    }

    fn counters(&mut self) -> Counters {
        let mut counters = Counters::from([("submissions", self.commits)]);
        if self.copies > 0 {
            counters.insert("copies", self.copies);
        }
        for (&name, &count) in &self.per_commit {
            counters.insert(name, count * self.commits);
        }
        counters
    }

    fn lines(&self) -> Vec<String> {
        Vec::new()
    }

    fn live(&mut self) -> Counters {
        Counters::from([("calls", self.calls.len() as u64)])
    }

    fn recorded_unified_memory(&mut self, unified: bool) {
        self.unified = Some(unified);
    }

    fn managed_texture(&self, texture: u64) -> Option<bool> {
        Some(self.managed.contains(&texture))
    }

    fn skips(&self) -> u64 {
        self.skipped
    }

    fn locate(&mut self, frame: u64, call: u64) {
        self.located.push((frame, call));
    }

    fn texture_writes(&self, pipeline: u64, stage: SlotStage) -> Option<Vec<u64>> {
        (stage == SlotStage::Kernel).then(|| self.kernel_writes.get(&pipeline).cloned())?
    }
}

/// `build`'s stream, played on `backend`.
fn played(
    backend: &mut Kept,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> (Report, Result<(), PlayError>) {
    played_through(backend, None, build)
}

/// `build`'s stream, played on `backend` through frame `last` where one is given.
fn played_through(
    backend: &mut Kept,
    last: Option<usize>,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> (Report, Result<(), PlayError>) {
    let dir = ScratchDir::new("replay-player").unwrap();
    let path = written(&dir, "s.m2vr", build);
    let mut reader = StreamReader::open(&[path], None).unwrap();
    let player = Player::new(backend);
    let player = match last {
        Some(last) => player.until_frame(last),
        None => player,
    };
    player.play(&mut reader)
}

const BLOCK: u64 = 1 << 46;

fn addressed(address: u64, length: u64) -> [(&'static str, Value); 2] {
    [
        ("gpuAddress", Value::Address(address)),
        ("gpuLength", Value::Uint(length)),
    ]
}

/// A heap descriptor of `size` bytes and type `kind` (`MTLHeapType`).
fn heap_of(size: u64, kind: u64) -> [Value; 1] {
    [map(&[
        ("size", Value::Uint(size)),
        ("heapType", Value::Uint(kind)),
        ("options", Value::List(Vec::new())),
    ])]
}

/// Two frames, each one command buffer committed; `inside` adds its calls to the first frame,
/// after the command buffer's making and before its commit.
fn two_frames(
    w: &mut StreamWriter<File>,
    inside: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
    w.call("newCommandQueue", 1, 2, &[], &[])?;
    w.call("commandBuffer", 2, 3, &[], &[])?;
    inside(w)?;
    w.call("commit", 3, 0, &[], &[])?;
    w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
    w.call("commandBuffer", 2, 4, &[], &[])?;
    w.call("commit", 4, 0, &[], &[])?;
    w.call("m2v.present", 1, 0, &[Value::Nil], &[])
}

/// The counts each submission adds on the backend the judged metrics' tests play on.
fn judged_counts() -> Counters {
    Counters::from([
        ("passes", 3),
        ("barriers", 5),
        ("texture_clear_barriers", 1),
        ("texture_clears", 2),
        ("texture_clear_bytes", 100),
        ("bytes_copied", 7),
        ("blocks_made", 9),
    ])
}

/// The matrix's judged metrics of a play on the runtime are the play's executed totals, one
/// record each under the stream's first segment's stem (whitespace made `_`) and its range: the
/// barriers less the one after each upload's clears, the clears and their bytes, a counter the
/// backend does not report as 0, and no other
/// counter (here `blocks_made`). A play through a last frame names it.
#[test]
fn the_judged_metrics_total_the_play_less_the_zeroing_barrier() {
    let first = std::path::Path::new("streams/a b.0000.m2vr");
    let mut kept = Kept {
        per_commit: judged_counts(),
        ..Kept::default()
    };
    let (report, ended) = played(&mut kept, |w| two_frames(w, |_| Ok(())));
    ended.as_ref().unwrap();
    let records = judged_records(first, &report, &ended, None);
    let whole = [
        "M2V-MATRIX metric replay-rt/a_b/whole/submissions 2",
        "M2V-MATRIX metric replay-rt/a_b/whole/passes 6",
        "M2V-MATRIX metric replay-rt/a_b/whole/barriers 8",
        "M2V-MATRIX metric replay-rt/a_b/whole/bytes_copied 14",
        "M2V-MATRIX metric replay-rt/a_b/whole/pipelines_created 0",
        "M2V-MATRIX metric replay-rt/a_b/whole/new_textures_zeroed 4",
        "M2V-MATRIX metric replay-rt/a_b/whole/new_texture_zeroed_bytes 200",
    ];
    assert_eq!(records, whole);
    let mut kept = Kept {
        per_commit: judged_counts(),
        ..Kept::default()
    };
    let (report, ended) = played_through(&mut kept, Some(0), |w| two_frames(w, |_| Ok(())));
    assert_eq!(report.frames.len(), 1);
    let records = judged_records(first, &report, &ended, Some(0));
    assert_eq!(
        records[0],
        "M2V-MATRIX metric replay-rt/a_b/through-0/submissions 1"
    );
    assert_eq!(
        records[2],
        "M2V-MATRIX metric replay-rt/a_b/through-0/barriers 4"
    );
}

/// A play that did not finish its range gives no judged metric, so the matrix fails it for the
/// base's records it lacks: one that ended before its last frame's present, one that stopped, and
/// one whose backend skipped anything (no gate result). Each played its frames' counts.
#[test]
fn a_short_play_gives_no_judged_metric() {
    let first = std::path::Path::new("s.0000.m2vr");
    let none: [String; 0] = [];
    let mut kept = Kept {
        per_commit: judged_counts(),
        ..Kept::default()
    };
    let (report, ended) = played_through(&mut kept, Some(2), |w| two_frames(w, |_| Ok(())));
    ended.as_ref().unwrap();
    assert_eq!(report.frames.len(), 2);
    assert_eq!(judged_records(first, &report, &ended, Some(2)), none);
    let mut kept = Kept {
        per_commit: judged_counts(),
        ..Kept::default()
    };
    let (report, ended) = played(&mut kept, |w| {
        two_frames(w, |w| w.call("notPlayed", 1, 0, &[], &[]))
    });
    assert!(ended.is_err());
    assert_eq!(judged_records(first, &report, &ended, None), none);
    let mut kept = Kept {
        per_commit: judged_counts(),
        skips_draws: true,
        ..Kept::default()
    };
    let (report, ended) = played(&mut kept, |w| {
        two_frames(w, |w| {
            w.call("MTLRenderPassDescriptor.new", 0, 5, &[], &[])?;
            let pass = [Value::Object(5)];
            w.call("renderCommandEncoderWithDescriptor:", 3, 6, &pass, &[])?;
            let draw = [Value::Uint(3), Value::Uint(0), Value::Uint(3)];
            w.call("drawPrimitives:vertexStart:vertexCount:", 6, 0, &draw, &[])?;
            w.call("endEncoding", 6, 0, &[], &[])
        })
    });
    ended.as_ref().unwrap();
    assert_eq!((report.frames.len(), kept.skipped), (2, 1));
    assert_eq!(judged_records(first, &report, &ended, None), none);
}

/// The gauges the runtime's backend reads once its device dropped go out as the matrix's
/// lines, `M2V-MATRIX gauge replay-rt/device/GAUGE@drop N`: three fields after the marker, the
/// key ending in `@drop`, as `scripts/matrix/results.py` reads a record, whose rule fails a gauge
/// not at zero there. The end's error names such a gauge.
#[test]
fn the_gauges_at_device_drop_go_out_as_the_matrix_lines() {
    let ended = Ended {
        gauges: vec![("buffers", 0), ("blocks", 2)],
        zero: false,
        idle: true,
    };
    let lines = [
        "M2V-MATRIX gauge replay-rt/device/buffers@drop 0",
        "M2V-MATRIX gauge replay-rt/device/blocks@drop 2",
    ];
    assert_eq!(ended.lines(), lines);
    let error = ended.result().unwrap_err();
    assert!(error.contains("[blocks=2]"), "{error}");
    let zero = Ended {
        gauges: vec![("buffers", 0)],
        zero: true,
        idle: true,
    };
    assert_eq!(zero.result(), Ok(()));
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

/// A write into a block and a constant upload carry words that look like recorded addresses: the
/// player writes them as recorded on a backend that moved the block, and counts them, an address 4
/// past a multiple of 8 apart; a write into a buffer of no address is written as recorded and not
/// counted. No address argument moved.
#[test]
fn written_words_play_as_recorded_and_are_counted() {
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
    assert_eq!(kept.writes[0], (2, 0, pointer.to_vec()), "as recorded");
    assert_eq!(kept.writes[1], (2, 16, misaligned), "4 past 8 as recorded");
    assert_eq!(kept.writes[2], (4, 0, pointer.to_vec()), "no address");
    let constants = kept.args.last().and_then(|args| args.first());
    assert_eq!(constants, Some(&Value::Bytes(pointer.to_vec())));
    assert_eq!(report.total.stream["address_looking_words"], 2);
    assert_eq!(report.total.stream["misaligned_address_words"], 1);
    assert_eq!(report.total.stream["written_bytes"], 8 + 16 + 8);
    assert_eq!(report.total.stream["relocated_words"], 0);
}

/// A typed address argument in a live range goes where the backend made its innermost object (a
/// buffer in a block), counted as moved; one in no live range stays, and is not.
#[test]
fn a_typed_address_argument_moves_with_its_object() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let block = [Value::Uint(0x1_0000), Value::Uint(0)];
        let at = addressed(BLOCK, 0x1_0000);
        w.call("m2v.newBlockWithLength:storageMode:", 1, 2, &block, &at)?;
        let inner = [
            Value::Object(2),
            Value::Uint(0x100),
            Value::Uint(0x100),
            Value::Uint(0),
        ];
        let at = addressed(BLOCK + 0x100, 0x100);
        let placed = "m2v.newBufferInBlock:offset:length:options:";
        w.call(placed, 2, 3, &inner, &at)?;
        let set = [Value::Address(BLOCK + 0x108), Value::Uint(0)];
        w.call("setAddress:atIndex:", 9, 0, &set, &[])?;
        let set = [Value::Address(1 << 40), Value::Uint(1)];
        w.call("setAddress:atIndex:", 9, 0, &set, &[])
    });
    ended.unwrap();
    let address = |call: usize| kept.args[call][0].clone();
    assert_eq!(address(3), Value::Address(0x1000_0000 + 0x108));
    assert_eq!(address(4), Value::Address(1 << 40), "in no range");
    assert_eq!(report.total.stream["address_arguments"], 2);
    assert_eq!(report.total.stream["relocated_words"], 1);
    assert_eq!(report.total.stream["address_looking_words"], 0);
}

/// Once the stream notes a buffer escaped (`m2v.bufferEscaped`), a written word whose value lies in
/// its recorded range moves to where the backend made the buffer, plus its offset, in an
/// `m2v.write` and in `set*Bytes` data alike, and is counted as noted; the same word written
/// before the note, and a word into a buffer with no note, play as written. A note of a kind past
/// 3, or of an object no call made, is an invalid stream.
#[test]
fn a_word_into_a_buffer_noted_escaped_moves_with_it() {
    let mut kept = Kept::default();
    let noted = BLOCK;
    let plain = BLOCK + 0x1_0000;
    let word = |value: u64| value.to_le_bytes().to_vec();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let length = [Value::Uint(0x100), Value::Uint(0)];
        let made = "newBufferWithLength:options:";
        w.call(made, 1, 2, &length, &addressed(noted, 0x100))?;
        w.call(made, 1, 3, &length, &addressed(plain, 0x100))?;
        w.call(made, 1, 4, &length, &addressed(BLOCK + 0x2_0000, 0x100))?;
        let into = |offset: u64, value: u64| [Value::Uint(offset), Value::Bytes(word(value))];
        w.call("m2v.write", 4, 0, &into(0, noted + 0x10), &[])?;
        w.call("m2v.bufferEscaped", 2, 0, &[Value::Uint(1)], &[])?;
        w.call("m2v.write", 4, 0, &into(8, plain + 8), &[])?;
        w.call("m2v.write", 4, 0, &into(16, noted + 0x10), &[])?;
        let bytes = [
            Value::Bytes(word(noted + 0x20)),
            Value::Uint(8),
            Value::Uint(0),
        ];
        w.call("setBytes:length:atIndex:", 9, 0, &bytes, &[])
    });
    ended.unwrap();
    // The backend made the noted buffer at 0x1000_0000.
    assert_eq!(
        kept.writes,
        [
            (4, 0, word(noted + 0x10)),
            (4, 8, word(plain + 8)),
            (4, 16, word(0x1000_0010)),
        ]
    );
    let constants = kept.args.last().and_then(|args| args.first());
    assert_eq!(constants, Some(&Value::Bytes(word(0x1000_0020))));
    assert_eq!(report.total.stream["address_looking_words"], 4);
    assert_eq!(report.total.stream["noted_words"], 2);
    assert_eq!(report.escape_notes, 1);

    for (receiver, kind) in [(2, 4), (9, 0)] {
        let (_, ended) = played(&mut Kept::default(), |w| {
            w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
            let length = [Value::Uint(0x100), Value::Uint(0)];
            let made = "newBufferWithLength:options:";
            w.call(made, 1, 2, &length, &addressed(noted, 0x100))?;
            let note = [Value::Uint(kind)];
            w.call("m2v.bufferEscaped", receiver, 0, &note, &[])
        });
        assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");
    }
}

/// A copy from the guest's RAM lands each piece at its byte, as recorded: into a block, a word that
/// looks like an address is counted at a multiple of 8 and apart 4 past one; into a buffer of no
/// address, it is not counted. A copy of another form (one the platform refused, recorded before
/// the recorder left those out) stops the play.
#[test]
fn guest_ram_copies_play_as_recorded_and_are_counted() {
    let pointer = (BLOCK + 0x10).to_le_bytes();
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
    let copied = [pointer.to_vec(), vec![0; 8]].concat();
    assert_eq!(data(3, 0), Value::Bytes(copied));
    assert_eq!(data(4, 0), Value::Bytes(pointer.to_vec()));
    assert_eq!(data(4, 1), Value::Bytes(pointer.to_vec()), "4 past 8");
    assert_eq!(data(5, 0), Value::Bytes(pointer.to_vec()), "no address");
    assert_eq!(report.total.stream["address_looking_words"], 2);
    assert_eq!(report.total.stream["misaligned_address_words"], 1);
    assert_eq!(report.total.stream["written_bytes"], 16 + 16 + 8);
    assert_eq!(report.total.stream["relocated_words"], 0);

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
        ("address_looking_words", 0),
        ("noted_words", 0),
        ("written_bytes", 0),
        ("address_arguments", 0),
        ("relocated_words", 0),
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

/// Metal's objects keep their device: a program may release its device first and then what it
/// made. The device's release drops what it made, and the program's own release of one of them
/// later plays as nothing: the backend releases it once, with the device. An object no call
/// made still stops the play.
#[test]
fn an_object_released_after_its_device_plays_as_released_with_it() {
    let mut kept = Kept::default();
    let (_, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newSamplerStateWithDescriptor:", 1, 2, &[], &[])?;
        w.call("newSamplerStateWithDescriptor:", 1, 3, &[], &[])?;
        w.call("release", 1, 0, &[], &[])?;
        w.call("release", 3, 0, &[], &[])?;
        w.call("release", 2, 0, &[], &[])
    });
    ended.unwrap();
    let released: Vec<&str> = (kept.calls.iter())
        .filter(|call| call.ends_with(" release"))
        .map(String::as_str)
        .collect();
    assert_eq!(released, ["#3 release", "#2 release", "#1 release"]);

    let (_, ended) = played(&mut Kept::default(), |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newSamplerStateWithDescriptor:", 1, 2, &[], &[])?;
        w.call("release", 1, 0, &[], &[])?;
        w.call("release", 2, 0, &[], &[])?;
        w.call("release", 2, 0, &[], &[])
    });
    match ended {
        Err(PlayError::Invalid(why)) => assert!(why.contains("#2, made by no call"), "{why}"),
        other => panic!("a second release of a dropped object played: {other:?}"),
    }
}

/// A draw the backend skips is counted, and the play is no verdict past the first: the frame and
/// stream position of the call at which the backend's skips first grew, which it was told before
/// it played that call. A play with no skip has no such point.
#[test]
fn a_skipped_draw_ends_the_verdict_at_its_frame_and_call() {
    let mut kept = Kept {
        skips_draws: true,
        ..Kept::default()
    };
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        w.call("commandBuffer", 2, 3, &[], &[])?;
        w.call("MTLRenderPassDescriptor.new", 0, 4, &[], &[])?;
        let pass = [Value::Object(4)];
        w.call("renderCommandEncoderWithDescriptor:", 3, 5, &pass, &[])?;
        // MTLPrimitiveTypeTriangle, from vertex 0, three vertices: calls 6 and 7.
        let draw = [Value::Uint(3), Value::Uint(0), Value::Uint(3)];
        w.call("drawPrimitives:vertexStart:vertexCount:", 5, 0, &draw, &[])?;
        w.call("drawPrimitives:vertexStart:vertexCount:", 5, 0, &draw, &[])?;
        w.call("endEncoding", 5, 0, &[], &[])?;
        w.call("commit", 3, 0, &[], &[])
    });
    ended.unwrap();
    assert_eq!(kept.skipped, 2);
    assert!(kept.located.contains(&(1, 6)), "{:?}", kept.located);
    assert_eq!(report.first_skip, Some((1, 6)));
    let note = report.verdict_note();
    assert_eq!(note.as_deref(), Some("not a verdict past frame 1, call 6"));
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])
    });
    ended.unwrap();
    assert_eq!((report.first_skip, report.verdict_note()), (None, None));
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
            ("address_looking_words", 0),
            ("noted_words", 0),
            ("written_bytes", 0),
            ("address_arguments", 0),
            ("relocated_words", 0),
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
        "completed #3 stream address_arguments=0 address_looking_words=0 calls=10 commits=1 \
         dispatches=1 draws=2 encoders=2 misaligned_address_words=0 noted_words=0 \
         relocated_words=0 written_bytes=0 executed submissions=1"
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
/// inline bytes: the backend's making takes them as bytes (a buffer that plays with two copies
/// starts its GPU copy from them), and the player writes them from its first byte as recorded,
/// counting a word that looks like an address where it answered a `gpuAddress` (here a pointer
/// into itself); a buffer of no address is written as it is, and a no-copy buffer's nil contents
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
    let made_with = [
        Value::Bytes(long.clone()),
        Value::Bytes(pointer.to_vec()),
        Value::Nil,
    ];
    assert_eq!(kept.args.len(), 1 + made_with.len());
    for (args, made_with) in kept.args[1..].iter().zip(&made_with) {
        let contents = args[0]
            .as_map()
            .unwrap()
            .iter()
            .find(|(k, _)| &**k == "contents");
        assert_eq!(contents.map(|(_, v)| v), Some(made_with));
    }
    let untouched = (3, 0, pointer.to_vec());
    assert_eq!(kept.writes, [(2, 0, long), untouched]);
    assert_eq!(report.total.stream["written_bytes"], 128 + 8);
    assert_eq!(report.total.stream["address_looking_words"], 1);
    assert_eq!(report.total.stream["relocated_words"], 0);
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
        word: None,
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

/// Where the backend gives the documented high word of an object's ID, the player compares the
/// ID's low word with the recording's and its high word with that word, never with the
/// recording's high word: an ID of the recording's index and the word passes whatever high word
/// the recording holds; another index, or another high word, is reported with the word.
#[test]
fn a_resource_id_s_high_word_is_checked_against_the_documented_word() {
    let mut kept = Kept {
        ids: HashMap::from([
            (2, 0x0001_0800_0000_0005),
            (3, 0x0001_0800_0000_0006),
            (4, 0x0002_0000_0000_0007),
        ]),
        words: HashMap::from([(2, 0x0001_0800), (3, 0x0001_0800), (4, 0x0001_0800)]),
        ..Kept::default()
    };
    let id = |id: u64| [("gpuResourceID", Value::Uint(id))];
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        for object in 2..=4 {
            w.call("newSamplerStateWithDescriptor:", 1, object, &[], &[])?;
        }
        // The recording's high words are its runtime's: 0 and 1.
        w.call("gpuResourceID", 2, 0, &[], &id(5))?;
        w.call("gpuResourceID", 3, 0, &[], &id(0x1_0000_0005))?;
        w.call("gpuResourceID", 4, 0, &[], &id(7))
    });
    ended.unwrap();
    let differs = |object, recorded, played| IdDiffers {
        object,
        recorded,
        played: Some(played),
        word: Some(0x0001_0800),
    };
    let ids = [
        differs(3, 0x1_0000_0005, 0x0001_0800_0000_0006),
        differs(4, 7, 0x0002_0000_0000_0007),
    ];
    assert_eq!(report.ids, ids);
    let lines = report.lines();
    let line = "id #4 0x7 0x2000000000007 word 0x10800";
    assert!(lines.contains(&line.to_owned()), "{lines:?}");
}

/// A texture its descriptor marks `outside` is outside the program's IDs: a `gpuResourceID`
/// record of it, where the mark or the record is wrong, stops the play, naming the call; a plain
/// texture's is compared as ever.
#[test]
fn a_texture_marked_outside_whose_id_the_program_saw_stops_the_play() {
    let mut kept = Kept {
        ids: HashMap::from([(5, 0x10)]),
        ..Kept::default()
    };
    let id = |id: u64| [("gpuResourceID", Value::Uint(id))];
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("m2v.textureDescriptor", 0, 2, &[map(&[])], &[])?;
        let outside = map(&[("outside", Value::Bool(true))]);
        w.call("m2v.textureDescriptor", 0, 3, &[outside], &[])?;
        w.call("newTextureWithDescriptor:", 1, 4, &[Value::Object(3)], &[])?;
        w.call("newTextureWithDescriptor:", 1, 5, &[Value::Object(2)], &[])?;
        w.call("gpuResourceID", 5, 0, &[], &id(0x10))?;
        w.call("gpuResourceID", 4, 0, &[], &id(0x11))
    });
    match ended {
        Err(PlayError::Invalid(why)) => assert!(why.contains("gpuResourceID of #4"), "{why}"),
        other => panic!("the play went on past a marked texture's ID: {other:?}"),
    }
    assert_eq!(kept.asked, [5]);
    assert_eq!(report.ids, []);
}

/// The runtime's backend places a texture marked `outside` at the highest free index of its
/// class and leaves its sequence where it was: the plain texture made after it takes the index
/// it takes where the stream makes no marked one, and a second marked texture the next free
/// index down.
#[test]
fn a_texture_marked_outside_takes_no_index_of_the_sequence() {
    let class = rt::IndexClass::Texture;
    let make = |index: u32| Ok::<u32, rt::Error>(index);
    let placed = |indices: &mut Indices, object| {
        let made = indices.place(object, class, None, make);
        made.expect("a free index").expect("made")
    };
    let topped = |indices: &mut Indices, object| {
        let made = indices.top(object, class, 0, make);
        made.expect("a free index").expect("made")
    };
    let mut indices = Indices::default();
    let unmarked = [placed(&mut indices, 1), placed(&mut indices, 2)];
    let mut indices = Indices::default();
    let first = placed(&mut indices, 1);
    let marked = topped(&mut indices, 2);
    let second = placed(&mut indices, 3);
    let marked_again = topped(&mut indices, 4);
    assert_eq!([first, second], unmarked);
    let end = class.range().end;
    assert_eq!((marked, marked_again), (end - 1, end - 2));
}

/// The runtime's backend gives a constexpr sampler, which the stream names no index for, the
/// highest free sampler index, past those the same call took, and leaves the sequence the
/// program's sampler states take where it was: a sampler state at its index would take it.
#[test]
fn a_constexpr_sampler_takes_no_index_of_the_sequence() {
    let class = rt::IndexClass::Sampler;
    let (start, end) = (class.range().start, class.range().end);
    let make = |index: u32| Ok::<u32, rt::Error>(index);
    let mut indices = Indices::default();
    let first = indices.place(1, class, None, make);
    assert_eq!(first.expect("a free index").expect("made"), start);
    let constexpr = indices.highest(class, |_| false);
    assert_eq!(constexpr, Some(end - 1));
    indices.hold(None, class, end - 1);
    assert_eq!(
        indices.highest(class, |index| index == end - 2),
        Some(end - 3)
    );
    let second = indices.place(2, class, None, make);
    assert_eq!(second.expect("a free index").expect("made"), start + 1);
}

/// An object takes the index its record names; else, and for an object with none, the lowest
/// index of its class that no live object holds and the program did not reserve. A named index an
/// object holds is taken from it, and the earlier object's release leaves it; a released object's
/// index is free again.
#[test]
fn objects_take_the_named_index_else_the_lowest_free_one() {
    let class = rt::IndexClass::Sampler;
    let start = class.range().start;
    let make = |index: u32| Ok::<u32, rt::Error>(index);
    let mut indices = Indices::default();
    indices.reserve(class, start + 2);
    let take = |indices: &mut Indices, object, named| {
        let made = indices.place(object, class, named, make);
        made.expect("a free index").expect("made")
    };
    assert_eq!(
        take(&mut indices, 1, Some(start + 7)),
        start + 7,
        "the named index"
    );
    assert_eq!(take(&mut indices, 2, None), start, "the lowest");
    // start + 2 is reserved.
    assert_eq!(take(&mut indices, 3, None), start + 1);
    assert_eq!(take(&mut indices, 4, None), start + 3);
    // A named index an object holds: taken, and the earlier object's release leaves it.
    assert_eq!(take(&mut indices, 5, Some(start + 7)), start + 7);
    indices.give(1, 0);
    assert_eq!(indices.lowest(class, start + 7), Some(start + 8));
    assert_eq!(
        take(&mut indices, 6, Some(start + 2)),
        start + 2,
        "the stream names a reserved index"
    );
    indices.give(2, 0);
    assert_eq!(
        take(&mut indices, 7, None),
        start,
        "free again after the release"
    );
    indices.give(5, 0);
    assert_eq!(indices.lowest(class, start + 7), Some(start + 7));
    assert!(
        indices.line().starts_with("indices named 3:"),
        "{}",
        indices.line()
    );
}

/// An index is owed to the last commit of a command buffer that named an object while it held the
/// index, the object live or released since, and to every commit before its holder's release; a
/// placement there waits only while the owed commit has not retired, and the line counts the
/// waits and their time. A command buffer released uncommitted owes nothing, and nor does one that
/// named an object after a later one took its index: of two live objects placed at one index, the
/// index is the later one's.
#[test]
fn an_index_is_owed_to_the_commits_that_named_its_holders() {
    let class = rt::IndexClass::Texture;
    let made = |index: u32| Ok::<u32, rt::Error>(index);
    let mut indices = Indices::default();
    let placed = indices.place(1, class, Some(10), made);
    assert_eq!(placed.expect("a free index").expect("made"), 10);
    indices.named(100, &[1]);
    indices.committed(100, 3);
    assert_eq!(
        indices.owed(class, 10, 2),
        Some(3),
        "a live holder's commit"
    );
    // Released while one more command buffer not yet committed named it.
    indices.named(101, &[1]);
    indices.give(1, 3);
    indices.committed(101, 5);
    assert_eq!(
        indices.owed(class, 10, 4),
        Some(5),
        "the later user's commit"
    );
    assert_eq!(indices.owed(class, 10, 5), None, "retired");
    assert_eq!(indices.owed(class, 10, 0), None, "owed no more");
    let placed = indices.place(2, class, Some(10), made);
    assert_eq!(placed.expect("a free index").expect("made"), 10);
    indices.named(102, &[2]);
    indices.committed(102, 6);
    assert_eq!(indices.owed(class, 10, 5), Some(6));
    indices.waited(class, 10, 7);
    assert_eq!(indices.owed(class, 10, 0), None, "waited for");
    // An object only an uncommitted command buffer named, which is released uncommitted.
    let placed = indices.place(3, class, Some(11), made);
    assert_eq!(placed.expect("a free index").expect("made"), 11);
    indices.named(103, &[3]);
    indices.give(3, 6);
    indices.ended(103);
    indices.committed(103, 9);
    assert_eq!(
        indices.owed(class, 11, 5),
        Some(6),
        "every commit before the release"
    );
    assert_eq!(indices.owed(class, 11, 6), None, "and nothing of #103");
    // Two live objects placed at 12, #5 after #4: a command buffer that names #4 then, which is
    // released after, owes 12 nothing; one that names #5 owes it.
    for object in [4, 5] {
        let placed = indices.place(object, class, Some(12), made);
        assert_eq!(placed.expect("a free index").expect("made"), 12);
    }
    indices.named(104, &[4]);
    indices.give(4, 9);
    indices.committed(104, 10);
    assert_eq!(indices.owed(class, 12, 0), None, "the earlier object's");
    indices.named(105, &[5]);
    indices.committed(105, 11);
    assert_eq!(indices.owed(class, 12, 0), Some(11), "the later object's");
    assert_eq!(
        indices.target(class, Some(0)),
        Some(1),
        "outside the class: the lowest"
    );
    let line = indices.line();
    assert!(
        line.ends_with(
            "1 placed at an index owed to commits in flight, after waiting for them (7 ns); 0 \
             sampler states placed without waiting at an index whose last sampler state had the \
             same descriptor while its commits were in flight"
        ),
        "{line}"
    );
}

/// A released object's index is owed to every commit before the release, which may reach it
/// through an ID in memory, though no command buffer named the object. An index the player chooses
/// itself (`top`) is never one owed to a commit in flight, and a placement kept back is counted.
#[test]
fn a_released_index_is_owed_to_every_commit_before_the_release() {
    let class = rt::IndexClass::Texture;
    let made = |index: u32| Ok::<u32, rt::Error>(index);
    let end = class.range().end;
    let mut indices = Indices::default();
    let placed = indices.place(1, class, Some(end - 1), made);
    assert_eq!(placed.expect("a free index").expect("made"), end - 1);
    // Released after commit 3, which named nothing.
    indices.give(1, 3);
    assert!(indices.owes(class, end - 1, 2), "commit 3 in flight");
    assert!(!indices.owes(class, end - 1, 3), "commit 3 retired");
    let topped = indices.top(2, class, 2, made);
    let topped = topped.expect("a free index").expect("made");
    assert_eq!(topped, end - 2, "the owed index skipped");
    indices.kept();
    let line = indices.kept_line();
    assert!(line.starts_with("indices kept back 1:"), "{line}");
    assert_eq!(indices.owed(class, end - 1, 2), Some(3), "still owed");
    let topped = indices.top(3, class, 3, made);
    let topped = topped.expect("a free index").expect("made");
    assert_eq!(topped, end - 1, "free again once commit 3 retired");
}

#[test]
fn an_equal_sampler_state_takes_an_owed_index_and_leaves_it_owed() {
    let class = rt::IndexClass::Sampler;
    let made = |index: u32| Ok::<u32, rt::Error>(index);
    let mut indices = Indices::default();
    let placed = indices.place(1, class, Some(16), made);
    assert_eq!(placed.expect("a free index").expect("made"), 16);
    assert_eq!(indices.index_of(1), Some(16));
    indices.named(100, &[1]);
    indices.committed(100, 4);
    indices.give(1, 4);
    // An equal sampler state at 16 while commit 4 is in flight: counted, and 16 stays owed.
    indices.equal(class, 16, 3);
    assert_eq!(indices.owed(class, 16, 3), Some(4), "still owed");
    // Once commit 4 retired, nothing is owed and nothing counted.
    indices.equal(class, 16, 4);
    assert_eq!(indices.owed(class, 16, 0), None, "owed no more");
    let line = indices.line();
    assert!(
        line.ends_with(
            "1 sampler states placed without waiting at an index whose last sampler state had \
             the same descriptor while its commits were in flight"
        ),
        "{line}"
    );
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

/// A device identity record plays as nothing and tells the backend how the recording device kept
/// Managed buffers.
#[test]
fn a_device_identity_record_tells_the_backend_how_managed_buffers_were_kept() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let answers = [
            ("name", Value::Str("GPU".into())),
            ("hasUnifiedMemory", Value::Bool(false)),
        ];
        w.call("m2v.deviceIdentity", 1, 0, &[], &answers)?;
        Ok(())
    });
    ended.unwrap();
    assert_eq!(kept.calls, ["#1 = MTLCreateSystemDefaultDevice"]);
    assert_eq!(kept.unified, Some(false));
    assert_eq!(report.total.stream["calls"], 2);
}

/// Of two recorded devices, the one the guest makes its objects on tells the Managed form, whichever
/// identity record comes last; the other is named on the report's line.
#[test]
fn the_guest_s_device_tells_the_managed_form() {
    for (guest, unified) in [(1, true), (2, false)] {
        let mut kept = Kept::default();
        let (report, ended) = played(&mut kept, |w| {
            w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
            w.call("MTLCreateSystemDefaultDevice", 0, 2, &[], &[])?;
            // One identity record per live device, in id order: device 1 unified, device 2 not.
            for (device, unified) in [(1, true), (2, false)] {
                let answers = [("hasUnifiedMemory", Value::Bool(unified))];
                w.call("m2v.deviceIdentity", device, 0, &[], &answers)?;
            }
            for queue in [3, 4] {
                w.call("newCommandQueue", guest, queue, &[], &[])?;
            }
            Ok(())
        });
        ended.unwrap();
        assert_eq!(kept.unified, Some(unified), "the guest on #{guest}");
        assert_eq!(report.other_devices, [3 - guest]);
        let line = format!("devices apart from the guest's: #{}", 3 - guest);
        assert!(report.lines().contains(&line), "{:?}", report.lines());
    }
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

/// A call of `receiver` with `args`, making nothing.
fn encoder_call(selector: &str, receiver: u64, args: Vec<Value>) -> Call {
    Call {
        selector: selector.into(),
        receiver,
        result: 0,
        args,
        answers: Vec::new(),
    }
}

/// Buffer `id` as a resource argument: `{Buffer: object}`.
fn buffer_resource(id: u64) -> Value {
    map(&[("Buffer", Value::Object(id))])
}

/// `setBuffer:offset:atIndex:` of buffer `id` at slot `slot` of encoder `encoder`.
fn bind_at(encoder: u64, id: u64, slot: u64) -> Call {
    let args = vec![Value::Object(id), Value::Uint(0), Value::Uint(slot)];
    encoder_call("setBuffer:offset:atIndex:", encoder, args)
}

fn dispatch_of(encoder: u64) -> Call {
    encoder_call("dispatchThreads:threadsPerThreadgroup:", encoder, vec![])
}

/// The `unwritten` line of `count` buffers, `unsure` of them only counted as read, the first
/// named as `first`.
fn unwritten_line(count: u64, unsure: u64, first: &str) -> String {
    format!(
        "unwritten {count}: Shared buffers a dispatch read before any recorded write, upload or GPU \
         pass reached their bytes, {unsure} of them counted as read though they may not be (named \
         by an encoder call that neither binds nor uses them, or bound under a skipped \
         state); first {first}"
    )
}

/// The runtime's `unwritten` line counts, once, a Shared buffer a committed dispatch read
/// before anything recorded reached its bytes: not one written through its block, one a blit
/// names, one written before the commit, one an earlier commit's dispatch bound or one released
/// before the commit, nor one made over its block's bytes that an earlier commit's dispatch may
/// have written. With no state set, a dispatch reads and may write every bound buffer, and the
/// line says it only counted them as read.
#[test]
fn a_dispatch_reading_a_buffer_nothing_recorded_reached_is_counted_once() {
    let mut unwritten = Unwritten::default();
    // Shared block 1 holds buffers 2, 3 and 4; buffers 5, 6 and 7 lie in none.
    unwritten.block(1);
    for (id, range) in [(2, 0..256), (3, 256..512), (4, 512..768)] {
        unwritten.made(id, Some((1, range)));
    }
    for id in [5, 6, 7] {
        unwritten.made(id, None);
    }
    unwritten.written(1, 300..304);
    let fill = encoder_call("fillBuffer:range:value:", 11, vec![Value::Object(5)]);
    unwritten.encoded(&fill, 10, false);
    for (slot, id) in (0..).zip([2, 3, 4, 5, 6, 7]) {
        unwritten.encoded(&bind_at(12, id, slot), 10, true);
    }
    unwritten.encoded(&dispatch_of(12), 10, true);
    unwritten.encoded(&dispatch_of(12), 10, true);
    unwritten.written(6, 0..4);
    unwritten.released(7);
    unwritten.committed(10);
    // Buffer 8 lies over buffer 2's bytes, which #10's dispatches may have written.
    unwritten.made(8, Some((1, 0..128)));
    for (slot, id) in (0..).zip([2, 8]) {
        unwritten.encoded(&bind_at(13, id, slot), 14, true);
    }
    unwritten.encoded(&dispatch_of(13), 14, true);
    unwritten.committed(14);
    assert_eq!(
        unwritten.line(),
        unwritten_line(2, 2, "#2 in #10, #4 in #10")
    );
}

/// A dispatch reads a bound buffer only at a slot its state reads (`ReadOnly`, `ReadWrite`); a
/// `WriteOnly` slot writes it, so the later dispatches of its command buffer and the later
/// command buffers find it written, and a slot the state does not use touches nothing. A buffer
/// `useResource:usage:` names is read or written by its usage's bits. A buffer another call of
/// the encoder names, or one bound under a skipped state, is counted as read, and the line says
/// how many were.
#[test]
fn a_dispatch_reads_a_buffer_only_where_its_state_or_its_usage_reads_it() {
    use metal2vulkan_runtime::BindingAccess::{ReadOnly, ReadWrite, WriteOnly};
    let mut unwritten = Unwritten::default();
    for id in 2..=13 {
        unwritten.made(id, None);
    }
    // #12 in #10: 2 read, 3 written, 4 read and written, 5 at a slot the state does not use.
    unwritten.state(
        12,
        Some(vec![(0, ReadOnly), (1, WriteOnly), (2, ReadWrite)]),
    );
    for (slot, id) in (0..).zip([2, 3, 4, 5]) {
        unwritten.encoded(&bind_at(12, id, slot), 10, true);
    }
    unwritten.encoded(&dispatch_of(12), 10, true);
    unwritten.committed(10);
    // #13 in #14: 3, which #10 wrote, then 5, read at slot 0.
    unwritten.state(13, Some(vec![(0, ReadOnly)]));
    unwritten.encoded(&bind_at(13, 3, 0), 14, true);
    unwritten.encoded(&dispatch_of(13), 14, true);
    unwritten.encoded(&bind_at(13, 5, 0), 14, true);
    unwritten.encoded(&dispatch_of(13), 14, true);
    unwritten.committed(14);
    // #15 in #16: 6 written by a dispatch, then read by the next; 7 read, then written.
    let (writes, reads) = (vec![(0, WriteOnly)], vec![(0, ReadOnly)]);
    for (id, first, second) in [(6, &writes, &reads), (7, &reads, &writes)] {
        unwritten.encoded(&bind_at(15, id, 0), 16, true);
        unwritten.state(15, Some(first.clone()));
        unwritten.encoded(&dispatch_of(15), 16, true);
        unwritten.state(15, Some(second.clone()));
        unwritten.encoded(&dispatch_of(15), 16, true);
    }
    unwritten.committed(16);
    // #17 in #18: 8 used for writing, 9 for reading; 10 named by a barrier; under a skipped
    // state, 11 bound.
    unwritten.state(17, Some(Vec::new()));
    let used = |id: u64, usage: u64| {
        encoder_call(
            "useResource:usage:",
            17,
            vec![buffer_resource(id), Value::Uint(usage)],
        )
    };
    unwritten.encoded(&used(8, 2), 18, true);
    unwritten.encoded(&used(9, 1), 18, true);
    let barrier = vec![Value::List(vec![Value::Object(10)]), Value::Uint(1)];
    let barrier = encoder_call("memoryBarrierWithResources:count:", 17, barrier);
    unwritten.encoded(&barrier, 18, true);
    unwritten.encoded(&dispatch_of(17), 18, true);
    unwritten.state(17, None);
    unwritten.encoded(&bind_at(17, 11, 0), 18, true);
    unwritten.encoded(&dispatch_of(17), 18, true);
    unwritten.committed(18);
    // 3, 6 and 8 were written, 12 and 13 never bound: none of them is counted, later either.
    for id in [3, 6, 8, 12] {
        unwritten.state(19, Some(vec![(0, ReadOnly)]));
        unwritten.encoded(&bind_at(19, id, 0), 20, true);
        unwritten.encoded(&dispatch_of(19), 20, true);
    }
    unwritten.committed(20);
    assert_eq!(
        unwritten.line(),
        unwritten_line(
            8,
            2,
            "#2 in #10, #4 in #10, #5 in #14, #7 in #16, #9 in #18, #10 in #18, #11 in #18, \
             #12 in #20"
        )
    );
}

/// A buffer `useResources:count:usage:` names is read or written by its usage's bits, each of the
/// list as `useResource:usage:` reads or writes its one (the stream records the C array and its
/// count as one list, then the usage): the two used for writing are written, not read, and the
/// two used for reading are read, surely.
#[test]
fn a_dispatch_reads_or_writes_each_buffer_a_plural_use_names_by_its_usage() {
    let mut unwritten = Unwritten::default();
    for id in 2..=5 {
        unwritten.made(id, None);
    }
    unwritten.state(12, Some(Vec::new()));
    let used = |ids: &[u64], usage: u64| {
        let list = Value::List(ids.iter().map(|&id| buffer_resource(id)).collect());
        encoder_call(
            "useResources:count:usage:",
            12,
            vec![list, Value::Uint(usage)],
        )
    };
    unwritten.encoded(&used(&[2, 3], 2), 10, true);
    unwritten.encoded(&used(&[4, 5], 1), 10, true);
    unwritten.encoded(&dispatch_of(12), 10, true);
    unwritten.committed(10);
    assert_eq!(
        unwritten.line(),
        unwritten_line(2, 0, "#4 in #10, #5 in #10")
    );
}

/// A render encoder's call reaches each buffer it names, a residency declaration's resources
/// among them as the stream records them (`{Buffer: object}`, each of a list): the buffers a
/// render encoder's `useResource:usage:stages:` and `useResources:count:usage:` name are not
/// unwritten when a later dispatch reads them; the one nothing named is.
#[test]
fn a_render_encoder_s_residency_declarations_reach_the_buffers_they_name() {
    use metal2vulkan_runtime::BindingAccess::ReadOnly;
    let mut unwritten = Unwritten::default();
    for id in 2..=4 {
        unwritten.made(id, None);
    }
    // MTLResourceUsageRead; MTLRenderStageFragment.
    let used = vec![buffer_resource(2), Value::Uint(1), Value::Uint(2)];
    unwritten.encoded(
        &encoder_call("useResource:usage:stages:", 11, used),
        10,
        false,
    );
    let listed = vec![Value::List(vec![buffer_resource(3)]), Value::Uint(1)];
    unwritten.encoded(
        &encoder_call("useResources:count:usage:", 11, listed),
        10,
        false,
    );
    unwritten.committed(10);
    let reads = (0..3).map(|slot| (slot, ReadOnly)).collect();
    unwritten.state(12, Some(reads));
    for (slot, id) in (0..).zip([2, 3, 4]) {
        unwritten.encoded(&bind_at(12, id, slot), 14, true);
    }
    unwritten.encoded(&dispatch_of(12), 14, true);
    unwritten.committed(14);
    assert_eq!(unwritten.line(), unwritten_line(1, 0, "#4 in #14"));
}

/// A buffer made in a Shared block over bytes of it that a recorded write or a GPU pass reached
/// before its making is not unwritten: the block's bytes are what the program's were. One made
/// over bytes nothing reached is.
#[test]
fn a_buffer_made_over_its_block_s_reached_bytes_is_not_unwritten() {
    use metal2vulkan_runtime::BindingAccess::ReadOnly;
    let mut unwritten = Unwritten::default();
    unwritten.block(1);
    // An m2v.write into the block before any buffer of it.
    unwritten.written(1, 0..64);
    // 2 lies in the written bytes, 3 past them, 4 over both.
    for (id, range) in [(2, 0..64), (3, 64..128), (4, 32..96)] {
        unwritten.made(id, Some((1, range)));
    }
    // A blit names 5, whose bytes 6 lies in once 5 is released.
    unwritten.made(5, Some((1, 256..512)));
    let copy = encoder_call("fillBuffer:range:value:", 11, vec![Value::Object(5)]);
    unwritten.encoded(&copy, 10, false);
    unwritten.released(5);
    unwritten.made(6, Some((1, 300..400)));
    unwritten.state(
        12,
        Some(vec![
            (0, ReadOnly),
            (1, ReadOnly),
            (2, ReadOnly),
            (3, ReadOnly),
        ]),
    );
    for (slot, id) in (0..).zip([2, 3, 4, 6]) {
        unwritten.encoded(&bind_at(12, id, slot), 10, true);
    }
    unwritten.encoded(&dispatch_of(12), 10, true);
    unwritten.committed(10);
    assert_eq!(unwritten.line(), unwritten_line(1, 0, "#3 in #10"));
}

/// A CPU write into a Managed texture counts with its user not seen complete while a committed
/// command buffer that used it (through a view) has no recorded completion, by the presents since
/// that commit; after the CPU waited for it, it counts without. A write into a texture that is not
/// Managed is not counted.
#[test]
fn a_write_into_a_managed_texture_its_user_may_still_read_is_counted() {
    let mut kept = Kept {
        managed: vec![3],
        ..Kept::default()
    };
    let region = map(&[
        ("x", Value::Uint(0)),
        ("y", Value::Uint(0)),
        ("width", Value::Uint(2)),
        ("height", Value::Uint(2)),
    ]);
    let replace = |w: &mut StreamWriter<File>, texture: u64| {
        let args = [
            region.clone(),
            Value::Uint(0),
            Value::Bytes(vec![0; 16]),
            Value::Uint(8),
        ];
        w.call(
            "replaceRegion:mipmapLevel:withBytes:bytesPerRow:",
            texture,
            0,
            &args,
            &[],
        )
    };
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let descriptor = [map(&[("width", Value::Uint(2))])];
        w.call("newTextureWithDescriptor:", 1, 3, &descriptor, &[])?;
        w.call("newTextureWithDescriptor:", 1, 4, &descriptor, &[])?;
        w.call(
            "newTextureViewWithPixelFormat:",
            3,
            5,
            &[Value::Uint(70)],
            &[],
        )?;
        w.call("commandBuffer", 2, 6, &[], &[])?;
        w.call("computeCommandEncoder", 6, 7, &[], &[])?;
        let bind = [Value::Object(5), Value::Uint(0)];
        w.call("setTexture:atIndex:", 7, 0, &bind, &[])?;
        w.call("endEncoding", 7, 0, &[], &[])?;
        w.call("commit", 6, 0, &[], &[])?;
        replace(w, 3)?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        replace(w, 3)?;
        replace(w, 4)?;
        w.call(WAIT_COMPLETED, 6, 0, &[], &[])?;
        replace(w, 3)
    });
    ended.unwrap();
    let writes = TextureWrites {
        writes: 3,
        bytes: 48,
        textures: 1,
        pending: 2,
        pending_bytes: 32,
        pending_textures: 1,
        by_presents: [1, 1, 0, 0],
        first: vec![(10, 3, 6, 0), (12, 3, 6, 1)],
        unknown: 0,
    };
    assert_eq!(report.texture_writes, writes);
    assert!(
        (report.lines()).contains(&writes.line()),
        "{:?}",
        report.lines()
    );
}

/// A player told its last frame stops after that frame's present, and one given an output writes
/// each frame's lines there as the frame ends: the report's other lines leave them out.
#[test]
fn a_play_stops_after_its_last_frame_and_writes_each_frame_as_it_ends() {
    let mut kept = Kept::default();
    let dir = ScratchDir::new("replay-until").unwrap();
    let path = written(&dir, "s.m2vr", |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        w.call("newCommandQueue", 1, 3, &[], &[])
    });
    let mut reader = StreamReader::open(&[path], None).unwrap();
    let mut out = Vec::new();
    let player = Player::new(&mut kept).with_frame_lines(&mut out);
    let (report, ended) = player.until_frame(1).play(&mut reader);
    ended.unwrap();
    assert_eq!(report.frames.len(), 2);
    assert_eq!(
        kept.calls,
        [
            "#1 = MTLCreateSystemDefaultDevice",
            "#2 = #1 newCommandQueue"
        ]
    );
    let out = String::from_utf8(out).unwrap();
    let frames: Vec<String> = (0..2).flat_map(|at| report.frame_lines(at)).collect();
    assert_eq!(out.lines().collect::<Vec<_>>(), frames, "{out}");
    assert_eq!(frames.len(), 4);
    let rest = report.lines_but_frames();
    assert!(
        !(rest.iter()).any(|line| line.starts_with("frame ") || line.starts_with("end ")),
        "{rest:?}"
    );
}

/// A play that stops after its last frame, a command buffer committed and held there on an event
/// only a later call signals, ends at the stop with that command buffer named held; played on, the
/// signal lets it complete. A stream that ends with it held, under a last frame past the end too,
/// is an invalid stream.
#[test]
fn a_play_stopped_after_its_last_frame_names_the_command_buffers_held_there() {
    let held = |w: &mut StreamWriter<File>| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("newSharedEvent", 1, 3, &[], &[])?;
        w.call("commandBuffer", 2, 4, &[], &[])?;
        w.call(
            "encodeWaitForEvent:value:",
            4,
            0,
            &[Value::Object(3), Value::Uint(1)],
            &[],
        )?;
        w.call("commit", 4, 0, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])
    };
    let signalled = |w: &mut StreamWriter<File>| {
        held(w)?;
        w.call("setSignaledValue:", 3, 0, &[Value::Uint(1)], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])
    };
    let mut kept = Kept::default();
    let (report, ended) = played_through(&mut kept, Some(0), signalled);
    ended.unwrap();
    assert_eq!(report.frames.len(), 1);
    assert_eq!(report.held, [4]);
    assert!(kept.waits.is_empty(), "{:?}", kept.waits);
    let line =
        "held #4: committed before the play stopped after its last frame, waiting on events \
                no call played signals";
    let lines = report.lines_but_frames();
    assert!(lines.iter().any(|l| l == line), "{lines:?}");

    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, signalled);
    ended.unwrap();
    assert!(report.held.is_empty(), "{:?}", report.held);
    assert_eq!(kept.waits, [4]);

    for last in [None, Some(5)] {
        let (report, ended) = played_through(&mut Kept::default(), last, held);
        assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");
        assert!(report.held.is_empty(), "{:?}", report.held);
    }
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
    assert!(
        line.contains(" gpu_busy=400 gpu_longest=200 gpu_longest_cb=3 live"),
        "{line}"
    );
}

/// A frame's end names the longest GPU interval of one command buffer it waited for, and that
/// command buffer; a frame that waited for none names 0 and 0.
#[test]
fn a_frame_end_names_its_longest_command_buffer() {
    let mut kept = Kept {
        gpu: HashMap::from([(3, (100, 300)), (4, (150, 600)), (5, (1000, 1100))]),
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
    let longest: Vec<_> = (report.ends.iter())
        .map(|end| (end.times.gpu_longest, end.times.gpu_longest_cb))
        .collect();
    assert_eq!(longest, [(450, 4), (0, 0)]);
    let lines = report.lines();
    let line = |at: usize| {
        let prefix = format!("end {at} ");
        lines.iter().find(|line| line.starts_with(&prefix)).cloned()
    };
    let first = line(0).expect("frame 0's end line");
    assert!(
        first.contains(" gpu_longest=450 gpu_longest_cb=4 live"),
        "{first}"
    );
    let second = line(1).expect("frame 1's end line");
    assert!(
        second.contains(" gpu_longest=0 gpu_longest_cb=0 live"),
        "{second}"
    );
}

/// A frame's end counts the command buffers it waited for that completed with no GPU interval,
/// or one of no length (#4), and the wait spent on them, which its GPU busy time leaves out; the
/// report sums them.
#[test]
fn a_frame_end_counts_its_waits_for_untimed_command_buffers() {
    let mut kept = Kept {
        gpu: HashMap::from([(3, (100, 300)), (4, (500, 500))]),
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
        w.call("commandBuffer", 2, 6, &[], &[])?;
        w.call("commit", 6, 0, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        Ok(())
    });
    ended.unwrap();
    let untimed: Vec<_> = (report.ends.iter())
        .map(|end| (end.times.untimed_waits, end.times.gpu_busy))
        .collect();
    assert_eq!(untimed, [(2, 200), (1, 0)]);
    for end in &report.ends {
        assert!(end.times.untimed_wait <= end.times.wait, "{end:?}");
    }
    let lines = report.lines();
    let line = lines.iter().find(|line| line.starts_with("end 0 "));
    let line = line.expect("an end line");
    assert!(line.contains(" untimed_waits=2 untimed_wait="), "{line}");
    let total = lines.iter().find(|line| line.starts_with("untimed waits "));
    let total = total.expect("the untimed waits line");
    assert!(total.starts_with("untimed waits 3: "), "{total}");
}

/// A dispatch an `m2v.privateBytes` record names plays as the program made it, and the report
/// counts it.
#[test]
fn a_dispatch_a_private_bytes_record_names_plays_as_it_is() {
    let bytes: Vec<u8> = (0..4).collect();
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
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
        w.call("m2v.privateBytes", 3, 0, &args, &[])
    });
    ended.unwrap();
    let at = kept
        .calls
        .iter()
        .position(|c| c == "#5 = #4 computeCommandEncoder")
        .expect("the encoder's making");
    let expected = [
        "#5 = #4 computeCommandEncoder",
        "#5 dispatchThreadgroups:threadsPerThreadgroup: {width: 1, height: 1} {width: 1, height: 1}",
        "#5 endEncoding",
        "#4 commit",
    ];
    assert_eq!(&kept.calls[at..at + 4], expected);
    assert_eq!(report.unsnapshotted, 1);
    assert!(report.lines().contains(&"unsnapshotted 1".to_string()));
}

/// A record must name an earlier call: else the play stops.
#[test]
fn a_private_bytes_record_names_an_earlier_call() {
    let stream = |dispatch: u64| {
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
            let range = map(&[("location", Value::Uint(0)), ("length", Value::Uint(8))]);
            let args = [
                Value::Uint(dispatch),
                range,
                Value::Blob(bytes),
                Value::Blob(bytes),
            ];
            w.call("m2v.privateBytes", 3, 0, &args, &[])
        }
    };
    let (report, ended) = played(&mut Kept::default(), stream(5));
    ended.unwrap();
    assert_eq!(report.unsnapshotted, 1);
    let (_, ended) = played(&mut Kept::default(), stream(8));
    assert!(matches!(ended, Err(PlayError::Invalid(_))), "{ended:?}");
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

/// A released buffer's place waits for each command buffer that named it and is not yet
/// committed: it is let go only when the last of them is committed or released uncommitted, never
/// on a count of commits that those users are not part of.
#[test]
fn a_place_waits_for_its_uncommitted_users() {
    let mut waiting = Waiting::default();
    // No user waits: the place goes to the commit count at once.
    assert_eq!(waiting.wait("free", Vec::new()), Some("free"));
    // Two command buffers named it, neither committed.
    assert_eq!(waiting.wait("held", vec![7, 9]), None);
    assert_eq!(waiting.wait("other", vec![9]), None);
    assert!(
        waiting.ended(8).is_empty(),
        "a command buffer that named neither"
    );
    assert!(waiting.ended(7).is_empty(), "the place still waits for 9");
    let mut done = waiting.ended(9);
    done.sort();
    assert_eq!(done, vec!["held", "other"]);
    assert!(waiting.ended(9).is_empty(), "each place is let go once");
}

/// The ranges a second copy names merge where they overlap or touch, in order, as one copy-in
/// each at the next commit.
#[test]
fn named_ranges_merge_where_they_overlap_or_touch() {
    let mut ranges = Vec::new();
    for (start, end) in [
        (256, 260),
        (0, 4),
        (4, 8),
        (100, 120),
        (110, 300),
        (512, 516),
    ] {
        merge(&mut ranges, start, end);
    }
    assert_eq!(ranges, vec![(0, 8), (100, 300), (512, 516)]);
    merge(&mut ranges, 8, 100);
    assert_eq!(ranges, vec![(0, 300), (512, 516)]);
}

/// A dispatch whose pipeline state writes texture slot 0 covers what the texture there meets in
/// its placement heap: a buffer and a texture placed at offset 0 too. A check of either is not
/// compared until it is written again: the buffer, written whole, is compared again. A dispatch
/// of a state that only reads the slot covers nothing. A binding of the buffer covered again
/// takes the cover off too, and is counted apart, as it may be a read.
#[test]
fn a_heap_texture_write_covers_the_buffer_and_texture_it_meets_until_they_are_written() {
    let mut kept = Kept::default();
    kept.heap_answers.insert(4, (0x1000, 0x1000));
    kept.kernel_writes.insert(9, vec![0]);
    kept.kernel_writes.insert(11, Vec::new());
    let encoded = |w: &mut StreamWriter<File>, at: u64, state: u64| {
        w.call("commandBuffer", 6, at, &[], &[])?;
        w.call("computeCommandEncoder", at, at + 1, &[], &[])?;
        w.call(
            "setComputePipelineState:",
            at + 1,
            0,
            &[Value::Object(state)],
            &[],
        )?;
        let slot = [Value::Object(5), Value::Uint(0)];
        w.call("setTexture:atIndex:", at + 1, 0, &slot, &[])?;
        let size = map(&[("width", Value::Uint(1)), ("height", Value::Uint(1))]);
        let groups = [size.clone(), size];
        w.call(
            "dispatchThreadgroups:threadsPerThreadgroup:",
            at + 1,
            0,
            &groups,
            &[],
        )?;
        w.call("endEncoding", at + 1, 0, &[], &[])?;
        w.call("commit", at, 0, &[], &[])
    };
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newHeapWithDescriptor:", 1, 2, &heap_of(0x1_0000, 1), &[])?;
        let buffer = [Value::Uint(0x1000), Value::Uint(0), Value::Uint(0)];
        w.call("newBufferWithLength:options:offset:", 2, 3, &buffer, &[])?;
        w.call("m2v.textureDescriptor", 0, 4, &[map(&[])], &[])?;
        let texture = [Value::Object(4), Value::Uint(0)];
        w.call("newTextureWithDescriptor:offset:", 2, 5, &texture, &[])?;
        w.call("newTextureWithDescriptor:offset:", 2, 10, &texture, &[])?;
        w.call("newCommandQueue", 1, 6, &[], &[])?;
        w.call("newComputePipelineStateWithFunction:error:", 1, 9, &[], &[])?;
        w.call(
            "newComputePipelineStateWithFunction:error:",
            1,
            11,
            &[],
            &[],
        )?;
        encoded(w, 20, 9)?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(3), Value::Object(10)],
            &[],
        )?;
        let whole = [Value::Uint(0), Value::Bytes(vec![7; 0x1000])];
        w.call("m2v.write", 3, 0, &whole, &[])?;
        encoded(w, 30, 11)?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(3), Value::Object(10)],
            &[],
        )?;
        // Covered again, then bound: the binding, which may be a read, takes the cover off.
        encoded(w, 40, 9)?;
        w.call("commandBuffer", 6, 50, &[], &[])?;
        w.call("computeCommandEncoder", 50, 51, &[], &[])?;
        let bound = [Value::Object(3), Value::Uint(0), Value::Uint(0)];
        w.call("setBuffer:offset:atIndex:", 51, 0, &bound, &[])?;
        w.call("endEncoding", 51, 0, &[], &[])?;
        w.call("commit", 50, 0, &[], &[])?;
        w.call("m2v.check", 0, 0, &[Value::Object(3)], &[])
    });
    ended.unwrap();
    let covered: Vec<(u64, bool)> = (report.checks.iter())
        .map(|check| (check.object, check.covered))
        .collect();
    assert_eq!(
        covered,
        [(3, true), (10, true), (3, false), (10, true), (3, false)]
    );
    assert_eq!(
        report.covered, 4,
        "each write covers the buffer and the other texture"
    );
    assert_eq!(report.binding_uncovers, 1);
    let lines = report.lines();
    let line = "covered 4: texture writes over other objects of a placement heap; 3 checks they \
                covered not compared; 1 covers taken off by a binding, which may be a read";
    assert!(lines.iter().any(|l| l == line), "{lines:?}");
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.starts_with("check #") && l.ends_with(" covered"))
            .count(),
        3
    );
}

/// A compute encoder's plural residency declarations, and a render encoder's with stages, take
/// covers off as their single forms do, each of the list (the stream records each C array and its
/// count as one list, then the usage, then the stages): `useResources:count:usage:` with write
/// usage writes each object it names, and `useHeaps:count:` binds each heap's objects, counted
/// apart as a binding.
#[test]
fn plural_uses_of_compute_and_render_encoders_take_covers_off_as_the_single_forms_do() {
    for render in [false, true] {
        plural_uses_take_covers_off(render);
    }
}

/// [`plural_uses_of_compute_and_render_encoders_take_covers_off_as_the_single_forms_do`] with the
/// uses on a compute encoder, or on a render encoder with `stages:` where `render`.
fn plural_uses_take_covers_off(render: bool) {
    let mut kept = Kept::default();
    kept.heap_answers.insert(4, (0x1000, 0x1000));
    kept.kernel_writes.insert(9, vec![0]);
    let encoded = |w: &mut StreamWriter<File>, at: u64| {
        w.call("commandBuffer", 6, at, &[], &[])?;
        w.call("computeCommandEncoder", at, at + 1, &[], &[])?;
        w.call(
            "setComputePipelineState:",
            at + 1,
            0,
            &[Value::Object(9)],
            &[],
        )?;
        let slot = [Value::Object(5), Value::Uint(0)];
        w.call("setTexture:atIndex:", at + 1, 0, &slot, &[])?;
        let size = map(&[("width", Value::Uint(1)), ("height", Value::Uint(1))]);
        let groups = [size.clone(), size];
        let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
        w.call(dispatch, at + 1, 0, &groups, &[])?;
        w.call("endEncoding", at + 1, 0, &[], &[])?;
        w.call("commit", at, 0, &[], &[])
    };
    let used = |w: &mut StreamWriter<File>, at: u64, selector: &str, args: &[Value]| {
        w.call("commandBuffer", 6, at, &[], &[])?;
        let mut args = args.to_vec();
        let selector = match render {
            false => {
                w.call("computeCommandEncoder", at, at + 1, &[], &[])?;
                selector.to_owned()
            }
            true => {
                w.call("MTLRenderPassDescriptor.new", 0, at + 2, &[], &[])?;
                let pass = [Value::Object(at + 2)];
                w.call(
                    "renderCommandEncoderWithDescriptor:",
                    at,
                    at + 1,
                    &pass,
                    &[],
                )?;
                // MTLRenderStageVertex | MTLRenderStageFragment.
                args.push(Value::Uint(3));
                format!("{selector}stages:")
            }
        };
        w.call(&selector, at + 1, 0, &args, &[])?;
        w.call("endEncoding", at + 1, 0, &[], &[])?;
        w.call("commit", at, 0, &[], &[])
    };
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newHeapWithDescriptor:", 1, 2, &heap_of(0x1_0000, 1), &[])?;
        let buffer = [Value::Uint(0x1000), Value::Uint(0), Value::Uint(0)];
        w.call("newBufferWithLength:options:offset:", 2, 3, &buffer, &[])?;
        w.call("m2v.textureDescriptor", 0, 4, &[map(&[])], &[])?;
        let texture = [Value::Object(4), Value::Uint(0)];
        w.call("newTextureWithDescriptor:offset:", 2, 5, &texture, &[])?;
        w.call("newCommandQueue", 1, 6, &[], &[])?;
        let made = "newComputePipelineStateWithFunction:error:";
        w.call(made, 1, 9, &[], &[])?;
        // Covered by the write of #5, then written by the plural use with write usage (2).
        encoded(w, 20)?;
        let list = Value::List(vec![buffer_resource(3)]);
        used(w, 30, "useResources:count:usage:", &[list, Value::Uint(2)])?;
        w.call("m2v.check", 0, 0, &[Value::Object(3)], &[])?;
        // Covered again, then bound by the plural use of its heap.
        encoded(w, 40)?;
        let heaps = Value::List(vec![Value::Object(2)]);
        used(w, 50, "useHeaps:count:", &[heaps])?;
        w.call("m2v.check", 0, 0, &[Value::Object(3)], &[])
    });
    ended.unwrap();
    let covered: Vec<(u64, bool)> = (report.checks.iter())
        .map(|check| (check.object, check.covered))
        .collect();
    assert_eq!(covered, [(3, false), (3, false)], "render {render}");
    assert_eq!(
        report.covered, 2,
        "each write of #5 covers #3; render {render}"
    );
    let uncovers = report.binding_uncovers;
    assert_eq!(uncovers, 1, "the heap's use is a binding; render {render}");
}

/// A texture bound through a range takes the slot of the range's location plus its place in the
/// list: a dispatch of a state that writes slot 1 covers what the texture meets in its placement
/// heap where the range put it at slot 1, and nothing where it put it at slot 0.
#[test]
fn a_texture_bound_through_a_range_takes_the_slot_its_location_names() {
    let mut kept = Kept::default();
    kept.heap_answers.insert(4, (0x1000, 0x1000));
    kept.kernel_writes.insert(9, vec![1]);
    let encoded = |w: &mut StreamWriter<File>, at: u64, location: u64| {
        w.call("commandBuffer", 6, at, &[], &[])?;
        w.call("computeCommandEncoder", at, at + 1, &[], &[])?;
        let state = [Value::Object(9)];
        w.call("setComputePipelineState:", at + 1, 0, &state, &[])?;
        let range = map(&[
            ("location", Value::Uint(location)),
            ("length", Value::Uint(1)),
        ]);
        let slots = [Value::List(vec![Value::Object(5)]), range];
        w.call("setTextures:withRange:", at + 1, 0, &slots, &[])?;
        let size = map(&[("width", Value::Uint(1)), ("height", Value::Uint(1))]);
        let groups = [size.clone(), size];
        let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
        w.call(dispatch, at + 1, 0, &groups, &[])?;
        w.call("endEncoding", at + 1, 0, &[], &[])?;
        w.call("commit", at, 0, &[], &[])
    };
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newHeapWithDescriptor:", 1, 2, &heap_of(0x1_0000, 1), &[])?;
        let buffer = [Value::Uint(0x1000), Value::Uint(0), Value::Uint(0)];
        w.call("newBufferWithLength:options:offset:", 2, 3, &buffer, &[])?;
        w.call("m2v.textureDescriptor", 0, 4, &[map(&[])], &[])?;
        let texture = [Value::Object(4), Value::Uint(0)];
        w.call("newTextureWithDescriptor:offset:", 2, 5, &texture, &[])?;
        w.call("newCommandQueue", 1, 6, &[], &[])?;
        let state = "newComputePipelineStateWithFunction:error:";
        w.call(state, 1, 9, &[], &[])?;
        encoded(w, 20, 0)?;
        w.call("m2v.check", 0, 0, &[Value::Object(3)], &[])?;
        encoded(w, 30, 1)?;
        w.call("m2v.check", 0, 0, &[Value::Object(3)], &[])
    });
    ended.unwrap();
    let covered: Vec<bool> = report.checks.iter().map(|check| check.covered).collect();
    assert_eq!(covered, [false, true]);
}

/// A buffer a texture over a surface's plane was made on stands for the surface's pages: its checks
/// are compared on no runtime, the texture's are, and once the buffer is released and made again
/// as an ordinary buffer, its checks are compared again.
#[test]
fn a_surface_s_receiver_buffer_is_not_compared() {
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let pages = [Value::Uint(0x1000), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 2, &pages, &[])?;
        w.call("m2v.textureDescriptor", 0, 3, &[map(&[])], &[])?;
        let plane = map(&[
            ("plane", Value::Uint(0)),
            ("offset", Value::Uint(0)),
            ("bytesPerRow", Value::Uint(256)),
            ("planar", Value::Bool(false)),
            ("facts", Value::Nil),
        ]);
        let made = "newTextureWithDescriptor:iosurface:plane:";
        w.call(made, 2, 4, &[Value::Object(3), plane], &[])?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(2), Value::Object(4)],
            &[],
        )?;
        w.call("release", 4, 0, &[], &[])?;
        w.call("release", 2, 0, &[], &[])?;
        w.call("newBufferWithLength:options:", 1, 2, &pages, &[])?;
        w.call("m2v.check", 0, 0, &[Value::Object(2)], &[])
    });
    ended.unwrap();
    let receivers: Vec<(u64, bool)> = (report.checks.iter())
        .map(|check| (check.object, check.receiver))
        .collect();
    assert_eq!(receivers, [(2, true), (4, false), (2, false)]);
    let lines = report.lines();
    let line = "receivers: 1 checks of surfaces' receiver buffers not compared; the textures \
                over the surfaces hold the texels";
    assert!(lines.iter().any(|l| l == line), "{lines:?}");
    assert!(lines
        .iter()
        .any(|l| l.starts_with("check #2 ") && l.ends_with(" receiver")));
}

/// A listed frame's present checks the texture it shows, at that present, as a check with no
/// recorded checksum: a check of that frame, which two plays' digests compare by
/// (`compare_plays`); an unlisted present, and a listed one of nil, read nothing.
#[test]
fn a_listed_present_checks_the_shown_texture() {
    let build = |w: &mut StreamWriter<File>| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newTextureWithDescriptor:", 1, 2, &[map(&[])], &[])?;
        for shown in [
            Value::Object(2),
            Value::Object(2),
            Value::Nil,
            Value::Object(2),
        ] {
            w.call("m2v.present", 1, 0, &[shown], &[])?;
        }
        Ok(())
    };
    let dir = ScratchDir::new("replay-presents").unwrap();
    let path = written(&dir, "s.m2vr", build);
    let play = |frames: &[u64]| {
        let mut kept = Kept::default();
        let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
        let player = Player::new(&mut kept).checking_presents(frames.iter().copied());
        let (report, ended) = player.play(&mut reader);
        ended.unwrap();
        report
    };
    assert!(
        play(&[]).checks.is_empty(),
        "a present reads nothing unlisted"
    );
    let report = play(&[1, 2, 3]);
    let checked: Vec<(u64, u64, Option<[u8; 32]>)> = (report.checks.iter())
        .map(|check| (check.object, check.frame, check.recorded))
        .collect();
    assert_eq!(checked, [(2, 1, None), (2, 3, None)]);
    // The backend's bytes of the texture: `Kept` reads an object it holds no memory of as its id.
    let sha: [u8; 32] = Sha256::digest(2u64.to_le_bytes()).into();
    let line = format!("check #2 frame 1 {} unrecorded", hex(&sha));
    let lines = report.lines();
    assert!(lines.contains(&line), "{lines:?}");
    assert_eq!(report.frames.len(), 4, "every present still ends its frame");
}

/// A listed present the player keeps (`Player::keeping_presents`): the shown texture's bytes go to
/// `frame-F.bytes` in the directory, the report names it with its shape (unknown to a backend
/// that gives none) and every frame's end carries the system's clock.
#[test]
fn a_kept_present_writes_the_shown_texture_s_bytes_and_each_end_its_time() {
    let build = |w: &mut StreamWriter<File>| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newTextureWithDescriptor:", 1, 2, &[map(&[])], &[])?;
        for _ in 0..2 {
            w.call("m2v.present", 1, 0, &[Value::Object(2)], &[])?;
        }
        Ok(())
    };
    let dir = ScratchDir::new("replay-kept-presents").unwrap();
    let path = written(&dir, "s.m2vr", build);
    let kept_dir = dir.path().join("presents");
    let mut kept = Kept::default();
    let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
    let player = (Player::new(&mut kept).checking_presents([1])).keeping_presents(kept_dir.clone());
    let (report, ended) = player.play(&mut reader);
    ended.unwrap();
    let shown = Shown {
        object: 2,
        frame: 1,
        shape: None,
    };
    assert_eq!(report.shown, [shown]);
    assert_eq!(
        std::fs::read(kept_dir.join("frame-1.bytes")).unwrap(),
        2u64.to_le_bytes(),
        "`Kept` reads an object it holds no memory of as its id"
    );
    assert!(
        !kept_dir.join("frame-0.bytes").exists(),
        "an unlisted present keeps nothing"
    );
    let lines = report.lines();
    assert!(
        lines.contains(&"shown #2 frame 1 shape unknown".to_owned()),
        "{lines:?}"
    );
    let ends: Vec<&String> = lines
        .iter()
        .filter(|line| line.starts_with("end "))
        .collect();
    assert_eq!(ends.len(), 2, "{lines:?}");
    for end in ends {
        let time = end
            .split_whitespace()
            .find_map(|word| word.strip_prefix("unix_nanos="));
        let time: u64 = time.expect("a time").parse().unwrap();
        assert!(time > 1_700_000_000_000_000_000, "{end}");
    }
}

/// A picture difference's size: the texels that differ and their share, the largest difference
/// of each channel in the format's units, in its channels' order, and the box the differing
/// texels lie in; a packed format reads its channels from its words; with no shape, bytes.
#[test]
fn a_picture_difference_is_sized_by_texels_channels_and_box() {
    // BGRA8Unorm, 3 by 2: texel (1, 0) is 7 redder, texel (2, 1) 2 bluer.
    let a = vec![0u8; 24];
    let mut b = a.clone();
    b[4 + 2] = 7;
    b[12 + 8] = 2;
    let bgra = Shape {
        pixel_format: 80,
        width: 3,
        height: 2,
    };
    assert_eq!(
        difference(Some(bgra), &a, &b),
        "2 of 6 texels of format 80 3x2 differ (33.33%); largest difference per channel B 2, G 0, \
         R 7, A 0 (levels of 1/255); box x 1..2, y 0..1"
    );
    // RGB10A2Unorm, one texel: red from 0 to 1023, alpha from 0 to 3.
    let packed = Shape {
        pixel_format: 90,
        width: 1,
        height: 1,
    };
    let word = (0x3ffu32 | (0x3 << 30)).to_le_bytes();
    assert_eq!(
        difference(Some(packed), &[0; 4], &word),
        "1 of 1 texels of format 90 1x1 differ (100.00%); largest difference per channel R 1023, \
         G 0, B 0, A 3 (levels of 1/1023, A of 1/3); box x 0..0, y 0..0"
    );
    assert_eq!(
        difference(None, &[1, 2, 3], &[1, 5, 3]),
        "1 of 3 bytes differ (33.33%); largest difference per channel byte 3 (bytes); box x 1..1, \
         y 0..0"
    );
    assert_eq!(
        difference(None, &[1], &[1, 2]),
        "sizes differ: a 1 bytes, b 2 bytes"
    );
}

/// The compare sizes a frame's picture difference where both reports name the shown texture they
/// kept and the frame's digests differ; equal digests print nothing.
#[test]
fn two_plays_kept_pictures_that_differ_are_sized() {
    let dir = ScratchDir::new("replay-shown-differences").unwrap();
    let (a_kept, b_kept) = (dir.path().join("a"), dir.path().join("b"));
    for (kept, red) in [(&a_kept, 0u8), (&b_kept, 9)] {
        std::fs::create_dir_all(kept).unwrap();
        std::fs::write(kept.join("frame-1.bytes"), [0, 0, red, 255, 0, 0, 0, 255]).unwrap();
        std::fs::write(kept.join("frame-2.bytes"), [1; 8]).unwrap();
    }
    let report = |frame_1: &str| {
        format!(
            "shown #4 frame 1 format 80 width 2 height 1\ncheck #4 frame 1 {frame_1} unrecorded\n\
             shown #4 frame 2 format 80 width 2 height 1\ncheck #4 frame 2 cc unrecorded\n"
        )
    };
    let lines = shown_differences(&report("aa"), &report("bb"), &a_kept, &b_kept);
    assert_eq!(
        lines,
        ["pixels at frame 1: 1 of 2 texels of format 80 2x1 differ (50.00%); largest difference \
          per channel B 0, G 0, R 9, A 0 (levels of 1/255); box x 0..0, y 0..0"]
    );
}

/// A play of `build` on `backend` that dumps command buffers 20 to 29 into `dir`.
fn dumped(
    backend: &mut Kept,
    dir: &std::path::Path,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> (Report, Result<(), PlayError>) {
    dumped_listed(backend, dir, "20..29", build)
}

/// A play of `build` on `backend` that dumps the command buffers `listed` names, as
/// `--dump-command-buffers` lists them, into `dir`.
fn dumped_listed(
    backend: &mut Kept,
    dir: &std::path::Path,
    listed: &str,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> (Report, Result<(), PlayError>) {
    let streams = ScratchDir::new("replay-dumped").unwrap();
    let path = written(&streams, "s.m2vr", build);
    let mut reader = StreamReader::open(&[path], None).unwrap();
    let listed = CommandBufferList::parse(listed).unwrap();
    let player = Player::new(backend).dumping(listed, dir.to_path_buf());
    player.play(&mut reader)
}

/// `--dump-command-buffers` lists `FIRST..LAST` ranges and single ids joined by commas, in any
/// order: the list holds each and nothing between them. An empty item, a range with an end
/// missing or its first id past its last, or any other text is no list.
#[test]
fn a_dump_list_holds_its_ranges_and_ids_and_nothing_between() {
    let list = CommandBufferList::parse("30,20..22,23,40..40").unwrap();
    let held: Vec<u64> = (0..50).filter(|&id| list.contains(id)).collect();
    assert_eq!(held, [20, 21, 22, 23, 30, 40]);
    for bad in ["", "20,", "20..", "..22", "x", "22..20", "20;22", "-1"] {
        assert_eq!(CommandBufferList::parse(bad), None, "{bad:?}");
    }
}

/// A play dumps the listed command buffers alone: of command buffers 20, 22 and 24, each naming
/// buffer #3, the list `20,24..24` dumps 20's and 24's and not 22's.
#[test]
fn a_play_dumps_the_listed_command_buffers_alone() {
    let dir = ScratchDir::new("dumps-listed").unwrap();
    let (_, ended) = dumped_listed(&mut Kept::default(), dir.path(), "20,24..24", |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let plain = [Value::Uint(16), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 3, &plain, &[])?;
        w.call("newCommandQueue", 1, 6, &[], &[])?;
        for command_buffer in [20, 22, 24] {
            let encoder = command_buffer + 1;
            w.call("commandBuffer", 6, command_buffer, &[], &[])?;
            w.call("computeCommandEncoder", command_buffer, encoder, &[], &[])?;
            let bound = [Value::Object(3), Value::Uint(0), Value::Uint(0)];
            w.call("setBuffer:offset:atIndex:", encoder, 0, &bound, &[])?;
            w.call("endEncoding", encoder, 0, &[], &[])?;
            w.call("commit", command_buffer, 0, &[], &[])?;
        }
        Ok(())
    });
    ended.unwrap();
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let wanted = [
        "cb20-obj3.bin",
        "cb20-obj3.cover",
        "cb24-obj3.bin",
        "cb24-obj3.cover",
    ];
    assert_eq!(names, wanted);
}

/// The dump tests' play of `bytes` into buffer #3 (0x2000 bytes at offset 0 of a placement heap,
/// with a 0x1000-byte texture #5 at offset 0 too), into `dir`: command buffer #20 binds the
/// buffer, then dispatches a state that writes the texture's slot, which covers the buffer's first
/// 0x1000 bytes. Where `rebound`, the encoder binds the buffer again after the dispatch: a binding
/// counts as a write, so the buffer's last write follows the cover, and the cover is void.
fn play_covered(dir: &std::path::Path, bytes: Vec<u8>, rebound: bool) {
    let mut kept = Kept::default();
    kept.heap_answers.insert(4, (0x1000, 0x1000));
    kept.kernel_writes.insert(9, vec![0]);
    let (_, ended) = dumped(&mut kept, dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newHeapWithDescriptor:", 1, 2, &heap_of(0x1_0000, 1), &[])?;
        let buffer = [Value::Uint(0x2000), Value::Uint(0), Value::Uint(0)];
        w.call("newBufferWithLength:options:offset:", 2, 3, &buffer, &[])?;
        w.call("m2v.textureDescriptor", 0, 4, &[map(&[])], &[])?;
        let texture = [Value::Object(4), Value::Uint(0)];
        w.call("newTextureWithDescriptor:offset:", 2, 5, &texture, &[])?;
        w.call("newCommandQueue", 1, 6, &[], &[])?;
        w.call("newComputePipelineStateWithFunction:error:", 1, 9, &[], &[])?;
        let written = [Value::Uint(0), Value::Bytes(bytes)];
        w.call("m2v.write", 3, 0, &written, &[])?;
        w.call("commandBuffer", 6, 20, &[], &[])?;
        w.call("computeCommandEncoder", 20, 21, &[], &[])?;
        let state = [Value::Object(9)];
        w.call("setComputePipelineState:", 21, 0, &state, &[])?;
        let bound = [Value::Object(3), Value::Uint(0), Value::Uint(0)];
        w.call("setBuffer:offset:atIndex:", 21, 0, &bound, &[])?;
        let slot = [Value::Object(5), Value::Uint(0)];
        w.call("setTexture:atIndex:", 21, 0, &slot, &[])?;
        let size = map(&[("width", Value::Uint(1)), ("height", Value::Uint(1))]);
        let groups = [size.clone(), size];
        let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
        w.call(dispatch, 21, 0, &groups, &[])?;
        if rebound {
            w.call("setBuffer:offset:atIndex:", 21, 0, &bound, &[])?;
        }
        w.call("endEncoding", 21, 0, &[], &[])?;
        w.call("commit", 20, 0, &[], &[])
    });
    ended.unwrap();
}

/// Two plays' buffer #3, alike but for the words at bytes 0x10 and 0x1800.
fn two_dumps(rebound: bool) -> (ScratchDir, ScratchDir) {
    let (a, b) = (
        ScratchDir::new("dumps-a").unwrap(),
        ScratchDir::new("dumps-b").unwrap(),
    );
    let mut other = vec![0u8; 0x2000];
    other[0x10] = 1;
    other[0x1800] = 1;
    play_covered(a.path(), vec![0; 0x2000], rebound);
    play_covered(b.path(), other, rebound);
    (a, b)
}

/// The per-command-buffer compare of two plays (`compare_dumps`) leaves out a differing word
/// under a cover and reports one outside every cover: a texture's write covered the first 0x1000
/// bytes of buffer #3 after the command buffer bound it, so of the two words that differ, the one
/// at 0x10 is covered and the one at 0x1800 (word 0x600) is reported, and the plays differ.
#[test]
fn the_dump_compare_reports_a_differing_word_outside_every_cover() {
    let (a, b) = two_dumps(false);
    let cover = std::fs::read_to_string(a.path().join("cb20-obj3.cover")).unwrap();
    assert_eq!(cover, "0..4096");
    let comparison = compare_dumps(a.path(), b.path()).unwrap();
    assert_eq!(comparison.compared, 1);
    let difference = Difference {
        command_buffer: 20,
        object: 3,
        differing: 2,
        covered: 1,
        first_uncovered: Some(0x600),
        lengths: (0x2000, 0x2000),
    };
    assert_eq!(comparison.differences, [difference]);
    assert_eq!(comparison.uncovered(), 1);
    assert!(!comparison.agrees());
    let lines = comparison.lines();
    assert_eq!(
        lines,
        [
            "differs cb #20 obj #3: 1 words outside every cover, the first word 1536; 1 covered; \
             lengths 8192 and 8192",
            "compared 1 buffers (0 as both plays' CPU copies, 0 as their device copies, read \
             back): 1 differ, 1 words outside every cover, 1 covered; 0 dumps in one play only, 0 \
             unread, 0 of different copies: DIFFER",
        ]
    );
}

/// A cover the buffer was written after is void, and the compare reports what differs under it:
/// the command buffer binds buffer #3 again after the texture's write covered it, a binding counts
/// as a write, so the dump holds no cover and both differing words are reported.
#[test]
fn the_dump_compare_reports_a_word_whose_cover_the_buffer_s_last_write_voided() {
    let (a, b) = two_dumps(true);
    let cover = std::fs::read_to_string(a.path().join("cb20-obj3.cover")).unwrap();
    assert_eq!(cover, "-");
    let comparison = compare_dumps(a.path(), b.path()).unwrap();
    let difference = Difference {
        command_buffer: 20,
        object: 3,
        differing: 2,
        covered: 0,
        first_uncovered: Some(4),
        lengths: (0x2000, 0x2000),
    };
    assert_eq!(comparison.differences, [difference]);
    assert_eq!(comparison.uncovered(), 2);
    assert!(!comparison.agrees());
}

/// A listed command buffer's dump holds every buffer its encoders named, whatever made it: a
/// plain buffer, a buffer in a block and a host's padded buffer; a texture it names is no dump.
#[test]
fn a_listed_command_buffer_dumps_every_buffer_it_names_whatever_made_it() {
    let dir = ScratchDir::new("dumps-makers").unwrap();
    let (_, ended) = dumped(&mut Kept::default(), dir.path(), |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let block = [Value::Uint(0x1_0000), Value::Uint(0)];
        let at = addressed(BLOCK, 0x1_0000);
        w.call("m2v.newBlockWithLength:storageMode:", 1, 2, &block, &at)?;
        let plain = [Value::Uint(16), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 3, &plain, &[])?;
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
            4,
            &inner,
            &at,
        )?;
        let padded = [Value::Uint(64), Value::Uint(0), Value::Nil];
        w.call("m2v.newBufferOfLengthPaddingUnobserved", 1, 5, &padded, &[])?;
        w.call("newCommandQueue", 1, 6, &[], &[])?;
        w.call("m2v.textureDescriptor", 0, 7, &[map(&[])], &[])?;
        w.call("newTextureWithDescriptor:", 1, 8, &[Value::Object(7)], &[])?;
        w.call("commandBuffer", 6, 20, &[], &[])?;
        w.call("computeCommandEncoder", 20, 21, &[], &[])?;
        for (buffer, index) in [(3, 0), (4, 1), (5, 2)] {
            let bound = [Value::Object(buffer), Value::Uint(0), Value::Uint(index)];
            w.call("setBuffer:offset:atIndex:", 21, 0, &bound, &[])?;
        }
        let slot = [Value::Object(8), Value::Uint(0)];
        w.call("setTexture:atIndex:", 21, 0, &slot, &[])?;
        w.call("endEncoding", 21, 0, &[], &[])?;
        w.call("commit", 20, 0, &[], &[])
    });
    ended.unwrap();
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let dumps = ["cb20-obj3", "cb20-obj4", "cb20-obj5"];
    let wanted: Vec<String> = (dumps.iter())
        .flat_map(|dump| [format!("{dump}.bin"), format!("{dump}.cover")])
        .collect();
    assert_eq!(names, wanted);
}

/// A listed command buffer's dump holds each colour attachment its render passes stored, the parts
/// of it they stored one after another, and no other attachment: of command buffer #20's two
/// passes, the first stores texture #8 (attachment 0, `MTLStoreActionStore`), not #9 (1,
/// `MTLStoreActionDontCare`), the resolve texture #11's level 1 and slice 2 of multisample #10,
/// not #10 itself (2, `MTLStoreActionMultisampleResolve`), #12's level 1 by the encoder's
/// `setColorStoreAction:atIndex:` (3, `MTLStoreActionUnknown`), and no depth attachment (#13);
/// the second stores #8's level 2 and depth plane 3. A texture's dump covers nothing.
#[test]
fn a_listed_command_buffer_dumps_the_colour_attachments_its_render_passes_store() {
    let dir = ScratchDir::new("dumps-stores").unwrap();
    let (_, ended) = dumped(&mut Kept::default(), dir.path(), |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 6, &[], &[])?;
        w.call("m2v.textureDescriptor", 0, 7, &[map(&[])], &[])?;
        for texture in 8..=13 {
            w.call(
                "newTextureWithDescriptor:",
                1,
                texture,
                &[Value::Object(7)],
                &[],
            )?;
        }
        let colour = |w: &mut StreamWriter<File>, pass, at, set: &[(&str, Value)]| {
            let args = [Value::Uint(at), map(set)];
            w.call("m2v.setColorAttachment", pass, 0, &args, &[])
        };
        w.call("MTLRenderPassDescriptor.new", 0, 30, &[], &[])?;
        let stored = [
            ("texture", Value::Object(8)),
            ("storeAction", Value::Uint(1)),
        ];
        colour(w, 30, 0, &stored)?;
        let dropped = [
            ("texture", Value::Object(9)),
            ("storeAction", Value::Uint(0)),
        ];
        colour(w, 30, 1, &dropped)?;
        let resolved = [
            ("texture", Value::Object(10)),
            ("resolveTexture", Value::Object(11)),
            ("resolveLevel", Value::Uint(1)),
            ("resolveSlice", Value::Uint(2)),
            ("storeAction", Value::Uint(2)),
        ];
        colour(w, 30, 2, &resolved)?;
        let unknown = [
            ("texture", Value::Object(12)),
            ("level", Value::Uint(1)),
            ("storeAction", Value::Uint(4)),
        ];
        colour(w, 30, 3, &unknown)?;
        let depth = map(&[
            ("texture", Value::Object(13)),
            ("storeAction", Value::Uint(1)),
        ]);
        w.call("m2v.setDepthAttachment", 30, 0, &[depth], &[])?;
        w.call("MTLRenderPassDescriptor.new", 0, 31, &[], &[])?;
        let level = [
            ("texture", Value::Object(8)),
            ("level", Value::Uint(2)),
            ("depthPlane", Value::Uint(3)),
            ("storeAction", Value::Uint(1)),
        ];
        colour(w, 31, 0, &level)?;
        w.call("commandBuffer", 6, 20, &[], &[])?;
        w.call(
            "renderCommandEncoderWithDescriptor:",
            20,
            21,
            &[Value::Object(30)],
            &[],
        )?;
        let store = [Value::Uint(1), Value::Uint(3)];
        w.call("setColorStoreAction:atIndex:", 21, 0, &store, &[])?;
        w.call("endEncoding", 21, 0, &[], &[])?;
        w.call(
            "renderCommandEncoderWithDescriptor:",
            20,
            22,
            &[Value::Object(31)],
            &[],
        )?;
        w.call("endEncoding", 22, 0, &[], &[])?;
        w.call("commit", 20, 0, &[], &[])
    });
    ended.unwrap();
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let dumps = ["cb20-obj11", "cb20-obj12", "cb20-obj8"];
    let wanted: Vec<String> = (dumps.iter())
        .flat_map(|dump| [format!("{dump}.bin"), format!("{dump}.cover")])
        .collect();
    assert_eq!(names, wanted);
    let read = |name: &str| std::fs::read(dir.path().join(name)).unwrap();
    let parts = |parts: &[[u64; 4]]| -> Vec<u8> {
        (parts.iter().flatten())
            .flat_map(|word| word.to_le_bytes())
            .collect()
    };
    assert_eq!(read("cb20-obj8.bin"), parts(&[[8, 0, 0, 0], [8, 2, 0, 3]]));
    assert_eq!(read("cb20-obj11.bin"), parts(&[[11, 1, 2, 0]]));
    assert_eq!(read("cb20-obj12.bin"), parts(&[[12, 1, 0, 0]]));
    for dump in dumps {
        assert_eq!(read(&format!("{dump}.cover")), b"-");
    }
}

/// Two plays alike agree; a dump one play has alone is reported.
#[test]
fn the_dump_compare_agrees_on_alike_plays_and_reports_a_dump_one_play_lacks() {
    let (a, b) = (
        ScratchDir::new("dumps-a").unwrap(),
        ScratchDir::new("dumps-b").unwrap(),
    );
    play_covered(a.path(), vec![5; 0x2000], false);
    play_covered(b.path(), vec![5; 0x2000], false);
    let comparison = compare_dumps(a.path(), b.path()).unwrap();
    assert_eq!((comparison.compared, comparison.agrees()), (1, true));
    std::fs::write(a.path().join("cb21-obj3.bin"), [0u8; 4]).unwrap();
    let comparison = compare_dumps(a.path(), b.path()).unwrap();
    assert_eq!(comparison.only_one, [("cb21-obj3".to_owned(), 'a')]);
    assert!(!comparison.agrees());
}

/// A play whose backend reads buffer #3 as `copy` of its two copies and keeps one copy of #4:
/// command buffer #20 binds both, then the stream checks both; dumped into `dir`.
fn play_copies(dir: &std::path::Path, copy: ReadCopy) -> Report {
    let mut kept = Kept::default();
    kept.read_copies.insert(3, copy);
    let (report, ended) = dumped(&mut kept, dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        for buffer in [3, 4] {
            let made = [Value::Uint(16), Value::Uint(0)];
            w.call("newBufferWithLength:options:", 1, buffer, &made, &[])?;
        }
        w.call("newCommandQueue", 1, 6, &[], &[])?;
        w.call("commandBuffer", 6, 20, &[], &[])?;
        w.call("computeCommandEncoder", 20, 21, &[], &[])?;
        for (buffer, index) in [(3, 0), (4, 1)] {
            let bound = [Value::Object(buffer), Value::Uint(0), Value::Uint(index)];
            w.call("setBuffer:offset:atIndex:", 21, 0, &bound, &[])?;
        }
        w.call("endEncoding", 21, 0, &[], &[])?;
        w.call("commit", 20, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 20, 0, &[], &[])?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(3), Value::Object(4)],
            &[],
        )
    });
    ended.unwrap();
    report
}

/// A read of a buffer with two copies says which copy it took, in its check line and in its dump,
/// and the dump compare compares only the same copy of both plays: #3's check line ends `(CPU
/// copy)` or `(device copy, read back)` and its dump holds a `.copy` file with those words; #4,
/// one copy, has neither. Two plays that read #3's CPU copy compare it as such; a play that read
/// its device copy against one that read its CPU copy differs, #3 not compared.
#[test]
fn a_read_of_a_buffer_with_two_copies_says_which_copy_it_took() {
    let dirs = ["cpu-a", "cpu-b", "device"].map(|name| ScratchDir::new(name).unwrap());
    let copies = [ReadCopy::Cpu, ReadCopy::Cpu, ReadCopy::Device];
    let reports: Vec<Report> = (dirs.iter().zip(copies))
        .map(|(dir, copy)| play_copies(dir.path(), copy))
        .collect();
    let checks = |report: &Report| -> Vec<String> {
        let lines = report.lines().into_iter();
        let checks = lines.filter(|line| line.starts_with("check "));
        checks
            .map(|line| line.split(' ').skip(6).collect::<Vec<_>>().join(" "))
            .collect()
    };
    assert_eq!(checks(&reports[0]), ["(CPU copy)", ""]);
    assert_eq!(checks(&reports[2]), ["(device copy, read back)", ""]);
    let copy_file = |dir: &ScratchDir, name: &str| std::fs::read_to_string(dir.path().join(name));
    assert_eq!(copy_file(&dirs[0], "cb20-obj3.copy").unwrap(), "CPU copy");
    assert_eq!(
        copy_file(&dirs[2], "cb20-obj3.copy").unwrap(),
        "device copy, read back"
    );
    assert!(copy_file(&dirs[0], "cb20-obj4.copy").is_err());
    let alike = compare_dumps(dirs[0].path(), dirs[1].path()).unwrap();
    assert_eq!(
        (alike.compared, alike.cpu_copies, alike.agrees()),
        (2, 1, true)
    );
    let crossed = compare_dumps(dirs[0].path(), dirs[2].path()).unwrap();
    assert_eq!((crossed.compared, crossed.agrees()), (1, false));
    let differ = "copies differ: cb20-obj3: a read its CPU copy, b its device copy, read back";
    assert_eq!(crossed.lines()[0], differ);
}

/// A Managed buffer descriptor of `length` bytes made with `contents`, a no-copy one where
/// `no_copy`.
fn managed_descriptor(length: u64, contents: Value, no_copy: bool) -> [Value; 1] {
    let managed = Value::List(vec![map(&[("ResourceOptions", Value::Uint(16))])]);
    let fields = [
        ("length", Value::Uint(length)),
        ("options", managed),
        ("contents", contents),
        ("noCopy", Value::Bool(no_copy)),
    ];
    [map(&fields)]
}

/// A command buffer of queue #2, `command_buffer`, whose compute encoder `encoder` binds
/// `buffers`, committed.
fn bound(
    w: &mut StreamWriter<File>,
    command_buffer: u64,
    encoder: u64,
    buffers: &[u64],
) -> std::io::Result<()> {
    w.call("commandBuffer", 2, command_buffer, &[], &[])?;
    w.call("computeCommandEncoder", command_buffer, encoder, &[], &[])?;
    for (at, &buffer) in buffers.iter().enumerate() {
        let args = [
            Value::Object(buffer),
            Value::Uint(0),
            Value::Uint(at as u64),
        ];
        w.call("setBuffer:offset:atIndex:", encoder, 0, &args, &[])?;
    }
    w.call("endEncoding", encoder, 0, &[], &[])?;
    w.call("commit", command_buffer, 0, &[], &[])
}

/// The GPU reads of Managed bytes no announcement took to the GPU copy, counted from the stream
/// alone, under a two-copy identity record: a copied buffer's contents read twice before any
/// announcement (made-with, served), and a no-copy buffer's write read once before its
/// `didModifyRange:` and once after (stale once).
#[test]
fn a_two_copy_stream_s_managed_reads_are_counted_by_kind() {
    let mut kept = Kept::default();
    let write = [Value::Uint(0), Value::Bytes(vec![7; 4])];
    let (report, ended) = played(&mut kept, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let identity = [("hasUnifiedMemory", Value::Bool(false))];
        w.call("m2v.deviceIdentity", 1, 0, &[], &identity)?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let copied = managed_descriptor(4, Value::Bytes(vec![9; 4]), false);
        w.call("newBufferWithDescriptor:", 1, 3, &copied, &[])?;
        let no_copy = managed_descriptor(4, Value::Nil, true);
        w.call("newBufferWithDescriptor:", 1, 4, &no_copy, &[])?;
        w.call("m2v.write", 4, 0, &write, &[])?;
        bound(w, 10, 11, &[3, 4])?;
        w.call("m2v.present", 0, 0, &[], &[])?;
        let whole = map(&[("location", Value::Uint(0)), ("length", Value::Uint(4))]);
        w.call("didModifyRange:", 4, 0, &[whole], &[])?;
        bound(w, 12, 13, &[3, 4])
    });
    ended.unwrap();
    let reads = report.managed_reads;
    assert_eq!((reads.made_with.objects, reads.made_with.uses), (1, 2));
    assert_eq!((reads.stale.objects, reads.stale.uses), (1, 1));
    let line = "managed reads on two copies: made-with 1 objects, 2 uses, first at frame 0, \
                served; stale 1 objects, 1 uses, first at frame 0, played as recorded";
    assert!(
        report.lines().iter().any(|l| l == line),
        "{:?}",
        report.lines()
    );
}

/// With no identity record, a no-copy Managed buffer's writes before the command buffer that
/// first names it commits are served, not stale; a write after that commit is stale. With a
/// one-copy identity record nothing is counted.
#[test]
fn a_no_copy_buffer_s_writes_before_first_use_are_not_stale_with_no_identity() {
    let write = [Value::Uint(0), Value::Bytes(vec![7; 4])];
    let stream = |w: &mut StreamWriter<File>, identity: Option<bool>| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        if let Some(unified) = identity {
            let answers = [("hasUnifiedMemory", Value::Bool(unified))];
            w.call("m2v.deviceIdentity", 1, 0, &[], &answers)?;
        }
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let no_copy = managed_descriptor(4, Value::Nil, true);
        w.call("newBufferWithDescriptor:", 1, 4, &no_copy, &[])?;
        w.call("m2v.write", 4, 0, &write, &[])?;
        bound(w, 10, 11, &[4])?;
        w.call("m2v.present", 0, 0, &[], &[])?;
        w.call("m2v.write", 4, 0, &write, &[])?;
        bound(w, 12, 13, &[4])
    };
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| stream(w, None));
    ended.unwrap();
    let reads = report.managed_reads;
    assert_eq!(reads.stale.first_frame, Some(1));
    assert_eq!((reads.stale.objects, reads.stale.uses), (1, 1));
    let line = "managed reads on two copies: made-with none, served; stale 1 objects, 1 uses, \
                first at frame 1, played as recorded; writes before first use served (no identity \
                record)";
    assert!(
        report.lines().iter().any(|l| l == line),
        "{:?}",
        report.lines()
    );
    let mut kept = Kept::default();
    let (report, ended) = played(&mut kept, |w| stream(w, Some(true)));
    ended.unwrap();
    let line = "managed reads on two copies: none counted: the stream's identity record keeps one \
                copy";
    assert!(
        report.lines().iter().any(|l| l == line),
        "{:?}",
        report.lines()
    );
}

/// Two plays' reports compared: a verdict at a frame with equal player-side counts and equal
/// digests (other executed counts may differ); none where the skips differ, where a frame has no
/// digest in both, where the digests differ (the stale reads named as the first suspect on a
/// two-copy play) or where a play's player predates the served read classes.
#[test]
fn two_plays_compare_for_a_verdict_only_where_counts_and_digests_agree() {
    let report = |busy: u64, skipped: u64, sha: &str, extra: &str| {
        format!(
            "managed two copies (the stream's device identity): played as recorded\n\
             managed reads on two copies: made-with none, served; stale 2 objects, 3 uses, first \
             at frame 1, played as recorded\n\
             frame 0 stream calls=10 draws=1 dispatches=0 executed barriers={busy} skipped_draws=0\n\
             end 0 wall_nanos=100 times stream=1 gpu_busy={busy} live\n\
             check #7 frame 0 aa match\n\
             frame 1 stream calls=5 draws=1 dispatches=0 executed skipped_draws={skipped}\n\
             end 1 wall_nanos=90 times gpu_busy=30 live\n\
             check #7 frame 1 {sha} match\n\
             frame 2 stream calls=5 draws=1 dispatches=0 executed\n\
             end 2 wall_nanos=90 times gpu_busy=30 live\n{extra}"
        )
    };
    let a = report(50, 0, "bb", "");
    let (lines, verdict) = compare_plays(&a, &report(40, 0, "bb", ""), Some(0));
    assert!(verdict, "{lines:?}");
    let line = "verdict at frame 0: b is 20.0% faster than a by GPU-busy time (a 50 ns, b 40 ns); \
                wall a 100 ns, b 100 ns";
    assert_eq!(lines.last().map(String::as_str), Some(line), "{lines:?}");
    let (lines, verdict) = compare_plays(&a, &report(50, 1, "bb", ""), Some(1));
    assert!(!verdict);
    let line = "no verdict at frame 1: counts differ (executed.skipped_draws a=0 b=1)";
    assert_eq!(lines.last().map(String::as_str), Some(line), "{lines:?}");
    let (lines, verdict) = compare_plays(&a, &a, Some(2));
    assert!(!verdict);
    let line = "no verdict at frame 2: no digest there in both plays";
    assert_eq!(lines.last().map(String::as_str), Some(line), "{lines:?}");
    let (lines, verdict) = compare_plays(&a, &report(50, 0, "cc", ""), Some(1));
    assert!(!verdict);
    let line = "no verdict at frame 1: the digests differ; stale reads from frame 1, the first \
                suspect";
    assert_eq!(lines.last().map(String::as_str), Some(line), "{lines:?}");
    let old: String = (a.lines())
        .filter(|l| !l.starts_with("managed reads"))
        .map(|l| format!("{l}\n"))
        .collect();
    let (lines, verdict) = compare_plays(&old, &a, Some(0));
    assert!(!verdict);
    assert!(lines.contains(
        &"served a: no: no managed reads line, so a player older than the \
                              made-with and first-use rules"
            .to_owned()
    ));
    let line = "no verdict at frame 0: a play's player does not serve every read class";
    assert_eq!(lines.last().map(String::as_str), Some(line), "{lines:?}");
}
