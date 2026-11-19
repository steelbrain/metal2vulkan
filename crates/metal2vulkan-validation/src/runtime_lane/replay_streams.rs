//! Device tests of the replay player on the old runtime (`replay::OldRuntime`), on the device the
//! cfg selects: streams the tests write (`docs/REPLAY-STREAM.md`), each played and its report
//! printed as `REPLAY <test> <line>`.
//!
//! - A kernel follows a device pointer that a write put in a block at a recorded address: the
//!   backend makes the block and every buffer of a recorded range at its recorded address, so
//!   the word plays as written, and the kernel's output is the case's Metal output. Two plays on
//!   two devices report the same.
//! - A data word whose value lies in a live recorded range plays as written, and the report says
//!   every object with a recorded range was made at it.
//! - A recorded address the runtime refuses (it meets another pin's span) stops the play, naming
//!   the object, with no other placement tried.
//! - A command buffer waiting on an event the CPU signals later does not stop another queue's.
//! - Textures made from descriptor records read back as written.
//! - A render pass its setter records describe clears a drawable the stream never saw made.
//! - The copies from the guest's RAM and a device copy's fill land where their pieces say.
//! - A pass setter's object 0 clears the attachment's texture it sets to nil.
//! - A buffer made as a copy of host memory starts with its contents, and the IDs the program saw
//!   are compared with the runtime's.
//! - A texture its descriptor marks `outside` takes none of the program's IDs.
//! - A Private buffer's bytes a stream holds around a dispatch go in before the dispatch, and the
//!   bytes it leaves are compared with the stream's after.
//! - A view made through a swizzle, by the public call and by Apple's private one at a resource
//!   index, is sampled through it.
//! - A blit's generated mipmaps fill a texture's levels below its first.
//! - A store action an encoder sets over its pass's unknown one stores the pass's clear, with
//!   fragment samplers bound with LOD clamps; the depth and stencil store actions an encoder sets
//!   store a depth-only and a stencil-only pass's clears.
//!
//! Each device takes its queue plan ([`at_plan`]): the player's readbacks take a command queue of
//! their own beside the stream's.

use super::host_memory::on_device_with;
use super::{at_plan, public_case};
use crate::replay::{
    OldRuntime, PlayError, Player, Report, StreamReader, StreamWriter, Value, WAIT_COMPLETED,
};
use crate::ScratchDir;
use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};
use metal2vulkan_runtime::Device;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::{Path, PathBuf};

/// The address the recording put the block at, in the guest's range of Metal addresses.
const BLOCK: u64 = 1 << 46;
/// Where the recording put the output buffer.
const OUTPUT: u64 = (1 << 46) + (1 << 32);

fn addressed(address: u64, length: u64) -> [(&'static str, Value); 2] {
    [
        ("gpuAddress", Value::Address(address)),
        ("gpuLength", Value::Uint(length)),
    ]
}

fn sha(bytes: &[u8]) -> Value {
    Value::List(vec![Value::Bytes(Sha256::digest(bytes).to_vec())])
}

/// A recorder struct's map of unsigned fields.
fn uints(fields: &[(&str, u64)]) -> Value {
    let fields = fields
        .iter()
        .map(|&(key, value)| (key.into(), Value::Uint(value)));
    Value::Map(fields.collect())
}

/// A stream `build` writes, in `dir`.
fn written(
    dir: &ScratchDir,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> PathBuf {
    let path = dir.path().join("stream.m2vr");
    let mut writer = StreamWriter::new(File::create(&path).unwrap()).unwrap();
    build(&mut writer).unwrap();
    writer.finish().unwrap();
    path
}

/// `path` played on `device`, its report printed; the play's end unwrapped.
fn play(test: &str, device: &Device, path: &Path) -> Report {
    let mut backend = OldRuntime::new(device.clone(), crate::gpu::TEST_DEVICE);
    let mut reader = StreamReader::open(&[path.to_path_buf()], None).unwrap();
    let (report, ended) = Player::new(&mut backend).play(&mut reader);
    for line in report.lines() {
        eprintln!("REPLAY {test} {} {line}", device.name());
    }
    ended.unwrap();
    report
}

/// What of `report` the stream fixes: its executed counts and frame ends left out. Those differ
/// between plays (the runtime's pipeline and shader-cache nanoseconds among them), and only the
/// stream's counts compare between plays (`docs/REPLAY-STREAM.md`, section 10).
fn fixed_by_stream(report: &Report) -> Report {
    let mut fixed = report.clone();
    for (_, counts) in &mut fixed.completed {
        counts.executed.clear();
    }
    for counts in &mut fixed.frames {
        counts.executed.clear();
    }
    fixed.total.executed.clear();
    fixed.split_executed.clear();
    fixed.ends.clear();
    fixed
}

/// The pointer chain: `root` (a buffer at the block's start) holds the recorded address of
/// `node` (a buffer 0x100 into the block) and 7, `node` holds null and 11; two steps from `root`
/// sum 7 + 11 + 11 = 29 in each of four lanes. The write comes before the commit, as a recorder
/// captures it.
fn pointer_chain(w: &mut StreamWriter<File>) -> std::io::Result<()> {
    let case = public_case("walk-a-device-pointer-chain-through-a-loop-phi").unwrap();
    let expected: Vec<u8> = [29u32; 4].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(case.metal_output, expected, "the case's Metal output");
    let library = metallib(&[MtlbFunction {
        name: &case.case.entry,
        type_code: 2,
        bitcode: wrap(&case.bitcode),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    }]);
    let library = w.blob(&library)?;
    w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
    w.call("newCommandQueue", 1, 2, &[], &[])?;
    w.call(
        "newLibraryWithData:error:",
        1,
        3,
        &[Value::Blob(library)],
        &[],
    )?;
    let entry = [Value::Str(case.case.entry.clone())];
    w.call("newFunctionWithName:", 3, 4, &entry, &[])?;
    w.call(
        "newComputePipelineStateWithFunction:error:",
        1,
        5,
        &[Value::Object(4)],
        &[],
    )?;
    let block = [Value::Uint(0x1_0000), Value::Uint(0)];
    let at = addressed(BLOCK, 0x1_0000);
    w.call("m2v.newBlockWithLength:storageMode:", 1, 6, &block, &at)?;
    for (id, offset) in [(7, 0), (8, 0x100)] {
        let args = [
            Value::Object(6),
            Value::Uint(offset),
            Value::Uint(16),
            Value::Uint(0),
        ];
        let at = addressed(BLOCK + offset, 16);
        w.call(
            "m2v.newBufferInBlock:offset:length:options:",
            6,
            id,
            &args,
            &at,
        )?;
    }
    let output = [Value::Uint(16), Value::Uint(0)];
    w.call(
        "newBufferWithLength:options:",
        1,
        9,
        &output,
        &addressed(OUTPUT, 16),
    )?;
    w.call("commandBuffer", 2, 10, &[], &[])?;
    w.call("computeCommandEncoder", 10, 11, &[], &[])?;
    w.call("setComputePipelineState:", 11, 0, &[Value::Object(5)], &[])?;
    for (buffer, index) in [(7, 0), (9, 1)] {
        let args = [Value::Object(buffer), Value::Uint(0), Value::Uint(index)];
        w.call("setBuffer:offset:atIndex:", 11, 0, &args, &[])?;
    }
    let steps = [
        Value::Bytes(2u32.to_le_bytes().to_vec()),
        Value::Uint(4),
        Value::Uint(2),
    ];
    w.call("setBytes:length:atIndex:", 11, 0, &steps, &[])?;
    let node = Value::Map(vec![("Buffer".into(), Value::Object(8))]);
    w.call("useResource:usage:", 11, 0, &[node, Value::Uint(1)], &[])?;
    let grid = uints(&[("width", 4), ("height", 1), ("depth", 1)]);
    let args = [grid.clone(), grid];
    w.call("dispatchThreads:threadsPerThreadgroup:", 11, 0, &args, &[])?;
    w.call("endEncoding", 11, 0, &[], &[])?;
    let node = |next: u64, value: u32| {
        let mut bytes = next.to_le_bytes().to_vec();
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        Value::Bytes(bytes)
    };
    let pairs = [
        Value::Uint(0),
        node(BLOCK + 0x100, 7),
        Value::Uint(0x100),
        node(0, 11),
    ];
    w.call("m2v.write", 6, 0, &pairs, &[])?;
    w.call("commit", 10, 0, &[], &[])?;
    w.call(
        "m2v.check",
        0,
        0,
        &[Value::Object(9)],
        &[("sha256", sha(&expected))],
    )?;
    for id in (2..=10).rev() {
        w.call("release", id, 0, &[], &[])?;
    }
    Ok(())
}

/// The kernel reads the node through the pointer as written, the block and its buffers and the
/// output at their recorded addresses: its output is the case's Metal output, the address map
/// found the one word in a live range and moved it nowhere, the write's 32 bytes counted; the
/// same stream on a second device reports the same counts of the stream's and the same checks.
#[test]
fn a_pointer_written_at_a_recorded_address_reaches_its_node() {
    let dir = ScratchDir::new("replay-pointer-chain").unwrap();
    let path = written(&dir, pointer_chain);
    let mut reports = Vec::new();
    for _ in 0..2 {
        on_device_with(at_plan, |device| {
            reports.push(play("pointer-chain", device, &path))
        });
    }
    let report = &reports[0];
    assert_eq!(report.checks.len(), 1);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    assert_eq!(report.total.stream["relocated_words"], 1);
    assert_eq!(report.total.stream["misaligned_address_words"], 0);
    assert_eq!(report.total.stream["written_bytes"], 32);
    assert_eq!(
        report.pinned,
        Some(4),
        "the block, its two buffers and the output"
    );
    assert_eq!(report.completed.len(), 1);
    assert!(
        report.total.executed["dispatches"] >= 1,
        "{:?}",
        report.total
    );
    let fixed = reports.iter().map(fixed_by_stream).collect::<Vec<_>>();
    assert_eq!(fixed[0], fixed[1], "two plays on two devices");
}

/// A descriptor buffer pinned at `address`, `length` bytes, made as the stream records it: the
/// pin in its descriptor and the range it answered.
fn pinned_buffer(
    w: &mut StreamWriter<File>,
    id: u64,
    address: u64,
    length: u64,
) -> std::io::Result<()> {
    let descriptor = Value::Map(vec![
        ("length".into(), Value::Uint(length)),
        ("options".into(), Value::List(Vec::new())),
        ("pinnedGpuAddress".into(), Value::Address(address)),
        ("noCopy".into(), Value::Bool(false)),
    ]);
    let made = addressed(address, length);
    w.call("newBufferWithDescriptor:", 1, id, &[descriptor], &made)
}

/// Two pinned buffers: a write into the first holds two words, one an address 0x20 into the
/// second and one two 32-bit indices whose pair lies in the first's own range (as pixel data
/// can). Both play as written, and the report counts both objects pinned.
#[test]
fn a_data_word_in_a_live_range_plays_as_written() {
    let dir = ScratchDir::new("replay-pinned-words").unwrap();
    let (first, second) = (BLOCK, BLOCK + (1 << 20));
    let mut bytes = (second + 0x20).to_le_bytes().to_vec();
    bytes.extend_from_slice(&(first + 0x10).to_le_bytes());
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        pinned_buffer(w, 2, first, 0x100)?;
        pinned_buffer(w, 3, second, 0x100)?;
        let pairs = [Value::Uint(0), Value::Bytes(bytes.clone())];
        w.call("m2v.write", 2, 0, &pairs, &[])?;
        let mut whole = bytes.clone();
        whole.resize(0x100, 0);
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(2)],
            &[("sha256", sha(&whole))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("pinned-words", device, &path);
        assert_eq!(report.checks.len(), 1);
        assert_eq!(
            Some(report.checks[0].sha256),
            report.checks[0].recorded,
            "the words as written"
        );
        assert_eq!(report.total.stream["relocated_words"], 2);
        assert_eq!(report.pinned, Some(2));
        let line = "pinned 2 of 2: every object with a recorded range at its recorded address; \
                    the address map moves no word";
        assert!(report.lines().contains(&line.to_string()), "{report:?}");
    });
}

/// A buffer pinned 32 KiB into another's span (a 32-KiB buffer's span holds its two pages and a
/// guard page, rounded up to 64 KiB) is refused by the runtime: the play stops at it, naming it
/// and its recorded address, and makes it nowhere else.
#[test]
fn a_recorded_address_the_runtime_refuses_stops_the_play() {
    let dir = ScratchDir::new("replay-pinned-conflict").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        pinned_buffer(w, 2, BLOCK, 32 << 10)?;
        pinned_buffer(w, 3, BLOCK + (32 << 10), 0x100)
    });
    on_device_with(at_plan, |device| {
        let mut backend = OldRuntime::new(device.clone(), crate::gpu::TEST_DEVICE);
        let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
        let (report, ended) = Player::new(&mut backend).play(&mut reader);
        let stopped = matches!(
            ended,
            Err(PlayError::Unpinned { object: 3, recorded, made: None, .. })
                if recorded == BLOCK + (32 << 10)
        );
        assert!(stopped, "{ended:?}");
        assert_eq!(report.pinned, Some(1));
    });
}

/// Queue 2's command buffer waits for event 4 at 1, which the CPU sets only after queue 3's
/// command buffer is committed and its output checked: the player waits for queue 3's at its
/// commit, and for queue 2's once the value is set. Each fills its buffer.
#[test]
fn a_command_buffer_held_on_an_event_does_not_stop_another_queue() {
    let dir = ScratchDir::new("replay-held").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("newCommandQueue", 1, 3, &[], &[])?;
        w.call("newSharedEvent", 1, 4, &[], &[])?;
        for id in [5, 6] {
            w.call(
                "newBufferWithLength:options:",
                1,
                id,
                &[Value::Uint(16), Value::Uint(0)],
                &[],
            )?;
        }
        for (queue, command_buffer, buffer, fill) in [(2, 7, 5, 0x11), (3, 9, 6, 0x22)] {
            w.call("commandBuffer", queue, command_buffer, &[], &[])?;
            if queue == 2 {
                let wait = [Value::Object(4), Value::Uint(1)];
                w.call("encodeWaitForEvent:value:", command_buffer, 0, &wait, &[])?;
            }
            let blit = command_buffer + 1;
            w.call("blitCommandEncoder", command_buffer, blit, &[], &[])?;
            let range = uints(&[("location", 0), ("length", 16)]);
            let args = [Value::Object(buffer), range, Value::Uint(fill)];
            w.call("fillBuffer:range:value:", blit, 0, &args, &[])?;
            w.call("endEncoding", blit, 0, &[], &[])?;
            w.call("commit", command_buffer, 0, &[], &[])?;
        }
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(6)],
            &[("sha256", sha(&[0x22; 16]))],
        )?;
        w.call("setSignaledValue:", 4, 0, &[Value::Uint(1)], &[])?;
        w.call(WAIT_COMPLETED, 7, 0, &[], &[])?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(5)],
            &[("sha256", sha(&[0x11; 16]))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("held", device, &path);
        let order: Vec<&[u64]> = report.completed.iter().map(|(ids, _)| &ids[..]).collect();
        assert_eq!(order, [&[9][..], &[7][..]]);
        for check in &report.checks {
            assert_eq!(Some(check.sha256), check.recorded, "#{}", check.object);
        }
    });
}

/// A texture of a 2D descriptor whose setter records set its usage and options, and one of a
/// descriptor record placed at resource index 77, each read back as `replaceRegion` wrote it.
#[test]
fn textures_are_made_from_descriptor_records() {
    let dir = ScratchDir::new("replay-texture-descriptors").unwrap();
    let texels: Vec<u8> = (0..64).collect();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        // RGBA8Unorm, 4 by 4, not mipmapped.
        let two_d = [70, 4, 4].map(Value::Uint);
        let two_d = [&two_d[..], &[Value::Bool(false)]].concat();
        let selector = "MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:\
                        mipmapped:";
        w.call(selector, 0, 2, &two_d, &[])?;
        w.call("setUsage:", 2, 0, &[Value::Uint(1)], &[])?;
        w.call("setResourceOptions:", 2, 0, &[Value::Uint(0)], &[])?;
        w.call("newTextureWithDescriptor:", 1, 3, &[Value::Object(2)], &[])?;
        let Value::Map(mut fields) = uints(&[
            ("pixelFormat", 70),
            ("width", 4),
            ("height", 4),
            ("usage", 1),
            ("resourceOptions", 0),
        ]) else {
            unreachable!("a map");
        };
        let index = Value::List(vec![Value::Uint(77), Value::Bool(true)]);
        fields.push(("resourceIndex".into(), index));
        fields.push(("rotation".into(), Value::Nil));
        w.call("m2v.textureDescriptor", 0, 4, &[Value::Map(fields)], &[])?;
        w.call("newTextureWithDescriptor:", 1, 5, &[Value::Object(4)], &[])?;
        let region = Value::Map(vec![
            ("origin".into(), uints(&[("x", 0), ("y", 0), ("z", 0)])),
            (
                "size".into(),
                uints(&[("width", 4), ("height", 4), ("depth", 1)]),
            ),
        ]);
        for texture in [3, 5] {
            let args = [
                region.clone(),
                Value::Uint(0),
                Value::Bytes(texels.clone()),
                Value::Uint(16),
            ];
            let selector = "replaceRegion:mipmapLevel:withBytes:bytesPerRow:";
            w.call(selector, texture, 0, &args, &[])?;
        }
        let textures = [Value::Object(3), Value::Object(5)];
        let sums = [sha(&texels), sha(&texels)].map(|sum| match sum {
            Value::List(mut one) => one.remove(0),
            _ => unreachable!("a list"),
        });
        w.call(
            "m2v.check",
            0,
            0,
            &textures,
            &[("sha256", Value::List(sums.to_vec()))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("texture-descriptors", device, &path);
        assert_eq!(report.checks.len(), 2);
        for check in &report.checks {
            assert_eq!(Some(check.sha256), check.recorded, "#{}", check.object);
        }
    });
}

/// A drawable the stream never saw made (a `newTextureWithDescriptor:` of an inline descriptor,
/// answered `m2v.drawable`) is a texture of its descriptor, and a render pass whose setter
/// records clear it to green and store it leaves it green.
#[test]
fn a_pass_from_setter_records_clears_a_drawable() {
    let dir = ScratchDir::new("replay-pass").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        // RGBA8Unorm, 4 by 4, shader-read and render-target, Private.
        let drawable = uints(&[
            ("textureType", 2),
            ("pixelFormat", 70),
            ("width", 4),
            ("height", 4),
            ("usage", 5),
            ("storageMode", 2),
        ]);
        let answer = [("m2v.drawable", Value::Bool(true))];
        w.call("newTextureWithDescriptor:", 0, 3, &[drawable], &answer)?;
        w.call("MTLRenderPassDescriptor.new", 0, 4, &[], &[])?;
        let pass = Value::Map(vec![
            ("visibilityResultBuffer".into(), Value::Nil),
            ("renderTargetWidth".into(), Value::Uint(4)),
            ("renderTargetHeight".into(), Value::Uint(4)),
        ]);
        w.call("m2v.setPass", 4, 0, &[pass], &[])?;
        let clear = Value::Map(
            [("red", 0.0), ("green", 1.0), ("blue", 0.0), ("alpha", 1.0)]
                .map(|(key, value)| (key.into(), Value::F64(value)))
                .to_vec(),
        );
        let Value::Map(mut color) = uints(&[("level", 0), ("loadAction", 2), ("storeAction", 1)])
        else {
            unreachable!("a map");
        };
        color.push(("texture".into(), Value::Object(3)));
        color.push(("resolveTexture".into(), Value::Nil));
        color.push(("clearColor".into(), clear));
        let args = [Value::Uint(0), Value::Map(color)];
        w.call("m2v.setColorAttachment", 4, 0, &args, &[])?;
        w.call("commandBuffer", 2, 5, &[], &[])?;
        let pass = [Value::Object(4)];
        w.call("renderCommandEncoderWithDescriptor:", 5, 6, &pass, &[])?;
        w.call("endEncoding", 6, 0, &[], &[])?;
        w.call("commit", 5, 0, &[], &[])?;
        w.call("m2v.present", 0, 0, &[Value::Object(3)], &[])?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(3)],
            &[("sha256", sha(&green))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("pass", device, &path);
        assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
        assert_eq!(report.frames.len(), 1);
        assert!(
            report.total.executed["render_passes"] >= 1,
            "{:?}",
            report.total
        );
    });
}

/// The copies from the guest's RAM and a device copy's fill land their bytes where their pieces
/// say, in order with a fill before them, and a copy back to guest RAM runs: the buffer reads back
/// as laid out.
#[test]
fn guest_ram_copies_land_where_their_pieces_say() {
    let dir = ScratchDir::new("replay-guest-ram").unwrap();
    let mut expected = vec![0u8; 64];
    expected[8..24].copy_from_slice(&(1..=16).collect::<Vec<u8>>());
    expected[32..40].copy_from_slice(&(105..=112).collect::<Vec<u8>>());
    expected[40..44].copy_from_slice(&(101..=104).collect::<Vec<u8>>());
    let piece = |at: u64, data: Vec<u8>| Value::List(vec![Value::Uint(at), Value::Bytes(data)]);
    let placed = |at: u64, start: u64, length: u64| {
        let run = Value::List(vec![Value::Uint(start), Value::Uint(start + length)]);
        Value::List(vec![Value::Uint(at), run])
    };
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let shared = [Value::Uint(64), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 3, &shared, &[])?;
        w.call("commandBuffer", 2, 4, &[], &[])?;
        w.call("blitCommandEncoder", 4, 5, &[], &[])?;
        let range = uints(&[("location", 0), ("length", 64)]);
        let fill = [Value::Object(3), range, Value::Uint(0)];
        w.call("fillBuffer:range:value:", 5, 0, &fill, &[])?;
        let pieces = Value::List(vec![piece(8, (1..=16).collect())]);
        let args = [Value::Object(3), pieces];
        w.call("m2v.copyFromGuestRam", 5, 0, &args, &[])?;
        let pieces = Value::List(vec![
            piece(40, (101..=104).collect()),
            piece(32, (105..=112).collect()),
        ]);
        let args = [Value::Object(3), pieces];
        w.call("m2v.copyFromGuestRamAt", 5, 0, &args, &[])?;
        let runs = Value::List(vec![placed(0, 0x5000, 16), placed(48, 0x6000, 16)]);
        w.call("m2v.copyToGuestRam", 5, 0, &[Value::Object(3), runs], &[])?;
        w.call("endEncoding", 5, 0, &[], &[])?;
        w.call("commit", 4, 0, &[], &[])?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(3)],
            &[("sha256", sha(&expected))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("guest-ram", device, &path);
        assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
        assert_eq!(report.total.stream["written_bytes"], 16 + 12);
        assert_eq!(report.total.stream["relocated_words"], 0);
    });
}

/// A 4 by 4 RGBA8Unorm texture descriptor record's map, shader-read and render-target, Shared,
/// with `extra` fields.
fn render_target(extra: Vec<(&str, Value)>) -> Value {
    let Value::Map(mut fields) = uints(&[
        ("pixelFormat", 70),
        ("width", 4),
        ("height", 4),
        ("usage", 5),
        ("resourceOptions", 0),
    ]) else {
        unreachable!("a map");
    };
    fields.extend(extra.into_iter().map(|(key, value)| (key.into(), value)));
    Value::Map(fields)
}

/// The whole of a 4 by 4 texture.
fn whole() -> Value {
    Value::Map(vec![
        ("origin".into(), uints(&[("x", 0), ("y", 0), ("z", 0)])),
        (
            "size".into(),
            uints(&[("width", 4), ("height", 4), ("depth", 1)]),
        ),
    ])
}

/// A pass whose setter records set attachment 1's texture and then set it to nil (object 0)
/// clears attachment 0's texture to green and leaves the other's blue, as Metal leaves a texture
/// no attachment holds; object 0 for the visibility result buffer, the rate map and a resolve
/// texture plays too.
#[test]
fn a_pass_setter_clears_an_attachment_set_to_nil() {
    let dir = ScratchDir::new("replay-pass-nil").unwrap();
    let [red, blue] = [[255u8, 0, 0, 255], [0, 0, 255, 255]].map(|texel| texel.repeat(16));
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("m2v.textureDescriptor", 0, 3, &[render_target(vec![])], &[])?;
        for (texture, texels) in [(4, &red), (5, &blue)] {
            w.call(
                "newTextureWithDescriptor:",
                1,
                texture,
                &[Value::Object(3)],
                &[],
            )?;
            let args = [
                whole(),
                Value::Uint(0),
                Value::Bytes(texels.clone()),
                Value::Uint(16),
            ];
            let selector = "replaceRegion:mipmapLevel:withBytes:bytesPerRow:";
            w.call(selector, texture, 0, &args, &[])?;
        }
        let shared = [Value::Uint(64), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 6, &shared, &[])?;
        w.call("MTLRenderPassDescriptor.new", 0, 7, &[], &[])?;
        let pass = |visibility: u64, rate_map: Value| {
            Value::Map(vec![
                ("visibilityResultBuffer".into(), Value::Object(visibility)),
                ("rasterizationRateMap".into(), rate_map),
                ("renderTargetWidth".into(), Value::Uint(4)),
                ("renderTargetHeight".into(), Value::Uint(4)),
            ])
        };
        w.call("m2v.setPass", 7, 0, &[pass(6, Value::Nil)], &[])?;
        w.call("m2v.setPass", 7, 0, &[pass(0, Value::Object(0))], &[])?;
        let clear = |color: [f64; 4]| {
            let channels = ["red", "green", "blue", "alpha"].into_iter().zip(color);
            Value::Map(
                channels
                    .map(|(key, value)| (key.into(), Value::F64(value)))
                    .collect(),
            )
        };
        let attachment = |texture: u64, color: [f64; 4]| {
            let Value::Map(mut fields) =
                uints(&[("level", 0), ("loadAction", 2), ("storeAction", 1)])
            else {
                unreachable!("a map");
            };
            fields.push(("texture".into(), Value::Object(texture)));
            fields.push(("resolveTexture".into(), Value::Object(0)));
            fields.push(("clearColor".into(), clear(color)));
            Value::Map(fields)
        };
        let set = [Value::Uint(0), attachment(4, [0.0, 1.0, 0.0, 1.0])];
        w.call("m2v.setColorAttachment", 7, 0, &set, &[])?;
        let set = [Value::Uint(1), attachment(5, [1.0, 0.0, 0.0, 1.0])];
        w.call("m2v.setColorAttachment", 7, 0, &set, &[])?;
        let nil = Value::Map(vec![("texture".into(), Value::Object(0))]);
        w.call("m2v.setColorAttachment", 7, 0, &[Value::Uint(1), nil], &[])?;
        w.call("commandBuffer", 2, 8, &[], &[])?;
        let pass = [Value::Object(7)];
        w.call("renderCommandEncoderWithDescriptor:", 8, 9, &pass, &[])?;
        w.call("endEncoding", 9, 0, &[], &[])?;
        w.call("commit", 8, 0, &[], &[])?;
        let sums = [&green, &blue].map(|texels| Value::Bytes(Sha256::digest(texels).to_vec()));
        let answer = [("sha256", Value::List(sums.to_vec()))];
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(4), Value::Object(5)],
            &answer,
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("pass-nil", device, &path);
        assert_eq!(report.checks.len(), 2);
        for check in &report.checks {
            assert_eq!(Some(check.sha256), check.recorded, "#{}", check.object);
        }
    });
}

/// A buffer `newBufferWithDescriptor:` made as a copy of host memory reads back as its
/// `contents`, a blob and inline bytes; a texture placed at its resource index gives the ID the
/// program saw, and one the runtime placed where it had room another, which the report names.
#[test]
fn a_copied_buffer_starts_with_its_contents_and_ids_are_compared() {
    let dir = ScratchDir::new("replay-contents-ids").unwrap();
    let long: Vec<u8> = (0..=255).collect();
    let short = [9u8; 8];
    let recorded = 0xdead_0000_0001;
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let blob = Value::Blob(w.blob(&long)?);
        for (buffer, length, contents) in [(2, 256, blob), (3, 8, Value::Bytes(short.to_vec()))] {
            let descriptor = Value::Map(vec![
                ("length".into(), Value::Uint(length)),
                ("options".into(), Value::List(Vec::new())),
                ("contents".into(), contents),
                ("noCopy".into(), Value::Bool(false)),
            ]);
            w.call("newBufferWithDescriptor:", 1, buffer, &[descriptor], &[])?;
        }
        let placed = Value::List(vec![Value::Uint(77), Value::Bool(true)]);
        let at_77 = render_target(vec![("resourceIndex", placed)]);
        w.call("m2v.textureDescriptor", 0, 4, &[at_77], &[])?;
        w.call("m2v.textureDescriptor", 0, 5, &[render_target(vec![])], &[])?;
        for (texture, descriptor, id) in [(6, 4, 77), (7, 5, recorded)] {
            let args = [Value::Object(descriptor)];
            w.call("newTextureWithDescriptor:", 1, texture, &args, &[])?;
            let answer = [("gpuResourceID", Value::Uint(id))];
            w.call("gpuResourceID", texture, 0, &[], &answer)?;
        }
        let sums =
            [&long[..], &short[..]].map(|bytes| Value::Bytes(Sha256::digest(bytes).to_vec()));
        let answer = [("sha256", Value::List(sums.to_vec()))];
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(2), Value::Object(3)],
            &answer,
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("contents-ids", device, &path);
        for check in &report.checks {
            assert_eq!(Some(check.sha256), check.recorded, "#{}", check.object);
        }
        assert_eq!(report.total.stream["written_bytes"], 256 + 8);
        assert_eq!(report.ids.len(), 1, "{:?}", report.ids);
        assert_eq!(
            (report.ids[0].object, report.ids[0].recorded),
            (7, recorded)
        );
        assert!(report.ids[0].played.is_some_and(|id| id != recorded));
    });
}

/// Two plain textures and, where `marked`, a texture its descriptor marks `outside` made between
/// them, the IDs the program saw of the plain ones `recorded`.
fn outside_stream(dir: &ScratchDir, marked: bool, recorded: [u64; 2]) -> PathBuf {
    written(dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("m2v.textureDescriptor", 0, 2, &[render_target(vec![])], &[])?;
        let outside = render_target(vec![("outside", Value::Bool(true))]);
        w.call("m2v.textureDescriptor", 0, 3, &[outside], &[])?;
        w.call("newTextureWithDescriptor:", 1, 4, &[Value::Object(2)], &[])?;
        if marked {
            w.call("newTextureWithDescriptor:", 1, 5, &[Value::Object(3)], &[])?;
        }
        w.call("newTextureWithDescriptor:", 1, 6, &[Value::Object(2)], &[])?;
        for (texture, id) in [(4, recorded[0]), (6, recorded[1])] {
            let answer = [("gpuResourceID", Value::Uint(id))];
            w.call("gpuResourceID", texture, 0, &[], &answer)?;
        }
        Ok(())
    })
}

/// A texture its descriptor marks `outside` takes none of the program's IDs: the plain textures
/// made around it give the IDs they give where the stream makes no marked one, which a play of
/// that stream names (its records' IDs, 0, differ from them).
#[test]
fn a_texture_marked_outside_leaves_the_program_its_ids() {
    let (plain, marked) = (
        ScratchDir::new("replay-plain").unwrap(),
        ScratchDir::new("replay-outside").unwrap(),
    );
    let reference = outside_stream(&plain, false, [0, 0]);
    on_device_with(at_plan, |device| {
        let report = play("outside-reference", device, &reference);
        let given: Vec<Option<u64>> = report.ids.iter().map(|ids| ids.played).collect();
        let [Some(first), Some(second)] = given[..] else {
            panic!("the plain textures gave no IDs: {:?}", report.ids);
        };
        let path = outside_stream(&marked, true, [first, second]);
        let report = play("outside", device, &path);
        assert_eq!(
            report.ids,
            [],
            "the marked texture moved the plain ones' IDs"
        );
    });
}

/// Two dispatches of a kernel that copies part of a Private buffer over another part, in one
/// encoder, each with `m2v.privateBytes` records of the whole buffer: before each, the player puts
/// the case's input in, and after each it compares what the kernel left with the bytes after, the
/// case's Metal output for the first and that output with one word changed for the second, which
/// is the one word the report counts. The group that held the splits reports its executed counts
/// apart, in no group, frame or total: they hold the program's two dispatches and one
/// submission, as the same stream without the records executes them.
#[test]
fn a_private_buffer_takes_its_bytes_before_a_dispatch_and_is_compared_after() {
    let case = public_case("copy-twelve-bytes-through-a-four-byte-element-pointer").unwrap();
    let input = case.compute.buffers[&0].clone();
    let output = case.metal_output.clone();
    assert_eq!((input.len(), output.len()), (32, 32));
    assert_ne!(input, output, "the kernel changes the buffer");
    let mut changed = output.clone();
    changed[20] ^= 1;
    let library = metallib(&[MtlbFunction {
        name: &case.case.entry,
        type_code: 2,
        bitcode: wrap(&case.bitcode),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    }]);
    let (library, input, output, changed) = (&library, &input, &output, &changed);
    let entry = &case.case.entry;
    let stream = |records: bool| {
        move |w: &mut StreamWriter<File>| {
            let library = w.blob(library)?;
            let before = w.blob(input)?;
            let afters = [w.blob(output)?, w.blob(changed)?];
            w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
            w.call("newCommandQueue", 1, 2, &[], &[])?;
            let library = [Value::Blob(library)];
            w.call("newLibraryWithData:error:", 1, 3, &library, &[])?;
            let entry = [Value::Str(entry.clone())];
            w.call("newFunctionWithName:", 3, 4, &entry, &[])?;
            let function = [Value::Object(4)];
            w.call(
                "newComputePipelineStateWithFunction:error:",
                1,
                5,
                &function,
                &[],
            )?;
            let private = [Value::Uint(32), Value::Uint(0x20)];
            w.call("newBufferWithLength:options:", 1, 6, &private, &[])?;
            w.call("commandBuffer", 2, 7, &[], &[])?;
            w.call("computeCommandEncoder", 7, 8, &[], &[])?;
            w.call("setComputePipelineState:", 8, 0, &[Value::Object(5)], &[])?;
            let bound = [Value::Object(6), Value::Uint(0), Value::Uint(0)];
            w.call("setBuffer:offset:atIndex:", 8, 0, &bound, &[])?;
            let one = uints(&[("width", 1), ("height", 1), ("depth", 1)]);
            for _ in 0..2 {
                let args = [one.clone(), one.clone()];
                w.call("dispatchThreads:threadsPerThreadgroup:", 8, 0, &args, &[])?;
            }
            w.call("endEncoding", 8, 0, &[], &[])?;
            w.call("commit", 7, 0, &[], &[])?;
            w.call("release", 6, 0, &[], &[])?;
            for (dispatch, after) in [(10, afters[0]), (11, afters[1])]
                .into_iter()
                .filter(|_| records)
            {
                let range = uints(&[("location", 0), ("length", 32)]);
                let args = [
                    Value::Uint(dispatch),
                    range,
                    Value::Blob(before),
                    Value::Blob(after),
                ];
                w.call("m2v.privateBytes", 6, 0, &args, &[])?;
            }
            Ok(())
        }
    };
    let mut reports = Vec::new();
    for (name, records) in [("private-bytes", true), ("no-private-bytes", false)] {
        let dir = ScratchDir::new("replay-private-bytes").unwrap();
        let path = written(&dir, stream(records));
        on_device_with(at_plan, |device| reports.push(play(name, device, &path)));
    }
    let (report, plain) = (&reports[0], &reports[1]);
    let differing = report.private.iter().map(|p| (p.dispatch, p.differing));
    let differing: Vec<_> = differing.collect();
    assert_eq!(differing, [(10, 0), (11, 1)], "{:?}", report.private);
    assert_eq!(report.total.stream["written_bytes"], 64);
    assert_eq!((report.splits, plain.splits), (4, 0));
    assert!(plain.split_executed.is_empty());
    assert_eq!(report.completed.len(), 1);
    assert!(report.completed[0].1.executed.is_empty());
    // The pipeline the runtime makes in the background may land in any span.
    for key in ["barriers", "copies", "dispatches", "submissions"] {
        assert_eq!(report.total.executed[key], 0, "{key}");
    }
    for key in ["dispatches", "submissions"] {
        let (split, alone) = (report.split_executed[key], plain.total.executed[key]);
        assert_eq!(split, alone, "{key}");
    }
}

/// Each view through a swizzle of a 4 by 4 red `RGBA8Unorm` texture, the public call's
/// (blue, red, one, zero) and Apple's private call's at resource index 50 (alpha, green, red,
/// red), sampled at every pixel by `a4_pixel_fetch` (the owned fixture
/// `engine/a4_pixel_fetch.ll`): (0, 1, 1, 0) and (1, 0, 1, 1), where the texture itself samples
/// as (1, 0, 0, 1).
#[test]
fn a_view_s_swizzle_is_what_a_kernel_samples() {
    let ll = metal2vulkan_devtools::engine_dir().join("a4_pixel_fetch.ll");
    let library = metallib(&[MtlbFunction {
        name: "a4_pixel_fetch",
        type_code: 2,
        bitcode: wrap(&metal2vulkan_devtools::llvm_assemble(&ll).unwrap()),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    }]);
    // MTLTextureSwizzle: zero 0, one 1, red 2, green 3, blue 4, alpha 5.
    let views = [
        (
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:",
            [4, 2, 1, 0],
            None,
            [0.0f32, 1.0, 1.0, 0.0],
        ),
        (
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:",
            [5, 3, 2, 2],
            Some(50),
            [1.0, 0.0, 1.0, 1.0],
        ),
    ];
    let sampled = |color: [f32; 4]| -> Vec<u8> {
        let texel: Vec<u8> = color.iter().flat_map(|c| c.to_le_bytes()).collect();
        texel.repeat(16)
    };
    let dir = ScratchDir::new("replay-swizzled-views").unwrap();
    let path = written(&dir, |w| {
        let library = w.blob(&library)?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let data = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &data, &[])?;
        let entry = [Value::Str("a4_pixel_fetch".into())];
        w.call("newFunctionWithName:", 3, 4, &entry, &[])?;
        let function = [Value::Object(4)];
        let selector = "newComputePipelineStateWithFunction:error:";
        w.call(selector, 1, 5, &function, &[])?;
        w.call("m2v.textureDescriptor", 0, 6, &[render_target(vec![])], &[])?;
        w.call("newTextureWithDescriptor:", 1, 7, &[Value::Object(6)], &[])?;
        let args = [
            whole(),
            Value::Uint(0),
            Value::Bytes([255, 0, 0, 255].repeat(16)),
            Value::Uint(16),
        ];
        w.call(
            "replaceRegion:mipmapLevel:withBytes:bytesPerRow:",
            7,
            0,
            &args,
            &[],
        )?;
        let one = uints(&[("location", 0), ("length", 1)]);
        for (view, &(selector, [red, green, blue, alpha], index, _)) in (8..).zip(&views) {
            let swizzle = uints(&[
                ("red", red),
                ("green", green),
                ("blue", blue),
                ("alpha", alpha),
            ]);
            let mut args = vec![
                Value::Uint(70),
                Value::Uint(2),
                one.clone(),
                one.clone(),
                swizzle,
            ];
            args.extend(index.map(Value::Uint));
            w.call(selector, 7, view, &args, &[])?;
        }
        // A granule apart: each buffer is made at its recorded address, which a pin's span holds.
        for (buffer, at) in [(10, OUTPUT), (11, OUTPUT + (32 << 10))] {
            let length = [Value::Uint(256), Value::Uint(0)];
            let at = addressed(at, 256);
            w.call("newBufferWithLength:options:", 1, buffer, &length, &at)?;
        }
        w.call("commandBuffer", 2, 12, &[], &[])?;
        w.call("computeCommandEncoder", 12, 13, &[], &[])?;
        w.call("setComputePipelineState:", 13, 0, &[Value::Object(5)], &[])?;
        for (view, buffer) in [(8, 10), (9, 11)] {
            let texture = [Value::Object(view), Value::Uint(0)];
            w.call("setTexture:atIndex:", 13, 0, &texture, &[])?;
            let output = [Value::Object(buffer), Value::Uint(0), Value::Uint(0)];
            w.call("setBuffer:offset:atIndex:", 13, 0, &output, &[])?;
            let grid = uints(&[("width", 4), ("height", 4), ("depth", 1)]);
            let args = [grid.clone(), grid];
            w.call("dispatchThreads:threadsPerThreadgroup:", 13, 0, &args, &[])?;
        }
        w.call("endEncoding", 13, 0, &[], &[])?;
        w.call("commit", 12, 0, &[], &[])?;
        let sums = views.map(|(.., color)| Value::Bytes(Sha256::digest(sampled(color)).to_vec()));
        let buffers = [Value::Object(10), Value::Object(11)];
        w.call(
            "m2v.check",
            0,
            0,
            &buffers,
            &[("sha256", Value::List(sums.to_vec()))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("swizzled-views", device, &path);
        assert_eq!(report.checks.len(), 2);
        for check in &report.checks {
            assert_eq!(Some(check.sha256), check.recorded, "#{}", check.object);
        }
    });
}

/// A blit's `generateMipmapsForTexture:` of a 4 by 4 `RGBA8Unorm` texture of three levels whose
/// first holds one colour: a view of each level below reads back that colour in every pixel.
#[test]
fn a_blit_s_generated_mipmaps_fill_the_levels_below_the_first() {
    let texel = [0x40u8, 0x80, 0xc0, 0xff];
    let dir = ScratchDir::new("replay-mipmaps").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let levels = render_target(vec![("mipmapLevelCount", Value::Uint(3))]);
        w.call("m2v.textureDescriptor", 0, 3, &[levels], &[])?;
        w.call("newTextureWithDescriptor:", 1, 4, &[Value::Object(3)], &[])?;
        let args = [
            whole(),
            Value::Uint(0),
            Value::Bytes(texel.repeat(16)),
            Value::Uint(16),
        ];
        w.call(
            "replaceRegion:mipmapLevel:withBytes:bytesPerRow:",
            4,
            0,
            &args,
            &[],
        )?;
        w.call("commandBuffer", 2, 5, &[], &[])?;
        w.call("blitCommandEncoder", 5, 6, &[], &[])?;
        w.call("generateMipmapsForTexture:", 6, 0, &[Value::Object(4)], &[])?;
        w.call("endEncoding", 6, 0, &[], &[])?;
        w.call("commit", 5, 0, &[], &[])?;
        for (view, level) in [(7, 1), (8, 2)] {
            let args = [
                Value::Uint(70),
                Value::Uint(2),
                uints(&[("location", level), ("length", 1)]),
                uints(&[("location", 0), ("length", 1)]),
            ];
            let selector = "newTextureViewWithPixelFormat:textureType:levels:slices:";
            w.call(selector, 4, view, &args, &[])?;
        }
        let sums = [4, 1].map(|pixels| Value::Bytes(Sha256::digest(texel.repeat(pixels)).to_vec()));
        let views = [Value::Object(7), Value::Object(8)];
        w.call(
            "m2v.check",
            0,
            0,
            &views,
            &[("sha256", Value::List(sums.to_vec()))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("mipmaps", device, &path);
        assert_eq!(report.checks.len(), 2);
        for check in &report.checks {
            assert_eq!(Some(check.sha256), check.recorded, "#{}", check.object);
        }
    });
}

/// A pass whose colour attachment's store action is unknown until its encoder sets it: the
/// encoder binds two fragment samplers with LOD clamps, sets the store action to store, and the
/// cleared green is what the texture holds after.
#[test]
fn an_encoder_s_store_action_and_clamped_samplers_play() {
    let dir = ScratchDir::new("replay-store-action").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("m2v.textureDescriptor", 0, 3, &[render_target(vec![])], &[])?;
        w.call("newTextureWithDescriptor:", 1, 4, &[Value::Object(3)], &[])?;
        for sampler in [5, 6] {
            let plain = [Value::Map(Vec::new())];
            w.call("newSamplerStateWithDescriptor:", 1, sampler, &plain, &[])?;
        }
        w.call("MTLRenderPassDescriptor.new", 0, 7, &[], &[])?;
        let clear = Value::Map(
            [("red", 0.0), ("green", 1.0), ("blue", 0.0), ("alpha", 1.0)]
                .map(|(key, value)| (key.into(), Value::F64(value)))
                .to_vec(),
        );
        // Load action clear (2), store action unknown (4).
        let Value::Map(mut color) = uints(&[("level", 0), ("loadAction", 2), ("storeAction", 4)])
        else {
            unreachable!("a map");
        };
        color.push(("texture".into(), Value::Object(4)));
        color.push(("resolveTexture".into(), Value::Nil));
        color.push(("clearColor".into(), clear));
        let args = [Value::Uint(0), Value::Map(color)];
        w.call("m2v.setColorAttachment", 7, 0, &args, &[])?;
        w.call("commandBuffer", 2, 8, &[], &[])?;
        let pass = [Value::Object(7)];
        w.call("renderCommandEncoderWithDescriptor:", 8, 9, &pass, &[])?;
        let clamped = [
            Value::List(vec![Value::Object(5), Value::Object(6)]),
            Value::List(vec![Value::F64(1.0), Value::F64(0.0)]),
            Value::List(vec![Value::F64(2.0), Value::F64(0.5)]),
            uints(&[("location", 1), ("length", 2)]),
        ];
        let selector = "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:";
        w.call(selector, 9, 0, &clamped, &[])?;
        // Store (1) at colour attachment 0.
        let store = [Value::Uint(1), Value::Uint(0)];
        w.call("setColorStoreAction:atIndex:", 9, 0, &store, &[])?;
        w.call("endEncoding", 9, 0, &[], &[])?;
        w.call("commit", 8, 0, &[], &[])?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(4)],
            &[("sha256", sha(&green))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("store-action", device, &path);
        assert_eq!(report.checks.len(), 1);
        assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    });
}

/// A depth-only pass and a stencil-only pass whose attachment's store action is unknown until its
/// encoder sets it to store: the Private `Depth32Float` texture holds the pass's clear depth, 0.25,
/// and the Private `Stencil8` texture its clear stencil, 0x5a.
#[test]
fn an_encoder_s_depth_and_stencil_store_actions_play() {
    let dir = ScratchDir::new("replay-depth-stencil-store").unwrap();
    let depth: Vec<u8> = 0.25f32.to_le_bytes().repeat(16);
    let stencil = vec![0x5au8; 16];
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("commandBuffer", 2, 3, &[], &[])?;
        // Depth32Float (252) and Stencil8 (253): textures 5 and 9, passes 6 and 10, encoders 7 and
        // 11.
        let passes = [
            (
                252,
                "m2v.setDepthAttachment",
                "clearDepth",
                Value::F64(0.25),
            ),
            (
                253,
                "m2v.setStencilAttachment",
                "clearStencil",
                Value::Uint(0x5a),
            ),
        ];
        for (at, (format, setter, key, clear)) in [4u64, 8].into_iter().zip(passes) {
            let (texture, pass, encoder) = (at + 1, at + 2, at + 3);
            let d = uints(&[
                ("textureType", 2),
                ("pixelFormat", format),
                ("width", 4),
                ("height", 4),
                ("usage", 5),
                ("storageMode", 2),
            ]);
            w.call("m2v.textureDescriptor", 0, at, &[d], &[])?;
            w.call(
                "newTextureWithDescriptor:",
                1,
                texture,
                &[Value::Object(at)],
                &[],
            )?;
            w.call("MTLRenderPassDescriptor.new", 0, pass, &[], &[])?;
            // Load action clear (2), store action unknown (4).
            let Value::Map(mut a) = uints(&[("level", 0), ("loadAction", 2), ("storeAction", 4)])
            else {
                unreachable!("a map");
            };
            a.push(("texture".into(), Value::Object(texture)));
            a.push(("resolveTexture".into(), Value::Nil));
            a.push((key.into(), clear));
            w.call(setter, pass, 0, &[Value::Map(a)], &[])?;
            let pass = [Value::Object(pass)];
            w.call(
                "renderCommandEncoderWithDescriptor:",
                3,
                encoder,
                &pass,
                &[],
            )?;
            // Store (1).
            let selector = match format {
                252 => "setDepthStoreAction:",
                _ => "setStencilStoreAction:",
            };
            w.call(selector, encoder, 0, &[Value::Uint(1)], &[])?;
            w.call("endEncoding", encoder, 0, &[], &[])?;
        }
        w.call("commit", 3, 0, &[], &[])?;
        let sums = [&depth, &stencil].map(|bytes| Value::Bytes(Sha256::digest(bytes).to_vec()));
        let textures = [Value::Object(5), Value::Object(9)];
        w.call(
            "m2v.check",
            0,
            0,
            &textures,
            &[("sha256", Value::List(sums.to_vec()))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("depth-stencil-store", device, &path);
        assert_eq!(report.checks.len(), 2);
        for check in &report.checks {
            assert_eq!(Some(check.sha256), check.recorded, "#{}", check.object);
        }
    });
}
