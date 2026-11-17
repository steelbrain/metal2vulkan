//! Device tests of the replay player on the old runtime (`replay::OldRuntime`), on the device the
//! cfg selects: streams the tests write (`docs/REPLAY-STREAM.md`), each played and its report
//! printed as `REPLAY <test> <line>`.
//!
//! - A kernel follows a device pointer that a write put in a block at a recorded address: the
//!   player relocates it to the runtime's, and the kernel's output is the case's Metal output.
//!   Two plays on two devices report the same.
//! - A command buffer waiting on an event the CPU signals later does not stop another queue's.
//! - Textures made from descriptor records read back as written.
//! - A render pass its setter records describe clears a drawable the stream never saw made.
//! - The copies from the guest's RAM and a device copy's fill land where their pieces say.
//!
//! Each device takes its queue plan ([`at_plan`]): the player's readbacks take a command queue of
//! their own beside the stream's.

use super::host_memory::on_device_with;
use super::{at_plan, public_case};
use crate::replay::{
    OldRuntime, Player, Report, StreamReader, StreamWriter, Value, WAIT_COMPLETED,
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
            1,
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

/// The kernel reads the node through the relocated pointer: its output is the case's Metal
/// output, one word was relocated, the write's 32 bytes counted; the same stream on a second
/// device reports the same.
#[test]
fn a_pointer_written_at_a_recorded_address_is_relocated() {
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
    assert_eq!(report.total["relocated_words"], 1);
    assert_eq!(report.total["misaligned_address_words"], 0);
    assert_eq!(report.total["written_bytes"], 32);
    assert_eq!(report.completed.len(), 1);
    assert!(report.total["dispatches"] >= 1, "{:?}", report.total);
    assert_eq!(reports[0], reports[1], "two plays on two devices");
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
        assert!(report.total["render_passes"] >= 1, "{:?}", report.total);
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
        assert_eq!(report.total["written_bytes"], 16 + 12);
        assert_eq!(report.total["relocated_words"], 0);
    });
}
