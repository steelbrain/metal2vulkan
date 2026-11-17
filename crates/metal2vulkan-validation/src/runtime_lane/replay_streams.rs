//! Device tests of the replay player on the old runtime (`replay::OldRuntime`), on the device the
//! cfg selects: streams the tests write (`docs/REPLAY-STREAM.md`), each played and its report
//! printed as `REPLAY <test> <line>`.
//!
//! - A kernel follows a device pointer that a write put in a block at a recorded address: the
//!   player relocates it to the runtime's, and the kernel's output is the case's Metal output.
//!   Two plays on two devices report the same.
//! - A command buffer waiting on an event the CPU signals later does not stop another queue's.
//! - Textures are made at their recorded resource IDs, and read back as written.
//!
//! Each device takes its queue plan ([`at_plan`]): the player's readbacks take a command queue of
//! their own beside the stream's.

use super::host_memory::on_device_with;
use super::{at_plan, public_case};
use crate::replay::{OldRuntime, Player, Report, StreamReader, StreamWriter, Value};
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
    let mut backend = OldRuntime::new(device.clone());
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
    w.call(
        "useResource:usage:",
        11,
        0,
        &[Value::Object(8), Value::Uint(1)],
        &[],
    )?;
    let grid = Value::List(vec![Value::Uint(4), Value::Uint(1), Value::Uint(1)]);
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
            let range = Value::List(vec![Value::Uint(0), Value::Uint(16)]);
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
        w.call("waitUntilCompleted", 7, 0, &[], &[])?;
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

/// Two Shared textures made at recorded resource IDs 77 and 5 have them, and one reads back the
/// texels `replaceRegion` gave it.
#[test]
fn textures_are_made_at_their_recorded_resource_ids() {
    let dir = ScratchDir::new("replay-texture-ids").unwrap();
    let texels: Vec<u8> = (0..64).collect();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        // RGBA8Unorm, 4 by 4, shader-read, Shared.
        let descriptor = Value::Map(
            [
                ("pixelFormat", 70),
                ("width", 4),
                ("height", 4),
                ("usage", 1),
            ]
            .into_iter()
            .map(|(key, value)| (key.into(), Value::Uint(value)))
            .collect(),
        );
        for (id, resource_id) in [(2, 77), (3, 5)] {
            let answer = [("gpuResourceID", Value::Uint(resource_id))];
            w.call(
                "newTextureWithDescriptor:",
                1,
                id,
                std::slice::from_ref(&descriptor),
                &answer,
            )?;
        }
        let region = Value::List([0, 0, 0, 4, 4, 1].map(Value::Uint).to_vec());
        let args = [
            region,
            Value::Uint(0),
            Value::Bytes(texels.clone()),
            Value::Uint(16),
        ];
        w.call(
            "replaceRegion:mipmapLevel:withBytes:bytesPerRow:",
            2,
            0,
            &args,
            &[],
        )?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(2)],
            &[("sha256", sha(&texels))],
        )
    });
    on_device_with(at_plan, |device| {
        let report = play("texture-ids", device, &path);
        assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    });
}
