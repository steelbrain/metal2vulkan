//! Device tests of the replay player on the new runtime (`replay::NewRuntime`), on the device the
//! cfg selects: streams the tests write (`docs/REPLAY-STREAM.md`), each played and its report
//! printed as `REPLAY <test> <line>`.
//!
//! - A draw of a render pipeline state, its vertices in buffers bound to the vertex stage, covers
//!   a pass's target that its setter records clear to red: a blit copies the target to a buffer,
//!   which the stream checks green, and nothing is skipped.
//! - A draw of points after it, which the runtime does not build, is skipped and counted, and the
//!   report says the play is no verdict past it: its gap, frame and call.
//! - A Managed buffer of a stream recorded with two copies keeps both on any device (the player's
//!   own second copy where the device has one): a blit's fill reaches the GPU's copy and not the
//!   CPU's, an announcement with no write puts the CPU's bytes back into the GPU's copy at the next
//!   commit, and a synchronize brings the GPU's bytes into the CPU's copy.
//! - In a stream whose header marks its announced bytes recorded, an announcement after a
//!   read-back puts the stream's bytes into the GPU's copy, not the bytes read back, which a check
//!   reads until then; without the mark it puts the bytes read back.
//! - Sampler states take the lowest index of their class that no live object holds and the
//!   program did not reserve: made and released past the class's 3,999 indices, they reuse the
//!   released ones. A reserved index is taken only where the stream names it, by a descriptor's
//!   resource index or by the ID the program saw of the state.
//! - A texture over a surface, made as an ordinary texture in its place, takes the index its
//!   descriptor names, inside a reservation of every storage-capable index (as a program that
//!   reserves Metal's texture IDs 1 to 16,320 for its own textures makes them): made and released
//!   past the class's 4,095 indices, each takes its own.
//!
//! The new runtime's device is opened behind the hang tripwire, as every device open in this crate
//! is (`gpu.rs`), its kernels kept in a cache the test owns ([`ScratchCache`]).

use super::ScratchCache;
use crate::gpu::{self, DeviceType};
use crate::replay::{
    Backend, NewRuntime, Player, Report, StreamReader, StreamWriter, Value, WAIT_COMPLETED,
};
use crate::ScratchDir;
use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::{Path, PathBuf};

/// The new runtime's first device of the cfg's kind (a GPU: discrete, then integrated, then
/// virtual; else the CPU), its kernels kept in `cache`.
fn rt_device(cache: &Path) -> metal2vulkan_rt::Device {
    use ash::vk::PhysicalDeviceType as Kind;
    let kinds: &[Kind] = match gpu::TEST_DEVICE {
        DeviceType::Gpu => {
            gpu::tripwire::refuse_if_tripped().unwrap();
            &[Kind::DISCRETE_GPU, Kind::INTEGRATED_GPU, Kind::VIRTUAL_GPU]
        }
        DeviceType::Cpu => &[Kind::CPU],
    };
    let instance = metal2vulkan_rt::Instance::new(&[]).unwrap();
    let (_, vk_instance) = instance.raw();
    let mut physical = instance.physical_devices().unwrap();
    // SAFETY: each is one of the instance's physical devices, and the instance is live.
    let kind_of = |p: &metal2vulkan_rt::PhysicalDevice| unsafe {
        vk_instance
            .get_physical_device_properties(p.raw())
            .device_type
    };
    physical.retain(|p| kinds.contains(&kind_of(p)));
    physical.sort_by_key(|p| kinds.iter().position(|&kind| kind == kind_of(p)));
    let physical = physical.first().expect("a device of the cfg's kind");
    let mut options = metal2vulkan_rt::DeviceOptions::default();
    options.pipeline_cache_dir = Some(cache.to_path_buf());
    metal2vulkan_rt::Device::new(physical, options).unwrap()
}

fn written(
    dir: &ScratchDir,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> PathBuf {
    flagged(dir, 0, build)
}

/// A stream `build` writes in `dir`, its header's flags `flags`.
fn flagged(
    dir: &ScratchDir,
    flags: u32,
    build: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> PathBuf {
    let path = dir.path().join("stream.m2vr");
    let file = File::create(&path).unwrap();
    let mut writer = StreamWriter::with_flags(file, flags).unwrap();
    build(&mut writer).unwrap();
    writer.finish().unwrap();
    path
}

/// `path` played on the new runtime's device, its report and the backend's lines printed; the
/// play's end unwrapped.
fn play(test: &str, path: &Path) -> (Report, Vec<String>) {
    let (report, lines, _) = played(test, path, &[]);
    (report, lines)
}

/// `path` played as [`play`] plays it, and the index (the low half of the `gpuResourceID`) the
/// runtime then gives each of `objects`.
fn played(test: &str, path: &Path, objects: &[u64]) -> (Report, Vec<String>, Vec<u32>) {
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path())).unwrap();
    let mut reader = StreamReader::open(&[path.to_path_buf()], None).unwrap();
    let (report, ended) = Player::new(&mut backend).play(&mut reader);
    let lines = backend.lines();
    for line in report.lines().iter().chain(&lines) {
        eprintln!("REPLAY {test} {line}");
    }
    ended.unwrap();
    let mut index = |object: &u64| backend.resource_id(*object).unwrap().expect("an ID") as u32;
    let indices = objects.iter().map(&mut index).collect();
    (report, lines, indices)
}

/// A sampler descriptor's map: the default descriptor, its resource index `index` where given.
fn sampler(index: Option<u64>) -> Value {
    let index = index.map(|index| ("resourceIndex".into(), Value::Uint(index)));
    Value::Map(index.into_iter().collect())
}

/// A recorder struct's map of unsigned fields.
fn uints(fields: &[(&str, u64)]) -> Value {
    Value::Map(
        fields
            .iter()
            .map(|&(key, value)| (key.into(), Value::Uint(value)))
            .collect(),
    )
}

fn floats(values: &[f32]) -> Value {
    Value::Bytes(values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// The render fixture's `colored_vertex` (a vertex function of positions and colours by vertex
/// index, from buffers 0 and 1) and `colored_fragment` (its colour), in a metallib. A metallib
/// function's module declares that function as its one entry, so each function's module is the
/// fixture with its entry lists cut to that function's, assembled in `dir`.
fn render_library(dir: &ScratchDir) -> Vec<u8> {
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let text = std::fs::read_to_string(ll).unwrap();
    let lists = ["!air.kernel = ", "!air.vertex = ", "!air.fragment = "];
    let function = |name: &'static str, type_code, list: &str| {
        let mut module: String = (text.lines())
            .filter(|line| !lists.iter().any(|list| line.starts_with(list)))
            .flat_map(|line| [line, "\n"])
            .collect();
        // The function's own entry node: `!N = !{ptr @name, ...}`.
        let node = text.lines().find_map(|line| {
            let (node, rest) = line.split_once(" = ")?;
            rest.starts_with(&format!("!{{ptr @{name}, "))
                .then_some(node)
        });
        module.push_str(&format!("{list}!{{{}}}\n", node.expect("the entry node")));
        let path = dir.path().join(format!("{name}.ll"));
        std::fs::write(&path, module).unwrap();
        MtlbFunction {
            name,
            type_code,
            bitcode: wrap(&metal2vulkan_devtools::llvm_assemble(&path).unwrap()),
            constants: Vec::new(),
            public: Vec::new(),
            tags: Vec::new(),
        }
    };
    metallib(&[
        function("colored_vertex", 0, lists[1]),
        function("colored_fragment", 1, lists[2]),
    ])
}

/// A stream of a draw into a pass, its target copied to a buffer that the stream checks green
/// (`green`); with `points`, a draw of points (call 19) after the draw.
fn draw_into_a_pass(dir: &ScratchDir, green: &[u8], points: bool) -> PathBuf {
    written(dir, |w| {
        let library = w.blob(&render_library(dir))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call(
            "newLibraryWithData:error:",
            1,
            3,
            &[Value::Blob(library)],
            &[],
        )?;
        let vertex = [Value::Str("colored_vertex".into())];
        w.call("newFunctionWithName:", 3, 4, &vertex, &[])?;
        let fragment = [Value::Str("colored_fragment".into())];
        w.call("newFunctionWithName:", 3, 5, &fragment, &[])?;
        // The short descriptor: the two functions and colour 0's format, RGBA8Unorm.
        let pipeline = Value::Map(vec![
            ("vertex".into(), Value::Object(4)),
            ("fragment".into(), Value::Object(5)),
            ("color0PixelFormat".into(), Value::Uint(70)),
        ]);
        let made = "newRenderPipelineStateWithDescriptor:error:";
        w.call(made, 1, 6, &[pipeline], &[])?;
        // RGBA8Unorm, 4 by 4, shader-read and render-target, Private.
        let target = uints(&[
            ("textureType", 2),
            ("pixelFormat", 70),
            ("width", 4),
            ("height", 4),
            ("usage", 5),
            ("storageMode", 2),
        ]);
        w.call("newTextureWithDescriptor:", 1, 7, &[target], &[])?;
        // One triangle over the whole target, green at each corner.
        let positions = floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0, 0.0, 0.0]);
        let colors = floats(&[0.0, 1.0, 0.0, 1.0].repeat(3));
        for (id, bytes) in [(8, positions), (9, colors)] {
            let Value::Bytes(data) = &bytes else {
                unreachable!("bytes");
            };
            let args = [
                bytes.clone(),
                Value::Uint(data.len() as u64),
                Value::Uint(0),
            ];
            w.call("newBufferWithBytes:length:options:", 1, id, &args, &[])?;
        }
        let shared = [Value::Uint(64), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 10, &shared, &[])?;
        w.call("MTLRenderPassDescriptor.new", 0, 11, &[], &[])?;
        let pass = uints(&[("renderTargetWidth", 4), ("renderTargetHeight", 4)]);
        w.call("m2v.setPass", 11, 0, &[pass], &[])?;
        let Value::Map(mut color) = uints(&[("level", 0), ("loadAction", 2), ("storeAction", 1)])
        else {
            unreachable!("a map");
        };
        color.push(("texture".into(), Value::Object(7)));
        let red = [("red", 1.0), ("green", 0.0), ("blue", 0.0), ("alpha", 1.0)]
            .map(|(key, value)| (key.into(), Value::F64(value)));
        color.push(("clearColor".into(), Value::Map(red.to_vec())));
        let args = [Value::Uint(0), Value::Map(color)];
        w.call("m2v.setColorAttachment", 11, 0, &args, &[])?;
        w.call("commandBuffer", 2, 12, &[], &[])?;
        let pass = [Value::Object(11)];
        w.call("renderCommandEncoderWithDescriptor:", 12, 13, &pass, &[])?;
        w.call("setRenderPipelineState:", 13, 0, &[Value::Object(6)], &[])?;
        for (buffer, index) in [(8, 0), (9, 1)] {
            let args = [Value::Object(buffer), Value::Uint(0), Value::Uint(index)];
            w.call("setVertexBuffer:offset:atIndex:", 13, 0, &args, &[])?;
        }
        // MTLPrimitiveTypeTriangle, from vertex 0, three vertices.
        let draw = [Value::Uint(3), Value::Uint(0), Value::Uint(3)];
        w.call("drawPrimitives:vertexStart:vertexCount:", 13, 0, &draw, &[])?;
        if points {
            // MTLPrimitiveTypePoint: call 19.
            let draw = [Value::Uint(0), Value::Uint(0), Value::Uint(3)];
            w.call("drawPrimitives:vertexStart:vertexCount:", 13, 0, &draw, &[])?;
        }
        w.call("endEncoding", 13, 0, &[], &[])?;
        w.call("blitCommandEncoder", 12, 14, &[], &[])?;
        let copy = [
            Value::Object(7),
            Value::Uint(0),
            Value::Uint(0),
            uints(&[("x", 0), ("y", 0), ("z", 0)]),
            uints(&[("width", 4), ("height", 4), ("depth", 1)]),
            Value::Object(10),
            Value::Uint(0),
            Value::Uint(16),
            Value::Uint(64),
        ];
        let selector = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:\
                        destinationOffset:destinationBytesPerRow:destinationBytesPerImage:";
        w.call(selector, 14, 0, &copy, &[])?;
        w.call("endEncoding", 14, 0, &[], &[])?;
        w.call("commit", 12, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 12, 0, &[], &[])?;
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(green).to_vec())]);
        w.call("m2v.check", 0, 0, &[Value::Object(10)], &[("sha256", sum)])
    })
}

#[test]
fn a_draw_into_a_pass_plays_on_the_new_runtime() {
    let dir = ScratchDir::new("rt-replay-draw").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let path = draw_into_a_pass(&dir, &green, false);
    let (report, _) = play("draw", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    for skipped in [
        "skipped_encoders",
        "skipped_render_states",
        "skipped_draws",
        "skipped_calls",
        "skipped_objects",
    ] {
        assert_eq!(executed[skipped], 0, "{skipped}: {executed:?}");
    }
    assert!(executed["passes"] >= 1, "{executed:?}");
    assert_eq!(report.first_skip, None);
}

#[test]
fn a_draw_the_runtime_does_not_build_is_skipped_and_ends_the_verdict() {
    let dir = ScratchDir::new("rt-replay-points").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let path = draw_into_a_pass(&dir, &green, true);
    let (report, lines) = play("points", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    assert_eq!(executed["skipped_render_commands"], 1, "{executed:?}");
    assert_eq!(report.first_skip, Some((0, 19)));
    let note = report.verdict_note();
    assert_eq!(note.as_deref(), Some("not a verdict past frame 0, call 19"));
    let line = "skipped render commands Draw { kind: Points } 1, the first at frame 0, call 19";
    assert!(lines.iter().any(|l| l == line), "{lines:?}");
}

#[test]
fn a_two_copy_stream_s_managed_buffer_keeps_both_copies() {
    let dir = ScratchDir::new("rt-replay-two-copies").unwrap();
    let sum = |bytes: &[u8]| Value::List(vec![Value::Bytes(Sha256::digest(bytes).to_vec())]);
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let identity = [("hasUnifiedMemory", Value::Bool(false))];
        w.call("m2v.deviceIdentity", 1, 0, &[], &identity)?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        // #3 Managed (MTLResourceStorageModeManaged, 1 << 4), #4 and #5 Shared, 4 bytes each.
        let made = "newBufferWithLength:options:";
        for (id, options) in [(3, 16), (4, 0), (5, 0)] {
            w.call(made, 1, id, &[Value::Uint(4), Value::Uint(options)], &[])?;
        }
        let range = uints(&[("location", 0), ("length", 4)]);
        let fill = |value| vec![Value::Object(3), range.clone(), Value::Uint(value)];
        let copy = |to| {
            let (zero, four) = (Value::Uint(0), Value::Uint(4));
            vec![
                Value::Object(3),
                zero.clone(),
                Value::Object(to),
                zero,
                four,
            ]
        };
        let copy_from = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
        let check = |w: &mut StreamWriter<File>, id, bytes: [u8; 4]| {
            w.call(
                "m2v.check",
                0,
                0,
                &[Value::Object(id)],
                &[("sha256", sum(&bytes))],
            )
        };
        // Each blit command in an encoder of its own, the encoders in command buffer `cb`.
        let blits = |w: &mut StreamWriter<File>, cb, commands: &[(&str, Vec<Value>)]| {
            w.call("commandBuffer", 2, cb, &[], &[])?;
            for (at, (selector, args)) in commands.iter().enumerate() {
                let encoder = cb + 1 + at as u64;
                w.call("blitCommandEncoder", cb, encoder, &[], &[])?;
                w.call(selector, encoder, 0, args, &[])?;
                w.call("endEncoding", encoder, 0, &[], &[])?;
            }
            w.call("commit", cb, 0, &[], &[])?;
            w.call(WAIT_COMPLETED, cb, 0, &[], &[])
        };
        // A fill of 7s reaches the GPU's copy, which a copy into #4 reads; the CPU's copy keeps
        // the zeros it was made with.
        let first = [("fillBuffer:range:value:", fill(7)), (copy_from, copy(4))];
        blits(w, 10, &first)?;
        check(w, 4, [7; 4])?;
        check(w, 3, [0; 4])?;
        // An announcement with no write: the next commit puts the CPU's zeros back, which a copy
        // into #5 reads. Then a fill of 9s and a synchronize: the CPU's copy takes the GPU's bytes.
        w.call("didModifyRange:", 3, 0, std::slice::from_ref(&range), &[])?;
        let buffer = Value::Map(vec![("Buffer".into(), Value::Object(3))]);
        let second = [
            (copy_from, copy(5)),
            ("fillBuffer:range:value:", fill(9)),
            ("synchronizeResource:", vec![buffer]),
        ];
        blits(w, 20, &second)?;
        check(w, 5, [0; 4])?;
        check(w, 3, [9; 4])
    });
    let (report, lines) = play("two-copies", &path);
    assert_eq!(report.checks.len(), 4, "{lines:?}");
    for check in &report.checks {
        let object = check.object;
        assert_eq!(Some(check.sha256), check.recorded, "#{object}: {lines:?}");
    }
}

/// A two-copy stream's Managed buffer #3 of 4 bytes, its header's flags `flags`: the program
/// writes 5s into its CPU copy and announces them; a fill of 9s reaches the GPU's copy, and a
/// synchronize reads it back, which the program's check of #3 sees; the program then puts its 5s
/// back without the stream writing them again (the stream's last write of the bytes holds them, as
/// the announce invariant promises where the header marks it) and announces; a copy of #3 into
/// Shared #4 follows, and checks of #4 and #3 the 5s the program held. The play's checks, in
/// order: #3, #4, #3.
fn read_back_then_announced(test: &str, flags: u32) -> Vec<crate::replay::Check> {
    let dir = ScratchDir::new(test).unwrap();
    let sum = |bytes: [u8; 4]| Value::List(vec![Value::Bytes(Sha256::digest(bytes).to_vec())]);
    let path = flagged(&dir, flags, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let identity = [("hasUnifiedMemory", Value::Bool(false))];
        w.call("m2v.deviceIdentity", 1, 0, &[], &identity)?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        // #3 Managed (MTLResourceStorageModeManaged, 1 << 4), #4 Shared, 4 bytes each.
        let made = "newBufferWithLength:options:";
        for (id, options) in [(3, 16), (4, 0)] {
            w.call(made, 1, id, &[Value::Uint(4), Value::Uint(options)], &[])?;
        }
        let range = uints(&[("location", 0), ("length", 4)]);
        let announce = |w: &mut StreamWriter<File>| {
            w.call("didModifyRange:", 3, 0, std::slice::from_ref(&range), &[])
        };
        let check = |w: &mut StreamWriter<File>, id, bytes| {
            w.call(
                "m2v.check",
                0,
                0,
                &[Value::Object(id)],
                &[("sha256", sum(bytes))],
            )
        };
        // Each blit command in an encoder of its own, the encoders in command buffer `cb`.
        let blits = |w: &mut StreamWriter<File>, cb, commands: &[(&str, Vec<Value>)]| {
            w.call("commandBuffer", 2, cb, &[], &[])?;
            for (at, (selector, args)) in commands.iter().enumerate() {
                let encoder = cb + 1 + at as u64;
                w.call("blitCommandEncoder", cb, encoder, &[], &[])?;
                w.call(selector, encoder, 0, args, &[])?;
                w.call("endEncoding", encoder, 0, &[], &[])?;
            }
            w.call("commit", cb, 0, &[], &[])?;
            w.call(WAIT_COMPLETED, cb, 0, &[], &[])
        };
        let write = [Value::Uint(0), Value::Bytes(vec![5; 4])];
        w.call("m2v.write", 3, 0, &write, &[])?;
        announce(w)?;
        let fill = vec![Value::Object(3), range.clone(), Value::Uint(9)];
        let buffer = Value::Map(vec![("Buffer".into(), Value::Object(3))]);
        let read_back = [
            ("fillBuffer:range:value:", fill),
            ("synchronizeResource:", vec![buffer]),
        ];
        blits(w, 10, &read_back)?;
        check(w, 3, [9; 4])?;
        announce(w)?;
        let (zero, four) = (Value::Uint(0), Value::Uint(4));
        let copy = vec![Value::Object(3), zero.clone(), Value::Object(4), zero, four];
        let copy_from = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
        blits(w, 20, &[(copy_from, copy)])?;
        check(w, 4, [5; 4])?;
        check(w, 3, [5; 4])
    });
    let (report, lines) = play(test, &path);
    assert_eq!(report.checks.len(), 3, "{lines:?}");
    let rule = match flags {
        0 => "announced bytes: the CPU copy's, read-backs included (no header flag); 0 read-backs",
        _ => "announced bytes: the stream's (header flag bit 0); 1 read-backs into 1 asides",
    };
    assert!(lines.iter().any(|line| line.starts_with(rule)), "{lines:?}");
    report.checks
}

#[test]
fn a_marked_stream_s_announcement_after_a_read_back_puts_the_stream_s_bytes() {
    let flags = crate::replay::FLAG_ANNOUNCED_BYTES_RECORDED;
    for check in read_back_then_announced("rt-replay-marked", flags) {
        let object = check.object;
        assert_eq!(Some(check.sha256), check.recorded, "#{object}");
    }
}

#[test]
fn an_unmarked_stream_s_announcement_after_a_read_back_puts_the_bytes_read_back() {
    let checks = read_back_then_announced("rt-replay-unmarked", 0);
    let held: Vec<_> = checks
        .iter()
        .map(|c| c.recorded == Some(c.sha256))
        .collect();
    // The read-back's 9s, which the program's check saw, then 9s where it held 5s.
    assert_eq!(held, [true, false, false]);
    assert_eq!(checks[1].sha256[..], Sha256::digest([9u8; 4])[..]);
}

#[test]
fn sampler_states_made_and_released_past_their_class_reuse_the_lowest_free_index() {
    let dir = ScratchDir::new("rt-replay-samplers").unwrap();
    let made = "newSamplerStateWithDescriptor:";
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call(made, 1, 2, &[sampler(None)], &[])?;
        // More than the class's 3,999 indices, each released before the next is made.
        for object in 10..4110 {
            w.call(made, 1, object, &[sampler(None)], &[])?;
            w.call("release", object, 0, &[], &[])?;
        }
        w.call(made, 1, 3, &[sampler(None)], &[])
    });
    let (_, _, indices) = played("samplers", &path, &[2, 3]);
    assert_eq!(indices, [1, 2]);
}

#[test]
fn a_reserved_sampler_index_is_taken_only_where_the_stream_names_it() {
    let dir = ScratchDir::new("rt-replay-reserved").unwrap();
    let made = "newSamplerStateWithDescriptor:";
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        // The sampler type's indices 1 and 2, reserved.
        let indices = Value::List(vec![Value::Uint(1), Value::Uint(2)]);
        let reserve = [Value::Uint(1), indices, Value::Uint(2)];
        let selector = "reserveResourceIndicesForResourceType:indices:indexCount:";
        w.call(selector, 1, 0, &reserve, &[])?;
        w.call(made, 1, 2, &[sampler(None)], &[])?;
        w.call(made, 1, 3, &[sampler(Some(2))], &[])
    });
    let (_, _, indices) = played("reserved", &path, &[2, 3]);
    assert_eq!(indices, [3, 2]);
}

#[test]
fn surface_textures_take_the_reserved_index_their_descriptor_names() {
    let dir = ScratchDir::new("rt-replay-surfaces").unwrap();
    // RGBA8Unorm, 1 by 1, shader-read and shader-write, at index `index`, forced.
    let texture = |index: u64| {
        let index = Value::List(vec![Value::Uint(index), Value::Bool(true)]);
        let Value::Map(mut fields) = uints(&[
            ("textureType", 2),
            ("pixelFormat", 70),
            ("width", 1),
            ("height", 1),
            ("usage", 3),
        ]) else {
            unreachable!("a map");
        };
        fields.push(("resourceIndex".into(), index));
        Value::Map(fields)
    };
    // Plane 0 of the surface's pages from their start, 4 bytes a row, its facts not read.
    let plane = Value::Map(vec![
        ("plane".into(), Value::Uint(0)),
        ("offset".into(), Value::Uint(0)),
        ("bytesPerRow".into(), Value::Uint(4)),
        ("planar".into(), Value::Bool(false)),
        ("facts".into(), Value::Nil),
    ]);
    let made = "newTextureWithDescriptor:iosurface:plane:";
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        // The texture type's indices 1 to 16,320 reserved: every storage-capable texture's index.
        let indices = Value::List((1..=16_320).map(Value::Uint).collect());
        let reserve = [Value::Uint(0), indices, Value::Uint(16_320)];
        let selector = "reserveResourceIndicesForResourceType:indices:indexCount:";
        w.call(selector, 1, 0, &reserve, &[])?;
        // #2, the surface's pages: a Shared buffer of 4 bytes.
        let pages = [Value::Uint(4), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 2, &pages, &[])?;
        // More than the class's 4,095 indices, each texture released before the next is made.
        for object in 10..4110 {
            w.call(
                made,
                2,
                object,
                &[texture(object % 4095 + 1), plane.clone()],
                &[],
            )?;
            w.call("release", object, 0, &[], &[])?;
        }
        w.call(made, 2, 3, &[texture(4095), plane.clone()], &[])?;
        w.call(made, 2, 4, &[texture(7), plane.clone()], &[])
    });
    let (_, lines, indices) = played("surfaces", &path, &[3, 4]);
    assert_eq!(indices, [4095, 7]);
    let named = "indices named 4102:";
    assert!(
        lines.iter().any(|line| line.starts_with(named)),
        "{lines:?}"
    );
}

#[test]
fn a_sampler_state_takes_the_index_of_the_id_the_program_saw() {
    let dir = ScratchDir::new("rt-replay-recorded").unwrap();
    let made = "newSamplerStateWithDescriptor:";
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call(made, 1, 2, &[sampler(None)], &[])?;
        w.call(made, 1, 3, &[sampler(None)], &[])?;
        // The program saw #2 at index 7, the default descriptor's word 0 (docs/RESOURCE-ID.md).
        let id = [("gpuResourceID", Value::Uint(7))];
        w.call("gpuResourceID", 2, 0, &[], &id)
    });
    let (report, _, indices) = played("recorded", &path, &[2, 3]);
    assert_eq!(indices, [7, 1]);
    assert!(report.ids.is_empty(), "{:?}", report.ids);
}
