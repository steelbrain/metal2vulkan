//! Device tests of the replay player on the new runtime (`replay::NewRuntime`), on the device the
//! cfg selects: streams the tests write (`docs/REPLAY-STREAM.md`), each played and its report
//! printed as `REPLAY <test> <line>`.
//!
//! - A draw of a render pipeline state, its vertices in buffers bound to the vertex stage, covers
//!   a pass's target that its setter records clear to red: a blit copies the target to a buffer,
//!   which the stream checks green, and nothing is skipped.
//! - A draw of points after it, which the runtime does not build, is skipped and counted, and the
//!   report says the play is no verdict past it: its gap, frame and call.
//! - `setLineWidth:` plays: at width 1 a draw of lines after it plays; at width 2 a draw of
//!   triangles plays and one of lines is the runtime's gap, skipped and counted.
//! - A compute pipeline state the runtime answers with a typed gap is skipped and counted by the
//!   gap, and a dispatch of it is skipped, and the play goes on.
//! - A compute encoder's `setStageInRegion:` plays: the runtime takes it.
//! - The host's library from source is made of its display library where the backend has it, a
//!   function of it with it; without it both are skipped.
//! - A sampler bound with LOD clamps whose samples read as its own plays with them (its own
//!   clamps, and (0, 1000) over its (0, FLT_MAX)); one bound with clamps that read otherwise is
//!   the runtime's typed gap, skipped and counted, and the draw after it is skipped.
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
//! - A texture over a surface, a view of the runtime's texture under the surface's range, takes the
//!   index its descriptor names, inside a reservation of every storage-capable index (as a program
//!   that reserves Metal's texture IDs 1 to 16,320 for its own textures makes them): made and
//!   released past the class's 4,095 indices, each takes its own.
//! - Two textures of different formats over one surface's plane share its texels: what a kernel
//!   writes through the one, a later kernel reads through the other. A texture of another extent
//!   over the range starts from those texels by byte offset, a copy from the guest's RAM into the
//!   range reaches it, and a binding of the older extent's texture after it is counted.
//! - A buffer placed in a block of the stream that the runtime may take from a released block's
//!   bytes reads zero where nothing wrote it: the player zeroes each block and heap of the stream
//!   once at its making, as the live system does.
//! - A sampler state placed at an index a held commit owes (its command buffer waits for an event
//!   only a later record signals) takes it without waiting: where its descriptor is the one placed
//!   there last, as Metal hands an equal descriptor the same ID; otherwise as the recording
//!   device's index policy gave the index away before the commit completed, counted on a line of
//!   its own.
//! - A command buffer that names a sampler state after a later one took its index owes the index
//!   nothing: of two live sampler states placed at one index, the index is the later one's.
//! - A play that ends with a command buffer held (it waits for an event no record signals, and
//!   holds a sampler state, whose handle holds the device) ends its device all the same: the
//!   backend's end quiesces it and drops every handle, and the gauges read zero.
//! - A buffer's linear texture (the SPI `newLinearTextureWithDescriptor:offset:bytesPerRow:
//!   bytesPerImage:`) of a 2D descriptor plays as the public texture over the buffer, and one of a
//!   2D array, whose slices Metal lays out itself, is refused.
//! - A no-copy buffer plays by its creator. A bytes-form one is made over host pages the player
//!   imports: a texture over it is the runtime's counted copy (one copy of its 64 bytes, on the
//!   imported side), a blit of the texture reads the stream's bytes, and the imports end with the
//!   device. A descriptor-form one is a pool buffer with no import: recorded inside a block's
//!   recorded range, a buffer of the block, which reads the bytes written into the block there;
//!   outside every block, a place of its own, a texture over which reads the stream's bytes with
//!   no copy on the imported side. The report's gap line bounds the ones that may be the program's
//!   own memory by those outside every block with a pinned address and no `zeroFrom`.
//! - Buffers keep the slack of their recorded granule: two recorded at addresses of 32 KiB and
//!   64 KiB granules take places that span their lengths rounded up to those granules, the slack
//!   the places hold the sum of both.
//! - A word that looks like an address (its value lies in a live recorded range, as two 32-bit
//!   indices can) plays as written, and the census of such words is the same on both runtimes.
//!
//! The new runtime's device is opened behind the hang tripwire, as every device open in this crate
//! is (`gpu.rs`), its kernels kept in a cache the test owns ([`ScratchCache`]).

use super::host_memory::on_device_with;
use super::{at_plan, ScratchCache};
use crate::gpu::{self, DeviceType};
use crate::replay::{
    Backend, NewRuntime, OldRuntime, PlayError, Player, Report, StreamReader, StreamWriter, Value,
    WAIT_COMPLETED,
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
    let (report, lines, ()) = play_then(test, path, |_| ());
    (report, lines)
}

/// `path` played as [`play`] plays it, and the index (the low half of the `gpuResourceID`) the
/// runtime then gives each of `objects`.
fn played(test: &str, path: &Path, objects: &[u64]) -> (Report, Vec<String>, Vec<u32>) {
    play_then(test, path, |backend| {
        let mut index = |object: &u64| backend.resource_id(*object).unwrap().expect("an ID") as u32;
        objects.iter().map(&mut index).collect()
    })
}

/// `path` played as [`play`] plays it, then `then` of the backend.
fn play_then<T>(
    test: &str,
    path: &Path,
    then: impl FnOnce(&mut NewRuntime) -> T,
) -> (Report, Vec<String>, T) {
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let mut reader = StreamReader::open(&[path.to_path_buf()], None).unwrap();
    let (report, ended) = Player::new(&mut backend).play(&mut reader);
    let lines = backend.lines();
    for line in report.lines().iter().chain(&lines) {
        eprintln!("REPLAY {test} {line}");
    }
    ended.unwrap();
    let then = then(&mut backend);
    (report, lines, then)
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
    library(dir, &[("colored_vertex", 0, 1), ("colored_fragment", 1, 2)])
}

/// A metallib of `functions` of the runtime examples ([`library_of`]).
fn library(dir: &ScratchDir, functions: &[(&'static str, u8, usize)]) -> Vec<u8> {
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    library_of(dir, &ll, functions)
}

/// A metallib of `functions` of the fixture `ll`, each `(name, type code, list)`: `list` 0 the
/// kernels', 1 the vertex functions', 2 the fragment functions'.
pub(super) fn library_of(
    dir: &ScratchDir,
    ll: &Path,
    functions: &[(&'static str, u8, usize)],
) -> Vec<u8> {
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
    let made: Vec<MtlbFunction> = (functions.iter())
        .map(|&(name, type_code, list)| function(name, type_code, lists[list]))
        .collect();
    metallib(&made)
}

/// A stream of a draw into a pass, its target copied to a buffer that the stream checks green
/// (`green`); with `points`, a draw of points (call 19) after the draw.
fn draw_into_a_pass(dir: &ScratchDir, green: &[u8], then: Then) -> PathBuf {
    draw_with_clamps(dir, green, then, None)
}

/// What [`draw_with_clamps`]' stream draws after its triangle.
#[derive(Clone, Copy)]
enum Then {
    Nothing,
    /// The triangle's vertices as points: call 19.
    Points,
    /// `setLineWidth:` of the width (call 19), the triangle again (call 20), and its first two
    /// vertices as a line (call 21).
    Lines(f32),
}

/// [`draw_into_a_pass`]'s stream; where `clamp` is given, a sampler state of the default
/// descriptor (#15) bound at fragment slot 0 with LOD clamps 0 and `clamp` before the draw.
fn draw_with_clamps(dir: &ScratchDir, green: &[u8], then: Then, clamp: Option<f32>) -> PathBuf {
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
        if let Some(clamp) = clamp {
            w.call(
                "newSamplerStateWithDescriptor:",
                1,
                15,
                &[sampler(None)],
                &[],
            )?;
            let args = [
                Value::List(vec![Value::Object(15)]),
                Value::List(vec![Value::F32(0.0)]),
                Value::List(vec![Value::F32(clamp)]),
                uints(&[("location", 0), ("length", 1)]),
            ];
            let bind = "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:";
            w.call(bind, 13, 0, &args, &[])?;
        }
        // MTLPrimitiveTypeTriangle, from vertex 0, three vertices.
        let draw = [Value::Uint(3), Value::Uint(0), Value::Uint(3)];
        w.call("drawPrimitives:vertexStart:vertexCount:", 13, 0, &draw, &[])?;
        match then {
            Then::Nothing => {}
            Then::Points => {
                // MTLPrimitiveTypePoint: call 19.
                let draw = [Value::Uint(0), Value::Uint(0), Value::Uint(3)];
                w.call("drawPrimitives:vertexStart:vertexCount:", 13, 0, &draw, &[])?;
            }
            Then::Lines(width) => {
                w.call("setLineWidth:", 13, 0, &[Value::F32(width)], &[])?;
                w.call("drawPrimitives:vertexStart:vertexCount:", 13, 0, &draw, &[])?;
                // MTLPrimitiveTypeLine, vertices 0 and 1, along the target's bottom edge.
                let line = [Value::Uint(1), Value::Uint(0), Value::Uint(2)];
                w.call("drawPrimitives:vertexStart:vertexCount:", 13, 0, &line, &[])?;
            }
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
    let path = draw_into_a_pass(&dir, &green, Then::Nothing);
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
    let path = draw_into_a_pass(&dir, &green, Then::Points);
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

/// `setLineWidth:` plays: at width 1 the triangle and the line after it play, nothing skipped; at
/// width 2 the triangle draws and the line is the runtime's gap, skipped and counted.
#[test]
fn a_line_width_plays_and_a_line_drawn_at_another_width_is_skipped() {
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let dir = ScratchDir::new("rt-replay-line-width-1").unwrap();
    let (report, _) = play(
        "line-width-1",
        &draw_into_a_pass(&dir, &green, Then::Lines(1.0)),
    );
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    for skipped in ["skipped_render_commands", "skipped_draws", "skipped_calls"] {
        assert_eq!(executed[skipped], 0, "{skipped}: {executed:?}");
    }
    assert_eq!(report.first_skip, None);
    let dir = ScratchDir::new("rt-replay-line-width-2").unwrap();
    let (report, lines) = play(
        "line-width-2",
        &draw_into_a_pass(&dir, &green, Then::Lines(2.0)),
    );
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    assert_eq!(executed["skipped_render_commands"], 1, "{executed:?}");
    assert_eq!(report.first_skip, Some((0, 21)));
    let line = "skipped render commands Draw { kind: LineWidth } 1, the first at frame 0, call 21";
    assert!(lines.iter().any(|l| l == line), "{lines:?}");
}

#[test]
fn the_host_s_library_from_source_is_made_of_its_display_library() {
    let dir = ScratchDir::new("rt-replay-display").unwrap();
    // The host's library from source (#2, no arguments), and a function of it.
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newLibraryWithSource:options:error:", 1, 2, &[], &[])?;
        let vertex = [Value::Str("colored_vertex".into())];
        w.call("newFunctionWithName:", 2, 3, &vertex, &[])
    });
    let played = |display: Option<Vec<u8>>| {
        let cache = ScratchCache::new().unwrap();
        let backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
        let mut backend = match display {
            Some(metallib) => backend.with_display_library(metallib),
            None => backend,
        };
        let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
        let (report, ended) = Player::new(&mut backend).play(&mut reader);
        ended.unwrap();
        let executed = &report.total.executed;
        (executed["skipped_objects"], executed["skipped_calls"])
    };
    assert_eq!(played(Some(render_library(&dir))), (0, 0));
    // Without it, the library is skipped, and so is the call on it that makes the function.
    assert_eq!(played(None), (1, 1));
}

#[test]
fn a_sampler_bound_with_lod_clamps_plays_where_the_runtime_takes_them() {
    let dir = ScratchDir::new("rt-replay-clamps").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    // The sampler's own clamps, and (0, 1000), which reads as its own (0, FLT_MAX): both bind.
    for (test, clamp) in [("clamps-own", f32::MAX), ("clamps-1000", 1000.0)] {
        let (report, _) = play(
            test,
            &draw_with_clamps(&dir, &green, Then::Nothing, Some(clamp)),
        );
        assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
        let executed = &report.total.executed;
        assert_eq!(executed["skipped_clamped_samplers"], 0, "{executed:?}");
        assert_eq!(executed["skipped_draws"], 0, "{executed:?}");
    }
    // (0, 2) reads otherwise: the runtime's gap, the bind and the draw skipped.
    let (report, _) = play(
        "clamps-2",
        &draw_with_clamps(&dir, &green, Then::Nothing, Some(2.0)),
    );
    let executed = &report.total.executed;
    assert_eq!(executed["skipped_clamped_samplers"], 1, "{executed:?}");
    assert_eq!(executed["skipped_draws"], 1, "{executed:?}");
}

#[test]
fn a_render_state_the_runtime_does_not_build_is_counted_by_its_gap() {
    let dir = ScratchDir::new("rt-replay-state-gap").unwrap();
    let path = written(&dir, |w| {
        let library = w.blob(&render_library(&dir))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let library = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &library, &[])?;
        let vertex = [Value::Str("colored_vertex".into())];
        w.call("newFunctionWithName:", 3, 4, &vertex, &[])?;
        let fragment = [Value::Str("colored_fragment".into())];
        w.call("newFunctionWithName:", 3, 5, &fragment, &[])?;
        // Colour 0 RGBA8Unorm (the short descriptor's key) rasterized at 3 samples, which no
        // Vulkan sample count is: the runtime refuses it.
        let pipeline = Value::Map(vec![
            ("vertex".into(), Value::Object(4)),
            ("fragment".into(), Value::Object(5)),
            ("color0PixelFormat".into(), Value::Uint(70)),
            ("rasterSampleCount".into(), Value::Uint(3)),
        ]);
        let made = "newRenderPipelineStateWithDescriptor:error:";
        w.call(made, 1, 6, &[pipeline], &[])
    });
    let (report, lines) = play("state-gap", &path);
    assert_eq!(report.total.executed["skipped_render_states"], 1);
    let line = "skipped render states RenderPipeline { kind: SampleCount } 1, the first #6: ";
    assert!(lines.iter().any(|l| l.starts_with(line)), "{lines:?}");
}

#[test]
fn a_compute_state_the_runtime_does_not_build_is_counted_and_its_dispatch_skipped() {
    let dir = ScratchDir::new("rt-replay-compute-gap").unwrap();
    let path = written(&dir, |w| {
        let library = w.blob(&library(&dir, &[("scale_and_offset", 2, 0)]))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call(
            "newLibraryWithData:error:",
            1,
            3,
            &[Value::Blob(library)],
            &[],
        )?;
        let kernel = [Value::Str("scale_and_offset".into())];
        w.call("newFunctionWithName:", 3, 4, &kernel, &[])?;
        // A state for indirect command buffers, which the runtime refuses.
        let state = Value::Map(vec![
            ("function".into(), Value::Object(4)),
            ("supportIndirectCommandBuffers".into(), Value::Bool(true)),
        ]);
        let made = "newComputePipelineStateWithDescriptor:options:reflection:error:";
        let options = [state, Value::Uint(0), Value::Nil];
        w.call(made, 1, 5, &options, &[])?;
        w.call("commandBuffer", 2, 6, &[], &[])?;
        w.call("computeCommandEncoder", 6, 7, &[], &[])?;
        w.call("setComputePipelineState:", 7, 0, &[Value::Object(5)], &[])?;
        let size = |n| uints(&[("width", n), ("height", 1), ("depth", 1)]);
        let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
        w.call(dispatch, 7, 0, &[size(1), size(1)], &[])?;
        w.call("endEncoding", 7, 0, &[], &[])?;
        w.call("commit", 6, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 6, 0, &[], &[])
    });
    let (report, lines) = play("compute-gap", &path);
    let executed = &report.total.executed;
    assert_eq!(executed["skipped_compute_states"], 1, "{executed:?}");
    assert_eq!(executed["skipped_dispatches"], 1, "{executed:?}");
    let line = "skipped compute states IndirectCommandBuffers 1, the first #5: ";
    assert!(lines.iter().any(|l| l.starts_with(line)), "{lines:?}");
}

#[test]
fn a_stage_in_region_plays_on_a_compute_encoder() {
    let dir = ScratchDir::new("rt-replay-stage-in").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("commandBuffer", 2, 3, &[], &[])?;
        w.call("computeCommandEncoder", 3, 4, &[], &[])?;
        let region = Value::Map(vec![
            ("origin".into(), uints(&[("x", 0), ("y", 0), ("z", 0)])),
            (
                "size".into(),
                uints(&[("width", 64), ("height", 1), ("depth", 1)]),
            ),
        ]);
        w.call("setStageInRegion:", 4, 0, &[region], &[])?;
        w.call("endEncoding", 4, 0, &[], &[])?;
        w.call("commit", 3, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 3, 0, &[], &[])
    });
    let (report, _) = play("stage-in", &path);
    let executed = &report.total.executed;
    assert_eq!(executed["skipped_calls"], 0, "{executed:?}");
    assert_eq!(report.first_skip, None);
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

/// A command buffer of queue #2 whose kernel (#5) reads `from` and writes the inverse into
/// `to`, waited for.
fn invert(
    w: &mut StreamWriter<File>,
    command_buffer: u64,
    from: u64,
    to: u64,
) -> std::io::Result<()> {
    let encoder = command_buffer + 1;
    w.call("commandBuffer", 2, command_buffer, &[], &[])?;
    w.call("computeCommandEncoder", command_buffer, encoder, &[], &[])?;
    let state = [Value::Object(5)];
    w.call("setComputePipelineState:", encoder, 0, &state, &[])?;
    for (texture, index) in [(from, 0), (to, 1)] {
        let bind = [Value::Object(texture), Value::Uint(index)];
        w.call("setTexture:atIndex:", encoder, 0, &bind, &[])?;
    }
    let size = |n| uints(&[("width", n), ("height", n), ("depth", 1)]);
    let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
    w.call(dispatch, encoder, 0, &[size(1), size(2)], &[])?;
    w.call("endEncoding", encoder, 0, &[], &[])?;
    w.call("commit", command_buffer, 0, &[], &[])?;
    w.call(WAIT_COMPLETED, command_buffer, 0, &[], &[])
}

/// Two textures over plane 0 of one surface (#6, 2 by 2 texels, 8 bytes a row): #8, RGBA8Unorm,
/// which a kernel writes the inverse of #7's texels into, and #9, BGRA8Unorm, a view of the same
/// shape, which a later command buffer's kernel reads and writes the inverse of into #10, so that
/// #10 holds #7's texels with red and blue swapped. Then #15, RGBA8Unorm of 1 by 2 over the same
/// plane, a new shape, which starts from the texels at its bytes once #8's write is in the buffer
/// (the first and third of #8's), and a copy from the guest's RAM into the plane's first 8 bytes,
/// which reaches both shapes: a kernel that then reads #8 into #20 reads the copy's bytes in #8's
/// first row. Last, a blit of #7's first column into #15: at its encoder's end the write goes into
/// #8's shape through the buffer, so a kernel reading #8 into #24 reads it in #8's first column.
#[test]
fn textures_over_one_surface_share_its_texels() {
    let dir = ScratchDir::new("rt-replay-surface-views").unwrap();
    let source = [0x10, 0x20, 0x30, 0x40].repeat(4);
    let written_through = [0xEF, 0xDF, 0xCF, 0x40].repeat(4);
    let read_through = [0x30, 0x20, 0x10, 0x40].repeat(4);
    let copied: Vec<u8> = (1..=8).collect();
    let path = written(&dir, |w| {
        let library = w.blob(&library(&dir, &[("invert_texels", 2, 0)]))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let data = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &data, &[])?;
        let kernel = [Value::Str("invert_texels".into())];
        w.call("newFunctionWithName:", 3, 4, &kernel, &[])?;
        let function = [Value::Object(4)];
        let made = "newComputePipelineStateWithFunction:error:";
        w.call(made, 1, 5, &function, &[])?;
        let pages = [Value::Uint(16), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 6, &pages, &[])?;
        // Shared, of `width` by 2 texels.
        let texture = |format, usage, width| {
            uints(&[
                ("textureType", 2),
                ("pixelFormat", format),
                ("width", width),
                ("height", 2),
                ("usage", usage),
                ("storageMode", 0),
            ])
        };
        w.call("newTextureWithDescriptor:", 1, 7, &[texture(70, 1, 2)], &[])?;
        let whole = Value::Map(vec![
            ("origin".into(), uints(&[("x", 0), ("y", 0), ("z", 0)])),
            (
                "size".into(),
                uints(&[("width", 2), ("height", 2), ("depth", 1)]),
            ),
        ]);
        let args = [
            whole,
            Value::Uint(0),
            Value::Bytes(source.clone()),
            Value::Uint(8),
        ];
        let replace = "replaceRegion:mipmapLevel:withBytes:bytesPerRow:";
        w.call(replace, 7, 0, &args, &[])?;
        // Plane 0 of a surface of one plane, from the pages' start, 8 bytes a row.
        let facts = Value::Map(vec![
            (
                "pixelFormat".into(),
                Value::Uint(u64::from(u32::from_be_bytes(*b"BGRA"))),
            ),
            ("planeCount".into(), Value::Uint(0)),
            ("width".into(), Value::Uint(2)),
            ("height".into(), Value::Uint(2)),
            ("chromaOffset".into(), Value::Uint(0)),
            ("chromaBytesPerRow".into(), Value::Uint(0)),
            ("ycbcrMatrix".into(), Value::Nil),
        ]);
        let plane = Value::Map(vec![
            ("plane".into(), Value::Uint(0)),
            ("offset".into(), Value::Uint(0)),
            ("bytesPerRow".into(), Value::Uint(8)),
            ("planar".into(), Value::Bool(false)),
            ("facts".into(), facts),
        ]);
        let surface = "newTextureWithDescriptor:iosurface:plane:";
        w.call(surface, 6, 8, &[texture(70, 2, 2), plane.clone()], &[])?;
        w.call(surface, 6, 9, &[texture(80, 1, 2), plane.clone()], &[])?;
        w.call(
            "newTextureWithDescriptor:",
            1,
            10,
            &[texture(70, 2, 2)],
            &[],
        )?;
        invert(w, 11, 7, 8)?;
        invert(w, 13, 9, 10)?;
        w.call(surface, 6, 15, &[texture(70, 1, 1), plane], &[])?;
        w.call("commandBuffer", 2, 16, &[], &[])?;
        w.call("blitCommandEncoder", 16, 17, &[], &[])?;
        let piece = Value::List(vec![Value::Uint(0), Value::Bytes(copied.clone())]);
        let pieces = [Value::Object(6), Value::List(vec![piece])];
        w.call("m2v.copyFromGuestRam", 17, 0, &pieces, &[])?;
        w.call("endEncoding", 17, 0, &[], &[])?;
        w.call("commit", 16, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 16, 0, &[], &[])?;
        w.call(
            "newTextureWithDescriptor:",
            1,
            20,
            &[texture(70, 2, 2)],
            &[],
        )?;
        invert(w, 18, 8, 20)?;
        w.call("commandBuffer", 2, 22, &[], &[])?;
        w.call("blitCommandEncoder", 22, 23, &[], &[])?;
        let origin = uints(&[("x", 0), ("y", 0), ("z", 0)]);
        let column = uints(&[("width", 1), ("height", 2), ("depth", 1)]);
        let copy = [
            Value::Object(7),
            Value::Uint(0),
            Value::Uint(0),
            origin.clone(),
            column,
            Value::Object(15),
            Value::Uint(0),
            Value::Uint(0),
            origin,
        ];
        let blit = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:\
                    destinationSlice:destinationLevel:destinationOrigin:";
        w.call(blit, 23, 0, &copy, &[])?;
        w.call("endEncoding", 23, 0, &[], &[])?;
        w.call("commit", 22, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 22, 0, &[], &[])?;
        w.call(
            "newTextureWithDescriptor:",
            1,
            24,
            &[texture(70, 2, 2)],
            &[],
        )?;
        invert(w, 25, 8, 24)?;
        w.call("m2v.check", 0, 0, &[Value::Object(6)], &[])
    });
    // Each texture's rows as a check reads them: `width` texels a row, the rows as far apart as
    // half the read's length (each texture has 2 rows).
    let (report, lines, rows) = play_then("surface-views", &path, |backend| {
        let mut rows = |object: u64, width: usize| {
            let bytes = backend.read(object).unwrap();
            let pitch = bytes.len() / 2;
            [&bytes[..width * 4], &bytes[pitch..pitch + width * 4]].concat()
        };
        [rows(10, 2), rows(20, 2), rows(24, 2)]
    });
    // The inverse of RGBA8 texels' colours, as the kernel writes it.
    let inverse = |bytes: &[u8]| -> Vec<u8> {
        (bytes.chunks(4))
            .flat_map(|t| [0xFF - t[0], 0xFF - t[1], 0xFF - t[2], t[3]])
            .collect()
    };
    // #8 after the copy: the copy's bytes in its first row, its written texels in its second.
    let copied_over = [&copied[..], &written_through[8..]].concat();
    // #8 after the blit through #15: #7's texels in its first column.
    let blitted = [
        &source[..4],
        &copied[4..8],
        &source[8..12],
        &written_through[12..],
    ]
    .concat();
    assert_eq!(
        rows,
        [read_through, inverse(&copied_over), inverse(&blitted)],
        "{lines:?}"
    );
    let executed = &report.total.executed;
    let counted = |key: &str| executed.get(key).copied().unwrap_or(0);
    let counts = [
        "surface_textures",
        "surface_shapes",
        "surface_shapes_found",
        "surface_written_back",
        "surface_copies_in",
        "surface_overlap_copies",
        "surface_overlap_bytes",
        "surface_straddled",
        "surface_unordered_writes",
        "substitutes_read",
    ]
    .map(counted);
    assert_eq!(counts, [3, 2, 1, 1, 2, 1, 8, 0, 0, 0], "{executed:?}");
    let receivers: Vec<(u64, bool)> = (report.checks.iter())
        .map(|check| (check.object, check.receiver))
        .collect();
    assert_eq!(receivers, [(6, true)], "{lines:?}");
}

/// A surface's pictures fill its shapes at their first use, not at their landing: over Shared
/// buffer #6 (2 by 2 RGBA8Unorm texels, 8 bytes a row), #8 is a texture of plane 0 that a kernel
/// reads. Two pictures land (`m2v.write` of the bytes, then `m2v.externalWriteDone`) before any
/// command buffer uses #8: the first is never copied, and the kernel that then reads #8 into #10
/// reads the second, filled once before its command buffer.
#[test]
fn a_surface_picture_fills_its_shapes_at_their_first_use() {
    let dir = ScratchDir::new("rt-replay-surface-lazy").unwrap();
    let first = [0x10, 0x20, 0x30, 0x40].repeat(4);
    let second = [0x50, 0x60, 0x70, 0x80].repeat(4);
    let path = written(&dir, |w| {
        let library = w.blob(&library(&dir, &[("invert_texels", 2, 0)]))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call(
            "newLibraryWithData:error:",
            1,
            3,
            &[Value::Blob(library)],
            &[],
        )?;
        let kernel = [Value::Str("invert_texels".into())];
        w.call("newFunctionWithName:", 3, 4, &kernel, &[])?;
        let made = "newComputePipelineStateWithFunction:error:";
        w.call(made, 1, 5, &[Value::Object(4)], &[])?;
        let pages = [Value::Uint(16), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 6, &pages, &[])?;
        let texture = |usage| {
            uints(&[
                ("textureType", 2),
                ("pixelFormat", 70),
                ("width", 2),
                ("height", 2),
                ("usage", usage),
                ("storageMode", 0),
            ])
        };
        let plane = Value::Map(vec![
            ("plane".into(), Value::Uint(0)),
            ("offset".into(), Value::Uint(0)),
            ("bytesPerRow".into(), Value::Uint(8)),
            ("planar".into(), Value::Bool(false)),
            ("facts".into(), Value::Nil),
        ]);
        let surface = "newTextureWithDescriptor:iosurface:plane:";
        w.call(surface, 6, 8, &[texture(1), plane], &[])?;
        for picture in [&first, &second] {
            let bytes = [Value::Uint(0), Value::Bytes(picture.clone())];
            w.call("m2v.write", 6, 0, &bytes, &[])?;
            w.call("m2v.externalWriteDone", 6, 0, &[], &[])?;
        }
        w.call("newTextureWithDescriptor:", 1, 10, &[texture(2)], &[])?;
        invert(w, 11, 8, 10)
    });
    let (report, lines, rows) = play_then("surface-lazy", &path, |backend| {
        let bytes = backend.read(10).unwrap();
        let pitch = bytes.len() / 2;
        [&bytes[..8], &bytes[pitch..pitch + 8]].concat()
    });
    let inverse: Vec<u8> = (second.chunks(4))
        .flat_map(|t| [0xFF - t[0], 0xFF - t[1], 0xFF - t[2], t[3]])
        .collect();
    assert_eq!(rows, inverse, "{lines:?}");
    let executed = &report.total.executed;
    let counted = |key: &str| executed.get(key).copied().unwrap_or(0);
    let counts = [
        "surface_external_writes",
        "surface_refills",
        "surface_pictures_filled",
        "surface_pictures_unfilled",
    ]
    .map(counted);
    assert_eq!(counts, [2, 1, 1, 1], "{executed:?}");
}

/// Textures over plane 0 of one surface (#2), 2 by 2, none declaring shader write: #3,
/// RGBA8Unorm, makes a plain shape, and #4, its sRGB sibling, is a view of it; then `last`, a
/// texture of BGRA8Unorm over the plane or a view of #3 as BGRA8Unorm, is refused and counted:
/// a plain shape views as its own format and that format's sRGB sibling alone.
#[test]
fn a_plain_surface_shape_views_as_its_format_and_its_srgb_sibling_alone() {
    let dir = ScratchDir::new("rt-replay-plain-shape").unwrap();
    let texture = |format, usage| {
        uints(&[
            ("textureType", 2),
            ("pixelFormat", format),
            ("width", 2),
            ("height", 2),
            ("usage", usage),
            ("storageMode", 0),
        ])
    };
    let plane = Value::Map(vec![
        ("plane".into(), Value::Uint(0)),
        ("offset".into(), Value::Uint(0)),
        ("bytesPerRow".into(), Value::Uint(8)),
        ("planar".into(), Value::Bool(false)),
        ("facts".into(), Value::Nil),
    ]);
    let surface = "newTextureWithDescriptor:iosurface:plane:";
    let stream = |view: bool| {
        written(&dir, |w| {
            w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
            let pages = [Value::Uint(16), Value::Uint(0)];
            w.call("newBufferWithLength:options:", 1, 2, &pages, &[])?;
            w.call(surface, 2, 3, &[texture(70, 1), plane.clone()], &[])?;
            w.call(surface, 2, 4, &[texture(71, 5), plane.clone()], &[])?;
            match view {
                true => w.call(
                    "newTextureViewWithPixelFormat:",
                    3,
                    5,
                    &[Value::Uint(80)],
                    &[],
                ),
                false => w.call(surface, 2, 5, &[texture(80, 1), plane.clone()], &[]),
            }
        })
    };
    for view in [false, true] {
        let cache = ScratchCache::new().unwrap();
        let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
        let mut reader = StreamReader::open(&[stream(view)], None).unwrap();
        let (report, ended) = Player::new(&mut backend).play(&mut reader);
        match ended {
            Err(PlayError::Refused { why, .. }) => {
                assert!(why.starts_with("a plain surface shape of "), "{why}");
            }
            other => panic!("a plain shape viewed as another format (view {view}): {other:?}"),
        }
        let executed = &report.total.executed;
        let counted = |key: &str| executed.get(key).copied().unwrap_or(0);
        let counts = [
            "surface_shapes",
            "surface_shapes_found",
            "surface_format_refusals",
        ]
        .map(counted);
        assert_eq!(counts, [1, 1, 1], "{executed:?}");
    }
}

/// A stream of a placement at an index a held commit owes: command buffer #6 of queue #2 waits for
/// event #4 to reach 1, binds sampler state #5 (the default descriptor, index 16) and is
/// committed; #5 is released and sampler state #8 made at index 16 with `descriptor`; only then
/// does command buffer #10 of queue #3 signal #4 to 1, and the CPU waits for #6. Where
/// `displaced`, sampler state #11 (a linear min filter) takes index 16 from #5 before #6 names
/// #5, and lives on: two live sampler states placed at 16, #5 released after #11 took it.
fn held_placement(dir: &ScratchDir, descriptor: Value, displaced: bool) -> PathBuf {
    written(dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("newCommandQueue", 1, 3, &[], &[])?;
        w.call("newEvent", 1, 4, &[], &[])?;
        let made = "newSamplerStateWithDescriptor:";
        let id = [("gpuResourceID", Value::Uint(16))];
        w.call(made, 1, 5, &[sampler(None)], &[])?;
        w.call("gpuResourceID", 5, 0, &[], &id)?;
        if displaced {
            w.call(made, 1, 11, &[uints(&[("minFilter", 1)])], &[])?;
            w.call("gpuResourceID", 11, 0, &[], &id)?;
        }
        let event = |value| [Value::Object(4), Value::Uint(value)];
        w.call("commandBuffer", 2, 6, &[], &[])?;
        w.call("encodeWaitForEvent:value:", 6, 0, &event(1), &[])?;
        w.call("computeCommandEncoder", 6, 7, &[], &[])?;
        let bound = [
            Value::List(vec![Value::Object(5)]),
            uints(&[("location", 0), ("length", 1)]),
        ];
        w.call("setSamplerStates:withRange:", 7, 0, &bound, &[])?;
        w.call("endEncoding", 7, 0, &[], &[])?;
        w.call("commit", 6, 0, &[], &[])?;
        w.call("release", 5, 0, &[], &[])?;
        w.call(made, 1, 8, &[descriptor], &[])?;
        w.call("gpuResourceID", 8, 0, &[], &id)?;
        w.call("commandBuffer", 3, 10, &[], &[])?;
        w.call("encodeSignalEvent:value:", 10, 0, &event(1), &[])?;
        w.call("commit", 10, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 6, 0, &[], &[])
    })
}

#[test]
fn an_equal_sampler_state_takes_an_index_a_held_commit_owes_without_waiting() {
    let dir = ScratchDir::new("rt-replay-equal-sampler").unwrap();
    let path = held_placement(&dir, sampler(None), false);
    let (_, lines) = play("equal-sampler", &path);
    let indices = lines.iter().find(|line| line.starts_with("indices named "));
    let indices = indices.unwrap_or_else(|| panic!("an indices line: {lines:?}"));
    assert!(
        indices.contains("0 placed at an index whose holders' commits were in flight")
            && indices.ends_with(
                "1 sampler states placed without waiting at an index whose last sampler state \
                 had the same descriptor while its commits were in flight"
            ),
        "{indices}"
    );
}

#[test]
fn a_placement_a_held_commit_owes_takes_the_index_without_waiting_and_is_counted() {
    let dir = ScratchDir::new("rt-replay-held-placement").unwrap();
    // A sampler state of another descriptor at index 16: #6, which reads 16, waits for a signal
    // only a later record makes, so the recording device gave 16 away before #6 completed.
    let other = uints(&[("minFilter", 1), ("magFilter", 1)]);
    let path = held_placement(&dir, other, false);
    let (_, lines, indices) = played("held-placement", &path, &[8]);
    assert_eq!(indices, [16]);
    let unwaited = "indices given while a commit they owe is held 1: placed without waiting at an \
                    index owed to a command buffer no record played so far lets complete, as the \
                    recording device's index policy gave the index to a new object before that \
                    command buffer completed";
    assert!(lines.iter().any(|line| line == unwaited), "{lines:?}");
    let indices = lines.iter().find(|line| line.starts_with("indices named "));
    let indices = indices.unwrap_or_else(|| panic!("an indices line: {lines:?}"));
    assert!(
        indices.contains("0 placed at an index whose holders' commits were in flight"),
        "{indices}"
    );
}

/// A play that ends with a command buffer held on an event no record signals still ends its
/// device: once it dropped, every gauge reads zero, and each goes out as the matrix's line,
/// `M2V-MATRIX gauge KEY@drop 0`.
#[test]
fn a_play_that_ends_with_a_command_buffer_held_ends_its_device() {
    let dir = ScratchDir::new("rt-replay-end-held").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("newEvent", 1, 3, &[], &[])?;
        w.call(
            "newSamplerStateWithDescriptor:",
            1,
            4,
            &[sampler(None)],
            &[],
        )?;
        w.call("commandBuffer", 2, 5, &[], &[])?;
        let event = [Value::Object(3), Value::Uint(1)];
        w.call("encodeWaitForEvent:value:", 5, 0, &event, &[])?;
        w.call("computeCommandEncoder", 5, 6, &[], &[])?;
        let bound = [
            Value::List(vec![Value::Object(4)]),
            uints(&[("location", 0), ("length", 1)]),
        ];
        w.call("setSamplerStates:withRange:", 6, 0, &bound, &[])?;
        w.call("endEncoding", 6, 0, &[], &[])?;
        w.call("commit", 5, 0, &[], &[])?;
        w.call("release", 4, 0, &[], &[])
    });
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let mut reader = StreamReader::open(&[path], None).unwrap();
    let (_, ended) = Player::new(&mut backend).play(&mut reader);
    let ended = format!("{ended:?}");
    assert!(
        ended.contains("still wait on events at the stream's end"),
        "{ended}"
    );
    let ended = backend.end();
    ended.result().unwrap();
    for line in ended.lines() {
        let fields: Vec<&str> = line.split(' ').collect();
        assert!(
            matches!(fields[..], ["M2V-MATRIX", "gauge", key, "0"] if key.ends_with("@drop")),
            "{line}"
        );
    }
}

#[test]
fn a_command_buffer_naming_an_object_after_a_later_one_took_its_index_owes_it_nothing() {
    let dir = ScratchDir::new("rt-replay-displaced").unwrap();
    // #6 names #5 after #11 took index 16 from it: 16 is #11's, so #8 owes #6 nothing there.
    let other = uints(&[("minFilter", 1), ("magFilter", 1)]);
    let path = held_placement(&dir, other, true);
    let (_, lines, indices) = played("displaced", &path, &[8]);
    assert_eq!(indices, [16]);
    let unwaited = "indices given while a commit they owe is held 0:";
    assert!(
        lines.iter().any(|line| line.starts_with(unwaited)),
        "{lines:?}"
    );
    let indices = lines.iter().find(|line| line.starts_with("indices named "));
    let indices = indices.unwrap_or_else(|| panic!("an indices line: {lines:?}"));
    assert!(
        indices.contains("0 placed at an index whose holders' commits were in flight"),
        "{indices}"
    );
}

#[test]
fn a_linear_texture_over_a_buffer_plays_where_2d_and_is_refused_as_an_array() {
    let dir = ScratchDir::new("rt-replay-linear").unwrap();
    // An RGBA8Unorm 4 by 4 texture of `fields`, shader-read, over Shared buffer #2's first 1 KiB,
    // 256 bytes a row.
    let linear = |fields: &[(&str, u64)]| {
        written(&dir, |w| {
            w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
            let made = [Value::Uint(4096), Value::Uint(0)];
            w.call("newBufferWithLength:options:", 1, 2, &made, &[])?;
            let mut keys = vec![
                ("pixelFormat", 70),
                ("width", 4),
                ("height", 4),
                ("usage", 1),
            ];
            keys.extend_from_slice(fields);
            w.call("m2v.textureDescriptor", 0, 3, &[uints(&keys)], &[])?;
            let at = [
                Value::Object(3),
                Value::Uint(0),
                Value::Uint(256),
                Value::Uint(0),
            ];
            let selector = "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:";
            w.call(selector, 2, 4, &at, &[])
        })
    };
    let (report, _) = play("linear", &linear(&[("textureType", 2)]));
    let executed = &report.total.executed;
    assert_eq!(executed["skipped_objects"], 0, "{executed:?}");
    let array = linear(&[("textureType", 3), ("arrayLength", 2)]);
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let mut reader = StreamReader::open(&[array], None).unwrap();
    let (_, ended) = Player::new(&mut backend).play(&mut reader);
    match ended {
        Err(PlayError::Refused { why, .. }) => {
            assert_eq!(why, "a linear texture of a type other than 2D");
        }
        other => panic!("a linear 2D array's refusal: {other:?}"),
    }
}

/// A bytes-form no-copy buffer is an import: a texture over it is the runtime's counted copy,
/// which a blit of the texture reads the stream's bytes through, and the imports end with the
/// device. A descriptor-form one is a pool buffer with no import: recorded inside a block's
/// recorded range, a buffer of that block, which reads the bytes written into the block there;
/// outside every block, a place of its own, a texture over which reads the stream's bytes and is
/// no copy on the imported side. The gap line counts, of those outside, the pinned ones with no
/// `zeroFrom`.
#[test]
fn a_bytes_form_no_copy_buffer_is_an_import_and_a_descriptor_form_one_a_pool_buffer() {
    let dir = ScratchDir::new("rt-replay-no-copy").unwrap();
    // An RGBA8Unorm 4 by 4 texture's rows, 16 bytes each, 256 bytes apart in the buffer.
    let texels: Vec<u8> = (0..64).collect();
    let mut pages = vec![0u8; 1024];
    for (row, bytes) in texels.chunks(16).enumerate() {
        pages[row * 256..][..16].copy_from_slice(bytes);
    }
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        // #3 Shared, no-copy, its bytes as the stream gives them: in writes.
        let no_copy = [Value::Nil, Value::Uint(1024), Value::Uint(0), Value::Nil];
        let selector = "newBufferWithBytesNoCopy:length:options:deallocator:";
        w.call(selector, 1, 3, &no_copy, &[])?;
        w.call(
            "m2v.write",
            3,
            0,
            &[Value::Uint(0), Value::Bytes(pages.clone())],
            &[],
        )?;
        // Descriptor-form no-copy buffers outside every block, 4096 bytes each: #4 pinned with no
        // zeroFrom (one that may be the program's own memory), its bytes as #3's; #12 with no
        // pinned address and #15 pinned with zeroFrom (the host's own).
        let outside = |pinned: Option<u64>, zero_from: Option<u64>| {
            let fields = [
                ("length", Value::Uint(4096)),
                ("options", Value::List(Vec::new())),
                (
                    "pinnedGpuAddress",
                    pinned.map_or(Value::Nil, Value::Address),
                ),
                ("contents", Value::Nil),
                ("noCopy", Value::Bool(true)),
                ("zeroFrom", zero_from.map_or(Value::Nil, Value::Uint)),
            ];
            Value::Map(Vec::from(fields.map(|(key, value)| (key.into(), value))))
        };
        const ELSEWHERE: u64 = 0x20_0000_0000;
        w.call(
            "newBufferWithDescriptor:",
            1,
            4,
            &[outside(Some(ELSEWHERE), None)],
            &[],
        )?;
        w.call(
            "m2v.write",
            4,
            0,
            &[Value::Uint(0), Value::Bytes(pages)],
            &[],
        )?;
        w.call(
            "newBufferWithDescriptor:",
            1,
            12,
            &[outside(None, None)],
            &[],
        )?;
        let zeroed = outside(Some(ELSEWHERE + 8192), Some(1024));
        w.call("newBufferWithDescriptor:", 1, 15, &[zeroed], &[])?;
        // #10 a Shared block recorded at BASE, and #11 a descriptor-form no-copy buffer recorded
        // 8 KiB into it, whose first bytes a write into the block sets.
        const BASE: u64 = 0x10_0000_0000;
        let block = [Value::Uint(65536), Value::Uint(0)];
        let at = |address, length| {
            [
                ("gpuAddress", Value::Address(address)),
                ("gpuLength", Value::Uint(length)),
            ]
        };
        let selector = "m2v.newBlockWithLength:storageMode:";
        w.call(selector, 1, 10, &block, &at(BASE, 65536))?;
        let fields = [
            ("length", Value::Uint(4096)),
            ("options", Value::List(Vec::new())),
            ("pinnedGpuAddress", Value::Address(BASE + 8192)),
            ("contents", Value::Nil),
            ("noCopy", Value::Bool(true)),
        ];
        let inside = Value::Map(Vec::from(fields.map(|(key, value)| (key.into(), value))));
        let pinned = at(BASE + 8192, 4096);
        w.call("newBufferWithDescriptor:", 1, 11, &[inside], &pinned)?;
        let set = [Value::Uint(8192), Value::Bytes(vec![1, 2, 3, 4])];
        w.call("m2v.write", 10, 0, &set, &[])?;
        let keys = [
            ("pixelFormat", 70),
            ("width", 4),
            ("height", 4),
            ("usage", 1),
        ];
        w.call("m2v.textureDescriptor", 0, 5, &[uints(&keys)], &[])?;
        let over = [Value::Object(5), Value::Uint(0), Value::Uint(256)];
        w.call(
            "newTextureWithDescriptor:offset:bytesPerRow:",
            3,
            6,
            &over,
            &[],
        )?;
        w.call(
            "newTextureWithDescriptor:offset:bytesPerRow:",
            4,
            13,
            &over,
            &[],
        )?;
        let made = [Value::Uint(64), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 7, &made, &[])?;
        w.call("newBufferWithLength:options:", 1, 14, &made, &[])?;
        // The textures, packed into #7 and #14.
        w.call("commandBuffer", 2, 8, &[], &[])?;
        w.call("blitCommandEncoder", 8, 9, &[], &[])?;
        let selector = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:\
                        toBuffer:destinationOffset:destinationBytesPerRow:\
                        destinationBytesPerImage:";
        for (texture, buffer) in [(6, 7), (13, 14)] {
            let copy = [
                Value::Object(texture),
                Value::Uint(0),
                Value::Uint(0),
                uints(&[("x", 0), ("y", 0), ("z", 0)]),
                uints(&[("width", 4), ("height", 4), ("depth", 1)]),
                Value::Object(buffer),
                Value::Uint(0),
                Value::Uint(16),
                Value::Uint(64),
            ];
            w.call(selector, 9, 0, &copy, &[])?;
        }
        w.call("endEncoding", 9, 0, &[], &[])?;
        w.call("commit", 8, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 8, 0, &[], &[])?;
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(&texels).to_vec())]);
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(7)],
            &[("sha256", sum.clone())],
        )?;
        w.call("m2v.check", 0, 0, &[Value::Object(14)], &[("sha256", sum)])?;
        let mut held = vec![0u8; 4096];
        held[..4].copy_from_slice(&[1, 2, 3, 4]);
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(&held).to_vec())]);
        w.call("m2v.check", 0, 0, &[Value::Object(11)], &[("sha256", sum)])?;
        w.call("release", 6, 0, &[], &[])?;
        w.call("release", 3, 0, &[], &[])
    });
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let mut reader = StreamReader::open(&[path], None).unwrap();
    let (report, ended) = Player::new(&mut backend).play(&mut reader);
    let lines = backend.lines();
    for line in report.lines().iter().chain(&lines) {
        eprintln!("REPLAY no-copy {line}");
    }
    ended.unwrap();
    assert_eq!(report.checks.len(), 3, "{lines:?}");
    for check in &report.checks {
        let object = check.object;
        assert_eq!(Some(check.sha256), check.recorded, "#{object}");
    }
    // #3 imported; #11 a buffer of block #10; #4, #12 and #15 places of their own, #4 alone
    // within the gap's bound.
    let counted = "no-copy buffers: descriptor-form 1 in a recorded block (buffers of it) and 3 \
                   outside every block (places of their own, 12288 bytes in all), pool buffers \
                   with no import; bytes-form 1 (the host's own objects, each imported, ";
    assert!(
        lines.iter().any(|line| line.starts_with(counted)),
        "{lines:?}"
    );
    let gap = "gap: descriptor-form buffers that may be the app's own no-copy memory are played \
               as pool buffers, count unknown: at most 1, those outside every block with a \
               pinnedGpuAddress and no zeroFrom (4096 bytes in all)";
    assert!(lines.iter().any(|line| line == gap), "{lines:?}");
    // The texture over the import is held in an image of its own; the one over #4 is in place
    // where the device's linear image takes it, and else the runtime's copy of a pool buffer's.
    let over = "textures over buffers 2: ";
    assert!(lines.iter().any(|line| line.starts_with(over)), "{lines:?}");
    // The import's texture's bytes into its image of its own, once, on the imported side; #4's,
    // if copied, on the pool side.
    let executed = &report.total.executed;
    assert_eq!(executed["over_copies_imported"], 1, "{executed:?}");
    assert_eq!(executed["over_copy_bytes_imported"], 64, "{executed:?}");
    let ended = backend.end();
    ended.result().unwrap();
    assert!(ended.gauges.contains(&("imports", 0)), "{:?}", ended.gauges);
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

#[test]
fn a_buffer_placed_in_a_block_taken_again_reads_zero() {
    let dir = ScratchDir::new("rt-replay-zero-once").unwrap();
    let length = 64 * 1024;
    let piece = 4096u64;
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let block = [Value::Uint(length), Value::Uint(0)];
        let placed = "m2v.newBufferInBlock:offset:length:options:";
        // A Shared block, a buffer in it written whole, both released.
        w.call("m2v.newBlockWithLength:storageMode:", 1, 2, &block, &[])?;
        let at = |block| {
            [
                Value::Object(block),
                Value::Uint(0),
                Value::Uint(piece),
                Value::Uint(0),
            ]
        };
        w.call(placed, 2, 3, &at(2), &[])?;
        let bytes = Value::Bytes(vec![0xab; piece as usize]);
        w.call("m2v.write", 3, 0, &[Value::Uint(0), bytes], &[])?;
        w.call("release", 3, 0, &[], &[])?;
        w.call("release", 2, 0, &[], &[])?;
        // A block of the same length, which the runtime may take from the same bytes, and a
        // buffer in it that nothing writes.
        w.call("m2v.newBlockWithLength:storageMode:", 1, 4, &block, &[])?;
        w.call(placed, 4, 5, &at(4), &[])?;
        let zeros = vec![0u8; piece as usize];
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(&zeros).to_vec())]);
        w.call("m2v.check", 0, 0, &[Value::Object(5)], &[("sha256", sum)])
    });
    let (report, _) = play("zero-once", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    // Each block reads zero at its making, as the player's buffers zeroed it.
    let zeroed = report.total.executed["bytes_zeroed"];
    assert!(
        zeroed >= 2 * length,
        "each block zeroed at its making: {zeroed}"
    );
}

#[test]
fn buffers_keep_the_slack_of_their_recorded_granule() {
    let dir = ScratchDir::new("rt-replay-granule").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let made = "newBufferWithLength:options:";
        let at = |address, length| {
            [
                ("gpuAddress", Value::Address(address)),
                ("gpuLength", Value::Uint(length)),
            ]
        };
        // #2 and #3, 4 bytes each, recorded 32 KiB apart: granules of 32 KiB and 64 KiB.
        for (id, address) in [(2, 0x48000), (3, 0x50000)] {
            w.call(
                made,
                1,
                id,
                &[Value::Uint(4), Value::Uint(0)],
                &at(address, 4),
            )?;
        }
        Ok(())
    });
    let (_, lines) = play("granule", &path);
    let places = lines
        .iter()
        .find(|line| line.starts_with("places by recorded granule: "));
    let places = places.unwrap_or_else(|| panic!("a places line: {lines:?}"));
    // Each place spans 4 bytes rounded up to its granule: 32,764 and 65,532 bytes of slack.
    let slack = "32768 B 1, 65536 B 1; most slack held 98296 bytes";
    assert!(places.contains(slack), "{places}");
}

/// A Shared block at a recorded address, a buffer in it, and a write into the buffer of a word
/// whose value lies in the buffer's recorded range (two 32-bit indices, as an index array holds,
/// can make one) and one 4 past a multiple of 8 that lies in the block's; then a check of the
/// buffer's bytes as written.
fn census_stream(dir: &ScratchDir) -> PathBuf {
    let block = 1u64 << 46;
    let mut bytes = (block + 0x108).to_le_bytes().to_vec();
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&(block + 0x10).to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    written(dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let at = |address: u64, length: u64| {
            [
                ("gpuAddress", Value::Address(address)),
                ("gpuLength", Value::Uint(length)),
            ]
        };
        let made = [Value::Uint(0x1_0000), Value::Uint(0)];
        let selector = "m2v.newBlockWithLength:storageMode:";
        w.call(selector, 1, 2, &made, &at(block, 0x1_0000))?;
        let inner = [
            Value::Object(2),
            Value::Uint(0x100),
            Value::Uint(0x100),
            Value::Uint(0),
        ];
        let selector = "m2v.newBufferInBlock:offset:length:options:";
        w.call(selector, 2, 3, &inner, &at(block + 0x100, 0x100))?;
        let write = [Value::Uint(0), Value::Bytes(bytes.clone())];
        w.call("m2v.write", 3, 0, &write, &[])?;
        let mut whole = bytes.clone();
        whole.resize(0x100, 0);
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(&whole).to_vec())]);
        w.call("m2v.check", 0, 0, &[Value::Object(3)], &[("sha256", sum)])
    })
}

/// The census's counts of a play: the address-looking words, those 4 past a multiple of 8, and
/// the address arguments moved.
fn census(report: &Report) -> [u64; 3] {
    let stream = &report.total.stream;
    [
        "address_looking_words",
        "misaligned_address_words",
        "relocated_words",
    ]
    .map(|key| stream[key])
}

/// On the new runtime, which makes the block elsewhere, the word that looks like an address
/// plays as written: the buffer reads back the recorded bytes, the word is counted, and nothing
/// moved.
#[test]
fn a_data_word_in_a_live_range_plays_as_written_on_the_new_runtime() {
    let dir = ScratchDir::new("rt-replay-data-word").unwrap();
    let (report, _) = play("data-word", &census_stream(&dir));
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    assert_eq!(census(&report), [1, 1, 0]);
}

/// The same stream on the old runtime, which makes the block at its recorded address, and on the
/// new runtime, which does not: the census counts the same words on both, and both play them as
/// written.
#[test]
fn the_address_census_is_the_same_on_both_runtimes() {
    let dir = ScratchDir::new("rt-replay-census").unwrap();
    let path = census_stream(&dir);
    let (rt, _) = play("census", &path);
    let mut old = None;
    on_device_with(at_plan, |device| {
        let mut backend = OldRuntime::new(device.clone(), gpu::TEST_DEVICE);
        let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
        let (report, ended) = Player::new(&mut backend).play(&mut reader);
        for line in report.lines() {
            eprintln!("REPLAY census-old {line}");
        }
        ended.unwrap();
        old = Some(report);
    });
    let old = old.expect("an old runtime's device");
    for report in [&rt, &old] {
        assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    }
    assert_eq!(census(&rt), census(&old));
    assert_eq!(census(&old), [1, 1, 0]);
    assert_eq!(old.pinned, Some(2), "the block and its buffer");
}

/// A copy back to the guest's RAM plays as the runtime's buffer copies into a sink of the play's
/// own, each run widened to the whole words it touches within the buffer: nothing is skipped, the
/// bytes it copied are counted, and the buffer keeps its bytes.
#[test]
fn a_copy_back_to_the_guest_s_ram_plays_as_buffer_copies() {
    let dir = ScratchDir::new("rt-replay-copy-back").unwrap();
    let bytes = vec![0x5a_u8; 100];
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let made = [Value::Uint(100), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 3, &made, &[])?;
        let write = [Value::Uint(0), Value::Bytes(bytes.clone())];
        w.call("m2v.write", 3, 0, &write, &[])?;
        w.call("commandBuffer", 2, 4, &[], &[])?;
        w.call("blitCommandEncoder", 4, 5, &[], &[])?;
        let run = |at: u64, start: u64, end: u64| {
            let host = Value::List(vec![Value::Uint(start), Value::Uint(end)]);
            Value::List(vec![Value::Uint(at), host])
        };
        // Bytes 1..17, and 90..100, the buffer's last.
        let runs = Value::List(vec![run(1, 0x1000, 0x1010), run(90, 0x2000, 0x200a)]);
        w.call("m2v.copyToGuestRam", 5, 0, &[Value::Object(3), runs], &[])?;
        w.call("endEncoding", 5, 0, &[], &[])?;
        w.call("commit", 4, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 4, 0, &[], &[])?;
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(&bytes).to_vec())]);
        w.call("m2v.check", 0, 0, &[Value::Object(3)], &[("sha256", sum)])
    });
    let (report, _) = play("copy-back", &path);
    assert_eq!(report.first_skip, None);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    // 1..17 is the words 0..20, and 90..100 the words 88..100.
    let copied = report.total.executed["copied_back_guest_ram_bytes"];
    assert_eq!(copied, 20 + 12);
}
