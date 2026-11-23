//! Device tests of the replay player on the runtime (`replay::NewRuntime`), on the device the
//! cfg selects: streams the tests write (`docs/REPLAY-STREAM.md`), each played and its report
//! printed as `REPLAY <test> <line>`.
//!
//! - A draw of a render pipeline state, its vertices in buffers bound to the vertex stage, covers
//!   a pass's target that its setter records clear to red: a blit copies the target to a buffer,
//!   which the stream checks green, and nothing is skipped.
//! - A draw of points after it plays, nothing skipped.
//! - A render encoder's residency declarations play through the runtime: after each of its eight
//!   `useResource`, `useResources`, `useHeap` and `useHeaps` forms, with stages and without, of
//!   buffers, a texture and a heap, with read, read and sample, and read and write usage, a draw
//!   plays, nothing refused or skipped. A skipped texture they name is a skipped call, alone or in
//!   a list, and binds nothing: the draw after them plays.
//! - `setLineWidth:` plays: at width 1 a draw of lines after it plays; at width 2 a draw of
//!   triangles plays and one of lines is the runtime's gap, skipped and counted, and the report
//!   says the play is no verdict past it: its gap, frame and call.
//! - A pass's visibility results play as the runtime's: in a `Reset` pass, `Counting` and
//!   `Boolean` modes before draws of the whole target write its 16 samples into their words, and
//!   a word no mode took keeps its bytes, nothing skipped; the backend's gauges read the one
//!   occlusion pool at the frame's end and none at the device's drop. A mode at an offset the
//!   encoder took before is the runtime's refusal, Metal's own, skipped and counted by its kind;
//!   in an `Accumulate` pass both modes are the runtime's gap, skipped and counted, and the buffer
//!   keeps its bytes.
//! - A compute encoder's residency declarations play through the runtime: after
//!   `useResource:usage:`, `useResources:count:usage:`, `useHeap:` and `useHeaps:count:`, a
//!   dispatch plays, nothing refused or skipped.
//! - A compute pipeline state the runtime answers with a typed gap is skipped and counted by the
//!   gap, and a dispatch of it is skipped, and the play goes on.
//! - Each pipeline state's making is split by the runtime's counters into the translator's time,
//!   the driver's and the rest, beside what it made: a kernel's first state and a render state
//!   are of contents new to the device, which lowered their functions and made a `VkPipeline`,
//!   with both halves' times, and a second state of the kernel is held, which made nothing and
//!   took no driver time; the slowest are listed by object, with their stages' kinds.
//! - A compute encoder's `setStageInRegion:` plays: the runtime takes it.
//! - The host's library from source is made of its display library where the backend has it, a
//!   function of it with it; without it both are skipped.
//! - A sampler bound with LOD clamps whose samples read as its own plays with them (its own
//!   clamps, and (0, 1000) over its (0, FLT_MAX)); one bound with clamps that read otherwise is
//!   the runtime's typed gap, skipped and counted, and the draw after it is skipped.
//! - The range forms of buffer and texture binds bind from the range's location, each slot as its
//!   single-slot form does: vertex buffers at their offsets draw; fragment buffers at location 1
//!   fill slots 1 and 2 (an offset of slot 2 plays, one of slot 0 is refused); a skipped texture
//!   bound to the vertex stage at location 1 skips the draws until a nil at location 1 takes it
//!   off; a fragment function reads the texture at the slot its range names; a kernel reads and
//!   writes the textures a compute encoder's ranges bind.
//! - A Managed buffer of a stream recorded with two copies keeps both on any device (the player's
//!   own second copy where the device has one): a blit's fill reaches the GPU's copy and not the
//!   CPU's, an announcement with no write puts the CPU's bytes back into the GPU's copy at the next
//!   commit, and a synchronize brings the GPU's bytes into the CPU's copy.
//! - A check of a two-copy stream's Managed buffer names the copy it read: its CPU copy, which a
//!   blit's fill does not reach, or where the play reads device copies, its device copy, read
//!   back, which holds the fill; a Shared buffer's check names none. On a device that keeps a
//!   device copy, the backend's header line says which copy the play reads.
//! - A two-copy stream's Managed buffers start their GPU copies from the bytes they were made with:
//!   a no-copy descriptor buffer and a copied one, each made with 7s and written 7s before its
//!   first use with no `didModifyRange:`, read as 7s through a blit.
//! - A Private buffer, which has no CPU mapping, reads back through a copy into a Shared buffer of
//!   the play's own: a blit's fill of it reads as the fill's bytes.
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
//!   only a later record signals) takes it without waiting where its descriptor is the one placed
//!   there last, as Metal hands an equal descriptor the same ID. Otherwise it is kept back: it
//!   takes the highest free sampler index, counted on a line of its own, whether the held command
//!   buffer named the released state or, committed before the release, named nothing.
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
//!   indices can) plays as written, and is counted.
//! - A tiled texture over a buffer of two levels (the SPI
//!   `newTiledTextureWithDescriptor:offset:bytesPerRow:`) takes the buffer's bytes into its first
//!   level and makes its second from them at its first use: blits of both levels read the
//!   expected texels, and the take is counted once.
//! - A states-only play writes each pipeline state's make as it happens (`with_states`): on two
//!   threads of the backend's own, each of three render states' makes writes its begin line, then
//!   its modules line, then its end line, made, and each begin's key is the name of the record
//!   its content wrote under `renders/`; with one of those keys listed, only that state is made,
//!   on the play's thread, and the other two are passed over, each on a pass line. With the
//!   digests on, each modules line ends in its state's content digest: three contents, three
//!   digests, which a second play on a device of its own, on two threads, writes again; without
//!   them, no modules line carries one.
//!
//! The runtime's device is opened behind the hang tripwire, as every device open in this crate
//! is (`gpu.rs`), its kernels kept in a cache the test owns ([`ScratchCache`]).

use crate::gpu::{self, DeviceType};
use crate::replay::{
    Backend, NewRuntime, PlayError, Player, ReadCopy, Report, StreamReader, StreamWriter, Value,
    WAIT_COMPLETED,
};
use crate::ScratchDir;
use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

/// A pipeline cache directory one test owns: a new, empty directory in the project's scratch
/// root, removed when dropped, and by the next run of any suite where a killed test left it
/// ([`metal2vulkan_devtools::scratch`]).
pub(in crate::replay) struct ScratchCache(metal2vulkan_devtools::scratch::Owned);

impl ScratchCache {
    /// A new directory, not yet made (the device makes it).
    pub(in crate::replay) fn new() -> std::io::Result<ScratchCache> {
        metal2vulkan_devtools::scratch::Owned::reserved("rt-replay-cache").map(ScratchCache)
    }

    /// The directory.
    pub(in crate::replay) fn path(&self) -> &Path {
        self.0.path()
    }
}

/// The runtime's first device of the cfg's kind (a GPU: discrete, then integrated, then
/// virtual; else the CPU), its kernels kept in `cache`.
pub(in crate::replay) fn rt_device(cache: &Path) -> metal2vulkan_runtime::Device {
    use ash::vk::PhysicalDeviceType as Kind;
    let kinds: &[Kind] = match gpu::TEST_DEVICE {
        DeviceType::Gpu => {
            gpu::tripwire::refuse_if_tripped().unwrap();
            &[Kind::DISCRETE_GPU, Kind::INTEGRATED_GPU, Kind::VIRTUAL_GPU]
        }
        DeviceType::Cpu => &[Kind::CPU],
    };
    let instance = metal2vulkan_runtime::Instance::new(&[]).unwrap();
    let (_, vk_instance) = instance.raw();
    let mut physical = instance.physical_devices().unwrap();
    // SAFETY: each is one of the instance's physical devices, and the instance is live.
    let kind_of = |p: &metal2vulkan_runtime::PhysicalDevice| unsafe {
        vk_instance
            .get_physical_device_properties(p.raw())
            .device_type
    };
    physical.retain(|p| kinds.contains(&kind_of(p)));
    physical.sort_by_key(|p| kinds.iter().position(|&kind| kind == kind_of(p)));
    let physical = physical.first().expect("a device of the cfg's kind");
    let mut options = metal2vulkan_runtime::DeviceOptions::default();
    options.pipeline_cache_dir = Some(cache.to_path_buf());
    metal2vulkan_runtime::Device::new(physical, options).unwrap()
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

/// `path` played on the runtime's device, its report and the backend's lines printed; the
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

/// A resource argument, `{Buffer: object}` or `{Texture: object}`.
fn resource(kind: &str, id: u64) -> Value {
    Value::Map(vec![(kind.into(), Value::Object(id))])
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
    /// A heap (#16, call 19) and a texture (#17, call 20), the eight residency declarations of a
    /// render encoder naming them and the vertex buffers (calls 21 to 28), and the triangle again
    /// (call 29).
    Uses,
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
            Then::Uses => {
                let block = [Value::Uint(0x1_0000), Value::Uint(0)];
                w.call("m2v.newBlockWithLength:storageMode:", 1, 16, &block, &[])?;
                // RGBA8Unorm, 4 by 4, shader-read, Private.
                let texture = uints(&[
                    ("textureType", 2),
                    ("pixelFormat", 70),
                    ("width", 4),
                    ("height", 4),
                    ("usage", 1),
                    ("storageMode", 2),
                ]);
                w.call("newTextureWithDescriptor:", 1, 17, &[texture], &[])?;
                // MTLResourceUsageRead, Read | Sample and Read | Write; MTLRenderStageVertex,
                // Fragment and both.
                let [read, sampled, written] = [1, 5, 3].map(Value::Uint);
                let [vertex, fragment, both] = [1, 2, 3].map(Value::Uint);
                let heap = Value::Object(16);
                let heaps = Value::List(vec![Value::Object(16)]);
                let list = |ids: [(&str, u64); 2]| {
                    Value::List(ids.map(|(kind, id)| resource(kind, id)).to_vec())
                };
                let uses = [
                    (
                        "useResource:usage:",
                        vec![resource("Buffer", 8), read.clone()],
                    ),
                    (
                        "useResource:usage:stages:",
                        vec![resource("Texture", 17), sampled, fragment],
                    ),
                    (
                        "useResources:count:usage:",
                        vec![list([("Buffer", 8), ("Buffer", 9)]), written],
                    ),
                    (
                        "useResources:count:usage:stages:",
                        vec![list([("Buffer", 9), ("Texture", 17)]), read, both.clone()],
                    ),
                    ("useHeap:", vec![heap.clone()]),
                    ("useHeap:stages:", vec![heap, vertex]),
                    ("useHeaps:count:", vec![heaps.clone()]),
                    ("useHeaps:count:stages:", vec![heaps, both]),
                ];
                for (selector, args) in uses {
                    w.call(selector, 13, 0, &args, &[])?;
                }
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

/// A play's report carries the runtime's stage sums among its executed counters, with the part
/// from scheduling to GPU start under earlier spans and the completion thread's steps, which add
/// up to the span from GPU end to handlers, and its stage histograms as each frame ended and at
/// the play's end: two frames of one fill each count one command buffer more in each histogram by
/// each frame's end.
#[test]
fn a_play_reports_the_runtime_s_command_buffer_stages() {
    let dir = ScratchDir::new("rt-replay-stages").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let made = [Value::Uint(4), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 3, &made, &[])?;
        let range = uints(&[("location", 0), ("length", 4)]);
        for (cb, value) in [(10, 7), (20, 9)] {
            w.call("commandBuffer", 2, cb, &[], &[])?;
            w.call("blitCommandEncoder", cb, cb + 1, &[], &[])?;
            let fill = [Value::Object(3), range.clone(), Value::Uint(value)];
            w.call("fillBuffer:range:value:", cb + 1, 0, &fill, &[])?;
            w.call("endEncoding", cb + 1, 0, &[], &[])?;
            w.call("commit", cb, 0, &[], &[])?;
            w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        }
        Ok(())
    });
    let (report, lines) = play("stages", &path);
    let executed = &report.total.executed;
    let sums = [
        "commit_to_scheduled_nanos",
        "scheduled_to_gpu_start_nanos",
        "gpu_run_nanos",
        "gpu_end_to_handler_nanos",
        "held_starts",
        "held_start_gap_nanos",
        "runtime_command_buffers",
        "runtime_command_buffer_gpu_nanos",
        "untimed_command_buffers",
        "scheduled_to_gpu_start_behind_nanos",
        "gpu_end_to_handler_tail",
        "calibrations",
        "calibration_nanos",
        "over_copies_pitch",
        "over_copy_bytes_image_memory",
    ];
    for key in sums {
        assert!(executed.contains_key(key), "{key}: {executed:?}");
    }
    // The completion thread's six steps add up to each span from a GPU end to its handlers.
    let steps = [
        "seen",
        "told",
        "stamp_read",
        "calibration",
        "earlier",
        "own",
    ];
    let sum = |infix: &str| -> u64 {
        (steps.iter())
            .map(|step| executed[format!("gpu_end_to_handler_{infix}{step}_nanos").as_str()])
            .sum()
    };
    assert_eq!(
        sum(""),
        executed["gpu_end_to_handler_nanos"],
        "{executed:?}"
    );
    assert!(sum("tail_") <= sum(""), "{executed:?}");
    assert!(
        executed["scheduled_to_gpu_start_behind_nanos"] <= executed["scheduled_to_gpu_start_nanos"],
        "{executed:?}"
    );
    let counted = |at: &str| -> Vec<u64> {
        let prefix = format!("stages {at} ");
        let line = lines.iter().find(|line| line.starts_with(&prefix));
        let line = line.unwrap_or_else(|| panic!("no stages {at} line: {lines:?}"));
        let histograms = line[prefix.len()..].split(' ').map(|named| {
            let (_, counts) = named.split_once('=').expect("a histogram");
            let counts: Vec<u64> = counts.split(',').map(|c| c.parse().unwrap()).collect();
            assert_eq!(counts.len(), 23, "{line}");
            counts.iter().sum::<u64>()
        });
        histograms.collect()
    };
    // Frame 0's line, then the play's end, which frame 1 ended.
    let (first, end) = (counted("0"), counted("end"));
    assert_eq!(first.len(), 4, "{lines:?}");
    for (at, (first, end)) in first.iter().zip(&end).enumerate() {
        assert!(*first >= 1, "{at}: {lines:?}");
        assert!(*end > *first, "{at}: {lines:?}");
    }
}

/// A draw of points after the triangle (call 19) plays, nothing skipped, the target still green.
#[test]
fn a_draw_of_points_plays_on_the_new_runtime() {
    let dir = ScratchDir::new("rt-replay-points").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let path = draw_into_a_pass(&dir, &green, Then::Points);
    let (report, _) = play("points", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    for skipped in ["skipped_render_commands", "skipped_draws", "skipped_calls"] {
        assert_eq!(executed[skipped], 0, "{skipped}: {executed:?}");
    }
    assert_eq!(report.first_skip, None);
}

/// A render encoder's eight residency declarations (calls 21 to 28) play through the runtime, as
/// the stream records them (each C array and its count one list, the usage after it, the stages
/// last): `useResource:usage:` and `useResources:count:usage:`, with stages and without, each
/// listed buffer or texture with the call's usage (read, read and sample, read and write) through
/// `use_resource`; `useHeap:` and `useHeaps:count:` through `use_heap`. The play ends, no call
/// refused, nothing skipped, and the triangle drawn after them (call 29) leaves the target green.
#[test]
fn a_render_encoder_s_residency_declarations_play_through_the_runtime() {
    let dir = ScratchDir::new("rt-replay-render-uses").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let path = draw_into_a_pass(&dir, &green, Then::Uses);
    let (report, _) = play("render-uses", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    for skipped in ["skipped_render_commands", "skipped_draws", "skipped_calls"] {
        assert_eq!(executed[skipped], 0, "{skipped}: {executed:?}");
    }
    assert_eq!(report.first_skip, None);
}

/// `setLineWidth:` plays: at width 1 the triangle and the line after it play, nothing skipped; at
/// width 2 the triangle draws and the line is the runtime's gap, skipped and counted, and the
/// report says the play is no verdict past it.
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
    let note = report.verdict_note();
    assert_eq!(note.as_deref(), Some("not a verdict past frame 0, call 21"));
    let line = "skipped render commands Draw { kind: LineWidth } 1, the first at frame 0, call 21";
    assert!(lines.iter().any(|l| l == line), "{lines:?}");
}

/// A word of a visibility result buffer that no mode wrote: its bytes as made.
const UNWRITTEN: u64 = 0xaaaa_aaaa_aaaa_aaaa;

/// A stream of a pass over a 4 by 4 target whose visibility result buffer (#10, a Shared one of
/// three words, made as `0xaa` bytes) is of `visibilityResultType` `kind` (0 `Reset`, 1
/// `Accumulate`): `Counting` at offset 0 (call 18) and a draw of the whole target (call 19), then
/// `Boolean` at offset 8 (call 20) and the draw again (call 21), and, with `revisit`, `Counting` at
/// offset 0 again (call 22), which Metal's validation stops; no mode takes offset 16. Once the
/// command buffer completes, the stream checks the buffer as `words`, then presents, which ends
/// the frame.
fn visibility_results(dir: &ScratchDir, kind: u64, revisit: bool, words: [u64; 3]) -> PathBuf {
    written(dir, |w| {
        let library = w.blob(&render_library(dir))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let library = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &library, &[])?;
        let vertex = [Value::Str("colored_vertex".into())];
        w.call("newFunctionWithName:", 3, 4, &vertex, &[])?;
        let fragment = [Value::Str("colored_fragment".into())];
        w.call("newFunctionWithName:", 3, 5, &fragment, &[])?;
        let pipeline = Value::Map(vec![
            ("vertex".into(), Value::Object(4)),
            ("fragment".into(), Value::Object(5)),
            ("color0PixelFormat".into(), Value::Uint(70)),
        ]);
        let made = "newRenderPipelineStateWithDescriptor:error:";
        w.call(made, 1, 6, &[pipeline], &[])?;
        // RGBA8Unorm, 4 by 4, render-target, Private.
        let target = uints(&[
            ("textureType", 2),
            ("pixelFormat", 70),
            ("width", 4),
            ("height", 4),
            ("usage", 4),
            ("storageMode", 2),
        ]);
        w.call("newTextureWithDescriptor:", 1, 7, &[target], &[])?;
        // One triangle over the whole target, and the visibility result buffer.
        let positions = floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
        let colors = floats(&[0.0, 1.0, 0.0, 1.0].repeat(3));
        let unwritten = Value::Bytes(vec![0xaa; 24]);
        for (id, bytes) in [(8, positions), (9, colors), (10, unwritten)] {
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
        w.call("MTLRenderPassDescriptor.new", 0, 11, &[], &[])?;
        let Value::Map(mut pass) = uints(&[
            ("renderTargetWidth", 4),
            ("renderTargetHeight", 4),
            ("visibilityResultType", kind),
        ]) else {
            unreachable!("a map");
        };
        pass.push(("visibilityResultBuffer".into(), Value::Object(10)));
        w.call("m2v.setPass", 11, 0, &[Value::Map(pass)], &[])?;
        let Value::Map(mut color) = uints(&[("level", 0), ("loadAction", 0), ("storeAction", 1)])
        else {
            unreachable!("a map");
        };
        color.push(("texture".into(), Value::Object(7)));
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
        // MTLVisibilityResultModeCounting at 0, then MTLVisibilityResultModeBoolean at 8, each
        // before a draw of the triangle (MTLPrimitiveTypeTriangle, from vertex 0, three).
        let mode = "setVisibilityResultMode:offset:";
        let draw = [Value::Uint(3), Value::Uint(0), Value::Uint(3)];
        for (raw, offset) in [(2, 0), (1, 8)] {
            w.call(mode, 13, 0, &[Value::Uint(raw), Value::Uint(offset)], &[])?;
            w.call("drawPrimitives:vertexStart:vertexCount:", 13, 0, &draw, &[])?;
        }
        if revisit {
            w.call(mode, 13, 0, &[Value::Uint(2), Value::Uint(0)], &[])?;
        }
        w.call("endEncoding", 13, 0, &[], &[])?;
        w.call("commit", 12, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 12, 0, &[], &[])?;
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(&bytes).to_vec())]);
        w.call("m2v.check", 0, 0, &[Value::Object(10)], &[("sha256", sum)])?;
        w.call("m2v.present", 1, 0, &[Value::Nil], &[])
    })
}

/// A `Reset` pass's visibility results play as the runtime's: `Counting` at offset 0 writes the
/// 16 samples of the target that the draw covers, `Boolean` at offset 8 the same count (the
/// runtime's `Boolean` word is the count, nonzero exactly where a sample passed), and offset 16,
/// which no mode took, keeps its bytes. The check reads those words, and nothing is skipped. The
/// backend's gauges carry the occlusion pools: the frame's end reads the one pool the command
/// buffer took, and the device's drop none.
#[test]
fn a_pass_s_visibility_results_play_on_the_new_runtime() {
    let dir = ScratchDir::new("rt-replay-visibility").unwrap();
    let path = visibility_results(&dir, 0, false, [16, 16, UNWRITTEN]);
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let mut reader = StreamReader::open(&[path], None).unwrap();
    let (report, ended) = Player::new(&mut backend).play(&mut reader);
    for line in report.lines() {
        eprintln!("REPLAY visibility {line}");
    }
    ended.unwrap();
    assert_eq!(
        report.ends[0].live["occlusion_pools"], 1,
        "{:?}",
        report.ends
    );
    let ended = backend.end();
    ended.result().unwrap();
    let pools = ended
        .gauges
        .iter()
        .find(|(gauge, _)| *gauge == "occlusion_pools");
    assert_eq!(pools, Some(&("occlusion_pools", 0)), "{:?}", ended.gauges);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    for skipped in [
        "skipped_encoders",
        "skipped_render_commands",
        "skipped_draws",
        "skipped_calls",
    ] {
        assert_eq!(executed[skipped], 0, "{skipped}: {executed:?}");
    }
    assert_eq!(report.first_skip, None);
}

/// A mode at an offset the encoder took before (call 22) is the runtime's refusal, Metal's own:
/// skipped and counted by its kind, with its frame and call, and the play goes on, the words those
/// of the modes before it.
#[test]
fn a_visibility_result_mode_the_runtime_refuses_is_counted_by_its_kind() {
    let dir = ScratchDir::new("rt-replay-visibility-revisit").unwrap();
    let path = visibility_results(&dir, 0, true, [16, 16, UNWRITTEN]);
    let (report, lines) = play("visibility-revisit", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    assert_eq!(report.total.executed["skipped_render_commands"], 1);
    assert_eq!(report.first_skip, Some((0, 22)));
    let line =
        "skipped render commands Refused(VisibilityResultOffsetTaken) 1, the first at frame 0, \
                call 22";
    assert!(lines.iter().any(|l| l == line), "{lines:?}");
}

/// In an `Accumulate` pass, `Counting` (call 18) and `Boolean` (call 20) are the runtime's gap,
/// skipped and counted, the draws drawn and the buffer's bytes as they were.
#[test]
fn an_accumulated_visibility_result_mode_is_counted_by_its_gap() {
    let dir = ScratchDir::new("rt-replay-visibility-accumulate").unwrap();
    let path = visibility_results(&dir, 1, false, [UNWRITTEN; 3]);
    let (report, lines) = play("visibility-accumulate", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    assert_eq!(executed["skipped_render_commands"], 2, "{executed:?}");
    assert_eq!(executed["skipped_draws"], 0, "{executed:?}");
    assert_eq!(report.first_skip, Some((0, 18)));
    let line =
        "skipped render commands Draw { kind: AccumulatedVisibilityResults } 2, the first at \
                frame 0, call 18";
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
fn a_sampler_bound_with_lod_clamps_plays_with_them() {
    let dir = ScratchDir::new("rt-replay-clamps").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    // The sampler's own clamps, (0, 1000), which reads as its own (0, FLT_MAX), and (0, 2), which
    // the state's variant of those clamps samples with: each binds, and the draw plays.
    for (test, clamp) in [
        ("clamps-own", f32::MAX),
        ("clamps-1000", 1000.0),
        ("clamps-2", 2.0),
    ] {
        let (report, _) = play(
            test,
            &draw_with_clamps(&dir, &green, Then::Nothing, Some(clamp)),
        );
        assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
        let executed = &report.total.executed;
        assert_eq!(executed["skipped_draws"], 0, "{executed:?}");
    }
}

/// An `NSRange` of `length` objects from `location`.
fn range(location: u64, length: u64) -> Value {
    uints(&[("location", location), ("length", length)])
}

/// A draw of one triangle from vertex 0: the selector and its arguments.
const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
const TRIANGLE: [Value; 3] = [Value::Uint(3), Value::Uint(0), Value::Uint(3)];

/// A stream of a draw of `colored_vertex` and `fragment` (a function of the runtime examples) into
/// a 4 by 4 RGBA8Unorm target (#7) its pass clears to red, the target copied to buffer #10, which
/// the stream checks against `expected`. The positions (#8) start 16 bytes into their buffer,
/// after four floats of -1, which as positions draw nothing; the colours (#9) are green. `make`
/// writes calls before command buffer #12, `encode` those of its render encoder #13 after its
/// pipeline state (#6): the binds and the draws. Their objects take ids from 15.
fn range_draw(
    dir: &ScratchDir,
    fragment: &'static str,
    expected: &[u8],
    make: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
    encode: impl FnOnce(&mut StreamWriter<File>) -> std::io::Result<()>,
) -> PathBuf {
    written(dir, |w| {
        let functions = [("colored_vertex", 0, 1), (fragment, 1, 2)];
        let library = w.blob(&library(dir, &functions))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let data = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &data, &[])?;
        for (id, name) in [(4, "colored_vertex"), (5, fragment)] {
            let name = [Value::Str(name.into())];
            w.call("newFunctionWithName:", 3, id, &name, &[])?;
        }
        let pipeline = Value::Map(vec![
            ("vertex".into(), Value::Object(4)),
            ("fragment".into(), Value::Object(5)),
            ("color0PixelFormat".into(), Value::Uint(70)),
        ]);
        let made = "newRenderPipelineStateWithDescriptor:error:";
        w.call(made, 1, 6, &[pipeline], &[])?;
        let target = uints(&[
            ("textureType", 2),
            ("pixelFormat", 70),
            ("width", 4),
            ("height", 4),
            ("usage", 5),
            ("storageMode", 2),
        ]);
        w.call("newTextureWithDescriptor:", 1, 7, &[target], &[])?;
        let positions = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let colors = [0.0, 1.0, 0.0, 1.0].repeat(3);
        for (id, values) in [(8, &positions[..]), (9, &colors[..])] {
            let bytes = floats(values);
            let length = Value::Uint(4 * values.len() as u64);
            let args = [bytes, length, Value::Uint(0)];
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
        make(w)?;
        w.call("commandBuffer", 2, 12, &[], &[])?;
        let pass = [Value::Object(11)];
        w.call("renderCommandEncoderWithDescriptor:", 12, 13, &pass, &[])?;
        w.call("setRenderPipelineState:", 13, 0, &[Value::Object(6)], &[])?;
        encode(w)?;
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
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(expected).to_vec())]);
        w.call("m2v.check", 0, 0, &[Value::Object(10)], &[("sha256", sum)])
    })
}

/// [`range_draw`]'s vertices bound one slot at a time: the positions at 16 bytes into theirs.
fn vertices(w: &mut StreamWriter<File>) -> std::io::Result<()> {
    for (buffer, offset, index) in [(8, 16, 0), (9, 0, 1)] {
        let args = [
            Value::Object(buffer),
            Value::Uint(offset),
            Value::Uint(index),
        ];
        w.call("setVertexBuffer:offset:atIndex:", 13, 0, &args, &[])?;
    }
    Ok(())
}

/// `setVertexBuffers:offsets:withRange:` binds each buffer at its offset, from the range's
/// location: the positions 16 bytes into theirs at slot 0 and the colours at slot 1 draw the green
/// triangle over the whole target.
#[test]
fn vertex_buffers_bound_through_a_range_take_their_offsets_and_slots() {
    let dir = ScratchDir::new("rt-replay-vertex-buffers-range").unwrap();
    let green = [0u8, 255, 0, 255].repeat(16);
    let path = range_draw(
        &dir,
        "colored_fragment",
        &green,
        |_| Ok(()),
        |w| {
            let args = [
                Value::List(vec![Value::Object(8), Value::Object(9)]),
                Value::List(vec![Value::Uint(16), Value::Uint(0)]),
                range(0, 2),
            ];
            w.call("setVertexBuffers:offsets:withRange:", 13, 0, &args, &[])?;
            w.call(DRAW, 13, 0, &TRIANGLE, &[])
        },
    );
    let (report, _) = play("vertex-buffers-range", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    assert_eq!(report.first_skip, None);
}

/// `setFragmentBuffers:offsets:withRange:` binds from the range's location: two buffers at
/// location 1 fill fragment slots 1 and 2, so a later offset of slot 2 plays, and one of slot 0,
/// where nothing is bound, is the runtime's refusal.
#[test]
fn fragment_buffers_bound_through_a_range_fill_the_slots_from_its_location() {
    let green = [0u8, 255, 0, 255].repeat(16);
    let stream = |dir: &ScratchDir, offset_at: u64| {
        range_draw(
            dir,
            "colored_fragment",
            &green,
            |_| Ok(()),
            |w| {
                vertices(w)?;
                let args = [
                    Value::List(vec![Value::Object(8), Value::Object(9)]),
                    Value::List(vec![Value::Uint(0), Value::Uint(0)]),
                    range(1, 2),
                ];
                w.call("setFragmentBuffers:offsets:withRange:", 13, 0, &args, &[])?;
                let offset = [Value::Uint(16), Value::Uint(offset_at)];
                w.call("setFragmentBufferOffset:atIndex:", 13, 0, &offset, &[])?;
                w.call(DRAW, 13, 0, &TRIANGLE, &[])
            },
        )
    };
    let dir = ScratchDir::new("rt-replay-fragment-buffers-range").unwrap();
    let (report, _) = play("fragment-buffers-range", &stream(&dir, 2));
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let dir = ScratchDir::new("rt-replay-fragment-buffers-range-0").unwrap();
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let mut reader = StreamReader::open(&[stream(&dir, 0)], None).unwrap();
    let (_, ended) = Player::new(&mut backend).play(&mut reader);
    match ended {
        Err(PlayError::Runtime { selector, .. }) => {
            assert_eq!(selector, "setFragmentBufferOffset:atIndex:");
        }
        other => panic!("an offset of an empty slot's refusal: {other:?}"),
    }
}

/// A texture the play skips (#16): RGB10A8_2P_XR10 over a surface of two planes (buffer #15's),
/// which reads across them.
fn across_planes(w: &mut StreamWriter<File>) -> std::io::Result<()> {
    let pages = [Value::Uint(4096), Value::Uint(0)];
    w.call("newBufferWithLength:options:", 1, 15, &pages, &[])?;
    let texture = uints(&[
        ("textureType", 2),
        ("pixelFormat", 550),
        ("width", 2),
        ("height", 2),
        ("usage", 1),
        ("storageMode", 0),
    ]);
    let facts = Value::Map(vec![
        (
            "pixelFormat".into(),
            Value::Uint(u64::from(u32::from_be_bytes(*b"&b3a"))),
        ),
        ("planeCount".into(), Value::Uint(2)),
        ("width".into(), Value::Uint(2)),
        ("height".into(), Value::Uint(2)),
        ("chromaOffset".into(), Value::Uint(2048)),
        ("chromaBytesPerRow".into(), Value::Uint(8)),
        ("ycbcrMatrix".into(), Value::Nil),
    ]);
    let plane = Value::Map(vec![
        ("plane".into(), Value::Uint(0)),
        ("offset".into(), Value::Uint(0)),
        ("bytesPerRow".into(), Value::Uint(8)),
        ("planar".into(), Value::Bool(true)),
        ("facts".into(), facts),
    ]);
    let surface = "newTextureWithDescriptor:iosurface:plane:";
    w.call(surface, 15, 16, &[texture, plane], &[])
}

/// A render encoder's residency declarations of a skipped object count it as a skipped call, as
/// a compute encoder's do, each of a list as its single form: the skipped texture
/// ([`across_planes`]) used by `useResource:usage:stages:` and listed beside a buffer by
/// `useResources:count:usage:` is two skipped calls. A residency declaration binds no slot, so
/// the draw after them plays.
#[test]
fn a_render_encoder_s_residency_declarations_count_a_skipped_object_as_a_skipped_call() {
    let dir = ScratchDir::new("rt-replay-render-uses-skipped").unwrap();
    let green = [0u8, 255, 0, 255].repeat(16);
    let path = range_draw(&dir, "colored_fragment", &green, across_planes, |w| {
        vertices(w)?;
        // MTLResourceUsageRead; MTLRenderStageFragment.
        let (read, fragment) = (Value::Uint(1), Value::Uint(2));
        let used = [resource("Texture", 16), read.clone(), fragment];
        w.call("useResource:usage:stages:", 13, 0, &used, &[])?;
        let listed = Value::List(vec![resource("Buffer", 8), resource("Texture", 16)]);
        w.call("useResources:count:usage:", 13, 0, &[listed, read], &[])?;
        w.call(DRAW, 13, 0, &TRIANGLE, &[])
    });
    let (report, _) = play("render-uses-skipped", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    assert_eq!(
        executed["skipped_surfaces_across_planes"], 1,
        "{executed:?}"
    );
    assert_eq!(executed["skipped_calls"], 2, "{executed:?}");
    assert_eq!(executed["skipped_draws"], 0, "{executed:?}");
}

/// `setVertexTextures:withRange:` binds from the range's location, each slot apart: a skipped
/// texture ([`across_planes`]) bound at location 1 skips the draws while its slot holds it, a nil
/// bound at location 0 leaves it there, and a nil bound at location 1 takes it off, so the second
/// draw plays.
#[test]
fn a_skipped_texture_bound_to_the_vertex_stage_through_a_range_skips_draws_while_its_slot_holds_it()
{
    let dir = ScratchDir::new("rt-replay-vertex-textures-range").unwrap();
    let green = [0u8, 255, 0, 255].repeat(16);
    let path = range_draw(&dir, "colored_fragment", &green, across_planes, |w| {
        vertices(w)?;
        let bind = "setVertexTextures:withRange:";
        let skipped = [Value::List(vec![Value::Object(16)]), range(1, 1)];
        w.call(bind, 13, 0, &skipped, &[])?;
        w.call(
            bind,
            13,
            0,
            &[Value::List(vec![Value::Nil]), range(0, 1)],
            &[],
        )?;
        w.call(DRAW, 13, 0, &TRIANGLE, &[])?;
        w.call(
            bind,
            13,
            0,
            &[Value::List(vec![Value::Nil]), range(1, 1)],
            &[],
        )?;
        w.call(DRAW, 13, 0, &TRIANGLE, &[])
    });
    let (report, _) = play("vertex-textures-range", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    assert_eq!(
        executed["skipped_surfaces_across_planes"], 1,
        "{executed:?}"
    );
    assert_eq!(executed["skipped_draws"], 1, "{executed:?}");
}

/// `setFragmentTextures:withRange:` binds from the range's location: of a red texel's texture
/// bound at location 0 and a blue one's at location 1, `first_texel_fragment` (fragment texture
/// 0's first texel plus the green colour) reads the red one, so the target is yellow.
#[test]
fn fragment_textures_bound_through_a_range_are_read_at_the_slot_its_location_names() {
    let dir = ScratchDir::new("rt-replay-fragment-textures-range").unwrap();
    let yellow = [255u8, 255, 0, 255].repeat(16);
    let make = |w: &mut StreamWriter<File>| {
        let texel = uints(&[
            ("textureType", 2),
            ("pixelFormat", 70),
            ("width", 1),
            ("height", 1),
            ("usage", 1),
            ("storageMode", 0),
        ]);
        let region = Value::Map(vec![
            ("origin".into(), uints(&[("x", 0), ("y", 0), ("z", 0)])),
            (
                "size".into(),
                uints(&[("width", 1), ("height", 1), ("depth", 1)]),
            ),
        ]);
        for (id, bytes) in [(15, [255, 0, 0, 0]), (16, [0, 0, 255, 0])] {
            w.call(
                "newTextureWithDescriptor:",
                1,
                id,
                std::slice::from_ref(&texel),
                &[],
            )?;
            let args = [
                region.clone(),
                Value::Uint(0),
                Value::Bytes(bytes.to_vec()),
                Value::Uint(4),
            ];
            let replace = "replaceRegion:mipmapLevel:withBytes:bytesPerRow:";
            w.call(replace, id, 0, &args, &[])?;
        }
        Ok(())
    };
    let path = range_draw(&dir, "first_texel_fragment", &yellow, make, |w| {
        vertices(w)?;
        let bind = "setFragmentTextures:withRange:";
        w.call(
            bind,
            13,
            0,
            &[Value::List(vec![Value::Object(15)]), range(0, 1)],
            &[],
        )?;
        w.call(
            bind,
            13,
            0,
            &[Value::List(vec![Value::Object(16)]), range(1, 1)],
            &[],
        )?;
        w.call(DRAW, 13, 0, &TRIANGLE, &[])
    });
    let (report, _) = play("fragment-textures-range", &path);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
}

/// A compute encoder's `setTextures:withRange:` binds from the range's location: the destination
/// bound at location 1, then the source at location 0, and `invert_texels` writes the source's
/// inverse into the destination.
#[test]
fn a_kernel_reads_and_writes_the_textures_a_range_binds() {
    let dir = ScratchDir::new("rt-replay-kernel-textures-range").unwrap();
    let source = [0x10, 0x20, 0x30, 0x40].repeat(4);
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
        // Shared RGBA8Unorm, 2 by 2: the source shader-read, the destination shader-write.
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
        w.call("newTextureWithDescriptor:", 1, 7, &[texture(1)], &[])?;
        w.call("newTextureWithDescriptor:", 1, 8, &[texture(2)], &[])?;
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
        w.call("commandBuffer", 2, 9, &[], &[])?;
        w.call("computeCommandEncoder", 9, 10, &[], &[])?;
        w.call("setComputePipelineState:", 10, 0, &[Value::Object(5)], &[])?;
        for (texture, location) in [(8, 1), (7, 0)] {
            let bind = [
                Value::List(vec![Value::Object(texture)]),
                range(location, 1),
            ];
            w.call("setTextures:withRange:", 10, 0, &bind, &[])?;
        }
        let size = |n| uints(&[("width", n), ("height", n), ("depth", 1)]);
        let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
        w.call(dispatch, 10, 0, &[size(1), size(2)], &[])?;
        w.call("endEncoding", 10, 0, &[], &[])?;
        w.call("commit", 9, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 9, 0, &[], &[])
    });
    // The destination's two rows of two texels, as far apart as half the read's length.
    let (report, lines, rows) = play_then("kernel-textures-range", &path, |backend| {
        let bytes = backend.read(8).unwrap();
        let pitch = bytes.len() / 2;
        [&bytes[..8], &bytes[pitch..pitch + 8]].concat()
    });
    let inverse: Vec<u8> = (source.chunks(4))
        .flat_map(|t| [0xFF - t[0], 0xFF - t[1], 0xFF - t[2], t[3]])
        .collect();
    assert_eq!(rows, inverse, "{lines:?}");
    assert_eq!(report.first_skip, None);
}

/// A compute encoder's four residency declarations play through the runtime before a dispatch:
/// `useResource:usage:` and `useResources:count:usage:` (each listed buffer with its usage) through
/// `use_resource`, `useHeap:` and `useHeaps:count:` (each listed heap) through `use_heap`, as the
/// stream records them (each C array and its count one list): the play ends, no call refused,
/// nothing skipped.
#[test]
fn a_compute_encoder_s_residency_declarations_play_through_the_runtime() {
    let dir = ScratchDir::new("rt-replay-compute-uses").unwrap();
    let workloads = metal2vulkan_devtools::fixtures_dir().join("harness/workloads.ll");
    let path = written(&dir, |w| {
        let library = library_of(&dir, &workloads, &[("scale_offset", 2, 0)]);
        let library = w.blob(&library)?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let library = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &library, &[])?;
        let kernel = [Value::Str("scale_offset".into())];
        w.call("newFunctionWithName:", 3, 4, &kernel, &[])?;
        let made = "newComputePipelineStateWithFunction:error:";
        w.call(made, 1, 5, &[Value::Object(4)], &[])?;
        let made = [Value::Uint(64), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 6, &made, &[])?;
        w.call("newBufferWithLength:options:", 1, 7, &made, &[])?;
        let block = [Value::Uint(0x1_0000), Value::Uint(0)];
        w.call("m2v.newBlockWithLength:storageMode:", 1, 10, &block, &[])?;
        w.call("commandBuffer", 2, 8, &[], &[])?;
        w.call("computeCommandEncoder", 8, 9, &[], &[])?;
        w.call("setComputePipelineState:", 9, 0, &[Value::Object(5)], &[])?;
        let affine = [Value::Bytes(vec![0; 8]), Value::Uint(8), Value::Uint(2)];
        w.call("setBytes:length:atIndex:", 9, 0, &affine, &[])?;
        let bind = |buffer, at| [Value::Object(buffer), Value::Uint(0), Value::Uint(at)];
        w.call("setBuffer:offset:atIndex:", 9, 0, &bind(6, 0), &[])?;
        w.call("setBuffer:offset:atIndex:", 9, 0, &bind(7, 1), &[])?;
        // MTLResourceUsageRead and Write.
        let (read, write) = (Value::Uint(1), Value::Uint(2));
        let buffers = Value::List(vec![resource("Buffer", 6), resource("Buffer", 7)]);
        let heaps = Value::List(vec![Value::Object(10)]);
        let uses = [
            ("useResource:usage:", vec![resource("Buffer", 6), read]),
            ("useResources:count:usage:", vec![buffers, write]),
            ("useHeap:", vec![Value::Object(10)]),
            ("useHeaps:count:", vec![heaps]),
        ];
        for (selector, args) in uses {
            w.call(selector, 9, 0, &args, &[])?;
        }
        let size = |n| uints(&[("width", n), ("height", 1), ("depth", 1)]);
        let dispatch = "dispatchThreads:threadsPerThreadgroup:";
        w.call(dispatch, 9, 0, &[size(4), size(4)], &[])?;
        w.call("endEncoding", 9, 0, &[], &[])?;
        w.call("commit", 8, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 8, 0, &[], &[])
    });
    let (report, _) = play("compute-uses", &path);
    let executed = &report.total.executed;
    for skipped in ["skipped_dispatches", "skipped_calls"] {
        assert_eq!(executed[skipped], 0, "{skipped}: {executed:?}");
    }
    assert_eq!(report.first_skip, None);
}

/// The `unwritten` line counts a Shared buffer a dispatch reads before anything reached it, but
/// not one the dispatch's state only writes, nor one made in a block over bytes an `m2v.write`
/// put there before its making: `scale_offset` reads #6 at buffer 0 and writes #7 at buffer 1,
/// then reads #11, made in block #10 over the bytes written there.
#[test]
fn the_unwritten_line_counts_what_a_state_reads_before_anything_reached_it() {
    let dir = ScratchDir::new("rt-replay-unwritten").unwrap();
    let workloads = metal2vulkan_devtools::fixtures_dir().join("harness/workloads.ll");
    let path = written(&dir, |w| {
        let library = library_of(&dir, &workloads, &[("scale_offset", 2, 0)]);
        let library = w.blob(&library)?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let library = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &library, &[])?;
        let kernel = [Value::Str("scale_offset".into())];
        w.call("newFunctionWithName:", 3, 4, &kernel, &[])?;
        let made = "newComputePipelineStateWithFunction:error:";
        w.call(made, 1, 5, &[Value::Object(4)], &[])?;
        let made = [Value::Uint(64), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 6, &made, &[])?;
        w.call("newBufferWithLength:options:", 1, 7, &made, &[])?;
        // A Shared block, written at 0..64 before #11 is made there.
        let block = [Value::Uint(65536), Value::Uint(0)];
        w.call("m2v.newBlockWithLength:storageMode:", 1, 10, &block, &[])?;
        let bytes = [Value::Uint(0), Value::Bytes(vec![1; 64])];
        w.call("m2v.write", 10, 0, &bytes, &[])?;
        let placed = [
            Value::Object(10),
            Value::Uint(0),
            Value::Uint(64),
            Value::Uint(0),
        ];
        w.call(
            "m2v.newBufferInBlock:offset:length:options:",
            10,
            11,
            &placed,
            &[],
        )?;
        w.call("commandBuffer", 2, 8, &[], &[])?;
        w.call("computeCommandEncoder", 8, 9, &[], &[])?;
        w.call("setComputePipelineState:", 9, 0, &[Value::Object(5)], &[])?;
        let affine = [Value::Bytes(vec![0; 8]), Value::Uint(8), Value::Uint(2)];
        w.call("setBytes:length:atIndex:", 9, 0, &affine, &[])?;
        let size = |n| uints(&[("width", n), ("height", 1), ("depth", 1)]);
        let bind = |buffer, at| [Value::Object(buffer), Value::Uint(0), Value::Uint(at)];
        for source in [6, 11] {
            w.call("setBuffer:offset:atIndex:", 9, 0, &bind(source, 0), &[])?;
            w.call("setBuffer:offset:atIndex:", 9, 0, &bind(7, 1), &[])?;
            let dispatch = "dispatchThreads:threadsPerThreadgroup:";
            w.call(dispatch, 9, 0, &[size(4), size(4)], &[])?;
        }
        w.call("endEncoding", 9, 0, &[], &[])?;
        w.call("commit", 8, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 8, 0, &[], &[])
    });
    let (_, lines) = play("unwritten", &path);
    let line = lines.iter().find(|line| line.starts_with("unwritten "));
    let line = line.unwrap_or_else(|| panic!("an unwritten line: {lines:?}"));
    assert!(
        line.starts_with("unwritten 1: ") && line.ends_with("; first #6 in #8"),
        "{line}"
    );
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
    let line =
        "skipped compute states IndirectCommandBuffers { kind: ComputeCommands } 1, the first #5: ";
    assert!(lines.iter().any(|l| l.starts_with(line)), "{lines:?}");
}

/// The milliseconds `line` gives after `label`: its making's or makings' `wall W ms =
/// translation T ms + driver D ms + other O ms`, as `[W, T, D, O]`.
fn split_after(line: &str, label: &str) -> [f64; 4] {
    let (_, rest) = line.split_once(label).expect(label);
    let numbers: Vec<f64> = (rest.split(' '))
        .take(16)
        .filter_map(|word| word.parse().ok())
        .collect();
    numbers[..4].try_into().unwrap()
}

#[test]
fn each_pipeline_state_s_making_is_split_into_the_translator_s_and_the_driver_s_time() {
    let dir = ScratchDir::new("rt-replay-makes").unwrap();
    let functions = [
        ("scale_and_offset", 2, 0),
        ("colored_vertex", 0, 1),
        ("colored_fragment", 1, 2),
    ];
    let path = written(&dir, |w| {
        let library = w.blob(&library(&dir, &functions))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let library = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &library, &[])?;
        for (id, (name, _, _)) in (4..).zip(functions) {
            w.call(
                "newFunctionWithName:",
                3,
                id,
                &[Value::Str(name.into())],
                &[],
            )?;
        }
        // Two states of one kernel: the device holds the second's content.
        let made = "newComputePipelineStateWithFunction:error:";
        w.call(made, 1, 7, &[Value::Object(4)], &[])?;
        w.call(made, 1, 8, &[Value::Object(4)], &[])?;
        let pipeline = Value::Map(vec![
            ("vertex".into(), Value::Object(5)),
            ("fragment".into(), Value::Object(6)),
            ("color0PixelFormat".into(), Value::Uint(70)),
        ]);
        let made = "newRenderPipelineStateWithDescriptor:error:";
        w.call(made, 1, 9, &[pipeline], &[])
    });
    let (_, lines) = play("makes", &path);
    let summary = (lines.iter())
        .find(|line| line.starts_with("pipeline makes "))
        .unwrap_or_else(|| panic!("{lines:?}"));
    assert!(
        summary.starts_with("pipeline makes 3 states: 2 new ("),
        "{summary}"
    );
    let end = "; 0 beside a pre-pass call, whose work their figures count";
    assert!(summary.ends_with(end), "{summary}");
    // The kernel's one or two lowerings and the render state's two, and its triangle class.
    let (_, made) = summary.split_once(" new (").unwrap();
    let (made, _) = made.split_once(')').unwrap();
    let count = |what: &str| -> u64 {
        let word = made.split(", ").find(|part| part.ends_with(what));
        word.and_then(|part| part.split(' ').next()?.parse().ok())
            .unwrap()
    };
    assert!(count(" lowered") >= 3, "{summary}");
    assert_eq!(count(" loaded"), 0, "{summary}");
    assert!(count(" VkPipelines") >= 1, "{summary}");
    let [wall, translation, driver, other] = split_after(summary, "VkPipelines), ");
    assert!(
        translation > 0.0 && driver > 0.0 && other >= 0.0,
        "{summary}"
    );
    assert!(wall >= translation + driver, "{summary}");
    let [.., other] = split_after(summary, " held, ");
    let (_, held) = summary.split_once(" held, ").unwrap();
    assert!(
        held.contains(" + driver 0.000 ms + ") && other >= 0.0,
        "{summary}"
    );
    let slowest: Vec<&String> = (lines.iter())
        .filter(|line| line.starts_with("pipeline make slowest "))
        .collect();
    assert_eq!(slowest.len(), 3, "{lines:?}");
    for (state, kinds) in [
        (7, "kernel new ("),
        (8, "kernel held (0 lowered, 0 loaded, 0 VkPipelines): wall "),
        (9, "vertex+fragment new ("),
    ] {
        let named = format!(" #{state} {kinds}");
        assert!(
            slowest.iter().any(|line| line.contains(&named)),
            "{slowest:?}"
        );
    }
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

#[test]
fn a_check_of_a_managed_buffer_names_the_copy_it_read() {
    let dir = ScratchDir::new("rt-replay-read-copies").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let identity = [("hasUnifiedMemory", Value::Bool(false))];
        w.call("m2v.deviceIdentity", 1, 0, &[], &identity)?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        // #3 Managed (MTLResourceStorageModeManaged, 1 << 4), #4 Shared, 4 bytes each, zeros.
        let made = "newBufferWithLength:options:";
        for (id, options) in [(3, 16), (4, 0)] {
            w.call(made, 1, id, &[Value::Uint(4), Value::Uint(options)], &[])?;
        }
        // A fill of 7s reaches #3's GPU copy; its CPU copy keeps its zeros.
        let range = uints(&[("location", 0), ("length", 4)]);
        let fill = [Value::Object(3), range, Value::Uint(7)];
        w.call("commandBuffer", 2, 10, &[], &[])?;
        w.call("blitCommandEncoder", 10, 11, &[], &[])?;
        w.call("fillBuffer:range:value:", 11, 0, &fill, &[])?;
        w.call("endEncoding", 11, 0, &[], &[])?;
        w.call("commit", 10, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 10, 0, &[], &[])?;
        w.call(
            "m2v.check",
            0,
            0,
            &[Value::Object(3), Value::Object(4)],
            &[],
        )
    });
    for device_reads in [false, true] {
        let cache = ScratchCache::new().unwrap();
        let device = rt_device(cache.path());
        let unified = device.has_unified_memory();
        let backend = NewRuntime::new(device, gpu::TEST_DEVICE).unwrap();
        let mut backend = match device_reads {
            true => backend.reading_device_copies(),
            false => backend,
        };
        let header = backend.managed_digests();
        let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
        let (report, ended) = Player::new(&mut backend).play(&mut reader);
        let lines = report.lines();
        for line in header.iter().chain(&lines) {
            eprintln!("REPLAY read-copies {device_reads} {line}");
        }
        ended.unwrap();
        let (copy, bytes, said) = match device_reads {
            false => (ReadCopy::Cpu, [0u8; 4], "reads its CPU copy only"),
            true => (ReadCopy::Device, [7u8; 4], "reads the device copy back"),
        };
        assert_eq!(report.checks.len(), 2, "{lines:?}");
        let (managed, shared) = (&report.checks[0], &report.checks[1]);
        assert_eq!((managed.object, managed.copy), (3, Some(copy)), "{lines:?}");
        assert_eq!(managed.sha256[..], Sha256::digest(bytes)[..], "{lines:?}");
        assert_eq!((shared.object, shared.copy), (4, None), "{lines:?}");
        let label = format!("({})", copy.label());
        let check = lines.iter().find(|line| line.starts_with("check #3 "));
        assert!(
            check.is_some_and(|line| line.ends_with(&label)),
            "{lines:?}"
        );
        assert_eq!(header.is_some(), !unified, "{header:?}");
        if let Some(header) = header {
            assert!(header.contains(said), "{header}");
        }
    }
}

#[test]
fn a_private_buffer_reads_back_through_a_copy() {
    let dir = ScratchDir::new("rt-replay-private-read").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        // #3 Private (MTLResourceStorageModePrivate, 2 << 4), 16 bytes, filled with 0x5a.
        let made = [Value::Uint(16), Value::Uint(32)];
        w.call("newBufferWithLength:options:", 1, 3, &made, &[])?;
        w.call("commandBuffer", 2, 10, &[], &[])?;
        w.call("blitCommandEncoder", 10, 11, &[], &[])?;
        let range = uints(&[("location", 0), ("length", 16)]);
        let fill = [Value::Object(3), range, Value::Uint(0x5a)];
        w.call("fillBuffer:range:value:", 11, 0, &fill, &[])?;
        w.call("endEncoding", 11, 0, &[], &[])?;
        w.call("commit", 10, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 10, 0, &[], &[])
    });
    let (_, lines, bytes) = play_then("private-read", &path, |backend| backend.read(3));
    let bytes = bytes.unwrap_or_else(|error| panic!("#3 read back: {error}: {lines:?}"));
    assert_eq!(bytes, vec![0x5a; 16]);
}

/// A Private texture the program cleared to red in a render pass and shows at present 0: the play
/// listing present 0 (`Player::checking_presents`) reads it back through a copy, which the CPU's
/// read of a Private texture is not, and checks its 16 red texels: the shown region read as tight
/// rows of texels, so a pitch or a row's padding cannot read as a picture that differs.
#[test]
fn a_private_shown_texture_is_checked_through_a_copy() {
    let dir = ScratchDir::new("rt-replay-private-shown").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        // RGBA8Unorm, 4 by 4, shader-read and render-target, Private.
        let target = uints(&[
            ("textureType", 2),
            ("pixelFormat", 70),
            ("width", 4),
            ("height", 4),
            ("usage", 5),
            ("storageMode", 2),
        ]);
        w.call("newTextureWithDescriptor:", 1, 3, &[target], &[])?;
        w.call("MTLRenderPassDescriptor.new", 0, 4, &[], &[])?;
        let pass = uints(&[("renderTargetWidth", 4), ("renderTargetHeight", 4)]);
        w.call("m2v.setPass", 4, 0, &[pass], &[])?;
        // MTLLoadActionClear, MTLStoreActionStore.
        let Value::Map(mut color) = uints(&[("level", 0), ("loadAction", 2), ("storeAction", 1)])
        else {
            unreachable!("a map");
        };
        color.push(("texture".into(), Value::Object(3)));
        let red = [("red", 1.0), ("green", 0.0), ("blue", 0.0), ("alpha", 1.0)]
            .map(|(key, value)| (key.into(), Value::F64(value)));
        color.push(("clearColor".into(), Value::Map(red.to_vec())));
        let args = [Value::Uint(0), Value::Map(color)];
        w.call("m2v.setColorAttachment", 4, 0, &args, &[])?;
        w.call("commandBuffer", 2, 5, &[], &[])?;
        w.call(
            "renderCommandEncoderWithDescriptor:",
            5,
            6,
            &[Value::Object(4)],
            &[],
        )?;
        w.call("endEncoding", 6, 0, &[], &[])?;
        w.call("commit", 5, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 5, 0, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Object(3)], &[])
    });
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
    let player = Player::new(&mut backend).checking_presents([0]);
    let (rt, ended) = player.play(&mut reader);
    let lines = rt.lines();
    ended.unwrap_or_else(|error| panic!("the play ended: {error}: {lines:?}"));
    let checked = |report: &Report| -> Vec<(u64, u64, [u8; 32])> {
        (report.checks.iter())
            .map(|check| (check.object, check.frame, check.sha256))
            .collect()
    };
    let red: [u8; 32] = Sha256::digest([255u8, 0, 0, 255].repeat(16)).into();
    assert_eq!(checked(&rt), [(3, 0, red)], "{lines:?}");
}

/// A tiled texture over Shared buffer #3 (`newTiledTextureWithDescriptor:offset:bytesPerRow:`) of
/// descriptor object #9, 128 by 2 RGBA8Unorm texels of two levels, 512 bytes a row: the buffer's texels come in 2 by 2
/// squares of one colour each, so the second level made from the first by a filter of each square
/// is that square's colour exactly. A blit encoder copies the first level into #5 and the second
/// into #6, each shown at a present. On the runtime the use takes the buffer's bytes into the
/// first level and makes the second from it, once.
#[test]
fn a_tiled_texture_takes_its_buffer_s_bytes_and_levels_at_first_use() {
    let dir = ScratchDir::new("rt-replay-tiled").unwrap();
    let colour = |x: u64| [(x / 2 * 4) as u8, 0xff - (x / 2 * 4) as u8, 0x40, 0xff];
    let first: Vec<u8> = (0..2).flat_map(|_| (0..128).flat_map(colour)).collect();
    let second: Vec<u8> = (0..64).flat_map(|x| colour(2 * x)).collect();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let made = [Value::Uint(1024), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 3, &made, &[])?;
        let write = [Value::Uint(0), Value::Bytes(first.clone())];
        w.call("m2v.write", 3, 0, &write, &[])?;
        // Shared, shader-read.
        let texture = |width, height, levels| {
            uints(&[
                ("textureType", 2),
                ("pixelFormat", 70),
                ("width", width),
                ("height", height),
                ("mipmapLevelCount", levels),
                ("usage", 1),
                ("storageMode", 0),
            ])
        };
        w.call("m2v.textureDescriptor", 0, 9, &[texture(128, 2, 2)], &[])?;
        let tiled = [Value::Object(9), Value::Uint(0), Value::Uint(512)];
        let selector = "newTiledTextureWithDescriptor:offset:bytesPerRow:";
        w.call(selector, 3, 4, &tiled, &[])?;
        w.call(
            "newTextureWithDescriptor:",
            1,
            5,
            &[texture(128, 2, 1)],
            &[],
        )?;
        w.call("newTextureWithDescriptor:", 1, 6, &[texture(64, 1, 1)], &[])?;
        w.call("commandBuffer", 2, 7, &[], &[])?;
        w.call("blitCommandEncoder", 7, 8, &[], &[])?;
        let blit = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:\
                    destinationSlice:destinationLevel:destinationOrigin:";
        let origin = uints(&[("x", 0), ("y", 0), ("z", 0)]);
        for (level, (width, height), into) in [(0, (128, 2), 5), (1, (64, 1), 6)] {
            let size = uints(&[("width", width), ("height", height), ("depth", 1)]);
            let copy = [
                Value::Object(4),
                Value::Uint(0),
                Value::Uint(level),
                origin.clone(),
                size,
                Value::Object(into),
                Value::Uint(0),
                Value::Uint(0),
                origin.clone(),
            ];
            w.call(blit, 8, 0, &copy, &[])?;
        }
        w.call("endEncoding", 8, 0, &[], &[])?;
        w.call("commit", 7, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 7, 0, &[], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Object(5)], &[])?;
        w.call("m2v.present", 1, 0, &[Value::Object(6)], &[])
    });
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let mut reader = StreamReader::open(std::slice::from_ref(&path), None).unwrap();
    let player = Player::new(&mut backend).checking_presents([0, 1]);
    let (rt, ended) = player.play(&mut reader);
    let lines = rt.lines();
    ended.unwrap_or_else(|error| panic!("the play ended: {error}: {lines:?}"));
    let checked = |report: &Report| -> Vec<(u64, u64, [u8; 32])> {
        (report.checks.iter())
            .map(|check| (check.object, check.frame, check.sha256))
            .collect()
    };
    let digest = |bytes: &[u8]| -> [u8; 32] { Sha256::digest(bytes).into() };
    let shown = [(5, 0, digest(&first)), (6, 1, digest(&second))];
    assert_eq!(checked(&rt), shown, "{lines:?}");
    let executed = &rt.total.executed;
    let counted = |key: &str| executed.get(key).copied().unwrap_or(0);
    let counts = [
        "tiled_textures",
        "tiled_takes",
        "tiled_take_bytes",
        "tiled_mipmap_generations",
        "tiled_levels_kept",
        "substitutes_read",
    ]
    .map(counted);
    assert_eq!(counts, [1, 1, 1024, 1, 0, 0], "{executed:?}");
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

/// The surface picture of [`a_surface_picture_fills_its_shapes_at_their_first_use`] behind
/// another queue's long work: queue #20's command buffer #24, 256 dispatches over 4 MiB, is
/// committed first, so #8's making fill can run behind it on the device and the refill the
/// kernel's commit owes be held behind both, where queue #2, idle, ranks first for the room
/// (`metal2vulkan_runtime`'s submit thread), as Metal's hazard tracking allows: it orders a
/// command buffer of another queue only against work already submitted. Whatever the order the
/// device takes, the commit waits on the CPU for the refill, once (`surface_host_waits`), so the
/// kernel reads the second picture.
#[test]
fn a_surface_picture_fills_before_its_use_behind_another_queues_long_work() {
    let dir = ScratchDir::new("rt-replay-surface-behind").unwrap();
    let first = [0x10, 0x20, 0x30, 0x40].repeat(4);
    let second = [0x50, 0x60, 0x70, 0x80].repeat(4);
    let path = written(&dir, |w| {
        let functions = [("invert_texels", 2, 0), ("scale_and_offset", 2, 0)];
        let library = w.blob(&library(&dir, &functions))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("newCommandQueue", 1, 20, &[], &[])?;
        w.call(
            "newLibraryWithData:error:",
            1,
            3,
            &[Value::Blob(library)],
            &[],
        )?;
        let made = "newComputePipelineStateWithFunction:error:";
        for (name, function, state) in [("invert_texels", 4, 5), ("scale_and_offset", 21, 22)] {
            let kernel = [Value::Str(name.into())];
            w.call("newFunctionWithName:", 3, function, &kernel, &[])?;
            w.call(made, 1, state, &[Value::Object(function)], &[])?;
        }
        let long = [Value::Uint(4 << 20), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, 23, &long, &[])?;
        w.call("newBufferWithLength:options:", 1, 26, &long, &[])?;
        w.call("commandBuffer", 20, 24, &[], &[])?;
        w.call("computeCommandEncoder", 24, 25, &[], &[])?;
        w.call("setComputePipelineState:", 25, 0, &[Value::Object(22)], &[])?;
        let scale = [Value::Bytes(vec![0; 4]), Value::Uint(4), Value::Uint(2)];
        w.call("setBytes:length:atIndex:", 25, 0, &scale, &[])?;
        for (buffer, at) in [(23, 0), (26, 1)] {
            let bind = [Value::Object(buffer), Value::Uint(0), Value::Uint(at)];
            w.call("setBuffer:offset:atIndex:", 25, 0, &bind, &[])?;
        }
        let size = |n| uints(&[("width", n), ("height", 1), ("depth", 1)]);
        let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
        for _ in 0..256 {
            w.call(dispatch, 25, 0, &[size(1 << 14), size(64)], &[])?;
        }
        w.call("endEncoding", 25, 0, &[], &[])?;
        w.call("commit", 24, 0, &[], &[])?;
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
        invert(w, 11, 8, 10)?;
        w.call(WAIT_COMPLETED, 24, 0, &[], &[])
    });
    let (report, lines, rows) = play_then("surface-behind", &path, |backend| {
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
    let counts = ["surface_refills", "surface_host_waits"].map(counted);
    assert_eq!(counts, [1, 1], "{executed:?}");
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

/// What command buffer #6 of [`held_placement`] names.
#[derive(Clone, Copy, PartialEq)]
enum Held {
    /// It binds sampler state #5, which holds index 16.
    Named,
    /// Sampler state #11 (a linear min filter) takes index 16 from #5 before #6 binds #5, and
    /// lives on: two live sampler states placed at 16, #5 released after #11 took it.
    Displaced,
    /// It binds nothing: it may reach #5 only through its ID in memory.
    Unnamed,
}

/// A stream of a placement at an index a held commit owes: command buffer #6 of queue #2 waits for
/// event #4 to reach 1, has a compute encoder that names what `held` says, and is committed; #5
/// (the default descriptor, index 16) is released and sampler state #8 made at index 16 with
/// `descriptor`; only then does command buffer #10 of queue #3 signal #4 to 1, and the CPU waits
/// for #6.
fn held_placement(dir: &ScratchDir, descriptor: Value, held: Held) -> PathBuf {
    written(dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        w.call("newCommandQueue", 1, 3, &[], &[])?;
        w.call("newEvent", 1, 4, &[], &[])?;
        let made = "newSamplerStateWithDescriptor:";
        let id = [("gpuResourceID", Value::Uint(16))];
        w.call(made, 1, 5, &[sampler(None)], &[])?;
        w.call("gpuResourceID", 5, 0, &[], &id)?;
        if held == Held::Displaced {
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
        if held != Held::Unnamed {
            w.call("setSamplerStates:withRange:", 7, 0, &bound, &[])?;
        }
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
    let path = held_placement(&dir, sampler(None), Held::Named);
    let (_, lines) = play("equal-sampler", &path);
    let indices = lines.iter().find(|line| line.starts_with("indices named "));
    let indices = indices.unwrap_or_else(|| panic!("an indices line: {lines:?}"));
    assert!(
        indices.contains("0 placed at an index owed to commits in flight")
            && indices.ends_with(
                "1 sampler states placed without waiting at an index whose last sampler state \
                 had the same descriptor while its commits were in flight"
            ),
        "{indices}"
    );
}

/// `path` played: its sampler state #8 is kept off index 16, which a held commit owes, at the
/// highest sampler index, waits for nothing and is counted.
fn kept_back(test: &str, path: &Path) {
    let (_, lines, indices) = played(test, path, &[8]);
    let highest = metal2vulkan_runtime::IndexClass::Sampler.range().end - 1;
    assert_eq!(indices, [highest]);
    let kept = "indices kept back 1: objects placed at the highest free index, not at the one the \
                stream gives them, which is owed to a command buffer no record played so far lets \
                complete";
    assert!(lines.iter().any(|line| line == kept), "{lines:?}");
    let indices = lines.iter().find(|line| line.starts_with("indices named "));
    let indices = indices.unwrap_or_else(|| panic!("an indices line: {lines:?}"));
    assert!(
        indices.contains("0 placed at an index owed to commits in flight"),
        "{indices}"
    );
}

#[test]
fn a_placement_a_held_commit_owes_is_kept_back_and_counted() {
    let dir = ScratchDir::new("rt-replay-held-placement").unwrap();
    // A sampler state of another descriptor at index 16: #6, which reads 16, waits for a signal
    // only a later record makes, so no wait for it can end before #8's placement.
    let other = uints(&[("minFilter", 1), ("magFilter", 1)]);
    kept_back("held-placement", &held_placement(&dir, other, Held::Named));
}

#[test]
fn an_index_released_after_a_held_commit_is_kept_back_though_it_names_nothing() {
    let dir = ScratchDir::new("rt-replay-held-unnamed").unwrap();
    // #6 binds nothing, but was committed before #5's release: it may read 16 through #5's ID in
    // memory, and its event holds it past #8's placement.
    let other = uints(&[("minFilter", 1), ("magFilter", 1)]);
    kept_back("held-unnamed", &held_placement(&dir, other, Held::Unnamed));
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
    let path = held_placement(&dir, other, Held::Displaced);
    let (_, lines, indices) = played("displaced", &path, &[8]);
    assert_eq!(indices, [16]);
    let kept = "indices kept back 0:";
    assert!(lines.iter().any(|line| line.starts_with(kept)), "{lines:?}");
    let indices = lines.iter().find(|line| line.starts_with("indices named "));
    let indices = indices.unwrap_or_else(|| panic!("an indices line: {lines:?}"));
    assert!(
        indices.contains("0 placed at an index owed to commits in flight"),
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

/// On the runtime, which makes the block elsewhere, the word that looks like an address
/// plays as written: the buffer reads back the recorded bytes, the word is counted, and nothing
/// moved.
#[test]
fn a_data_word_in_a_live_range_plays_as_written_on_the_new_runtime() {
    let dir = ScratchDir::new("rt-replay-data-word").unwrap();
    let (report, _) = play("data-word", &census_stream(&dir));
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    assert_eq!(census(&report), [1, 1, 0]);
}

/// A buffer the stream noted escaped (`m2v.bufferEscaped`, a `gpuAddress` read) in a block at a
/// recorded address, and a buffer of the same block whose written word is that buffer's recorded
/// address plus 16. The new runtime makes the block elsewhere, so the player moves the word to the
/// buffer's address in the play plus 16, counted as noted, and `ab_address` (`out[i] =
/// held.p[i]`, the word its constant buffer) follows it: `out` holds the buffer's words from the
/// fifth on, exactly.
#[test]
fn a_noted_buffer_s_address_in_written_bytes_moves_and_a_kernel_follows_it() {
    const THREADS: u64 = 16;
    let dir = ScratchDir::new("rt-replay-noted-address").unwrap();
    let block = 1u64 << 46;
    let data: Vec<u32> = (0..64).map(|i| 3 * i + 1).collect();
    let path = written(&dir, |w| {
        let ll = metal2vulkan_devtools::engine_dir().join("ab_pointers.ll");
        let library = w.blob(&library_of(&dir, &ll, &[("ab_address", 2, 0)]))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        let data_of = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &data_of, &[])?;
        let kernel = [Value::Str("ab_address".into())];
        w.call("newFunctionWithName:", 3, 4, &kernel, &[])?;
        let made = "newComputePipelineStateWithFunction:error:";
        w.call(made, 1, 5, &[Value::Object(4)], &[])?;
        let at = |address: u64, length: u64| {
            [
                ("gpuAddress", Value::Address(address)),
                ("gpuLength", Value::Uint(length)),
            ]
        };
        let length = [Value::Uint(0x1_0000), Value::Uint(0)];
        let selector = "m2v.newBlockWithLength:storageMode:";
        w.call(selector, 1, 6, &length, &at(block, 0x1_0000))?;
        // The data, the held word and the output, each of 256 bytes in the Shared block.
        let selector = "m2v.newBufferInBlock:offset:length:options:";
        for (buffer, offset) in [(7, 0x100), (8, 0x200), (9, 0x300)] {
            let inner = [
                Value::Object(6),
                Value::Uint(offset),
                Value::Uint(0x100),
                Value::Uint(0),
            ];
            w.call(selector, 6, buffer, &inner, &at(block + offset, 0x100))?;
        }
        let words: Vec<u8> = data.iter().flat_map(|word| word.to_le_bytes()).collect();
        w.call(
            "m2v.write",
            7,
            0,
            &[Value::Uint(0), Value::Bytes(words)],
            &[],
        )?;
        w.call("m2v.bufferEscaped", 7, 0, &[Value::Uint(0)], &[])?;
        let held = (block + 0x100 + 16).to_le_bytes().to_vec();
        w.call(
            "m2v.write",
            8,
            0,
            &[Value::Uint(0), Value::Bytes(held)],
            &[],
        )?;
        w.call("commandBuffer", 2, 10, &[], &[])?;
        w.call("computeCommandEncoder", 10, 11, &[], &[])?;
        w.call("setComputePipelineState:", 11, 0, &[Value::Object(5)], &[])?;
        for (buffer, index) in [(8, 0), (9, 1)] {
            let bind = [Value::Object(buffer), Value::Uint(0), Value::Uint(index)];
            w.call("setBuffer:offset:atIndex:", 11, 0, &bind, &[])?;
        }
        let size = |width| uints(&[("width", width), ("height", 1), ("depth", 1)]);
        let dispatch = "dispatchThreadgroups:threadsPerThreadgroup:";
        w.call(dispatch, 11, 0, &[size(1), size(THREADS)], &[])?;
        w.call("endEncoding", 11, 0, &[], &[])?;
        w.call("commit", 10, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 10, 0, &[], &[])
    });
    let (report, lines, out) =
        play_then("noted-address", &path, |backend| backend.read(9).unwrap());
    let followed: Vec<u8> = (data[4..4 + THREADS as usize].iter())
        .flat_map(|word| word.to_le_bytes())
        .collect();
    assert_eq!(out[..followed.len()], followed[..], "{lines:?}");
    assert_eq!(report.total.stream["noted_words"], 1);
    assert_eq!(report.escape_notes, 1);
    assert_eq!(report.first_skip, None);
}

/// A copy back to the guest's RAM plays in the host's one-copy form: one buffer copy of the whole
/// words its runs span within the buffer into a scratch of the play's own, after its command
/// buffer, which a wait for that command buffer waits for; then the CPU copies each run out of the
/// scratch. Nothing is skipped, the copy and the runs are counted, and the buffer keeps its bytes.
#[test]
fn a_copy_back_to_the_guest_s_ram_plays_as_one_copy_then_a_cpu_copy_a_run() {
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
    let (report, lines) = play("copy-back", &path);
    assert_eq!(report.first_skip, None);
    assert_eq!(Some(report.checks[0].sha256), report.checks[0].recorded);
    let executed = &report.total.executed;
    // 1..17 is the words 0..20, and 90..100 the words 88..100: one copy of 0..100.
    assert_eq!(executed["guest_ram_downloads"], 1);
    assert_eq!(executed["guest_ram_download_bytes"], 100);
    assert_eq!(executed["guest_ram_runs_out"], 2);
    assert_eq!(executed["guest_ram_run_bytes_out"], 16 + 10);
    let line =
        "guest RAM: copy-back played in the one-copy form: 1 copies back (100 bytes), 2 runs \
                (26 bytes)";
    assert!(lines.iter().any(|l| l.starts_with(line)), "{lines:?}");
}

/// A copy from the guest's RAM plays in the host's one-copy form: its pieces gathered by the CPU
/// into a scratch of the play's own and one buffer copy into the buffer, before its command
/// buffer. Into Shared #3, the gap between the pieces and a piece's edges up to whole words keep
/// the buffer's bytes, as its mapping holds them, in one copy. Into Private #4, which has no
/// mapping, a piece that is not whole words is skipped and counted, and the pieces go as one copy
/// for each run of touching words, so no gap is written.
#[test]
fn a_copy_from_the_guest_s_ram_plays_as_a_cpu_gather_then_one_copy() {
    let dir = ScratchDir::new("rt-replay-copy-in").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        w.call("newCommandQueue", 1, 2, &[], &[])?;
        // #3 Shared, #4 Private (MTLResourceStorageModePrivate, 2 << 4), 32 bytes each.
        let made = "newBufferWithLength:options:";
        w.call(made, 1, 3, &[Value::Uint(32), Value::Uint(0)], &[])?;
        w.call(made, 1, 4, &[Value::Uint(32), Value::Uint(32)], &[])?;
        let write = [Value::Uint(0), Value::Bytes(vec![0x11; 32])];
        w.call("m2v.write", 3, 0, &write, &[])?;
        w.call("commandBuffer", 2, 10, &[], &[])?;
        w.call("blitCommandEncoder", 10, 11, &[], &[])?;
        let range = uints(&[("location", 0), ("length", 32)]);
        let fill = [Value::Object(4), range, Value::Uint(0x5a)];
        w.call("fillBuffer:range:value:", 11, 0, &fill, &[])?;
        w.call("endEncoding", 11, 0, &[], &[])?;
        w.call("commit", 10, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 10, 0, &[], &[])?;
        w.call("commandBuffer", 2, 20, &[], &[])?;
        w.call("blitCommandEncoder", 20, 21, &[], &[])?;
        let piece = |at: u64, byte: u8, len: usize| {
            Value::List(vec![Value::Uint(at), Value::Bytes(vec![byte; len])])
        };
        let into = |buffer: u64, pieces: Vec<Value>| [Value::Object(buffer), Value::List(pieces)];
        let shared = into(3, vec![piece(4, 0xa0, 4), piece(13, 0xb0, 6)]);
        w.call("m2v.copyFromGuestRam", 21, 0, &shared, &[])?;
        let private = into(
            4,
            vec![piece(0, 0xc0, 4), piece(16, 0xd0, 8), piece(9, 0xe0, 2)],
        );
        w.call("m2v.copyFromGuestRam", 21, 0, &private, &[])?;
        w.call("endEncoding", 21, 0, &[], &[])?;
        w.call("commit", 20, 0, &[], &[])?;
        w.call(WAIT_COMPLETED, 20, 0, &[], &[])
    });
    let (report, lines, read) = play_then("copy-in", &path, |backend| {
        [backend.read(3), backend.read(4)]
    });
    let [shared, private] = read.map(|bytes| bytes.unwrap_or_else(|e| panic!("{e}: {lines:?}")));
    let mut want = vec![0x11; 32];
    want[4..8].fill(0xa0);
    want[13..19].fill(0xb0);
    assert_eq!(shared, want);
    let mut want = vec![0x5a; 32];
    want[0..4].fill(0xc0);
    want[16..24].fill(0xd0);
    assert_eq!(private, want);
    let executed = &report.total.executed;
    assert_eq!(executed["skipped_guest_ram_writes"], 1);
    assert_eq!(executed["guest_ram_fills"], 2);
    assert_eq!(executed["guest_ram_pieces_in"], 2 + 2);
    // #3: one copy of the words 4..20; #4: 0..4 and 16..24.
    assert_eq!(executed["guest_ram_fill_copies"], 1 + 2);
    assert_eq!(executed["guest_ram_fill_bytes"], 16 + 4 + 8);
}

/// A stream recorded with two copies (`hasUnifiedMemory` false), with Managed #3, a no-copy
/// descriptor buffer, and Managed #5, a copied one, 4 bytes each, made with `[7; 4]` and
/// `[9; 4]` (their descriptors' `contents`) and written so before their first use with no
/// `didModifyRange:`, as a recorder writes every buffer's bytes before its first use; a blit
/// copies each into Shared #4 and #6, which a check reads as the bytes each was made with: the
/// recording device's GPU copy started from them.
pub(super) fn made_with(w: &mut StreamWriter<File>) -> std::io::Result<()> {
    w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
    let identity = [("hasUnifiedMemory", Value::Bool(false))];
    w.call("m2v.deviceIdentity", 1, 0, &[], &identity)?;
    w.call("newCommandQueue", 1, 2, &[], &[])?;
    let managed = Value::List(vec![Value::Map(vec![(
        "ResourceOptions".into(),
        Value::Uint(16),
    )])]);
    let copy_from = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
    let mut sums = Vec::new();
    w.call("commandBuffer", 2, 10, &[], &[])?;
    for (id, no_copy, byte) in [(3, true, 7u8), (5, false, 9)] {
        let bytes = vec![byte; 4];
        let descriptor = Value::Map(vec![
            ("length".into(), Value::Uint(4)),
            ("options".into(), managed.clone()),
            ("contents".into(), Value::Bytes(bytes.clone())),
            ("noCopy".into(), Value::Bool(no_copy)),
        ]);
        w.call("newBufferWithDescriptor:", 1, id, &[descriptor], &[])?;
        let shared = [Value::Uint(4), Value::Uint(0)];
        w.call("newBufferWithLength:options:", 1, id + 1, &shared, &[])?;
        let pairs = [Value::Uint(0), Value::Bytes(bytes.clone())];
        w.call("m2v.write", id, 0, &pairs, &[])?;
        let encoder = 10 + id;
        w.call("blitCommandEncoder", 10, encoder, &[], &[])?;
        let (zero, four) = (Value::Uint(0), Value::Uint(4));
        let copy = [
            Value::Object(id),
            zero.clone(),
            Value::Object(id + 1),
            zero,
            four,
        ];
        w.call(copy_from, encoder, 0, &copy, &[])?;
        w.call("endEncoding", encoder, 0, &[], &[])?;
        sums.push(Value::Bytes(Sha256::digest(&bytes).to_vec()));
    }
    w.call("commit", 10, 0, &[], &[])?;
    w.call(WAIT_COMPLETED, 10, 0, &[], &[])?;
    let answer = [("sha256", Value::List(sums))];
    let checked = [Value::Object(4), Value::Object(6)];
    w.call("m2v.check", 0, 0, &checked, &answer)
}

#[test]
fn a_two_copy_stream_s_managed_buffers_start_from_the_bytes_they_were_made_with() {
    let dir = ScratchDir::new("rt-replay-made-with").unwrap();
    let path = written(&dir, made_with);
    let (report, lines) = play("made-with", &path);
    assert_eq!(report.checks.len(), 2);
    for check in &report.checks {
        let object = check.object;
        assert_eq!(Some(check.sha256), check.recorded, "#{object}'s GPU copy");
    }
    let managed = "managed two copies (the stream's device identity): 2 Managed buffers; 0 \
                   no-copy ones with contents from first use, not from create";
    // A device with unified memory adds that it played the two copies as its one.
    assert!(lines.iter().any(|l| l.starts_with(managed)), "{lines:?}");
}

/// A buffer made as a video surface's pages plays through the surface call: one surface for its
/// two textures over its planes, each picture landed the surface's acquire, and a surface made in
/// place waited for once at its ready point, as the consumer waits before the other API's first
/// write into it. A dma-buf is a Unix host's handle.
#[cfg(unix)]
#[test]
fn a_video_surface_is_waited_for_at_its_ready_point_once() {
    let dir = ScratchDir::new("rt-replay-video").unwrap();
    let path = written(&dir, |w| {
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        // Private pages of a 64 by 64 two-plane 4:2:0 picture.
        let pages = [Value::Uint(6144), Value::Uint(32), Value::Nil];
        let made = "m2v.newDmaBufBufferOfLengthPaddingUnobserved";
        w.call(made, 1, 2, &pages, &[])?;
        let facts = Value::Map(vec![
            (
                "pixelFormat".into(),
                Value::Uint(u64::from(u32::from_be_bytes(*b"420v"))),
            ),
            ("planeCount".into(), Value::Uint(2)),
            ("width".into(), Value::Uint(64)),
            ("height".into(), Value::Uint(64)),
            ("chromaOffset".into(), Value::Uint(4096)),
            ("chromaBytesPerRow".into(), Value::Uint(64)),
            ("ycbcrMatrix".into(), Value::Nil),
        ]);
        let plane = |at: u64, offset: u64| {
            Value::Map(vec![
                ("plane".into(), Value::Uint(at)),
                ("offset".into(), Value::Uint(offset)),
                ("bytesPerRow".into(), Value::Uint(64)),
                ("planar".into(), Value::Bool(true)),
                ("facts".into(), facts.clone()),
            ])
        };
        // Private, shader read and write.
        let texture = |format, side| {
            uints(&[
                ("textureType", 2),
                ("pixelFormat", format),
                ("width", side),
                ("height", side),
                ("usage", 3),
                ("storageMode", 2),
            ])
        };
        let surface = "newTextureWithDescriptor:iosurface:plane:";
        w.call(surface, 2, 3, &[texture(10, 64), plane(0, 0)], &[])?;
        w.call(surface, 2, 4, &[texture(30, 32), plane(1, 4096)], &[])?;
        for _ in 0..2 {
            w.call("m2v.externalWriteDone", 2, 0, &[], &[])?;
            w.call("m2v.present", 1, 0, &[Value::Nil], &[])?;
        }
        Ok(())
    });
    let (report, lines) = play("video-ready", &path);
    let executed = &report.total.executed;
    let in_place = executed["video_surfaces_in_place"];
    assert_eq!(in_place + executed["video_surfaces_copied"], 1, "{lines:?}");
    assert_eq!(executed["video_surface_textures"], 2, "{executed:?}");
    assert_eq!(executed["video_surface_acquires"], 2, "{executed:?}");
    // A wait counts only in place; the copy form's returns at once.
    assert_eq!(executed["surface_waits"], in_place, "{executed:?}");
    eprintln!("REPLAY video-ready in place {in_place}");
}

/// A stream of three render states of the render fixture's functions (#6, #7 and #8), colour 0 of
/// each another pixel format: RGBA8Unorm, BGRA8Unorm and RGBA16Float.
fn three_render_states(dir: &ScratchDir) -> PathBuf {
    written(dir, |w| {
        let library = w.blob(&render_library(dir))?;
        w.call("MTLCreateSystemDefaultDevice", 0, 1, &[], &[])?;
        let library = [Value::Blob(library)];
        w.call("newLibraryWithData:error:", 1, 3, &library, &[])?;
        let vertex = [Value::Str("colored_vertex".into())];
        w.call("newFunctionWithName:", 3, 4, &vertex, &[])?;
        let fragment = [Value::Str("colored_fragment".into())];
        w.call("newFunctionWithName:", 3, 5, &fragment, &[])?;
        for (object, format) in [(6, 70), (7, 80), (8, 115)] {
            let pipeline = Value::Map(vec![
                ("vertex".into(), Value::Object(4)),
                ("fragment".into(), Value::Object(5)),
                ("color0PixelFormat".into(), Value::Uint(format)),
            ]);
            let made = "newRenderPipelineStateWithDescriptor:error:";
            w.call(made, 1, object, &[pipeline], &[])?;
        }
        Ok(())
    })
}

/// `path` played states-only on a device of an empty cache directory, its makes written as
/// `with_states(log, threads, keys, digests)` writes them: the state log's lines, the backend's
/// lines, and the names of the records the runtime wrote under `renders/` by the device's end.
fn states_play(
    test: &str,
    path: &Path,
    threads: usize,
    keys: Option<HashSet<String>>,
    digests: bool,
) -> (Vec<String>, Vec<String>, BTreeSet<String>) {
    let dir = ScratchDir::new(&format!("rt-replay-{test}-log")).unwrap();
    let log = dir.path().join("states.log");
    let cache = ScratchCache::new().unwrap();
    let backend = NewRuntime::new(rt_device(cache.path()), gpu::TEST_DEVICE).unwrap();
    let log_file = Some(File::create(&log).unwrap());
    let mut backend = backend.with_states(log_file, threads, keys, digests);
    let mut reader = StreamReader::open(&[path.to_path_buf()], None).unwrap();
    let (report, ended) = Player::new(&mut backend).states_only().play(&mut reader);
    let lines = backend.lines();
    for line in report.lines().iter().chain(&lines) {
        eprintln!("REPLAY {test} {line}");
    }
    ended.unwrap();
    backend.end().result().unwrap();
    let logged: Vec<String> = (std::fs::read_to_string(&log).unwrap().lines())
        .map(str::to_owned)
        .collect();
    let mut records = BTreeSet::new();
    for generation in std::fs::read_dir(cache.path()).unwrap() {
        let renders = generation.unwrap().path().join("renders");
        for record in std::fs::read_dir(&renders).into_iter().flatten() {
            records.insert(record.unwrap().file_name().to_string_lossy().into_owned());
        }
    }
    (logged, lines, records)
}

/// The log's lines of `kind` (`begin`, `pass`, ...).
fn logged<'a>(lines: &'a [String], kind: &str) -> Vec<&'a String> {
    let prefix = format!("M2V-STATE {kind} ");
    lines.iter().filter(|l| l.starts_with(&prefix)).collect()
}

/// The content key a begin or pass line names.
fn key_of(line: &str) -> String {
    let after = line.split(" key ").nth(1).expect("a key");
    after.split(' ').next().unwrap().to_owned()
}

#[test]
fn a_states_only_play_on_its_threads_writes_each_make_between_its_lines() {
    let dir = ScratchDir::new("rt-replay-states-pool").unwrap();
    let path = three_render_states(&dir);
    let (lines, backend, records) = states_play("states-pool", &path, 2, None, false);
    let begins = logged(&lines, "begin");
    assert_eq!(begins.len(), 3, "{lines:?}");
    for begin in &begins {
        // `M2V-STATE begin SEQ THREAD ...`: the make's sequence number and its pool thread.
        let made: Vec<&str> = begin.split(' ').skip(2).take(2).collect();
        let made = made.join(" ");
        assert!(made.contains(" m2v-state-"), "{begin}");
        let at = |prefix: &str| lines.iter().position(|l| l.starts_with(prefix));
        let begun = at(begin).unwrap();
        let modules = at(&format!("M2V-STATE modules {made}")).expect("its modules line");
        let ended = at(&format!("M2V-STATE end {made} made in ")).expect("its end, made");
        assert!(begun < modules && modules < ended, "{lines:?}");
    }
    let keys: BTreeSet<String> = begins.iter().map(|l| key_of(l)).collect();
    assert_eq!(keys.len(), 3, "{lines:?}");
    assert_eq!(keys, records, "each begin's key names its content's record");
    let states = "states begun 3 made 3 refused 0 failed 0 passed over 0 on 2 threads";
    assert!(backend.iter().any(|l| l == states), "{backend:?}");
}

#[test]
fn a_states_only_play_of_listed_keys_makes_those_states_alone() {
    let dir = ScratchDir::new("rt-replay-states-keys").unwrap();
    let path = three_render_states(&dir);
    let (all, ..) = states_play("states-keys-all", &path, 0, None, false);
    let begins = logged(&all, "begin");
    assert_eq!(begins.len(), 3, "{all:?}");
    assert!(
        begins.iter().all(|l| l.split(' ').nth(3) == Some("play")),
        "{all:?}"
    );
    // The BGRA8Unorm state's content, #7.
    let listed = (begins.iter()).find(|l| l.contains(" render #7 ")).unwrap();
    let key = key_of(listed);
    let (lines, backend, records) = states_play(
        "states-keys-one",
        &path,
        0,
        Some(HashSet::from([key.clone()])),
        false,
    );
    let made = logged(&lines, "begin");
    assert_eq!(made.len(), 1, "{lines:?}");
    assert_eq!(key_of(made[0]), key);
    let passed = logged(&lines, "pass");
    assert_eq!(passed.len(), 2, "{lines:?}");
    assert!(passed.iter().all(|l| !l.contains(&key)), "{lines:?}");
    assert_eq!(records, BTreeSet::from([key]));
    let states = "states begun 1 made 1 refused 0 failed 0 passed over 2 on the play's thread";
    assert!(backend.iter().any(|l| l == states), "{backend:?}");
}

/// The content digest each modules line of `lines` ends in.
fn digests_of(lines: &[String]) -> BTreeSet<String> {
    (logged(lines, "modules").iter())
        .map(|l| l.split(" digest ").nth(1).expect("a digest").to_owned())
        .collect()
}

#[test]
fn a_states_only_play_writes_each_render_state_s_content_digest() {
    let dir = ScratchDir::new("rt-replay-states-digests").unwrap();
    let path = three_render_states(&dir);
    let (lines, ..) = states_play("states-digests", &path, 0, None, true);
    assert_eq!(logged(&lines, "modules").len(), 3, "{lines:?}");
    let digests = digests_of(&lines);
    let hex = |d: &String| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit());
    assert!(digests.iter().all(hex), "{lines:?}");
    assert_eq!(digests.len(), 3, "three contents, three digests: {lines:?}");
    // A play of its own, its makes on two threads, writes the same digests.
    let (pooled, ..) = states_play("states-digests-pooled", &path, 2, None, true);
    assert_eq!(digests_of(&pooled), digests, "{pooled:?}");
    let (plain, ..) = states_play("states-digests-off", &path, 0, None, false);
    let modules = logged(&plain, "modules");
    assert!(
        modules.len() == 3 && modules.iter().all(|l| !l.contains(" digest ")),
        "{plain:?}"
    );
}
