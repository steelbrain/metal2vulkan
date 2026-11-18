//! Device tests of the replay player on the new runtime (`replay::NewRuntime`), on the device the
//! cfg selects: streams the tests write (`docs/REPLAY-STREAM.md`), each played and its report
//! printed as `REPLAY <test> <line>`.
//!
//! - A draw of a render pipeline state, its vertices in buffers bound to the vertex stage, covers
//!   a pass's target that its setter records clear to red: a blit copies the target to a buffer,
//!   which the stream checks green, and nothing is skipped.
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
    let path = dir.path().join("stream.m2vr");
    let mut writer = StreamWriter::new(File::create(&path).unwrap()).unwrap();
    build(&mut writer).unwrap();
    writer.finish().unwrap();
    path
}

/// `path` played on the new runtime's device, its report printed; the play's end unwrapped.
fn play(test: &str, path: &Path) -> Report {
    let cache = ScratchCache::new().unwrap();
    let mut backend = NewRuntime::new(rt_device(cache.path())).unwrap();
    let mut reader = StreamReader::open(&[path.to_path_buf()], None).unwrap();
    let (report, ended) = Player::new(&mut backend).play(&mut reader);
    for line in report.lines().into_iter().chain(backend.lines()) {
        eprintln!("REPLAY {test} {line}");
    }
    ended.unwrap();
    report
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

/// The render library's `colored_vertex` (a vertex function of positions and colours by vertex
/// index, from buffers 0 and 1) and `colored_fragment` (its colour), in a metallib.
fn render_library() -> Vec<u8> {
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let bitcode = wrap(&metal2vulkan_devtools::llvm_assemble(&ll).unwrap());
    let function = |name, type_code| MtlbFunction {
        name,
        type_code,
        bitcode: bitcode.clone(),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    };
    metallib(&[
        function("colored_vertex", 0),
        function("colored_fragment", 1),
    ])
}

#[test]
fn a_draw_into_a_pass_plays_on_the_new_runtime() {
    let dir = ScratchDir::new("rt-replay-draw").unwrap();
    let green: Vec<u8> = [0u8, 255, 0, 255].repeat(16);
    let path = written(&dir, |w| {
        let library = w.blob(&render_library())?;
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
        let sum = Value::List(vec![Value::Bytes(Sha256::digest(&green).to_vec())]);
        w.call("m2v.check", 0, 0, &[Value::Object(10)], &[("sha256", sum)])
    });
    let report = play("draw", &path);
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
}
