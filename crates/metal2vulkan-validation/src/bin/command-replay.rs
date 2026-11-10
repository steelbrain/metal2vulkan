//! The command replay harness's runtime leg: run one recorded stretch of Metal commands (a
//! render pass's draws and tile dispatches, compute dispatches, across command buffers) on a
//! Vulkan device through the runtime, from a manifest that names every object, every byte the
//! commands start from and every call in order, and write what they leave behind, so the same
//! manifest run on Metal (the Metal leg, a Swift program kept beside the probes) and on each
//! Vulkan device can be compared byte for byte.
//!
//! `command-replay run --manifest FILE --out DIR --device-type cpu|gpu --shader-cache DIR`
//!
//! **The manifest** (version 1), one JSON object; every Metal enum is its raw number, files are
//! relative to the manifest's directory, and each table's ids are free strings:
//!
//! - `library`: the metallib a function comes from unless it names another; `libraries`?: `{LID:
//!   file}`, other metallibs. `functions`: `{FID: {"name", "library"?: LID, "constants": [{"index",
//!   "type" (MTLDataType), "hex"}]}}`.
//! - `heaps`: `{HID: {"size", "type" (MTLHeapType), "options" (MTLResourceOptions)}}`.
//! - `buffers`: `{BID: {"length", "options", "file"?, "heap"?, "offset"?}}` (`offset` places it in
//!   a placement heap).
//! - `textures`: `{TID: {"type", "format", "width", "height", "depth", "levels", "array",
//!   "samples", "usage", "options", "heap"?, "offset"?, "file"?}}`; the file holds level 0 of each
//!   slice in turn, rows packed (`width` texels each), copied in through a blit (a two-plane
//!   format's plane 0 only). A two-plane YCbCr texture over a surface takes `"surface": {"ostype"
//!   (four characters), "file", "lumaRow", "chromaOffset", "chromaRow", "matrix"?: "601" | "709" |
//!   "2020" (none: the surface has no `kIOSurfaceYCbCrMatrix`)}` instead: the file holds the
//!   surface's luma plane at byte 0, rows `lumaRow` bytes apart, and its chroma plane at
//!   `chromaOffset`, rows `chromaRow` apart; the texture is made over its plane 0
//!   (`newTextureWithDescriptor:iosurface:plane:`).
//! - `views`: `{TID: {"of" (a texture), "format", "type"?, "levels"?: [first, count], "slices"?:
//!   [first, count], "swizzle"?: [r, g, b, a] (MTLTextureSwizzle)}}`: views made after every
//!   texture, in the textures' namespace (`newTextureViewWithPixelFormat:textureType:levels:
//!   slices:swizzle:`, or `newTextureViewWithPixelFormat:` with no other key).
//! - `samplers`: `{SID: {"min", "mag", "mip", "s", "t", "r", "normalized", "border", "lodMin",
//!   "lodMax", "maxAniso", "compare"}}`.
//! - `renderPipelines`: `{PID: {"vertex", "fragment" (or null), "colors": [{"format",
//!   "blending", "srcRGB", "dstRGB", "rgbOp", "srcA", "dstA", "alphaOp", "mask"} or null],
//!   "depth", "stencil" (0 for none), "samples", "alphaToCoverage", "topology"}}`;
//!   `tilePipelines`: `{PID: {"function", "colors": [format, ...], "matchesTileSize"}}`;
//!   `computePipelines`: `{PID: {"function"}}`; `depthStencilStates`: `{DID: {"compare",
//!   "write", "stencil"?: {"compare", "fail", "depthFail", "pass", "read", "write"}}}` (the
//!   stencil descriptor both faces take).
//! - `commandBuffers`: a list of command buffers, each a list of encoders, committed in order and
//!   each waited on: `{"render": {"colors": [{"texture", "load", "store", "clear": [r, g, b, a]}
//!   or null], "depth"?, "stencil"?, "tileWidth"?, "tileHeight"?, "imageblockSize"?,
//!   "threadgroupMemoryLength"?, "width"?, "height"?, "commands": [...]}}` or `{"compute": {"commands": [...]}}`;
//!   a render pass's `defaultRasterSampleCount` is 1 (a pass without attachments rasterizes one
//!   sample; Metal ends the process at the default 0).
//! - The commands, applied in order, each one of the encoder's own calls: `{"pipeline": PID}`;
//!   `{"buffer" | "bytes" | "texture" | "sampler": {"stage": "vertex" | "fragment" | "tile" |
//!   "kernel", "index", ...}}`; `{"threadgroupMemory": {"index", "length"}}`; `{"depthStencil":
//!   DID}`; `{"stencilRef": n}`; `{"blendColor": [r, g, b, a]}`; `{"viewport": [x, y, w, h,
//!   znear, zfar]}`; `{"scissor": [x, y, w, h]}`; `{"cull": n}`; `{"winding": n}`;
//!   `{"useResource": {"id", "usage", "stages"}}`; `{"useHeap": HID}`; `{"draw": {"primitive",
//!   "start", "count", "instances", "baseInstance"}}`; `{"drawIndexed": {"primitive", "count",
//!   "indexType", "buffer", "offset", "instances", "baseVertex", "baseInstance"}}`; `{"tile":
//!   {"threads": [w, h]}}`; `{"dispatchThreadgroups": {"groups", "threads"}}`;
//!   `{"dispatchThreads": {"grid", "threads"}}`.
//! - `readback`: the ids written out, each to `DIR/<id>.bin`: a texture's level 0, every slice
//!   in turn, rows packed, through a blit into a shared buffer (a two-plane format's plane 0; a
//!   memoryless or multisampled texture is not read), a buffer's whole length.
//!
//! `DIR/status.txt` says `leg runtime`, `device <name>`, what the run's pipelines took from the
//! shader cache (`cache` and its counts: functions translated, variant pipelines made, directory
//! lookups hit and missed, driver records loaded, and the driver's pipeline cache hits and misses
//! its creation feedback reported), each pipeline creation's line where
//! `METAL2VULKAN_LOG_PIPELINES` is set (`pipeline <line>`), each degrade notice the run raised
//! (`notice <code>`), and `ok` or `error <where>: <message>`: the first failure stops the run,
//! and the readbacks done so far stay. The exit status is 0 for `ok`, 1 for a run that stopped,
//! and 2 for a harness failure (a bad manifest or command line).
//!
//! GUEST DATA: a manifest made from a guest's commands, its files and the outputs are the
//! consumer's: they stay on the hosts the guest data rule allows and are never committed.

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, CompareFunction, CullMode, DataType, IndexType,
    PixelFormat, PrimitiveTopologyClass, PrimitiveType, RenderStages, ResourceUsage,
    SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
    StencilOperation, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage, Winding,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputeCommandEncoder,
    ComputePipelineState, DepthStencilState, Device, DeviceOptions, Function, Heap, HeapDescriptor,
    HeapType, IoSurfacePlane, IoSurfaceYCbCr, Library, LoadAction, MessageSink, Origin,
    PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ResourceRef, SamplerState, ScissorRect, ShaderCacheOptions, Size, StoreAction,
    SurfaceYCbCrMatrix, Texture, TextureDescriptor, Viewport,
};
use metal2vulkan_validation::candidate::runtime_case::open_device;
use metal2vulkan_validation::gpu::DeviceType;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const USAGE: &str = "usage: command-replay run --manifest FILE --out DIR --device-type cpu|gpu \
                     --shader-cache DIR";

/// How long a command buffer may run.
const WAIT: Duration = Duration::from_secs(300);

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("run") => match options(&args[1..]) {
            Ok(options) => run(&options),
            Err(why) => {
                eprintln!("command-replay: {why}\n{USAGE}");
                2
            }
        },
        _ => {
            eprintln!("{USAGE}");
            2
        }
    };
    std::process::exit(code);
}

fn options(args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let name = flag
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected {flag:?}"))?;
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        out.insert(name.to_string(), value.clone());
    }
    for name in ["manifest", "out", "device-type", "shader-cache"] {
        if !out.contains_key(name) {
            return Err(format!("--{name} is required"));
        }
    }
    Ok(out)
}

/// The status file, written as the run goes.
struct Status {
    file: std::fs::File,
}

impl Status {
    fn line(&mut self, line: &str) {
        let _ = writeln!(self.file, "{line}");
        let _ = self.file.flush();
    }
}

fn run(options: &BTreeMap<String, String>) -> i32 {
    let manifest_path = PathBuf::from(&options["manifest"]);
    let out = PathBuf::from(&options["out"]);
    if let Err(error) = std::fs::create_dir_all(&out) {
        eprintln!("command-replay: {}: {error}", out.display());
        return 2;
    }
    let file = match std::fs::File::create(out.join("status.txt")) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("command-replay: status.txt: {error}");
            return 2;
        }
    };
    let mut status = Status { file };
    status.line("leg runtime");
    let manifest: Value = match std::fs::read(&manifest_path)
        .map_err(|e| e.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|e| e.to_string()))
    {
        Ok(v) => v,
        Err(why) => {
            status.line(&format!("error manifest: {why}"));
            return 2;
        }
    };
    let dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let device_type = match DeviceType::parse(&options["device-type"]) {
        Ok(t) => t,
        Err(why) => {
            status.line(&format!("error device-type: {why}"));
            return 2;
        }
    };
    let notices: Arc<Mutex<Vec<String>>> = Arc::default();
    let mut device_options = DeviceOptions::default();
    device_options.shader_cache = ShaderCacheOptions::in_directory(&options["shader-cache"]);
    let sink = Arc::clone(&notices);
    device_options.messages = Some(MessageSink::new(move |m| {
        if m.id_name.starts_with("metal2vulkan-degrade") || m.id_name.starts_with("VUID") {
            if let Ok(mut n) = sink.lock() {
                n.push(format!("notice {} {}", m.id_name, m.text));
            }
        } else if m.id_name == "metal2vulkan-pipeline-creation" {
            if let Ok(mut n) = sink.lock() {
                n.push(format!("pipeline {}", m.text));
            }
        }
    }));
    device_options.announce_each = metal2vulkan_runtime::Degradation::ALL
        .iter()
        .copied()
        .collect();
    let device = match open_device(device_type, device_options) {
        Ok(d) => d,
        Err(error) => {
            status.line(&format!("error device: {error}"));
            return 1;
        }
    };
    status.line(&format!("device {}", device.name()));
    let mut replay = Replay {
        dir,
        out,
        device,
        queue: None,
        libraries: BTreeMap::new(),
        functions: BTreeMap::new(),
        heaps: BTreeMap::new(),
        buffers: BTreeMap::new(),
        textures: BTreeMap::new(),
        samplers: BTreeMap::new(),
        render: BTreeMap::new(),
        compute: BTreeMap::new(),
        depth: BTreeMap::new(),
    };
    let result = replay.all(&manifest, &mut status);
    let read = replay.readback(&manifest, &mut status);
    let (stats, counts) = (
        replay.device.shader_cache_statistics(),
        replay.device.work_counts(),
    );
    status.line(&format!(
        "cache translated {} variant-pipelines {} lookups-hit {} lookups-missed {} driver-loads {} \
         driver-pipeline-hits {} driver-pipeline-misses {}",
        counts.functions_translated,
        counts.variant_pipelines_made,
        stats.hits,
        stats.misses,
        stats.driver_loads,
        stats.driver_pipeline_hits,
        stats.driver_pipeline_misses
    ));
    for line in notices.lock().map(|n| n.clone()).unwrap_or_default() {
        status.line(&line);
    }
    match result.and(read) {
        Ok(()) => {
            status.line("ok");
            0
        }
        Err(why) => {
            status.line(&format!("error {why}"));
            1
        }
    }
}

/// The objects a run made, by their manifest ids.
struct Replay {
    dir: PathBuf,
    out: PathBuf,
    device: Device,
    queue: Option<CommandQueue>,
    /// The manifest's libraries: its `library` under the empty id, and its `libraries`.
    libraries: BTreeMap<String, Library>,
    functions: BTreeMap<String, Function>,
    heaps: BTreeMap<String, Heap>,
    buffers: BTreeMap<String, Buffer>,
    textures: BTreeMap<String, (Texture, TextureDescriptor)>,
    samplers: BTreeMap<String, SamplerState>,
    render: BTreeMap<String, RenderPipelineState>,
    compute: BTreeMap<String, ComputePipelineState>,
    depth: BTreeMap<String, DepthStencilState>,
}

// ---- Manifest access ----

fn table(v: &Value, key: &str) -> Vec<(String, Value)> {
    v.get(key)
        .and_then(Value::as_object)
        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

fn u(v: &Value, key: &str, default: u64) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn f(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn b(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn s<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} is missing"))
}

fn nums(v: &Value, key: &str) -> Vec<f64> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn size3(v: &Value, key: &str) -> Result<Size, String> {
    let n = nums(v, key);
    let at = |k: usize| n.get(k).copied().unwrap_or(1.0) as u64;
    match n.is_empty() {
        true => Err(format!("{key} is missing")),
        false => Ok(Size::new(at(0), at(1), at(2))),
    }
}

/// A Metal enum of raw value `key` (`default` where absent).
fn raw<T: TryFrom<u32>>(v: &Value, key: &str, default: u32) -> Result<T, String>
where
    T::Error: std::fmt::Display,
{
    let r = u32::try_from(u(v, key, u64::from(default))).map_err(|e| e.to_string())?;
    T::try_from(r).map_err(|e| format!("{key} {r}: {e}"))
}

fn hex(text: &str) -> Result<Vec<u8>, String> {
    let text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if !text.len().is_multiple_of(2) {
        return Err("odd hex length".into());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

/// The bytes of one texel of `format` (plane 0's for a two-plane format).
fn texel_bytes(format: PixelFormat) -> Option<u64> {
    use PixelFormat as P;
    Some(match format {
        P::A8Unorm | P::R8Unorm | P::R8UnormSRGB | P::R8Snorm | P::R8Uint | P::R8Sint => 1,
        P::R16Unorm
        | P::R16Snorm
        | P::R16Uint
        | P::R16Sint
        | P::R16Float
        | P::RG8Unorm
        | P::RG8UnormSRGB
        | P::RG8Snorm
        | P::RG8Uint
        | P::RG8Sint
        | P::B5G6R5Unorm
        | P::A1BGR5Unorm
        | P::ABGR4Unorm
        | P::BGR5A1Unorm => 2,
        P::R32Uint
        | P::R32Sint
        | P::R32Float
        | P::RG16Unorm
        | P::RG16Snorm
        | P::RG16Uint
        | P::RG16Sint
        | P::RG16Float
        | P::RGBA8Unorm
        | P::RGBA8UnormSRGB
        | P::RGBA8Snorm
        | P::RGBA8Uint
        | P::RGBA8Sint
        | P::BGRA8Unorm
        | P::BGRA8UnormSRGB
        | P::ARGB8Unorm
        | P::RGB10A2Unorm
        | P::RGB10A2Uint
        | P::RG11B10Float
        | P::RGB9E5Float
        | P::BGR10A2Unorm
        | P::BGR10XR
        | P::BGR10XRSRGB
        | P::RGB10A82PXR10
        | P::RGB10A82PXR10SRGB
        | P::Depth32Float => 4,
        P::RG32Uint
        | P::RG32Sint
        | P::RG32Float
        | P::RGBA16Unorm
        | P::RGBA16Snorm
        | P::RGBA16Uint
        | P::RGBA16Sint
        | P::RGBA16Float
        | P::BGRA10XR
        | P::BGRA10XRSRGB => 8,
        P::RGBA32Uint | P::RGBA32Sint | P::RGBA32Float => 16,
        _ => return None,
    })
}

fn e<E: std::fmt::Display>(at: &str) -> impl Fn(E) -> String + '_ {
    move |error| format!("{at}: {error}")
}

impl Replay {
    fn all(&mut self, m: &Value, status: &mut Status) -> Result<(), String> {
        self.queue = Some(self.device.new_command_queue().map_err(e("queue"))?);
        let named = m
            .get("library")
            .and_then(Value::as_str)
            .map(|file| (String::new(), Value::from(file)));
        for (id, file) in named.into_iter().chain(table(m, "libraries")) {
            let at = format!("library {id:?}");
            let file = file
                .as_str()
                .ok_or_else(|| format!("{at}: not a file name"))?;
            let bytes = std::fs::read(self.dir.join(file)).map_err(e(&at))?;
            let library = self.device.new_library_with_data(&bytes).map_err(e(&at))?;
            self.libraries.insert(id, library);
        }
        self.functions(m)?;
        self.heaps(m)?;
        self.buffers(m)?;
        self.textures(m)?;
        self.views(m)?;
        self.samplers(m)?;
        self.pipelines(m)?;
        let buffers = m
            .get("commandBuffers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for (k, encoders) in buffers.iter().enumerate() {
            let at = format!("command buffer {k}");
            let cb = self.queue().command_buffer().map_err(e(&at))?;
            for (j, encoder) in encoders.as_array().into_iter().flatten().enumerate() {
                let at = format!("command buffer {k} encoder {j}");
                if let Some(r) = encoder.get("render") {
                    self.render_encoder(&cb, r)
                        .map_err(|w| format!("{at}: {w}"))?;
                } else if let Some(c) = encoder.get("compute") {
                    self.compute_encoder(&cb, c)
                        .map_err(|w| format!("{at}: {w}"))?;
                } else {
                    return Err(format!("{at}: neither render nor compute"));
                }
            }
            finish(&cb).map_err(|w| format!("{at}: {w}"))?;
            status.line(&format!("committed {k}"));
        }
        Ok(())
    }

    fn queue(&self) -> &CommandQueue {
        self.queue.as_ref().expect("the queue is made first")
    }

    fn functions(&mut self, m: &Value) -> Result<(), String> {
        for (id, fv) in table(m, "functions") {
            let name = s(&fv, "name")?;
            let lid = fv.get("library").and_then(Value::as_str).unwrap_or("");
            let library = self
                .libraries
                .get(lid)
                .ok_or_else(|| format!("function {id}: no library {lid:?}"))?;
            let mut constants = FunctionConstantValues::defaults();
            for c in fv
                .get("constants")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let ty: DataType = raw(c, "type", 0)?;
                let bytes = hex(s(c, "hex")?)?;
                let index = u32::try_from(u(c, "index", 0)).map_err(|e| e.to_string())?;
                constants = constants.with_index(index, FunctionConstantValue::new(ty, bytes));
            }
            let function = match constants.is_empty() {
                true => library.new_function_with_name(name),
                false => library.new_function_with_name_constant_values(name, &constants),
            }
            .map_err(|error| format!("function {id} ({name}): {error}"))?;
            self.functions.insert(id, function);
        }
        Ok(())
    }

    fn function(&self, id: &str) -> Result<StageSource, String> {
        let f = self
            .functions
            .get(id)
            .ok_or_else(|| format!("no function {id}"))?;
        Ok(StageSource::Function(f.function_descriptor()))
    }

    fn heaps(&mut self, m: &Value) -> Result<(), String> {
        for (id, hv) in table(m, "heaps") {
            let options =
                ResourceOptions::from_raw(u(&hv, "options", 0)).map_err(e("heap options"))?;
            let heap_type = HeapType::from_raw(u(&hv, "type", 0)).map_err(e("heap type"))?;
            let descriptor = HeapDescriptor::new(heap_type, u(&hv, "size", 0), options);
            let heap = self
                .device
                .new_heap(&descriptor)
                .map_err(|error| format!("heap {id}: {error}"))?;
            self.heaps.insert(id, heap);
        }
        Ok(())
    }

    fn buffers(&mut self, m: &Value) -> Result<(), String> {
        let mut uploads = Vec::new();
        for (id, bv) in table(m, "buffers") {
            let at = format!("buffer {id}");
            let length = u(&bv, "length", 0);
            let options = ResourceOptions::from_raw(u(&bv, "options", 0)).map_err(e(&at))?;
            let buffer = match bv.get("heap").and_then(Value::as_str) {
                Some(h) => {
                    let heap = self
                        .heaps
                        .get(h)
                        .ok_or_else(|| format!("{at}: no heap {h}"))?;
                    match bv.get("offset").and_then(Value::as_u64) {
                        Some(offset) => heap.new_buffer_at_offset(length, options, offset),
                        None => heap.new_buffer(length, options),
                    }
                }
                None => self.device.new_buffer(length, options),
            }
            .map_err(e(&at))?;
            if let Some(file) = bv.get("file").and_then(Value::as_str) {
                let bytes = std::fs::read(self.dir.join(file)).map_err(e(&at))?;
                match buffer.contents() {
                    Some(at_ptr) => {
                        let n = bytes.len().min(length as usize);
                        // SAFETY: a host-visible buffer of `length` bytes no command uses yet.
                        unsafe {
                            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at_ptr.as_ptr(), n);
                        }
                    }
                    None => uploads.push((buffer.clone(), bytes)),
                }
            }
            self.buffers.insert(id, buffer);
        }
        if uploads.is_empty() {
            return Ok(());
        }
        let cb = self.queue().command_buffer().map_err(e("buffer uploads"))?;
        let mut staging = Vec::new();
        let mut blit = cb.blit_command_encoder().map_err(e("buffer uploads"))?;
        for (buffer, bytes) in &uploads {
            let shared = self
                .device
                .new_buffer_with_bytes(bytes, ResourceOptions::shared())
                .map_err(e("buffer uploads"))?;
            let n = (bytes.len() as u64).min(buffer.length());
            blit.copy_from_buffer_to_buffer(&shared, 0, buffer, 0, n)
                .map_err(e("buffer uploads"))?;
            staging.push(shared);
        }
        blit.end_encoding().map_err(e("buffer uploads"))?;
        finish(&cb).map_err(|w| format!("buffer uploads: {w}"))
    }

    fn views(&mut self, m: &Value) -> Result<(), String> {
        for (id, v) in table(m, "views") {
            let at = format!("view {id}");
            let (of, of_d) = self
                .textures
                .get(s(&v, "of")?)
                .cloned()
                .ok_or(format!("{at}: no texture"))?;
            let format: PixelFormat = raw(&v, "format", 0)?;
            let pair = |key: &str, all: u64| -> std::ops::Range<u64> {
                let p: Vec<u64> = v
                    .get(key)
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default();
                match p.as_slice() {
                    [first, count] => *first..first + count,
                    _ => 0..all,
                }
            };
            let whole = ["type", "levels", "slices", "swizzle"]
                .iter()
                .all(|k| v.get(*k).is_none());
            let view = if whole {
                of.new_texture_view(format)
            } else {
                let texture_type = raw::<TextureType>(&v, "type", of_d.texture_type.as_raw())?;
                let levels = pair("levels", of_d.mipmap_level_count);
                let slices = pair(
                    "slices",
                    of_d.array_length.max(1)
                        * match of_d.texture_type {
                            TextureType::Cube | TextureType::CubeArray => 6,
                            _ => 1,
                        },
                );
                let channel = |at: usize, own: u32| -> Result<TextureSwizzle, String> {
                    let r = v
                        .get("swizzle")
                        .and_then(|s| s.get(at))
                        .and_then(Value::as_u64)
                        .map_or(own, |r| r as u32);
                    TextureSwizzle::try_from(r).map_err(|e| format!("swizzle {r}: {e}"))
                };
                let swizzle = TextureSwizzleChannels {
                    red: channel(0, 2)?,
                    green: channel(1, 3)?,
                    blue: channel(2, 4)?,
                    alpha: channel(3, 5)?,
                };
                of.new_texture_view_with_swizzle(format, texture_type, levels, slices, swizzle)
            }
            .map_err(e(&at))?;
            let mut d = of_d;
            d.pixel_format = format;
            self.textures.insert(id, (view, d));
        }
        Ok(())
    }

    fn textures(&mut self, m: &Value) -> Result<(), String> {
        let mut uploads = Vec::new();
        for (id, tv) in table(m, "textures") {
            let at = format!("texture {id}");
            let format: PixelFormat = raw(&tv, "format", 0)?;
            let mut d = TextureDescriptor::texture_2d(
                format,
                u(&tv, "width", 1),
                u(&tv, "height", 1),
                false,
            );
            d.texture_type = raw::<TextureType>(&tv, "type", 2)?;
            d.depth = u(&tv, "depth", 1);
            d.mipmap_level_count = u(&tv, "levels", 1);
            d.array_length = u(&tv, "array", 1);
            d.sample_count = u(&tv, "samples", 1);
            d.usage = raw::<TextureUsage>(&tv, "usage", 1)?;
            d.resource_options = ResourceOptions::from_raw(u(&tv, "options", 0)).map_err(e(&at))?;
            if let Some(sv) = tv.get("surface") {
                let (buffer, texture) = self.surface_texture(&at, &d, sv)?;
                self.buffers.insert(format!("surface:{id}"), buffer);
                self.textures.insert(id, (texture, d));
                continue;
            }
            let texture = match tv.get("heap").and_then(Value::as_str) {
                Some(h) => {
                    let heap = self
                        .heaps
                        .get(h)
                        .ok_or_else(|| format!("{at}: no heap {h}"))?;
                    match tv.get("offset").and_then(Value::as_u64) {
                        Some(offset) => heap.new_texture_at_offset(&d, offset),
                        None => heap.new_texture(&d),
                    }
                }
                None => self.device.new_texture(&d),
            }
            .map_err(e(&at))?;
            if let Some(file) = tv.get("file").and_then(Value::as_str) {
                let bytes = std::fs::read(self.dir.join(file)).map_err(e(&at))?;
                uploads.push((texture.clone(), d, bytes));
            }
            self.textures.insert(id, (texture, d));
        }
        if uploads.is_empty() {
            return Ok(());
        }
        let cb = self
            .queue()
            .command_buffer()
            .map_err(e("texture uploads"))?;
        let mut blit = cb.blit_command_encoder().map_err(e("texture uploads"))?;
        let mut staging = Vec::new();
        for (texture, d, bytes) in &uploads {
            let texel = texel_bytes(d.pixel_format)
                .ok_or_else(|| format!("upload: no texel size for {:?}", d.pixel_format))?;
            let row = texel * d.width;
            let image = row * d.height;
            let slices = d.array_length.max(1)
                * match d.texture_type {
                    TextureType::Cube | TextureType::CubeArray => 6,
                    _ => 1,
                };
            let shared = self
                .device
                .new_buffer_with_bytes(bytes, ResourceOptions::shared())
                .map_err(e("texture uploads"))?;
            for slice in 0..slices {
                if (slice + 1) * image > bytes.len() as u64 {
                    break;
                }
                blit.copy_from_buffer_to_texture(
                    &shared,
                    slice * image,
                    row,
                    image,
                    Size::new(d.width, d.height, d.depth.max(1)),
                    texture,
                    slice,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )
                .map_err(e("texture uploads"))?;
            }
            staging.push(shared);
        }
        blit.end_encoding().map_err(e("texture uploads"))?;
        finish(&cb).map_err(|w| format!("texture uploads: {w}"))
    }

    /// The surface `sv` describes (the manifest's `surface`) as a shared buffer of its file's
    /// bytes, and the texture of `d` over its plane 0 with its YCbCr facts.
    fn surface_texture(
        &self,
        at: &str,
        d: &TextureDescriptor,
        sv: &Value,
    ) -> Result<(Buffer, Texture), String> {
        let ostype: [u8; 4] = s(sv, "ostype")?
            .as_bytes()
            .try_into()
            .map_err(|_| format!("{at}: an ostype is four characters"))?;
        let bytes = std::fs::read(self.dir.join(s(sv, "file")?)).map_err(e(at))?;
        let buffer = self
            .device
            .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
            .map_err(e(at))?;
        let facts = IoSurfaceYCbCr::new(
            u32::from_be_bytes(ostype),
            0,
            [d.width, d.height],
            u(sv, "chromaOffset", 0),
            u(sv, "chromaRow", 0),
        );
        let facts = match sv.get("matrix").and_then(Value::as_str) {
            None => facts,
            Some("601") => facts.with_matrix(SurfaceYCbCrMatrix::Itu601),
            Some("709") => facts.with_matrix(SurfaceYCbCrMatrix::Itu709),
            Some("2020") => facts.with_matrix(SurfaceYCbCrMatrix::Itu2020),
            Some(other) => return Err(format!("{at}: no matrix {other:?}")),
        };
        let plane = IoSurfacePlane::new(0, u(sv, "lumaRow", 0), true).with_ycbcr(facts);
        let texture = buffer
            .new_texture_with_iosurface_plane(d, plane)
            .map_err(e(at))?;
        Ok((buffer, texture))
    }

    fn samplers(&mut self, m: &Value) -> Result<(), String> {
        for (id, sv) in table(m, "samplers") {
            let mut d = SamplerDescriptor::default();
            d.min_filter = raw::<SamplerMinMagFilter>(&sv, "min", 0)?;
            d.mag_filter = raw::<SamplerMinMagFilter>(&sv, "mag", 0)?;
            d.mip_filter = raw::<SamplerMipFilter>(&sv, "mip", 0)?;
            d.s_address_mode = raw::<SamplerAddressMode>(&sv, "s", 0)?;
            d.t_address_mode = raw::<SamplerAddressMode>(&sv, "t", 0)?;
            d.r_address_mode = raw::<SamplerAddressMode>(&sv, "r", 0)?;
            d.border_color = raw::<SamplerBorderColor>(&sv, "border", 0)?;
            d.normalized_coordinates = b(&sv, "normalized", true);
            d.lod_min_clamp = f(&sv, "lodMin", 0.0) as f32;
            d.lod_max_clamp = f(&sv, "lodMax", f64::from(f32::MAX)) as f32;
            d.max_anisotropy = u(&sv, "maxAniso", 1) as u32;
            d.compare_function = raw::<CompareFunction>(&sv, "compare", 0)?;
            let sampler = self
                .device
                .new_sampler_state(&d)
                .map_err(|error| format!("sampler {id}: {error}"))?;
            self.samplers.insert(id, sampler);
        }
        Ok(())
    }

    fn pipelines(&mut self, m: &Value) -> Result<(), String> {
        let options = PipelineOptions::default();
        for (id, pv) in table(m, "renderPipelines") {
            let at = format!("render pipeline {id}");
            let mut d = RenderPipelineDescriptor::new(self.function(s(&pv, "vertex")?)?);
            d.fragment = match pv.get("fragment").and_then(Value::as_str) {
                Some(fid) => Some(self.function(fid)?),
                None => None,
            };
            for (k, c) in pv
                .get("colors")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                if c.is_null() || k >= 8 {
                    continue;
                }
                let mut color = ColorAttachmentDescriptor::new(raw::<PixelFormat>(c, "format", 0)?);
                color.blending_enabled = b(c, "blending", false);
                color.source_rgb_blend_factor = raw::<BlendFactor>(c, "srcRGB", 1)?;
                color.destination_rgb_blend_factor = raw::<BlendFactor>(c, "dstRGB", 0)?;
                color.rgb_blend_operation = raw::<BlendOperation>(c, "rgbOp", 0)?;
                color.source_alpha_blend_factor = raw::<BlendFactor>(c, "srcA", 1)?;
                color.destination_alpha_blend_factor = raw::<BlendFactor>(c, "dstA", 0)?;
                color.alpha_blend_operation = raw::<BlendOperation>(c, "alphaOp", 0)?;
                color.write_mask = raw::<ColorWriteMask>(c, "mask", 15)?;
                d.color_attachments[k] = Some(color);
            }
            let format = |key: &str| -> Result<Option<PixelFormat>, String> {
                match u(&pv, key, 0) {
                    0 => Ok(None),
                    _ => raw::<PixelFormat>(&pv, key, 0).map(Some),
                }
            };
            d.depth_format = format("depth")?;
            d.stencil_format = format("stencil")?;
            d.raster_sample_count = u(&pv, "samples", 1) as u32;
            d.alpha_to_coverage = b(&pv, "alphaToCoverage", false);
            d.input_primitive_topology = raw::<PrimitiveTopologyClass>(&pv, "topology", 0)?;
            let state = self
                .device
                .new_render_pipeline_state(&d, &options)
                .map_err(e(&at))?;
            self.render.insert(id, state);
        }
        for (id, pv) in table(m, "tilePipelines") {
            let at = format!("tile pipeline {id}");
            let mut d = TileRenderPipelineDescriptor::new(self.function(s(&pv, "function")?)?);
            for (k, c) in pv
                .get("colors")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                if let (Some(raw_format), true) = (c.as_u64(), k < 8) {
                    let pf = u32::try_from(raw_format).map_err(e(&at))?;
                    d.color_attachment_formats[k] =
                        Some(PixelFormat::try_from(pf).map_err(e(&at))?);
                }
            }
            d.threadgroup_size_matches_tile_size = b(&pv, "matchesTileSize", false);
            let state = self
                .device
                .new_tile_render_pipeline_state(&d, &options)
                .map_err(e(&at))?;
            self.render.insert(id, state);
        }
        for (id, pv) in table(m, "computePipelines") {
            let at = format!("compute pipeline {id}");
            let fid = s(&pv, "function")?;
            let function = self
                .functions
                .get(fid)
                .ok_or_else(|| format!("{at}: no function {fid}"))?;
            let state = self
                .device
                .new_compute_pipeline_state_with_function(function, &options)
                .map_err(e(&at))?;
            self.compute.insert(id, state);
        }
        for (id, dv) in table(m, "depthStencilStates") {
            let mut d = DepthStencilDescriptor::default();
            d.depth_compare_function = raw::<CompareFunction>(&dv, "compare", 7)?;
            d.depth_write_enabled = b(&dv, "write", false);
            if let Some(sv) = dv.get("stencil").filter(|v| v.is_object()) {
                let mut face = StencilDescriptor::default();
                face.stencil_compare_function = raw::<CompareFunction>(sv, "compare", 7)?;
                face.stencil_failure_operation = raw::<StencilOperation>(sv, "fail", 0)?;
                face.depth_failure_operation = raw::<StencilOperation>(sv, "depthFail", 0)?;
                face.depth_stencil_pass_operation = raw::<StencilOperation>(sv, "pass", 0)?;
                face.read_mask = u(sv, "read", 0xffff_ffff) as u32;
                face.write_mask = u(sv, "write", 0xffff_ffff) as u32;
                d.front_face_stencil = face;
                d.back_face_stencil = face;
            }
            let state = self
                .device
                .new_depth_stencil_state(&d)
                .map_err(|error| format!("depth-stencil state {id}: {error}"))?;
            self.depth.insert(id, state);
        }
        Ok(())
    }

    fn texture(&self, id: &str) -> Result<&Texture, String> {
        self.textures
            .get(id)
            .map(|(t, _)| t)
            .ok_or_else(|| format!("no texture {id}"))
    }

    fn buffer(&self, id: &str) -> Result<&Buffer, String> {
        self.buffers
            .get(id)
            .ok_or_else(|| format!("no buffer {id}"))
    }

    fn pass(&self, r: &Value) -> Result<RenderPassDescriptor, String> {
        let mut pass = RenderPassDescriptor::new();
        for (k, c) in r
            .get("colors")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            if c.is_null() || k >= pass.color_attachments.len() {
                continue;
            }
            let a = &mut pass.color_attachments[k];
            a.texture = Some(self.texture(s(c, "texture")?)?.clone());
            a.load_action = LoadAction::from_raw(u(c, "load", 0)).map_err(e("load"))?;
            a.store_action = StoreAction::from_raw(u(c, "store", 1)).map_err(e("store"))?;
            let cc = nums(c, "clear");
            let at = |i: usize| cc.get(i).copied().unwrap_or(0.0);
            a.clear_color = metal2vulkan_runtime::ClearColor::new(at(0), at(1), at(2), at(3));
        }
        if let Some(dv) = r.get("depth").filter(|v| !v.is_null()) {
            let a = &mut pass.depth_attachment;
            a.texture = Some(self.texture(s(dv, "texture")?)?.clone());
            a.load_action = LoadAction::from_raw(u(dv, "load", 0)).map_err(e("depth load"))?;
            a.store_action = StoreAction::from_raw(u(dv, "store", 0)).map_err(e("depth store"))?;
            a.clear_depth = f(dv, "clear", 1.0);
        }
        if let Some(sv) = r.get("stencil").filter(|v| !v.is_null()) {
            let a = &mut pass.stencil_attachment;
            a.texture = Some(self.texture(s(sv, "texture")?)?.clone());
            a.load_action = LoadAction::from_raw(u(sv, "load", 0)).map_err(e("stencil load"))?;
            a.store_action =
                StoreAction::from_raw(u(sv, "store", 0)).map_err(e("stencil store"))?;
            a.clear_stencil = u(sv, "clear", 0) as u32;
        }
        pass.tile_width = u(r, "tileWidth", 0);
        pass.tile_height = u(r, "tileHeight", 0);
        pass.imageblock_sample_length = u(r, "imageblockSize", 0);
        pass.threadgroup_memory_length = u(r, "threadgroupMemoryLength", 0);
        pass.render_target_width = u(r, "width", 0);
        pass.render_target_height = u(r, "height", 0);
        // A pass without attachments rasterizes one sample: Metal ends the process at the
        // default 0; a pass with attachments takes theirs.
        pass.default_raster_sample_count = 1;
        Ok(pass)
    }

    fn render_encoder(&self, cb: &CommandBuffer, r: &Value) -> Result<(), String> {
        let pass = self.pass(r)?;
        let mut enc = cb
            .render_command_encoder(&pass)
            .map_err(e("render_command_encoder"))?;
        for (k, cmd) in r
            .get("commands")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            self.render_command(&mut enc, cmd)
                .map_err(|w| format!("command {k}: {w}"))?;
        }
        enc.end_encoding().map_err(e("end_encoding"))
    }

    fn render_command(&self, enc: &mut RenderCommandEncoder, cmd: &Value) -> Result<(), String> {
        let (name, v) = cmd
            .as_object()
            .and_then(|o| o.iter().next())
            .ok_or("an empty command")?;
        match name.as_str() {
            "pipeline" => {
                let id = v.as_str().ok_or("pipeline id")?;
                let state = self
                    .render
                    .get(id)
                    .ok_or_else(|| format!("no pipeline {id}"))?;
                enc.set_render_pipeline_state(state).map_err(e(name))
            }
            "buffer" => {
                let index = u(v, "index", 0) as u32;
                let buffer = self.buffer(s(v, "buffer")?)?;
                let offset = u(v, "offset", 0);
                match s(v, "stage")? {
                    "vertex" => enc.set_vertex_buffer(Some(buffer), offset, index),
                    "fragment" => enc.set_fragment_buffer(Some(buffer), offset, index),
                    "tile" => enc.set_tile_buffer(Some(buffer), offset, index),
                    other => return Err(format!("stage {other}")),
                }
                .map_err(e(name))
            }
            "bytes" => {
                let index = u(v, "index", 0) as u32;
                let bytes = hex(s(v, "hex")?)?;
                match s(v, "stage")? {
                    "vertex" => enc.set_vertex_bytes(&bytes, index),
                    "fragment" => enc.set_fragment_bytes(&bytes, index),
                    "tile" => enc.set_tile_bytes(&bytes, index),
                    other => return Err(format!("stage {other}")),
                }
                .map_err(e(name))
            }
            "texture" => {
                let index = u(v, "index", 0) as u32;
                let texture = self.texture(s(v, "texture")?)?;
                match s(v, "stage")? {
                    "vertex" => enc.set_vertex_texture(Some(texture), index),
                    "fragment" => enc.set_fragment_texture(Some(texture), index),
                    "tile" => enc.set_tile_texture(Some(texture), index),
                    other => return Err(format!("stage {other}")),
                }
                .map_err(e(name))
            }
            "sampler" => {
                let index = u(v, "index", 0) as u32;
                let id = s(v, "sampler")?;
                let sampler = self
                    .samplers
                    .get(id)
                    .ok_or_else(|| format!("no sampler {id}"))?;
                match s(v, "stage")? {
                    "vertex" => enc.set_vertex_sampler_state(Some(sampler), index),
                    "fragment" => enc.set_fragment_sampler_state(Some(sampler), index),
                    "tile" => enc.set_tile_sampler_state(Some(sampler), index),
                    other => return Err(format!("stage {other}")),
                }
                .map_err(e(name))
            }
            "threadgroupMemory" => enc
                .set_threadgroup_memory_length(u(v, "length", 0), 0, u(v, "index", 0) as u32)
                .map_err(e(name)),
            "depthStencil" => {
                let id = v.as_str().ok_or("depth-stencil id")?;
                let state = self
                    .depth
                    .get(id)
                    .ok_or_else(|| format!("no depth-stencil state {id}"))?;
                enc.set_depth_stencil_state(state).map_err(e(name))
            }
            "stencilRef" => {
                enc.set_stencil_reference_value(v.as_u64().unwrap_or(0) as u32);
                Ok(())
            }
            "blendColor" => {
                let c: Vec<f32> = v
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_f64)
                    .map(|x| x as f32)
                    .collect();
                let at = |i: usize| c.get(i).copied().unwrap_or(0.0);
                enc.set_blend_color(at(0), at(1), at(2), at(3));
                Ok(())
            }
            "viewport" => {
                let n: Vec<f64> = v
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_f64)
                    .collect();
                let at = |i: usize| n.get(i).copied().unwrap_or(0.0);
                enc.set_viewport(Viewport::new(at(0), at(1), at(2), at(3), at(4), at(5)))
                    .map_err(e(name))
            }
            "scissor" => {
                let n: Vec<u64> = v
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_u64)
                    .collect();
                let at = |i: usize| n.get(i).copied().unwrap_or(0);
                enc.set_scissor_rect(ScissorRect::new(at(0), at(1), at(2), at(3)))
                    .map_err(e(name))
            }
            "cull" => {
                let mode = CullMode::try_from(v.as_u64().unwrap_or(0) as u32).map_err(e(name))?;
                enc.set_cull_mode(mode);
                Ok(())
            }
            "winding" => {
                let w = Winding::try_from(v.as_u64().unwrap_or(0) as u32).map_err(e(name))?;
                enc.set_front_facing_winding(w);
                Ok(())
            }
            "useResource" => {
                let id = s(v, "id")?;
                let usage = raw::<ResourceUsage>(v, "usage", 1)?;
                let stages = raw::<RenderStages>(v, "stages", 3)?;
                let resource = match (self.textures.get(id), self.buffers.get(id)) {
                    (Some((t, _)), _) => ResourceRef::Texture(t),
                    (None, Some(b)) => ResourceRef::Buffer(b),
                    _ => return Err(format!("no resource {id}")),
                };
                enc.use_resource(resource, usage, stages).map_err(e(name))
            }
            "useHeap" => {
                let id = v.as_str().ok_or("heap id")?;
                let heap = self.heaps.get(id).ok_or_else(|| format!("no heap {id}"))?;
                enc.use_heap(heap, RenderStages::try_from(3).map_err(e(name))?)
                    .map_err(e(name))
            }
            "draw" => enc
                .draw_primitives(
                    raw::<PrimitiveType>(v, "primitive", 3)?,
                    u(v, "start", 0) as u32,
                    u(v, "count", 3) as u32,
                    u(v, "instances", 1) as u32,
                    u(v, "baseInstance", 0) as u32,
                )
                .map_err(e(name)),
            "drawIndexed" => {
                let buffer = self.buffer(s(v, "buffer")?)?;
                enc.draw_indexed_primitives(
                    raw::<PrimitiveType>(v, "primitive", 3)?,
                    u(v, "count", 0) as u32,
                    raw::<IndexType>(v, "indexType", 0)?,
                    buffer,
                    u(v, "offset", 0),
                    u(v, "instances", 1) as u32,
                    v.get("baseVertex").and_then(Value::as_i64).unwrap_or(0) as i32,
                    u(v, "baseInstance", 0) as u32,
                )
                .map_err(e(name))
            }
            "tile" => {
                let n = nums(v, "threads");
                let at = |i: usize| n.get(i).copied().unwrap_or(1.0) as u64;
                enc.dispatch_threads_per_tile(Size::new(at(0), at(1), 1))
                    .map_err(e(name))
            }
            other => Err(format!("unknown render command {other}")),
        }
    }

    fn compute_encoder(&self, cb: &CommandBuffer, c: &Value) -> Result<(), String> {
        let mut enc = cb
            .compute_command_encoder()
            .map_err(e("compute_command_encoder"))?;
        for (k, cmd) in c
            .get("commands")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            self.compute_command(&mut enc, cmd)
                .map_err(|w| format!("command {k}: {w}"))?;
        }
        enc.end_encoding().map_err(e("end_encoding"))
    }

    fn compute_command(&self, enc: &mut ComputeCommandEncoder, cmd: &Value) -> Result<(), String> {
        let (name, v) = cmd
            .as_object()
            .and_then(|o| o.iter().next())
            .ok_or("an empty command")?;
        match name.as_str() {
            "pipeline" => {
                let id = v.as_str().ok_or("pipeline id")?;
                let state = self
                    .compute
                    .get(id)
                    .ok_or_else(|| format!("no pipeline {id}"))?;
                enc.set_compute_pipeline_state(state).map_err(e(name))
            }
            "buffer" => enc
                .set_buffer(
                    Some(self.buffer(s(v, "buffer")?)?),
                    u(v, "offset", 0),
                    u(v, "index", 0) as u32,
                )
                .map_err(e(name)),
            "bytes" => enc
                .set_bytes(&hex(s(v, "hex")?)?, u(v, "index", 0) as u32)
                .map_err(e(name)),
            "texture" => enc
                .set_texture(
                    Some(self.texture(s(v, "texture")?)?),
                    u(v, "index", 0) as u32,
                )
                .map_err(e(name)),
            "sampler" => {
                let id = s(v, "sampler")?;
                let sampler = self
                    .samplers
                    .get(id)
                    .ok_or_else(|| format!("no sampler {id}"))?;
                enc.set_sampler_state(Some(sampler), u(v, "index", 0) as u32)
                    .map_err(e(name))
            }
            "threadgroupMemory" => enc
                .set_threadgroup_memory_length(u(v, "length", 0) as u32, u(v, "index", 0) as u32)
                .map_err(e(name)),
            "useResource" => {
                let id = s(v, "id")?;
                let usage = raw::<ResourceUsage>(v, "usage", 1)?;
                let resource = match (self.textures.get(id), self.buffers.get(id)) {
                    (Some((t, _)), _) => ResourceRef::Texture(t),
                    (None, Some(b)) => ResourceRef::Buffer(b),
                    _ => return Err(format!("no resource {id}")),
                };
                enc.use_resource(resource, usage).map_err(e(name))
            }
            "useHeap" => {
                let id = v.as_str().ok_or("heap id")?;
                let heap = self.heaps.get(id).ok_or_else(|| format!("no heap {id}"))?;
                enc.use_heap(heap).map_err(e(name))
            }
            "dispatchThreadgroups" => enc
                .dispatch_threadgroups(size3(v, "groups")?, size3(v, "threads")?)
                .map_err(e(name)),
            "dispatchThreads" => enc
                .dispatch_threads(size3(v, "grid")?, size3(v, "threads")?)
                .map_err(e(name)),
            other => Err(format!("unknown compute command {other}")),
        }
    }

    /// Every readback the manifest asks for, written to the output directory.
    fn readback(&self, m: &Value, status: &mut Status) -> Result<(), String> {
        let ids: Vec<String> = m
            .get("readback")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        if self.queue.is_none() {
            return Ok(());
        }
        for id in ids {
            let bytes = if let Some((texture, d)) = self.textures.get(&id) {
                if d.sample_count > 1
                    || d.resource_options
                        .storage_mode()
                        .is_ok_and(|m| m.raw() == 3)
                {
                    status.line(&format!("skip {id}: memoryless or multisampled"));
                    continue;
                }
                self.read_texture(texture, d)
            } else if let Some(buffer) = self.buffers.get(&id) {
                self.read_buffer(buffer)
            } else {
                Err(format!("no resource {id}"))
            };
            match bytes {
                Ok(bytes) => {
                    std::fs::write(self.out.join(format!("{id}.bin")), &bytes)
                        .map_err(e("readback"))?;
                    status.line(&format!("read {id} {}", bytes.len()));
                }
                Err(why) => status.line(&format!("skip {id}: {why}")),
            }
        }
        Ok(())
    }

    fn read_texture(&self, texture: &Texture, d: &TextureDescriptor) -> Result<Vec<u8>, String> {
        let texel = texel_bytes(d.pixel_format)
            .ok_or_else(|| format!("no texel size for {:?}", d.pixel_format))?;
        let row = texel * d.width;
        let image = row * d.height * d.depth.max(1);
        let slices = d.array_length.max(1)
            * match d.texture_type {
                TextureType::Cube | TextureType::CubeArray => 6,
                _ => 1,
            };
        let out = self
            .device
            .new_buffer(image * slices, ResourceOptions::shared())
            .map_err(e("readback buffer"))?;
        let cb = self.queue().command_buffer().map_err(e("readback"))?;
        let mut blit = cb.blit_command_encoder().map_err(e("readback"))?;
        for slice in 0..slices {
            blit.copy_from_texture_to_buffer(
                texture,
                slice,
                0,
                Origin::new(0, 0, 0),
                Size::new(d.width, d.height, d.depth.max(1)),
                &out,
                slice * image,
                row,
                row * d.height,
                BlitOption::None,
            )
            .map_err(e("readback"))?;
        }
        blit.end_encoding().map_err(e("readback"))?;
        finish(&cb)?;
        contents(&out)
    }

    fn read_buffer(&self, buffer: &Buffer) -> Result<Vec<u8>, String> {
        if buffer.contents().is_some() {
            return contents(buffer);
        }
        let out = self
            .device
            .new_buffer(buffer.length(), ResourceOptions::shared())
            .map_err(e("readback buffer"))?;
        let cb = self.queue().command_buffer().map_err(e("readback"))?;
        let mut blit = cb.blit_command_encoder().map_err(e("readback"))?;
        blit.copy_from_buffer_to_buffer(buffer, 0, &out, 0, buffer.length())
            .map_err(e("readback"))?;
        blit.end_encoding().map_err(e("readback"))?;
        finish(&cb)?;
        contents(&out)
    }
}

fn finish(command_buffer: &CommandBuffer) -> Result<(), String> {
    command_buffer.commit().map_err(e("commit"))?;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("the command buffer did not complete within {WAIT:?}"))?;
    match command_buffer.status() {
        CommandBufferStatus::Completed => Ok(()),
        status => Err(format!(
            "the command buffer ended {status:?}: {:?}",
            command_buffer.error()
        )),
    }
}

/// The bytes of `buffer`, which no command buffer writes any more.
fn contents(buffer: &Buffer) -> Result<Vec<u8>, String> {
    let at = buffer.contents().ok_or("a buffer without host contents")?;
    let length = usize::try_from(buffer.length()).map_err(|error| error.to_string())?;
    // SAFETY: the buffer is shared, `contents` holds its `length` bytes, and the command buffer
    // that wrote it completed.
    Ok(unsafe { std::slice::from_raw_parts(at.as_ptr(), length) }.to_vec())
}
