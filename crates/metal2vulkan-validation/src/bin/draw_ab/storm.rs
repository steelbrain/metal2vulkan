//! `corpus-draw-ab storm`: the compile-storm benchmark (for the runtime's budget of pipeline
//! makings in flight).
//!
//! One process, one device, one storm of `N` pipeline makings of one large fragment function (an
//! owned fixture: `--air`, `--vertex`, `--fragment`), driven by `C` threads of the program at
//! once, each taking the storm's next making until none is left:
//!
//! - **Storm A, distinct fragments**: making `i` creates a render pipeline state of the fragment
//!   specialized with function constant 0 = `i` (a translation of its own, so `N` distinct
//!   modules), made at the state's creation on the creating thread.
//! - **Storm B, one module's specializations**: one state (created before the storm), drawn once
//!   per making, making `i` binding a texture of the `i`-th pixel format to texture 0, which the
//!   fixture samples with a `half` result: each format class is a specialization of the same
//!   module (`SpecKey::TexelFormat`), made by the device's pipeline pool for the draw. Each
//!   thread commits its draw's command buffer and waits for it.
//! - **Storm C, identical modules**: making `i` creates a state of the same function (constant
//!   0 = 0) whose color attachment blends with the `i`-th source factor: the same module and
//!   specialization, a pipeline of its own, made at the state's creation.
//! - **Storm D, topologies**: one state of an unspecified topology class (created before the
//!   storm), drawn once per making, making `i` drawing with the `i`-th primitive type (triangles,
//!   lines, a triangle strip, a line strip) and one texture format: the same module and
//!   specialization, a pipeline per topology, made by the device's pipeline pool for the draw.
//! - **Storm E, listed draws** (`--draws FILE --ids ID,... --sources DIR` in place of the
//!   fixture): making `i` creates a state of draw `ids[i mod n]` of the draws file, its fragment
//!   (`DIR/<fragment>.bc` and `.entry`) under its logged template with the passthrough vertex
//!   function generated for it, as `time` draws it, made at the state's creation; a draw's later
//!   makings write its first color attachment through another write mask (`WRITE_MASKS`), a
//!   pipeline of its own. Distinct draws are full makings each (their own translation, MSL and
//!   back-end compile). One draw repeated shares its module, specialization and MSL only where
//!   its fragment does not fetch a color attachment: previously a fetching fragment on MoltenVK
//!   (the interlocked storage form) compiled each mask into a module of its own, so storm E's mask
//!   rounds there measured cold MSL compiles. The masks that keep some of an attachment's channels
//!   share one module (an extended-range or sRGB attachment's still compile per mask).
//!
//! Before the threads start, `--turns T --ahead A` set the process's budget of makings in flight
//! through the runtime's measurement hook (`set_making_budget_for_measurement`); without them the
//! build's own budget holds. Meanwhile a ticker thread commits a small command buffer (one
//! `fillBuffer` of 256 bytes) on a queue of its own every `--tick-ms` (16 by default, a fraction
//! allowed: 16.667 is 60 Hz), each after the last completed, and records its commit-to-completion
//! time: what the storm does to unrelated work on the device.
//!
//! - `--tick-frame WxH:DRAWS` makes each tick a small frame instead: one render pass over a `W` by
//!   `H` `RGBA8Unorm` target, cleared, of `DRAWS` textured rectangles side by side covering it (the
//!   owned `tob_rect` and `tob_sample` of `tob_renders.ll`, a 256-square texture), as a program
//!   rendering at the tick's rate beside the compiles would.
//! - `--hold-ms MS` keeps the ticker going until `MS` after the storm's start, past the storm's
//!   end: a baseline of the ticker with (almost) nothing compiling, from a storm whose makings are
//!   all in the driver's cache.
//!
//! `--qos user-interactive|user-initiated|default|utility|background` sets the quality of service
//! of every thread the harness runs (the program's threads, the ticker, and the thread that opens
//! the device, which the runtime's own threads may inherit), on macOS; the main thread's class
//! before it is a `meta` line.
//!
//! What a making cost comes from the runtime's creation log (`METAL2VULKAN_LOG_PIPELINES=1`,
//! required; with `MVK_CONFIG_PERFORMANCE_TRACKING=1` its lines carry MoltenVK's split): each
//! line, received through the device's message sink as the making ends, gives the making's end
//! instant, its time, the time it waited for its turn (builds), where it was made, and the split
//! (SPIR-V to MSL, MSL compile, MSL load, a library from the module's cache, function retrieval,
//! specialization, pipeline compile; device-wide, so a making that overlapped another holds both,
//! and says so).
//!
//! `--out FILE` gets the table: `meta` lines (storm, makings, threads, budget, device, the
//! fragment's SPIR-V words, the device's fills, its variant pipeline caches' counts, its driver
//! pipeline cache's, and its pipelines and libraries made and linked by the storm's end), one
//! `making` line per logged making that ended after the storm began, one `job` line per thread's
//! piece of work (its start and end, from the storm's start), one `tick` line per tick (its start
//! from the storm's, and the instants, from its commit's call, at which `commit` returned, the
//! command buffer was scheduled, and it completed), the `ticker` line (count, median, 90th
//! percentile and maximum commit-to-completion), and the `summary` line: the storm's time (start to
//! the last job's end), the makings logged and their throughput, their mean time, the spread of
//! their end instants, and the largest group of makings whose ends fall within 2 ms of the next
//! (siblings ending at one instant: a shared serialization point). Every time is in milliseconds.
//!
//! Storms A to D draw the caller's own fixture. Storm E reads the draws and sources the caller
//! names: when they are the consumer's, they and every output are guest data, which stay on the
//! hosts the guest data rule allows and are never committed.

use super::{bytes, draw_parts, number, options, required, DrawParts, Palette, Stop};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, DataType, PixelFormat, PrimitiveTopologyClass, PrimitiveType,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Buffer, CommandQueue, DebugMessage, Device, DeviceOptions, Function, Library, LoadAction,
    MessageSink, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, ShaderCacheOptions, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan_validation::candidate::engine::graphics::RenderTemplate;
use metal2vulkan_validation::candidate::runtime_case::open_device;
use metal2vulkan_validation::gpu::DeviceType;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The creation log's message id (`runtime/src/pipeline/creation_log.rs`).
const CREATION_LOG: &str = "metal2vulkan-pipeline-creation";

/// Storm B's formats, one class each for a `half` sample (`metal::texel_format`).
const FORMATS: [PixelFormat; 12] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA32Float,
    PixelFormat::RGBA8Snorm,
    PixelFormat::RGBA16Unorm,
    PixelFormat::RGBA16Snorm,
    PixelFormat::RGB10A2Unorm,
    PixelFormat::RG11B10Float,
    PixelFormat::B5G6R5Unorm,
    PixelFormat::A1BGR5Unorm,
    PixelFormat::ABGR4Unorm,
    PixelFormat::RG16Unorm,
];

/// Storm D's primitive types, one pipeline each.
const TOPOLOGIES: [PrimitiveType; 4] = [
    PrimitiveType::Triangle,
    PrimitiveType::Line,
    PrimitiveType::TriangleStrip,
    PrimitiveType::LineStrip,
];

/// Storm C's source color blend factors, one pipeline each.
const FACTORS: [BlendFactor; 12] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::OneMinusDestinationColor,
    BlendFactor::DestinationAlpha,
    BlendFactor::OneMinusDestinationAlpha,
    BlendFactor::SourceAlphaSaturated,
    BlendFactor::BlendColor,
];

/// Storm E's write masks of a draw's first color attachment after its first making, one
/// pipeline each (none is `ALL`, the logged templates' usual mask).
const WRITE_MASKS: [ColorWriteMask; 12] = {
    use ColorWriteMask as M;
    [
        M::RED.union(M::GREEN).union(M::BLUE),
        M::RED.union(M::GREEN).union(M::ALPHA),
        M::RED.union(M::BLUE).union(M::ALPHA),
        M::GREEN.union(M::BLUE).union(M::ALPHA),
        M::RED.union(M::GREEN),
        M::RED.union(M::BLUE),
        M::RED.union(M::ALPHA),
        M::GREEN.union(M::BLUE),
        M::GREEN.union(M::ALPHA),
        M::BLUE.union(M::ALPHA),
        M::RED,
        M::GREEN,
    ]
};

/// The attachments' side, small: the storm times makings, not draws.
const SIDE: u64 = 64;

/// The longest a storm's command buffer is waited for.
const WAIT: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Storm {
    Distinct,
    Specialized,
    Identical,
    Topologies,
    Draws,
}

impl Storm {
    fn parse(text: &str) -> Result<Storm, String> {
        match text {
            "a" | "A" => Ok(Storm::Distinct),
            "b" | "B" => Ok(Storm::Specialized),
            "c" | "C" => Ok(Storm::Identical),
            "d" | "D" => Ok(Storm::Topologies),
            "e" | "E" => Ok(Storm::Draws),
            _ => Err(format!("--storm {text:?}: a, b, c, d or e")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Storm::Distinct => "A distinct fragments",
            Storm::Specialized => "B one module's specializations",
            Storm::Identical => "C identical modules",
            Storm::Topologies => "D topologies",
            Storm::Draws => "E listed draws",
        }
    }
}

/// Set the process's budget of makings in flight (`set_making_budget_for_measurement`).
fn set_budget(turns: usize, ahead: usize) -> Result<(), String> {
    metal2vulkan_runtime::set_making_budget_for_measurement(turns, ahead)
        .map_err(|error| error.to_string())
}

/// Set this thread's quality of service to `name` (`--qos`).
#[cfg(target_vendor = "apple")]
pub(crate) fn set_qos(name: &str) -> Result<(), String> {
    use libc::qos_class_t as Q;
    let class = match name {
        "user-interactive" => Q::QOS_CLASS_USER_INTERACTIVE,
        "user-initiated" => Q::QOS_CLASS_USER_INITIATED,
        "default" => Q::QOS_CLASS_DEFAULT,
        "utility" => Q::QOS_CLASS_UTILITY,
        "background" => Q::QOS_CLASS_BACKGROUND,
        _ => {
            return Err(format!(
                "--qos {name:?}: user-interactive, user-initiated, default, utility or background"
            ))
        }
    };
    // SAFETY: sets the calling thread's own class; no pointer is passed.
    match unsafe { libc::pthread_set_qos_class_self_np(class, 0) } {
        0 => Ok(()),
        error => Err(format!("pthread_set_qos_class_self_np: {error}")),
    }
}

/// `--qos` is macOS's.
#[cfg(not(target_vendor = "apple"))]
pub(crate) fn set_qos(name: &str) -> Result<(), String> {
    Err(format!(
        "--qos {name:?}: quality of service classes are macOS's"
    ))
}

/// This thread's quality of service class, as `pthread_get_qos_class_np` answers it.
#[cfg(target_vendor = "apple")]
pub(crate) fn qos_now() -> String {
    let mut class = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
    let mut relative = 0;
    // SAFETY: both pointers are to locals that outlive the call; the thread is this one.
    let error =
        unsafe { libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative) };
    match error {
        0 => format!("0x{:02x} (relative {relative})", class as u32),
        error => format!("unread ({error})"),
    }
}

/// No quality of service classes here.
#[cfg(not(target_vendor = "apple"))]
pub(crate) fn qos_now() -> String {
    "none".into()
}

/// Storms A to D's fixture.
struct Fixture {
    library: Library,
    vertex: Function,
    fragment: String,
    /// The fragment with constant 0 = 0 (storms B and C).
    plain: Function,
}

/// What the storm's threads share.
struct Shared {
    storm: Storm,
    device: Device,
    /// Storms A to D's fixture.
    fixture: Option<Fixture>,
    /// Storm E's draws, in `--ids` order.
    draws: Vec<DrawParts>,
    /// Storm B's state, created before the storm.
    state: Option<RenderPipelineState>,
    queue: CommandQueue,
    buffers: Vec<Buffer>,
    sampler: SamplerState,
    textures: Vec<Texture>,
    targets: Vec<Texture>,
}

/// The fragment's function constant values for variant `value`.
fn constants(value: u32) -> FunctionConstantValues {
    FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, value.to_le_bytes().to_vec()),
    )
}

/// The descriptor of `vertex` and `fragment` drawing into one `RGBA8Unorm` attachment, blending
/// with `factor` where given.
fn descriptor(
    vertex: &Function,
    fragment: &Function,
    factor: Option<BlendFactor>,
    topology: PrimitiveTopologyClass,
) -> Result<RenderPipelineDescriptor, String> {
    let mut template = RenderTemplate::new(topology);
    let mut attachment = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    if let Some(factor) = factor {
        attachment.blending_enabled = true;
        attachment.source_rgb_blend_factor = factor;
    }
    template.color_attachments.insert(0, attachment);
    template
        .descriptor(
            StageSource::Function(vertex.function_descriptor()),
            Some(StageSource::Function(fragment.function_descriptor())),
        )
        .map_err(|error| error.to_string())
}

impl Shared {
    /// Making `job`'s work; `Ok` holds what must live until the storm ends.
    fn job(&self, job: usize) -> Result<Option<RenderPipelineState>, String> {
        let make = |descriptor: &RenderPipelineDescriptor| {
            self.device
                .new_render_pipeline_state(descriptor, &PipelineOptions::default())
                .map(Some)
                .map_err(|error| format!("state {job}: {error}"))
        };
        if self.storm == Storm::Draws {
            return make(&self.draw_variant(job)?);
        }
        let fixture = self
            .fixture
            .as_ref()
            .ok_or("a fixture storm without its fixture")?;
        let create = |fragment: &Function, factor: Option<BlendFactor>| {
            make(&descriptor(
                &fixture.vertex,
                fragment,
                factor,
                PrimitiveTopologyClass::Triangle,
            )?)
        };
        match self.storm {
            Storm::Distinct => {
                let fragment = fixture
                    .library
                    .new_function_with_name_constant_values(
                        &fixture.fragment,
                        &constants(job as u32),
                    )
                    .map_err(|error| format!("function {job}: {error}"))?;
                create(&fragment, None)
            }
            Storm::Identical => create(&fixture.plain, Some(FACTORS[job])),
            Storm::Specialized => self.draw(job, PrimitiveType::Triangle).map(|()| None),
            Storm::Topologies => self.draw(job, TOPOLOGIES[job]).map(|()| None),
            Storm::Draws => Err("storm E is made above".into()),
        }
    }

    /// Storm E's making `job`: draw `job mod n`'s descriptor, its first color attachment's write
    /// mask the `job / n`-th of [`WRITE_MASKS`] after the draw's first making.
    fn draw_variant(&self, job: usize) -> Result<RenderPipelineDescriptor, String> {
        let n = self.draws.len().max(1);
        let parts = self.draws.get(job % n).ok_or("storm E without draws")?;
        let mut template = parts.template.clone();
        if let Some(round) = (job / n).checked_sub(1) {
            let mask = *WRITE_MASKS
                .get(round)
                .ok_or_else(|| format!("making {job}: no write mask left"))?;
            let (_, first) = template
                .color_attachments
                .iter_mut()
                .next()
                .ok_or_else(|| format!("making {job}: a draw without a color attachment"))?;
            first.write_mask = mask;
        }
        parts.descriptor(&template)
    }

    /// Storm B's or D's making `job`: one draw of `primitive` sampling `textures[job]`,
    /// committed and waited for.
    fn draw(&self, job: usize, primitive: PrimitiveType) -> Result<(), String> {
        let e = |error: metal2vulkan_runtime::Error| format!("draw {job}: {error}");
        let state = self
            .state
            .as_ref()
            .ok_or("a drawn storm without its state")?;
        let command_buffer = self.queue.command_buffer().map_err(e)?;
        let mut pass = RenderPassDescriptor::new();
        let color = pass
            .color_attachments
            .get_mut(0)
            .ok_or("no color attachment 0")?;
        color.texture = Some(self.targets[job].clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).map_err(e)?;
        encoder.set_render_pipeline_state(state).map_err(e)?;
        for (index, buffer) in self.buffers.iter().enumerate() {
            encoder
                .set_fragment_buffer(Some(buffer), 0, index as u32)
                .map_err(e)?;
        }
        encoder
            .set_fragment_texture(Some(&self.textures[job]), 0)
            .map_err(e)?;
        encoder
            .set_fragment_sampler_state(Some(&self.sampler), 0)
            .map_err(e)?;
        encoder.draw_primitives(primitive, 0, 3, 1, 0).map_err(e)?;
        encoder.end_encoding().map_err(e)?;
        command_buffer
            .commit()
            .map_err(|error| format!("commit {job}: {error}"))?;
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| format!("draw {job} did not complete within {WAIT:?}"))?;
        match command_buffer.status() {
            metal2vulkan_runtime::CommandBufferStatus::Completed => Ok(()),
            status => Err(format!(
                "draw {job} ended {status:?}: {:?}",
                command_buffer.error()
            )),
        }
    }
}

/// One logged making.
#[derive(Default)]
struct Making {
    /// From the storm's start; `None` before it.
    end: Option<Duration>,
    head: String,
    made_ms: Option<f64>,
    failed: bool,
    waited_ms: f64,
    place: &'static str,
    overlapped: bool,
    /// MoltenVK's split: activity -> (count, milliseconds).
    split: BTreeMap<String, (u64, f64)>,
    /// The fragment's SPIR-V words, where the line prints the state's first key.
    words: Option<u64>,
}

/// The number at the start of `text` (digits, a point).
fn leading_number(text: &str) -> Option<f64> {
    let end = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    text[..end].parse().ok()
}

/// Parse a creation log line.
fn parse(text: &str, end: Option<Duration>) -> Making {
    let mut making = Making {
        end,
        place: "draw",
        ..Making::default()
    };
    let body = text
        .strip_prefix("metal2vulkan: pipeline creation: ")
        .unwrap_or(text);
    if let Some(at) = body.find(": made in ") {
        making.head = body[..at].to_string();
        making.made_ms = leading_number(&body[at + ": made in ".len()..]);
    } else if let Some(at) = body.find(": failed in ") {
        making.head = body[..at].to_string();
        making.failed = true;
        making.made_ms = leading_number(&body[at + ": failed in ".len()..]);
    }
    if let Some(at) = body.find(" after waiting ") {
        making.waited_ms = leading_number(&body[at + " after waiting ".len()..]).unwrap_or(0.0);
    }
    if body.contains("made at its state's creation, the predicted key") {
        making.place = "creation-predicted";
    }
    making.overlapped = body.contains("overlapped another creation");
    if let Some(at) = body.find("; MoltenVK") {
        let rest = &body[at + "; MoltenVK".len()..];
        let rest = rest.split_once(": ").map_or("", |(_, after)| after);
        let rest = rest.split("; ").next().unwrap_or("");
        for part in rest.split(", ") {
            // "<activity> <count>x <ms> ms"
            let words: Vec<&str> = part.split(' ').collect();
            if words.len() < 4 || words[words.len() - 1] != "ms" {
                continue;
            }
            let ms = words[words.len() - 2].parse::<f64>().ok();
            let count = words[words.len() - 3]
                .strip_suffix('x')
                .and_then(|count| count.parse::<u64>().ok());
            if let (Some(ms), Some(count)) = (ms, count) {
                let activity = words[..words.len() - 3].join(" ");
                making.split.insert(activity, (count, ms));
            }
        }
    }
    // The key's fields: "spirv <stage>: <hash> (<n> words)", the stage as its debug print.
    for field in body.split("; ") {
        let field = field.strip_prefix("its first, key: ").unwrap_or(field);
        let Some((name, value)) = field.split_once(": ") else {
            continue;
        };
        let fragment = name
            .strip_prefix("spirv ")
            .is_some_and(|stage| stage.to_ascii_lowercase().contains("frag"));
        if let (true, Some(open)) = (fragment, value.find('(')) {
            making.words = leading_number(&value[open + 1..]).map(|words| words as u64);
        }
    }
    making
}

/// The `q`-quantile of sorted `values`.
fn quantile(values: &[f64], q: f64) -> f64 {
    match values.len() {
        0 => f64::NAN,
        n => values[((n - 1) as f64 * q).round() as usize],
    }
}

/// One job's record: its making, its start and end from the storm's start, and its error.
type JobRecord = (usize, Duration, Duration, Option<String>);

pub(crate) fn storm(args: &[String]) -> Result<(), String> {
    let options = options(args)?;
    let storm = Storm::parse(required(&options, "storm")?)?;
    let makings = number(&options, "makings", 8)? as usize;
    let threads = (number(&options, "concurrency", 1)? as usize).max(1);
    let tick = match options.get("tick-ms") {
        None => Duration::from_millis(16),
        Some(text) => text
            .parse::<f64>()
            .ok()
            .filter(|ms| ms.is_finite() && *ms > 0.0)
            .map(|ms| Duration::from_secs_f64(ms / 1000.0))
            .ok_or_else(|| format!("--tick-ms {text}: a positive number of milliseconds"))?,
    };
    let hold = Duration::from_millis(number(&options, "hold-ms", 0)?);
    let frame_shape = options
        .get("tick-frame")
        .map(|text| FrameShape::parse(text))
        .transpose()?;
    // Storm E's draws, by id.
    let ids: Vec<String> = match storm {
        Storm::Draws => required(&options, "ids")?
            .split(',')
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    let limit = match storm {
        Storm::Specialized => FORMATS.len(),
        Storm::Identical => FACTORS.len(),
        Storm::Topologies => TOPOLOGIES.len(),
        Storm::Distinct => usize::MAX,
        Storm::Draws => ids.len() * (1 + WRITE_MASKS.len()),
    };
    if makings == 0 || makings > limit {
        return Err(format!("--makings {makings}: 1 to {limit} for this storm"));
    }
    if !metal2vulkan::env_vars::log_pipelines() {
        return Err("the storm reads the creation log: set METAL2VULKAN_LOG_PIPELINES=1".into());
    }
    let out = PathBuf::from(required(&options, "out")?);
    let qos = options.get("qos").map(String::as_str);
    let qos_at_start = qos_now();
    if let Some(qos) = qos {
        set_qos(qos)?;
    }
    // The budget first: the hook fails once any device or making used the process's budget.
    let budget = match (options.get("turns"), options.get("ahead")) {
        (None, None) => "the build's default".to_string(),
        (turns, ahead) => {
            let parse = |value: Option<&String>, name: &str| -> Result<usize, String> {
                value
                    .ok_or_else(|| format!("--turns and --ahead go together; --{name} is missing"))?
                    .parse()
                    .map_err(|error| format!("--{name}: {error}"))
            };
            let (turns, ahead) = (parse(turns, "turns")?, parse(ahead, "ahead")?);
            set_budget(turns, ahead)?;
            format!("{turns} turns, {ahead} ahead")
        }
    };
    // Every message, with the instant it arrived.
    let received: Arc<Mutex<Vec<(Instant, DebugMessage)>>> = Arc::default();
    let sink = {
        let received = Arc::clone(&received);
        MessageSink::new(move |message: &DebugMessage| {
            let now = Instant::now();
            if let Ok(mut received) = received.lock() {
                received.push((now, message.clone()));
            }
        })
    };
    let mut device_options = DeviceOptions::default();
    device_options.shader_cache =
        ShaderCacheOptions::in_directory(required(&options, "shader-cache")?);
    device_options.messages = Some(sink);
    let device_type = DeviceType::parse(required(&options, "device-type")?)?;
    let device =
        open_device(device_type, device_options).map_err(|error| format!("device: {error}"))?;
    let (fixture, draws) = match storm {
        Storm::Draws => {
            let file = required(&options, "draws")?;
            let sources = Path::new(required(&options, "sources")?);
            let draws = ids
                .iter()
                .map(|id| {
                    draw_parts(&device, file, id, sources).map_err(|stop| match stop {
                        Stop::Skip(why) | Stop::Refused(why) | Stop::Fatal(why) => {
                            format!("draw {id}: {why}")
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            (None, draws)
        }
        _ => (Some(fixture(&device, &options)?), Vec::new()),
    };
    let e =
        |what: &'static str| move |error: metal2vulkan_runtime::Error| format!("{what}: {error}");
    let queue = device.new_command_queue().map_err(e("queue"))?;
    let ticker_queue = device.new_command_queue().map_err(e("ticker queue"))?;
    let buffers = (0..3u64)
        .map(|slot| {
            device
                .new_buffer_with_bytes(
                    &bytes(Palette::Timing, slot, 4096),
                    ResourceOptions::shared(),
                )
                .map_err(e("buffer"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sampler = device
        .new_sampler_state(&SamplerDescriptor::new())
        .map_err(e("sampler"))?;
    let (mut textures, mut targets) = (Vec::new(), Vec::new());
    if matches!(storm, Storm::Specialized | Storm::Topologies) {
        // Storm B samples one format per making; storm D one format throughout.
        let formats = match storm {
            Storm::Specialized => FORMATS.to_vec(),
            _ => vec![PixelFormat::RGBA8Unorm; makings],
        };
        for &format in &formats[..makings] {
            let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            textures.push(device.new_texture(&d).map_err(e("texture"))?);
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            targets.push(device.new_texture(&d).map_err(e("target"))?);
        }
    }
    let class = match storm {
        Storm::Topologies => PrimitiveTopologyClass::Unspecified,
        _ => PrimitiveTopologyClass::Triangle,
    };
    let state = match (storm, &fixture) {
        (Storm::Specialized | Storm::Topologies, Some(fixture)) => Some(
            device
                .new_render_pipeline_state(
                    &descriptor(&fixture.vertex, &fixture.plain, None, class)?,
                    &PipelineOptions::default(),
                )
                .map_err(|error| format!("the drawn storm's state: {error}"))?,
        ),
        _ => None,
    };
    let shared = Shared {
        storm,
        device: device.clone(),
        fixture,
        draws,
        state,
        queue,
        buffers,
        sampler,
        textures,
        targets,
    };
    // The ticker's command buffer, once before the storm (anything it makes, made here).
    let fill = device
        .new_buffer(256, ResourceOptions::private())
        .map_err(e("ticker buffer"))?;
    let frame = frame_shape
        .map(|shape| Frame::new(&device, shape))
        .transpose()?;
    // Each tick's phases: `commit` returning, then the command buffer scheduled, then completed.
    let tick_once = |value: u8| -> Result<TickPhases, String> {
        let command_buffer = ticker_queue.command_buffer().map_err(e("ticker"))?;
        match &frame {
            Some(frame) => frame.encode(&command_buffer, &shared.sampler)?,
            None => {
                let mut blit = command_buffer.blit_command_encoder().map_err(e("ticker"))?;
                blit.fill_buffer(&fill, 0..256, value)
                    .map_err(e("ticker"))?;
                blit.end_encoding().map_err(e("ticker"))?;
            }
        }
        let start = Instant::now();
        command_buffer
            .commit()
            .map_err(|error| format!("commit: {error}"))?;
        let committed = start.elapsed();
        command_buffer
            .wait_until_scheduled()
            .map_err(|error| format!("wait until scheduled: {error}"))?;
        let scheduled = start.elapsed();
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| format!("a tick did not complete within {WAIT:?}"))?;
        if command_buffer.status() != metal2vulkan_runtime::CommandBufferStatus::Completed {
            return Err(format!("a tick failed: {:?}", command_buffer.error()));
        }
        Ok(TickPhases {
            committed,
            scheduled,
            completed: start.elapsed(),
        })
    };
    tick_once(0)?;
    let next = AtomicUsize::new(0);
    let done = AtomicBool::new(false);
    let start = Barrier::new(threads + 2);
    // The storm's start, set before the barrier lets the threads go.
    let began: OnceLock<Instant> = OnceLock::new();
    let (jobs, ticks) = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    if let Some(qos) = qos {
                        let _ = set_qos(qos);
                    }
                    start.wait();
                    let origin = *began
                        .get()
                        .expect("the storm's start is set before the barrier");
                    let (mut records, mut kept): (Vec<JobRecord>, Vec<_>) =
                        (Vec::new(), Vec::new());
                    loop {
                        let job = next.fetch_add(1, Ordering::SeqCst);
                        if job >= makings {
                            break;
                        }
                        let begun = origin.elapsed();
                        let outcome = shared.job(job);
                        let ended = origin.elapsed();
                        match outcome {
                            Ok(state) => {
                                kept.push(state);
                                records.push((job, begun, ended, None));
                            }
                            Err(why) => records.push((job, begun, ended, Some(why))),
                        }
                    }
                    (records, kept)
                })
            })
            .collect();
        let ticker = scope.spawn(|| {
            if let Some(qos) = qos {
                let _ = set_qos(qos);
            }
            start.wait();
            let origin = *began
                .get()
                .expect("the storm's start is set before the barrier");
            let mut ticks = Vec::new();
            let mut n = 1u8;
            while !done.load(Ordering::SeqCst) || origin.elapsed() < hold {
                let at = Instant::now();
                match tick_once(n) {
                    Ok(phases) => ticks.push((at.duration_since(origin), phases)),
                    Err(why) => return Err(why),
                }
                n = n.wrapping_add(1);
                if let Some(rest) = tick.checked_sub(at.elapsed()) {
                    std::thread::sleep(rest);
                }
            }
            Ok(ticks)
        });
        let _ = began.set(Instant::now());
        start.wait();
        let mut jobs = Vec::new();
        let mut kept = Vec::new();
        for worker in workers {
            match worker.join() {
                Ok((records, states)) => {
                    jobs.extend(records);
                    kept.extend(states);
                }
                Err(_) => jobs.push((
                    usize::MAX,
                    Duration::ZERO,
                    Duration::ZERO,
                    Some("a storm thread panicked".into()),
                )),
            }
        }
        done.store(true, Ordering::SeqCst);
        let ticks = ticker
            .join()
            .unwrap_or_else(|_| Err("the ticker panicked".into()));
        drop(kept);
        (jobs, ticks)
    });
    let origin = *began.get().ok_or("the storm never started")?;
    let ticks = ticks?;
    let messages = received.lock().map_err(|_| "the sink's lock")?.clone();
    let qos_line = format!("{qos_at_start} at start; {}", qos.unwrap_or("not set"));
    // The fills the device recorded, the ticker's among them, each one `vkCmdFillBuffer`.
    let counts = device.work_counts();
    let fills = format!("recorded={}", counts.fills_recorded);
    // What the makings' pipeline caches did by the storm's end: those seeded from their records
    // and the records written, the makings their caches held and those they compiled, and why
    // a save kept nothing (on every driver, MoltenVK's included).
    let cache = device.shader_cache_statistics();
    let driver_cache = format!(
        "loads={} saves={} pipeline_hits={} pipeline_misses={} truncated={} mismatched={} \
         rejected={} write_failures={}",
        cache.driver_loads,
        cache.driver_saves,
        cache.driver_pipeline_hits,
        cache.driver_pipeline_misses,
        cache.driver_data_truncated,
        cache.driver_data_mismatched,
        cache.rejected,
        cache.write_failures,
    );
    // The pipelines made and linked, and the libraries made, fragment shaders' among them: how many
    // fragment compiles the storm's states shared.
    let work = format!(
        "creation_pipelines_made={} variant_pipelines_made={} pipelines_linked={} \
         pipeline_libraries_made={} fragment_shader_libraries_made={}",
        counts.creation_pipelines_made,
        counts.variant_pipelines_made,
        counts.pipelines_linked,
        counts.pipeline_libraries_made,
        counts.fragment_shader_libraries_made,
    );
    report(
        &out,
        (storm, makings, threads, &budget, device.name(), &qos_line),
        origin,
        jobs,
        &ticks,
        &messages,
        &[
            ("fills", fills),
            ("driver_cache", driver_cache),
            ("work_counts", work),
        ],
    )
}

/// The ticker's frame (`--tick-frame WxH:DRAWS`, the module docs).
#[derive(Clone, Copy)]
struct FrameShape {
    width: u64,
    height: u64,
    draws: usize,
}

impl FrameShape {
    fn parse(text: &str) -> Result<FrameShape, String> {
        let bad = || format!("--tick-frame {text}: WIDTHxHEIGHT:DRAWS, each at least 1");
        let (size, draws) = text.split_once(':').ok_or_else(bad)?;
        let (width, height) = size.split_once('x').ok_or_else(bad)?;
        let shape = FrameShape {
            width: width.parse().map_err(|_| bad())?,
            height: height.parse().map_err(|_| bad())?,
            draws: draws.parse().map_err(|_| bad())?,
        };
        match shape.width > 0 && shape.height > 0 && shape.draws > 0 {
            true => Ok(shape),
            false => Err(bad()),
        }
    }
}

/// What the ticker's frame draws with: the owned `tob_rect` and `tob_sample` pipeline, a texture
/// it samples, the target, and each rectangle's inline vertex bytes.
struct Frame {
    pipeline: RenderPipelineState,
    texture: Texture,
    target: Texture,
    rects: Vec<Vec<u8>>,
}

impl Frame {
    /// The side of the texture the frame samples.
    const TEXTURE_SIDE: u64 = 256;

    fn new(device: &Device, shape: FrameShape) -> Result<Frame, String> {
        let e = |what: &'static str| {
            move |error: metal2vulkan_runtime::Error| format!("the ticker's frame: {what}: {error}")
        };
        let ll = metal2vulkan_devtools::engine_dir().join("tob_renders.ll");
        let air = metal2vulkan_devtools::llvm_assemble(&ll)
            .map_err(|error| format!("the ticker's frame: {}: {error:?}", ll.display()))?;
        use metal2vulkan_validation::air_library::AirLibrary;
        let library = device.new_library_with_air(&air).map_err(e("library"))?;
        let function = |name| -> Result<StageSource, String> {
            Ok(StageSource::Function(
                library
                    .new_function_with_name(name)
                    .map_err(e("function"))?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect")?);
        descriptor.fragment = Some(function("tob_sample")?);
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(e("pipeline"))?;
        let mut d = TextureDescriptor::texture_2d(
            PixelFormat::RGBA8Unorm,
            Frame::TEXTURE_SIDE,
            Frame::TEXTURE_SIDE,
            false,
        );
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&d).map_err(e("texture"))?;
        let mut d = TextureDescriptor::texture_2d(
            PixelFormat::RGBA8Unorm,
            shape.width,
            shape.height,
            false,
        );
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        let target = device.new_texture(&d).map_err(e("target"))?;
        // Rectangles side by side, covering the target: x0, y0, x1, y1 in clip space.
        let n = shape.draws as f32;
        let rects = (0..shape.draws)
            .map(|i| {
                let x = |at: f32| at * 2.0 / n - 1.0;
                [x(i as f32), -1.0, x(i as f32 + 1.0), 1.0]
                    .iter()
                    .flat_map(|v: &f32| v.to_le_bytes())
                    .collect()
            })
            .collect();
        Ok(Frame {
            pipeline,
            texture,
            target,
            rects,
        })
    }

    /// Encode the frame's pass into `command_buffer`, sampling through `sampler`.
    fn encode(
        &self,
        command_buffer: &metal2vulkan_runtime::CommandBuffer,
        sampler: &SamplerState,
    ) -> Result<(), String> {
        let e = |error: metal2vulkan_runtime::Error| format!("the ticker's frame: {error}");
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).map_err(e)?;
        encoder
            .set_render_pipeline_state(&self.pipeline)
            .map_err(e)?;
        encoder
            .set_fragment_sampler_state(Some(sampler), 0)
            .map_err(e)?;
        encoder
            .set_fragment_texture(Some(&self.texture), 0)
            .map_err(e)?;
        for rect in &self.rects {
            encoder.set_vertex_bytes(rect, 0).map_err(e)?;
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .map_err(e)?;
        }
        encoder.end_encoding().map_err(e)
    }
}

/// One tick's phases, each from its commit's call: `commit` returned, the command buffer was
/// scheduled (its transition ran on its queue's context), and it completed.
#[derive(Clone, Copy)]
struct TickPhases {
    committed: Duration,
    scheduled: Duration,
    completed: Duration,
}

/// Storms A to D's fixture: `--air`'s vertex function `--vertex` and fragment function
/// `--fragment`.
fn fixture(device: &Device, options: &BTreeMap<String, String>) -> Result<Fixture, String> {
    let air = required(options, "air")?;
    let bitcode = std::fs::read(air).map_err(|error| format!("read {air}: {error}"))?;
    let parsed =
        metal2vulkan::library::Library::parse(&bitcode).map_err(|error| error.to_string())?;
    let library = device.new_library_with_compile_library(parsed);
    let vertex = library
        .new_function_with_name(required(options, "vertex")?)
        .map_err(|error| format!("vertex: {error}"))?;
    let fragment = required(options, "fragment")?.to_string();
    let plain = library
        .new_function_with_name_constant_values(&fragment, &constants(0))
        .map_err(|error| format!("fragment: {error}"))?;
    Ok(Fixture {
        library,
        vertex,
        fragment,
        plain,
    })
}

/// Write the table.
fn report(
    out: &std::path::Path,
    (storm, makings, threads, budget, device, qos): (Storm, usize, usize, &str, &str, &str),
    origin: Instant,
    mut jobs: Vec<JobRecord>,
    ticks: &[(Duration, TickPhases)],
    messages: &[(Instant, DebugMessage)],
    counted: &[(&str, String)],
) -> Result<(), String> {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut text = String::new();
    let mut line = |fields: &[String]| {
        let fields: Vec<String> = fields
            .iter()
            .map(|f| f.replace(['\t', '\n'], " "))
            .collect();
        let _ = writeln!(text, "{}", fields.join("\t"));
    };
    let s = |v: &str| v.to_string();
    line(&[s("meta"), s("storm"), s(storm.name())]);
    line(&[s("meta"), s("makings"), makings.to_string()]);
    line(&[s("meta"), s("threads"), threads.to_string()]);
    line(&[s("meta"), s("budget"), s(budget)]);
    line(&[s("meta"), s("device"), s(device)]);
    line(&[s("meta"), s("qos"), s(qos)]);
    for (name, value) in counted {
        line(&[s("meta"), s(name), value.clone()]);
    }
    let mut logged = Vec::new();
    let mut other = 0usize;
    for (at, message) in messages {
        if message.id_name != CREATION_LOG {
            other += 1;
            if other <= 20 {
                line(&[
                    s("message"),
                    message.id_name.clone(),
                    message.text.chars().take(300).collect(),
                ]);
            }
            continue;
        }
        let end = at.checked_duration_since(origin);
        let making = parse(&message.text, end);
        if let Some(words) = making.words {
            line(&[
                s("meta"),
                s("fragment_words"),
                making.head.clone(),
                words.to_string(),
            ]);
        }
        logged.push(making);
    }
    line(&[s("meta"), s("other_messages"), other.to_string()]);
    let before = logged.iter().filter(|m| m.end.is_none()).count();
    line(&[s("meta"), s("makings_before_the_storm"), before.to_string()]);
    let activities = [
        "SPIR-V to MSL",
        "MSL compile",
        "MSL load",
        "library from cache",
        "function retrieval",
        "specialization",
        "pipeline compile",
    ];
    let mut storm_makings: Vec<&Making> = logged.iter().filter(|m| m.end.is_some()).collect();
    storm_makings.sort_by_key(|m| m.end);
    for (n, m) in storm_makings.iter().enumerate() {
        let mut fields = vec![
            s("making"),
            n.to_string(),
            m.head.clone(),
            format!("{:.1}", ms(m.end.unwrap_or_default())),
            m.made_ms.map_or_else(|| s("-"), |v| format!("{v:.1}")),
            format!("{:.1}", m.waited_ms),
            s(m.place),
            s(if m.failed { "failed" } else { "made" }),
            s(if m.overlapped { "overlapped" } else { "alone" }),
        ];
        for activity in activities {
            fields.push(
                m.split
                    .get(activity)
                    .map_or_else(|| s("-"), |(count, v)| format!("{count}x{v:.1}")),
            );
        }
        line(&fields);
    }
    jobs.sort_by_key(|job| job.0);
    for (job, begun, ended, error) in &jobs {
        line(&[
            s("job"),
            job.to_string(),
            format!("{:.1}", ms(*begun)),
            format!("{:.1}", ms(*ended)),
            format!("{:.1}", ms(ended.saturating_sub(*begun))),
            error.clone().unwrap_or_else(|| s("ok")),
        ]);
    }
    // Each tick: its start from the storm's, and its commit, scheduled and completed instants
    // from its commit's call.
    for (at, phases) in ticks {
        line(&[
            s("tick"),
            format!("{:.1}", ms(*at)),
            format!("{:.3}", ms(phases.committed)),
            format!("{:.3}", ms(phases.scheduled)),
            format!("{:.3}", ms(phases.completed)),
        ]);
    }
    let mut ticks: Vec<f64> = ticks
        .iter()
        .map(|(_, phases)| ms(phases.completed))
        .collect();
    ticks.sort_by(f64::total_cmp);
    line(&[
        s("ticker"),
        ticks.len().to_string(),
        format!("{:.3}", quantile(&ticks, 0.5)),
        format!("{:.3}", quantile(&ticks, 0.9)),
        format!("{:.3}", ticks.last().copied().unwrap_or(f64::NAN)),
    ]);
    let storm_ms = jobs.iter().map(|job| ms(job.2)).fold(0.0, f64::max);
    let ends: Vec<f64> = storm_makings
        .iter()
        .map(|m| ms(m.end.unwrap_or_default()))
        .collect();
    let made: Vec<f64> = storm_makings.iter().filter_map(|m| m.made_ms).collect();
    let mean = if made.is_empty() {
        f64::NAN
    } else {
        made.iter().sum::<f64>() / made.len() as f64
    };
    let spread = match (ends.first(), ends.last()) {
        (Some(first), Some(last)) => last - first,
        _ => f64::NAN,
    };
    // The largest run of makings whose ends lie within 2 ms of the next.
    let (mut group, mut largest) = (1usize, usize::from(!ends.is_empty()));
    for pair in ends.windows(2) {
        group = if pair[1] - pair[0] <= 2.0 {
            group + 1
        } else {
            1
        };
        largest = largest.max(group);
    }
    let failed = jobs.iter().filter(|job| job.3.is_some()).count();
    line(&[
        s("summary"),
        format!("storm_ms={storm_ms:.1}"),
        format!("makings={}", storm_makings.len()),
        format!(
            "per_s={:.3}",
            storm_makings.len() as f64 / (storm_ms / 1e3).max(f64::MIN_POSITIVE)
        ),
        format!("mean_made_ms={mean:.1}"),
        format!("end_spread_ms={spread:.1}"),
        format!("same_instant_group={largest}"),
        format!("jobs_failed={failed}"),
    ]);
    std::fs::write(out, text).map_err(|error| format!("write {}: {error}", out.display()))
}

#[cfg(test)]
mod tests {
    use super::{parse, ColorWriteMask, WRITE_MASKS};

    #[test]
    fn a_line_gives_its_times_place_and_split() {
        let line = "metal2vulkan: pipeline creation: state 2 (storm_vertex, storm_fragment) \
                    pipeline 1: made in 1234.5 ms after waiting 12.5 ms for its turn; made at \
                    its state's creation, the predicted key; MoltenVK, overlapped another \
                    creation (the split holds both): SPIR-V to MSL 1x 40.2 ms, MSL compile 2x \
                    900.0 ms, specialization 1x 3.5 ms; its first, key: spirv Fragment: \
                    00ff00ff00ff00ff (319844 words); topology: Triangle";
        let making = parse(line, None);
        assert_eq!(
            making.head,
            "state 2 (storm_vertex, storm_fragment) pipeline 1"
        );
        assert_eq!(making.made_ms, Some(1234.5));
        assert_eq!(making.waited_ms, 12.5);
        assert_eq!(making.place, "creation-predicted");
        assert!(making.overlapped);
        assert_eq!(making.split.get("MSL compile"), Some(&(2, 900.0)));
        assert_eq!(making.split.get("SPIR-V to MSL"), Some(&(1, 40.2)));
        assert_eq!(making.split.get("specialization"), Some(&(1, 3.5)));
        assert_eq!(making.words, Some(319_844));
    }

    #[test]
    fn the_fragment_words_whatever_the_stage_prints_as() {
        let making = parse(
            "metal2vulkan: pipeline creation: state 1 (v, f) pipeline 1: made in 9.0 ms; its \
             first, key: fixed: Fixed; spirv FRAGMENT: 0011223344556677 (1234 words); spirv \
             VERTEX: 8899aabbccddeeff (99 words); variant: V",
            None,
        );
        assert_eq!(making.words, Some(1234));
    }

    /// Storm E's later makings of a draw are pipelines of their own: each write mask differs
    /// from the others and from `ALL`.
    #[test]
    fn storm_e_write_masks_differ_from_each_other_and_from_all() {
        let raw: std::collections::BTreeSet<u32> =
            WRITE_MASKS.iter().map(|mask| mask.as_raw()).collect();
        assert_eq!(raw.len(), WRITE_MASKS.len());
        assert!(!raw.contains(&ColorWriteMask::ALL.as_raw()));
        assert!(!raw.contains(&ColorWriteMask::NONE.as_raw()));
    }

    #[test]
    fn a_line_without_a_split_or_a_wait() {
        let making = parse(
            "metal2vulkan: pipeline creation: state 1 (v, f) pipeline 3: made in 7.0 ms; differs \
             from the state's pipeline 1 in 1 field(s): spec 4 TexelFormat(…): 0 -> 5",
            None,
        );
        assert_eq!(making.made_ms, Some(7.0));
        assert_eq!(making.waited_ms, 0.0);
        assert_eq!(making.place, "draw");
        assert!(!making.overlapped);
        assert!(making.split.is_empty());
        assert_eq!(making.words, None);
    }
}
