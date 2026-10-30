//! The `metal2vulkan` command-line tool (feature `cli`): a metallib, AIR bitcode, or LLVM text in;
//! the SPIR-V of a compute or render pipeline compiled for a Khronos device profile out, through
//! the library's API (`Device::compile_compute_pipeline`, `compile_render_pipeline`, and
//! `compile_tile_pipeline`).
//!
//! Usage: `metal2vulkan <in.metallib|in.air|in.ll> [out.spv] --profile <profile.json> [options]`.
//! A success prints a line containing `PASS`. Any failure prints `metal2vulkan: FALLBACK (why)`
//! and exits with status 2; a failure after the inputs are read also writes a repro bundle
//! (`METAL2VULKAN_REPRO_DIR`, by default `$TMPDIR/metal2vulkan-repros`). The tool writes no
//! scratch file: `llvm-as` and `spirv-val` run over pipes.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{env, fs};

use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::error::{DescriptorError, Error, RuntimeInput};
use metal2vulkan::library::{
    Function, FunctionConstantValue, FunctionConstantValues, FunctionStage, Library,
};
use metal2vulkan::metal::{DataType, PixelFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipeline, ComputePipelineDescriptor,
    ComputePipelineMetalInfo, FunctionDescriptor, GraphicsState, PipelineInterface, PipelineLayout,
    PrecisionNote, Reconvergence, RenderPipeline, RenderPipelineDescriptor, RuntimeDependency,
    SpecializationLayout, StageInputDescriptor, StageModule, StageSource,
    TileRenderPipelineDescriptor, VertexDescriptor, VertexInputState,
};
use metal2vulkan::{vk, Budget};

const USAGE: &str =
    "usage: metal2vulkan <in.metallib|in.air|in.ll> [out.spv] --profile <profile.json> [options]";

const HELP: &str = r#"usage: metal2vulkan <in.metallib|in.air|in.ll> [out.spv] --profile <profile.json> [options]

Compiles one pipeline of the input library for the device a Khronos profile describes, and writes
its SPIR-V. A kernel writes out.spv; a render pipeline writes one module per stage, named by
inserting the stage before the extension (out.vert.spv, out.frag.spv; a post-tessellation vertex
function adds out.tesc.spv and out.tese.spv on a device with tessellationShader, and is out.vert.spv
itself, drawing the tessellator's triangles, on one without). Without out.spv the output is
<input-without-extension>.vk.spv. A .ll input is assembled with llvm-as (LLVM 21 or newer).

options:
  --profile FILE           Khronos profiles JSON for the target device (required; the repository's
                           crates/metal2vulkan-validation/profiles/ holds examples)
  --profile-name NAME      the profile to read when FILE holds more than one
  --function NAME          the kernel or vertex function (default: the library's only one)
  --fragment NAME          the fragment function of a render pipeline
  --constant ID:TYPE:VALUE function-constant value (repeatable): ID is an index or a name, TYPE a
                           Metal data type (uint, float4, bool, half2, ...), VALUE comma-separated
                           components (decimal, true/false, or 0x raw bits)
  --stage-input FILE       a kernel's stage-input descriptor (the JSON form of
                           metal2vulkan::pipeline::StageInputDescriptor)
  --vertex-descriptor FILE a render pipeline's vertex descriptor (the JSON form of
                           metal2vulkan::pipeline::VertexDescriptor)
  --color-format I:FORMAT  pixel format of color attachment I (repeatable), a Metal pixel format
                           name (RGBA8Unorm, ...); with a kernel, compiles a tile pipeline
  --depth-format FORMAT    the depth attachment's pixel format
  --stencil-format FORMAT  the stencil attachment's pixel format
  --raster-samples N       the render pipeline's sample count (default: 1)
  --vertex-amplification-count N
                           the render pipeline's maximum vertex amplification count (default: 1)
  --timeout-ms N           the compile's deadline (default: 20000)
  --emit-meta FILE         write the pipeline's interface, layout, dependencies, and stages as JSON
  --no-spirv-val           skip spirv-val (by default it runs when found)
  -h, --help               show this help
"#;

/// The options that name a descriptor file, and the name of its copy in a repro bundle.
const DESCRIPTOR_FILES: [(&str, &str); 2] = [
    ("--stage-input", "stage-input.json"),
    ("--vertex-descriptor", "vertex-descriptor.json"),
];

/// What the command line asks for.
#[derive(Default)]
struct Args {
    input: Option<String>,
    output: Option<String>,
    profile: Option<String>,
    profile_name: Option<String>,
    function: Option<String>,
    fragment: Option<String>,
    constants: Vec<(String, FunctionConstantValue)>,
    stage_input: Option<String>,
    vertex_descriptor: Option<String>,
    color_formats: BTreeMap<usize, PixelFormat>,
    depth_format: Option<PixelFormat>,
    stencil_format: Option<PixelFormat>,
    raster_samples: Option<u32>,
    vertex_amplification_count: Option<u32>,
    timeout: Option<Duration>,
    emit_meta: Option<String>,
    spirv_val: bool,
    /// The options a repro bundle replays: every one but `--profile` and `--emit-meta`, with each
    /// descriptor file named as the bundle's copy of it ([`DESCRIPTOR_FILES`]).
    replay: Vec<String>,
}

/// The pipeline the inputs compile to, boxed: the two differ in size by hundreds of bytes.
enum Compiled {
    Compute(Box<ComputePipeline>),
    Render(Box<RenderPipeline>),
}

impl Compiled {
    fn stages(&self) -> Vec<&StageModule> {
        match self {
            Compiled::Compute(p) => vec![&p.stage],
            Compiled::Render(p) => p.stages.iter().collect(),
        }
    }
}

/// A failure after the inputs are read: its one-line reason and the detail its bundle records.
struct Failure {
    why: String,
    detail: Option<String>,
}

impl Failure {
    fn new(why: impl Into<String>) -> Failure {
        Failure {
            why: why.into(),
            detail: None,
        }
    }

    fn detailed(why: impl Into<String>, detail: impl Into<String>) -> Failure {
        Failure {
            why: why.into(),
            detail: Some(detail.into()),
        }
    }
}

fn main() {
    let original: Vec<String> = env::args().collect();
    let args = parse_args(original.get(1..).unwrap_or_default());
    let Some(input) = args.input.clone() else {
        eprintln!("{USAGE}");
        fail("no input");
    };
    let Some(profile_path) = args.profile.clone() else {
        fail(
            "no --profile: name a Khronos profiles JSON for the target device (the repository's \
             crates/metal2vulkan-validation/profiles/ holds examples, and the Vulkan SDK and \
             vulkaninfo --json produce more)",
        );
    };
    let source = fs::read(&input).unwrap_or_else(|e| fail(&format!("cannot read {input}: {e}")));
    let profile_json = fs::read_to_string(&profile_path)
        .unwrap_or_else(|e| fail(&format!("cannot read {profile_path}: {e}")));
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(&input));
    let descriptors = [&args.stage_input, &args.vertex_descriptor]
        .into_iter()
        .zip(DESCRIPTOR_FILES)
        .filter_map(|(path, (_, name))| Some((path.as_deref()?, name)))
        .collect::<Vec<_>>();
    let bundle = Bundle {
        original: &original,
        input: &input,
        source: &source,
        profile_json: &profile_json,
        descriptors: &descriptors,
        output: &output,
        replay: &args.replay,
    };
    match run(&args, &source, &profile_json, &input, &output) {
        Ok(line) => println!("{line}"),
        Err(failure) => {
            if let Some(detail) = &failure.detail {
                eprintln!("metal2vulkan: {detail}");
            }
            match bundle.write(&failure) {
                Ok(dir) => eprintln!("metal2vulkan: wrote repro {}", dir.display()),
                Err(e) => eprintln!("metal2vulkan: repro write failed: {e}"),
            }
            fail(&failure.why);
        }
    }
}

/// Compile, write, and check; the success line, or why not.
fn run(
    args: &Args,
    source: &[u8],
    profile_json: &str,
    input: &str,
    output: &str,
) -> Result<String, Failure> {
    let description =
        DeviceDescription::from_khronos_json(profile_json, args.profile_name.as_deref())
            .map_err(|e| Failure::detailed("profile not read", e.to_string()))?;
    let device = Device::new(description, DevicePolicy::default())
        .map_err(|e| Failure::detailed("profile refused", e.to_string()))?;
    let assembled;
    let bitcode: &[u8] = if input.ends_with(".ll") {
        assembled = assemble(source)?;
        &assembled
    } else {
        source
    };
    let library =
        Library::parse(bitcode).map_err(|e| Failure::detailed("input not read", e.to_string()))?;
    let compiled = compile(args, &device, &library)?;

    let stages = compiled.stages();
    let paths: Vec<String> = stages
        .iter()
        .map(|stage| match &compiled {
            Compiled::Compute(_) => output.to_string(),
            Compiled::Render(_) => stage_output_path(output, stage_suffix(stage.stage)),
        })
        .collect();
    // Every module is checked before any is written: a failure leaves no output. The library
    // states what a module of the device is valid under: each target environment, with the
    // relaxations the device's features give.
    let spirv_val = if args.spirv_val {
        let plan = device.profile().spirv_validation();
        let mut found = true;
        for (stage, path) in stages.iter().zip(&paths) {
            for env_name in &plan.target_envs {
                match spirv_val(&stage.spirv_bytes(), env_name, &plan.relaxations) {
                    Ok(true) => {}
                    Ok(false) => found = false,
                    Err(detail) => {
                        return Err(Failure::detailed(
                            format!("spirv-val {env_name} rejected {path}"),
                            detail,
                        ))
                    }
                }
            }
        }
        if found {
            format!("; spirv-val {}: PASS", plan.target_envs.join(", "))
        } else {
            "; spirv-val: not found".to_string()
        }
    } else {
        String::new()
    };
    for (stage, path) in stages.iter().zip(&paths) {
        fs::write(path, stage.spirv_bytes())
            .map_err(|e| Failure::detailed("write failed", format!("cannot write {path}: {e}")))?;
    }
    if let Some(meta) = &args.emit_meta {
        write_meta(meta, &compiled, &device, &stages, &paths)?;
    }
    let mut line = format!(
        "metal2vulkan: wrote {}; verified: PASS{spirv_val}",
        paths.join(", ")
    );
    if let Some(meta) = &args.emit_meta {
        line.push_str(&format!("\nmetal2vulkan: wrote meta {meta}"));
    }
    Ok(line)
}

/// The pipeline `args` select from `library`, compiled on `device`.
fn compile(args: &Args, device: &Device, library: &Library) -> Result<Compiled, Failure> {
    let function = select(library, args.function.as_deref())?;
    let stage = function.info().stage;
    let render_options = args.fragment.is_some()
        || args.vertex_descriptor.is_some()
        || args.depth_format.is_some()
        || args.stencil_format.is_some()
        || args.raster_samples.is_some()
        || args.vertex_amplification_count.is_some();
    // A deadline past what `Instant` can hold is no deadline.
    let budget = Budget::with_deadline(
        Instant::now().checked_add(args.timeout.unwrap_or(Duration::from_secs(20))),
    );
    let constants = constant_values(&args.constants);
    match stage {
        FunctionStage::Kernel => {
            if render_options {
                return Err(Failure::new(
                    "a kernel takes no render pipeline option (--fragment, --vertex-descriptor, \
                     --depth-format, --stencil-format, --raster-samples, \
                     --vertex-amplification-count)",
                ));
            }
            let kernel = with_constants(function, &constants);
            if args.color_formats.is_empty() {
                let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel));
                descriptor.stage_input = match &args.stage_input {
                    Some(path) => Some(read_json::<StageInputDescriptor>(path)?),
                    None => None,
                };
                device
                    .compile_compute_pipeline(&descriptor, &budget)
                    .map(|p| Compiled::Compute(Box::new(p)))
                    .map_err(refusal)
            } else {
                if args.stage_input.is_some() {
                    return Err(Failure::new(
                        "a tile pipeline (a kernel with --color-format) takes no --stage-input",
                    ));
                }
                let mut descriptor =
                    TileRenderPipelineDescriptor::new(StageSource::Function(kernel));
                for (&index, &format) in &args.color_formats {
                    descriptor.color_attachment_formats[index] = Some(format);
                }
                device
                    .compile_tile_pipeline(&descriptor, &budget)
                    .map(|p| Compiled::Compute(Box::new(p)))
                    .map_err(refusal)
            }
        }
        FunctionStage::Vertex => {
            if args.stage_input.is_some() {
                return Err(Failure::new(
                    "--stage-input is a kernel's; a vertex function takes --vertex-descriptor",
                ));
            }
            let vertex = with_constants(function, &constants);
            let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
            if let Some(name) = &args.fragment {
                let fragment = library
                    .function(name)
                    .map_err(|e| Failure::detailed("no such fragment function", e.to_string()))?;
                if fragment.info().stage != FunctionStage::Fragment {
                    return Err(Failure::new(format!(
                        "--fragment {name} is a {:?} function",
                        fragment.info().stage
                    )));
                }
                descriptor.fragment =
                    Some(StageSource::Function(with_constants(fragment, &constants)));
            }
            descriptor.vertex_descriptor = match &args.vertex_descriptor {
                Some(path) => Some(read_json::<VertexDescriptor>(path)?),
                None => None,
            };
            for (&index, &format) in &args.color_formats {
                descriptor.color_attachments[index] = Some(ColorAttachmentDescriptor::new(format));
            }
            descriptor.depth_format = args.depth_format;
            descriptor.stencil_format = args.stencil_format;
            if let Some(samples) = args.raster_samples {
                descriptor.raster_sample_count = samples;
            }
            if let Some(count) = args.vertex_amplification_count {
                descriptor.max_vertex_amplification_count = count;
            }
            device
                .compile_render_pipeline(&descriptor, &budget)
                .map(|p| Compiled::Render(Box::new(p)))
                .map_err(refusal)
        }
        FunctionStage::Fragment => Err(Failure::new(format!(
            "{} is a fragment function, which compiles in a render pipeline: name its vertex \
             function with --function and this one with --fragment",
            function.info().name
        ))),
        other => Err(Failure::new(format!(
            "{} is a {other:?} function; the tool compiles kernels and render pipelines",
            function.info().name
        ))),
    }
}

/// The function `name` names, or the library's only kernel or vertex function.
fn select(library: &Library, name: Option<&str>) -> Result<Function, Failure> {
    if let Some(name) = name {
        return library
            .function(name)
            .map_err(|e| Failure::detailed(format!("no function {name}"), e.to_string()));
    }
    let starts: Vec<u32> = (0u32..)
        .zip(library.functions())
        .filter(|(_, f)| matches!(f.stage, FunctionStage::Kernel | FunctionStage::Vertex))
        .map(|(index, _)| index)
        .collect();
    match starts[..] {
        [index] => library
            .function_at(index)
            .map_err(|e| Failure::detailed("input not read", e.to_string())),
        _ => {
            let listed: Vec<String> = library
                .functions()
                .map(|f| format!("{} ({:?})", f.name, f.stage))
                .collect();
            Err(Failure::new(format!(
                "the library has {} kernel or vertex functions; name one with --function \
                 (functions: {})",
                starts.len(),
                listed.join(", ")
            )))
        }
    }
}

fn with_constants(function: Function, constants: &FunctionConstantValues) -> FunctionDescriptor {
    let mut descriptor = FunctionDescriptor::new(function);
    descriptor.constants = constants.clone();
    descriptor
}

fn constant_values(constants: &[(String, FunctionConstantValue)]) -> FunctionConstantValues {
    let mut values = FunctionConstantValues::defaults();
    for (id, value) in constants {
        values = match id.parse::<u32>() {
            Ok(index) => values.with_index(index, value.clone()),
            Err(_) => values.with_name(id.clone(), value.clone()),
        };
    }
    values
}

/// A compile error as a failure, naming the option that supplies a missing pipeline input.
fn refusal(error: Error) -> Failure {
    let hint = match &error {
        Error::NeedsInput(RuntimeInput::FunctionConstant { index }) => {
            format!("; supply it with --constant {index}:TYPE:VALUE")
        }
        Error::NeedsInput(RuntimeInput::StageInputDescriptor { .. }) => {
            "; supply a stage-input descriptor with --stage-input".to_string()
        }
        Error::NeedsInput(RuntimeInput::VertexDescriptor { .. })
        | Error::InvalidDescriptor(DescriptorError::VertexAttribute { .. }) => {
            "; give each attribute a format with --vertex-descriptor".to_string()
        }
        Error::NeedsInput(RuntimeInput::TileAttachments) => {
            "; a tile function compiles as a tile pipeline: give its attachments with \
             --color-format"
                .to_string()
        }
        _ => String::new(),
    };
    Failure::new(format!("compile failed: {error}{hint}"))
}

/// `spirv-val --target-env <env>` on `spirv` over a pipe: `Ok(true)` when it accepts the module,
/// `Ok(false)` when there is no `spirv-val` to run, and its diagnostic when it rejects it.
fn spirv_val(spirv: &[u8], env_name: &str, relaxations: &[&str]) -> Result<bool, String> {
    let mut command = tool("spirv-val");
    command
        .args(["--target-env", env_name])
        .args(relaxations)
        .arg("-");
    match pipe(command, spirv) {
        Ok(_) => Ok(true),
        Err(ToolFailure::NotFound) => Ok(false),
        Err(ToolFailure::Failed(detail)) => Err(detail),
    }
}

/// `source` (LLVM text) assembled to bitcode by `llvm-as` over a pipe.
fn assemble(source: &[u8]) -> Result<Vec<u8>, Failure> {
    let mut command = tool("llvm-as");
    command.args(["-o", "-", "-"]);
    pipe(command, source).map_err(|failure| match failure {
        ToolFailure::NotFound => Failure::new(
            "a .ll input needs llvm-as (LLVM 21 or newer) on PATH or in METAL2VULKAN_LLVM_AS",
        ),
        ToolFailure::Failed(detail) => Failure::detailed("llvm-as failed", detail),
    })
}

enum ToolFailure {
    NotFound,
    Failed(String),
}

/// The command for external tool `name`: `METAL2VULKAN_<NAME>` when set, else `name` on `PATH`.
fn tool(name: &str) -> Command {
    let program = metal2vulkan::env_vars::tool_path_override(name).unwrap_or_else(|| name.into());
    Command::new(program)
}

/// Run `command` with `input` on its standard input; its standard output, or why not.
fn pipe(mut command: Command, input: &[u8]) -> Result<Vec<u8>, ToolFailure> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => ToolFailure::NotFound,
            _ => ToolFailure::Failed(format!("cannot run {command:?}: {e}")),
        })?;
    // Write from a thread so a tool that answers before reading all of its input cannot block on
    // a full output pipe while this process blocks on a full input pipe.
    let mut stdin = child.stdin.take();
    let input = input.to_vec();
    let writer = std::thread::spawn(move || match stdin.as_mut() {
        Some(pipe) => pipe.write_all(&input),
        None => Ok(()),
    });
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let read = child
        .stdout
        .take()
        .map_or(Ok(0), |mut out| out.read_to_end(&mut stdout))
        .and_then(|_| {
            child
                .stderr
                .take()
                .map_or(Ok(0), |mut err| err.read_to_end(&mut stderr))
        });
    let status = child.wait();
    let written = writer.join();
    let status = status.map_err(|e| ToolFailure::Failed(format!("{command:?}: {e}")))?;
    if !status.success() {
        return Err(ToolFailure::Failed(format!(
            "{command:?} exited with {status}:\n{}{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        )));
    }
    read.map_err(|e| ToolFailure::Failed(format!("{command:?}: {e}")))?;
    match written {
        Ok(Ok(())) => Ok(stdout),
        Ok(Err(e)) => Err(ToolFailure::Failed(format!("{command:?}: {e}"))),
        Err(_) => Err(ToolFailure::Failed(format!(
            "{command:?}: the input writer panicked"
        ))),
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, Failure> {
    let text = fs::read_to_string(path)
        .map_err(|e| Failure::detailed(format!("cannot read {path}"), e.to_string()))?;
    serde_json::from_str(&text)
        .map_err(|e| Failure::detailed(format!("{path} does not parse"), e.to_string()))
}

/// The pipeline as `--emit-meta` writes it.
#[derive(serde::Serialize)]
struct Meta<'a> {
    kind: &'static str,
    /// SHA-256 of the device profile it was compiled for, as hex.
    profile: String,
    stages: Vec<StageMeta<'a>>,
    layout: &'a PipelineLayout,
    interface: &'a PipelineInterface,
    dependencies: &'a [RuntimeDependency],
    #[serde(skip_serializing_if = "Option::is_none")]
    metal_info: Option<&'a ComputePipelineMetalInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vertex_input: Option<&'a VertexInputState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    graphics: Option<&'a GraphicsState>,
    variant_bits: u32,
}

#[derive(serde::Serialize)]
struct StageMeta<'a> {
    stage: &'static str,
    spirv: &'a str,
    entry_point: &'a str,
    specialization: &'a SpecializationLayout,
    required_subgroup_size: Option<u32>,
    stage_flags: vk::PipelineShaderStageCreateFlags,
    reconvergence: Reconvergence,
    precision_notes: &'a [PrecisionNote],
    /// The constant data the stage reads at its constant-blob binding, as hex.
    #[serde(skip_serializing_if = "Option::is_none")]
    constant_blob: Option<String>,
}

fn write_meta(
    path: &str,
    compiled: &Compiled,
    device: &Device,
    stages: &[&StageModule],
    paths: &[String],
) -> Result<(), Failure> {
    let stages = stages
        .iter()
        .zip(paths)
        .map(|(stage, spirv)| StageMeta {
            stage: stage_suffix(stage.stage),
            spirv,
            entry_point: &stage.entry_point,
            specialization: &stage.specialization,
            required_subgroup_size: stage.required_subgroup_size,
            stage_flags: stage.stage_flags,
            reconvergence: stage.reconvergence,
            precision_notes: &stage.precision_notes,
            constant_blob: stage.constant_blob.as_deref().map(hex),
        })
        .collect();
    let profile = hex(&device.profile().fingerprint());
    let meta = match compiled {
        Compiled::Compute(p) => Meta {
            kind: "compute",
            profile,
            stages,
            layout: &p.layout,
            interface: &p.interface,
            dependencies: &p.dependencies,
            metal_info: Some(&p.metal_info),
            vertex_input: None,
            graphics: None,
            variant_bits: p.variant_bits,
        },
        Compiled::Render(p) => Meta {
            kind: "render",
            profile,
            stages,
            layout: &p.layout,
            interface: &p.interface,
            dependencies: &p.dependencies,
            metal_info: None,
            vertex_input: Some(&p.vertex_input),
            graphics: Some(&p.graphics),
            variant_bits: p.variant_bits,
        },
    };
    let json = serde_json::to_string_pretty(&meta)
        .map_err(|e| Failure::detailed("emit-meta failed", e.to_string()))?;
    fs::write(path, json)
        .map_err(|e| Failure::detailed("emit-meta failed", format!("cannot write {path}: {e}")))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The file-name suffix of a pipeline stage (glslang's stage extensions).
fn stage_suffix(stage: vk::ShaderStage) -> &'static str {
    match stage {
        vk::ShaderStage::VERTEX => "vert",
        vk::ShaderStage::TESSELLATION_CONTROL => "tesc",
        vk::ShaderStage::TESSELLATION_EVALUATION => "tese",
        vk::ShaderStage::FRAGMENT => "frag",
        vk::ShaderStage::COMPUTE => "comp",
        _ => "stage",
    }
}

/// `output` with `.<suffix>` inserted before its extension (`out.spv` -> `out.vert.spv`).
fn stage_output_path(output: &str, suffix: &str) -> String {
    let path = Path::new(output);
    match path.extension().and_then(|e| e.to_str()) {
        Some(extension) => path
            .with_extension(format!("{suffix}.{extension}"))
            .to_string_lossy()
            .into_owned(),
        None => format!("{output}.{suffix}"),
    }
}

fn default_output_path(input: &str) -> String {
    Path::new(input)
        .with_extension("vk.spv")
        .to_string_lossy()
        .into_owned()
}

fn parse_args(args: &[String]) -> Args {
    let mut parsed = Args {
        spirv_val: true,
        ..Args::default()
    };
    let mut positional = Vec::new();
    let mut i = 0;
    while let Some(arg) = args.get(i) {
        let value = || {
            args.get(i + 1)
                .filter(|v| !v.starts_with("--"))
                .cloned()
                .unwrap_or_else(|| fail(&format!("{arg} requires a value")))
        };
        let mut replayed = true;
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                print!("{}", metal2vulkan::env_vars::help_text());
                process::exit(0);
            }
            "--no-spirv-val" => {
                parsed.spirv_val = false;
                parsed.replay.push(arg.clone());
                i += 1;
                continue;
            }
            "--profile" => {
                parsed.profile = Some(value());
                replayed = false;
            }
            "--profile-name" => parsed.profile_name = Some(value()),
            "--function" => parsed.function = Some(value()),
            "--fragment" => parsed.fragment = Some(value()),
            "--constant" => {
                let constant = value();
                parsed
                    .constants
                    .push(parse_constant(&constant).unwrap_or_else(|e| fail(&e)));
            }
            "--stage-input" => parsed.stage_input = Some(value()),
            "--vertex-descriptor" => parsed.vertex_descriptor = Some(value()),
            "--color-format" => {
                let spec = value();
                let (index, format) = spec
                    .split_once(':')
                    .and_then(|(i, f)| Some((i.parse::<usize>().ok().filter(|&i| i < 8)?, f)))
                    .unwrap_or_else(|| fail("--color-format requires INDEX:FORMAT, INDEX 0-7"));
                parsed
                    .color_formats
                    .insert(index, pixel_format(format).unwrap_or_else(|e| fail(&e)));
            }
            "--depth-format" => {
                parsed.depth_format = Some(pixel_format(&value()).unwrap_or_else(|e| fail(&e)));
            }
            "--stencil-format" => {
                parsed.stencil_format = Some(pixel_format(&value()).unwrap_or_else(|e| fail(&e)));
            }
            "--raster-samples" => {
                parsed.raster_samples = Some(
                    value()
                        .parse()
                        .unwrap_or_else(|_| fail("--raster-samples requires a sample count")),
                );
            }
            "--vertex-amplification-count" => {
                parsed.vertex_amplification_count = Some(
                    value()
                        .parse()
                        .unwrap_or_else(|_| fail("--vertex-amplification-count requires a count")),
                );
            }
            "--timeout-ms" => {
                parsed.timeout =
                    Some(Duration::from_millis(value().parse().unwrap_or_else(
                        |_| fail("--timeout-ms requires milliseconds"),
                    )));
            }
            "--emit-meta" => {
                parsed.emit_meta = Some(value());
                replayed = false;
            }
            // A leading dash is always an option: an unknown one is a mistake, never a path (a
            // misspelled option must not become an output file name).
            other if other.starts_with('-') && other.len() > 1 => fail(&format!(
                "unrecognized option `{other}`; the input and output are positional, and --help \
                 lists every option"
            )),
            _ => {
                positional.push(arg.clone());
                i += 1;
                continue;
            }
        }
        if replayed {
            let copy = DESCRIPTOR_FILES.iter().find(|(flag, _)| flag == arg);
            let value = copy.map_or_else(|| args[i + 1].clone(), |(_, name)| name.to_string());
            parsed.replay.extend([arg.clone(), value]);
        }
        i += 2;
    }
    let mut positional = positional.into_iter();
    parsed.input = positional.next();
    parsed.output = positional.next();
    if let Some(extra) = positional.next() {
        fail(&format!("unexpected argument `{extra}`; {USAGE}"));
    }
    parsed
}

/// A Metal pixel format by its name, with or without the `MTLPixelFormat` prefix.
fn pixel_format(name: &str) -> Result<PixelFormat, String> {
    let name = name.strip_prefix("MTLPixelFormat").unwrap_or(name);
    PixelFormat::from_name(&format!("MTLPixelFormat{name}"))
        .ok_or_else(|| format!("unknown Metal pixel format `{name}`"))
}

/// `ID:TYPE:VALUE[,VALUE...]` as a constant's index or name and its value.
fn parse_constant(spec: &str) -> Result<(String, FunctionConstantValue), String> {
    let mut parts = spec.splitn(3, ':');
    let (Some(id), Some(ty), Some(values)) = (parts.next(), parts.next(), parts.next()) else {
        return Err(format!("--constant {spec}: expected ID:TYPE:VALUE"));
    };
    let name = ty.strip_prefix("MTLDataType").unwrap_or(ty);
    let data_type = DataType::ALL
        .iter()
        .copied()
        .find(|t| {
            t.name()
                .strip_prefix("MTLDataType")
                .is_some_and(|n| n.eq_ignore_ascii_case(name))
        })
        .ok_or_else(|| format!("--constant {spec}: unknown Metal data type `{ty}`"))?;
    let (scalar, lanes) = scalar_of(data_type)
        .ok_or_else(|| format!("--constant {spec}: {ty} is not a scalar or vector type"))?;
    let components: Vec<&str> = values.split(',').map(str::trim).collect();
    if components.len() != lanes {
        return Err(format!(
            "--constant {spec}: {ty} takes {lanes} components, not {}",
            components.len()
        ));
    }
    let mut bytes = Vec::new();
    for component in components {
        let bits = scalar
            .bits(component)
            .ok_or_else(|| format!("--constant {spec}: `{component}` is not a {ty} component"))?;
        bytes.extend_from_slice(&bits.to_le_bytes()[..scalar.width()]);
    }
    Ok((id.to_string(), FunctionConstantValue::new(data_type, bytes)))
}

/// The scalar kinds of Metal's function-constant data types.
#[derive(Clone, Copy)]
enum Scalar {
    Bool,
    Signed(usize),
    Unsigned(usize),
    Half,
    BFloat,
    Float,
}

/// The scalar kind and lane count of `ty`, from its Metal name (`MTLDataTypeUInt4`).
fn scalar_of(ty: DataType) -> Option<(Scalar, usize)> {
    let name = ty.name().strip_prefix("MTLDataType")?;
    let (base, lanes) = match name.as_bytes().last() {
        Some(digit @ b'2'..=b'4') => (&name[..name.len() - 1], usize::from(digit - b'0')),
        _ => (name, 1),
    };
    let scalar = match base {
        "Bool" => Scalar::Bool,
        "Char" => Scalar::Signed(1),
        "UChar" => Scalar::Unsigned(1),
        "Short" => Scalar::Signed(2),
        "UShort" => Scalar::Unsigned(2),
        "Int" => Scalar::Signed(4),
        "UInt" => Scalar::Unsigned(4),
        "Long" => Scalar::Signed(8),
        "ULong" => Scalar::Unsigned(8),
        "Half" => Scalar::Half,
        "BFloat" => Scalar::BFloat,
        "Float" => Scalar::Float,
        _ => return None,
    };
    Some((scalar, lanes))
}

impl Scalar {
    /// Bytes per component.
    fn width(self) -> usize {
        match self {
            Scalar::Bool => 1,
            Scalar::Signed(w) | Scalar::Unsigned(w) => w,
            Scalar::Half | Scalar::BFloat => 2,
            Scalar::Float => 4,
        }
    }

    /// The little-endian bits of one component written as `text`: `0x` raw bits of the
    /// component's width, or its decimal (or `true`/`false`) value.
    fn bits(self, text: &str) -> Option<u64> {
        let width_bits = 8 * self.width() as u32;
        let fits = |bits: u64| width_bits == 64 || bits >> width_bits == 0;
        if let Some(raw) = text.strip_prefix("0x") {
            return u64::from_str_radix(raw, 16).ok().filter(|&b| fits(b));
        }
        match self {
            Scalar::Bool => match text {
                "true" | "1" => Some(1),
                "false" | "0" => Some(0),
                _ => None,
            },
            Scalar::Signed(w) => {
                let v = text.parse::<i64>().ok()?;
                let bits = 8 * w as u32;
                let in_range = bits == 64 || (v >> (bits - 1) == 0 || v >> (bits - 1) == -1);
                in_range.then(|| (v as u64) & (u64::MAX >> (64 - bits)))
            }
            Scalar::Unsigned(_) => text.parse::<u64>().ok().filter(|&b| fits(b)),
            Scalar::Half => Some(u64::from(f32_to_f16(text.parse().ok()?))),
            Scalar::BFloat => Some(u64::from(f32_to_bf16(text.parse().ok()?))),
            Scalar::Float => Some(u64::from(text.parse::<f32>().ok()?.to_bits())),
        }
    }
}

/// `value` rounded to the nearest `half` (ties to even), as Metal converts a `float` literal.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        let nan = if mantissa == 0 {
            0
        } else {
            0x200 | (mantissa >> 13) as u16
        };
        return sign | 0x7c00 | nan;
    }
    let biased = exponent - 127 + 15;
    if biased >= 0x1f {
        return sign | 0x7c00;
    }
    // A subnormal or zero half keeps the implicit bit in its mantissa.
    let (significand, shift) = if biased <= 0 {
        if biased < -10 {
            return sign;
        }
        (mantissa | 0x80_0000, (14 - biased) as u32)
    } else {
        ((biased as u32) << 23 | mantissa, 13)
    };
    let kept = significand >> shift;
    let rest = significand & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    // A carry out of the mantissa raises the exponent, up to infinity, as rounding does.
    let rounded = if rest > half || (rest == half && kept & 1 == 1) {
        kept + 1
    } else {
        kept
    };
    sign | rounded as u16
}

/// `value` rounded to the nearest `bfloat` (ties to even).
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// What a repro bundle replays.
struct Bundle<'a> {
    original: &'a [String],
    input: &'a str,
    source: &'a [u8],
    profile_json: &'a str,
    /// Each descriptor file the command named, and the name of its copy.
    descriptors: &'a [(&'a str, &'a str)],
    output: &'a str,
    replay: &'a [String],
}

impl Bundle<'_> {
    /// A directory holding the input, the profile, the descriptor files, the command that replays
    /// the failure on them, and why it failed.
    fn write(&self, failure: &Failure) -> Result<PathBuf, String> {
        let base = metal2vulkan::env_vars::repro_dir()
            .map(PathBuf::from)
            .unwrap_or_else(|| env::temp_dir().join("metal2vulkan-repros"));
        fs::create_dir_all(&base).map_err(|e| format!("create {}: {e}", base.display()))?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("time before UNIX_EPOCH: {e}"))?
            .as_nanos();
        let dir = base.join(format!("metal2vulkan-repro-{}-{stamp}", process::id()));
        fs::create_dir(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;

        let input_name = repro_input_name(self.input);
        let write = |name: &str, bytes: &[u8]| {
            fs::write(dir.join(name), bytes).map_err(|e| format!("write {name}: {e}"))
        };
        write(&input_name, self.source)?;
        write("profile.json", self.profile_json.as_bytes())?;
        for (path, name) in self.descriptors {
            // A descriptor file that cannot be read is the failure the bundle records: its replay
            // reads the missing copy and fails the same way.
            if let Ok(bytes) = fs::read(path) {
                write(name, &bytes)?;
            }
        }
        let options = if self.replay.is_empty() {
            String::new()
        } else {
            format!(" {}", shell_join(self.replay))
        };
        write(
            "repro-command.txt",
            format!("metal2vulkan {input_name} out.spv --profile profile.json{options}\n")
                .as_bytes(),
        )?;
        write(
            "repro.sh",
            format!(
                "#!/usr/bin/env sh\nset -eu\ncd -- \"$(dirname -- \"$0\")\"\n\
                 metal2vulkan {input_name} out.spv --profile profile.json{options}\n"
            )
            .as_bytes(),
        )?;
        make_executable(&dir.join("repro.sh"))?;
        let mut text = format!(
            "why: {}\nsource: {}\noutput: {}\noriginal_command: {}\n",
            failure.why,
            self.input,
            self.output,
            shell_join(self.original)
        );
        if let Some(detail) = &failure.detail {
            text.push_str("\ndetail:\n");
            text.push_str(detail);
            if !detail.ends_with('\n') {
                text.push('\n');
            }
        }
        write("failure.txt", text.as_bytes())?;
        Ok(dir)
    }
}

fn repro_input_name(input: &str) -> String {
    match Path::new(input).extension().and_then(|e| e.to_str()) {
        Some(ext @ ("air" | "ll" | "metallib")) => format!("input.{ext}"),
        _ => "input.bin".to_string(),
    }
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    if s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b':' | b'='))
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("chmod {}: {e}", path.display()))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn fail(why: &(impl std::fmt::Display + ?Sized)) -> ! {
    eprintln!("metal2vulkan: FALLBACK ({why})");
    process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_preserves_dotted_directories_and_hidden_file_names() {
        for (input, expected) in [
            ("source.dir/input", "source.dir/input.vk.spv"),
            ("source.dir/input.air", "source.dir/input.vk.spv"),
            ("source.dir/.shader", "source.dir/.shader.vk.spv"),
        ] {
            assert_eq!(default_output_path(input), expected);
        }
    }

    #[test]
    fn stage_outputs_insert_the_stage_before_the_extension() {
        for (output, expected) in [
            ("dir.d/out.spv", "dir.d/out.vert.spv"),
            ("dir.d/in.vk.spv", "dir.d/in.vk.vert.spv"),
            ("dir.d/out", "dir.d/out.vert"),
        ] {
            assert_eq!(stage_output_path(output, "vert"), expected);
        }
    }

    #[test]
    fn constants_pack_their_components_little_endian() {
        let (id, value) = parse_constant("3:uint2:1,0x10").expect("uint2");
        assert_eq!(id, "3");
        assert_eq!(value.ty, DataType::UInt2);
        assert_eq!(&value.bytes[..], &[1, 0, 0, 0, 16, 0, 0, 0]);
        let (id, value) = parse_constant("scale:half:1.5").expect("half");
        assert_eq!(id, "scale");
        assert_eq!(&value.bytes[..], &0x3e00u16.to_le_bytes());
        let (_, value) = parse_constant("0:Bool3:true,false,1").expect("bool3");
        assert_eq!(&value.bytes[..], &[1, 0, 1]);
        let (_, value) = parse_constant("0:char:-1").expect("char");
        assert_eq!(&value.bytes[..], &[0xff]);
        for bad in [
            "0:uchar:256",
            "0:char:128",
            "0:float2:1",
            "0:texture:1",
            "0:uint",
        ] {
            assert!(parse_constant(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn halves_round_to_nearest_even() {
        for (value, bits) in [
            (1.0f32, 0x3c00u16),
            (-2.0, 0xc000),
            (65504.0, 0x7bff),
            (65520.0, 0x7c00),
            (f32::INFINITY, 0x7c00),
            (5.960_464_5e-8, 0x0001),
            (2.980_232_2e-8, 0x0000),
            (1.0 + 1.0 / 2048.0, 0x3c00),
            (1.0 + 3.0 / 2048.0, 0x3c02),
            (0.0, 0x0000),
        ] {
            assert_eq!(f32_to_f16(value), bits, "{value}");
        }
        assert_eq!(f32_to_f16(f32::NAN) & 0x7c00, 0x7c00);
        assert_ne!(f32_to_f16(f32::NAN) & 0x3ff, 0);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn pixel_formats_take_metal_names_with_or_without_the_prefix() {
        assert_eq!(pixel_format("RGBA8Unorm"), Ok(PixelFormat::RGBA8Unorm));
        assert_eq!(
            pixel_format("MTLPixelFormatDepth32Float"),
            Ok(PixelFormat::Depth32Float)
        );
        assert!(pixel_format("RGBA9Unorm").is_err());
    }
}
