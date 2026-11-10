//! Device tests of render command encoders (its first part, the pass descriptor, load and store
//! actions, the deferred begin, and direct draws; its second, the depth/stencil state and the rest
//! of the render state) on the device the cfg selects, under the validation layer.
//!
//! - **The census.** Every committed public render case whose inputs these parts of the runtime
//!   bind draws through the runtime as the Metal oracle draws it (`metal::platform`), by the case
//!   runner the engine executor's runtime path shares (`candidate::runtime_case`'s
//!   `commit_render`, which states the draw): a library of its AIR and one of its generated
//!   companion's, each stage's function specialized with its constant values and the case's own
//!   linking its functions, a render pipeline state of the oracle's descriptor, its attachments,
//!   state, vertex buffers and resources, its one draw, and its output read back after the command
//!   buffer completed, a multisampled pass's sample by sample; a fragment case's explicit
//!   imageblock between the oracle's generated initializer and resolver, and a rate-mapped pass
//!   drawn region by region. A case the executor's route keeps for a later step
//!   (`runtime_run::render_prerequisites`: argument buffers, visible-function tables, out-of-bounds
//!   targets) is counted, not run. So is a case of out-of-bounds targets the route runs on the
//!   runtime: its output is judged by its containment, which the runtime path decides from what its
//!   draw executed and the census does not, so the runtime path's tests run it
//!   (`runtime_run::tests`, `runs_as_the_executor_compiles_it`; a fetch past a vertex buffer placed
//!   in a heap reads the heap's bytes, which no byte comparison admits). So is a case whose
//!   argument buffers hold pipeline states or command buffers: their templates' states are the
//!   runtime path's to make (`runtime_run::template_states`), and so are the command buffers'
//!   executions after the draw; its tests run it. Where the runtime's output on this device is not
//!   Metal's, the engine candidate runs the case afresh, as the kernel census does
//!   (`kernel_textures.rs`), and the case passes only when the runtime's bytes are the candidate's.
//!   A case with no current Metal row (one awaiting Metal qualification) goes the same way, and
//!   says so.
//! - **Clears** are the device's conversion of the clear value, compared with Metal's measured
//!   bytes: each color format, depth and stencil value an M4 was measured with, the NaNs of float
//!   formats among them (a Metal probe, the M3 Max alike). A texel the device converts otherwise
//!   than Metal is within a code (a float's adjacent value) of Metal's in each channel and
//!   announced (`device-color-conversion`, a `Depth16Unorm` depth's `unorm-depth-rounding`), and
//!   printed; a NaN of a half-float or small-float format is the device's, announced. A format the
//!   device cannot render is counted.
//! - **Load and store actions**, the encoder's late store action, the passes' order against blits
//!   in one command buffer, and Metal's refusals of passes and draws, typed. What a `DontCare`
//!   load or store leaves is printed: Metal leaves it undefined, and Apple's GPU keeps the old
//!   texels (measured).

use super::{
    at_plan, buffer_with, layer_reported_nothing, layer_reported_nothing_at_all, metal_output,
    read_buffer, texture_usage, validated_device, validated_device_with, Messages, RuntimeRun,
    ScratchCache, WAIT,
};
use crate::air::reflect::FragmentImageblock;
use crate::candidate::engine::graphics::{render_case, RenderCase};
use crate::candidate::engine::runtime_run::render_prerequisites;
use crate::candidate::engine::{linked_modules, LinkedModule, ENGINE_EXECUTOR_ABI};
use crate::candidate::runtime_case::{
    commit_render, render_pipeline_descriptor, render_pipeline_state, RenderInputs, TemplateState,
};
use crate::candidate::tests::engine_render;
use crate::case::{AuthoredCase, Stage};
use crate::check::interface::AirInterface;
use crate::gpu::TEST_DEVICE;
use crate::store::CorpusStore;
use metal2vulkan::bind::StatusReport;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::RenderPipelineDescriptor;
use metal2vulkan_runtime::{
    AppleOnly, Attachment, BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus,
    Degradation, Device, DrawProblem, Error, Invalid, LoadAction, Origin, Problem, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPassProblem, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeMap;
use std::path::Path;

/// A committed public render case for the runtime: the case, its render mapping, its AIR
/// bitcode and its generated companion's, the usage the Metal oracle gives each of its textures,
/// its AIR interface (whose argument-buffer members give its argument-buffer textures' usage),
/// and the output its current Metal observation records, or why it has none (a case awaiting
/// Metal qualification, such as the macOS 27 cases).
pub(super) struct PublicRender {
    pub(super) case: AuthoredCase,
    pub(super) render: RenderCase,
    bitcode: Vec<u8>,
    companion: Option<Vec<u8>>,
    linked: Vec<LinkedModule>,
    texture_usage: BTreeMap<u32, TextureUsage>,
    interface: AirInterface,
    /// A fragment case's imageblock master, the checked reflection's.
    imageblock: Option<FragmentImageblock>,
    pub(super) metal_output: Result<Vec<u8>, String>,
}

/// The template states of a lane case, which names no pipeline template (the census leaves those
/// to the runtime path's tests, the module's note).
static NO_TEMPLATES: BTreeMap<u64, TemplateState> = BTreeMap::new();

impl PublicRender {
    /// What the case runner draws of the case.
    pub(super) fn inputs(&self) -> RenderInputs<'_> {
        RenderInputs {
            case: &self.case,
            render: &self.render,
            bitcode: &self.bitcode,
            companion: self.companion.as_deref(),
            linked: &self.linked,
            texture_usage: &self.texture_usage,
            interface: &self.interface,
            imageblock: self.imageblock.as_ref(),
            templates: &NO_TEMPLATES,
        }
    }
}

/// Every committed render case whose AIR is a public fixture, in shard order: mapped, or why
/// not (a later runtime step's input among them, the executor's route's
/// `runtime_run::render_prerequisites`).
pub(super) fn render_cases() -> Vec<(String, Result<PublicRender, String>)> {
    render_cases_where(is_public_render)
}

/// Whether `case` is a committed render case whose AIR is a public fixture.
pub(super) fn is_public_render(case: &AuthoredCase) -> bool {
    matches!(case.stage, Stage::Vertex | Stage::Fragment) && case.public_source.is_some()
}

/// The committed cases `select` takes (public render cases, [`is_public_render`], or some of
/// them), in shard order, mapped as [`render_cases`] maps them; only those are mapped.
pub(super) fn render_cases_where(
    select: impl Fn(&AuthoredCase) -> bool,
) -> Vec<(String, Result<PublicRender, String>)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let mut cases = Vec::new();
    for shard in 0..64 {
        for case in store.read_case_shard(shard).unwrap() {
            if select(&case) {
                let name = case.name.clone();
                cases.push((name, map_render(&root, &store, case)));
            }
        }
    }
    assert!(
        !cases.is_empty(),
        "no committed public render case selected"
    );
    cases
}

/// `case` mapped as [`render_cases`] maps it: refused where the executor's route keeps it for a
/// later runtime step.
pub(super) fn map_render(
    root: &Path,
    store: &CorpusStore,
    case: AuthoredCase,
) -> Result<PublicRender, String> {
    map_render_where(root, store, case, true)
}

/// The committed public render case named `name`, mapped whatever runtime step its inputs wait
/// for: a lane test that encodes those inputs itself draws it.
pub(super) fn public_render(name: &str) -> PublicRender {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let case = (0..64)
        .flat_map(|shard| store.read_case_shard(shard).unwrap())
        .find(|case| case.name == name && case.public_source.is_some())
        .unwrap_or_else(|| panic!("no committed public render case {name}"));
    map_render_where(&root, &store, case, false).unwrap()
}

/// `case` mapped, refused where `routed` and the executor's route keeps it for a later runtime
/// step.
fn map_render_where(
    root: &Path,
    store: &CorpusStore,
    case: AuthoredCase,
    routed: bool,
) -> Result<PublicRender, String> {
    let name = case.name.clone();
    let metal_output = metal_output(store, &case);
    let checked = crate::check::check_case(root, case)
        .map_err(|errors| format!("{name}: the checker: {}", errors.join("; ")))?;
    let primary = &checked.primary;
    let render = render_case(
        &primary.case,
        &primary.interface,
        primary.entry_ll(),
        primary.companion.as_ref(),
        primary.reflection.fragment_imageblock.as_ref(),
    )
    .map_err(|error| format!("{name}: render_case: {error}"))?;
    let later = render_prerequisites(&primary.case, &render);
    if routed && !later.is_empty() {
        let names: Vec<_> = later
            .iter()
            .map(|prerequisite| prerequisite.name())
            .collect();
        return Err(format!(
            "{name}: {}, a later runtime step",
            names.join(", ")
        ));
    }
    let scratch = crate::ScratchDir::new("runtime-lane-render")?;
    let ll = scratch.path().join("case.ll");
    std::fs::write(&ll, &primary.source.air_ll).map_err(|error| format!("{name}: {error}"))?;
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll)
        .map_err(|error| format!("{name}: llvm-as: {error:?}"))?;
    let companion = render
        .companion
        .as_ref()
        .map(crate::companion::Companion::opaque_bitcode)
        .transpose()
        .map_err(|error| format!("{name}: the companion: {error}"))?;
    let texture_usage = texture_usage(&primary.case, &primary.interface)
        .map_err(|error| format!("{name}: {error}"))?;
    let linked = linked_modules(&primary.linked_functions)
        .map_err(|error| format!("{name}: the linked functions: {error}"))?;
    Ok(PublicRender {
        case: primary.case.clone(),
        render,
        bitcode,
        companion,
        linked,
        texture_usage,
        interface: primary.interface.clone(),
        imageblock: primary.reflection.fragment_imageblock.clone(),
        metal_output,
    })
}

/// Whether `why` is an honest refusal of a run: a gap the runtime names (`NotImplemented`), or
/// what the device lacks (`Unsupported`, the runtime's or the binder's: a bind state the device
/// cannot run, such as a shader-side stencil's write that the depth test decides).
pub(super) fn refusal(why: &str) -> bool {
    why.contains("not implemented in the runtime yet")
        || why.contains("unsupported:")
        || why.contains("the bind state was refused: Unsupported(")
}

/// The render pipeline descriptor of `case`, its functions made on `device`, as the case runner
/// makes it (`runtime_case::render_pipeline_descriptor`).
pub(super) fn descriptor_of(
    device: &Device,
    case: &PublicRender,
) -> Result<RenderPipelineDescriptor, String> {
    render_pipeline_descriptor(device, &case.inputs())
        .map_err(|error| format!("{}: {error}", case.case.name))
}

/// Run `case` once on `device` by the case runner the executor's runtime path shares
/// (`runtime_case::commit_render`): one command buffer with one render encoder, waited for.
pub(super) fn run_render(device: &Device, case: &PublicRender) -> Result<RuntimeRun, String> {
    let name = &case.case.name;
    let inputs = case.inputs();
    let committed = commit_render(device, &inputs).map_err(|error| format!("{name}: {error}"))?;
    let command_buffer = &committed.command_buffer;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("{name}: the command buffer did not complete"))?;
    if command_buffer.status() != CommandBufferStatus::Completed {
        return Err(format!(
            "{name}: the command buffer failed: {:?}",
            command_buffer.error()
        ));
    }
    let report = command_buffer
        .execution_report()
        .ok_or("a completed command buffer has an execution report")?;
    let output = committed
        .output(&inputs)
        .map_err(|error| format!("{name}: {error}"))?;
    Ok(RuntimeRun {
        output,
        report,
        executions: committed.executions,
    })
}

/// The engine candidate's fresh run of `case` on the lane's device: the device it opened, its
/// output and its execution report, or why it gave none.
fn fresh_candidate(case: &PublicRender) -> Result<(String, Vec<u8>, StatusReport), String> {
    let run = engine_render::run_mapped(&case.render, &case.bitcode)
        .map_err(|error| error.to_string())?;
    let device = run.environment["device"]
        .as_str()
        .unwrap_or("an unnamed device")
        .to_owned();
    Ok((device, run.output, run.status))
}

/// Every committed public render case this part of the runtime binds draws through it with a
/// clean execution report and Metal's output, or the output of the candidate run afresh on the
/// case (the module's note), or is refused honestly (`refusal`); the validation layer reports
/// nothing, leaks included. A case of out-of-bounds targets, or of pipeline states or command
/// buffers, is left to the runtime path's tests (the module's note), and listed.
#[test]
fn public_render_cases_match_metal() {
    let cases = render_cases();
    let messages = Messages::default();
    let (mut matched, mut differing, mut refused, mut later, mut failed) =
        (0usize, Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut contained, mut templated) = (Vec::new(), Vec::new());
    let device_name;
    {
        let cache = ScratchCache::new().unwrap();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        device_name = device.name().to_owned();
        for (name, case) in &cases {
            let case = match case {
                Ok(case) => case,
                Err(why) if why.contains("later runtime step") => {
                    later.push(why.clone());
                    continue;
                }
                Err(why) => {
                    failed.push(why.clone());
                    continue;
                }
            };
            if case.case.out_of_bounds.is_some() {
                contained.push(name);
                continue;
            }
            if !case.case.pipeline_states.is_empty() || !case.case.command_buffers.is_empty() {
                templated.push(name);
                continue;
            }
            match run_render(&device, case) {
                Ok(run) if run.report != StatusReport::default() => {
                    failed.push(format!("{name}: the execution report is {:?}", run.report))
                }
                Ok(run) if Ok(&run.output) == case.metal_output.as_ref() => matched += 1,
                Ok(_) => differing.push((name, case)),
                Err(why) if why.contains("later runtime step") => later.push(why),
                Err(why) if refusal(&why) => refused.push(why),
                Err(why) => failed.push(why),
            }
            let reported = messages.take();
            if !reported.is_empty() {
                failed.push(format!(
                    "{name}: the validation layer reported {reported:#?}"
                ));
            }
        }
    }
    let leaked = messages.take();
    let mut as_candidate = Vec::new();
    for (name, case) in differing {
        let (candidate_device, candidate, candidate_report) = match fresh_candidate(case) {
            Ok(run) => run,
            Err(why) => {
                failed.push(format!("{name}: the fresh candidate gave no output: {why}"));
                continue;
            }
        };
        let rerun = {
            let cache = ScratchCache::new().unwrap();
            let device =
                validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
            run_render(&device, case)
        };
        let reported = messages.take();
        if !reported.is_empty() {
            failed.push(format!(
                "{name}: the validation layer reported {reported:#?}"
            ));
        }
        match rerun {
            _ if candidate_device != device_name => failed.push(format!(
                "{name}: the candidate ran on {candidate_device}, the runtime on {device_name}"
            )),
            Ok(run) if run.report != StatusReport::default() => {
                failed.push(format!("{name}: the execution report is {:?}", run.report))
            }
            Ok(run) if run.output == candidate => as_candidate.push(match &case.metal_output {
                Ok(_) => format!("{name}: the candidate's execution report {candidate_report:?}"),
                Err(why) => format!(
                    "{name}: the candidate's execution report {candidate_report:?}; no Metal \
                     answer to compare: {why}"
                ),
            }),
            Ok(run) => failed.push(format!(
                "{name}: differs from Metal and from the fresh candidate \
                 ({ENGINE_EXECUTOR_ABI}):\n runtime   {:02x?}\n candidate {:02x?}\n \
                 metal     {:02x?}",
                run.output, candidate, case.metal_output
            )),
            Err(why) => failed.push(format!("{name}: the run after the candidate's: {why}")),
        }
    }
    eprintln!(
        "runtime render cases on {device_name}: {matched} match Metal, {} match the fresh \
         candidate ({ENGINE_EXECUTOR_ABI}), {} refused, {} of later steps, {} of \
         out-of-bounds targets and {} of pipeline states or command buffers left to the runtime \
         path's tests, {} failed, of {}",
        as_candidate.len(),
        refused.len(),
        later.len(),
        contained.len(),
        templated.len(),
        failed.len(),
        cases.len()
    );
    for name in &contained {
        eprintln!("judged by containment: {name}");
    }
    for name in &templated {
        eprintln!("of pipeline states or command buffers: {name}");
    }
    for line in &as_candidate {
        eprintln!("as the candidate: {line}");
    }
    for why in &refused {
        eprintln!("refused: {why}");
    }
    for why in &later {
        eprintln!("later: {why}");
    }
    assert!(
        failed.is_empty() && leaked.is_empty(),
        "failed:\n{}\nat device destruction: {leaked:#?}",
        failed.join("\n")
    );
    layer_reported_nothing(
        &messages,
        "every case's draw was valid and left nothing behind",
    );
    assert!(matched > 0, "no case ran");
}

/// Destroy `device`, then check the validation layer reported nothing, leaks included.
fn quiet(device: Device, messages: &Messages) {
    drop(device);
    layer_reported_nothing_at_all(messages);
}

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// Commit `command_buffer` and wait for it to complete without an error.
pub(super) fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A `width × height` texture of `pf` with `RenderTarget` usage (and `ShaderRead`), of
/// `options`.
fn attachment(
    device: &Device,
    pf: PixelFormat,
    [width, height]: [u64; 2],
    options: ResourceOptions,
) -> Result<Texture, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = options;
    device.new_texture(&descriptor)
}

/// The bytes of `texture`'s first level and slice, `pixel` bytes a texel, copied by a blit into a
/// `Shared` buffer.
pub(super) fn blit_read(device: &Device, texture: &Texture, pixel: u64) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let buffer = device
        .new_buffer(width * height * pixel, ResourceOptions::shared())
        .unwrap();
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &buffer,
        0,
        width * pixel,
        width * height * pixel,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&buffer, 0, buffer.length()).unwrap()
}

/// One color clear Metal was measured with: the format, the clear color, and the texel's bytes.
type ColorClear = (PixelFormat, [f64; 4], &'static [u8]);

/// Color clears measured on an M4 (our Metal probes; the NaNs on an M4 and the M3 Max alike), each
/// format's bytes as Metal stored them.
const COLOR_CLEARS: &[ColorClear] = &[
    (
        PixelFormat::RGBA8Unorm,
        [0.5, 0.25, 1.5, -0.5],
        &[0x80, 0x40, 0xff, 0x00],
    ),
    (
        PixelFormat::RGBA8Unorm,
        [
            0.00196078431372549,
            0.0058823529411764705,
            0.00980392156862745,
            0.501960784,
        ],
        &[0x01, 0x02, 0x03, 0x80],
    ),
    (
        PixelFormat::RGBA8Unorm,
        [0.0019, 0.0021, 0.998, 0.996],
        &[0x00, 0x01, 0xfe, 0xfe],
    ),
    (
        PixelFormat::RGBA8Unorm,
        [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.75],
        &[0x00, 0xff, 0x00, 0xbf],
    ),
    (
        PixelFormat::BGRA8Unorm,
        [0.25, 0.5, 0.75, 1.0],
        &[0xbf, 0x80, 0x40, 0xff],
    ),
    (
        PixelFormat::RGBA8UnormSRGB,
        [0.5, 0.214, 0.0031308, 1.0],
        &[0xbc, 0x7f, 0x0a, 0xff],
    ),
    (
        PixelFormat::RGBA8UnormSRGB,
        [0.001, 0.01, 0.1, 0.5],
        &[0x03, 0x19, 0x59, 0x80],
    ),
    (
        PixelFormat::RGBA8UnormSRGB,
        [0.2158605, 0.2122308, 0.214, 0.25],
        &[0x80, 0x7f, 0x7f, 0x40],
    ),
    (
        PixelFormat::BGRA8UnormSRGB,
        [0.5, 0.214, 0.0031308, 0.5],
        &[0x0a, 0x7f, 0xbc, 0x80],
    ),
    (
        PixelFormat::RGBA8Snorm,
        [-1.5, -0.5, 0.5, 0.011811023622047244],
        &[0x81, 0xc0, 0x40, 0x01],
    ),
    (
        PixelFormat::RGBA8Snorm,
        [-1.0, -0.0039, 0.0039, 0.502],
        &[0x81, 0x00, 0x00, 0x40],
    ),
    (
        PixelFormat::RGBA8Snorm,
        [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0],
        &[0x00, 0x7f, 0x81, 0x00],
    ),
    (PixelFormat::R8Uint, [3.7, 0.0, 0.0, 0.0], &[0x03]),
    (PixelFormat::R8Uint, [300.0, 0.0, 0.0, 0.0], &[0xff]),
    (PixelFormat::R8Uint, [-1.0, 0.0, 0.0, 0.0], &[0x00]),
    (PixelFormat::R8Uint, [f64::NAN, 0.0, 0.0, 0.0], &[0x00]),
    (PixelFormat::R8Sint, [-3.7, 0.0, 0.0, 0.0], &[0xfd]),
    (PixelFormat::R8Sint, [200.0, 0.0, 0.0, 0.0], &[0x7f]),
    (PixelFormat::R8Sint, [-200.0, 0.0, 0.0, 0.0], &[0x80]),
    (
        PixelFormat::R16Uint,
        [65535.7, 0.0, 0.0, 0.0],
        &[0xff, 0xff],
    ),
    (PixelFormat::R16Sint, [-2.5, 0.0, 0.0, 0.0], &[0xfe, 0xff]),
    (
        PixelFormat::R32Uint,
        [4294967296.0, 0.0, 0.0, 0.0],
        &[0xff, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::R32Uint,
        [16777217.0, 0.0, 0.0, 0.0],
        &[0x01, 0x00, 0x00, 0x01],
    ),
    (
        PixelFormat::R32Uint,
        [4294967040.5, 0.0, 0.0, 0.0],
        &[0x00, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::R32Sint,
        [-16777217.0, 0.0, 0.0, 0.0],
        &[0xff, 0xff, 0xff, 0xfe],
    ),
    (
        PixelFormat::R32Sint,
        [-3000000000.0, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0x00, 0x80],
    ),
    (PixelFormat::R16Float, [0.1, 0.0, 0.0, 0.0], &[0x66, 0x2e]),
    (
        PixelFormat::R16Float,
        [70000.0, 0.0, 0.0, 0.0],
        &[0x00, 0x7c],
    ),
    (PixelFormat::R16Float, [-0.0, 0.0, 0.0, 0.0], &[0x00, 0x80]),
    (
        PixelFormat::R16Float,
        [1.001220703125, 0.0, 0.0, 0.0],
        &[0x01, 0x3c],
    ),
    (
        PixelFormat::R16Float,
        [1.0004882812509095, 0.0, 0.0, 0.0],
        &[0x00, 0x3c],
    ),
    (
        PixelFormat::R16Float,
        [1.00146484375, 0.0, 0.0, 0.0],
        &[0x02, 0x3c],
    ),
    (
        PixelFormat::R32Float,
        [0.1, 0.0, 0.0, 0.0],
        &[0xcd, 0xcc, 0xcc, 0x3d],
    ),
    (
        PixelFormat::R32Float,
        [1e-40, 0.0, 0.0, 0.0],
        &[0xc2, 0x16, 0x01, 0x00],
    ),
    (
        PixelFormat::R32Float,
        [1e39, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0x80, 0x7f],
    ),
    (
        PixelFormat::R32Float,
        [f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0xc0, 0x7f],
    ),
    (
        PixelFormat::R32Float,
        [-f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0xc0, 0xff],
    ),
    (
        PixelFormat::R16Float,
        [f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x7e],
    ),
    (
        PixelFormat::R16Float,
        [-f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x7e],
    ),
    (
        PixelFormat::RG16Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0x00, 0x7e, 0x00, 0x3c],
    ),
    (
        PixelFormat::RG16Float,
        [1.0, f64::NAN, 2.0, 0.25],
        &[0x00, 0x3c, 0x00, 0x7e],
    ),
    (
        PixelFormat::RGBA16Float,
        [-f64::NAN, -f64::NAN, -f64::NAN, -f64::NAN],
        &[0x00, 0x7e, 0x00, 0x7e, 0x00, 0x7e, 0x00, 0x7e],
    ),
    (
        PixelFormat::RGBA16Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0x00, 0x7e, 0x00, 0x3c, 0x00, 0x38, 0x00, 0x7e],
    ),
    (
        PixelFormat::RGBA16Float,
        [1.0, f64::NAN, 2.0, 0.25],
        &[0x00, 0x3c, 0x00, 0x7e, 0x00, 0x40, 0x00, 0x34],
    ),
    (
        PixelFormat::RGBA16Float,
        [0.1, -2.5, 65504.0, 1e-05],
        &[0x66, 0x2e, 0x00, 0xc1, 0xff, 0x7b, 0xa8, 0x00],
    ),
    (
        PixelFormat::RG16Snorm,
        [-1.5, 0.5, 0.0, 0.0],
        &[0x01, 0x80, 0x00, 0x40],
    ),
    (
        PixelFormat::RG16Snorm,
        [-1e-05, 0.99999, 0.0, 0.0],
        &[0x00, 0x00, 0xff, 0x7f],
    ),
    (
        PixelFormat::RGBA16Unorm,
        [0.5, 7.629510948348211e-06, 2.2888532845044633e-05, 1.2],
        &[0x00, 0x80, 0x00, 0x00, 0x01, 0x00, 0xff, 0xff],
    ),
    (
        PixelFormat::RGB10A2Unorm,
        [0.5, 0.25, 1.0, 0.5],
        &[0x00, 0x02, 0xf4, 0xbf],
    ),
    (
        PixelFormat::RGB10A2Unorm,
        [0.0005, 0.9995, 0.33, 0.17],
        &[0x01, 0xf8, 0x2f, 0x55],
    ),
    (
        PixelFormat::BGR10A2Unorm,
        [0.5, 0.25, 0.75, 0.4],
        &[0xff, 0x02, 0x04, 0x60],
    ),
    (
        PixelFormat::RGBA32Float,
        [0.1, -1e39, 1e-40, 3.0],
        &[
            0xcd, 0xcc, 0xcc, 0x3d, 0x00, 0x00, 0x80, 0xff, 0xc2, 0x16, 0x01, 0x00, 0x00, 0x00,
            0x40, 0x40,
        ],
    ),
    (
        PixelFormat::RGBA32Uint,
        [1.5, 4294967296.0, -1.0, 7.0],
        &[1, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 7, 0, 0, 0],
    ),
    (
        PixelFormat::RG11B10Float,
        [0.1, -1.0, 70000.0, 0.0],
        &[0xe6, 0x02, 0xc0, 0xf7],
    ),
    (
        PixelFormat::RG11B10Float,
        [65000.0, 0.5, 1e-05, 0.0],
        &[0xbe, 0x07, 0x5c, 0x01],
    ),
    (
        PixelFormat::RG11B10Float,
        [f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        &[0xe0, 0x07, 0x3f, 0xfc],
    ),
    (
        PixelFormat::RG11B10Float,
        [-f64::NAN, -f64::NAN, -f64::NAN, -f64::NAN],
        &[0xe0, 0x07, 0x3f, 0xfc],
    ),
    (
        PixelFormat::RG11B10Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0xe0, 0x07, 0x1e, 0x70],
    ),
    (
        PixelFormat::RG11B10Float,
        [1.0, f64::NAN, 2.0, 0.25],
        &[0xc0, 0x03, 0x3f, 0x80],
    ),
    (
        PixelFormat::RG11B10Float,
        [1.0078125, 1.015625, 1.0078125, 0.0],
        &[0xc0, 0x0b, 0x1e, 0x78],
    ),
    (
        PixelFormat::RGB9E5Float,
        [0.1, 0.5, 2.0, 0.0],
        &[0x0c, 0x80, 0x00, 0x8c],
    ),
    (
        PixelFormat::RGB9E5Float,
        [-1.0, 70000.0, 0.001, 0.0],
        &[0x00, 0xfe, 0x03, 0xf8],
    ),
    (
        PixelFormat::RGB9E5Float,
        [65408.0, 1e-08, 3.0, 0.0],
        &[0xff, 0x01, 0x00, 0xf8],
    ),
    (
        PixelFormat::RGB9E5Float,
        [f64::NAN, f64::INFINITY, 1.0, 0.0],
        &[0x00, 0xfe, 0x03, 0xf8],
    ),
    (
        PixelFormat::RGB9E5Float,
        [-f64::NAN, -f64::NAN, -f64::NAN, -f64::NAN],
        &[0x00, 0x00, 0x00, 0x00],
    ),
    (
        PixelFormat::RGB9E5Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0x00, 0x00, 0x02, 0x82],
    ),
    (
        PixelFormat::RGB9E5Float,
        [1.0, f64::NAN, 2.0, 0.25],
        &[0x80, 0x00, 0x00, 0x8c],
    ),
    (
        PixelFormat::B5G6R5Unorm,
        [0.5, 0.25, 0.75, 1.0],
        &[0x17, 0x82],
    ),
    (
        PixelFormat::A1BGR5Unorm,
        [0.5, 0.25, 0.75, 0.5],
        &[0x2e, 0x82],
    ),
    (
        PixelFormat::ABGR4Unorm,
        [0.5, 0.25, 0.75, 0.1],
        &[0xb1, 0x84],
    ),
    (
        PixelFormat::BGR5A1Unorm,
        [0.5, 0.25, 0.75, 0.6],
        &[0x17, 0xc1],
    ),
];

/// Clear `texture` (attached as the pass's only color attachment) to `color`, and store.
fn clear_color(device: &Device, texture: &Texture, color: [f64; 4]) -> Result<(), Error> {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    let [red, green, blue, alpha] = color;
    attachment.clear_color = ClearColor::new(red, green, blue, alpha);
    let command_buffer = device.new_command_queue()?.command_buffer()?;
    command_buffer
        .render_command_encoder(&pass)?
        .end_encoding()?;
    run(&command_buffer);
    Ok(())
}

/// The bytes of `texture` (one texel of `pixel` bytes) after a plain fill of the whole of it to
/// `color`: the device's clear command of the clear value a `Clear` load of it takes
/// (`blit_fill.rs`).
fn clear_command(device: &Device, texture: &Texture, color: [f64; 4], pixel: u64) -> Vec<u8> {
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let [red, green, blue, alpha] = color;
    let (width, height) = (texture.width(), texture.height());
    blit.fill_texture_color(
        texture,
        0,
        0,
        Region::new_2d(0, 0, width, height),
        ClearColor::new(red, green, blue, alpha),
        None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    blit_read(device, texture, pixel)
}

/// Whether `channel`'s bits, `exponent` exponent bits above `mantissa` mantissa bits, are a NaN.
pub(super) fn small_nan(channel: u32, exponent: u32, mantissa: u32) -> bool {
    let all = (1 << exponent) - 1;
    (channel >> mantissa) & all == all && channel & ((1 << mantissa) - 1) != 0
}

/// Each channel of a `pf` texel (a half-float format's halves, `RG11B10Float`'s three fields) as
/// its bits, its exponent's width and its mantissa's; empty for another format.
fn channels(pf: PixelFormat, texel: &[u8]) -> Vec<(u32, u32, u32)> {
    match pf {
        PixelFormat::R16Float | PixelFormat::RG16Float | PixelFormat::RGBA16Float => texel
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| (u32::from(u16::from_le_bytes([h[0], h[1]])), 5, 10))
            .collect(),
        PixelFormat::RG11B10Float => {
            let word = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
            vec![
                (word & 0x7ff, 5, 6),
                ((word >> 11) & 0x7ff, 5, 6),
                (word >> 22, 5, 5),
            ]
        }
        _ => Vec::new(),
    }
}

/// Whether texel `got` of `pf` differs from `want` only by one NaN for another in a half-float or
/// small-float channel: the device's conversion of a NaN clear (`device-nan-encoding`).
fn nan_for_nan(pf: PixelFormat, got: &[u8], want: &[u8]) -> bool {
    let (got, want) = (channels(pf, got), channels(pf, want));
    !want.is_empty()
        && got.len() == want.len()
        && got
            .iter()
            .zip(&want)
            .all(|(&(g, e, m), &(w, _, _))| g == w || (small_nan(g, e, m) && small_nan(w, e, m)))
}

/// The fields of a `pf` texel (each channel's bits, signed for a snorm format's) whose codes one
/// device's conversion may stand one apart from Metal's, or `None` for a format with no such
/// reading here (an integer or 32-bit float format's, whose clear converts nothing).
fn fields(pf: PixelFormat, texel: &[u8]) -> Option<Vec<i64>> {
    use PixelFormat as P;
    let word = |bytes: &[u8]| {
        bytes
            .iter()
            .rev()
            .fold(0u64, |w, &b| (w << 8) | u64::from(b))
    };
    let packed = |widths: &[u32]| {
        let w = word(texel);
        let mut at = 0;
        widths
            .iter()
            .map(|&bits| {
                let field = (w >> at) & ((1 << bits) - 1);
                at += bits;
                field as i64
            })
            .collect()
    };
    Some(match pf {
        P::RGBA8Unorm | P::BGRA8Unorm | P::RGBA8UnormSRGB | P::BGRA8UnormSRGB => {
            texel.iter().map(|&b| i64::from(b)).collect()
        }
        P::RGBA8Snorm => texel.iter().map(|&b| i64::from(b as i8)).collect(),
        P::RGBA16Unorm | P::R16Float | P::RG16Float | P::RGBA16Float => texel
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| i64::from(u16::from_le_bytes(*h)))
            .collect(),
        P::RG16Snorm => texel
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| i64::from(i16::from_le_bytes(*h)))
            .collect(),
        P::RGB10A2Unorm | P::BGR10A2Unorm => packed(&[10, 10, 10, 2]),
        P::RG11B10Float => packed(&[11, 11, 10]),
        // The mantissas under the shared exponent: one apart at the same exponent.
        P::RGB9E5Float => packed(&[9, 9, 9, 5]),
        P::B5G6R5Unorm => packed(&[5, 6, 5]),
        P::A1BGR5Unorm => packed(&[1, 5, 5, 5]),
        P::BGR5A1Unorm => packed(&[5, 5, 5, 1]),
        P::ABGR4Unorm => packed(&[4, 4, 4, 4]),
        P::Depth16Unorm => vec![word(texel) as i64],
        _ => return None,
    })
}

/// Whether texel `got` of `pf` stands at most one code (a float's adjacent value) from `want` in
/// each field ([`fields`]): the device's conversion of a clear whose rounding is in question.
fn a_code_apart(pf: PixelFormat, got: &[u8], want: &[u8]) -> bool {
    match (fields(pf, got), fields(pf, want)) {
        (Some(got), Some(want)) => {
            got.len() == want.len() && got.iter().zip(&want).all(|(g, w)| g.abs_diff(*w) <= 1)
        }
        _ => false,
    }
}

/// Each color clear Metal was measured with is the device's conversion of the clear value, depth
/// and stencil clears included: Metal's bytes, but a texel of a clear whose rounding is in
/// question, within a code (a float's adjacent value) of Metal's in each field and announced
/// (`device-color-conversion`; a `Depth16Unorm` depth's `unorm-depth-rounding`), and a NaN of a
/// half-float or small-float channel, which the device's clear converts as it likes, announced
/// (`device-nan-encoding`); each printed. A format the device cannot render to is counted, not
/// run. On a driver whose clears are Metal's (`DriverFact::ColorClearIsMetals`), every color
/// clear is Metal's bytes and none announces `device-color-conversion`, and so is the device's
/// clear command of the load operation's clear value over the whole of a texture: a plain fill
/// of it (`BlitCommandEncoder::fill_texture_color` with no pixel format, whose fill announces its
/// own notes, taken apart).
#[test]
fn clears_are_the_device_s_conversion_of_the_clear_value() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    let metals = device
        .properties()
        .driver_facts
        .contains(&metal2vulkan::device::DriverFact::ColorClearIsMetals);
    let (mut checked, mut unsupported, mut wrong) = (0usize, Vec::new(), Vec::new());
    let mut commanded = 0usize;
    let (mut announced, mut converted, mut rounded) = (false, false, 0usize);
    for &(pf, color, expected) in COLOR_CLEARS {
        let texture = match attachment(&device, pf, [1, 1], ResourceOptions::private()) {
            Ok(texture) => texture,
            Err(Error::Unsupported(why)) => {
                unsupported.push(format!("{pf:?}: {why:?}"));
                continue;
            }
            Err(error) => panic!("{pf:?}: {error}"),
        };
        // A texture the device makes but cannot render to is refused at the pass.
        match clear_color(&device, &texture, color) {
            Ok(()) => {}
            Err(Error::Unsupported(why)) => {
                unsupported.push(format!("{pf:?}: {why:?}"));
                continue;
            }
            Err(error) => panic!("{pf:?}: {error}"),
        }
        let bytes = blit_read(&device, &texture, expected.len() as u64);
        let counts = super::degrade_notices::counts(&messages.take_notices());
        announced |= counts.contains_key(&Degradation::DeviceNanEncoding);
        converted |= counts.contains_key(&Degradation::DeviceColorConversion);
        if bytes != expected && announced && nan_for_nan(pf, &bytes, expected) {
            eprintln!(
                "clear {pf:?} {color:?} on {}: {bytes:02x?}, Metal {expected:02x?} \
                 (device-nan-encoding)",
                device.name()
            );
        } else if bytes != expected && converted && a_code_apart(pf, &bytes, expected) {
            eprintln!(
                "clear {pf:?} {color:?} on {}: {bytes:02x?}, Metal {expected:02x?} \
                 (device-color-conversion)",
                device.name()
            );
            rounded += 1;
        } else if bytes != expected {
            wrong.push(format!(
                "{pf:?} {color:?}: {bytes:02x?}, Metal {expected:02x?}"
            ));
        }
        checked += 1;
        if metals {
            let filled = attachment(&device, pf, [1, 1], ResourceOptions::private()).unwrap();
            let bytes = clear_command(&device, &filled, color, expected.len() as u64);
            messages.take_notices();
            if bytes != expected {
                wrong.push(format!(
                    "{pf:?} {color:?}, the clear command: {bytes:02x?}, Metal {expected:02x?}"
                ));
            }
            commanded += 1;
        }
    }
    // Depth: through `float`, `Depth16Unorm` truncating, and outside [0, 1] the nearer end (Metal
    // probe case `depth_clear_range_load` on an M4, which Metal's validation stops); stencil: the
    // low 8 bits.
    let depth_clears: &[(PixelFormat, f64, &[u8])] = &[
        (PixelFormat::Depth32Float, 2.0, &[0x00, 0x00, 0x80, 0x3f]),
        (PixelFormat::Depth32Float, -1.0, &[0x00, 0x00, 0x00, 0x00]),
        (PixelFormat::Depth32Float, 0.3, &[0x9a, 0x99, 0x99, 0x3e]),
        (PixelFormat::Depth32Float, 0.1, &[0xcd, 0xcc, 0xcc, 0x3d]),
        (PixelFormat::Depth32Float, 1.0, &[0x00, 0x00, 0x80, 0x3f]),
        (PixelFormat::Depth16Unorm, 0.5, &[0xff, 0x7f]),
        (PixelFormat::Depth16Unorm, 0.3, &[0xcc, 0x4c]),
        (PixelFormat::Depth16Unorm, 0.1, &[0x99, 0x19]),
        (
            PixelFormat::Depth16Unorm,
            7.629510948348211e-06,
            &[0x00, 0x00],
        ),
        (
            PixelFormat::Depth16Unorm,
            2.2888532845044633e-05,
            &[0x01, 0x00],
        ),
        (PixelFormat::Depth16Unorm, 1.0, &[0xff, 0xff]),
    ];
    let mut converted_depth = false;
    for &(pf, depth, expected) in depth_clears {
        let texture = attachment(&device, pf, [1, 1], ResourceOptions::private()).unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(texture.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.depth_attachment.clear_depth = depth;
        let command_buffer = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        command_buffer
            .render_command_encoder(&pass)
            .unwrap()
            .end_encoding()
            .unwrap();
        run(&command_buffer);
        let bytes = blit_read(&device, &texture, expected.len() as u64);
        let depth_rounded = super::degrade_notices::counts(&messages.take_notices())
            .contains_key(&Degradation::UnormDepthRounding);
        converted_depth |= depth_rounded;
        if bytes != expected && converted_depth && a_code_apart(pf, &bytes, expected) {
            eprintln!(
                "clear {pf:?} {depth} on {}: {bytes:02x?}, Metal {expected:02x?} \
                 (unorm-depth-rounding)",
                device.name()
            );
            rounded += 1;
        } else if bytes != expected {
            wrong.push(format!(
                "{pf:?} {depth}: {bytes:02x?}, Metal {expected:02x?}"
            ));
        }
        checked += 1;
    }
    for (stencil, expected) in [(90u32, 0x5au8), (511, 0xff), (256, 0x00), (u32::MAX, 0xff)] {
        let texture = attachment(
            &device,
            PixelFormat::Stencil8,
            [1, 1],
            ResourceOptions::private(),
        )
        .unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.stencil_attachment.texture = Some(texture.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::Store;
        pass.stencil_attachment.clear_stencil = stencil;
        let command_buffer = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        command_buffer
            .render_command_encoder(&pass)
            .unwrap()
            .end_encoding()
            .unwrap();
        run(&command_buffer);
        let bytes = blit_read(&device, &texture, 1);
        if bytes != [expected] {
            wrong.push(format!(
                "Stencil8 {stencil}: {bytes:02x?}, Metal {expected:02x}"
            ));
        }
        checked += 1;
    }
    eprintln!(
        "clears on {}: {checked} checked, {rounded} a code from Metal's, {commanded} by the clear \
         command, {} formats the device cannot render to",
        device.name(),
        unsupported.len()
    );
    assert!(
        !(metals && converted),
        "{}: a color clear announced device-color-conversion where the load operation clears as \
         Metal's",
        device.name()
    );
    for why in &unsupported {
        eprintln!("unsupported: {why}");
    }
    assert!(
        wrong.is_empty(),
        "clears Metal stores otherwise:\n{}",
        wrong.join("\n")
    );
    quiet(device, &messages);
}

/// A 4×4 `RGBA8Unorm` pattern.
fn pattern(seed: u8) -> Vec<u8> {
    (0..64u8)
        .map(|i| i.wrapping_mul(7).wrapping_add(seed))
        .collect()
}

/// A 4×4 `RGBA8Unorm` `Shared` attachment holding `bytes`.
fn holding(device: &Device, bytes: &[u8]) -> Texture {
    let texture = attachment(
        device,
        PixelFormat::RGBA8Unorm,
        [4, 4],
        ResourceOptions::shared(),
    )
    .unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, bytes, 16, 64)
        .unwrap();
    texture
}

/// A 4×4 `RGBA32Float` `Shared` attachment (the format of [`CONSTANT_COLOR`]'s pipeline) holding
/// `bytes` (256 of them).
fn holding_float(device: &Device, bytes: &[u8]) -> Texture {
    let texture = attachment(
        device,
        PixelFormat::RGBA32Float,
        [4, 4],
        ResourceOptions::shared(),
    )
    .unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, bytes, 64, 256)
        .unwrap();
    texture
}

/// The 4×4 `RGBA32Float` texels of `texture`.
fn float_texels(texture: &Texture) -> Vec<u8> {
    let mut bytes = vec![0u8; 256];
    texture
        .get_bytes(&mut bytes, 64, 256, Region::new_2d(0, 0, 4, 4), 0, 0)
        .unwrap();
    bytes
}

/// The 4×4 `RGBA8Unorm` texels of `texture`.
fn texels(texture: &Texture) -> Vec<u8> {
    let mut bytes = vec![0u8; 64];
    texture
        .get_bytes(&mut bytes, 16, 64, Region::new_2d(0, 0, 4, 4), 0, 0)
        .unwrap();
    bytes
}

/// A pass whose only color attachment is `texture`, with `load` and `store`, clearing to a
/// color whose bytes are `[0x40, 0x80, 0xc0, 0xff]`.
fn one_target(texture: &Texture, load: LoadAction, store: StoreAction) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = load;
    attachment.store_action = store;
    attachment.clear_color = ClearColor::new(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
    pass
}

/// The clear color of [`one_target`], every texel.
fn cleared() -> Vec<u8> {
    [0x40u8, 0x80, 0xc0, 0xff].repeat(16)
}

/// Run one pass of `descriptor` without draws, `encode` applied to its encoder first.
fn empty_pass(
    device: &Device,
    descriptor: &RenderPassDescriptor,
    encode: impl FnOnce(&mut RenderCommandEncoder),
) {
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    run(&command_buffer);
}

/// `Load` keeps the texels and `Clear` writes the clear color; a store action the descriptor left
/// `Unknown` is the one the encoder sets; passes order against blits in one command buffer (a
/// pass loads what a blit before it wrote, a blit after it reads what it stored). What `DontCare`
/// leaves is printed.
#[test]
fn load_and_store_actions_keep_and_clear_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    let texture = holding(&device, &pattern(1));
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::Load, StoreAction::Store),
        |_| {},
    );
    assert_eq!(texels(&texture), pattern(1), "Load, Store");
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::Clear, StoreAction::Store),
        |_| {},
    );
    assert_eq!(texels(&texture), cleared(), "Clear, Store");

    // The late store action.
    let texture = holding(&device, &pattern(2));
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::Clear, StoreAction::Unknown),
        |encoder| {
            encoder
                .set_color_store_action(StoreAction::Store, 0)
                .unwrap()
        },
    );
    assert_eq!(texels(&texture), cleared(), "Clear, a late Store");

    // In one command buffer: a blit writes the attachment, a pass loads and stores it, a blit
    // reads it; then a pass clears it and a blit reads it again.
    let texture = attachment(
        &device,
        PixelFormat::RGBA8Unorm,
        [4, 4],
        ResourceOptions::private(),
    )
    .unwrap();
    let source = buffer_with(&device, &pattern(3)).unwrap();
    let loaded = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let after_clear = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let blit_copy = |command_buffer: &CommandBuffer, into: Option<&Buffer>| {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let (origin, size) = (Origin::new(0, 0, 0), Size::new(4, 4, 1));
        match into {
            None => blit.copy_from_buffer_to_texture(
                &source,
                0,
                16,
                64,
                size,
                &texture,
                0,
                0,
                origin,
                BlitOption::None,
            ),
            Some(buffer) => blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                origin,
                size,
                buffer,
                0,
                16,
                64,
                BlitOption::None,
            ),
        }
        .unwrap();
        blit.end_encoding().unwrap();
    };
    blit_copy(&command_buffer, None);
    command_buffer
        .render_command_encoder(&one_target(&texture, LoadAction::Load, StoreAction::Store))
        .unwrap()
        .end_encoding()
        .unwrap();
    blit_copy(&command_buffer, Some(&loaded));
    command_buffer
        .render_command_encoder(&one_target(&texture, LoadAction::Clear, StoreAction::Store))
        .unwrap()
        .end_encoding()
        .unwrap();
    blit_copy(&command_buffer, Some(&after_clear));
    run(&command_buffer);
    assert_eq!(
        read_buffer(&loaded, 0, 64).unwrap(),
        pattern(3),
        "a blit, then Load"
    );
    assert_eq!(
        read_buffer(&after_clear, 0, 64).unwrap(),
        cleared(),
        "then Clear"
    );

    // DontCare: undefined in Metal; printed.
    let texture = holding(&device, &pattern(4));
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::Load, StoreAction::DontCare),
        |_| {},
    );
    let after_store = texels(&texture);
    let texture = holding(&device, &pattern(5));
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::DontCare, StoreAction::Store),
        |_| {},
    );
    let after_load = texels(&texture);
    let verdict = |bytes: &[u8], old: &[u8]| {
        if bytes == old {
            "keeps the old texels".to_owned()
        } else {
            format!("leaves {:02x?}", &bytes[..16])
        }
    };
    eprintln!(
        "DontCare on {}: a DontCare store {}; a DontCare load with no draws {}",
        device.name(),
        verdict(&after_store, &pattern(4)),
        verdict(&after_load, &pattern(5))
    );
    quiet(device, &messages);
}

/// The public case whose pipeline the refusal test draws with: a constant fragment color through
/// its generated passthrough vertex, one `RGBA8Unorm` target.
const CONSTANT_COLOR: &str = "authored-fragment-constant-color";

/// Metal's refusals of passes and draws, typed: a NaN clear depth; a store action never set; a
/// pipeline state of other formats than the pass's; a draw without a pipeline (one of another
/// topology class draws as plain Metal's: `topology_classes`). A refused draw records nothing, and
/// a draw of no vertices or no instances records nothing; a misaligned index buffer offset reads
/// from the offset rounded down. A color store action set late over a known one, or for no
/// attachment, and a depth one for no attachment change nothing, as plain Metal's (Metal probes on
/// an M4, which Metal's validation stops). The render encoder's Apple-only setters answer by step
/// 12's table. (A draw whose indices run past its index buffer draws as Metal's runtime does:
/// `index_fetches`.)
#[test]
fn passes_and_draws_are_refused_as_metal_refuses_them() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let case = (0..64)
        .flat_map(|shard| store.read_case_shard(shard).unwrap())
        .find(|case| case.name == CONSTANT_COLOR)
        .expect("the constant color case");
    let case = map_render(&root, &store, case).unwrap();
    let pipeline = render_pipeline_state(&device, &case.inputs()).unwrap();
    let queue = device.new_command_queue().unwrap();

    // A NaN clear depth of a format whose store of one is not measured, where the device makes
    // it (a value outside [0, 1] clears to the nearer end:
    // `clears_are_the_device_s_conversion_of_the_clear_value`; a NaN of a measured format clears
    // as Metal's: `depth_denormals::a_nan_clear_depth_stores_as_metal_s`).
    match attachment(
        &device,
        PixelFormat::Depth24UnormStencil8,
        [4, 4],
        ResourceOptions::private(),
    ) {
        Ok(depth) => {
            let mut pass = RenderPassDescriptor::new();
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = f64::NAN;
            let command_buffer = queue.command_buffer().unwrap();
            assert_eq!(
                command_buffer.render_command_encoder(&pass).unwrap_err(),
                invalid(
                    "render_command_encoder",
                    Problem::RenderPass(RenderPassProblem::ClearDepth)
                )
            );
        }
        Err(error) => println!("NAN-CLEAR-DEPTH Depth24UnormStencil8 not made: {error}"),
    }

    // Late store actions, in a pass of the pipeline's formats: over the descriptor's `Store` (the
    // texels stay stored, `nothing drew` below), of a color attachment without a texture, and of no
    // depth attachment, each changes nothing (Metal probe cases `late_store_known`,
    // `late_store_no_texture` and `late_store_depth_no_texture` on an M4).
    let target = holding_float(&device, &pattern(6).repeat(4));
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&one_target(&target, LoadAction::Load, StoreAction::Store))
        .unwrap();
    assert_eq!(
        encoder.set_color_store_action(StoreAction::DontCare, 0),
        Ok(())
    );
    assert_eq!(
        encoder.set_color_store_action(StoreAction::Store, 1),
        Ok(())
    );
    assert_eq!(encoder.set_depth_store_action(StoreAction::Store), Ok(()));

    // Draws.
    let draw = |encoder: &mut RenderCommandEncoder, instances| {
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, instances, 0)
    };
    assert_eq!(
        draw(&mut encoder, 1),
        Err(invalid("draw_primitives", Problem::NoPipelineState))
    );
    // An empty draw without a pipeline state stops plain Metal too: refused alike.
    assert_eq!(
        draw(&mut encoder, 0),
        Err(invalid("draw_primitives", Problem::NoPipelineState))
    );
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    let draw_problem = |call, problem| Err(invalid(call, Problem::Draw(problem)));
    // No instances: plain Metal draws nothing, where its validation stops the draw (Metal probe
    // case `draw_zero` on an M4).
    assert_eq!(draw(&mut encoder, 0), Ok(()));
    // An empty draw is checked only as plain Metal checks one: past a pipeline state of its
    // kind, it draws nothing whatever the rest of the encoder's state, here an amplification
    // count above the pipeline's maximum of 1, which refuses a draw that draws (its validation
    // stops both).
    encoder.set_vertex_amplification_count(2, None).unwrap();
    assert_eq!(draw(&mut encoder, 0), Ok(()));
    assert_eq!(
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0),
        Ok(())
    );
    assert_eq!(
        draw(&mut encoder, 1),
        draw_problem(
            "draw_primitives",
            DrawProblem::AmplificationCount { count: 2, max: 1 }
        )
    );
    encoder.set_vertex_amplification_count(1, None).unwrap();
    let indices = buffer_with(&device, &[0, 0, 1, 0, 2, 0, 0, 0]).unwrap();
    let indexed = |encoder: &mut RenderCommandEncoder, ty, count, ty_index, offset| {
        encoder.draw_indexed_primitives(ty, count, ty_index, &indices, offset, 1, 0, 0)
    };
    // An index buffer offset that is not a multiple of the index size reads from the offset
    // rounded down, as plain Metal reads it: one index, no triangle.
    assert_eq!(
        indexed(
            &mut encoder,
            PrimitiveType::Triangle,
            1,
            IndexType::UInt32,
            2
        ),
        Ok(())
    );
    // The render encoder's Apple-only setters (step 12's table): Metal's defaults are accepted.
    let apple = |feature| Err(Error::AppleOnly(feature));
    assert_eq!(encoder.set_alpha_test_reference_value(0.5), Ok(()));
    // Every width is accepted (`runtime_lane::line_widths` draws them); the encoder's width is
    // set back to Metal's default for the draws below.
    assert_eq!(encoder.set_line_width(2.0), Ok(()));
    assert_eq!(encoder.set_line_width(1.0), Ok(()));
    // `setPointSize:` changes nothing Metal draws, at every size (088029d75).
    assert_eq!(encoder.set_point_size(4.0), Ok(()));
    // Every mode is drawn, as Metal draws every one; a mode past 1, which Metal draws a third
    // way, as mode 1 (`runtime_lane::provoking` checks the announcement).
    for mode in [0, 1, u64::MAX, 1 << 63, 2, i64::MAX as u64] {
        assert_eq!(encoder.set_provoking_vertex_mode(mode), Ok(()), "{mode:#x}");
    }
    assert_eq!(encoder.set_viewport_transform_enabled(true), Ok(()));
    assert_eq!(
        encoder.set_viewport_transform_enabled(false),
        apple(AppleOnly::ViewportTransformDisabled)
    );
    // A count of 0 (no set bits in mode 1) draws a number of copies that changes from run to
    // run on Metal: refused, the amplification set before kept.
    assert_eq!(
        encoder.set_vertex_amplification_mode(1, 0),
        apple(AppleOnly::VertexAmplificationMode)
    );
    // Depth and stencil "cleared" change nothing Metal draws.
    assert_eq!(encoder.set_depth_cleared(), Ok(()));
    assert_eq!(encoder.set_stencil_cleared(), Ok(()));
    // Nothing above drew; a draw of no vertices records nothing.
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    assert_eq!(float_texels(&target), pattern(6).repeat(4), "nothing drew");

    // A pipeline of another color format than the pass's (measured: Metal's validation stops
    // `setRenderPipelineState:`, and plain Metal stores the pipeline's encoding). The refused
    // call keeps the state set before. (A color the pipeline lacks is drawn, zeroed where
    // covered, and a depth or stencil format the pipeline names otherwise is the pass's, as
    // plain Metal draws them: `extra_colors.rs` and `hidden_depth.rs`.)
    let other = attachment(
        &device,
        PixelFormat::R32Float,
        [4, 4],
        ResourceOptions::private(),
    )
    .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&one_target(&other, LoadAction::Clear, StoreAction::Store))
        .unwrap();
    assert_eq!(
        encoder.set_render_pipeline_state(&pipeline),
        draw_problem(
            "set_render_pipeline_state",
            DrawProblem::AttachmentFormat(Attachment::Color(0))
        )
    );
    assert_eq!(
        draw(&mut encoder, 1),
        Err(invalid("draw_primitives", Problem::NoPipelineState))
    );
    encoder.end_encoding().unwrap();
    run(&command_buffer);

    // A color store action still Unknown at end_encoding stores as `DontCare` (Metal probe case
    // `store_unknown` on an M4, which Metal's validation stops: plain Metal kept the old texels
    // under a clear); a `DontCare` store's texels are the device's
    // (`load_and_store_actions_keep_and_clear_as_metal`).
    let command_buffer = queue.command_buffer().unwrap();
    let encoder = command_buffer
        .render_command_encoder(&one_target(
            &target,
            LoadAction::Clear,
            StoreAction::Unknown,
        ))
        .unwrap();
    assert_eq!(encoder.end_encoding(), Ok(()));
    run(&command_buffer);
    // So does a depth or stencil store action still Unknown without a resolve texture: plain
    // Metal left the texels under a clear, as its `DontCare` (Metal's
    // validation stops it).
    for (format, depth, stencil) in [
        (PixelFormat::Depth32Float, true, false),
        (PixelFormat::Stencil8, false, true),
        (PixelFormat::Depth32FloatStencil8, true, true),
    ] {
        let texture = attachment(&device, format, [4, 4], ResourceOptions::private()).unwrap();
        let mut pass = RenderPassDescriptor::new();
        if depth {
            pass.depth_attachment.texture = Some(texture.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.store_action = StoreAction::Unknown;
        }
        if stencil {
            pass.stencil_attachment.texture = Some(texture.clone());
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.store_action = StoreAction::Unknown;
        }
        let command_buffer = queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        assert_eq!(encoder.end_encoding(), Ok(()), "{format:?}");
        run(&command_buffer);
    }
    quiet(device, &messages);
}
