//! A tour of the API: the flow `docs/API.md` walks, on owned public fixtures and committed device
//! profiles, with no GPU. It needs no feature.
//!
//! ```sh
//! PATH=/opt/homebrew/opt/llvm/bin:$PATH \
//!   cargo run --example api_preview
//! ```
//!
//! The library reads AIR bitcode (a metallib, a bitcode wrapper, or a raw module). The owned
//! fixtures are committed as LLVM text, so the example assembles them with `llvm-as` (LLVM 21 or
//! newer) from `PATH` first, as a consumer holding `.ll` would.
//!
//! It compiles a kernel for the committed `moltenvk-m3` profile (and the same kernel as a bare
//! module's entry, linked alone), prints its pipeline interface, layout, and binding plan,
//! instantiates one dispatch, round-trips the pipeline through its cache encoding, answers a
//! pipeline-input request, shows a typed refusal on `vk12-floor`, dispatches a kernel with
//! `[[stage_in]]`, and compiles, plans, and draws a render pipeline and a tessellation pipeline. `tests/api_preview_example.rs` runs it in the gate and checks that
//! every code block of `docs/API.md` is quoted from it.

use std::path::{Path, PathBuf};
use std::process::Command;

use metal2vulkan::bind::{
    BindState, BoundBuffer, BoundSampler, BoundTexture, Dispatch, DrawCall, SamplerDescriptor,
    StageInRegion, StageTable, TessellationFactorBuffer, TessellationFactors, TextureFacts,
};
use metal2vulkan::cache::PipelineArtifact;
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::error::{BindError, Error, Needs, RuntimeInput};
use metal2vulkan::library::{
    CompiledFunction, FunctionConstantValue, FunctionConstantValues, Library,
};
use metal2vulkan::metal::{
    AttributeFormat, DataType, PixelFormat, StepFunction, TextureType, VertexFormat,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipeline, ComputePipelineDescriptor, EntryDescriptor,
    EntryRef, FunctionDescriptor, LoweringInputs, RenderPipelineDescriptor, Stage,
    StageInputAttributeDescriptor, StageInputDescriptor, StageInputLayoutDescriptor, StageSource,
    TessellationDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use metal2vulkan::{Budget, CancelToken};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The repository root: the committed profiles and fixtures live under it.
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Owned fixture `stem`
/// (`crates/metal2vulkan-validation/fixtures/public/engine/<stem>.ll`) as AIR bitcode.
fn fixture_bitcode(stem: &str) -> Result<Vec<u8>> {
    let ll: PathBuf = repo()
        .join("crates/metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{stem}.ll"));
    let out = Command::new("llvm-as")
        .arg("-o")
        .arg("-")
        .arg(&ll)
        .output()
        .map_err(|e| format!("cannot run llvm-as (put LLVM's bin directory on PATH): {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("llvm-as {}: {stderr}", ll.display()).into());
    }
    Ok(out.stdout)
}

/// A device for committed profile `stem`
/// (`crates/metal2vulkan-validation/profiles/<stem>.json`).
fn committed_device(stem: &str) -> Result<Device> {
    let json = std::fs::read_to_string(repo().join(format!(
        "crates/metal2vulkan-validation/profiles/{stem}.json"
    )))?;
    let description = DeviceDescription::from_khronos_json(&json, None)?;
    let device = Device::new(description, DevicePolicy::default())?;
    Ok(device)
}

/// A buffer bound at `offset` in a `length`-byte allocation whose Metal `gpuAddress` is
/// `0x1_0000`, without a Vulkan device address.
fn buffer(length: u64, offset: u64) -> BoundBuffer {
    BoundBuffer::Buffer {
        length,
        offset,
        gpu_address: 0x1_0000,
        device_address: None,
        attribute_stride: None,
    }
}

/// Lowercase hex of a digest.
fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn main() -> Result<()> {
    // 1. A device: what the consumer enabled, and its policy.
    let device = committed_device("moltenvk-m3")?;
    let enables = device.required_enables();
    println!(
        "device: profile {}, {} features and {} extensions to enable",
        hex(&device.profile().fingerprint()),
        enables.features.len(),
        enables.extensions.len(),
    );
    println!("  Metal properties: {:?}", device.metal_properties());

    // 2. A library from AIR bitcode, and its function table.
    let sampling = Library::from_air(&fixture_bitcode("sampling")?)?;
    for info in sampling.functions() {
        println!("library: {} ({:?})", info.name, info.stage);
    }

    // 3. A compute pipeline, under a 20-second deadline and a cancellation token.
    let cancel = CancelToken::new();
    let mut budget = Budget::standard();
    budget.cancel = Some(cancel.clone());
    let kernel = FunctionDescriptor::new(sampling.function("sample_level")?);
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel));
    let pipeline = device.compile_compute_pipeline(&descriptor, &budget)?;
    println!(
        "compute pipeline {:?}: {} SPIR-V words, entry point {:?}",
        pipeline.key,
        pipeline.stage.spirv.len(),
        pipeline.stage.entry_point,
    );

    // What the module the driver will compile contains: the verifier's counts of it.
    let stats = pipeline.stage.stats;
    println!(
        "{} blocks, {} instructions",
        stats.blocks, stats.instructions
    );
    assert!(stats.reachable_blocks <= stats.blocks);

    // The same kernel as an entry of a bare module, through the device's profile: no library,
    // the same compile, and its module linked alone.
    let bitcode = fixture_bitcode("sampling")?;
    let entry = EntryDescriptor::new(&bitcode, EntryRef::Name("sample_level"));
    let profile = device.profile();
    let (compiled, report) = profile.compile_entry(&entry, &LoweringInputs::new(), &budget);
    let alone = profile.link_standalone(&compiled?)?;
    println!(
        "entry sample_level: {} SPIR-V words linked alone, {} work units",
        alone.words.len(),
        report.work_units
    );

    // 4. What the pipeline binds, how, and what it depends on.
    for (slot, interface) in &pipeline.interface.slots {
        println!("  slot {slot:?}: {interface:?}");
    }
    for set in &pipeline.layout.sets {
        for binding in &set.bindings {
            println!(
                "  set {} binding {}: {:?} x{} from {:?}",
                set.set, binding.binding, binding.ty, binding.count, binding.source
            );
        }
    }
    for range in &pipeline.layout.push_constants {
        println!("  push constants: {range:?}");
    }
    for dependency in &pipeline.dependencies {
        let (fact, mechanism) = (dependency.fact, dependency.mechanism);
        println!("  depends on {fact:?} via {mechanism:?}");
    }
    println!("  precision notes: {:?}", pipeline.stage.precision_notes);
    println!("  Metal info: {:?}", pipeline.metal_info);

    // 5. What is bound (Metal-shaped), the plan for its shape, and one dispatch.
    let mut table = StageTable::default();
    table.buffers.insert(0, buffer(4096, 0));
    let facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm);
    let texture = BoundTexture::Texture {
        facts,
        resource_id: None,
    };
    table.textures.insert(0, texture);
    let sampler = BoundSampler::new(SamplerDescriptor::new());
    table.samplers.insert(0, sampler);
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, table);

    let plan = pipeline.plan(&device, &state.shape())?;
    assert!(plan.variant.is_empty(), "this shape needs no other variant");
    for write in &plan.descriptors {
        println!(
            "  write set {} binding {}[{}] {:?}: {:?}",
            write.set, write.binding, write.array_element, write.ty, write.source
        );
    }
    let dispatch = Dispatch::Threads {
        threads: [100, 1, 1],
        threads_per_group: [32, 1, 1],
    };
    let commands = plan.instantiate_dispatch(&state, &dispatch)?;
    for region in &commands.regions {
        println!(
            "  region: {:?} groups, specialization {:?}",
            region.group_count, region.specialization
        );
    }
    println!("  descriptor ranges: {:?}", commands.descriptor_ranges);
    println!("  push holes: {:?}", commands.push_holes);

    // 6. The cache: a pipeline encodes under its key and decodes for the same device.
    let bytes = pipeline.to_bytes();
    let cached = ComputePipeline::from_bytes_for_key(&bytes, &pipeline.key, &device)?;
    assert_eq!(cached, pipeline);
    println!(
        "cache: {} bytes round-trip under {:?}",
        bytes.len(),
        pipeline.key
    );

    // A function compiles on its own under a key known before compiling, and a pipeline links
    // the cached function without decoding or lowering it again.
    let kernel = FunctionDescriptor::new(sampling.function("sample_level")?);
    let inputs = LoweringInputs::new();
    let key = device.function_compile_key(&kernel, &inputs);
    let compiled = device.compile_function(&kernel, &inputs, &budget)?;
    let bytes = compiled.to_bytes();
    let function = CompiledFunction::from_bytes_for_key(&bytes, &key, &device)?;
    let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(function));
    let linked = device.compile_compute_pipeline(&descriptor, &budget)?;
    assert_eq!(linked, pipeline);
    println!(
        "cache: function {key:?} ({} bytes) links to the same pipeline",
        bytes.len()
    );

    // 7. A function-constant value, supplied by index. A constant left undefined compiles too, as
    // Metal's does: `is_function_constant_defined` names it false, and a value of it alone is zero.
    let library = Library::from_air(&fixture_bitcode("kernel_vector_function_constant")?)?;
    let function = library.function("kernel_vector_function_constant")?;
    let mut kernel = FunctionDescriptor::new(function);
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel.clone()));
    let undefined = device.compile_compute_pipeline(&descriptor, &budget)?;
    let words = [1u32, 2, 3, 4].map(u32::to_le_bytes).concat();
    let value = FunctionConstantValue::new(DataType::UInt4, words);
    kernel.constants = FunctionConstantValues::defaults().with_index(0, value);
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel));
    let specialized = device.compile_compute_pipeline(&descriptor, &budget)?;
    assert_ne!(undefined.key, specialized.key);
    println!("function constant supplied: {:?}", specialized.key);

    // 8. A typed refusal: fragment function `sample_min_lod` clamps its level of detail, which
    // needs `shaderResourceMinLod`; the `vk12-floor` profile lacks it.
    let floor = committed_device("vk12-floor")?;
    let min_lod = FunctionDescriptor::new(sampling.function("sample_min_lod")?);
    match floor.compile_function(&min_lod, &LoweringInputs::new(), &Budget::standard()) {
        Err(Error::Unsupported(refusal)) => match (&refusal.feature, &refusal.needs) {
            (feature, Needs::AnyOf(alternatives)) => {
                println!("refused {feature:?}: needs any of {alternatives:?}");
            }
            (feature, needs) => println!("refused {feature:?}: {needs:?}"),
        },
        Err(other) => return Err(other.into()),
        Ok(_) => return Err("vk12-floor compiled sample_min_lod".into()),
    }

    stage_in(&device, &budget)?;
    render(&device)?;
    tessellation(&device)
}

/// A kernel with `[[stage_in]]` on `device`: the stage-input descriptor it asks for, the kernel
/// compiled on its own under it, and a dispatch that needs the encoder's stage-in region.
fn stage_in(device: &Device, budget: &Budget) -> Result<()> {
    let library = Library::from_air(&fixture_bitcode("kernel_stage_in")?)?;
    let kernel = FunctionDescriptor::new(library.function("stage_in_region")?);
    let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel.clone()));
    match device.compile_compute_pipeline(&descriptor, budget) {
        Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor { attribute })) => {
            let mut stage_input = StageInputDescriptor::new();
            let fetch = StageInputAttributeDescriptor::new(AttributeFormat::Float, 0, 1);
            stage_input.attributes.insert(attribute, fetch);
            let mut layout = StageInputLayoutDescriptor::new(Some(4));
            layout.step_function = StepFunction::ThreadPositionInGridX;
            stage_input.layouts.insert(1, layout);
            descriptor.stage_input = Some(stage_input);
        }
        other => return Err(format!("expected a stage-input request: {other:?}").into()),
    }
    let pipeline = device.compile_compute_pipeline(&descriptor, budget)?;
    println!("stage-in pipeline {:?}", pipeline.key);

    // Compiled on its own, the kernel compiles under the pipeline's stage-input descriptor.
    let mut inputs = LoweringInputs::new();
    inputs.stage_input = descriptor.stage_input.clone();
    let compiled = device.compile_function(&kernel, &inputs, budget)?;
    let mut linked = descriptor.clone();
    linked.kernel = StageSource::Compiled(compiled);
    assert_eq!(device.compile_compute_pipeline(&linked, budget)?, pipeline);

    let mut table = StageTable::default();
    table.buffers.insert(0, buffer(4096, 0));
    table.buffers.insert(1, buffer(256, 0));
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, table);
    let plan = pipeline.plan(device, &state.shape())?;
    let dispatch = Dispatch::Threads {
        threads: [64, 1, 1],
        threads_per_group: [32, 1, 1],
    };
    match plan.instantiate_dispatch(&state, &dispatch) {
        Err(BindError::MissingStageInRegion) => {}
        other => return Err(format!("expected a missing stage-in region: {other:?}").into()),
    }
    state.stage_in_region = Some(StageInRegion::new([0, 0, 0], [64, 1, 1]));
    let commands = plan.instantiate_dispatch(&state, &dispatch)?;
    for region in &commands.regions {
        println!("  region push constants: {:?}", region.push_constants);
    }
    Ok(())
}

/// A render pipeline on `device`: compile, plan, and one draw.
fn render(device: &Device) -> Result<()> {
    let library = Library::from_air(&fixture_bitcode("stage_io")?)?;
    let vertex = FunctionDescriptor::new(library.function("stage_io_vertex")?);
    let fragment = FunctionDescriptor::new(library.function("stage_io_fragment")?);

    let mut vertex_descriptor = VertexDescriptor::new();
    let layout = VertexBufferLayoutDescriptor::new(Some(24));
    vertex_descriptor.layouts.insert(0, layout);
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        let attribute = VertexAttributeDescriptor::new(format, offset, 0);
        vertex_descriptor.attributes.insert(index, attribute);
    }
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    descriptor.fragment = Some(StageSource::Function(fragment));
    descriptor.vertex_descriptor = Some(vertex_descriptor);
    let color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    descriptor.color_attachments[0] = Some(color);
    let color = ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint);
    descriptor.color_attachments[2] = Some(color);
    descriptor.depth_format = Some(PixelFormat::Depth32Float);

    let pipeline = device.compile_render_pipeline(&descriptor, &Budget::standard())?;
    println!("render pipeline {:?}", pipeline.key);
    for stage in &pipeline.stages {
        println!(
            "  stage {:?}: {} SPIR-V words",
            stage.stage,
            stage.spirv.len()
        );
    }
    println!("  vertex input: {:?}", pipeline.vertex_input);

    let mut table = StageTable::default();
    table.buffers.insert(0, buffer(24 * 64, 0));
    table.buffers.insert(1, buffer(256, 16));
    let mut state = BindState::default();
    state.stages.insert(Stage::Vertex, table);
    let plan = pipeline.plan(device, &state.shape())?;
    let draw = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 6,
        instance_count: 1,
        base_instance: 0,
    };
    let commands = plan.instantiate_draw(&state, &draw)?;
    println!("  draw: {:?}", commands.draw);
    println!("  vertex buffer offsets: {:?}", commands.buffer_offsets);
    println!("  push constants: {:?}", commands.push_constants);
    Ok(())
}

/// A tessellation pipeline on `device`: a post-tessellation vertex function, and a patch draw.
fn tessellation(device: &Device) -> Result<()> {
    let library = Library::from_air(&fixture_bitcode("tessellation")?)?;
    let vertex = FunctionDescriptor::new(library.function("tess_quad_buffer")?);
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    descriptor.tessellation = Some(TessellationDescriptor::default());
    let pipeline = device.compile_render_pipeline(&descriptor, &Budget::standard())?;
    let stages: Vec<_> = pipeline.stages.iter().map(|stage| stage.stage).collect();
    println!("tessellation pipeline {:?}: {stages:?}", pipeline.key);
    println!("  {:?}", pipeline.graphics.tessellation);

    let mut table = StageTable::default();
    table.buffers.insert(0, buffer(4096, 0));
    let mut state = BindState::default();
    state.stages.insert(Stage::Vertex, table);
    let factors = TessellationFactorBuffer::new(4096, 0, 0);
    state.tessellation_factors = TessellationFactors::new(Some(factors), 1.0);
    let plan = pipeline.plan(device, &state.shape())?;
    let draw = DrawCall::Patches {
        control_points: 4,
        patch_start: 0,
        patch_count: 16,
        patch_index: None,
        instance_count: 1,
        base_instance: 0,
        control_point_index: None,
    };
    let commands = plan.instantiate_draw(&state, &draw)?;
    println!("  draw: {:?}", commands.draw);
    println!("  push constants: {:?}", commands.push_constants);
    Ok(())
}
