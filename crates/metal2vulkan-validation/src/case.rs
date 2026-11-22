pub mod indirect_commands;
pub mod mips;
pub mod out_of_bounds;
pub mod sparse;

use crate::hash::sha256_bytes;
use crate::jsonl::to_sorted_json_string;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// The sample counts a case's render pass may declare beside one
/// ([`AuthoredCase::raster_sample_count`]): the counts the Metal reference device rasterizes. An
/// Apple M3 Max supports 1, 2, and 4 samples (`supportsTextureSampleCount:`, measured 2026-10-03),
/// not 8, so no Metal answer exists for 8 samples, and a candidate's run of one would have nothing
/// to be compared with.
pub const RASTER_SAMPLE_COUNTS: [u32; 2] = [2, 4];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredCase {
    pub air_sha256: String,
    pub case_id: String,
    pub name: String,
    pub entry: String,
    pub stage: Stage,
    #[serde(default)]
    pub buffers: Vec<BufferResource>,
    #[serde(default)]
    pub argument_buffer_buffers: Vec<ArgumentBufferBufferResource>,
    #[serde(default)]
    pub device_buffer_arrays: Vec<DeviceBufferArrayResource>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command_buffers: Vec<indirect_commands::CommandBufferResource>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pipeline_states: Vec<indirect_commands::PipelineResource>,
    #[serde(default)]
    pub threadgroup_memory: Vec<ThreadgroupMemoryResource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imageblock: Option<ImageblockResource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fragment_imageblock: Option<FragmentImageblockResource>,
    #[serde(default)]
    pub acceleration_structures: Vec<AccelerationStructureResource>,
    #[serde(default)]
    pub visible_function_references: Vec<LinkedFunctionResource>,
    #[serde(default)]
    pub visible_function_tables: Vec<FunctionTableResource>,
    #[serde(default)]
    pub intersection_function_tables: Vec<IntersectionFunctionTableResource>,
    #[serde(default)]
    pub argument_buffer_intersection_function_tables:
        Vec<ArgumentBufferIntersectionFunctionTableResource>,
    #[serde(default)]
    pub textures: Vec<TextureResource>,
    #[serde(default)]
    pub texture_arrays: Vec<TextureArrayResource>,
    #[serde(default)]
    pub argument_buffer_textures: Vec<ArgumentBufferTextureResource>,
    /// Sampler states held in argument buffers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub argument_buffer_samplers: Vec<ArgumentBufferSamplerResource>,
    #[serde(default)]
    pub samplers: Vec<SamplerResource>,
    #[serde(default)]
    pub render_targets: Vec<RenderTargetResource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_stencil: Option<DepthStencilResource>,
    #[serde(default)]
    pub vertex_inputs: Vec<VertexInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertex_observation: Option<VertexObservation>,
    /// Values a fragment case's generated passthrough vertex writes to its varyings in place of
    /// the defaults ([`VaryingValue`]). Absent for every varying the defaults serve, so no case
    /// authored before this field changes identity. Part of the semantic identity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub varying_values: Vec<VaryingValue>,
    #[serde(default)]
    pub kernel_stage_inputs: Vec<KernelStageInput>,
    #[serde(default)]
    pub function_constants: Vec<FunctionConstant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch: Option<Dispatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw: Option<Draw>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tessellation: Option<TessellationDraw>,
    /// An object or mesh case's draw of its mesh render pipeline ([`MeshDraw`]). Absent from every
    /// other case, so no case authored before this field changes identity. Part of the semantic
    /// identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_draw: Option<MeshDraw>,
    pub output: OutputSelection,
    pub compare: Comparison,
    pub execution_safety: ExecutionSafety,
    /// Per-invocation loop-header visits a fuel-instrumented executor grants this case, in
    /// `1..=2^24` (checker cap). Absent means the executor default (2^20). Part of the semantic
    /// identity when present, because exhaustion changes the run's outcome. The Metal oracle
    /// grants it to a cyclic case through fuel-instrumented AIR; the old-engine candidate has only
    /// a per-loop budget, so it refuses a case that declares it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fuel: Option<u32>,
    /// The resources the case accesses past their range and what a run must show
    /// ([`out_of_bounds`]). Part of the semantic identity: it decides how the Metal oracle
    /// allocates the targets and how a candidate's run is judged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_of_bounds: Option<out_of_bounds::OutOfBounds>,
    /// The render pass's rasterization-rate map (`MTLRenderPassDescriptor.rasterizationRateMap`),
    /// in a rasterizing render case: the Metal oracle renders the pass through Metal's compile of
    /// the map, with the viewport the map's screen. A Vulkan candidate carries the pass region by
    /// region: every map of the case is the
    /// product's compile of it, and the pass is drawn region by region into the compile's physical
    /// size, Metal's pass over the identity and a near-equivalent over any other map. Part of the
    /// semantic identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_pass_rate_map: Option<RateMapLiteral>,
    /// The render pass's sample count: Metal's `rasterSampleCount` of the pipeline, the sample
    /// count of every attachment, and the pass's default raster sample count without one, in a
    /// fragment case. Absent for one sample, the only spelling of it, so no case authored before
    /// this field changes identity; otherwise one of [`RASTER_SAMPLE_COUNTS`]. Every attachment's
    /// bytes then hold each pixel's samples in order, the layout a multisampled texture literal
    /// uses, and a render-target or depth output selects every sample of its region, never
    /// resolved. Part of the semantic identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_sample_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    /// A kernel case's own bound on each driver creation of its pipelines, in place of the shared
    /// ceiling (`budgets::PIPELINE_CREATION_CEILING`), with what it was set from
    /// ([`CreationBound`]). A test bound the compute executors hold each creation to, never a
    /// product one. Execution never reads it, so like the name and the rationale it is not part of
    /// the identity (`case_id`, `input_sha256`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub creation_bound: Option<CreationBound>,
    /// Why the case's work is meant to leave its selected output exactly as authored: an early exit
    /// its parameters select, a callback that writes nothing, or a write of the bytes it read.
    /// Metal qualification refuses an answer equal to the selected initial bytes unless the case
    /// declares this, and refuses a declared case whose answer differs (`crate::no_effect`).
    /// Execution never reads it, so like the name and the rationale it is not part of the identity
    /// (`case_id`, `input_sha256`): declaring it annotates the case and keeps its observations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_effect: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored_by: Option<String>,
    /// The repository-relative path of the public fixture whose text is this case's AIR
    /// (`crates/metal2vulkan-validation/fixtures/public/...`), present exactly when `air_sha256` is a public fixture
    /// module's hash. Like the name, rationale, and author it is not part of the identity
    /// (`case_id`, `input_sha256`). `corpus-case-check --install` sets it from the AIR, and the
    /// checker refuses a case whose named fixture no longer hashes to `air_sha256`: an edited
    /// fixture whose case was not re-keyed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_source: Option<String>,
}

#[cfg(test)]
pub(crate) const ATTACHMENTLESS_FRAGMENT_AIR: &str = r#"target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @fragment_no_writes() {
  ret void
}

!air.fragment = !{!0}
!0 = !{ptr @fragment_no_writes, !1, !2}
!1 = !{}
!2 = !{}
"#;

#[cfg(test)]
pub(crate) fn attachmentless_fragment_test_case(air_sha256: String, entry: String) -> AuthoredCase {
    AuthoredCase {
        air_sha256,
        case_id: "test-attachmentless-fragment".into(),
        name: "attachmentless-fragment-smoke".into(),
        entry,
        stage: Stage::Fragment,
        buffers: vec![],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: None,
        draw: Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            index_buffer: None,
            base_vertex: 0,
            base_instance: 0,
        }),
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::None,
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    }
}

// Callers are the Metal device tests, which build only on macOS under `--cfg m2v_gpu`.
#[cfg(all(test, target_os = "macos", m2v_gpu))]
pub(crate) fn combined_depth_stencil_test_case(air_sha256: String, entry: String) -> AuthoredCase {
    AuthoredCase {
        air_sha256,
        case_id: "test-fragment-depth-stencil".into(),
        name: "fragment-depth-stencil-smoke".into(),
        entry,
        stage: Stage::Fragment,
        buffers: vec![],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: Some(DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: Some("AACAPw==".into()),
            initial_stencil_b64: Some("qw==".into()),
            state: None,
        }),
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: None,
        draw: Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            index_buffer: None,
            base_vertex: 0,
            base_instance: 0,
        }),
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Stencil {
            origin: [0, 0],
            dimensions: [1, 1],
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    }
}

#[cfg(test)]
pub(crate) fn vector_function_constant_test_case(
    air_sha256: String,
    entry: String,
) -> AuthoredCase {
    AuthoredCase {
        air_sha256,
        case_id: "test-vector-function-constant".into(),
        name: "vector-function-constant-smoke".into(),
        entry,
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![FunctionConstant {
            index: 0,
            scalar_type: ScalarType::U32,
            lanes: 4,
            bytes_b64: "AQAAAAIAAAADAAAABAAAAA==".into(),
        }],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    }
}

#[cfg(test)]
pub(crate) fn narrow_implicit_imageblock_test_case(
    air_sha256: String,
    entry: String,
) -> AuthoredCase {
    AuthoredCase {
        air_sha256,
        case_id: "test-narrow-implicit-imageblock".into(),
        name: "narrow-implicit-imageblock-smoke".into(),
        entry,
        stage: Stage::Kernel,
        buffers: vec![],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: Some(ImageblockResource {
            dimensions: [16, 16],
            implicit_coverage: Some(ImplicitImageblockCoverage::FullSingleSample),
        }),
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![RenderTargetResource {
            index: 0,
            format: TextureFormat::Rg16Float,
            dimensions: [16, 16],
            initial_bytes_b64: base64::engine::general_purpose::STANDARD
                .encode([0x00, 0x3c, 0x00, 0x40].repeat(16 * 16)),
        }],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [16, 16, 1],
            threads_per_threadgroup: [16, 16, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [1, 1],
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    }
}

#[cfg(all(test, target_os = "macos", m2v_gpu))]
pub(crate) fn fragment_imageblock_test_case(air_sha256: String, entry: String) -> AuthoredCase {
    let initial_half = base64::engine::general_purpose::STANDARD.encode([0x00, 0x3c]);
    AuthoredCase {
        air_sha256,
        case_id: "test-fragment-imageblock".into(),
        name: "fragment-imageblock-smoke".into(),
        entry,
        stage: Stage::Fragment,
        buffers: vec![],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: Some(FragmentImageblockResource {
            dimensions: [1, 1],
            members: vec![FragmentImageblockMemberResource {
                semantic: "user(depth)".into(),
                format: FragmentImageblockFormat::Half,
                role: ResourceRole::InOut,
                bytes_b64: None,
                initial_bytes_b64: Some(initial_half),
            }],
        }),
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba16Float,
            dimensions: [1, 1],
            initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode([0u8; 8]),
        }],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: None,
        draw: Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            index_buffer: None,
            base_vertex: 0,
            base_instance: 0,
        }),
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::FragmentImageblock {
            semantic: "user(depth)".into(),
            origin: [0, 0],
            dimensions: [1, 1],
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    }
}

/// A kernel case's own bound on each driver creation of its pipelines
/// ([`AuthoredCase::creation_bound`]), above the shared ceiling: for a module whose compile, on a
/// device and driver measured, takes more of its thread's own CPU time than the ceiling on a host
/// whose other work shares the CPU's cores, where the clock the executors read
/// (`budgets::CreationClock`) already leaves out the thread's waits for a CPU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreationBound {
    /// The bound, in milliseconds.
    pub ms: u64,
    /// The device and driver it was measured on, and the measurements.
    pub measured: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Kernel,
    Vertex,
    Fragment,
    /// An object function (`!air.object`), the first stage of a mesh render pipeline.
    Object,
    /// A mesh function (`!air.mesh`), a mesh render pipeline's geometry stage.
    Mesh,
}

impl Stage {
    /// Every stage, in the order a module's entry lists are read
    /// ([`crate::air::stage_entry_from_ll`]).
    pub const ALL: [Self; 5] = [
        Self::Kernel,
        Self::Vertex,
        Self::Fragment,
        Self::Object,
        Self::Mesh,
    ];

    /// The stage's label in a source row and the index (`Kernel`).
    pub fn metadata_label(self) -> &'static str {
        match self {
            Self::Kernel => "Kernel",
            Self::Vertex => "Vertex",
            Self::Fragment => "Fragment",
            Self::Object => "Object",
            Self::Mesh => "Mesh",
        }
    }

    /// The stage a source row or the index labels `label` ([`Stage::metadata_label`]).
    pub fn from_metadata_label(label: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|stage| stage.metadata_label() == label)
    }

    /// The suffix of the `!air.<list>` named metadata that lists the module's entries of this
    /// stage (`kernel`).
    pub fn entry_list(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Object => "object",
            Self::Mesh => "mesh",
        }
    }

    /// Whether the stage is one of a mesh render pipeline's object and mesh stages.
    pub fn is_mesh_pipeline_stage(self) -> bool {
        matches!(self, Self::Object | Self::Mesh)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRole {
    Input,
    Output,
    InOut,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferResource {
    pub binding: u32,
    pub role: ResourceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
    /// An input buffer holding a rasterization-rate map's parameter data, in place of `bytes_b64`:
    /// each executor builds the described map with its implementation and writes that
    /// implementation's data, since the layout is private to it (Metal's is Apple's, the product's
    /// is the AIR front end's `canon::rate_map_data`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rasterization_rate_map: Option<RateMapLiteral>,
}

/// A rasterization-rate map (`MTLRasterizationRateMapDescriptor`) whose parameter data a buffer
/// holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateMapLiteral {
    /// `screenSize`: width and height in screen pixels, each 1 to 65536.
    pub screen_size: [u32; 2],
    /// The layers, from layer 0; at least one.
    pub layers: Vec<RateMapLayerLiteral>,
}

/// One layer's quality samples (`MTLRasterizationRateLayerDescriptor`), each finite and in
/// `[0, 1]`, at least one per axis.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateMapLayerLiteral {
    pub horizontal: Vec<RateSample>,
    pub vertical: Vec<RateSample>,
}

/// A quality sample, equal to another when its bits are.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RateSample(pub f32);

impl PartialEq for RateSample {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for RateSample {}

impl RateMapLiteral {
    /// The product's descriptor of this map.
    pub fn descriptor(&self) -> metal2vulkan::bind::RateMapDescriptor {
        let samples = |axis: &[RateSample]| axis.iter().map(|sample| sample.0).collect();
        let mut descriptor = metal2vulkan::bind::RateMapDescriptor::new(self.screen_size);
        descriptor.layers = self
            .layers
            .iter()
            .map(|layer| {
                metal2vulkan::bind::RateMapLayerDescriptor::new(
                    samples(&layer.horizontal),
                    samples(&layer.vertical),
                )
            })
            .collect();
        descriptor
    }

    /// The product's compile of this map (`bind::RasterizationRateMap::new`, Apple M3's rule): the
    /// map a Vulkan executor answers every map of a case with and draws a rate-mapped pass with,
    /// region by region (the region path, the only one the executors run).
    pub fn compiled(&self) -> Result<metal2vulkan::bind::RasterizationRateMap, String> {
        metal2vulkan::bind::RasterizationRateMap::new(&self.descriptor())
            .map_err(|error| format!("rasterization-rate map: {error}"))
    }

    /// The parameter data the product's implementation writes for this map
    /// (`bind::RasterizationRateMap::to_bytes` of [`Self::compiled`]): what a Vulkan executor
    /// binds.
    pub fn parameter_data(&self) -> Result<Vec<u8>, String> {
        self.compiled().map(|map| map.to_bytes())
    }

    fn validate(&self, label: &str, errors: &mut Vec<String>) {
        let before = errors.len();
        for (axis, extent) in self.screen_size.into_iter().enumerate() {
            if extent == 0 || extent > metal2vulkan::bind::MAX_SCREEN_EXTENT {
                errors.push(format!(
                    "{label} screen extent {extent} (axis {axis}) is outside 1..={}",
                    metal2vulkan::bind::MAX_SCREEN_EXTENT
                ));
            }
        }
        if self.layers.is_empty() {
            errors.push(format!("{label} has no layer"));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            for (axis, samples) in [&layer.horizontal, &layer.vertical].into_iter().enumerate() {
                if samples.is_empty() {
                    errors.push(format!("{label} layer {index} axis {axis} has no sample"));
                }
                if samples
                    .iter()
                    .any(|sample| !(0.0..=1.0).contains(&sample.0))
                {
                    errors.push(format!(
                        "{label} layer {index} axis {axis} has a sample outside [0, 1]"
                    ));
                }
            }
        }
        if errors.len() == before {
            if let Err(error) = self.parameter_data() {
                errors.push(format!("{label}: {error}"));
            }
        }
    }
}

/// A device-buffer handle encoded inside an `air.indirect_buffer` argument. Its identity is the
/// owner buffer plus field offset; reflection supplies the Metal argument index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArgumentBufferBufferResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub role: ResourceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
    /// An input member buffer holding a rasterization-rate map's parameter data, in place of
    /// `bytes_b64`, as [`BufferResource::rasterization_rate_map`] is for a buffer: each executor
    /// encodes a buffer of its own implementation's data of the map into the member.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rasterization_rate_map: Option<RateMapLiteral>,
}

/// A top-level AIR `array_ref<void>` whose backing allocation contains GPU addresses of separately
/// owned device buffers. Slots may be sparse; every undeclared slot is encoded as a null address.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceBufferArrayResource {
    pub binding: u32,
    pub length: u32,
    pub elements: Vec<DeviceBufferArrayElementResource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceBufferArrayElementResource {
    pub index: u32,
    pub role: ResourceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

/// A dynamically sized Metal `threadgroup` buffer argument.
///
/// Its Metal `[[threadgroup(n)]]` index is distinct from `[[buffer(n)]]`; Vulkan emits it as
/// descriptor-free `Workgroup` storage. Contents are deliberately not authorable: both APIs leave
/// this memory uninitialized, so a meaningful case must initialize every byte it reads in the
/// shader.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThreadgroupMemoryResource {
    pub binding: u32,
    pub length: u32,
}

/// Compute imageblock dimensions and, for implicit layouts, the authored raster coverage that
/// makes attachment writes persistent. Explicit imageblock storage remains shader-initialized and
/// therefore has no host bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageblockResource {
    pub dimensions: [u32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implicit_coverage: Option<ImplicitImageblockCoverage>,
}

/// Per-pixel custom fragment imageblock planes. Members are identified by the stable AIR user
/// semantic shared by the master and its entry projections, never by a source struct/member name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentImageblockResource {
    pub dimensions: [u32; 2],
    pub members: Vec<FragmentImageblockMemberResource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentImageblockMemberResource {
    pub semantic: String,
    #[serde(default, skip_serializing_if = "FragmentImageblockFormat::is_half")]
    pub format: FragmentImageblockFormat,
    pub role: ResourceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

/// Exact scalar/vector texel representation of one custom fragment-imageblock plane.
///
/// This mirrors the product's structurally supported AIR master-member types. Keeping it in the
/// authored literal makes byte extents and backend formats explicit without relying on source
/// member names or guessing from payload length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FragmentImageblockFormat {
    #[default]
    Half,
    Half4,
    Uchar4,
    Ushort,
    /// A 32-bit float member.
    Float,
    /// A 32-bit unsigned integer member.
    Uint,
}

impl FragmentImageblockFormat {
    pub fn from_air_type(type_name: &str, size: u32) -> Option<Self> {
        match (type_name, size) {
            ("half", 2) => Some(Self::Half),
            ("half4", 8) => Some(Self::Half4),
            ("uchar4", 4) => Some(Self::Uchar4),
            ("ushort", 2) => Some(Self::Ushort),
            ("float", 4) => Some(Self::Float),
            ("uint", 4) => Some(Self::Uint),
            _ => None,
        }
    }

    pub fn air_type_name(self) -> &'static str {
        match self {
            Self::Half => "half",
            Self::Half4 => "half4",
            Self::Uchar4 => "uchar4",
            Self::Ushort => "ushort",
            Self::Float => "float",
            Self::Uint => "uint",
        }
    }

    pub fn byte_size(self) -> usize {
        self.texture_format().bytes_per_pixel()
    }

    pub fn texture_format(self) -> TextureFormat {
        match self {
            Self::Half => TextureFormat::R16Float,
            Self::Half4 => TextureFormat::Rgba16Float,
            Self::Uchar4 => TextureFormat::Rgba8Uint,
            Self::Ushort => TextureFormat::R16Uint,
            Self::Float => TextureFormat::R32Float,
            Self::Uint => TextureFormat::R32Uint,
        }
    }

    pub fn is_half(&self) -> bool {
        *self == Self::Half
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplicitImageblockCoverage {
    /// A deterministic fullscreen triangle covers every pixel and the sole raster sample before
    /// the authored tile kernel runs. Color writes are disabled during this coverage prepass, so
    /// the authored render-target bytes remain the imageblock's initial values.
    FullSingleSample,
}

/// A literal acceleration structure shared by the Metal oracle and Vulkan candidate.
///
/// Instance resources use the existing host-defined child-reference shadow ABI. Primitive
/// resources carry tightly packed little-endian `float3` triangle vertices (nine floats, or 36
/// bytes, per triangle), from which Metal builds the native BLAS and Vulkan builds its shadow.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccelerationStructureResource {
    pub binding: u32,
    #[serde(
        default,
        skip_serializing_if = "AccelerationStructureKind::is_instance"
    )]
    pub kind: AccelerationStructureKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primitive_triangles_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub child_references: Vec<u64>,
}

/// An explicitly sized Metal function table with zero or more populated slots.
///
/// Every populated entry resolves a retained, hash-identified non-entry AIR module and exact LLVM
/// function symbol from the same harvested metallib provenance as the stage entry. Indices absent
/// from `entries` are authored null slots. The index is the value consumed by AIR table lookup
/// intrinsics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionTableResource {
    pub binding: u32,
    /// Total table capacity, including explicitly null slots.
    pub size: u32,
    pub entries: Vec<FunctionTableEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionTableEntry {
    pub index: u32,
    pub module_sha256: String,
    pub function: String,
}

/// One logical function named by AIR's direct `air.visible_function_reference` metadata.
/// The dependency module is retained separately in the private library-module store.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkedFunctionResource {
    pub module_sha256: String,
    pub function: String,
}

/// An intersection-function table can contain linked procedural callbacks or Metal's explicit
/// opaque-triangle sentinel. The latter is a real table entry, not a missing/null slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntersectionFunctionTableResource {
    pub binding: u32,
    pub size: u32,
    pub entries: Vec<IntersectionFunctionTableEntry>,
}

/// An intersection-function table handle encoded inside an `air.indirect_buffer` argument.
/// Its stable identity is the owning Metal buffer binding plus the member's byte offset; the AIR
/// metadata supplies the backend argument-encoder index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArgumentBufferIntersectionFunctionTableResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub size: u32,
    pub entries: Vec<IntersectionFunctionTableEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IntersectionFunctionTableEntry {
    Linked {
        index: u32,
        module_sha256: String,
        function: String,
    },
    OpaqueTriangle {
        index: u32,
        signature: Vec<IntersectionFunctionSignature>,
    },
}

impl IntersectionFunctionTableEntry {
    pub fn index(&self) -> u32 {
        match self {
            Self::Linked { index, .. } | Self::OpaqueTriangle { index, .. } => *index,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntersectionFunctionSignature {
    Instancing,
    TriangleData,
    WorldSpaceData,
    InstanceMotion,
    PrimitiveMotion,
    ExtendedLimits,
    MaxLevels,
    IntersectionFunctionBuffer,
    UserData,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerationStructureKind {
    #[default]
    Instance,
    Primitive,
}

impl AccelerationStructureKind {
    /// The triangle each instance of an instance structure instantiates, as both executors build
    /// it: three vertices of three floats.
    pub const INSTANCED_TRIANGLE: [f32; 9] = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0];

    fn is_instance(&self) -> bool {
        *self == Self::Instance
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureResource {
    pub binding: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub dimensions: [u32; 3],
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub sample_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
    /// A sparse texture and what is mapped of it before the work ([`sparse`]). Part of the semantic
    /// identity; absent, the texture is an ordinary one, as before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<sparse::SparseTextureResource>,
    /// The texture's mipmap levels, every one authored in the bytes, level 0 first ([`mips`]). Part
    /// of the semantic identity when present; one level, the default, is written by leaving it out,
    /// so a case written before the field keeps its identity.
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub mip_levels: u32,
}

/// A texture handle encoded inside an `air.indirect_buffer` argument.
///
/// The structural identity is the owning Metal buffer binding plus field byte offset. Product
/// reflection supplies the backend-specific Metal argument index and synthetic Vulkan descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArgumentBufferTextureResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub dimensions: [u32; 3],
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub sample_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

/// A Metal texture-handle array. Element order is the descriptor/Metal handle index used by AIR.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureArrayResource {
    pub binding: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub sample_count: u32,
    /// The specialized fixed-array path owns the base Metal texture slot shared with its layered
    /// alternative. Metal binds texture arrays after direct textures, so this declaration makes
    /// that otherwise ambiguous overwrite explicit and validated.
    #[serde(default, skip_serializing_if = "is_false")]
    pub overrides_texture_at_base: bool,
    pub elements: Vec<TextureArrayElement>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureArrayElement {
    pub dimensions: [u32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureType {
    Buffer,
    D1,
    D1Array,
    D2,
    D2Array,
    D2Multisample,
    D2MultisampleArray,
    D3,
    Cube,
    CubeArray,
}

const fn default_one() -> u32 {
    1
}

const fn is_one(value: &u32) -> bool {
    *value == 1
}

const fn is_false(value: &bool) -> bool {
    !*value
}

fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

const fn texture_type_is_array_alternative(arrayed: TextureType, element: TextureType) -> bool {
    matches!(
        (arrayed, element),
        (TextureType::D1Array, TextureType::D1)
            | (TextureType::D2Array, TextureType::D2)
            | (TextureType::D2MultisampleArray, TextureType::D2Multisample)
            | (TextureType::CubeArray, TextureType::Cube)
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureFormat {
    R8Unorm,
    Rgba8Unorm,
    Rgba8Uint,
    Rgba8Sint,
    R16Float,
    R16Uint,
    Rg16Float,
    Rg32Float,
    Rgba16Float,
    Rgba16Uint,
    Rgba16Sint,
    R32Uint,
    R32Sint,
    R32Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Depth32Float,
    /// The format of Metal's `ulong` textures, the only one Metal binds to them: each texel's low
    /// word in R and its high word in G. A case binds it only to a single-sample texture whose
    /// operations spell a `ulong` texel ([`TextureFormat::is_ulong`]).
    Rg32Uint,
    R8Snorm,
    Rgba8Snorm,
    R16Unorm,
    Rgba16Unorm,
    R16Snorm,
    Rg16Snorm,
    Rgba16Snorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    R8UnormSrgb,
    Rg8Unorm,
    Rg8UnormSrgb,
    Rg8Snorm,
    Rg16Unorm,
    A8Unorm,
    B5g6r5Unorm,
    A1bgr5Unorm,
    Abgr4Unorm,
    Bgr5a1Unorm,
    Rgb10a2Unorm,
    Bgr10a2Unorm,
    Rg11b10Float,
    Rgb9e5Float,
}

impl TextureFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::R8Unorm | Self::R8Snorm | Self::R8UnormSrgb | Self::A8Unorm => 1,
            Self::R16Float | Self::R16Uint | Self::R16Unorm | Self::R16Snorm => 2,
            Self::Rg8Unorm | Self::Rg8UnormSrgb | Self::Rg8Snorm => 2,
            Self::B5g6r5Unorm | Self::A1bgr5Unorm | Self::Abgr4Unorm | Self::Bgr5a1Unorm => 2,
            Self::Rgba8Unorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Rgba8Snorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float => 4,
            Self::Rg16Float | Self::Rg16Snorm | Self::Rg16Unorm => 4,
            Self::Rgb10a2Unorm | Self::Bgr10a2Unorm | Self::Rg11b10Float | Self::Rgb9e5Float => 4,
            Self::Rg32Float | Self::Rg32Uint => 8,
            Self::Rgba16Float
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm => 8,
            Self::Rgba32Uint | Self::Rgba32Sint | Self::Rgba32Float => 16,
            Self::Depth32Float => 4,
        }
    }

    /// Whether a case may bind this format only to a texture it samples or reads: not to a
    /// storage texture, a multisampled texture, or a render target. The executors create these
    /// formats from their bytes alone, and neither writes one.
    pub fn sampled_only(self) -> bool {
        matches!(
            self,
            Self::R8Snorm
                | Self::Rgba8Snorm
                | Self::R16Unorm
                | Self::Rgba16Unorm
                | Self::R16Snorm
                | Self::Rg16Snorm
                | Self::Rgba16Snorm
                | Self::Rgba8UnormSrgb
                | Self::Bgra8Unorm
                | Self::Bgra8UnormSrgb
                | Self::R8UnormSrgb
                | Self::Rg8Unorm
                | Self::Rg8UnormSrgb
                | Self::Rg8Snorm
                | Self::Rg16Unorm
                | Self::A8Unorm
                | Self::B5g6r5Unorm
                | Self::A1bgr5Unorm
                | Self::Abgr4Unorm
                | Self::Bgr5a1Unorm
                | Self::Rgb10a2Unorm
                | Self::Bgr10a2Unorm
                | Self::Rg11b10Float
                | Self::Rgb9e5Float
        )
    }

    /// The AIR component class a texture declaration must carry to read or write this format.
    ///
    /// Every format has exactly one class, so this is the whole answer to the question the
    /// sampled/storage texture check and the colour render-target check both ask. Spelling it
    /// instead as one membership list per class in each of them let the two copies drift: the
    /// render-target list accepted `r16_float` and `rg16_float` for a `half` target while the
    /// texture list did not, so no case could bind a `texture2d<half, sample>` to a
    /// single-channel half texture even though both executors bind that format.
    pub fn component_class(self) -> crate::air::reflect::TextureComponent {
        use crate::air::reflect::TextureComponent;
        match self {
            Self::R8Unorm
            | Self::Rgba8Unorm
            | Self::R16Float
            | Self::Rg16Float
            | Self::Rg32Float
            | Self::Rgba16Float
            | Self::R32Float
            | Self::Rgba32Float
            | Self::Depth32Float
            | Self::R8Snorm
            | Self::Rgba8Snorm
            | Self::R16Unorm
            | Self::Rgba16Unorm
            | Self::R16Snorm
            | Self::Rg16Snorm
            | Self::Rgba16Snorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::R8UnormSrgb
            | Self::Rg8Unorm
            | Self::Rg8UnormSrgb
            | Self::Rg8Snorm
            | Self::Rg16Unorm
            | Self::A8Unorm
            | Self::B5g6r5Unorm
            | Self::A1bgr5Unorm
            | Self::Abgr4Unorm
            | Self::Bgr5a1Unorm
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float
            | Self::Rgb9e5Float => TextureComponent::Float,
            Self::Rgba8Uint
            | Self::R16Uint
            | Self::Rgba16Uint
            | Self::R32Uint
            | Self::Rgba32Uint
            | Self::Rg32Uint => TextureComponent::Uint,
            Self::Rgba8Sint | Self::R32Sint | Self::Rgba16Sint | Self::Rgba32Sint => {
                TextureComponent::Sint
            }
        }
    }

    /// Whether this is the format of Metal's `ulong` textures (`Rg32Uint`). It backs only a texture
    /// whose operations spell a `ulong` texel, and no such texture is backed by another format
    /// (`check::interface::TextureUse::accepts_format`); a case binds it to no render target and no
    /// multisampled texture.
    pub fn is_ulong(self) -> bool {
        matches!(self, Self::Rg32Uint)
    }

    /// Whether this format is a depth format. Depth carries a float component class but is not a
    /// colour attachment, so the render-target check excludes it on top of the component class.
    pub fn is_depth(self) -> bool {
        matches!(self, Self::Depth32Float)
    }
}

/// A sampler state encoded inside an `air.indirect_buffer` argument: one made with
/// `supportArgumentBuffers`, whose `gpuResourceID` the argument buffer holds at `field_offset` of
/// the buffer bound at `buffer_binding`. Its descriptor is a [`SamplerResource`]'s, with a
/// comparison function for a sampler a function compares through.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArgumentBufferSamplerResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub address_mode: SamplerAddressMode,
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub normalized_coordinates: bool,
    /// `MTLSamplerDescriptor.compareFunction`; absent is Metal's default, `never`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compare_function: Option<SamplerCompareFunction>,
}

/// `MTLCompareFunction` of a sampler descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerCompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplerResource {
    pub binding: u32,
    pub address_mode: SamplerAddressMode,
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub normalized_coordinates: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerAddressMode {
    ClampToEdge,
    ClampToZero,
    Repeat,
    MirroredRepeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerMipFilter {
    NotMipmapped,
    Nearest,
    Linear,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderTargetResource {
    pub index: u32,
    pub format: TextureFormat,
    pub dimensions: [u32; 2],
    pub initial_bytes_b64: String,
}

/// Literal graphics depth/stencil attachment. Each present aspect is tightly packed in pixel
/// order: little-endian `f32` depth values and one byte per stencil value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepthStencilResource {
    pub dimensions: [u32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_depth_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_stencil_b64: Option<String>,
    /// The render encoder's depth/stencil state for the draw. Absent, both executors set the state
    /// the fragment function's live outputs imply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<DepthStencilStateLiteral>,
}

/// A literal Metal render encoder depth/stencil state: the `MTLDepthStencilDescriptor` set with
/// `setDepthStencilState:`, the references of `setStencilFrontReferenceValue:backReferenceValue:`,
/// and the winding of `setFrontFacingWinding:`. Masks and references are 8-bit: Metal's stencil
/// attachments are 8-bit, and what Metal does with a wider mask is not read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepthStencilStateLiteral {
    pub depth_compare: CompareFunctionLiteral,
    pub depth_write: bool,
    pub front: StencilFaceLiteral,
    pub back: StencilFaceLiteral,
    pub front_reference: u32,
    pub back_reference: u32,
    pub front_facing_winding: WindingLiteral,
}

/// One face of a literal `MTLDepthStencilDescriptor`: an `MTLStencilDescriptor`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StencilFaceLiteral {
    pub compare: CompareFunctionLiteral,
    pub stencil_failure: StencilOperationLiteral,
    pub depth_failure: StencilOperationLiteral,
    pub depth_stencil_pass: StencilOperationLiteral,
    pub read_mask: u32,
    pub write_mask: u32,
}

/// `MTLCompareFunction`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareFunctionLiteral {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// `MTLStencilOperation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StencilOperationLiteral {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

/// `MTLWinding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindingLiteral {
    Clockwise,
    CounterClockwise,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributeInput {
    pub location: u32,
    pub format: AttributeFormat,
    pub stride: u32,
    /// The layout's stride is `MTLBufferLayoutStrideDynamic`: the fetch reads the stride the data's
    /// buffer is bound with, and the case binds it with the stride form
    /// (`setBuffer:offset:attributeStride:atIndex:`, `setVertexBuffer:...`) at `stride`, so a
    /// command that binds the buffer at another stride reads it at that one. Absent, the layout's
    /// stride is `stride`. A kernel stage input's or a vertex input's: no executor models a dynamic
    /// layout of tessellation inputs.
    #[serde(default, skip_serializing_if = "is_false")]
    pub dynamic_stride: bool,
    pub bytes_b64: String,
}

/// The value a fragment case's generated passthrough vertex writes to one of the fragment
/// function's varyings, in place of the default `crate::companion` gives it: the same value at all
/// three vertices of the fullscreen triangle, so every fragment receives it whatever the varying's
/// interpolation. The varying is named by its user key, Metal's interface-matching identity; the
/// checker refuses a key no fragment input declares and lanes that do not fit the input's IR type.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VaryingValue {
    /// The fragment input's user key as its AIR metadata spells it (`generated(...)` or
    /// `user(...)`).
    pub key: String,
    /// Each lane's bits at the varying's scalar width, in lane order: an IEEE binary32 or
    /// binary16 pattern for a float or half lane, two's complement for a 32- or 16-bit integer
    /// lane, and 0 or 1 for a boolean lane. As many lanes as the varying's IR type has.
    pub lanes: Vec<u32>,
}

/// Per-vertex graphics attributes and per-thread compute stage inputs share the same literal record
/// layout. Their containing manifest fields determine the stage-specific fetch semantics.
pub type VertexInput = AttributeInput;
pub type KernelStageInput = AttributeInput;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeFormat {
    Char,
    Char2,
    Char3,
    Char4,
    Uchar,
    Uchar2,
    Uchar3,
    Uchar4,
    Short,
    Short2,
    Short3,
    Short4,
    Ushort,
    Ushort2,
    Ushort3,
    Ushort4,
    Half,
    Half2,
    Half3,
    Half4,
    Float,
    Float2,
    Float3,
    Float4,
    Uint,
    Uint2,
    Uint3,
    Uint4,
    Int,
    Int2,
    Int3,
    Int4,
}

impl AttributeFormat {
    pub fn from_air_type_name(name: &str) -> Option<Self> {
        Some(match name {
            "char" => Self::Char,
            "char2" => Self::Char2,
            "char3" => Self::Char3,
            "char4" => Self::Char4,
            "uchar" => Self::Uchar,
            "uchar2" => Self::Uchar2,
            "uchar3" => Self::Uchar3,
            "uchar4" => Self::Uchar4,
            "short" => Self::Short,
            "short2" => Self::Short2,
            "short3" => Self::Short3,
            "short4" => Self::Short4,
            "ushort" => Self::Ushort,
            "ushort2" => Self::Ushort2,
            "ushort3" => Self::Ushort3,
            "ushort4" => Self::Ushort4,
            "half" => Self::Half,
            "half2" => Self::Half2,
            "half3" => Self::Half3,
            "half4" => Self::Half4,
            "float" => Self::Float,
            "float2" => Self::Float2,
            "float3" => Self::Float3,
            "float4" => Self::Float4,
            "uint" => Self::Uint,
            "uint2" => Self::Uint2,
            "uint3" => Self::Uint3,
            "uint4" => Self::Uint4,
            "int" => Self::Int,
            "int2" => Self::Int2,
            "int3" => Self::Int3,
            "int4" => Self::Int4,
            _ => return None,
        })
    }

    pub fn byte_size(self) -> u32 {
        match self {
            Self::Char | Self::Uchar => 1,
            Self::Char2 | Self::Uchar2 => 2,
            Self::Char3 | Self::Uchar3 => 3,
            Self::Char4 | Self::Uchar4 => 4,
            Self::Short | Self::Ushort | Self::Half => 2,
            Self::Short2 | Self::Ushort2 | Self::Half2 => 4,
            Self::Short3 | Self::Ushort3 | Self::Half3 => 6,
            Self::Short4 | Self::Ushort4 | Self::Half4 => 8,
            Self::Float | Self::Uint | Self::Int => 4,
            Self::Float2 | Self::Uint2 | Self::Int2 => 8,
            Self::Float3 | Self::Uint3 | Self::Int3 => 12,
            Self::Float4 | Self::Uint4 | Self::Int4 => 16,
        }
    }

    pub fn air_type_name(self) -> &'static str {
        match self {
            Self::Char => "char",
            Self::Char2 => "char2",
            Self::Char3 => "char3",
            Self::Char4 => "char4",
            Self::Uchar => "uchar",
            Self::Uchar2 => "uchar2",
            Self::Uchar3 => "uchar3",
            Self::Uchar4 => "uchar4",
            Self::Short => "short",
            Self::Short2 => "short2",
            Self::Short3 => "short3",
            Self::Short4 => "short4",
            Self::Ushort => "ushort",
            Self::Ushort2 => "ushort2",
            Self::Ushort3 => "ushort3",
            Self::Ushort4 => "ushort4",
            Self::Half => "half",
            Self::Half2 => "half2",
            Self::Half3 => "half3",
            Self::Half4 => "half4",
            Self::Float => "float",
            Self::Float2 => "float2",
            Self::Float3 => "float3",
            Self::Float4 => "float4",
            Self::Uint => "uint",
            Self::Uint2 => "uint2",
            Self::Uint3 => "uint3",
            Self::Uint4 => "uint4",
            Self::Int => "int",
            Self::Int2 => "int2",
            Self::Int3 => "int3",
            Self::Int4 => "int4",
        }
    }

    /// The scalar type and lane count of the shader-visible attribute value.
    pub fn scalar_lanes(self) -> (ScalarType, u32) {
        use ScalarType as S;
        match self {
            Self::Char => (S::I8, 1),
            Self::Char2 => (S::I8, 2),
            Self::Char3 => (S::I8, 3),
            Self::Char4 => (S::I8, 4),
            Self::Uchar => (S::U8, 1),
            Self::Uchar2 => (S::U8, 2),
            Self::Uchar3 => (S::U8, 3),
            Self::Uchar4 => (S::U8, 4),
            Self::Short => (S::I16, 1),
            Self::Short2 => (S::I16, 2),
            Self::Short3 => (S::I16, 3),
            Self::Short4 => (S::I16, 4),
            Self::Ushort => (S::U16, 1),
            Self::Ushort2 => (S::U16, 2),
            Self::Ushort3 => (S::U16, 3),
            Self::Ushort4 => (S::U16, 4),
            Self::Half => (S::F16, 1),
            Self::Half2 => (S::F16, 2),
            Self::Half3 => (S::F16, 3),
            Self::Half4 => (S::F16, 4),
            Self::Float => (S::F32, 1),
            Self::Float2 => (S::F32, 2),
            Self::Float3 => (S::F32, 3),
            Self::Float4 => (S::F32, 4),
            Self::Uint => (S::U32, 1),
            Self::Uint2 => (S::U32, 2),
            Self::Uint3 => (S::U32, 3),
            Self::Uint4 => (S::U32, 4),
            Self::Int => (S::I32, 1),
            Self::Int2 => (S::I32, 2),
            Self::Int3 => (S::I32, 3),
            Self::Int4 => (S::I32, 4),
        }
    }

    /// Runtime-array stride used by the product's Vulkan StorageBuffer lowering.
    pub fn storage_buffer_stride(self) -> u32 {
        match self {
            Self::Char3 | Self::Uchar3 => 4,
            Self::Short3 | Self::Ushort3 | Self::Half3 => 8,
            Self::Float3 | Self::Uint3 | Self::Int3 => 16,
            _ => self.byte_size(),
        }
    }

    pub fn supports_tessellation_system_value(self) -> bool {
        matches!(self, Self::Short | Self::Ushort | Self::Int | Self::Uint)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConstant {
    pub index: u32,
    pub scalar_type: ScalarType,
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub lanes: u32,
    pub bytes_b64: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl ScalarType {
    pub fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    pub fn from_metal_abi_type_encoding(encoding: &str) -> Option<(Self, u32)> {
        let (lanes, scalar) = if let Some(vector) = encoding.strip_prefix("Dv") {
            let (lanes, scalar) = vector.split_once('_')?;
            (lanes.parse().ok()?, scalar)
        } else {
            (1, encoding)
        };
        if !(1..=4).contains(&lanes) {
            return None;
        }
        let scalar = match scalar {
            "b" => Self::Bool,
            "c" => Self::I8,
            "h" => Self::U8,
            "s" => Self::I16,
            "t" => Self::U16,
            "i" => Self::I32,
            "j" => Self::U32,
            "l" => Self::I64,
            "m" => Self::U64,
            "Dh" => Self::F16,
            "f" => Self::F32,
            "d" => Self::F64,
            _ => return None,
        };
        Some((scalar, lanes))
    }

    pub fn supports_metal_function_constant(self) -> bool {
        self != Self::F64
    }

    /// The `MTLDataType` of a function constant of this scalar with `lanes` lanes, `None` for a
    /// type Metal function constants do not have.
    pub fn metal_data_type(self, lanes: u32) -> Option<metal2vulkan::metal::DataType> {
        use metal2vulkan::metal::DataType as D;
        let row: [D; 4] = match self {
            Self::Bool => [D::Bool, D::Bool2, D::Bool3, D::Bool4],
            Self::U8 => [D::UChar, D::UChar2, D::UChar3, D::UChar4],
            Self::I8 => [D::Char, D::Char2, D::Char3, D::Char4],
            Self::U16 => [D::UShort, D::UShort2, D::UShort3, D::UShort4],
            Self::I16 => [D::Short, D::Short2, D::Short3, D::Short4],
            Self::F16 => [D::Half, D::Half2, D::Half3, D::Half4],
            Self::U32 => [D::UInt, D::UInt2, D::UInt3, D::UInt4],
            Self::I32 => [D::Int, D::Int2, D::Int3, D::Int4],
            Self::F32 => [D::Float, D::Float2, D::Float3, D::Float4],
            Self::U64 => [D::ULong, D::ULong2, D::ULong3, D::ULong4],
            Self::I64 => [D::Long, D::Long2, D::Long3, D::Long4],
            Self::F64 => return None,
        };
        let lane = usize::try_from(lanes).ok()?.checked_sub(1)?;
        row.get(lane).copied()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dispatch {
    pub grid: [u32; 3],
    pub threads_per_threadgroup: [u32; 3],
    /// The stage-in region both executors set for a case with kernel stage inputs
    /// (`setStageInRegion:`), in place of the grid's from the origin; valid only with kernel stage
    /// inputs. A thread past it on x fetches its last element.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_in_region: Option<indirect_commands::HostStageInRegion>,
}

impl Dispatch {
    /// The stage-in region the dispatch sets for a case with kernel stage inputs: its
    /// [`Dispatch::stage_in_region`], or the grid's from the origin.
    pub fn region(&self) -> indirect_commands::HostStageInRegion {
        self.stage_in_region
            .unwrap_or(indirect_commands::HostStageInRegion {
                origin: [0; 3],
                size: self.grid,
            })
    }

    /// One past the last element a kernel stage input fetches over the dispatch (stepping by
    /// thread position in the grid on x at rate 1): the region's x origin plus the threads on x
    /// it keeps apart, the grid's or the region's size where that is smaller (a thread past the
    /// region fetches its last element, and a size of 0 clamps nothing). `None` when that overflows
    /// 32 bits.
    pub fn stage_in_elements(&self) -> Option<u32> {
        let region = self.region();
        let kept = match region.size[0] {
            0 => self.grid[0],
            size => size.min(self.grid[0]),
        };
        region.origin[0].checked_add(kept)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Draw {
    pub primitive: Primitive,
    /// The first vertex of a draw without an index buffer; 0 for an indexed draw.
    pub vertex_start: u32,
    /// The vertices of a draw without an index buffer, or the indices of an indexed draw.
    pub vertex_count: u32,
    pub instance_count: u32,
    /// An indexed draw's index buffer: the draw is Metal's
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
    /// baseVertex:baseInstance:` of `vertex_count` indices from it, with the draw's
    /// [`Draw::base_vertex`] and [`Draw::base_instance`]. Absent for a draw of `vertex_count`
    /// vertices from `vertex_start`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_buffer: Option<IndexBufferLiteral>,
    /// An indexed draw's `baseVertex`, which Metal adds to each index it reads
    /// (`drawIndexedPrimitives:...:instanceCount:baseVertex:baseInstance:`); 0 without an index
    /// buffer, whose base vertex is `vertex_start`. Omitted when 0, so a case without it keeps
    /// its identity.
    #[serde(default, skip_serializing_if = "is_default")]
    pub base_vertex: i32,
    /// The draw's `baseInstance` (`drawPrimitives:...:instanceCount:baseInstance:` and the indexed
    /// form). Omitted when 0.
    #[serde(default, skip_serializing_if = "is_default")]
    pub base_instance: u32,
}

/// The index buffer of an indexed draw ([`Draw::index_buffer`]): `bytes_b64` holds its indices,
/// and the draw reads them from byte `offset` (Metal's `indexBufferOffset`, a multiple of the
/// index size).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexBufferLiteral {
    pub index_type: IndexType,
    pub offset: u32,
    pub bytes_b64: String,
}

/// `MTLIndexType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    Uint16,
    Uint32,
}

impl IndexType {
    /// Bytes of one index.
    pub const fn size(self) -> u32 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

impl IndexBufferLiteral {
    /// The index buffer's bytes, when `bytes_b64` decodes.
    pub fn bytes(&self) -> Option<Vec<u8>> {
        base64::engine::general_purpose::STANDARD
            .decode(&self.bytes_b64)
            .ok()
    }

    /// The indices an indexed draw of `count` indices reads from the buffer: those whose bytes
    /// lie wholly inside it, from `offset` on.
    pub fn indices_inside(&self, count: u32) -> Vec<u32> {
        let Some(bytes) = self.bytes() else {
            return Vec::new();
        };
        let size = self.index_type.size() as usize;
        bytes
            .get(self.offset as usize..)
            .unwrap_or_default()
            .chunks_exact(size)
            .take(count as usize)
            .map(|index| match self.index_type {
                IndexType::Uint16 => u32::from(u16::from_le_bytes([index[0], index[1]])),
                IndexType::Uint32 => u32::from_le_bytes([index[0], index[1], index[2], index[3]]),
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TessellationDraw {
    pub factors: Vec<TessellationFactors>,
    pub instance_count: u32,
    pub amplification_count: u32,
    #[serde(default)]
    pub control_points: Vec<AttributeInput>,
    #[serde(default)]
    pub patch_inputs: Vec<AttributeInput>,
    /// The pipeline's `tessellationPartitionMode`; absent is `integer`.
    #[serde(default, skip_serializing_if = "is_default")]
    pub partition_mode: TessellationPartition,
    /// The pipeline's `tessellationOutputWindingOrder`; absent is `counter_clockwise`.
    #[serde(default, skip_serializing_if = "is_default")]
    pub output_winding: TessellationWinding,
    /// The render encoder's `setCullMode:` for the draw; absent is `none`, Metal's default. The
    /// front-facing winding stays Metal's default, clockwise.
    #[serde(default, skip_serializing_if = "is_default")]
    pub cull_mode: CullMode,
}

/// `MTLTessellationPartitionMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TessellationPartition {
    Pow2,
    #[default]
    Integer,
    FractionalOdd,
    FractionalEven,
}

/// `MTLWinding`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TessellationWinding {
    Clockwise,
    #[default]
    CounterClockwise,
}

/// `MTLCullMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

/// Metal's most threads of an object or mesh threadgroup (`maxTotalThreadsPerObjectThreadgroup`,
/// `maxTotalThreadsPerMeshThreadgroup`, measured on the M3 Max and an M4): its validation stops a
/// draw of a larger threadgroup, so a case of one has no defined answer.
pub const MESH_THREADGROUP_THREADS: u64 = 1024;

/// An object or mesh case's draw of its mesh render pipeline (`MTLMeshRenderPipelineDescriptor`
/// and one `drawMeshThreadgroups:` or `drawMeshThreads:`).
///
/// The case's own entry is the stage its [`AuthoredCase::stage`] names; the pipeline's other
/// stages are either partners, functions of the case's own AIR module named here, generated
/// stages its [`observation`](Self::observation) or [`object_launch`](Self::object_launch)
/// implies, or absent. An object case's absent mesh stage is a stage nothing supplies
/// (`ToolingRequirement::MeshCompanion`) unless the case states a [`MeshObservation::Launch`], as
/// is the absent object stage of a mesh function that reads a payload unless the case states the
/// generated object function's [`ObjectLaunch`]. A rasterizing draw has a fragment function: Metal's
/// descriptor validation stops a rasterizing mesh pipeline without one ("fragmentFunction must not
/// be nil", an abort on the M3 Max under macOS 27.0.1), so the schema refuses such a draw before
/// any executor runs. The draw names its fragment partner, or states the
/// [`observation`](Self::observation) whose generated stages write render target 0
/// (`companion::MeshStages`). A partner binds no
/// resource: every resource the case authors is its own entry's, bound through the stage's setters
/// (`setObjectBuffer:`, `setMeshBuffer:`, ...). The payload length is the pipeline's default
/// (`payloadMemoryLength` 0, under which the M3 Max's mesh function reads the object function's
/// whole payload): the case does not author it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshDraw {
    /// The object function: a partner of a mesh case, absent for an object case (its own entry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// The mesh function: a partner of an object case, absent for a mesh case (its own entry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<String>,
    /// The fragment function, a partner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fragment: Option<String>,
    /// The draw's grid and its kind.
    pub grid: MeshGrid,
    /// The draw's `threadsPerObjectThreadgroup`.
    pub threads_per_object_threadgroup: [u32; 3],
    /// The draw's `threadsPerMeshThreadgroup`.
    pub threads_per_mesh_threadgroup: [u32; 3],
    /// The pipeline's `rasterizationEnabled`; absent is `true`, which needs a
    /// [`fragment`](Self::fragment) or an [`observation`](Self::observation). Without
    /// rasterization the pipeline has no fragment stage and no attachment, and the case observes
    /// its entry's buffers.
    #[serde(default = "rasterizes", skip_serializing_if = "is_rasterizing")]
    pub rasterization: bool,
    /// What generated stages observe into render target 0, in place of a fragment partner (and,
    /// for an object case, of a mesh partner): only a rasterizing case that names neither states
    /// one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation: Option<MeshObservation>,
    /// What a generated object function launches in the object partner's place: only a mesh case
    /// that names no object function states one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_launch: Option<ObjectLaunch>,
}

/// Metal's largest object payload (16 KB an object threadgroup, its feature set tables' limit),
/// the most a generated object function writes.
pub const MESH_PAYLOAD_BYTES: usize = 16 * 1024;

/// What a mesh case's generated object function does in its object partner's place
/// ([`MeshDraw::object_launch`], `companion::PayloadObject`): thread 0 of each object threadgroup
/// writes the payload's bytes and launches the mesh grid, so each object threadgroup launches the
/// same mesh threadgroups over the same payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectLaunch {
    /// The payload's bytes: whole 4-byte words, at least as many as the mesh function reads
    /// (its `air.arg_type_size`) and at most [`MESH_PAYLOAD_BYTES`].
    pub payload_b64: String,
    /// The mesh threadgroups each object threadgroup launches
    /// (`mesh_grid_properties::set_threadgroups_per_grid`), at most
    /// [`MESH_GRID_THREADGROUPS`] in all.
    pub mesh_threadgroups: [u32; 3],
}

/// Metal's most mesh threadgroups an object threadgroup launches (its feature set tables' limit
/// on a mesh grid, which `max_total_threadgroups_per_mesh_grid` may lower).
pub const MESH_GRID_THREADGROUPS: u64 = 1024;

impl ObjectLaunch {
    /// The payload's bytes, when `payload_b64` decodes.
    pub fn payload(&self) -> Option<Vec<u8>> {
        base64::engine::general_purpose::STANDARD
            .decode(&self.payload_b64)
            .ok()
    }

    fn validate(&self, errors: &mut Vec<String>) {
        match self.payload() {
            None => errors.push("mesh_draw.object_launch.payload_b64 does not decode".into()),
            Some(bytes) if bytes.is_empty() || bytes.len() % 4 != 0 => errors.push(format!(
                "mesh_draw.object_launch.payload_b64 holds {} bytes: a generated object function \
                 writes whole 4-byte words, at least one",
                bytes.len()
            )),
            Some(bytes) if bytes.len() > MESH_PAYLOAD_BYTES => errors.push(format!(
                "mesh_draw.object_launch.payload_b64 holds {} bytes, past Metal's \
                 {MESH_PAYLOAD_BYTES}-byte payload",
                bytes.len()
            )),
            Some(_) => {}
        }
        let total = self
            .mesh_threadgroups
            .iter()
            .map(|axis| u64::from(*axis))
            .product::<u64>();
        if total > MESH_GRID_THREADGROUPS {
            errors.push(format!(
                "mesh_draw.object_launch.mesh_threadgroups {:?} holds {total} threadgroups, past \
                 Metal's {MESH_GRID_THREADGROUPS} a mesh grid",
                self.mesh_threadgroups
            ));
        }
    }
}

/// What a case's generated stages write to render target 0 ([`MeshDraw::observation`]): an output
/// of a mesh case's own mesh function, which a generated observer fragment reads as a vertex
/// case's observer reads a vertex output ([`VertexObservation`]), or what an object case's own
/// object function launches ([`Launch`](Self::Launch)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeshObservation {
    /// An object case's launch: the grid of mesh threadgroups its object function sets and the
    /// payload it writes, drawn by a generated mesh function in its mesh partner's place
    /// (`companion::LaunchObserver`) through a generated observer fragment. Each mesh threadgroup
    /// of the launched grid draws row `x + X * (y + Y * z)` of an `RGBA32Uint` render target 0
    /// (its position `(x, y, z)` in the grid of `(X, Y, Z)` threadgroups): pixel 0 holds
    /// `(X, Y, Z, row)`, and pixel `1 + k` the payload's words `4k` to `4k + 3`, read at the
    /// payload's declared size (`air.arg_type_size`, whole words), with a word past it 0. Every
    /// other pixel keeps the target's initial value, so a row past the grid or a pixel past the
    /// payload shows any write that lands there. The draw launches one object threadgroup: a mesh
    /// threadgroup cannot tell which object threadgroup launched it, so two would draw the same
    /// rows.
    Launch,
    /// `[[position]]`, as Metal's rasterizer delivers it.
    Position,
    /// The vertex struct's user data field `index`, counting its user data fields in declaration
    /// order (its builtins aside), interpolated as Metal's default for the field's type.
    VertexData { index: u32 },
    /// The primitive struct's user data field `index`, counting its user data fields in
    /// declaration order: one value over each primitive, read flat.
    PrimitiveData { index: u32 },
}

fn rasterizes() -> bool {
    true
}

fn is_rasterizing(rasterization: &bool) -> bool {
    *rasterization
}

/// A mesh draw's grid: of the first stage's threadgroups (`drawMeshThreadgroups:`), or of its
/// threads (`drawMeshThreads:`). The first stage is the object function where the pipeline has
/// one, else the mesh function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshGrid {
    Threadgroups([u32; 3]),
    Threads([u32; 3]),
}

impl MeshGrid {
    /// The grid's three extents, whichever kind it is.
    pub fn extents(self) -> [u32; 3] {
        match self {
            MeshGrid::Threadgroups(extents) | MeshGrid::Threads(extents) => extents,
        }
    }
}

impl MeshDraw {
    /// The function the pipeline runs at `stage` for a case of `case_stage` whose entry is
    /// `entry`: the entry at its own stage, the partner this draw names at another, `None` for a
    /// stage the draw leaves absent (or for a kernel, vertex or fragment `stage` other than the
    /// fragment).
    pub fn function<'a>(
        &'a self,
        case_stage: Stage,
        entry: &'a str,
        stage: Stage,
    ) -> Option<&'a str> {
        if stage == case_stage {
            return Some(entry);
        }
        self.partner(stage)
    }

    /// The partner this draw names at `stage`, if any.
    pub fn partner(&self, stage: Stage) -> Option<&str> {
        match stage {
            Stage::Object => self.object.as_deref(),
            Stage::Mesh => self.mesh.as_deref(),
            Stage::Fragment => self.fragment.as_deref(),
            Stage::Kernel | Stage::Vertex => None,
        }
    }

    /// Whether the draw takes generated stages (`companion::MeshStages`): it states an observation
    /// or an object launch.
    pub fn generates(&self) -> bool {
        self.observation.is_some() || self.object_launch.is_some()
    }

    /// The partners this draw names, with their stages, in pipeline order.
    pub fn partners(&self) -> impl Iterator<Item = (Stage, &str)> {
        [
            (Stage::Object, self.object.as_deref()),
            (Stage::Mesh, self.mesh.as_deref()),
            (Stage::Fragment, self.fragment.as_deref()),
        ]
        .into_iter()
        .filter_map(|(stage, name)| name.map(|name| (stage, name)))
    }

    /// The literal checks of the draw of a case of `stage` whose entry is `entry`.
    fn validate(&self, stage: Stage, entry: &str, errors: &mut Vec<String>) {
        match stage {
            Stage::Object if self.object.is_some() => errors.push(
                "mesh_draw.object of an object case: the case's entry is its object function"
                    .into(),
            ),
            Stage::Mesh if self.mesh.is_some() => errors.push(
                "mesh_draw.mesh of a mesh case: the case's entry is its mesh function".into(),
            ),
            _ => {}
        }
        for (partner_stage, name) in self.partners() {
            if name.trim().is_empty() {
                errors.push(format!(
                    "mesh_draw.{} must not be empty",
                    partner_stage.entry_list()
                ));
            } else if name == entry {
                errors.push(format!(
                    "mesh_draw.{} names the case's own entry {name:?}",
                    partner_stage.entry_list()
                ));
            }
        }
        match (self.rasterization, &self.fragment, self.observation) {
            (false, Some(_), _) => errors.push(
                "mesh_draw.fragment without rasterization: a pipeline that does not rasterize \
                 has no fragment stage"
                    .into(),
            ),
            (false, None, Some(_)) => errors.push(
                "mesh_draw.observation without rasterization: the observer is a fragment \
                 function, and a pipeline that does not rasterize has no fragment stage"
                    .into(),
            ),
            (true, Some(_), Some(_)) => errors.push(
                "mesh_draw names a fragment partner and an observation: the observation's \
                 generated fragment function takes the fragment partner's place"
                    .into(),
            ),
            (true, None, None) => errors.push(
                "mesh_draw rasterizes and names no fragment function: Metal's descriptor \
                 validation stops a rasterizing mesh pipeline without one (\"fragmentFunction \
                 must not be nil\"); name the fragment partner, state an observation, or set \
                 rasterization false"
                    .into(),
            ),
            _ => {}
        }
        match (stage, self.observation) {
            (Stage::Object, Some(MeshObservation::Launch)) => {
                if self.mesh.is_some() {
                    errors.push(
                        "mesh_draw names a mesh partner and a launch observation: the \
                         observation's generated mesh function takes the mesh partner's place"
                            .into(),
                    );
                }
                let launched = match self.grid {
                    MeshGrid::Threadgroups(groups) => groups,
                    MeshGrid::Threads(threads) => {
                        let mut groups = [0; 3];
                        for (axis, groups) in groups.iter_mut().enumerate() {
                            let per = self.threads_per_object_threadgroup[axis].max(1);
                            *groups = threads[axis].div_ceil(per);
                        }
                        groups
                    }
                };
                if launched != [1, 1, 1] {
                    errors.push(format!(
                        "mesh_draw's launch observation needs one object threadgroup, and its \
                         grid launches {launched:?}: a mesh threadgroup cannot tell which object \
                         threadgroup launched it, so two would draw the same rows"
                    ));
                }
            }
            (Stage::Object, Some(_)) => errors.push(
                "mesh_draw.observation of an object case other than its launch: the observer \
                 reads a mesh function's outputs, and the case's own function is its object \
                 function"
                    .into(),
            ),
            (_, Some(MeshObservation::Launch)) => errors.push(
                "mesh_draw's launch observation of a mesh case: it observes an object case's own \
                 object function"
                    .into(),
            ),
            _ => {}
        }
        if let Some(launch) = &self.object_launch {
            match stage {
                Stage::Mesh if self.object.is_some() => errors.push(
                    "mesh_draw names an object partner and an object launch: the launch's \
                     generated object function takes the object partner's place"
                        .into(),
                ),
                Stage::Mesh => launch.validate(errors),
                _ => errors.push(
                    "mesh_draw.object_launch of an object case: the case's entry is its object \
                     function"
                        .into(),
                ),
            }
        }
        validate_dimensions("mesh_draw.grid", &self.grid.extents(), errors);
        for (field, threads) in [
            (
                "mesh_draw.threads_per_object_threadgroup",
                &self.threads_per_object_threadgroup,
            ),
            (
                "mesh_draw.threads_per_mesh_threadgroup",
                &self.threads_per_mesh_threadgroup,
            ),
        ] {
            validate_dimensions(field, threads, errors);
            let total = threads.iter().map(|axis| u64::from(*axis)).product::<u64>();
            if total > MESH_THREADGROUP_THREADS {
                errors.push(format!(
                    "{field} {threads:?} holds {total} threads: Metal's validation stops a \
                     threadgroup of more than {MESH_THREADGROUP_THREADS}"
                ));
            }
        }
    }
}

/// Any binary16 bit patterns: Metal culls a patch one of whose edge factors is zero, negative, or
/// NaN, and clamps every other factor to its partition's range at most `maxTessellationFactor`, so
/// every pattern is a defined draw.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TessellationFactors {
    /// IEEE-754 binary16 bit patterns in Metal/Vulkan outer-factor order.
    pub edge_f16: Vec<u16>,
    /// IEEE-754 binary16 bit patterns in Metal/Vulkan inner-factor order.
    pub inside_f16: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Primitive {
    Point,
    Line,
    LineStrip,
    Triangle,
    TriangleStrip,
}

impl Primitive {
    /// The topology class of the primitive, which a render pipeline is compiled for.
    pub fn topology_class(self) -> metal2vulkan::metal::PrimitiveTopologyClass {
        use metal2vulkan::metal::PrimitiveTopologyClass as Class;
        match self {
            Self::Point => Class::Point,
            Self::Line | Self::LineStrip => Class::Line,
            Self::Triangle | Self::TriangleStrip => Class::Triangle,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputSelection {
    /// Execute the entry and compare the exact empty byte sequence. The shared checker accepts this
    /// only when reflection exposes no writable observable output.
    None,
    Buffer {
        binding: u32,
        offset: u64,
        length: u64,
    },
    ArgumentBufferBuffer {
        buffer_binding: u32,
        field_offset: u32,
        offset: u64,
        length: u64,
    },
    DeviceBufferArrayElement {
        binding: u32,
        element: u32,
        offset: u64,
        length: u64,
    },
    Texture {
        binding: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    TextureArrayElement {
        binding: u32,
        element: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    ArgumentBufferTexture {
        buffer_binding: u32,
        field_offset: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    RenderTarget {
        index: u32,
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    Depth {
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    Stencil {
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    FragmentImageblock {
        semantic: String,
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
}

/// Vertex-stage value routed through the generated fragment observer into render target zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VertexObservation {
    Position,
    Varying { location: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Comparison {
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionSafety {
    /// The AIR CFG is structurally acyclic.
    LoopFree,
    /// Authored inputs and function constants give every reachable loop a finite bound.
    AuthoredBounded,
}

#[derive(Serialize)]
struct SemanticCase<'a> {
    air_sha256: &'a str,
    entry: &'a str,
    stage: Stage,
    buffers: Vec<&'a BufferResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    argument_buffer_buffers: Vec<&'a ArgumentBufferBufferResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    device_buffer_arrays: Vec<&'a DeviceBufferArrayResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    command_buffers: Vec<&'a indirect_commands::CommandBufferResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pipeline_states: Vec<&'a indirect_commands::PipelineResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    threadgroup_memory: Vec<&'a ThreadgroupMemoryResource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    imageblock: &'a Option<ImageblockResource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fragment_imageblock: &'a Option<FragmentImageblockResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    acceleration_structures: Vec<&'a AccelerationStructureResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    visible_function_references: Vec<&'a LinkedFunctionResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    visible_function_tables: Vec<&'a FunctionTableResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    intersection_function_tables: Vec<&'a IntersectionFunctionTableResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    argument_buffer_intersection_function_tables:
        Vec<&'a ArgumentBufferIntersectionFunctionTableResource>,
    textures: Vec<&'a TextureResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    texture_arrays: Vec<&'a TextureArrayResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    argument_buffer_textures: Vec<&'a ArgumentBufferTextureResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    argument_buffer_samplers: Vec<&'a ArgumentBufferSamplerResource>,
    samplers: Vec<&'a SamplerResource>,
    render_targets: Vec<&'a RenderTargetResource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depth_stencil: &'a Option<DepthStencilResource>,
    vertex_inputs: Vec<&'a VertexInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vertex_observation: &'a Option<VertexObservation>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    varying_values: Vec<&'a VaryingValue>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    kernel_stage_inputs: Vec<&'a KernelStageInput>,
    function_constants: Vec<&'a FunctionConstant>,
    dispatch: &'a Option<Dispatch>,
    draw: &'a Option<Draw>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tessellation: &'a Option<TessellationDraw>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh_draw: &'a Option<MeshDraw>,
    output: &'a OutputSelection,
    compare: &'a Comparison,
    execution_safety: ExecutionSafety,
    #[serde(skip_serializing_if = "Option::is_none")]
    fuel: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    out_of_bounds: &'a Option<out_of_bounds::OutOfBounds>,
    #[serde(skip_serializing_if = "Option::is_none")]
    render_pass_rate_map: &'a Option<RateMapLiteral>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raster_sample_count: Option<u32>,
}

impl AuthoredCase {
    /// The render pass's sample count: the authored `raster_sample_count`, or one sample without
    /// it.
    pub fn raster_samples(&self) -> u32 {
        self.raster_sample_count.unwrap_or(1)
    }

    pub fn is_rasterization_disabled_vertex(&self) -> bool {
        self.stage == Stage::Vertex && self.vertex_observation.is_none()
    }

    /// The byte length of each buffer the case states bytes for, by its Metal index.
    pub fn buffer_lengths(&self) -> std::collections::BTreeMap<u32, usize> {
        self.buffers
            .iter()
            .filter_map(|buffer| {
                let bytes = initial_bytes(
                    buffer.bytes_b64.as_deref(),
                    buffer.initial_bytes_b64.as_deref(),
                )?;
                Some((buffer.binding, bytes.len()))
            })
            .collect()
    }

    /// The vertex elements the case's draw fetches from each per-vertex buffer: `vertex_start +
    /// vertex_count` for a draw without an index buffer, and one past the largest index an
    /// indexed draw reads inside its index buffer (an index read past it is the out-of-bounds
    /// target's, whose case fetches no vertex attribute). `None` without a draw.
    pub fn vertex_elements(&self) -> Option<u32> {
        let draw = self.draw.as_ref()?;
        match &draw.index_buffer {
            None => draw.vertex_start.checked_add(draw.vertex_count),
            Some(index_buffer) => Some(
                index_buffer
                    .indices_inside(draw.vertex_count)
                    .into_iter()
                    .max()
                    .map_or(0, |largest| largest.saturating_add(1)),
            ),
        }
    }

    /// The attribute locations the case declares out-of-bounds targets: their buffers hold fewer
    /// elements than the draw or dispatch fetches, by design.
    fn out_of_bounds_attribute_locations(&self) -> HashSet<u32> {
        self.out_of_bounds
            .iter()
            .flat_map(|declaration| &declaration.targets)
            .filter_map(|target| match target.resource {
                out_of_bounds::OutOfBoundsResource::AttributeBuffer { location } => Some(location),
                _ => None,
            })
            .collect()
    }

    /// An indexed draw's literal rules: a vertex case's draw of indices from no first vertex, over
    /// a buffer of whole indices, from an offset that is a multiple of the index size inside the
    /// buffer. Unless the case declares the index buffer an out-of-bounds target, every index the
    /// draw reads lies inside the buffer: Metal's API validation rejects a draw past it and its
    /// documentation states no result.
    fn validate_index_buffer(
        &self,
        draw: &Draw,
        index_buffer: &IndexBufferLiteral,
        errors: &mut Vec<String>,
    ) {
        if self.stage != Stage::Vertex {
            errors.push("an indexed draw is authorable for vertex cases only".into());
        }
        if draw.vertex_start != 0 {
            errors.push(
                "an indexed draw starts at no vertex: draw.vertex_start must be 0 (its \
                 vertex_count is the index count)"
                    .into(),
            );
        }
        let size = index_buffer.index_type.size();
        let Some(bytes) = index_buffer.bytes() else {
            errors.push("draw.index_buffer.bytes_b64 is not base64".into());
            return;
        };
        if bytes.is_empty() || bytes.len() % size as usize != 0 {
            errors.push(format!(
                "draw.index_buffer holds {} bytes, not a nonzero number of {size}-byte indices",
                bytes.len()
            ));
        }
        if !index_buffer.offset.is_multiple_of(size) || index_buffer.offset as usize >= bytes.len()
        {
            errors.push(format!(
                "draw.index_buffer.offset {} must be a multiple of the index size {size} inside \
                 the buffer's {} bytes",
                index_buffer.offset,
                bytes.len()
            ));
        }
        let end = u64::from(index_buffer.offset) + u64::from(draw.vertex_count) * u64::from(size);
        let target = self.out_of_bounds.iter().any(|declaration| {
            declaration
                .targets
                .iter()
                .any(|target| target.resource == out_of_bounds::OutOfBoundsResource::IndexBuffer)
        });
        if !target && end > bytes.len() as u64 {
            errors.push(format!(
                "the indexed draw reads indices to byte {end} of its {}-byte index buffer, which \
                 Metal leaves undefined: declare the index buffer an out-of-bounds target",
                bytes.len()
            ));
        }
    }

    fn semantic(&self) -> SemanticCase<'_> {
        SemanticCase {
            air_sha256: &self.air_sha256,
            entry: &self.entry,
            stage: self.stage,
            buffers: sorted_refs(&self.buffers, |resource| resource.binding),
            argument_buffer_buffers: sorted_refs(&self.argument_buffer_buffers, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            device_buffer_arrays: sorted_refs(&self.device_buffer_arrays, |resource| {
                resource.binding
            }),
            command_buffers: sorted_refs(&self.command_buffers, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            pipeline_states: sorted_refs(&self.pipeline_states, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            threadgroup_memory: sorted_refs(&self.threadgroup_memory, |resource| resource.binding),
            imageblock: &self.imageblock,
            fragment_imageblock: &self.fragment_imageblock,
            acceleration_structures: sorted_refs(&self.acceleration_structures, |resource| {
                resource.binding
            }),
            visible_function_references: sorted_refs(
                &self.visible_function_references,
                |resource| (resource.function.clone(), resource.module_sha256.clone()),
            ),
            visible_function_tables: sorted_refs(&self.visible_function_tables, |resource| {
                resource.binding
            }),
            intersection_function_tables: sorted_refs(
                &self.intersection_function_tables,
                |resource| resource.binding,
            ),
            argument_buffer_intersection_function_tables: sorted_refs(
                &self.argument_buffer_intersection_function_tables,
                |resource| (resource.buffer_binding, resource.field_offset),
            ),
            textures: sorted_refs(&self.textures, |resource| resource.binding),
            texture_arrays: sorted_refs(&self.texture_arrays, |resource| resource.binding),
            argument_buffer_textures: sorted_refs(&self.argument_buffer_textures, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            argument_buffer_samplers: sorted_refs(&self.argument_buffer_samplers, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            samplers: sorted_refs(&self.samplers, |resource| resource.binding),
            render_targets: sorted_refs(&self.render_targets, |resource| resource.index),
            depth_stencil: &self.depth_stencil,
            vertex_inputs: sorted_refs(&self.vertex_inputs, |resource| resource.location),
            vertex_observation: &self.vertex_observation,
            varying_values: sorted_refs(&self.varying_values, |value| value.key.clone()),
            kernel_stage_inputs: sorted_refs(&self.kernel_stage_inputs, |resource| {
                resource.location
            }),
            function_constants: sorted_refs(&self.function_constants, |constant| constant.index),
            dispatch: &self.dispatch,
            draw: &self.draw,
            tessellation: &self.tessellation,
            mesh_draw: &self.mesh_draw,
            output: &self.output,
            compare: &self.compare,
            execution_safety: self.execution_safety,
            fuel: self.fuel,
            out_of_bounds: &self.out_of_bounds,
            render_pass_rate_map: &self.render_pass_rate_map,
            raster_sample_count: self.raster_sample_count,
        }
    }

    pub fn computed_case_id(&self) -> Result<String, String> {
        let canonical = to_sorted_json_string(self.semantic())
            .map_err(|error| format!("canonicalize case: {error}"))?;
        Ok(sha256_bytes(canonical.as_bytes()))
    }

    pub fn computed_input_sha256(&self) -> Result<String, String> {
        #[derive(Serialize)]
        struct Inputs<'a> {
            case_id: &'a str,
            buffers: Vec<&'a BufferResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            argument_buffer_buffers: Vec<&'a ArgumentBufferBufferResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            device_buffer_arrays: Vec<&'a DeviceBufferArrayResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            command_buffers: Vec<&'a indirect_commands::CommandBufferResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            pipeline_states: Vec<&'a indirect_commands::PipelineResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            threadgroup_memory: Vec<&'a ThreadgroupMemoryResource>,
            #[serde(skip_serializing_if = "Option::is_none")]
            imageblock: &'a Option<ImageblockResource>,
            #[serde(skip_serializing_if = "Option::is_none")]
            fragment_imageblock: &'a Option<FragmentImageblockResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            acceleration_structures: Vec<&'a AccelerationStructureResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            visible_function_references: Vec<&'a LinkedFunctionResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            visible_function_tables: Vec<&'a FunctionTableResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            intersection_function_tables: Vec<&'a IntersectionFunctionTableResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            argument_buffer_intersection_function_tables:
                Vec<&'a ArgumentBufferIntersectionFunctionTableResource>,
            textures: Vec<&'a TextureResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            texture_arrays: Vec<&'a TextureArrayResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            argument_buffer_textures: Vec<&'a ArgumentBufferTextureResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            argument_buffer_samplers: Vec<&'a ArgumentBufferSamplerResource>,
            samplers: Vec<&'a SamplerResource>,
            render_targets: Vec<&'a RenderTargetResource>,
            #[serde(skip_serializing_if = "Option::is_none")]
            depth_stencil: &'a Option<DepthStencilResource>,
            vertex_inputs: Vec<&'a VertexInput>,
            #[serde(skip_serializing_if = "Option::is_none")]
            vertex_observation: &'a Option<VertexObservation>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            varying_values: Vec<&'a VaryingValue>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            kernel_stage_inputs: Vec<&'a KernelStageInput>,
            function_constants: Vec<&'a FunctionConstant>,
            dispatch: &'a Option<Dispatch>,
            draw: &'a Option<Draw>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tessellation: &'a Option<TessellationDraw>,
            #[serde(skip_serializing_if = "Option::is_none")]
            mesh_draw: &'a Option<MeshDraw>,
            output: &'a OutputSelection,
            #[serde(skip_serializing_if = "Option::is_none")]
            render_pass_rate_map: &'a Option<RateMapLiteral>,
            #[serde(skip_serializing_if = "Option::is_none")]
            raster_sample_count: Option<u32>,
        }
        let canonical = to_sorted_json_string(Inputs {
            case_id: &self.case_id,
            buffers: sorted_refs(&self.buffers, |resource| resource.binding),
            argument_buffer_buffers: sorted_refs(&self.argument_buffer_buffers, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            device_buffer_arrays: sorted_refs(&self.device_buffer_arrays, |resource| {
                resource.binding
            }),
            command_buffers: sorted_refs(&self.command_buffers, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            pipeline_states: sorted_refs(&self.pipeline_states, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            threadgroup_memory: sorted_refs(&self.threadgroup_memory, |resource| resource.binding),
            imageblock: &self.imageblock,
            fragment_imageblock: &self.fragment_imageblock,
            acceleration_structures: sorted_refs(&self.acceleration_structures, |resource| {
                resource.binding
            }),
            visible_function_references: sorted_refs(
                &self.visible_function_references,
                |resource| (resource.function.clone(), resource.module_sha256.clone()),
            ),
            visible_function_tables: sorted_refs(&self.visible_function_tables, |resource| {
                resource.binding
            }),
            intersection_function_tables: sorted_refs(
                &self.intersection_function_tables,
                |resource| resource.binding,
            ),
            argument_buffer_intersection_function_tables: sorted_refs(
                &self.argument_buffer_intersection_function_tables,
                |resource| (resource.buffer_binding, resource.field_offset),
            ),
            textures: sorted_refs(&self.textures, |resource| resource.binding),
            texture_arrays: sorted_refs(&self.texture_arrays, |resource| resource.binding),
            argument_buffer_textures: sorted_refs(&self.argument_buffer_textures, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            argument_buffer_samplers: sorted_refs(&self.argument_buffer_samplers, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            samplers: sorted_refs(&self.samplers, |resource| resource.binding),
            render_targets: sorted_refs(&self.render_targets, |resource| resource.index),
            depth_stencil: &self.depth_stencil,
            vertex_inputs: sorted_refs(&self.vertex_inputs, |resource| resource.location),
            vertex_observation: &self.vertex_observation,
            varying_values: sorted_refs(&self.varying_values, |value| value.key.clone()),
            kernel_stage_inputs: sorted_refs(&self.kernel_stage_inputs, |resource| {
                resource.location
            }),
            function_constants: sorted_refs(&self.function_constants, |constant| constant.index),
            dispatch: &self.dispatch,
            draw: &self.draw,
            tessellation: &self.tessellation,
            mesh_draw: &self.mesh_draw,
            output: &self.output,
            render_pass_rate_map: &self.render_pass_rate_map,
            raster_sample_count: self.raster_sample_count,
        })
        .map_err(|error| format!("canonicalize inputs: {error}"))?;
        Ok(sha256_bytes(canonical.as_bytes()))
    }

    /// The bound each driver creation of the case's own pipeline is held to: its
    /// [`creation_bound`](Self::creation_bound), or the shared ceiling
    /// (`budgets::PIPELINE_CREATION_CEILING`) where it declares none.
    pub fn pipeline_creation_bound(&self) -> std::time::Duration {
        self.creation_bound
            .as_ref()
            .map_or(crate::budgets::PIPELINE_CREATION_CEILING, |bound| {
                std::time::Duration::from_millis(bound.ms)
            })
    }

    pub fn validate_literal_resources(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        validate_hash("air_sha256", &self.air_sha256, &mut errors);
        validate_hash("case_id", &self.case_id, &mut errors);
        if self.name.trim().is_empty() {
            errors.push("name must not be empty".into());
        }
        if self.entry.trim().is_empty() {
            errors.push("entry must not be empty".into());
        }
        if self.execution_safety == ExecutionSafety::AuthoredBounded
            && self
                .rationale
                .as_deref()
                .is_none_or(|rationale| rationale.trim().is_empty())
        {
            errors.push(
                "authored_bounded execution_safety requires a rationale identifying the finite loop bound"
                    .into(),
            );
        }
        if let Some(bound) = &self.creation_bound {
            if self.stage != Stage::Kernel {
                errors.push("creation_bound is read by the compute executors alone".into());
            }
            if u128::from(bound.ms) <= crate::budgets::PIPELINE_CREATION_CEILING.as_millis() {
                errors.push(format!(
                    "creation_bound {} ms is not above the shared ceiling of {} ms",
                    bound.ms,
                    crate::budgets::PIPELINE_CREATION_CEILING.as_millis()
                ));
            }
            if bound.measured.trim().is_empty() {
                errors.push(
                    "creation_bound requires what it was measured on and the measurements".into(),
                );
            }
        }
        if [
            self.dispatch.is_some(),
            self.draw.is_some(),
            self.tessellation.is_some(),
            self.mesh_draw.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count()
            != 1
        {
            errors.push(
                "exactly one of dispatch, draw, tessellation, or mesh_draw is required".into(),
            );
        }
        match self.stage {
            Stage::Kernel if self.dispatch.is_none() => {
                errors.push("kernel case requires dispatch".into())
            }
            Stage::Vertex if self.draw.is_none() && self.tessellation.is_none() => {
                errors.push("vertex case requires draw or tessellation".into())
            }
            Stage::Fragment if self.draw.is_none() => {
                errors.push("fragment case requires draw".into())
            }
            Stage::Object | Stage::Mesh if self.mesh_draw.is_none() => {
                errors.push("object or mesh case requires mesh_draw".into())
            }
            _ => {}
        }
        if let Some(mesh_draw) = &self.mesh_draw {
            self.validate_mesh_draw(mesh_draw, &mut errors);
        }
        if let Some(dispatch) = &self.dispatch {
            validate_dimensions("dispatch.grid", &dispatch.grid, &mut errors);
            validate_dimensions(
                "dispatch.threads_per_threadgroup",
                &dispatch.threads_per_threadgroup,
                &mut errors,
            );
            if let Some(imageblock) = &self.imageblock {
                validate_dimensions("imageblock.dimensions", &imageblock.dimensions, &mut errors);
                // The tile is the threadgroup's, but a thread may own more than one cell of it: an
                // entry that stages a `k`x`k` output block per thread has a tile `k` times the
                // threadgroup in each axis, and the AIR says so (`imageblock_cell_scale`). The
                // factor has to be the same in both axes and a whole number, because a cell is
                // owned by exactly one thread either way.
                let threads = [
                    dispatch.threads_per_threadgroup[0],
                    dispatch.threads_per_threadgroup[1],
                ];
                let scales = [
                    imageblock.dimensions[0].checked_div(threads[0]),
                    imageblock.dimensions[1].checked_div(threads[1]),
                ];
                let exact = imageblock.dimensions[0] % threads[0].max(1) == 0
                    && imageblock.dimensions[1] % threads[1].max(1) == 0;
                if !exact || scales[0] != scales[1] || scales[0] == Some(0) {
                    errors.push(format!(
                        "imageblock dimensions {:?} must be the same whole multiple of the \
                         threadgroup x/y dimensions {:?} in both axes",
                        imageblock.dimensions,
                        &dispatch.threads_per_threadgroup[..2]
                    ));
                }
            }
        }
        if self.imageblock.is_some() && self.stage != Stage::Kernel {
            errors.push("imageblock is valid only for kernel cases".into());
        }
        if let Some(imageblock) = &self.fragment_imageblock {
            if self.stage != Stage::Fragment {
                errors.push("fragment_imageblock is valid only for fragment cases".into());
            }
            validate_dimensions(
                "fragment_imageblock.dimensions",
                &imageblock.dimensions,
                &mut errors,
            );
            let mut semantics = HashSet::new();
            for member in &imageblock.members {
                if !semantics.insert(&member.semantic) {
                    errors.push(format!(
                        "duplicate fragment imageblock member semantic {}",
                        member.semantic
                    ));
                }
            }
            for member in &imageblock.members {
                let expected = checked_extent_bytes(
                    "fragment imageblock member",
                    &imageblock.dimensions,
                    member.format.byte_size(),
                    &mut errors,
                );
                validate_role_bytes_with_len(
                    &format!("fragment imageblock member {}", member.semantic),
                    member.role,
                    member.bytes_b64.as_deref(),
                    member.initial_bytes_b64.as_deref(),
                    expected,
                    &mut errors,
                );
            }
        }
        if let Some(draw) = &self.draw {
            if draw.vertex_count == 0 {
                errors.push("draw.vertex_count must be nonzero".into());
            }
            if draw.instance_count == 0 {
                errors.push("draw.instance_count must be nonzero".into());
            }
            if let Some(index_buffer) = &draw.index_buffer {
                self.validate_index_buffer(draw, index_buffer, &mut errors);
            }
            // A draw's bases are a vertex case's: the fragment executors draw the generated
            // passthrough vertex with none. The fetch extents of vertex inputs do not model them.
            if draw.base_vertex != 0 || draw.base_instance != 0 {
                if self.stage != Stage::Vertex {
                    errors
                        .push("draw.base_vertex and draw.base_instance are a vertex case's".into());
                }
                if !self.vertex_inputs.is_empty() {
                    errors.push(
                        "draw.base_vertex and draw.base_instance are authorable without vertex_inputs"
                            .into(),
                    );
                }
            }
            if draw.base_vertex != 0 && draw.index_buffer.is_none() {
                errors.push(
                    "draw.base_vertex is an indexed draw's: a draw without an index buffer starts at \
                     draw.vertex_start"
                        .into(),
                );
            }
        }
        if let Some(tessellation) = &self.tessellation {
            if self.stage != Stage::Vertex {
                errors.push("tessellation is valid only for vertex cases".into());
            }
            if tessellation.factors.is_empty() {
                errors.push("tessellation.factors must contain at least one patch".into());
            }
            if tessellation.instance_count == 0 {
                errors.push("tessellation.instance_count must be nonzero".into());
            }
            if tessellation.amplification_count != 1 {
                errors.push(
                    "tessellation.amplification_count must be 1 for the single-layer observation contract"
                        .into(),
                );
            }
            ensure_unique(
                "tessellation control-point location",
                tessellation
                    .control_points
                    .iter()
                    .map(|input| input.location),
                &mut errors,
            );
            ensure_unique(
                "tessellation patch-input location",
                tessellation.patch_inputs.iter().map(|input| input.location),
                &mut errors,
            );
            validate_attribute_inputs(
                "tessellation control point",
                &tessellation.control_points,
                None,
                &HashSet::new(),
                false,
                &mut errors,
            );
            validate_attribute_inputs(
                "tessellation patch input",
                &tessellation.patch_inputs,
                None,
                &HashSet::new(),
                false,
                &mut errors,
            );
        }

        let mut buffer_bindings = HashSet::new();
        for buffer in &self.buffers {
            if !buffer_bindings.insert(buffer.binding) {
                errors.push(format!("duplicate buffer binding {}", buffer.binding));
            }
            validate_rate_map_or_bytes(
                &format!("buffer binding {}", buffer.binding),
                buffer.role,
                buffer.bytes_b64.as_deref(),
                buffer.initial_bytes_b64.as_deref(),
                buffer.rasterization_rate_map.as_ref(),
                &mut errors,
            );
        }
        let mut device_buffer_array_bindings = HashSet::new();
        let mut device_buffer_array_ranges = Vec::<(u64, u64)>::new();
        for array in &self.device_buffer_arrays {
            if !device_buffer_array_bindings.insert(array.binding) {
                errors.push(format!(
                    "duplicate device-buffer-array binding {}",
                    array.binding
                ));
            }
            if array.length == 0 {
                errors.push(format!(
                    "device-buffer-array binding {} length must be nonzero",
                    array.binding
                ));
            }
            let start = u64::from(array.binding);
            let end = start + u64::from(array.length);
            if end > u64::from(u32::MAX) + 1 {
                errors.push(format!(
                    "device-buffer-array binding {} length {} overflows the Metal buffer-index namespace",
                    array.binding, array.length
                ));
            }
            if buffer_bindings
                .iter()
                .any(|binding| (start..end).contains(&u64::from(*binding)))
            {
                errors.push(format!(
                    "device-buffer-array binding {} length {} overlaps a buffer binding",
                    array.binding, array.length
                ));
            }
            if device_buffer_array_ranges
                .iter()
                .any(|(other_start, other_end)| start < *other_end && *other_start < end)
            {
                errors.push(format!(
                    "device-buffer-array binding {} length {} overlaps another device-buffer array",
                    array.binding, array.length
                ));
            }
            device_buffer_array_ranges.push((start, end));
            let mut previous = None;
            for element in &array.elements {
                if element.index >= array.length {
                    errors.push(format!(
                        "device-buffer-array binding {} element {} exceeds length {}",
                        array.binding, element.index, array.length
                    ));
                }
                if previous.is_some_and(|index| element.index <= index) {
                    errors.push(format!(
                        "device-buffer-array binding {} elements must be sorted and unique by index",
                        array.binding
                    ));
                }
                previous = Some(element.index);
                validate_role_bytes(
                    &format!(
                        "device-buffer-array binding {} element {}",
                        array.binding, element.index
                    ),
                    element.role,
                    element.bytes_b64.as_deref(),
                    element.initial_bytes_b64.as_deref(),
                    &mut errors,
                );
            }
            if array.elements.is_empty() {
                errors.push(format!(
                    "device-buffer-array binding {} must declare at least one element",
                    array.binding
                ));
            }
        }
        let device_buffer_array_owns = |binding: u32| {
            device_buffer_array_ranges
                .iter()
                .any(|(start, end)| (*start..*end).contains(&u64::from(binding)))
        };
        let mut embedded_field_keys = HashSet::new();
        for buffer in &self.argument_buffer_buffers {
            let key = (buffer.buffer_binding, buffer.field_offset);
            if !embedded_field_keys.insert(key) {
                errors.push(format!(
                    "duplicate argument-buffer buffer at buffer {} offset {}",
                    buffer.buffer_binding, buffer.field_offset
                ));
            }
            if !buffer_bindings.contains(&buffer.buffer_binding) {
                errors.push(format!(
                    "argument-buffer buffer at offset {} owns undeclared buffer {}",
                    buffer.field_offset, buffer.buffer_binding
                ));
            }
            let label = format!(
                "argument-buffer buffer at buffer {} offset {}",
                buffer.buffer_binding, buffer.field_offset
            );
            validate_rate_map_or_bytes(
                &label,
                buffer.role,
                buffer.bytes_b64.as_deref(),
                buffer.initial_bytes_b64.as_deref(),
                buffer.rasterization_rate_map.as_ref(),
                &mut errors,
            );
        }
        let mut acceleration_structure_bindings = HashSet::new();
        let mut threadgroup_bindings = HashSet::new();
        for resource in &self.threadgroup_memory {
            if !threadgroup_bindings.insert(resource.binding) {
                errors.push(format!(
                    "duplicate threadgroup-memory binding {}",
                    resource.binding
                ));
            }
            if resource.length == 0 {
                errors.push(format!(
                    "threadgroup-memory binding {} length must be nonzero",
                    resource.binding
                ));
            }
        }
        for resource in &self.acceleration_structures {
            if !acceleration_structure_bindings.insert(resource.binding) {
                errors.push(format!(
                    "duplicate acceleration-structure binding {}",
                    resource.binding
                ));
            }
            if buffer_bindings.contains(&resource.binding) {
                errors.push(format!(
                    "binding {} is declared as both a buffer and an acceleration structure",
                    resource.binding
                ));
            }
            if device_buffer_array_owns(resource.binding) {
                errors.push(format!(
                    "binding {} is declared as both a device-buffer-array slot and an acceleration structure",
                    resource.binding
                ));
            }
            match resource.kind {
                AccelerationStructureKind::Instance => {
                    if resource.primitive_triangles_b64.is_some() {
                        errors.push(format!(
                            "instance acceleration-structure binding {} cannot declare primitive triangles",
                            resource.binding
                        ));
                    }
                    if resource.child_references.len() > u32::MAX as usize {
                        errors.push(format!(
                            "acceleration-structure binding {} has too many instances",
                            resource.binding
                        ));
                    }
                }
                AccelerationStructureKind::Primitive => {
                    if !resource.child_references.is_empty() {
                        errors.push(format!(
                            "primitive acceleration-structure binding {} cannot declare child references",
                            resource.binding
                        ));
                    }
                    validate_primitive_triangles(resource, &mut errors);
                }
            }
        }
        let mut visible_reference_names = HashSet::new();
        for reference in &self.visible_function_references {
            validate_hash(
                &format!(
                    "visible function reference {:?} module_sha256",
                    reference.function
                ),
                &reference.module_sha256,
                &mut errors,
            );
            if reference.function.trim().is_empty() {
                errors.push("visible function reference name must not be empty".into());
            } else if !visible_reference_names.insert(reference.function.as_str()) {
                errors.push(format!(
                    "duplicate visible function reference {:?}",
                    reference.function
                ));
            }
        }
        let mut function_table_bindings = HashSet::new();
        for table in &self.visible_function_tables {
            let kind = "visible-function";
            if !function_table_bindings.insert(table.binding) {
                errors.push(format!(
                    "duplicate function-table buffer binding {}",
                    table.binding
                ));
            }
            for (other, occupied) in [
                ("buffer", buffer_bindings.contains(&table.binding)),
                (
                    "acceleration structure",
                    acceleration_structure_bindings.contains(&table.binding),
                ),
                (
                    "device-buffer-array slot",
                    device_buffer_array_owns(table.binding),
                ),
            ] {
                if occupied {
                    errors.push(format!(
                        "binding {} is declared as both a {kind} table and {other}",
                        table.binding
                    ));
                }
            }
            if table.size == 0 {
                errors.push(format!(
                    "{kind} table binding {} size must be nonzero",
                    table.binding
                ));
            }
            let mut previous = None;
            for entry in &table.entries {
                if entry.index >= table.size {
                    errors.push(format!(
                        "{kind} table binding {} entry {} exceeds size {}",
                        table.binding, entry.index, table.size
                    ));
                }
                if previous.is_some_and(|index| index >= entry.index) {
                    errors.push(format!(
                        "{kind} table binding {} entries must be sorted by unique index",
                        table.binding
                    ));
                }
                previous = Some(entry.index);
                validate_hash(
                    &format!(
                        "{kind} table binding {} entry {} module_sha256",
                        table.binding, entry.index
                    ),
                    &entry.module_sha256,
                    &mut errors,
                );
                if entry.function.trim().is_empty() {
                    errors.push(format!(
                        "{kind} table binding {} entry {} function must not be empty",
                        table.binding, entry.index
                    ));
                }
            }
        }
        for table in &self.intersection_function_tables {
            let kind = "intersection-function";
            if !function_table_bindings.insert(table.binding) {
                errors.push(format!(
                    "duplicate function-table buffer binding {}",
                    table.binding
                ));
            }
            for (other, occupied) in [
                ("buffer", buffer_bindings.contains(&table.binding)),
                (
                    "acceleration structure",
                    acceleration_structure_bindings.contains(&table.binding),
                ),
                (
                    "device-buffer-array slot",
                    device_buffer_array_owns(table.binding),
                ),
            ] {
                if occupied {
                    errors.push(format!(
                        "binding {} is declared as both a {kind} table and {other}",
                        table.binding
                    ));
                }
            }
            if table.size == 0 {
                errors.push(format!(
                    "{kind} table binding {} size must be nonzero",
                    table.binding
                ));
            }
            let mut previous = None;
            for entry in &table.entries {
                let index = entry.index();
                if index >= table.size {
                    errors.push(format!(
                        "{kind} table binding {} entry {index} exceeds size {}",
                        table.binding, table.size
                    ));
                }
                if previous.is_some_and(|previous| previous >= index) {
                    errors.push(format!(
                        "{kind} table binding {} entries must be sorted by unique index",
                        table.binding
                    ));
                }
                previous = Some(index);
                match entry {
                    IntersectionFunctionTableEntry::Linked {
                        module_sha256,
                        function,
                        ..
                    } => {
                        validate_hash(
                            &format!(
                                "{kind} table binding {} entry {index} module_sha256",
                                table.binding
                            ),
                            module_sha256,
                            &mut errors,
                        );
                        if function.trim().is_empty() {
                            errors.push(format!(
                                "{kind} table binding {} entry {index} function must not be empty",
                                table.binding
                            ));
                        }
                    }
                    IntersectionFunctionTableEntry::OpaqueTriangle { signature, .. } => {
                        if signature.windows(2).any(|pair| pair[0] >= pair[1]) {
                            errors.push(format!(
                                "{kind} table binding {} opaque entry {index} signature flags must be sorted and unique",
                                table.binding
                            ));
                        }
                    }
                }
            }
        }
        let mut pipeline_tokens = HashSet::new();
        for resource in &self.pipeline_states {
            let key = (resource.buffer_binding, resource.field_offset);
            if !embedded_field_keys.insert(key) {
                errors.push(format!("duplicate pipeline-state field {key:?}"));
            }
            if !buffer_bindings.contains(&resource.buffer_binding) {
                errors.push(format!(
                    "pipeline-state field {key:?} owns an undeclared buffer"
                ));
            }
            if resource.field_offset.checked_add(8).is_none() {
                errors.push(format!("pipeline-state field {key:?} offset overflows"));
            }
            if resource.token == 0 || !pipeline_tokens.insert(resource.token) {
                errors.push("pipeline-state tokens must be nonzero and unique".into());
            }
            validate_hash("pipeline-state case_id", &resource.case_id, &mut errors);
            validate_hash(
                "pipeline-state air_sha256",
                &resource.air_sha256,
                &mut errors,
            );
        }
        for resource in &self.command_buffers {
            let key = (resource.buffer_binding, resource.field_offset);
            if !embedded_field_keys.insert(key) {
                errors.push(format!("duplicate indirect command-buffer field {key:?}"));
            }
            if !buffer_bindings.contains(&resource.buffer_binding) {
                errors.push(format!(
                    "indirect command-buffer field {key:?} owns an undeclared buffer"
                ));
            }
            if let Err(error) = resource.record_storage_bytes() {
                errors.push(format!("indirect command-buffer field {key:?}: {error}"));
            }
        }
        // What the host encodes and executes: every pipeline token is unique across the pipeline
        // states and the executions' parent pipelines, and each command buffer's host state is
        // checked against the case's buffers and tokens.
        let parents = self
            .command_buffers
            .iter()
            .filter_map(|resource| resource.execute.as_ref()?.pipeline.as_ref());
        let mut tokens = pipeline_tokens.clone();
        for template in parents {
            if template.token == 0 || !tokens.insert(template.token) {
                errors.push("pipeline-state tokens must be nonzero and unique".into());
            }
            validate_hash("parent pipeline case_id", &template.case_id, &mut errors);
            validate_hash(
                "parent pipeline air_sha256",
                &template.air_sha256,
                &mut errors,
            );
        }
        let tokens = tokens
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        let lengths = self.buffer_lengths();
        for resource in &self.command_buffers {
            let key = (resource.buffer_binding, resource.field_offset);
            if let Err(error) = resource.validate_host_state(&lengths, &tokens) {
                errors.push(format!("indirect command-buffer field {key:?}: {error}"));
            }
            // A kernel's dispatch or a vertex or fragment case's draw is the producer whose
            // commands the host encodes before it and executes after it; a tile function's is a
            // render pass's tile dispatch, whose commands no executor models.
            if (!resource.commands.is_empty() || resource.execute.is_some())
                && self.stage == Stage::Kernel
                && self
                    .imageblock
                    .as_ref()
                    .is_some_and(|imageblock| imageblock.implicit_coverage.is_some())
            {
                errors.push(format!(
                    "indirect command-buffer field {key:?}: a tile function's commands are \
                     neither host-encoded nor executed"
                ));
            }
        }
        for table in &self.argument_buffer_intersection_function_tables {
            let key = (table.buffer_binding, table.field_offset);
            let label = format!(
                "argument-buffer intersection-function table at buffer {} offset {}",
                table.buffer_binding, table.field_offset
            );
            if !embedded_field_keys.insert(key) {
                errors.push(format!("duplicate {label}"));
            }
            if !buffer_bindings.contains(&table.buffer_binding) {
                errors.push(format!("{label} owns an undeclared buffer"));
            }
            if table.size == 0 {
                errors.push(format!("{label} size must be nonzero"));
            }
            let mut previous = None;
            for entry in &table.entries {
                let index = entry.index();
                if index >= table.size {
                    errors.push(format!("{label} entry {index} exceeds size {}", table.size));
                }
                if previous.is_some_and(|previous| previous >= index) {
                    errors.push(format!("{label} entries must be sorted by unique index"));
                }
                previous = Some(index);
                match entry {
                    IntersectionFunctionTableEntry::Linked {
                        module_sha256,
                        function,
                        ..
                    } => {
                        validate_hash(
                            &format!("{label} entry {index} module_sha256"),
                            module_sha256,
                            &mut errors,
                        );
                        if function.trim().is_empty() {
                            errors
                                .push(format!("{label} entry {index} function must not be empty"));
                        }
                    }
                    IntersectionFunctionTableEntry::OpaqueTriangle { signature, .. } => {
                        if signature.windows(2).any(|pair| pair[0] >= pair[1]) {
                            errors.push(format!(
                                "{label} opaque entry {index} signature flags must be sorted and unique"
                            ));
                        }
                    }
                }
            }
        }
        let mut texture_bindings = HashSet::new();
        for texture in &self.textures {
            if !texture_bindings.insert(texture.binding) {
                errors.push(format!("duplicate texture binding {}", texture.binding));
            }
            validate_dimensions(
                &format!("texture binding {} dimensions", texture.binding),
                &texture.dimensions,
                &mut errors,
            );
            if let Err(error) = crate::literal::texture_layout(
                texture.texture_type,
                texture.dimensions,
                texture.sample_count,
            ) {
                errors.push(format!("texture binding {}: {error}", texture.binding));
            }
            if texture.sample_count > 1 && texture.role != ResourceRole::Input {
                errors.push(format!(
                    "multisample texture binding {} must have input role",
                    texture.binding
                ));
            }
            restricted_format_use(
                &format!("texture binding {}", texture.binding),
                texture.format,
                texture.role,
                texture.sample_count,
                &mut errors,
            );
            let expected = if texture.mip_levels > 1 {
                // Every level's bytes, level 0 first ([`mips`]).
                let len = mips::literal_len(
                    texture.texture_type,
                    texture.dimensions,
                    texture.mip_levels,
                    texture.format.bytes_per_pixel(),
                );
                if len.is_none() {
                    errors.push(format!(
                        "texture binding {} byte size overflows host limits",
                        texture.binding
                    ));
                }
                len
            } else {
                checked_texture_extent_bytes(
                    &format!("texture binding {}", texture.binding),
                    &texture.dimensions,
                    texture.sample_count,
                    texture.format.bytes_per_pixel(),
                    &mut errors,
                )
            };
            validate_role_bytes_with_len(
                &format!("texture binding {}", texture.binding),
                texture.role,
                texture.bytes_b64.as_deref(),
                texture.initial_bytes_b64.as_deref(),
                expected,
                &mut errors,
            );
            sparse::validate(texture, &mut errors);
            mips::validate(texture, &mut errors);
        }
        for array in &self.texture_arrays {
            restricted_format_use(
                &format!("texture-array binding {}", array.binding),
                array.format,
                array.role,
                array.sample_count,
                &mut errors,
            );
            if array.elements.is_empty() {
                errors.push(format!(
                    "texture-array binding {} must have at least one element",
                    array.binding
                ));
            }
            for element in 0..array.elements.len() {
                let Some(slot) = array.binding.checked_add(element as u32) else {
                    errors.push(format!(
                        "texture-array binding {} slot range overflows",
                        array.binding
                    ));
                    break;
                };
                let function_constant_texture_alternative = element == 0
                    && array.overrides_texture_at_base
                    && !self.function_constants.is_empty()
                    && self.textures.iter().any(|texture| {
                        texture.binding == array.binding
                            && texture.role == ResourceRole::Input
                            && array.role == ResourceRole::Input
                            && texture.format == array.format
                            && texture.sample_count == array.sample_count
                            && texture_type_is_array_alternative(
                                texture.texture_type,
                                array.texture_type,
                            )
                    });
                if !texture_bindings.insert(slot) && !function_constant_texture_alternative {
                    errors.push(format!(
                        "texture-array binding {} element {element} overlaps Metal texture slot {slot}",
                        array.binding
                    ));
                }
            }
            if array.overrides_texture_at_base
                && !self.textures.iter().any(|texture| {
                    texture.binding == array.binding
                        && texture.role == ResourceRole::Input
                        && array.role == ResourceRole::Input
                        && texture.format == array.format
                        && texture.sample_count == array.sample_count
                        && texture_type_is_array_alternative(
                            texture.texture_type,
                            array.texture_type,
                        )
                })
            {
                errors.push(format!(
                    "texture-array binding {} overrides_texture_at_base requires a matching input-only layered texture alternative",
                    array.binding
                ));
            }
            if array.elements.len()
                > crate::air::meta::TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT as usize
            {
                errors.push(format!(
                    "texture-array binding {} has {} elements, maximum is {}",
                    array.binding,
                    array.elements.len(),
                    crate::air::meta::TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT
                ));
            }
            for (element_index, element) in array.elements.iter().enumerate() {
                let label = format!(
                    "texture-array binding {} element {element_index}",
                    array.binding
                );
                validate_dimensions(
                    &format!("{label} dimensions"),
                    &element.dimensions,
                    &mut errors,
                );
                if let Err(error) = crate::literal::texture_layout(
                    array.texture_type,
                    element.dimensions,
                    array.sample_count,
                ) {
                    errors.push(format!("{label}: {error}"));
                }
                if array.sample_count > 1 && array.role != ResourceRole::Input {
                    errors.push(format!("multisample {label} must have input role"));
                }
                let expected = checked_texture_extent_bytes(
                    &label,
                    &element.dimensions,
                    array.sample_count,
                    array.format.bytes_per_pixel(),
                    &mut errors,
                );
                validate_role_bytes_with_len(
                    &label,
                    array.role,
                    element.bytes_b64.as_deref(),
                    element.initial_bytes_b64.as_deref(),
                    expected,
                    &mut errors,
                );
            }
        }
        for texture in &self.argument_buffer_textures {
            restricted_format_use(
                &format!(
                    "argument-buffer texture at buffer {} offset {}",
                    texture.buffer_binding, texture.field_offset
                ),
                texture.format,
                texture.role,
                texture.sample_count,
                &mut errors,
            );
            let key = (texture.buffer_binding, texture.field_offset);
            if !embedded_field_keys.insert(key) {
                errors.push(format!(
                    "duplicate argument-buffer texture at buffer {} offset {}",
                    texture.buffer_binding, texture.field_offset
                ));
            }
            if !buffer_bindings.contains(&texture.buffer_binding) {
                errors.push(format!(
                    "argument-buffer texture at offset {} owns undeclared buffer {}",
                    texture.field_offset, texture.buffer_binding
                ));
            }
            validate_dimensions(
                &format!(
                    "argument-buffer texture at buffer {} offset {} dimensions",
                    texture.buffer_binding, texture.field_offset
                ),
                &texture.dimensions,
                &mut errors,
            );
            if let Err(error) = crate::literal::texture_layout(
                texture.texture_type,
                texture.dimensions,
                texture.sample_count,
            ) {
                errors.push(format!(
                    "argument-buffer texture at buffer {} offset {}: {error}",
                    texture.buffer_binding, texture.field_offset
                ));
            }
            if texture.sample_count > 1 && texture.role != ResourceRole::Input {
                errors.push(format!(
                    "multisample argument-buffer texture at buffer {} offset {} must have input role",
                    texture.buffer_binding, texture.field_offset
                ));
            }
            let expected = checked_texture_extent_bytes(
                &format!(
                    "argument-buffer texture at buffer {} offset {}",
                    texture.buffer_binding, texture.field_offset
                ),
                &texture.dimensions,
                texture.sample_count,
                texture.format.bytes_per_pixel(),
                &mut errors,
            );
            validate_role_bytes_with_len(
                &format!(
                    "argument-buffer texture at buffer {} offset {}",
                    texture.buffer_binding, texture.field_offset
                ),
                texture.role,
                texture.bytes_b64.as_deref(),
                texture.initial_bytes_b64.as_deref(),
                expected,
                &mut errors,
            );
        }
        // A sampler state held in an argument buffer shares its owner's fields with every other
        // member.
        for sampler in &self.argument_buffer_samplers {
            let key = (sampler.buffer_binding, sampler.field_offset);
            if !embedded_field_keys.insert(key) {
                errors.push(format!(
                    "duplicate argument-buffer sampler at buffer {} offset {}",
                    sampler.buffer_binding, sampler.field_offset
                ));
            }
            if !buffer_bindings.contains(&sampler.buffer_binding) {
                errors.push(format!(
                    "argument-buffer sampler at offset {} owns undeclared buffer {}",
                    sampler.field_offset, sampler.buffer_binding
                ));
            }
        }
        ensure_unique(
            "sampler binding",
            self.samplers.iter().map(|sampler| sampler.binding),
            &mut errors,
        );
        ensure_unique(
            "render target index",
            self.render_targets.iter().map(|target| target.index),
            &mut errors,
        );
        // Every attachment holds the pass's samples of each pixel.
        let samples = self.raster_samples().max(1) as usize;
        for target in &self.render_targets {
            if target.format.sampled_only() {
                errors.push(format!(
                    "render target {} format {:?} is sampled-only",
                    target.index, target.format
                ));
            }
            if target.format.is_ulong() {
                errors.push(format!(
                    "render target {} format {:?} is the ulong texture format, which no render \
                     target holds",
                    target.index, target.format
                ));
            }
            validate_dimensions(
                &format!("render target {} dimensions", target.index),
                &target.dimensions,
                &mut errors,
            );
            let expected = checked_extent_bytes(
                &format!("render target {}", target.index),
                &target.dimensions,
                target.format.bytes_per_pixel() * samples,
                &mut errors,
            );
            validate_b64_len(
                &format!("render target {} initial_bytes_b64", target.index),
                &target.initial_bytes_b64,
                expected,
                &mut errors,
            );
        }
        if let Some(attachment) = &self.depth_stencil {
            validate_dimensions(
                "depth/stencil attachment",
                &attachment.dimensions,
                &mut errors,
            );
            let pixels = attachment.dimensions[0]
                .checked_mul(attachment.dimensions[1])
                .and_then(|pixels| pixels.checked_mul(samples as u32))
                .map(u64::from);
            if attachment.initial_depth_b64.is_none() && attachment.initial_stencil_b64.is_none() {
                errors
                    .push("depth/stencil attachment requires at least one authored aspect".into());
            }
            if let Some(bytes) = &attachment.initial_depth_b64 {
                validate_b64_len(
                    "depth/stencil initial_depth_b64",
                    bytes,
                    pixels.and_then(|count| usize::try_from(count.checked_mul(4)?).ok()),
                    &mut errors,
                );
            }
            if let Some(bytes) = &attachment.initial_stencil_b64 {
                validate_b64_len(
                    "depth/stencil initial_stencil_b64",
                    bytes,
                    pixels.and_then(|count| usize::try_from(count).ok()),
                    &mut errors,
                );
            }
            if self.stage != Stage::Fragment {
                errors.push("depth_stencil is valid only for fragment cases".into());
            }
            if let Some(state) = &attachment.state {
                validate_depth_stencil_state(attachment, state, &mut errors);
            }
        }
        ensure_unique(
            "vertex input location",
            self.vertex_inputs.iter().map(|input| input.location),
            &mut errors,
        );
        let targets = self.out_of_bounds_attribute_locations();
        validate_attribute_inputs(
            "vertex input",
            &self.vertex_inputs,
            self.vertex_elements(),
            &targets,
            true,
            &mut errors,
        );
        if !self.vertex_inputs.is_empty() && self.stage != Stage::Vertex {
            errors.push("vertex_inputs are valid only for vertex cases".into());
        }
        validate_varying_values(self, &mut errors);
        match (self.stage, self.vertex_observation) {
            (Stage::Vertex, None) => {
                if !self.render_targets.is_empty() || self.depth_stencil.is_some() {
                    errors.push(
                        "rasterization-disabled vertex execution does not accept attachments"
                            .into(),
                    );
                }
                if matches!(
                    self.output,
                    OutputSelection::RenderTarget { .. }
                        | OutputSelection::Depth { .. }
                        | OutputSelection::Stencil { .. }
                        | OutputSelection::FragmentImageblock { .. }
                ) {
                    errors.push(
                        "rasterization-disabled vertex output must select a shader resource".into(),
                    );
                }
            }
            (Stage::Vertex, Some(_)) => {
                if self.render_targets.len() != 1 || self.render_targets[0].index != 0 {
                    errors.push("vertex observation requires exactly render target zero".into());
                }
                if !matches!(self.output, OutputSelection::RenderTarget { index: 0, .. }) {
                    errors.push("vertex observation output must select render target zero".into());
                }
            }
            (_, Some(_)) => errors.push("vertex_observation is valid only for vertex cases".into()),
            (_, None) => {}
        }
        ensure_unique(
            "kernel stage-input location",
            self.kernel_stage_inputs.iter().map(|input| input.location),
            &mut errors,
        );
        let stage_in_elements = match &self.dispatch {
            Some(dispatch) => {
                if dispatch.stage_in_region.is_some() && self.kernel_stage_inputs.is_empty() {
                    errors.push(
                        "dispatch stage_in_region is valid only with kernel stage inputs".into(),
                    );
                }
                let elements = dispatch.stage_in_elements();
                if elements.is_none() {
                    errors.push(
                        "dispatch stage_in_region's x origin plus the elements it fetches \
                         overflows 32 bits"
                            .into(),
                    );
                }
                elements
            }
            None => None,
        };
        validate_attribute_inputs(
            "kernel stage input",
            &self.kernel_stage_inputs,
            stage_in_elements,
            &targets,
            true,
            &mut errors,
        );
        for input in &self.kernel_stage_inputs {
            let expected = input.format.storage_buffer_stride();
            if input.stride != expected {
                errors.push(format!(
                    "kernel stage input {} stride {} must equal product storage-buffer stride {expected}",
                    input.location, input.stride
                ));
            }
        }
        if !self.kernel_stage_inputs.is_empty() && self.stage != Stage::Kernel {
            errors.push("kernel_stage_inputs are valid only for kernel cases".into());
        }
        ensure_unique(
            "function constant index",
            self.function_constants.iter().map(|fc| fc.index),
            &mut errors,
        );
        for fc in &self.function_constants {
            if !(1..=4).contains(&fc.lanes) {
                errors.push(format!(
                    "function constant {} lanes must be in 1..=4",
                    fc.index
                ));
            }
            validate_b64_len(
                &format!("function constant {} bytes_b64", fc.index),
                &fc.bytes_b64,
                fc.scalar_type.byte_size().checked_mul(fc.lanes as usize),
                &mut errors,
            );
            if fc.scalar_type == ScalarType::Bool
                && base64::engine::general_purpose::STANDARD
                    .decode(&fc.bytes_b64)
                    .is_ok_and(|bytes| bytes.iter().any(|byte| !matches!(byte, 0 | 1)))
            {
                errors.push(format!(
                    "function constant {} bool must be encoded as 0 or 1",
                    fc.index
                ));
            }
        }
        self.validate_render_pass_rate_map(&mut errors);
        self.validate_raster_sample_count(&mut errors);
        self.validate_output_selection(&mut errors);
        out_of_bounds::validate(self, &mut errors);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// A mesh draw ([`MeshDraw`]) in an object or mesh case, and what such a case authors around
    /// it: color render targets only where it rasterizes, no indirect command buffer (whose
    /// producers are kernels, vertex and fragment functions), and threadgroup memory only for an
    /// object function, whose `setObjectThreadgroupMemoryLength:atIndex:` sets it (Metal has no
    /// such setter for a mesh function).
    fn validate_mesh_draw(&self, mesh_draw: &MeshDraw, errors: &mut Vec<String>) {
        if !self.stage.is_mesh_pipeline_stage() {
            errors.push("mesh_draw is valid only for object and mesh cases".into());
            return;
        }
        mesh_draw.validate(self.stage, &self.entry, errors);
        if !mesh_draw.rasterization {
            if !self.render_targets.is_empty() {
                errors.push(
                    "a mesh draw without rasterization does not accept render targets".into(),
                );
            }
            if matches!(self.output, OutputSelection::RenderTarget { .. }) {
                errors
                    .push("a mesh draw without rasterization must select a shader resource".into());
            }
        }
        if self.stage == Stage::Mesh && !self.threadgroup_memory.is_empty() {
            errors.push(
                "threadgroup_memory of a mesh case: Metal sets no mesh function's threadgroup \
                 memory"
                    .into(),
            );
        }
        // A depth/stencil attachment, a multisampled or rate-mapped pass, varying values, vertex
        // and kernel stage inputs and imageblocks are refused by their own checks, each valid
        // only for its stage.
        for (present, what) in [
            (!self.command_buffers.is_empty(), "command_buffers"),
            (!self.pipeline_states.is_empty(), "pipeline_states"),
        ] {
            if present {
                errors.push(format!(
                    "{what} is not modelled for an object or mesh case: an indirect command \
                     buffer's producer is a kernel, vertex or fragment function"
                ));
            }
        }
        // Metal's render encoder binds an object or mesh function's buffers, bytes, textures,
        // samplers and (an object function's) threadgroup memory, and has no object or mesh setter
        // for an acceleration structure or a function table; no executor links visible functions
        // into a mesh render pipeline or shadows a mesh draw's out-of-bounds targets.
        for (present, what) in [
            (
                !self.acceleration_structures.is_empty(),
                "acceleration_structures",
            ),
            (
                !self.visible_function_tables.is_empty(),
                "visible_function_tables",
            ),
            (
                !self.intersection_function_tables.is_empty(),
                "intersection_function_tables",
            ),
            (
                !self.argument_buffer_intersection_function_tables.is_empty(),
                "argument_buffer_intersection_function_tables",
            ),
            (
                !self.visible_function_references.is_empty(),
                "visible_function_references",
            ),
            (self.out_of_bounds.is_some(), "out_of_bounds"),
        ] {
            if present {
                errors.push(format!("{what} is not modelled for an object or mesh case"));
            }
        }
    }

    /// A render pass's rate map: a valid map of one layer, in a case that rasterizes through a
    /// fragment function (a fragment case, or a vertex case observed through its generated fragment
    /// function) and draws no patches or imageblock, whose every attachment holds the map's screen
    /// (so the physical extent of any device's compile, which never exceeds it).
    fn validate_render_pass_rate_map(&self, errors: &mut Vec<String>) {
        let Some(map) = &self.render_pass_rate_map else {
            return;
        };
        map.validate("render_pass_rate_map", errors);
        if map.layers.len() > 1 {
            errors.push(format!(
                "render_pass_rate_map has {} layers: a layered rate-mapped pass is not modelled",
                map.layers.len()
            ));
        }
        let rasterizes = match self.stage {
            Stage::Fragment => true,
            Stage::Vertex => self.vertex_observation.is_some(),
            Stage::Kernel | Stage::Object | Stage::Mesh => false,
        };
        if !rasterizes {
            errors.push(
                "render_pass_rate_map is valid only for a fragment case or an observed vertex case"
                    .into(),
            );
        }
        if self.tessellation.is_some() || self.fragment_imageblock.is_some() {
            errors.push(
                "render_pass_rate_map is not modelled with tessellation or a fragment imageblock"
                    .into(),
            );
        }
        // Metal sizes the pass's raster from its attachments: without one there is no physical
        // image for the map to rasterize into, and the pass would draw a default-sized raster
        // under a map of another screen.
        if self.render_targets.is_empty() && self.depth_stencil.is_none() {
            errors.push(
                "render_pass_rate_map needs a render target or a depth/stencil attachment: a pass \
                 without one has no raster for the map"
                    .into(),
            );
        }
        let screen = map.screen_size;
        let holds = |dimensions: [u32; 2]| dimensions[0] >= screen[0] && dimensions[1] >= screen[1];
        for target in &self.render_targets {
            if !holds(target.dimensions) {
                errors.push(format!(
                    "render target {} dimensions {:?} do not hold the rate map's screen {screen:?}",
                    target.index, target.dimensions
                ));
            }
        }
        if let Some(attachment) = &self.depth_stencil {
            if !holds(attachment.dimensions) {
                errors.push(format!(
                    "depth/stencil dimensions {:?} do not hold the rate map's screen {screen:?}",
                    attachment.dimensions
                ));
            }
        }
    }

    /// A multisampled render pass: a count of [`RASTER_SAMPLE_COUNTS`], written only above one, in
    /// a fragment case's draw. Its attachments are color targets and a depth aspect, filled and
    /// read back sample by sample, and a stencil aspect whose samples hold one value, which each
    /// executor clears it to and reads back sample by sample (no executor initializes a stencil
    /// sample by sample, [`crate::executor_contract::multisample::uniform_stencil`]). The case
    /// schema has no multisampled fragment imageblock or tile pass and no rate-mapped multisampled
    /// pass, so each of those is refused. (A layered pass is refused where the checker reads the
    /// function's stage metadata.)
    fn validate_raster_sample_count(&self, errors: &mut Vec<String>) {
        let Some(count) = self.raster_sample_count else {
            return;
        };
        if count == 1 {
            errors.push(
                "raster_sample_count 1 is written by omitting the field: a pass of one sample has \
                 one spelling"
                    .into(),
            );
        } else if !RASTER_SAMPLE_COUNTS.contains(&count) {
            errors.push(format!(
                "raster_sample_count {count} is not one of {RASTER_SAMPLE_COUNTS:?}: the Metal \
                 reference device (Apple M3 Max) rasterizes 1, 2, and 4 samples, so no Metal \
                 answer exists for another count"
            ));
        }
        if self.stage != Stage::Fragment || self.draw.is_none() {
            errors.push("raster_sample_count is valid only for a fragment case's draw".into());
        }
        if let Some(bytes) = self
            .depth_stencil
            .as_ref()
            .and_then(|attachment| initial_bytes(attachment.initial_stencil_b64.as_deref(), None))
        {
            if let Err(why) = crate::executor_contract::multisample::uniform_stencil(&bytes) {
                errors.push(format!("a multisampled render pass with {why}"));
            }
        }
        let refused = [
            (
                self.fragment_imageblock.is_some(),
                "a fragment imageblock: a multisampled imageblock is not modelled",
            ),
            (
                self.render_pass_rate_map.is_some(),
                "a rasterization-rate map: a multisampled rate-mapped pass is not modelled",
            ),
        ];
        for (present, what) in refused {
            if present {
                errors.push(format!("a multisampled render pass with {what}"));
            }
        }
    }

    fn validate_output_selection(&self, errors: &mut Vec<String>) {
        match &self.output {
            OutputSelection::None => {}
            OutputSelection::Buffer {
                binding,
                offset,
                length,
            } => {
                if *length == 0 {
                    errors.push("buffer output length must be nonzero".into());
                }
                let Some(resource) = self.buffers.iter().find(|item| item.binding == *binding)
                else {
                    errors.push(format!("buffer output binding {binding} is not declared"));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "buffer output binding {binding} has input-only role"
                    ));
                }
                if let Some(bytes) = initial_bytes(
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                ) {
                    if offset.saturating_add(*length) > bytes.len() as u64 {
                        errors.push(format!(
                            "buffer output [{offset}, {}) exceeds binding {binding} length {}",
                            offset.saturating_add(*length),
                            bytes.len()
                        ));
                    }
                }
            }
            OutputSelection::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
                offset,
                length,
            } => {
                let Some(resource) = self.argument_buffer_buffers.iter().find(|item| {
                    item.buffer_binding == *buffer_binding && item.field_offset == *field_offset
                }) else {
                    errors.push(format!(
                        "argument-buffer buffer at buffer {buffer_binding} offset {field_offset} is not declared"
                    ));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "argument-buffer buffer at buffer {buffer_binding} offset {field_offset} has input-only role"
                    ));
                }
                if *length == 0 {
                    errors.push("argument-buffer buffer output length must be nonzero".into());
                }
                if offset.checked_add(*length).is_none() {
                    errors.push("argument-buffer buffer output range overflows".into());
                }
            }
            OutputSelection::DeviceBufferArrayElement {
                binding,
                element,
                offset,
                length,
            } => {
                let Some(resource) = self
                    .device_buffer_arrays
                    .iter()
                    .find(|item| item.binding == *binding)
                    .and_then(|array| array.elements.iter().find(|item| item.index == *element))
                else {
                    errors.push(format!(
                        "device-buffer-array output binding {binding} element {element} is not declared"
                    ));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "device-buffer-array output binding {binding} element {element} has input-only role"
                    ));
                }
                if *length == 0 {
                    errors.push("device-buffer-array output length must be nonzero".into());
                }
                if let Some(bytes) = initial_bytes(
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                ) {
                    if offset.saturating_add(*length) > bytes.len() as u64 {
                        errors.push(format!(
                            "device-buffer-array output [{offset}, {}) exceeds binding {binding} element {element} length {}",
                            offset.saturating_add(*length),
                            bytes.len()
                        ));
                    }
                }
            }
            OutputSelection::Texture {
                binding,
                origin,
                dimensions,
            } => {
                let Some(resource) = self.textures.iter().find(|item| item.binding == *binding)
                else {
                    errors.push(format!("texture output binding {binding} is not declared"));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "texture output binding {binding} has input-only role"
                    ));
                }
                validate_region(
                    "texture output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::TextureArrayElement {
                binding,
                element,
                origin,
                dimensions,
            } => {
                let Some(resource) = self
                    .texture_arrays
                    .iter()
                    .find(|item| item.binding == *binding)
                else {
                    errors.push(format!(
                        "texture-array output binding {binding} is not declared"
                    ));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "texture-array output binding {binding} has input-only role"
                    ));
                }
                let Some(selected) = resource.elements.get(*element as usize) else {
                    errors.push(format!(
                        "texture-array output binding {binding} element {element} is not declared"
                    ));
                    return;
                };
                validate_region(
                    "texture-array element output",
                    origin,
                    dimensions,
                    &selected.dimensions,
                    errors,
                );
            }
            OutputSelection::ArgumentBufferTexture {
                buffer_binding,
                field_offset,
                origin,
                dimensions,
            } => {
                let Some(resource) = self.argument_buffer_textures.iter().find(|item| {
                    item.buffer_binding == *buffer_binding && item.field_offset == *field_offset
                }) else {
                    errors.push(format!(
                        "argument-buffer texture at buffer {buffer_binding} offset {field_offset} is not declared"
                    ));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "argument-buffer texture at buffer {buffer_binding} offset {field_offset} has input-only role"
                    ));
                }
                validate_region(
                    "argument-buffer texture output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::RenderTarget {
                index,
                origin,
                dimensions,
            } => {
                let Some(resource) = self.render_targets.iter().find(|item| item.index == *index)
                else {
                    errors.push(format!(
                        "render target output index {index} is not declared"
                    ));
                    return;
                };
                validate_region(
                    "render target output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::Depth { origin, dimensions } => {
                let Some(resource) = &self.depth_stencil else {
                    errors.push("depth output requires a depth/stencil attachment".into());
                    return;
                };
                if resource.initial_depth_b64.is_none() {
                    errors.push("depth output requires an authored depth aspect".into());
                }
                validate_region(
                    "depth output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::Stencil { origin, dimensions } => {
                let Some(resource) = &self.depth_stencil else {
                    errors.push("stencil output requires a depth/stencil attachment".into());
                    return;
                };
                if resource.initial_stencil_b64.is_none() {
                    errors.push("stencil output requires an authored stencil aspect".into());
                }
                validate_region(
                    "stencil output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::FragmentImageblock {
                semantic,
                origin,
                dimensions,
            } => {
                let Some(imageblock) = &self.fragment_imageblock else {
                    errors.push(
                        "fragment imageblock output requires a fragment_imageblock resource".into(),
                    );
                    return;
                };
                let Some(member) = imageblock
                    .members
                    .iter()
                    .find(|member| member.semantic == *semantic)
                else {
                    errors.push(format!(
                        "fragment imageblock output semantic {semantic} is not declared"
                    ));
                    return;
                };
                if member.role == ResourceRole::Input {
                    errors.push(format!(
                        "fragment imageblock output semantic {semantic} has input-only role"
                    ));
                }
                validate_region(
                    "fragment imageblock output",
                    origin,
                    dimensions,
                    &imageblock.dimensions,
                    errors,
                );
            }
        }
    }
}

fn sorted_refs<T, K: Ord>(values: &[T], key: impl Fn(&T) -> K) -> Vec<&T> {
    let mut values = values.iter().collect::<Vec<_>>();
    values.sort_by_key(|value| key(value));
    values
}

/// A sampled-only format ([`TextureFormat::sampled_only`]) bound where a case writes or
/// multisamples the texture, or the `ulong` format ([`TextureFormat::is_ulong`]) bound where it
/// multisamples it.
fn restricted_format_use(
    label: &str,
    format: TextureFormat,
    role: ResourceRole,
    sample_count: u32,
    errors: &mut Vec<String>,
) {
    if format.sampled_only() && (role != ResourceRole::Input || sample_count > 1) {
        errors.push(format!(
            "{label}: {format:?} is a sampled-only format, bound only as a single-sample input"
        ));
    }
    if format.is_ulong() && sample_count > 1 {
        errors.push(format!(
            "{label}: {format:?} is the ulong texture format, bound only to a single-sample texture"
        ));
    }
}

fn validate_hash(field: &str, value: &str, errors: &mut Vec<String>) {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        errors.push(format!("{field} must be 64 hexadecimal characters"));
    } else if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        errors.push(format!("{field} must use lowercase hexadecimal"));
    }
}

fn validate_dimensions<const N: usize>(field: &str, dims: &[u32; N], errors: &mut Vec<String>) {
    if dims.contains(&0) {
        errors.push(format!("{field} dimensions must be nonzero"));
    }
}

/// A literal depth/stencil state must state 8-bit masks and references, and test or write only
/// aspects the attachment authors: Metal attaches only the authored aspects, so a test of another
/// would run on Metal without an attachment and on Vulkan against the combined attachment's
/// unauthored aspect.
fn validate_depth_stencil_state(
    attachment: &DepthStencilResource,
    state: &DepthStencilStateLiteral,
    errors: &mut Vec<String>,
) {
    for (name, value) in [
        ("front.read_mask", state.front.read_mask),
        ("front.write_mask", state.front.write_mask),
        ("back.read_mask", state.back.read_mask),
        ("back.write_mask", state.back.write_mask),
        ("front_reference", state.front_reference),
        ("back_reference", state.back_reference),
    ] {
        if value > 0xFF {
            errors.push(format!(
                "depth/stencil state {name} {value:#x} is wider than the 8-bit stencil"
            ));
        }
    }
    let draw = crate::executor_contract::DrawDepthStencil::authored(state);
    if draw.tests_depth() && attachment.initial_depth_b64.is_none() {
        errors.push("depth/stencil state tests or writes depth without an authored depth".into());
    }
    if draw.tests_stencil() && attachment.initial_stencil_b64.is_none() {
        errors
            .push("depth/stencil state tests or writes stencil without an authored stencil".into());
    }
}

fn checked_extent_bytes<const N: usize>(
    field: &str,
    dimensions: &[u32; N],
    bytes_per_element: usize,
    errors: &mut Vec<String>,
) -> Option<usize> {
    let bytes = dimensions.iter().try_fold(1usize, |size, dimension| {
        size.checked_mul(*dimension as usize)
    });
    match bytes.and_then(|size| size.checked_mul(bytes_per_element)) {
        Some(bytes) => Some(bytes),
        None => {
            errors.push(format!("{field} byte size overflows host limits"));
            None
        }
    }
}

fn checked_texture_extent_bytes(
    field: &str,
    dimensions: &[u32; 3],
    sample_count: u32,
    bytes_per_element: usize,
    errors: &mut Vec<String>,
) -> Option<usize> {
    let bytes_per_texel = bytes_per_element.checked_mul(sample_count as usize);
    match bytes_per_texel {
        Some(bytes_per_texel) => checked_extent_bytes(field, dimensions, bytes_per_texel, errors),
        None => {
            errors.push(format!("{field} byte size overflows host limits"));
            None
        }
    }
}

/// Each attribute input's stride and bytes, holding at least `required_records` elements unless
/// its location is one of `targets`, an out-of-bounds target whose fetches past its buffer the
/// case declares.
fn validate_attribute_inputs(
    label: &str,
    inputs: &[AttributeInput],
    required_records: Option<u32>,
    targets: &HashSet<u32>,
    dynamic_strides: bool,
    errors: &mut Vec<String>,
) {
    for input in inputs {
        if input.dynamic_stride && !dynamic_strides {
            errors.push(format!(
                "{label} {} declares a dynamic stride, which no executor models for its layout",
                input.location
            ));
        }
        let required_records = required_records.filter(|_| !targets.contains(&input.location));
        if input.stride < input.format.byte_size() {
            errors.push(format!(
                "{label} {} stride {} is smaller than format size {}",
                input.location,
                input.stride,
                input.format.byte_size()
            ));
        }
        validate_b64_len(
            &format!("{label} {} bytes_b64", input.location),
            &input.bytes_b64,
            None,
            errors,
        );
        if let (Some(records), Ok(bytes)) = (
            required_records,
            base64::engine::general_purpose::STANDARD.decode(&input.bytes_b64),
        ) {
            match records
                .checked_mul(input.stride)
                .map(|required| required as usize)
            {
                Some(required) if bytes.len() < required => errors.push(format!(
                    "{label} {} has {} bytes, but execution requires at least {required}",
                    input.location,
                    bytes.len()
                )),
                None => errors.push(format!("{label} {} byte range overflows", input.location)),
                _ => {}
            }
        }
    }
}

/// The structural rules of `case.varying_values` ([`VaryingValue`]): a fragment case's, one
/// value per key, with 1 to 4 lanes. The checker matches each against the fragment function's
/// inputs (`crate::companion::derive`).
fn validate_varying_values(case: &AuthoredCase, errors: &mut Vec<String>) {
    if case.varying_values.is_empty() {
        return;
    }
    if case.stage != Stage::Fragment {
        errors.push("varying_values are valid only for fragment cases".into());
    }
    let mut keys = HashSet::new();
    for value in &case.varying_values {
        if value.key.is_empty() {
            errors.push("a varying value's key must not be empty".into());
        }
        if !keys.insert(value.key.as_str()) {
            errors.push(format!("duplicate varying value key {:?}", value.key));
        }
        if !(1..=4).contains(&value.lanes.len()) {
            errors.push(format!(
                "varying value {:?} has {} lanes; a varying has 1 to 4",
                value.key,
                value.lanes.len()
            ));
        }
    }
}

fn ensure_unique(field: &str, values: impl IntoIterator<Item = u32>, errors: &mut Vec<String>) {
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(value) {
            errors.push(format!("duplicate {field} {value}"));
        }
    }
}

fn validate_role_bytes(
    label: &str,
    role: ResourceRole,
    bytes: Option<&str>,
    initial: Option<&str>,
    errors: &mut Vec<String>,
) {
    validate_role_bytes_with_len(label, role, bytes, initial, None, errors);
}

/// A buffer's contents: a rasterization-rate map, which is an input's only contents, or its role's
/// bytes.
fn validate_rate_map_or_bytes(
    label: &str,
    role: ResourceRole,
    bytes: Option<&str>,
    initial: Option<&str>,
    map: Option<&RateMapLiteral>,
    errors: &mut Vec<String>,
) {
    let Some(map) = map else {
        return validate_role_bytes(label, role, bytes, initial, errors);
    };
    if role != ResourceRole::Input || bytes.is_some() || initial.is_some() {
        errors.push(format!(
            "{label} rasterization_rate_map is an input's only contents (no bytes_b64 or \
             initial_bytes_b64)"
        ));
    }
    map.validate(&format!("{label} rasterization_rate_map"), errors);
}

fn validate_role_bytes_with_len(
    label: &str,
    role: ResourceRole,
    bytes: Option<&str>,
    initial: Option<&str>,
    expected_len: Option<usize>,
    errors: &mut Vec<String>,
) {
    let selected = match role {
        ResourceRole::Input => match (bytes, initial) {
            (Some(value), None) => Some(("bytes_b64", value)),
            _ => {
                errors.push(format!(
                    "{label} input requires only bytes_b64 (no inferred or initial bytes)"
                ));
                None
            }
        },
        ResourceRole::Output | ResourceRole::InOut => match (bytes, initial) {
            (None, Some(value)) => Some(("initial_bytes_b64", value)),
            _ => {
                errors.push(format!(
                    "{label} output/in_out requires only initial_bytes_b64"
                ));
                None
            }
        },
    };
    if let Some((field, value)) = selected {
        validate_b64_len(&format!("{label} {field}"), value, expected_len, errors);
    }
}

fn validate_b64_len(
    field: &str,
    value: &str,
    expected_len: Option<usize>,
    errors: &mut Vec<String>,
) {
    match base64::engine::general_purpose::STANDARD.decode(value) {
        Ok(bytes) if bytes.is_empty() => errors.push(format!("{field} must not be empty")),
        Ok(bytes) if expected_len.is_some_and(|expected| bytes.len() != expected) => {
            errors.push(format!(
                "{field} decodes to {} bytes, expected {}",
                bytes.len(),
                expected_len.unwrap_or_default()
            ))
        }
        Ok(_) => {}
        Err(error) => errors.push(format!("{field} is invalid base64: {error}")),
    }
}

fn validate_primitive_triangles(
    resource: &AccelerationStructureResource,
    errors: &mut Vec<String>,
) {
    let label = format!(
        "primitive acceleration-structure binding {} triangles",
        resource.binding
    );
    let Some(encoded) = resource.primitive_triangles_b64.as_deref() else {
        errors.push(format!("{label} are required"));
        return;
    };
    let bytes = match base64::engine::general_purpose::STANDARD.decode(encoded) {
        Ok(bytes) => bytes,
        Err(error) => {
            errors.push(format!("{label} are not valid base64: {error}"));
            return;
        }
    };
    const TRIANGLE_BYTES: usize = 9 * std::mem::size_of::<f32>();
    if bytes.is_empty() || bytes.len() % TRIANGLE_BYTES != 0 {
        errors.push(format!(
            "{label} must contain one or more tightly packed 36-byte float3 triangles, got {} bytes",
            bytes.len()
        ));
        return;
    }
    for (index, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_le_bytes(*word);
        if !value.is_finite() {
            errors.push(format!("{label} float {index} is not finite"));
        }
    }
}

fn initial_bytes(bytes: Option<&str>, initial: Option<&str>) -> Option<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(bytes.or(initial)?)
        .ok()
}

fn validate_region<const N: usize>(
    label: &str,
    origin: &[u32; N],
    dimensions: &[u32; N],
    extent: &[u32; N],
    errors: &mut Vec<String>,
) {
    validate_dimensions(label, dimensions, errors);
    for axis in 0..N {
        if origin[axis]
            .checked_add(dimensions[axis])
            .is_none_or(|end| end > extent[axis])
        {
            errors.push(format!("{label} exceeds declared extent on axis {axis}"));
        }
    }
}

// Callers are the Metal platform tests and the engine render tests.
#[cfg(test)]
pub(crate) fn narrow_vertex_test_case(air_sha256: String, entry: String) -> AuthoredCase {
    AuthoredCase {
        air_sha256,
        case_id: "test-vertex-side-effect".into(),
        name: "vertex-narrow-attributes-smoke".into(),
        entry,
        stage: Stage::Vertex,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![
            AttributeInput {
                location: 0,
                format: AttributeFormat::Uchar,
                stride: 1,
                dynamic_stride: false,
                bytes_b64: "Ag==".into(),
            },
            AttributeInput {
                location: 1,
                format: AttributeFormat::Ushort2,
                stride: 4,
                dynamic_stride: false,
                bytes_b64: "AwAEAA==".into(),
            },
        ],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: None,
        draw: Some(Draw {
            primitive: Primitive::Point,
            vertex_start: 0,
            vertex_count: 1,
            instance_count: 1,
            index_buffer: None,
            base_vertex: 0,
            base_instance: 0,
        }),
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    }
}

#[cfg(test)]
mod tests;
