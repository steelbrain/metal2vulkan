//! One wrapper that forwards every [`Side`] call and lets something watch the boundary.
//!
//! Two parts of the suite need to do something **at each call** without changing what the call
//! does: [`crate::validated`] announces the call's name so the parent process can attribute
//! Metal's own output to it, and [`crate::times`] measures what the call cost. Both want the same
//! narrow span — around the Metal or Vulkan call and nothing else — and neither wants to reach
//! inside [`crate::metal_side::MetalSide`] or [`crate::runtime_side::RuntimeSide`], because a span
//! placed separately in each of them is two spans that can drift apart, and a cost comparison of
//! two legs measured in different places is not a comparison.
//!
//! So there is **one** forwarding impl, here, parameterised by what happens at the boundary. That
//! is not tidiness: `Side`'s methods mostly carry a default that answers
//! [`Failure::unavailable`](crate::case::Failure::unavailable), so a wrapper that forgets to
//! forward one answers "no such call" where the wrapped side answers fine, **and the compiler says
//! nothing**. A second hand-written wrapper would be a second copy of that hazard, in a trait
//! several streams are extending. One impl means one place to forget, and
//! `the_announcing_side_forwards_every_call` already watches that place.
//!
//! # What the span covers, and what it deliberately does not
//!
//! The span opens on entry to the `Side` method and closes when it returns, so it holds the Metal
//! or Vulkan call plus the side's own handle bookkeeping, and **excludes** building the transcript,
//! reading a fixture, hashing an input and everything else the harness does between calls. That is
//! the narrowest boundary both legs share.
//!
//! Two methods are forwarded without a span, because neither is a call to the implementation under
//! test: [`Side::venue`] answers a label, and [`Side::announced`] drains the notices our own side
//! collected, which Metal has no counterpart for at all. Timing them would put the harness's own
//! bookkeeping in the table.

use crate::argument_encoders::{ArgumentDescriptorSpec, ArgumentWrite, EncoderFacts};
use crate::case::{
    Answer, ArgumentSlot, Bindings, CoordinateProbe, Fill, FragmentFill, FragmentFilled, Handle,
    IndirectReadBack, MappedCoordinates, MeshCommand, MeshFunctions, MeshReflectionSpec, MeshStep,
    PassOutcome, PlaneTexture, RateMapFacts, RateMapRequest, SamplerSpec, ShaderSource, Side,
    StageInRegion, StageInput, TextureSpec, TileFacts, TilePipelineSpec, TileStep, Venue,
};
use crate::function_bindings::BindingFacts;
use crate::indirect_commands::{CommandsMade, EncodedCommand, IcbDescriptor};
use crate::transcript::Value;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{BindingAccess, PixelFormat, RenderStages};

/// What happens at a [`Side`] call's boundary.
///
/// `enter` is handed the call's name and answers a span. The span is dropped when the call
/// returns, so a watcher that wants the call's *end* does its work in the span's `Drop`. A span
/// must not borrow the watcher, because the wrapper holds the span across the forwarded call and
/// that call needs the side mutably; a watcher with state shares it through a handle of its own
/// instead.
pub trait Watcher {
    /// What `enter` answers, dropped when the call returns.
    type Span;

    /// `call` is about to be made.
    fn enter(&self, call: &'static str) -> Self::Span;
}

/// `inner`, with `watcher` watching every call it is asked to make. It observes nothing of the
/// answers and changes none of them.
pub struct Watching<S, W> {
    inner: S,
    watcher: W,
}

impl<S: Side, W: Watcher + Default> Watching<S, W> {
    /// `inner`, watched by a watcher that needs no setting up.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            watcher: W::default(),
        }
    }
}

impl<S: Side, W: Watcher> Watching<S, W> {
    /// `inner`, watched by `watcher`.
    pub fn with(inner: S, watcher: W) -> Self {
        Self { inner, watcher }
    }

    /// The watched side back.
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Open `call`'s span.
    ///
    /// It takes `&self` and answers a span that borrows nothing of `self`, which is what lets the
    /// caller hold the span across a forwarded call that needs `self.inner` mutably.
    fn watch(&self, call: &'static str) -> W::Span {
        self.watcher.enter(call)
    }
}

/// Every call is forwarded, and every call opens a span.
///
/// **A forwarded method that is left out is a defect, not a gap**: `Side` gives most of its calls a
/// default that answers [`Failure::unavailable`](crate::case::Failure::unavailable), so an
/// unforwarded method would make a wrapped side answer "no such call" where the side itself
/// answers perfectly well, and the compiler would say nothing. `the_announcing_side_forwards_every_call`
/// catches that by running the whole case set through both and comparing the transcripts, so the
/// trait can keep growing without this wrapper rotting.
///
/// The same test covers the timing side too, because there is one impl rather than one per
/// watcher. A call left out would also read as a *free* call in the cost table -- the worst
/// shape of the same defect, since a missing measurement there looks like good news.
impl<S: Side, W: Watcher> Side for Watching<S, W> {
    fn venue(&self) -> Venue {
        self.inner.venue()
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        let _span = self.watch("library");
        self.inner.library(source)
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let _span = self.watch("compute_pipeline");
        self.inner.compute_pipeline(library, entry)
    }

    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let _span = self.watch("static_threadgroup_memory_length");
        self.inner.static_threadgroup_memory_length(pipeline)
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle> {
        let _span = self.watch("compute_pipeline_with_stage_input");
        self.inner
            .compute_pipeline_with_stage_input(library, entry, stage_input)
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        let _span = self.watch("buffer");
        self.inner.buffer(bytes)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        let _span = self.watch("zeroed_buffer");
        self.inner.zeroed_buffer(length)
    }

    fn host_buffer(&mut self, spec: &crate::host_pages::HostBufferSpec) -> Answer<Handle> {
        let _span = self.watch("host_buffer");
        self.inner.host_buffer(spec)
    }

    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle> {
        let _span = self.watch("texture_2d");
        self.inner.texture_2d(format, size, row_pitch, bytes)
    }

    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let _span = self.watch("read_texture");
        self.inner.read_texture(texture, size, row_pitch)
    }

    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        let _span = self.watch("texture_2d_in_role");
        self.inner.texture_2d_in_role(spec, bytes)
    }

    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        let _span = self.watch("replace_region");
        self.inner.replace_region(texture, size, row_pitch, bytes)
    }

    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        let _span = self.watch("texture");
        self.inner.texture(request)
    }

    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let _span = self.watch("sampler");
        self.inner.sampler(spec)
    }

    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let _span = self.watch("argument_buffer");
        self.inner.argument_buffer(slots)
    }

    fn argument_encoder(
        &mut self,
        library: Handle,
        entry: &str,
        buffer_index: u32,
    ) -> Answer<Handle> {
        let _span = self.watch("argument_encoder");
        self.inner.argument_encoder(library, entry, buffer_index)
    }

    fn argument_encoder_with_arguments(
        &mut self,
        arguments: &[ArgumentDescriptorSpec],
    ) -> Answer<Handle> {
        let _span = self.watch("argument_encoder_with_arguments");
        self.inner.argument_encoder_with_arguments(arguments)
    }

    fn argument_encoder_facts(&mut self, encoder: Handle, label: &str) -> Answer<EncoderFacts> {
        let _span = self.watch("argument_encoder_facts");
        self.inner.argument_encoder_facts(encoder, label)
    }

    fn set_argument_buffer(
        &mut self,
        encoder: Handle,
        buffer: Handle,
        offset: u64,
        element: Option<u64>,
    ) -> Answer<()> {
        let _span = self.watch("set_argument_buffer");
        self.inner
            .set_argument_buffer(encoder, buffer, offset, element)
    }

    fn nested_argument_encoder(&mut self, encoder: Handle, index: u32) -> Answer<Option<Handle>> {
        let _span = self.watch("nested_argument_encoder");
        self.inner.nested_argument_encoder(encoder, index)
    }

    fn encode_argument(&mut self, encoder: Handle, write: &ArgumentWrite) -> Answer<()> {
        let _span = self.watch("encode_argument");
        self.inner.encode_argument(encoder, write)
    }

    fn compute_pipeline_with_descriptor(
        &mut self,
        library: Handle,
        spec: &crate::pipeline_facts::ComputeStateSpec<'_>,
    ) -> Answer<Handle> {
        let _span = self.watch("compute_pipeline_with_descriptor");
        self.inner.compute_pipeline_with_descriptor(library, spec)
    }

    fn compute_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<crate::pipeline_facts::ComputeStateFacts> {
        let _span = self.watch("compute_pipeline_facts");
        self.inner.compute_pipeline_facts(state, imageblock)
    }

    fn render_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<crate::pipeline_facts::RenderStateFacts> {
        let _span = self.watch("render_pipeline_facts");
        self.inner.render_pipeline_facts(state, imageblock)
    }

    fn render_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::render_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::render_descriptors::RenderDescriptorFacts> {
        let _span = self.watch("render_descriptor_facts");
        self.inner.render_descriptor_facts(library, spec)
    }

    fn compute_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::compute_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::compute_descriptors::ComputeDescriptorFacts> {
        let _span = self.watch("compute_descriptor_facts");
        self.inner.compute_descriptor_facts(library, spec)
    }

    fn tile_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::tile_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::tile_descriptors::TileDescriptorFacts> {
        let _span = self.watch("tile_descriptor_facts");
        self.inner.tile_descriptor_facts(library, spec)
    }

    fn pass_descriptor_facts(
        &mut self,
        spec: &crate::pass_descriptors::PassSpec,
    ) -> Answer<crate::pass_descriptors::PassDescriptorFacts> {
        let _span = self.watch("pass_descriptor_facts");
        self.inner.pass_descriptor_facts(spec)
    }

    fn mesh_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::mesh_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::mesh_descriptors::MeshDescriptorFacts> {
        let _span = self.watch("mesh_descriptor_facts");
        self.inner.mesh_descriptor_facts(library, spec)
    }

    fn heap_buffer_size_and_align(
        &mut self,
        query: &crate::device_queries::HeapBufferQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let _span = self.watch("heap_buffer_size_and_align");
        self.inner.heap_buffer_size_and_align(query)
    }

    fn supports_vertex_amplification_count(&mut self, count: u64) -> Answer<bool> {
        let _span = self.watch("supports_vertex_amplification_count");
        self.inner.supports_vertex_amplification_count(count)
    }

    fn supports_rasterization_rate_map_layers(&mut self, layers: u64) -> Answer<bool> {
        let _span = self.watch("supports_rasterization_rate_map_layers");
        self.inner.supports_rasterization_rate_map_layers(layers)
    }

    fn minimum_linear_texture_alignment(&mut self, format: PixelFormat) -> Answer<u64> {
        let _span = self.watch("minimum_linear_texture_alignment");
        self.inner.minimum_linear_texture_alignment(format)
    }

    fn convert_sparse_pixel_regions(
        &mut self,
        query: &crate::device_queries::RegionQuery,
    ) -> Answer<Vec<crate::device_queries::Box6>> {
        let _span = self.watch("convert_sparse_pixel_regions");
        self.inner.convert_sparse_pixel_regions(query)
    }

    fn convert_sparse_tile_regions(
        &mut self,
        query: &crate::device_queries::RegionQuery,
    ) -> Answer<Vec<crate::device_queries::Box6>> {
        let _span = self.watch("convert_sparse_tile_regions");
        self.inner.convert_sparse_tile_regions(query)
    }

    fn function_facts(
        &mut self,
        library: Handle,
        entry: &str,
        constants: &[crate::function_facts::ConstantValue],
    ) -> Answer<crate::function_facts::FunctionFacts> {
        let _span = self.watch("function_facts");
        self.inner.function_facts(library, entry, constants)
    }

    fn library_facts(&mut self, library: Handle) -> Answer<crate::library_facts::LibraryFacts> {
        let _span = self.watch("library_facts");
        self.inner.library_facts(library)
    }

    fn function_facts_from_descriptor(
        &mut self,
        library: Handle,
        descriptor: &crate::library_facts::FunctionDescriptorSpec,
    ) -> Answer<crate::function_facts::FunctionFacts> {
        let _span = self.watch("function_facts_from_descriptor");
        self.inner
            .function_facts_from_descriptor(library, descriptor)
    }

    fn compute_pipeline_bindings(
        &mut self,
        library: Handle,
        entry: &str,
    ) -> Answer<Vec<BindingFacts>> {
        let _span = self.watch("compute_pipeline_bindings");
        self.inner.compute_pipeline_bindings(library, entry)
    }

    fn archives(
        &mut self,
        program: &crate::archives::Program<'_>,
    ) -> Answer<Vec<crate::archives::Observation>> {
        let _span = self.watch("archives");
        self.inner.archives(program)
    }

    fn compute_pipeline_under_options(
        &mut self,
        library: Handle,
        entry: &str,
        options: u64,
    ) -> Answer<crate::pipeline_options::OptionOutcome> {
        let _span = self.watch("compute_pipeline_under_options");
        self.inner
            .compute_pipeline_under_options(library, entry, options)
    }

    fn visible_function_table(
        &mut self,
        caller: Handle,
        functions: Handle,
        spec: &crate::visible_tables::TableSpec<'_>,
    ) -> Answer<crate::visible_tables::TableOutcome> {
        let _span = self.watch("visible_function_table");
        self.inner.visible_function_table(caller, functions, spec)
    }

    fn render_pipeline_under_options(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
        options: u64,
    ) -> Answer<crate::pipeline_options::OptionOutcome> {
        let _span = self.watch("render_pipeline_under_options");
        self.inner
            .render_pipeline_under_options(library, vertex, fragment, attachment, options)
    }

    fn render_pipeline_functions_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::pipeline_functions::FunctionsDescriptorSpec<'_>,
    ) -> Answer<crate::pipeline_functions::FunctionsDescriptorFacts> {
        let _span = self.watch("render_pipeline_functions_descriptor_facts");
        self.inner
            .render_pipeline_functions_descriptor_facts(library, spec)
    }

    fn constant_data_offset(&mut self, encoder: Handle, index: u32) -> Answer<u64> {
        let _span = self.watch("constant_data_offset");
        self.inner.constant_data_offset(encoder, index)
    }

    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        let _span = self.watch("dispatch");
        self.inner
            .dispatch(pipeline, bindings, threadgroups, threads)
    }

    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        let _span = self.watch("dispatch_threads");
        self.inner
            .dispatch_threads(pipeline, bindings, stage_in, grid, threads)
    }

    fn wait(&mut self, command_buffer: Handle) -> Answer<Value> {
        let _span = self.watch("wait");
        self.inner.wait(command_buffer)
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        let _span = self.watch("read_buffer");
        self.inner.read_buffer(buffer, offset, length)
    }

    fn buffer_texture(
        &mut self,
        request: &crate::buffer_texture::BufferTextureRequest,
    ) -> Answer<crate::buffer_texture::BufferTextureFacts> {
        let _span = self.watch("buffer_texture");
        self.inner.buffer_texture(request)
    }

    fn write_plane(&mut self, target: Handle, bytes: &[u8]) -> Answer<()> {
        let _span = self.watch("write_plane");
        self.inner.write_plane(target, bytes)
    }

    fn read_plane(&mut self, target: Handle) -> Answer<Vec<u8>> {
        let _span = self.watch("read_plane");
        self.inner.read_plane(target)
    }

    fn announced(&mut self) -> Vec<String> {
        self.inner.announced()
    }

    fn indirect_draw_read_back(
        &mut self,
        primitive_type: u64,
        indexed: bool,
    ) -> Answer<IndirectReadBack> {
        let _span = self.watch("indirect_draw_read_back");
        self.inner.indirect_draw_read_back(primitive_type, indexed)
    }

    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        let _span = self.watch("work");
        self.inner.work(work)
    }

    fn fragment_fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let _span = self.watch("fragment_fill");
        self.inner.fragment_fill(fill)
    }

    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        let _span = self.watch("mesh_pipeline");
        self.inner.mesh_pipeline(library, functions)
    }

    fn mesh_pipeline_bindings(
        &mut self,
        library: Handle,
        spec: &MeshReflectionSpec<'_>,
        options: u64,
    ) -> Answer<crate::function_bindings::StageReflection> {
        let _span = self.watch("mesh_pipeline_bindings");
        self.inner.mesh_pipeline_bindings(library, spec, options)
    }

    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        let _span = self.watch("mesh_pipeline_for_commands");
        self.inner.mesh_pipeline_for_commands(library, functions)
    }

    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        let _span = self.watch("render_target");
        self.inner.render_target(width, height)
    }

    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        let _span = self.watch("mesh_pass");
        self.inner.mesh_pass(targets, steps)
    }

    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        let _span = self.watch("mesh_command_buffer");
        self.inner.mesh_command_buffer(commands)
    }

    fn compute_pipeline_for_commands(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: Option<&StageInput>,
    ) -> Answer<Handle> {
        let _span = self.watch("compute_pipeline_for_commands");
        self.inner
            .compute_pipeline_for_commands(library, entry, stage_input)
    }

    fn indirect_command_buffer(
        &mut self,
        descriptor: &IcbDescriptor,
        commands: &[EncodedCommand],
    ) -> Answer<CommandsMade> {
        let _span = self.watch("indirect_command_buffer");
        self.inner.indirect_command_buffer(descriptor, commands)
    }

    fn indirect_commands_size(&mut self, commands: Handle) -> Answer<u64> {
        let _span = self.watch("indirect_commands_size");
        self.inner.indirect_commands_size(commands)
    }

    fn reset_indirect_commands(&mut self, commands: Handle, first: u64, count: u64) -> Answer<()> {
        let _span = self.watch("reset_indirect_commands");
        self.inner.reset_indirect_commands(commands, first, count)
    }

    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        let _span = self.watch("execute_mesh_commands");
        self.inner.execute_mesh_commands(targets, commands, used)
    }

    fn plane_texture(&mut self, buffer: Handle, over: &PlaneTexture) -> Answer<Handle> {
        let _span = self.watch("plane_texture");
        self.inner.plane_texture(buffer, over)
    }

    fn fill_texture(&mut self, texture: Handle, fill: &Fill) -> Answer<Value> {
        let _span = self.watch("fill_texture");
        self.inner.fill_texture(texture, fill)
    }

    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        let _span = self.watch("rate_map");
        self.inner.rate_map(request)
    }

    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        let _span = self.watch("rate_map_facts");
        self.inner.rate_map_facts(map, layers)
    }

    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        let _span = self.watch("copy_rate_map_parameters");
        self.inner.copy_rate_map_parameters(map, into, offset)
    }

    fn rate_map_coordinates(
        &mut self,
        map: Handle,
        probes: &[CoordinateProbe],
    ) -> Answer<Vec<MappedCoordinates>> {
        let _span = self.watch("rate_map_coordinates");
        self.inner.rate_map_coordinates(map, probes)
    }

    fn rate_map_device_is_the_sides(&mut self, map: Handle) -> Answer<bool> {
        let _span = self.watch("rate_map_device_is_the_sides");
        self.inner.rate_map_device_is_the_sides(map)
    }

    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        let _span = self.watch("tile_pipeline");
        self.inner.tile_pipeline(library, spec)
    }

    fn tile_pipeline_bindings(
        &mut self,
        library: Handle,
        spec: &TilePipelineSpec<'_>,
        options: u64,
    ) -> Answer<crate::function_bindings::StageReflection> {
        let _span = self.watch("tile_pipeline_bindings");
        self.inner.tile_pipeline_bindings(library, spec, options)
    }

    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        let _span = self.watch("tile_facts");
        self.inner.tile_facts(state)
    }

    fn pipeline_both_paths(
        &mut self,
        library: Handle,
        form: &crate::pipeline_handlers::Form<'_>,
    ) -> Answer<crate::pipeline_handlers::BothPaths> {
        let _span = self.watch("pipeline_both_paths");
        self.inner.pipeline_both_paths(library, form)
    }

    fn compute_pipeline_reflection_under(
        &mut self,
        library: Handle,
        entry: &str,
        options: u64,
    ) -> Answer<crate::pipeline_option_reflection::UnderOptions> {
        let _span = self.watch("compute_pipeline_reflection_under");
        self.inner
            .compute_pipeline_reflection_under(library, entry, options)
    }

    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        let _span = self.watch("tile_pass");
        self.inner.tile_pass(targets, tile, steps)
    }

    fn compute_pass(
        &mut self,
        pass: &crate::compute_pass::ComputePass<'_>,
    ) -> Answer<crate::compute_pass::ComputePassOutcome> {
        let _span = self.watch("compute_pass");
        self.inner.compute_pass(pass)
    }

    fn heap_texture_size_and_align(
        &mut self,
        query: &crate::texture_query::HeapTextureQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let _span = self.watch("heap_texture_size_and_align");
        self.inner.heap_texture_size_and_align(query)
    }

    fn makes_texture(&mut self, query: &crate::texture_query::HeapTextureQuery) -> Answer<bool> {
        let _span = self.watch("makes_texture");
        self.inner.makes_texture(query)
    }

    fn sparse_tile_size(
        &mut self,
        query: &crate::sparse_query::SparseTileQuery,
    ) -> Answer<crate::sparse_query::TileSize> {
        let _span = self.watch("sparse_tile_size");
        self.inner.sparse_tile_size(query)
    }

    fn sparse_mapping(
        &mut self,
        program: &crate::sparse_mapping::SparseMappingProgram,
    ) -> Answer<crate::sparse_mapping::SparseMapped> {
        let _span = self.watch("sparse_mapping");
        self.inner.sparse_mapping(program)
    }

    fn command_buffer_facts(&mut self) -> Answer<crate::command_buffer_facts::CommandBufferFacts> {
        let _span = self.watch("command_buffer_facts");
        self.inner.command_buffer_facts()
    }

    fn sparse_tile_size_in_bytes(&mut self, page: crate::sparse_query::QueryPage) -> Answer<u64> {
        let _span = self.watch("sparse_tile_size_in_bytes");
        self.inner.sparse_tile_size_in_bytes(page)
    }

    fn sparse_heap(&mut self, spec: &crate::sparse_query::SparseHeapSpec) -> Answer<Handle> {
        let _span = self.watch("sparse_heap");
        self.inner.sparse_heap(spec)
    }

    fn heap_facts(&mut self, heap: Handle) -> Answer<crate::sparse_query::HeapFacts> {
        let _span = self.watch("heap_facts");
        self.inner.heap_facts(heap)
    }

    fn heap_texture(
        &mut self,
        heap: Handle,
        spec: &crate::sparse_query::SparseTextureSpec,
    ) -> Answer<Handle> {
        let _span = self.watch("heap_texture");
        self.inner.heap_texture(heap, spec)
    }

    fn sparse_texture_facts(
        &mut self,
        texture: Handle,
    ) -> Answer<crate::sparse_query::SparseTextureFacts> {
        let _span = self.watch("sparse_texture_facts");
        self.inner.sparse_texture_facts(texture)
    }

    fn queue_slots(
        &mut self,
        max: Option<u32>,
        cap: u32,
    ) -> Answer<crate::queue_slots::QueueSlots> {
        let _span = self.watch("queue_slots");
        self.inner.queue_slots(max, cap)
    }

    fn read_texture_over(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        before: &[u8],
    ) -> Answer<Vec<u8>> {
        let _span = self.watch("read_texture_over");
        self.inner
            .read_texture_over(texture, size, row_pitch, before)
    }

    fn render_pipeline_with_reflection(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
    ) -> Answer<Handle> {
        let _span = self.watch("render_pipeline_with_reflection");
        self.inner
            .render_pipeline_with_reflection(library, vertex, fragment, attachment)
    }

    fn texture_accesses(
        &mut self,
        pipeline: Handle,
        stage: RenderStages,
        indices: &[u64],
    ) -> Answer<Vec<Option<BindingAccess>>> {
        let _span = self.watch("texture_accesses");
        self.inner.texture_accesses(pipeline, stage, indices)
    }

    fn target(&mut self, spec: &crate::render::TargetSpec) -> Answer<Handle> {
        let _span = self.watch("target");
        self.inner.target(spec)
    }

    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &crate::render::RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let _span = self.watch("render_pipeline");
        self.inner.render_pipeline(library, spec)
    }

    fn depth_stencil_state(&mut self, spec: &DepthStencilDescriptor) -> Answer<Handle> {
        let _span = self.watch("depth_stencil_state");
        self.inner.depth_stencil_state(spec)
    }

    fn render(&mut self, work: &[crate::render::RenderWork]) -> Answer<PassOutcome> {
        let _span = self.watch("render");
        self.inner.render(work)
    }

    fn handler_wait(&mut self, wait: crate::handler_waits::OwnWait) -> Answer<Value> {
        let _span = self.watch("handler_wait");
        self.inner.handler_wait(wait)
    }

    fn detach_in_a_completed_handler(
        &mut self,
        buffer: Handle,
        after: std::time::Duration,
    ) -> Answer<crate::host_detach::DetachedInHandler> {
        let _span = self.watch("detach_in_a_completed_handler");
        self.inner.detach_in_a_completed_handler(buffer, after)
    }

    fn shared_event_handle_round_trip(
        &mut self,
        label: &str,
    ) -> Answer<crate::shared_events::EventHandleRoundTrip> {
        let _span = self.watch("shared_event_handle_round_trip");
        self.inner.shared_event_handle_round_trip(label)
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        let _span = self.watch("shared_event_rises");
        self.inner.shared_event_rises(label)
    }

    fn event_cross_queue_wait(&mut self) -> Answer<crate::event_orders::CrossQueueWait> {
        let _span = self.watch("event_cross_queue_wait");
        self.inner.event_cross_queue_wait()
    }

    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        let _span = self.watch("event_highest_signal");
        self.inner.event_highest_signal(values)
    }

    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        let _span = self.watch("held_command_buffer");
        self.inner.held_command_buffer()
    }

    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        let _span = self.watch("signal_before_a_wait");
        self.inner.signal_before_a_wait()
    }

    fn scheduled_handler_order(
        &mut self,
    ) -> Answer<crate::scheduled_handlers::ScheduledHandlerOrder> {
        let _span = self.watch("scheduled_handler_order");
        self.inner.scheduled_handler_order()
    }

    fn heap(&mut self, spec: &crate::heap_query::HeapSpec) -> Answer<Handle> {
        let _span = self.watch("heap");
        self.inner.heap(spec)
    }

    fn heap_own_facts(&mut self, heap: Handle) -> Answer<crate::heap_query::HeapOwnFacts> {
        let _span = self.watch("heap_own_facts");
        self.inner.heap_own_facts(heap)
    }

    fn heap_max_available_size(&mut self, heap: Handle, alignment: u64) -> Answer<u64> {
        let _span = self.watch("heap_max_available_size");
        self.inner.heap_max_available_size(heap, alignment)
    }

    fn set_heap_label(&mut self, heap: Handle, label: &str) -> Answer<()> {
        let _span = self.watch("set_heap_label");
        self.inner.set_heap_label(heap, label)
    }

    fn heap_purgeable_state(
        &mut self,
        heap: Handle,
        state: metal2vulkan_runtime::PurgeableState,
    ) -> Answer<u64> {
        let _span = self.watch("heap_purgeable_state");
        self.inner.heap_purgeable_state(heap, state)
    }

    fn heap_buffer(
        &mut self,
        heap: Handle,
        spec: &crate::heap_query::HeapBufferSpec,
    ) -> Answer<Handle> {
        let _span = self.watch("heap_buffer");
        self.inner.heap_buffer(heap, spec)
    }

    fn resource_facts(
        &mut self,
        of: crate::heap_query::QueriedResource,
        heap: Option<Handle>,
    ) -> Answer<crate::heap_query::ResourceFacts> {
        let _span = self.watch("resource_facts");
        self.inner.resource_facts(of, heap)
    }

    fn set_resource_label(
        &mut self,
        of: crate::heap_query::QueriedResource,
        label: &str,
    ) -> Answer<()> {
        let _span = self.watch("set_resource_label");
        self.inner.set_resource_label(of, label)
    }

    fn make_resource_aliasable(&mut self, of: crate::heap_query::QueriedResource) -> Answer<()> {
        let _span = self.watch("make_resource_aliasable");
        self.inner.make_resource_aliasable(of)
    }

    fn resource_purgeable_state(
        &mut self,
        of: crate::heap_query::QueriedResource,
        state: metal2vulkan_runtime::PurgeableState,
    ) -> Answer<u64> {
        let _span = self.watch("resource_purgeable_state");
        self.inner.resource_purgeable_state(of, state)
    }

    fn texture_view(
        &mut self,
        of: Handle,
        request: &crate::texture_views::ViewRequest,
    ) -> Answer<Handle> {
        let _span = self.watch("texture_view");
        self.inner.texture_view(of, request)
    }

    fn texture_view_facts(
        &mut self,
        texture: Handle,
        parent: Option<Handle>,
    ) -> Answer<crate::texture_views::ViewFacts> {
        let _span = self.watch("texture_view_facts");
        self.inner.texture_view_facts(texture, parent)
    }

    fn shared_texture(&mut self, query: &crate::texture_query::HeapTextureQuery) -> Answer<Handle> {
        let _span = self.watch("shared_texture");
        self.inner.shared_texture(query)
    }

    fn shared_texture_handle(
        &mut self,
        texture: Handle,
    ) -> Answer<crate::texture_views::SharedHandleFacts> {
        let _span = self.watch("shared_texture_handle");
        self.inner.shared_texture_handle(texture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::Failure;

    /// A shader source no side reads: the stub answers without looking at it.
    fn stub_source() -> ShaderSource<'static> {
        ShaderSource {
            bitcode: &[],
            air_text: "",
            entry: "main",
        }
    }

    /// A side that answers nothing and records which calls it was asked for: enough to prove the
    /// wrapper forwards a call, opens exactly one span for it, and changes neither the answer nor
    /// the order.
    #[derive(Default)]
    struct Stub {
        asked: Vec<&'static str>,
    }

    impl Side for Stub {
        fn venue(&self) -> Venue {
            Venue {
                device: "stub".to_owned(),
                driver: "stub".to_owned(),
                api: "stub".to_owned(),
                metal_level: "stub".to_owned(),
            }
        }

        fn library(&mut self, _source: &ShaderSource<'_>) -> Answer<Handle> {
            self.asked.push("library");
            Ok(Handle(7))
        }

        fn compute_pipeline(&mut self, _library: Handle, _entry: &str) -> Answer<Handle> {
            self.asked.push("compute_pipeline");
            Err(Failure::named("Stub", "nothing here".to_owned()))
        }

        fn compute_pipeline_with_stage_input(
            &mut self,
            _library: Handle,
            _entry: &str,
            _stage_input: &StageInput,
        ) -> Answer<Handle> {
            self.asked.push("compute_pipeline_with_stage_input");
            Ok(Handle(0))
        }

        fn buffer(&mut self, _bytes: &[u8]) -> Answer<Handle> {
            self.asked.push("buffer");
            Ok(Handle(1))
        }

        fn zeroed_buffer(&mut self, _length: usize) -> Answer<Handle> {
            self.asked.push("zeroed_buffer");
            Ok(Handle(2))
        }

        fn texture_2d(
            &mut self,
            _format: PixelFormat,
            _size: [u32; 2],
            _row_pitch: u32,
            _bytes: &[u8],
        ) -> Answer<Handle> {
            self.asked.push("texture_2d");
            Ok(Handle(3))
        }

        fn read_texture(
            &mut self,
            _texture: Handle,
            _size: [u32; 2],
            _row_pitch: u32,
        ) -> Answer<Vec<u8>> {
            self.asked.push("read_texture");
            Ok(Vec::new())
        }

        fn dispatch(
            &mut self,
            _pipeline: Handle,
            _bindings: &Bindings<'_>,
            _threadgroups: [u32; 3],
            _threads: [u32; 3],
        ) -> Answer<Handle> {
            self.asked.push("dispatch");
            Ok(Handle(4))
        }

        fn dispatch_threads(
            &mut self,
            _pipeline: Handle,
            _bindings: &Bindings<'_>,
            _stage_in: &[StageInRegion],
            _grid: [u32; 3],
            _threads: [u32; 3],
        ) -> Answer<Handle> {
            self.asked.push("dispatch_threads");
            Ok(Handle(5))
        }

        fn wait(&mut self, _command_buffer: Handle) -> Answer<Value> {
            self.asked.push("wait");
            Ok(Value::Uint(0))
        }

        fn read_buffer(
            &mut self,
            _buffer: Handle,
            _offset: u64,
            _length: usize,
        ) -> Answer<Vec<u8>> {
            self.asked.push("read_buffer");
            Ok(Vec::new())
        }

        fn announced(&mut self) -> Vec<String> {
            self.asked.push("announced");
            Vec::new()
        }
    }

    /// One span per call, named for the call, in the order the calls were made — and the answers
    /// are the wrapped side's own, including a typed error.
    ///
    /// The last part is what makes the wrapper safe to put under a measurement: a wrapper that
    /// swallowed an error would turn a refusal into a cheap call and the cost table would read it
    /// as good news.
    #[test]
    fn a_span_is_opened_for_each_call_and_the_answer_is_the_sides_own() {
        let timer = crate::times::Timer::new();
        let mut side = Watching::with(Stub::default(), timer.clone());
        assert_eq!(side.library(&stub_source()), Ok(Handle(7)));
        assert!(side.compute_pipeline(Handle(7), "main").is_err());
        assert_eq!(side.buffer(&[1, 2, 3]), Ok(Handle(1)));
        assert_eq!(side.wait(Handle(0)), Ok(Value::Uint(0)));
        let observed = timer.take();
        let names: Vec<&str> = observed.iter().map(|one| one.call).collect();
        assert_eq!(names, ["library", "compute_pipeline", "buffer", "wait"]);
        // The refused call is measured like any other: a refusal has a cost and hiding it would
        // make a gap look free.
        assert_eq!(observed.len(), 4);
    }

    /// `venue` and `announced` are forwarded with **no** span, because neither is a call to the
    /// implementation under test. Timing them would put the harness's own bookkeeping in the
    /// table, and `announced` has no Metal counterpart at all.
    #[test]
    fn the_two_calls_that_are_not_the_implementations_open_no_span() {
        let timer = crate::times::Timer::new();
        let mut side = Watching::with(Stub::default(), timer.clone());
        let _ = side.venue();
        let _ = side.announced();
        assert!(
            timer.is_empty(),
            "venue or announced opened a span: {:?}",
            timer.take()
        );
    }

    /// The span closes when the call returns, not before: a call that takes a known time is
    /// measured as at least that long.
    ///
    /// Without this the wrapper could record a zero for every call and every table would read as
    /// free — the one failure of a timing layer that looks like success.
    #[test]
    fn a_span_covers_the_call_rather_than_closing_early() {
        /// A side whose one call sleeps, so the span has something to measure.
        struct Slow(std::time::Duration);
        impl Side for Slow {
            fn venue(&self) -> Venue {
                Venue {
                    device: "slow".to_owned(),
                    driver: "slow".to_owned(),
                    api: "slow".to_owned(),
                    metal_level: "slow".to_owned(),
                }
            }
            fn library(&mut self, _source: &ShaderSource<'_>) -> Answer<Handle> {
                std::thread::sleep(self.0);
                Ok(Handle(0))
            }
            fn compute_pipeline(&mut self, _library: Handle, _entry: &str) -> Answer<Handle> {
                Err(Failure::unavailable("compute_pipeline"))
            }
            fn compute_pipeline_with_stage_input(
                &mut self,
                _library: Handle,
                _entry: &str,
                _stage_input: &StageInput,
            ) -> Answer<Handle> {
                Err(Failure::unavailable("compute_pipeline_with_stage_input"))
            }
            fn buffer(&mut self, _bytes: &[u8]) -> Answer<Handle> {
                Err(Failure::unavailable("buffer"))
            }
            fn zeroed_buffer(&mut self, _length: usize) -> Answer<Handle> {
                Err(Failure::unavailable("zeroed_buffer"))
            }
            fn texture_2d(
                &mut self,
                _format: PixelFormat,
                _size: [u32; 2],
                _row_pitch: u32,
                _bytes: &[u8],
            ) -> Answer<Handle> {
                Err(Failure::unavailable("texture_2d"))
            }
            fn read_texture(
                &mut self,
                _texture: Handle,
                _size: [u32; 2],
                _row_pitch: u32,
            ) -> Answer<Vec<u8>> {
                Err(Failure::unavailable("read_texture"))
            }
            fn dispatch(
                &mut self,
                _pipeline: Handle,
                _bindings: &Bindings<'_>,
                _threadgroups: [u32; 3],
                _threads: [u32; 3],
            ) -> Answer<Handle> {
                Err(Failure::unavailable("dispatch"))
            }
            fn dispatch_threads(
                &mut self,
                _pipeline: Handle,
                _bindings: &Bindings<'_>,
                _stage_in: &[StageInRegion],
                _grid: [u32; 3],
                _threads: [u32; 3],
            ) -> Answer<Handle> {
                Err(Failure::unavailable("dispatch_threads"))
            }
            fn wait(&mut self, _command_buffer: Handle) -> Answer<Value> {
                Err(Failure::unavailable("wait"))
            }
            fn read_buffer(
                &mut self,
                _buffer: Handle,
                _offset: u64,
                _length: usize,
            ) -> Answer<Vec<u8>> {
                Err(Failure::unavailable("read_buffer"))
            }
            fn announced(&mut self) -> Vec<String> {
                Vec::new()
            }
        }
        let slept = std::time::Duration::from_millis(20);
        let timer = crate::times::Timer::new();
        let mut side = Watching::with(Slow(slept), timer.clone());
        let _ = side.library(&stub_source());
        let observed = timer.take();
        let [one] = observed.as_slice() else {
            panic!("one call, one observation: {observed:?}");
        };
        let slept = u64::try_from(slept.as_nanos()).unwrap();
        assert!(
            one.wall >= slept,
            "the span read {} ns for a call that slept {slept} ns, so it closed before the call \
             returned",
            one.wall
        );
        // Asleep is not on the CPU, so nearly all of it must land in `blocked` rather than in the
        // caller's own CPU time. That is the attribution the whole design turns on.
        assert!(
            one.blocked() > one.on_cpu,
            "a sleeping call read {} ns on cpu and {} ns blocked",
            one.on_cpu,
            one.blocked()
        );
    }
}
