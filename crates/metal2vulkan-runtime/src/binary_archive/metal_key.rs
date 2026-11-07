//! The key Metal's binary archives find a pipeline by, so that
//! `PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS` refuses exactly where Metal's does.
//!
//! Measured on an M3 Max (macOS 26.7) and an M4 (macOS 27.0.1), which agree on every field: a
//! creation hits an archive a descriptor was added to unless a keyed field differs.
//!
//! - **Functions.** A function is its name and its bitcode's content (a library made again from
//!   the same bytes hits), and the value of each constant it declares, however supplied (by name
//!   or by index alike; a value it does not declare changes nothing; a constant without a value
//!   differs from one with). Its linked functions are keyed as sets (their order changes
//!   nothing, an empty set is none), each list apart: `functions`, `private_functions`,
//!   `binary_functions`, and each group by name.
//! - **Compute.** The kernel; the stage input descriptor in full (an empty one is none; every
//!   attribute and layout, used or not, and the index buffer's index, but not the index type);
//!   `max_total_threads_per_threadgroup`; `support_indirect_command_buffers`;
//!   `support_adding_binary_functions`; `required_threads_per_threadgroup` (an all-zero size is
//!   none); the buffers set `Mutable`. Not `threadgroup_size_is_multiple_of_execution_width`, nor
//!   the label.
//! - **Tile.** The tile function, apart from a compute key of the same function; each color
//!   attachment with a pixel format (its index and format); the raster sample count;
//!   `threadgroup_size_matches_tile_size`; `max_total_threads_per_threadgroup`;
//!   `support_adding_binary_functions`; `required_threads_per_threadgroup`; the buffers set
//!   `Mutable`. Not the label.
//! - **Buffer mutability**. A stage's key holds the index of each buffer set `Mutable`, whether its
//!   function writes, reads or does not use it, in compute, tile, vertex and fragment stages alike.
//!   `Immutable` and the default change nothing.
//! - **Render: each stage apart.** A render creation looks up its fragment stage's key, then its
//!   vertex stage's, each in every archive listed, and misses naming the first function not
//!   found: so two adds' stages combine. The fragment key: the fragment function, the vertex
//!   descriptor in full (an empty one is none), each color attachment with a pixel format (its
//!   index, format and write mask, whether it blends, and its blend factors and operations only
//!   when it blends), the raster sample count, alpha to coverage and to one, and
//!   `support_indirect_command_buffers`, `support_adding_fragment_binary_functions` and the
//!   fragment buffers set `Mutable`. The vertex key: the vertex function, the vertex
//!   descriptor, the input primitive topology class (line and triangle alike), whether
//!   rasterization is enabled, whether a fragment stage reads the vertex outputs, the vertex
//!   amplification count, `support_indirect_command_buffers`, the tessellation partition mode
//!   and factor step function, `support_adding_vertex_binary_functions` and the vertex buffers
//!   set `Mutable`. The control point index type is in both keys, only for a
//!   post-tessellation vertex function. Neither holds the depth and stencil formats, the maximum
//!   tessellation factor, factor scale or winding order, nor the label; a creation without a
//!   fragment function looks up no fragment key, so its color attachments change nothing.
//! - **Which vertex outputs the fragment stage reads.** Metal's vertex key holds the outputs its
//!   compiler finds the fragment reading: none for a fragment that reads only its position or
//!   nothing, or whose color the write mask or blend factors discard, the same set for two
//!   fragments reading the same outputs. A lookup here does not decode, so the key holds whether
//!   a fragment stage reads at all (a fragment function under enabled rasterization). That is
//!   Metal's answer whenever both stages come from one add. An add also keeps its two stages'
//!   pair key (`RenderKeys::pair`), so a lookup that finds both stages but no add's pair of them
//!   combines two adds' stages; that creation announces `combined-archive-stages`, since Metal
//!   misses it where the two fragments read different outputs. Where the key's answer makes the
//!   runtime miss and Metal hit, the creation is refused.
//! - **Mesh** (not measured: no probe has asked which of a mesh descriptor's fields Metal keys).
//!   One key of every field the descriptor holds: the object, mesh and fragment stages, each as a
//!   stage (its function, constants and linked functions), each color attachment with a pixel
//!   format as the render fragment key holds it, the depth and stencil formats, the raster
//!   sample count, rasterization, alpha to coverage and to one, the texture write rounding mode,
//!   `support_indirect_command_buffers`, both threadgroup maxima, the mesh-grid maximum, both
//!   multiple-of-execution-width flags, the payload length, the vertex amplification count and
//!   both required threadgroup sizes. So a creation hits only an add of an equal descriptor,
//!   where Metal's hits too, and one that differs from every add in a field Metal does not key
//!   misses here where Metal's may hit.
//! - **Dynamic libraries** (probes, an M3 Max and an M4). Neither a stage's preloaded libraries
//!   (used or not, in any order, of any contents or install name) nor the contents of the libraries
//!   its install names load are in its key: a dependency's file replaced hits, and the creation
//!   links the new file's code. A function's bitcode names the install names it was linked against,
//!   so they are in its content.
//! - **The rest.** Metal's descriptor fields the compile layer's descriptors do not have yet
//!   (call stack depths, shader validation) never reach a key.
//!
//! Every key also holds the profile it is for and the artifact and semantics versions, as an
//! archive made on another device, or by a build whose translations differ, misses on Metal's
//! GPUs (an M4's archive missed on the M3).
//!
//! Whether a vertex function is a post-tessellation one shows only once it is decoded, and a
//! lookup must not decode. So each render stage has two keys, without and with the control point
//! index type, and an archive keeps the ones that fit its vertex function, which it learns from
//! the compile its add makes. The vertex key holds the function's identity, so the variant an
//! archive holds of it tells a lookup which of the fragment's keys to look for.

use metal2vulkan::cache::{Hash256, ARTIFACT_FORMAT_VERSION, SEMANTICS_VERSION};
use metal2vulkan::library::{Function, FunctionConstantValues};
use metal2vulkan::metal::{
    AttributeFormat, Mutability, PixelFormat, PrimitiveTopologyClass, VertexFormat,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, LinkedFunction, LinkedFunctions,
    MeshPipelineDescriptor, RenderPipelineDescriptor, StageInputAttributeDescriptor,
    StageInputDescriptor, StageInputLayoutDescriptor, StageSource, TileRenderPipelineDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Whether a vertex function is a post-tessellation one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VertexKind {
    Plain,
    Patch,
}

/// A render stage's two keys, by the kind of the vertex function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StageKeys {
    /// The key beside a vertex function that is not a post-tessellation one.
    pub(crate) plain: Hash256,
    /// The key beside a post-tessellation vertex function: `plain`'s fields and the control
    /// point index type.
    pub(crate) patch: Hash256,
}

impl StageKeys {
    /// The key beside a vertex function of `kind`.
    pub(crate) fn of(&self, kind: VertexKind) -> Hash256 {
        match kind {
            VertexKind::Plain => self.plain,
            VertexKind::Patch => self.patch,
        }
    }

    /// The kind whose key `held` holds, if either's.
    pub(crate) fn held(&self, held: &dyn Fn(&Hash256) -> bool) -> Option<VertexKind> {
        [VertexKind::Plain, VertexKind::Patch]
            .into_iter()
            .find(|kind| held(&self.of(*kind)))
    }
}

/// A render descriptor's stage keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RenderKeys {
    /// The fragment stage's keys, when there is a fragment function.
    pub(crate) fragment: Option<StageKeys>,
    /// The vertex stage's keys.
    pub(crate) vertex: StageKeys,
}

impl RenderKeys {
    /// The key of both stages of one add beside a vertex function of `kind`, when there is a
    /// fragment function: an add keeps it, so a lookup whose two stages were found tells one
    /// add's stages from two adds' (which vertex outputs the fragment reads, the module docs).
    pub(crate) fn pair(&self, kind: VertexKind) -> Option<Hash256> {
        let fragment = self.fragment?;
        let mut key = Key(Sha256::new());
        key.bytes(b"metal2vulkan binary archive render pair 1");
        key.0.update(self.vertex.of(kind));
        key.0.update(fragment.of(kind));
        Some(key.finish())
    }
}

/// The archive key of `descriptor` on the device of profile `profile`.
pub(crate) fn compute(profile: &Hash256, descriptor: &ComputePipelineDescriptor) -> Hash256 {
    let mut key = Key::new(b"compute", profile);
    key.stage(&descriptor.kernel);
    match &descriptor.stage_input {
        Some(input) if *input != StageInputDescriptor::new() => {
            key.flag(true);
            stage_input(&mut key, input);
        }
        _ => key.flag(false),
    }
    key.u32(descriptor.max_total_threads_per_threadgroup.unwrap_or(0));
    key.flag(descriptor.support_indirect_command_buffers);
    key.flag(descriptor.support_adding_binary_functions);
    key.required(descriptor.required_threads_per_threadgroup);
    key.mutable(&descriptor.buffers);
    key.finish()
}

/// The archive key of the tile pipeline `descriptor` on the device of profile `profile`.
pub(crate) fn tile(profile: &Hash256, descriptor: &TileRenderPipelineDescriptor) -> Hash256 {
    let mut key = Key::new(b"tile", profile);
    key.stage(&descriptor.tile);
    for (index, format) in (0u32..).zip(&descriptor.color_attachment_formats) {
        match format {
            Some(format) if *format != PixelFormat::Invalid => {
                key.u32(index);
                key.u32(format.as_raw());
            }
            _ => {}
        }
    }
    key.u32(u32::MAX);
    key.u32(descriptor.raster_sample_count);
    key.flag(descriptor.threadgroup_size_matches_tile_size);
    key.u32(descriptor.max_total_threads_per_threadgroup.unwrap_or(0));
    key.flag(descriptor.support_adding_binary_functions);
    key.required(descriptor.required_threads_per_threadgroup);
    key.mutable(&descriptor.buffers);
    key.finish()
}

/// The archive keys of `descriptor`'s stages on the device of profile `profile`.
pub(crate) fn render(profile: &Hash256, descriptor: &RenderPipelineDescriptor) -> RenderKeys {
    let tessellation = descriptor.tessellation.unwrap_or_default();
    let control_points = tessellation.control_point_index_type.as_raw();
    let fragment = descriptor.fragment.as_ref().map(|fragment| {
        let mut key = Key::new(b"render fragment", profile);
        key.stage(fragment);
        vertex_descriptor(&mut key, descriptor.vertex_descriptor.as_ref());
        color_attachments(&mut key, &descriptor.color_attachments);
        key.u32(descriptor.raster_sample_count);
        key.flag(descriptor.alpha_to_coverage);
        key.flag(descriptor.alpha_to_one);
        key.flag(descriptor.support_indirect_command_buffers);
        key.flag(descriptor.support_adding_fragment_binary_functions);
        key.mutable(&descriptor.fragment_buffers);
        key.variants(control_points)
    });
    let mut key = Key::new(b"render vertex", profile);
    key.stage(&descriptor.vertex);
    vertex_descriptor(&mut key, descriptor.vertex_descriptor.as_ref());
    // Measured: a line and a triangle class make one key; no class, points and the two differ.
    let class = match descriptor.input_primitive_topology {
        PrimitiveTopologyClass::Triangle => PrimitiveTopologyClass::Line,
        class => class,
    };
    key.u32(class.as_raw());
    key.flag(descriptor.rasterization_enabled);
    key.flag(descriptor.fragment.is_some() && descriptor.rasterization_enabled);
    key.u32(descriptor.max_vertex_amplification_count);
    key.flag(descriptor.support_indirect_command_buffers);
    key.u32(tessellation.partition_mode.as_raw());
    key.u32(tessellation.factor_step_function.as_raw());
    key.flag(descriptor.support_adding_vertex_binary_functions);
    key.mutable(&descriptor.vertex_buffers);
    RenderKeys {
        fragment,
        vertex: key.variants(control_points),
    }
}

/// The archive key of the mesh pipeline `descriptor` on the device of profile `profile`: every
/// field it holds (the module docs).
pub(crate) fn mesh(profile: &Hash256, descriptor: &MeshPipelineDescriptor) -> Hash256 {
    let mut key = Key::new(b"mesh", profile);
    let stages = [
        descriptor.object.as_ref(),
        Some(&descriptor.mesh),
        descriptor.fragment.as_ref(),
    ];
    for stage in stages {
        match stage {
            Some(stage) => {
                key.flag(true);
                key.stage(stage);
            }
            None => key.flag(false),
        }
    }
    color_attachments(&mut key, &descriptor.color_attachments);
    for format in [descriptor.depth_format, descriptor.stencil_format] {
        key.u32(format.map_or(u32::MAX, PixelFormat::as_raw));
    }
    key.u32(descriptor.raster_sample_count);
    key.flag(descriptor.rasterization_enabled);
    key.flag(descriptor.alpha_to_coverage);
    key.flag(descriptor.alpha_to_one);
    // The mode by its name: the enum states no raw value of its own.
    key.bytes(format!("{:?}", descriptor.texture_write_rounding_mode).as_bytes());
    key.flag(descriptor.support_indirect_command_buffers);
    for limit in [
        descriptor.max_total_threads_per_object_threadgroup,
        descriptor.max_total_threads_per_mesh_threadgroup,
        descriptor.max_total_threadgroups_per_mesh_grid,
        descriptor.payload_memory_length,
    ] {
        key.u32(limit.unwrap_or(0));
    }
    key.flag(descriptor.object_threadgroup_size_is_multiple_of_thread_execution_width);
    key.flag(descriptor.mesh_threadgroup_size_is_multiple_of_thread_execution_width);
    key.u32(descriptor.max_vertex_amplification_count);
    key.required(descriptor.required_threads_per_object_threadgroup);
    key.required(descriptor.required_threads_per_mesh_threadgroup);
    key.finish()
}

/// Each color attachment of `attachments` with a pixel format: its index, format and write mask,
/// whether it blends, and its blend factors and operations only when it blends.
fn color_attachments(key: &mut Key, attachments: &[Option<ColorAttachmentDescriptor>]) {
    for (index, attachment) in (0u32..).zip(attachments) {
        let Some(attachment) = attachment else {
            continue;
        };
        if attachment.pixel_format == PixelFormat::Invalid {
            continue;
        }
        key.u32(index);
        key.u32(attachment.pixel_format.as_raw());
        key.u32(attachment.write_mask.as_raw());
        key.flag(attachment.blending_enabled);
        if attachment.blending_enabled {
            for raw in [
                attachment.source_rgb_blend_factor.as_raw(),
                attachment.destination_rgb_blend_factor.as_raw(),
                attachment.rgb_blend_operation.as_raw(),
                attachment.source_alpha_blend_factor.as_raw(),
                attachment.destination_alpha_blend_factor.as_raw(),
                attachment.alpha_blend_operation.as_raw(),
            ] {
                key.u32(raw);
            }
        }
    }
    key.u32(u32::MAX);
}

/// The stage input descriptor `input`: each layout but Metal's unset one (stride 0, per vertex,
/// step rate 1, which Metal's serializer leaves out) and each set attribute, then the index
/// buffer's index.
fn stage_input(key: &mut Key, input: &StageInputDescriptor) {
    let unset_layout = StageInputLayoutDescriptor::new(Some(0));
    let unset_attribute = StageInputAttributeDescriptor::new(AttributeFormat::Invalid, 0, 0);
    for (index, layout) in &input.layouts {
        if *layout != unset_layout {
            key.u32(*index);
            match layout.stride {
                Some(stride) => {
                    key.flag(true);
                    key.u32(stride);
                }
                None => key.flag(false),
            }
            key.u32(layout.step_function.as_raw());
            key.u32(layout.step_rate);
        }
    }
    key.u32(u32::MAX);
    for (index, attribute) in &input.attributes {
        if *attribute != unset_attribute {
            key.u32(*index);
            key.u32(attribute.format.as_raw());
            key.u32(attribute.offset);
            key.u32(attribute.buffer_index);
        }
    }
    key.u32(u32::MAX);
    key.u32(input.index_buffer_index);
}

/// The vertex descriptor `vertex`, an empty one as none.
fn vertex_descriptor(key: &mut Key, vertex: Option<&VertexDescriptor>) {
    let Some(vertex) = vertex.filter(|vertex| **vertex != VertexDescriptor::new()) else {
        key.flag(false);
        return;
    };
    key.flag(true);
    let unset_layout = VertexBufferLayoutDescriptor::new(Some(0));
    let unset_attribute = VertexAttributeDescriptor::new(VertexFormat::Invalid, 0, 0);
    for (index, layout) in &vertex.layouts {
        if *layout != unset_layout {
            key.u32(*index);
            match layout.stride {
                Some(stride) => {
                    key.flag(true);
                    key.u32(stride);
                }
                None => key.flag(false),
            }
            key.u32(layout.step_function.as_raw());
            key.u32(layout.step_rate);
        }
    }
    key.u32(u32::MAX);
    for (index, attribute) in &vertex.attributes {
        if *attribute != unset_attribute {
            key.u32(*index);
            key.u32(attribute.format.as_raw());
            key.u32(attribute.offset);
            key.u32(attribute.buffer_index);
        }
    }
    key.u32(u32::MAX);
}

/// A key being written: SHA-256 over a fixed-width, length-prefixed encoding.
#[derive(Clone)]
struct Key(Sha256);

impl Key {
    /// A key of pipeline `kind` for `profile`, under this build's artifact and semantics
    /// versions.
    fn new(kind: &[u8], profile: &Hash256) -> Key {
        let mut key = Key(Sha256::new());
        key.bytes(b"metal2vulkan binary archive key 1");
        key.bytes(kind);
        key.0.update(profile);
        key.u32(ARTIFACT_FORMAT_VERSION);
        key.u32(SEMANTICS_VERSION);
        key
    }

    fn finish(self) -> Hash256 {
        self.0.finalize().into()
    }

    /// The stage keys of what is written: as it is, and with the control point index type
    /// `control_points` appended.
    fn variants(self, control_points: u32) -> StageKeys {
        let mut patch = self.clone();
        patch.u32(control_points);
        StageKeys {
            plain: self.finish(),
            patch: patch.finish(),
        }
    }

    fn u32(&mut self, value: u32) {
        self.0.update(value.to_le_bytes());
    }

    fn flag(&mut self, value: bool) {
        self.0.update([u8::from(value)]);
    }

    /// The indices of a stage's buffers set `Mutable`: Metal keys those alone, whether the function
    /// writes, reads or does not use the buffer, and keys neither `Immutable` nor the default.
    fn mutable(&mut self, buffers: &BTreeMap<u32, Mutability>) {
        for (index, mutability) in buffers {
            if *mutability == Mutability::Mutable {
                self.u32(*index);
            }
        }
        self.u32(u32::MAX);
    }

    /// A required threadgroup size (`requiredThreadsPerThreadgroup`), or none: Metal's all-zero
    /// size is none.
    fn required(&mut self, required: Option<[u32; 3]>) {
        match required.filter(|axes| *axes != [0; 3]) {
            Some(axes) => {
                self.flag(true);
                for axis in axes {
                    self.u32(axis);
                }
            }
            None => self.flag(false),
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }

    /// A stage: its function, constant values and linked functions, or the compile key of a
    /// function compiled already.
    fn stage(&mut self, source: &StageSource) {
        match source {
            StageSource::Function(descriptor) => {
                self.flag(true);
                self.function(&descriptor.function, &descriptor.constants);
                self.linked(&descriptor.linked);
            }
            StageSource::Compiled(compiled) => {
                self.flag(false);
                self.0.update(compiled.key().digest());
            }
        }
    }

    /// `function`'s name and content, and the value of each constant it declares.
    fn function(&mut self, function: &Function, values: &FunctionConstantValues) {
        let info = function.info();
        self.bytes(info.name.as_bytes());
        self.0.update(info.air_hash);
        for constant in info.constants {
            let value = constant
                .index
                .and_then(|index| values.by_index.get(&index))
                .or_else(|| values.by_name.get(&constant.name));
            match value {
                Some(value) => {
                    self.flag(true);
                    self.u32(value.ty.as_raw());
                    self.bytes(&value.bytes);
                }
                None => self.flag(false),
            }
        }
    }

    /// Each list of `linked` as a set.
    fn linked(&mut self, linked: &LinkedFunctions) {
        if linked.is_empty() {
            self.flag(false);
            return;
        }
        self.flag(true);
        self.set(&linked.functions);
        self.set(&linked.private_functions);
        self.set(&linked.binary_functions);
        self.u32(linked.groups.len() as u32);
        for (name, functions) in &linked.groups {
            self.bytes(name.as_bytes());
            self.set(functions);
        }
    }

    /// `functions` in no order: each function's own key, sorted.
    fn set(&mut self, functions: &[LinkedFunction]) {
        let mut members: Vec<Hash256> = functions
            .iter()
            .map(|linked| {
                let mut member = Key(Sha256::new());
                member.function(&linked.function, &linked.constants);
                member.finish()
            })
            .collect();
        members.sort_unstable();
        self.u32(members.len() as u32);
        for member in members {
            self.0.update(member);
        }
    }
}
