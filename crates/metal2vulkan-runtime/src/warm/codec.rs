//! A pipeline record's bytes: the descriptor of a pipeline state's creation, field by field, its
//! libraries named by their content hash.
//!
//! ```text
//! magic "M2VPREC\0" | RECORD_FORMAT u32 | body | SHA-256 of all before it
//! body: libraries (count u32, each: content hash, length u64) | kind u8 | the kind's fields
//! ```
//!
//! - **Every field.** Each descriptor field is written in declaration order: integers little
//!   endian, a `bool` as one byte 0 or 1, an enum as its raw Metal value (`as_raw`, read back
//!   through its checked `TryFrom<u32>`), an `Option` as a byte 0 or 1 and the value, a list or a
//!   map as a `u32` count and its entries in order. A function is its library's place in the
//!   record's list and its index in the library's table. Bytes of an unknown enum value, a
//!   `bool` other than 0 or 1, a count past the bytes left, or bytes left over are no record.
//! - **What cannot be written.** A stage the compile layer already compiled
//!   (`StageSource::Compiled`), and a value of a non-exhaustive enum this codec does not know,
//!   make no record ([`Unwritable`]): the creation is made, unrecorded, and counted.
//! - **Checked at the making.** A record is read back the moment it is made, from its own bytes
//!   with the creation's own libraries, and kept only where what it reads equals the creation's
//!   descriptor (`PipelineRecord::of`): a field this codec missed makes no record, rather than a
//!   record of another pipeline.
//! - **Versioned by its format.** [`RECORD_FORMAT`] moves whenever these bytes change meaning (a
//!   field added, removed or re-typed); a record of another format is not read. A record holds
//!   inputs, no translation, so no translator or runtime version decides whether it is read.

use metal2vulkan::cache::Hash256;
use metal2vulkan::library::{
    DynamicLibrary, Function, FunctionConstantValue, FunctionConstantValues, Library,
};
use metal2vulkan::metal;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, DynamicLibraries, FunctionDescriptor,
    LinkedFunction, LinkedFunctions, MeshPipelineDescriptor, PrivateRenderState,
    RenderPipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource, TessellationDescriptor, TileRenderPipelineDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

/// The first bytes of a record.
pub(crate) const MAGIC: &[u8; 8] = b"M2VPREC\0";

/// The record format: bumped whenever a record's bytes change meaning (the module docs).
pub(crate) const RECORD_FORMAT: u32 = 1;

/// The length of a record's header (magic and format) and of its checksum.
const HEADER: usize = 8 + 4;
const CHECKSUM: usize = 32;

/// What a record describes: one pipeline state creation's descriptor, and how many of its
/// stages' last binary functions were added without the stage's support flag
/// (`new_*_pipeline_state_with_additional_binary_functions`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Pipeline {
    Compute(ComputePipelineDescriptor, usize),
    /// The vertex stage's count, then the fragment stage's.
    Render(RenderPipelineDescriptor, [usize; 2]),
    Mesh(MeshPipelineDescriptor),
    Tile(TileRenderPipelineDescriptor, usize),
}

impl Pipeline {
    /// The kind's byte.
    fn kind(&self) -> u8 {
        match self {
            Pipeline::Compute(..) => 1,
            Pipeline::Render(..) => 2,
            Pipeline::Mesh(..) => 3,
            Pipeline::Tile(..) => 4,
        }
    }
}

/// A creation this codec cannot write (the module docs).
#[derive(Debug)]
pub(crate) struct Unwritable;

/// Bytes that are no record of this format: damaged, of another format, or not a record.
#[derive(Debug)]
pub(crate) struct Unreadable;

/// `pipeline`'s record bytes, and the libraries they name, in the order their places number them.
pub(crate) fn encode(pipeline: &Pipeline) -> Result<(Vec<u8>, Vec<Library>), Unwritable> {
    let mut w = Writer::default();
    w.u8(pipeline.kind());
    match pipeline {
        Pipeline::Compute(d, added) => {
            w.compute(d)?;
            w.usize(*added);
        }
        Pipeline::Render(d, [vertex, fragment]) => {
            w.render(d)?;
            w.usize(*vertex);
            w.usize(*fragment);
        }
        Pipeline::Mesh(d) => w.mesh(d)?,
        Pipeline::Tile(d, added) => {
            w.tile(d)?;
            w.usize(*added);
        }
    }
    let Writer {
        out: fields,
        libraries,
        ..
    } = w;
    let mut out = Vec::with_capacity(HEADER + 4 + libraries.len() * 40 + fields.len() + CHECKSUM);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&RECORD_FORMAT.to_le_bytes());
    out.extend_from_slice(&count(libraries.len())?.to_le_bytes());
    for library in &libraries {
        out.extend_from_slice(library.content_hash());
        out.extend_from_slice(&(library.bytes().len() as u64).to_le_bytes());
    }
    out.extend_from_slice(&fields);
    let checksum: Hash256 = Sha256::digest(&out).into();
    out.extend_from_slice(&checksum);
    Ok((out, libraries))
}

/// Whether `bytes` are a whole record of this format: its magic, its format and its checksum.
pub(crate) fn is_record(bytes: &[u8]) -> bool {
    if bytes.len() < HEADER + CHECKSUM || &bytes[..8] != MAGIC {
        return false;
    }
    if bytes[8..HEADER] != RECORD_FORMAT.to_le_bytes() {
        return false;
    }
    let (signed, checksum) = bytes.split_at(bytes.len() - CHECKSUM);
    <[u8; 32]>::from(Sha256::digest(signed)) == checksum
}

/// What `bytes`, a record ([`is_record`]), describe, with `libraries` the libraries its table
/// names, in its order ([`libraries`]).
pub(crate) fn decode(bytes: &[u8], libraries: &[Library]) -> Result<Pipeline, Unreadable> {
    let mut r = Reader::body(bytes, libraries)?;
    let table = r.library_table()?;
    if table.len() != libraries.len()
        || table
            .iter()
            .zip(libraries)
            .any(|((hash, length), library)| {
                hash != library.content_hash() || *length != library.bytes().len() as u64
            })
    {
        return Err(Unreadable);
    }
    let pipeline = match r.u8()? {
        1 => Pipeline::Compute(r.compute()?, r.usize()?),
        2 => Pipeline::Render(r.render()?, [r.usize()?, r.usize()?]),
        3 => Pipeline::Mesh(r.mesh()?),
        4 => Pipeline::Tile(r.tile()?, r.usize()?),
        _ => return Err(Unreadable),
    };
    match r.at == r.bytes.len() {
        true => Ok(pipeline),
        false => Err(Unreadable),
    }
}

/// The libraries `bytes`, a record ([`is_record`]), names, each by its content hash and its
/// container's length, in their places' order: what [`decode`] is then handed.
pub(crate) fn libraries(bytes: &[u8]) -> Result<Vec<(Hash256, u64)>, Unreadable> {
    Reader::body(bytes, &[])?.library_table()
}

/// `n` as a record's count, where it fits.
fn count(n: usize) -> Result<u32, Unwritable> {
    u32::try_from(n).map_err(|_| Unwritable)
}

/// The encoder's state: the bytes of the fields so far, and the libraries they name.
#[derive(Default)]
struct Writer {
    out: Vec<u8>,
    libraries: Vec<Library>,
    places: HashMap<Hash256, u32>,
}

impl Writer {
    fn u8(&mut self, v: u8) {
        self.out.push(v);
    }

    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }

    fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }

    fn bytes(&mut self, v: &[u8]) -> Result<(), Unwritable> {
        self.u32(count(v.len())?);
        self.out.extend_from_slice(v);
        Ok(())
    }

    fn opt_u32(&mut self, v: Option<u32>) {
        self.bool(v.is_some());
        if let Some(v) = v {
            self.u32(v);
        }
    }

    fn threads(&mut self, v: Option<[u32; 3]>) {
        self.bool(v.is_some());
        for n in v.into_iter().flatten() {
            self.u32(n);
        }
    }

    fn pixel_format(&mut self, v: Option<metal::PixelFormat>) {
        self.opt_u32(v.map(metal::PixelFormat::as_raw));
    }

    fn mutability(&mut self, v: &BTreeMap<u32, metal::Mutability>) -> Result<(), Unwritable> {
        self.u32(count(v.len())?);
        for (index, mutability) in v {
            self.u32(*index);
            self.u32(mutability.as_raw());
        }
        Ok(())
    }

    fn rounding(&mut self, v: metal::TextureWriteRoundingMode) -> Result<(), Unwritable> {
        self.u32(match v {
            metal::TextureWriteRoundingMode::TowardZero => 0,
            metal::TextureWriteRoundingMode::NearestEven => 2,
            _ => return Err(Unwritable),
        });
        Ok(())
    }

    /// The place of `library` in the record's list, given one at its first use.
    fn library(&mut self, library: &Library) -> Result<u32, Unwritable> {
        if let Some(&place) = self.places.get(library.content_hash()) {
            return Ok(place);
        }
        let place = count(self.libraries.len())?;
        self.places.insert(*library.content_hash(), place);
        self.libraries.push(library.clone());
        Ok(place)
    }

    fn function(&mut self, f: &Function) -> Result<(), Unwritable> {
        let place = self.library(f.library())?;
        self.u32(place);
        self.u32(f.index());
        Ok(())
    }

    fn constant(&mut self, v: &FunctionConstantValue) -> Result<(), Unwritable> {
        self.u32(v.ty.as_raw());
        self.bytes(&v.bytes)
    }

    fn constants(&mut self, v: &FunctionConstantValues) -> Result<(), Unwritable> {
        self.u32(count(v.by_index.len())?);
        for (index, value) in &v.by_index {
            self.u32(*index);
            self.constant(value)?;
        }
        self.u32(count(v.by_name.len())?);
        for (name, value) in &v.by_name {
            self.bytes(name.as_bytes())?;
            self.constant(value)?;
        }
        Ok(())
    }

    fn linked_list(&mut self, v: &[LinkedFunction]) -> Result<(), Unwritable> {
        self.u32(count(v.len())?);
        for f in v {
            self.function(&f.function)?;
            self.constants(&f.constants)?;
            self.bool(f.precompiled);
        }
        Ok(())
    }

    fn linked(&mut self, v: &LinkedFunctions) -> Result<(), Unwritable> {
        self.linked_list(&v.functions)?;
        self.linked_list(&v.private_functions)?;
        self.linked_list(&v.binary_functions)?;
        self.u32(count(v.groups.len())?);
        for (name, functions) in &v.groups {
            self.bytes(name.as_bytes())?;
            self.linked_list(functions)?;
        }
        Ok(())
    }

    fn dynamic(&mut self, v: &DynamicLibraries) -> Result<(), Unwritable> {
        self.u32(count(v.preloaded.len())?);
        for library in &v.preloaded {
            let place = self.library(library.library())?;
            self.u32(place);
        }
        self.u32(count(v.loaded.len())?);
        for (name, library) in &v.loaded {
            self.bytes(name.as_bytes())?;
            let place = self.library(library.library())?;
            self.u32(place);
        }
        Ok(())
    }

    fn stage(&mut self, v: &StageSource) -> Result<(), Unwritable> {
        let StageSource::Function(f) = v else {
            return Err(Unwritable);
        };
        self.function(&f.function)?;
        self.constants(&f.constants)?;
        self.linked(&f.linked)?;
        self.dynamic(&f.libraries)
    }

    fn opt_stage(&mut self, v: Option<&StageSource>) -> Result<(), Unwritable> {
        self.bool(v.is_some());
        match v {
            Some(stage) => self.stage(stage),
            None => Ok(()),
        }
    }

    fn colors(&mut self, v: &[Option<ColorAttachmentDescriptor>; 8]) {
        for color in v {
            self.bool(color.is_some());
            if let Some(c) = color {
                self.u32(c.pixel_format.as_raw());
                self.bool(c.blending_enabled);
                self.u32(c.source_rgb_blend_factor.as_raw());
                self.u32(c.destination_rgb_blend_factor.as_raw());
                self.u32(c.rgb_blend_operation.as_raw());
                self.u32(c.source_alpha_blend_factor.as_raw());
                self.u32(c.destination_alpha_blend_factor.as_raw());
                self.u32(c.alpha_blend_operation.as_raw());
                self.u32(c.write_mask.as_raw());
            }
        }
    }

    fn vertex_descriptor(&mut self, v: Option<&VertexDescriptor>) -> Result<(), Unwritable> {
        self.bool(v.is_some());
        let Some(v) = v else {
            return Ok(());
        };
        self.u32(count(v.layouts.len())?);
        for (index, layout) in &v.layouts {
            self.u32(*index);
            self.opt_u32(layout.stride);
            self.u32(layout.step_function.as_raw());
            self.u32(layout.step_rate);
        }
        self.u32(count(v.attributes.len())?);
        for (index, attribute) in &v.attributes {
            self.u32(*index);
            self.u32(attribute.format.as_raw());
            self.u32(attribute.offset);
            self.u32(attribute.buffer_index);
        }
        Ok(())
    }

    fn stage_input(&mut self, v: Option<&StageInputDescriptor>) -> Result<(), Unwritable> {
        self.bool(v.is_some());
        let Some(v) = v else {
            return Ok(());
        };
        self.u32(count(v.layouts.len())?);
        for (index, layout) in &v.layouts {
            self.u32(*index);
            self.opt_u32(layout.stride);
            self.u32(layout.step_function.as_raw());
            self.u32(layout.step_rate);
        }
        self.u32(count(v.attributes.len())?);
        for (index, attribute) in &v.attributes {
            self.u32(*index);
            self.u32(attribute.format.as_raw());
            self.u32(attribute.offset);
            self.u32(attribute.buffer_index);
        }
        self.u32(v.index_type.as_raw());
        self.u32(v.index_buffer_index);
        Ok(())
    }

    fn tessellation(&mut self, v: Option<&TessellationDescriptor>) {
        self.bool(v.is_some());
        if let Some(t) = v {
            self.u32(t.max_tessellation_factor);
            self.u32(t.partition_mode.as_raw());
            self.u32(t.factor_format.as_raw());
            self.u32(t.factor_step_function.as_raw());
            self.u32(t.output_winding_order.as_raw());
            self.u32(t.control_point_index_type.as_raw());
            self.bool(t.factor_scale_enabled);
        }
    }

    fn private(&mut self, p: &PrivateRenderState) {
        self.u64(p.sample_mask);
        self.u32(p.sample_coverage.to_bits());
        self.bool(p.sample_coverage_invert);
        self.bool(p.alpha_test_enabled);
        self.u64(p.alpha_test_function);
        self.bool(p.logic_operation_enabled);
        self.u64(p.logic_operation);
        self.u8(p.clip_distance_enable_mask);
        self.bool(p.point_smooth_enabled);
        self.bool(p.point_coord_lower_left);
        self.bool(p.point_size_output_vs);
        self.bool(p.two_side_enabled);
        self.u32(p.vertex_depth_compare_clamp_mask);
        self.u32(p.fragment_depth_compare_clamp_mask);
        self.bool(p.depth_stencil_write_disabled);
        self.bool(p.needs_custom_border_color_samplers);
        self.u64(p.vertex_amplification_mode);
        self.u32(p.explicit_visibility_group_id);
    }

    fn compute(&mut self, d: &ComputePipelineDescriptor) -> Result<(), Unwritable> {
        self.stage(&d.kernel)?;
        self.stage_input(d.stage_input.as_ref())?;
        self.opt_u32(d.max_total_threads_per_threadgroup);
        self.bool(d.threadgroup_size_is_multiple_of_execution_width);
        self.bool(d.support_indirect_command_buffers);
        self.u32(d.max_call_stack_depth);
        self.threads(d.required_threads_per_threadgroup);
        self.bool(d.support_adding_binary_functions);
        self.mutability(&d.buffers)?;
        self.bool(d.needs_custom_border_color_samplers);
        Ok(())
    }

    fn render(&mut self, d: &RenderPipelineDescriptor) -> Result<(), Unwritable> {
        self.stage(&d.vertex)?;
        self.opt_stage(d.fragment.as_ref())?;
        self.vertex_descriptor(d.vertex_descriptor.as_ref())?;
        self.colors(&d.color_attachments);
        self.pixel_format(d.depth_format);
        self.pixel_format(d.stencil_format);
        self.u32(d.raster_sample_count);
        self.bool(d.alpha_to_coverage);
        self.bool(d.alpha_to_one);
        self.bool(d.rasterization_enabled);
        self.u32(d.input_primitive_topology.as_raw());
        self.u32(d.max_vertex_amplification_count);
        self.tessellation(d.tessellation.as_ref());
        self.bool(d.support_indirect_command_buffers);
        self.u32(d.max_vertex_call_stack_depth);
        self.u32(d.max_fragment_call_stack_depth);
        self.bool(d.support_adding_vertex_binary_functions);
        self.bool(d.support_adding_fragment_binary_functions);
        self.mutability(&d.vertex_buffers)?;
        self.mutability(&d.fragment_buffers)?;
        self.rounding(d.texture_write_rounding_mode)?;
        self.private(&d.private);
        Ok(())
    }

    fn mesh(&mut self, d: &MeshPipelineDescriptor) -> Result<(), Unwritable> {
        self.opt_stage(d.object.as_ref())?;
        self.stage(&d.mesh)?;
        self.opt_stage(d.fragment.as_ref())?;
        self.colors(&d.color_attachments);
        self.pixel_format(d.depth_format);
        self.pixel_format(d.stencil_format);
        self.u32(d.raster_sample_count);
        self.bool(d.rasterization_enabled);
        self.bool(d.alpha_to_coverage);
        self.bool(d.alpha_to_one);
        self.bool(d.support_indirect_command_buffers);
        self.rounding(d.texture_write_rounding_mode)
    }

    fn tile(&mut self, d: &TileRenderPipelineDescriptor) -> Result<(), Unwritable> {
        self.stage(&d.tile)?;
        for format in d.color_attachment_formats {
            self.pixel_format(format);
        }
        self.u32(d.raster_sample_count);
        self.bool(d.threadgroup_size_matches_tile_size);
        self.opt_u32(d.max_total_threads_per_threadgroup);
        self.u32(d.max_call_stack_depth);
        self.threads(d.required_threads_per_threadgroup);
        self.bool(d.support_adding_binary_functions);
        self.mutability(&d.buffers)
    }
}

/// The decoder's state: a record's body, how far it has read, and the libraries its places name.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    libraries: &'a [Library],
}

impl<'a> Reader<'a> {
    /// A reader of the body of `record`, a whole record ([`is_record`]), with its libraries.
    fn body(record: &'a [u8], libraries: &'a [Library]) -> Result<Reader<'a>, Unreadable> {
        if !is_record(record) {
            return Err(Unreadable);
        }
        Ok(Reader {
            bytes: &record[HEADER..record.len() - CHECKSUM],
            at: 0,
            libraries,
        })
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Unreadable> {
        let end = self.at.checked_add(n).ok_or(Unreadable)?;
        let taken = self.bytes.get(self.at..end).ok_or(Unreadable)?;
        self.at = end;
        Ok(taken)
    }

    fn u8(&mut self) -> Result<u8, Unreadable> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Unreadable> {
        let bytes = self.take(4)?.try_into().map_err(|_| Unreadable)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, Unreadable> {
        let bytes = self.take(8)?.try_into().map_err(|_| Unreadable)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn usize(&mut self) -> Result<usize, Unreadable> {
        usize::try_from(self.u64()?).map_err(|_| Unreadable)
    }

    fn bool(&mut self) -> Result<bool, Unreadable> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Unreadable),
        }
    }

    /// A count of entries each at least `least` bytes long: past the bytes left is no record.
    fn count(&mut self, least: usize) -> Result<usize, Unreadable> {
        let n = self.u32()? as usize;
        match n.saturating_mul(least) <= self.bytes.len() - self.at {
            true => Ok(n),
            false => Err(Unreadable),
        }
    }

    fn bytes(&mut self) -> Result<&'a [u8], Unreadable> {
        let n = self.count(1)?;
        self.take(n)
    }

    fn string(&mut self) -> Result<String, Unreadable> {
        let bytes = self.bytes()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Unreadable)
    }

    fn opt_u32(&mut self) -> Result<Option<u32>, Unreadable> {
        match self.bool()? {
            true => Ok(Some(self.u32()?)),
            false => Ok(None),
        }
    }

    fn threads(&mut self) -> Result<Option<[u32; 3]>, Unreadable> {
        match self.bool()? {
            true => Ok(Some([self.u32()?, self.u32()?, self.u32()?])),
            false => Ok(None),
        }
    }

    /// A Metal enum or option set, by its checked `TryFrom<u32>`.
    fn raw<T: TryFrom<u32>>(&mut self) -> Result<T, Unreadable> {
        T::try_from(self.u32()?).map_err(|_| Unreadable)
    }

    fn pixel_format(&mut self) -> Result<Option<metal::PixelFormat>, Unreadable> {
        match self.bool()? {
            true => Ok(Some(self.raw()?)),
            false => Ok(None),
        }
    }

    fn mutability(&mut self) -> Result<BTreeMap<u32, metal::Mutability>, Unreadable> {
        let n = self.count(8)?;
        let mut out = BTreeMap::new();
        for _ in 0..n {
            let index = self.u32()?;
            out.insert(index, self.raw()?);
        }
        Ok(out)
    }

    fn rounding(&mut self) -> Result<metal::TextureWriteRoundingMode, Unreadable> {
        match self.u32()? {
            0 => Ok(metal::TextureWriteRoundingMode::TowardZero),
            2 => Ok(metal::TextureWriteRoundingMode::NearestEven),
            _ => Err(Unreadable),
        }
    }

    fn library_table(&mut self) -> Result<Vec<(Hash256, u64)>, Unreadable> {
        let n = self.count(40)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let hash: Hash256 = self.take(32)?.try_into().map_err(|_| Unreadable)?;
            out.push((hash, self.u64()?));
        }
        Ok(out)
    }

    fn library(&mut self) -> Result<Library, Unreadable> {
        let place = self.u32()? as usize;
        self.libraries.get(place).cloned().ok_or(Unreadable)
    }

    fn function(&mut self) -> Result<Function, Unreadable> {
        let library = self.library()?;
        let index = self.u32()?;
        library.function_at(index).map_err(|_| Unreadable)
    }

    fn constant(&mut self) -> Result<FunctionConstantValue, Unreadable> {
        let ty: metal::DataType = self.raw()?;
        Ok(FunctionConstantValue::new(ty, self.bytes()?.to_vec()))
    }

    fn constants(&mut self) -> Result<FunctionConstantValues, Unreadable> {
        let mut out = FunctionConstantValues::defaults();
        for _ in 0..self.count(8)? {
            let index = self.u32()?;
            out = out.with_index(index, self.constant()?);
        }
        for _ in 0..self.count(8)? {
            let name = self.string()?;
            out = out.with_name(name, self.constant()?);
        }
        Ok(out)
    }

    fn linked_list(&mut self) -> Result<Vec<LinkedFunction>, Unreadable> {
        let n = self.count(17)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let function = self.function()?;
            let constants = self.constants()?;
            let precompiled = self.bool()?;
            out.push(
                LinkedFunction::new(function)
                    .with_constants(constants)
                    .with_precompiled(precompiled),
            );
        }
        Ok(out)
    }

    fn linked(&mut self) -> Result<LinkedFunctions, Unreadable> {
        let mut out = LinkedFunctions::new();
        out.functions = self.linked_list()?;
        out.private_functions = self.linked_list()?;
        out.binary_functions = self.linked_list()?;
        for _ in 0..self.count(8)? {
            let name = self.string()?;
            out.groups.insert(name, self.linked_list()?);
        }
        Ok(out)
    }

    fn dynamic_library(&mut self) -> Result<DynamicLibrary, Unreadable> {
        DynamicLibrary::loaded(self.library()?).map_err(|_| Unreadable)
    }

    fn dynamic(&mut self) -> Result<DynamicLibraries, Unreadable> {
        let mut out = DynamicLibraries::new();
        for _ in 0..self.count(4)? {
            out.preloaded.push(self.dynamic_library()?);
        }
        for _ in 0..self.count(8)? {
            let name = self.string()?;
            out.loaded.insert(name, self.dynamic_library()?);
        }
        Ok(out)
    }

    fn stage(&mut self) -> Result<StageSource, Unreadable> {
        let mut f = FunctionDescriptor::new(self.function()?);
        f.constants = self.constants()?;
        f.linked = self.linked()?;
        f.libraries = self.dynamic()?;
        Ok(StageSource::Function(f))
    }

    fn opt_stage(&mut self) -> Result<Option<StageSource>, Unreadable> {
        match self.bool()? {
            true => Ok(Some(self.stage()?)),
            false => Ok(None),
        }
    }

    fn colors(&mut self) -> Result<[Option<ColorAttachmentDescriptor>; 8], Unreadable> {
        let mut out: [Option<ColorAttachmentDescriptor>; 8] = Default::default();
        for slot in &mut out {
            if !self.bool()? {
                continue;
            }
            let mut c = ColorAttachmentDescriptor::new(self.raw()?);
            c.blending_enabled = self.bool()?;
            c.source_rgb_blend_factor = self.raw()?;
            c.destination_rgb_blend_factor = self.raw()?;
            c.rgb_blend_operation = self.raw()?;
            c.source_alpha_blend_factor = self.raw()?;
            c.destination_alpha_blend_factor = self.raw()?;
            c.alpha_blend_operation = self.raw()?;
            c.write_mask = self.raw()?;
            *slot = Some(c);
        }
        Ok(out)
    }

    fn vertex_descriptor(&mut self) -> Result<Option<VertexDescriptor>, Unreadable> {
        if !self.bool()? {
            return Ok(None);
        }
        let mut v = VertexDescriptor::new();
        for _ in 0..self.count(13)? {
            let index = self.u32()?;
            let mut layout = VertexBufferLayoutDescriptor::new(self.opt_u32()?);
            layout.step_function = self.raw()?;
            layout.step_rate = self.u32()?;
            v.layouts.insert(index, layout);
        }
        for _ in 0..self.count(16)? {
            let index = self.u32()?;
            let attribute = VertexAttributeDescriptor::new(self.raw()?, self.u32()?, self.u32()?);
            v.attributes.insert(index, attribute);
        }
        Ok(Some(v))
    }

    fn stage_input(&mut self) -> Result<Option<StageInputDescriptor>, Unreadable> {
        if !self.bool()? {
            return Ok(None);
        }
        let mut v = StageInputDescriptor::new();
        for _ in 0..self.count(13)? {
            let index = self.u32()?;
            let mut layout = StageInputLayoutDescriptor::new(self.opt_u32()?);
            layout.step_function = self.raw()?;
            layout.step_rate = self.u32()?;
            v.layouts.insert(index, layout);
        }
        for _ in 0..self.count(16)? {
            let index = self.u32()?;
            let attribute =
                StageInputAttributeDescriptor::new(self.raw()?, self.u32()?, self.u32()?);
            v.attributes.insert(index, attribute);
        }
        v.index_type = self.raw()?;
        v.index_buffer_index = self.u32()?;
        Ok(Some(v))
    }

    fn tessellation(&mut self) -> Result<Option<TessellationDescriptor>, Unreadable> {
        if !self.bool()? {
            return Ok(None);
        }
        let mut t = TessellationDescriptor::default();
        t.max_tessellation_factor = self.u32()?;
        t.partition_mode = self.raw()?;
        t.factor_format = self.raw()?;
        t.factor_step_function = self.raw()?;
        t.output_winding_order = self.raw()?;
        t.control_point_index_type = self.raw()?;
        t.factor_scale_enabled = self.bool()?;
        Ok(Some(t))
    }

    fn private(&mut self) -> Result<PrivateRenderState, Unreadable> {
        let mut p = PrivateRenderState::new();
        p.sample_mask = self.u64()?;
        p.sample_coverage = f32::from_bits(self.u32()?);
        p.sample_coverage_invert = self.bool()?;
        p.alpha_test_enabled = self.bool()?;
        p.alpha_test_function = self.u64()?;
        p.logic_operation_enabled = self.bool()?;
        p.logic_operation = self.u64()?;
        p.clip_distance_enable_mask = self.u8()?;
        p.point_smooth_enabled = self.bool()?;
        p.point_coord_lower_left = self.bool()?;
        p.point_size_output_vs = self.bool()?;
        p.two_side_enabled = self.bool()?;
        p.vertex_depth_compare_clamp_mask = self.u32()?;
        p.fragment_depth_compare_clamp_mask = self.u32()?;
        p.depth_stencil_write_disabled = self.bool()?;
        p.needs_custom_border_color_samplers = self.bool()?;
        p.vertex_amplification_mode = self.u64()?;
        p.explicit_visibility_group_id = self.u32()?;
        Ok(p)
    }

    fn compute(&mut self) -> Result<ComputePipelineDescriptor, Unreadable> {
        let mut d = ComputePipelineDescriptor::new(self.stage()?);
        d.stage_input = self.stage_input()?;
        d.max_total_threads_per_threadgroup = self.opt_u32()?;
        d.threadgroup_size_is_multiple_of_execution_width = self.bool()?;
        d.support_indirect_command_buffers = self.bool()?;
        d.max_call_stack_depth = self.u32()?;
        d.required_threads_per_threadgroup = self.threads()?;
        d.support_adding_binary_functions = self.bool()?;
        d.buffers = self.mutability()?;
        d.needs_custom_border_color_samplers = self.bool()?;
        Ok(d)
    }

    fn render(&mut self) -> Result<RenderPipelineDescriptor, Unreadable> {
        let mut d = RenderPipelineDescriptor::new(self.stage()?);
        d.fragment = self.opt_stage()?;
        d.vertex_descriptor = self.vertex_descriptor()?;
        d.color_attachments = self.colors()?;
        d.depth_format = self.pixel_format()?;
        d.stencil_format = self.pixel_format()?;
        d.raster_sample_count = self.u32()?;
        d.alpha_to_coverage = self.bool()?;
        d.alpha_to_one = self.bool()?;
        d.rasterization_enabled = self.bool()?;
        d.input_primitive_topology = self.raw()?;
        d.max_vertex_amplification_count = self.u32()?;
        d.tessellation = self.tessellation()?;
        d.support_indirect_command_buffers = self.bool()?;
        d.max_vertex_call_stack_depth = self.u32()?;
        d.max_fragment_call_stack_depth = self.u32()?;
        d.support_adding_vertex_binary_functions = self.bool()?;
        d.support_adding_fragment_binary_functions = self.bool()?;
        d.vertex_buffers = self.mutability()?;
        d.fragment_buffers = self.mutability()?;
        d.texture_write_rounding_mode = self.rounding()?;
        d.private = self.private()?;
        Ok(d)
    }

    fn mesh(&mut self) -> Result<MeshPipelineDescriptor, Unreadable> {
        let object = self.opt_stage()?;
        let mut d = MeshPipelineDescriptor::new(self.stage()?);
        d.object = object;
        d.fragment = self.opt_stage()?;
        d.color_attachments = self.colors()?;
        d.depth_format = self.pixel_format()?;
        d.stencil_format = self.pixel_format()?;
        d.raster_sample_count = self.u32()?;
        d.rasterization_enabled = self.bool()?;
        d.alpha_to_coverage = self.bool()?;
        d.alpha_to_one = self.bool()?;
        d.support_indirect_command_buffers = self.bool()?;
        d.texture_write_rounding_mode = self.rounding()?;
        Ok(d)
    }

    fn tile(&mut self) -> Result<TileRenderPipelineDescriptor, Unreadable> {
        let mut d = TileRenderPipelineDescriptor::new(self.stage()?);
        for format in &mut d.color_attachment_formats {
            *format = self.pixel_format()?;
        }
        d.raster_sample_count = self.u32()?;
        d.threadgroup_size_matches_tile_size = self.bool()?;
        d.max_total_threads_per_threadgroup = self.opt_u32()?;
        d.max_call_stack_depth = self.u32()?;
        d.required_threads_per_threadgroup = self.threads()?;
        d.support_adding_binary_functions = self.bool()?;
        d.buffers = self.mutability()?;
        Ok(d)
    }
}
