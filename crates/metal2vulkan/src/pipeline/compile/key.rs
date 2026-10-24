//! Compile keys.
//!
//! A key is SHA-256 over a domain tag, [`SEMANTICS_VERSION`], and a length-prefixed encoding of
//! exactly the inputs that determine the output, so equal keys mean equal output. Keys never need
//! decoded AIR:
//!
//! - [`function_key`] (`FunctionCompileKey`): the function's AIR hash, its entry name, the
//!   function-constant values **as supplied** (by index and by name), the profile fingerprint
//!   (policy included), the canonical lowering inputs ([`canonical_inputs`]), and, when the stage
//!   links functions, each linked function's list, name, AIR hash, and constant values. A stage
//!   without linked functions keeps the key it had before linking existed, and one compiled without
//!   `emulated_simdgroups`, `dropped_vertex_layer`, `storage_view_slots`,
//!   `signed_color_attachments`, `multisampled`, `replay_bindings`, or `framebuffer_fetch` the key
//!   it had before those inputs existed. A stage that links dynamic libraries records each library
//!   its imports are searched in, in order (install name and content hash), and the install name
//!   not loaded; one that links none keeps its key.
//! - [`pipeline_key`] (`PipelineKey`): the profile fingerprint, each stage's `FunctionCompileKey`
//!   in stage order, and every descriptor field the artifact depends on (its fixed-function
//!   state included, since the artifact carries it).

use sha2::{Digest, Sha256};

use crate::cache::{Hash256, PipelineKey, SEMANTICS_VERSION};
use crate::library::{FunctionCompileKey, FunctionConstantValues};
use crate::lower::select::LowerInputs;
use crate::pipeline::descriptor::{
    LinkedList, MeshPipelineDescriptor, PrivateRenderState, RenderPipelineDescriptor,
    StageInputDescriptor, StageSource, TessellationDescriptor,
};

/// What a compile key records of one linked function: the list it is in, its name, its AIR hash,
/// and the constant values it was specialized with.
#[derive(Clone, Copy, Debug)]
pub struct LinkedKey<'a> {
    pub list: LinkedList<'a>,
    pub name: &'a str,
    pub air_hash: Hash256,
    pub constants: &'a FunctionConstantValues,
}

/// What a compile key records of the dynamic libraries a stage links: each library its imports are
/// searched in, in order, by install name and content hash, and the first install name no library
/// was loaded for.
#[derive(Clone, Debug, Default)]
pub struct DynamicKey<'a> {
    pub libraries: Vec<(&'a str, &'a Hash256)>,
    pub unloaded: Option<&'a str>,
}

/// A key hasher with length-prefixed fields.
pub(crate) struct KeyHasher(Sha256);

impl KeyHasher {
    /// A hasher for keys of `domain`.
    pub(crate) fn new(domain: &[u8]) -> KeyHasher {
        let mut h = Sha256::new();
        h.update(domain);
        h.update([0]);
        h.update(SEMANTICS_VERSION.to_le_bytes());
        KeyHasher(h)
    }

    /// A length-prefixed byte field.
    pub(crate) fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.0.update((b.len() as u64).to_le_bytes());
        self.0.update(b);
        self
    }

    /// A fixed-width field.
    pub(crate) fn u64(&mut self, v: u64) -> &mut Self {
        self.0.update(v.to_le_bytes());
        self
    }

    /// The digest.
    pub(crate) fn finish(self) -> Hash256 {
        self.0.finalize().into()
    }
}

/// `inputs` with an empty vertex-fetch projection written as none: a vertex function that reads
/// no attribute compiles the same with or without a vertex descriptor; and the stage-input
/// descriptor canonical (`StageInputDescriptor::canonical`).
pub fn canonical_inputs(inputs: &LowerInputs) -> LowerInputs {
    let mut out = inputs.clone();
    if out
        .vertex_fetch
        .as_ref()
        .is_some_and(|v| v.attributes.is_empty())
    {
        out.vertex_fetch = None;
    }
    out.stage_input = inputs
        .stage_input
        .as_ref()
        .map(StageInputDescriptor::canonical);
    // A depth of 0 is taken as 1, Metal's default, which is no depth at all.
    out.max_call_stack_depth = inputs.max_call_stack_depth.filter(|&k| k > 1);
    // Metal's default tessellation state is no state at all.
    out.tessellation = inputs
        .tessellation
        .filter(|d| *d != TessellationDescriptor::default());
    // Only a tile function's threadgroup is the tile.
    out.threadgroup_size_matches_tile_size =
        inputs.threadgroup_size_matches_tile_size && inputs.tile_attachments.is_some();
    out
}

/// The encoding of the lowering inputs a key has hashed: the vertex-fetch projection, the
/// storage-image choices, and the device-address slots, written as the derived `Debug` of
/// `LoweringInputs` wrote them previously added a field. The maps and sets are canonical (ordered,
/// of plain enums), so their `Debug` form is a total, deterministic encoding of them, as the
/// profile fingerprint's is of the profile. Later fields are appended by [`function_key`] only when
/// set.
pub(crate) fn inputs_encoding(inputs: &LowerInputs) -> String {
    format!(
        "LoweringInputs {{ vertex_fetch: {:?}, storage_images: {:?}, psb_slots: {:?} }}",
        inputs.vertex_fetch, inputs.storage_images, inputs.psb_slots
    )
}

/// Function-constant values as supplied, by index and by name.
fn constants(h: &mut KeyHasher, consts: &FunctionConstantValues) {
    h.u64(consts.by_index.len() as u64);
    for (index, v) in &consts.by_index {
        h.u64(u64::from(*index))
            .u64(u64::from(v.ty.as_raw()))
            .bytes(&v.bytes);
    }
    h.u64(consts.by_name.len() as u64);
    for (name, v) in &consts.by_name {
        h.bytes(name.as_bytes())
            .u64(u64::from(v.ty.as_raw()))
            .bytes(&v.bytes);
    }
}

/// The `FunctionCompileKey` of entry `entry` of the function whose AIR hash is `air_hash`, with
/// `consts` as supplied, for the profile of fingerprint `profile`, under `inputs`, linking
/// `linked` in their order and searching `dynamic`.
pub fn function_key(
    air_hash: &Hash256,
    entry: &str,
    consts: &FunctionConstantValues,
    profile: &Hash256,
    inputs: &LowerInputs,
    (linked, dynamic): (&[LinkedKey<'_>], &DynamicKey<'_>),
) -> FunctionCompileKey {
    let mut h = KeyHasher::new(b"metal2vulkan/function-compile-key/v2");
    h.bytes(air_hash).bytes(entry.as_bytes());
    constants(&mut h, consts);
    h.bytes(profile);
    let inputs = canonical_inputs(inputs);
    h.bytes(inputs_encoding(&inputs).as_bytes());
    // Appended only when set, so a function compiled without them keeps its key.
    if inputs.emulated_simdgroups {
        h.bytes(b"emulated-simdgroups");
    }
    if inputs.dropped_vertex_layer {
        h.bytes(b"dropped-vertex-layer");
    }
    // Appended only when set, canonical, in its derived `Debug` form (ordered maps of plain enums
    // and integers, a total encoding as `inputs_encoding`'s).
    if let Some(d) = &inputs.stage_input {
        h.bytes(b"stage-input").bytes(format!("{d:?}").as_bytes());
    }
    // Appended only when set, as above.
    if !inputs.storage_view_slots.is_empty() {
        h.bytes(b"storage-view-slots")
            .bytes(format!("{:?}", inputs.storage_view_slots).as_bytes());
    }
    // Appended only when set, as above.
    if !inputs.signed_color_attachments.is_empty() {
        h.bytes(b"signed-color-attachments")
            .bytes(format!("{:?}", inputs.signed_color_attachments).as_bytes());
    }
    // Appended only when set, as above.
    if let Some(t) = &inputs.tile_attachments {
        h.bytes(b"tile-attachments")
            .bytes(format!("{t:?}").as_bytes());
    }
    // Appended only when set.
    if inputs.multisampled {
        h.bytes(b"multisampled");
    }
    // Appended only when set.
    if inputs.replay_bindings {
        h.bytes(b"replay-bindings");
    }
    // Appended only when set.
    if inputs.merged_draws {
        h.bytes(b"merged-draws");
    }
    // Appended only when set, in its derived `Debug` form (ordered maps of plain enums, flags, and
    // booleans, a total encoding as `inputs_encoding`'s).
    if let Some(f) = &inputs.framebuffer_fetch {
        h.bytes(b"framebuffer-fetch")
            .bytes(format!("{f:?}").as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.srgb_textures.is_empty() {
        h.bytes(b"srgb-textures")
            .bytes(format!("{:?}", inputs.srgb_textures).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.srgb_twins.is_empty() {
        h.bytes(b"srgb-twins")
            .bytes(format!("{:?}", inputs.srgb_twins).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.snapshot_textures.is_empty() {
        h.bytes(b"snapshot-textures")
            .bytes(format!("{:?}", inputs.snapshot_textures).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.ycbcr_textures.is_empty() {
        h.bytes(b"ycbcr-textures")
            .bytes(format!("{:?}", inputs.ycbcr_textures).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.extended_range.is_empty() {
        h.bytes(b"extended-range")
            .bytes(format!("{:?}", inputs.extended_range).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.extended_range_colors.is_empty() {
        h.bytes(b"extended-range-colors")
            .bytes(format!("{:?}", inputs.extended_range_colors).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.word_colors.is_empty() {
        h.bytes(b"word-colors")
            .bytes(format!("{:?}", inputs.word_colors).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.srgb_fetches.is_empty() {
        h.bytes(b"srgb-fetches")
            .bytes(format!("{:?}", inputs.srgb_fetches).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.rotated_colors.is_empty() {
        h.bytes(b"rotated-colors")
            .bytes(format!("{:?}", inputs.rotated_colors).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.alpha_in_red_colors.is_empty() {
        h.bytes(b"alpha-in-red-colors")
            .bytes(format!("{:?}", inputs.alpha_in_red_colors).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.rotated_factors.is_empty() {
        h.bytes(b"rotated-factors")
            .bytes(format!("{:?}", inputs.rotated_factors).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.rotated_blends.is_empty() {
        h.bytes(b"rotated-blends")
            .bytes(format!("{:?}", inputs.rotated_blends).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.half_output_lanes.is_empty() {
        h.bytes(b"half-output-lanes")
            .bytes(format!("{:?}", inputs.half_output_lanes).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.swizzled_storage.is_empty() {
        h.bytes(b"swizzled-storage")
            .bytes(format!("{:?}", inputs.swizzled_storage).as_bytes());
    }
    // Appended only when set.
    if inputs.viewport_index_varying {
        h.bytes(b"viewport-index-varying");
    }
    // Appended only when set: canonical, so never Metal's default state.
    if let Some(d) = inputs.tessellation {
        h.bytes(b"tessellation").bytes(format!("{d:?}").as_bytes());
    }
    // Appended only when set: canonical, so never 1, Metal's default.
    if let Some(k) = inputs.max_call_stack_depth {
        h.bytes(b"max-call-stack-depth").u64(u64::from(k));
    }
    // Appended only when set.
    if inputs.flat_grid {
        h.bytes(b"flat-grid");
    }
    // Appended only when set.
    if inputs.render_target_array_index_varying {
        h.bytes(b"render-target-array-index-varying");
    }
    // Appended only when set.
    if inputs.primitive_id_varying {
        h.bytes(b"primitive-id-varying");
    }
    // Appended only when set, with its index type.
    if let Some(ty) = inputs.primitive_id_indices {
        h.bytes(b"primitive-id-indices").u64(u64::from(ty.as_raw()));
    }
    // Appended only when set.
    if inputs.quad_primitive_ids {
        h.bytes(b"quad-primitive-ids");
    }
    // Appended only when set.
    if inputs.private_list {
        h.bytes(b"private-list");
    }
    // Appended only when set.
    if let Some(d) = inputs.amplification_draws {
        h.bytes(b"amplification-draws")
            .u64(u64::from(d.layered))
            .u64(u64::from(d.viewports));
    }
    // Appended only when set, in its derived `Debug` form (an ordered map of plain enums and
    // integers, a total encoding as `inputs_encoding`'s).
    if !inputs.color_writes.is_empty() {
        h.bytes(b"color-writes")
            .bytes(format!("{:?}", inputs.color_writes).as_bytes());
    }
    // Appended only when set, in its derived `Debug` form (an ordered set of byte keys, a total
    // encoding as `inputs_encoding`'s).
    if !inputs.primitive_varyings.is_empty() {
        h.bytes(b"primitive-varyings")
            .bytes(format!("{:?}", inputs.primitive_varyings).as_bytes());
    }
    // Appended only when set, in its derived `Debug` form (an ordered set of byte keys).
    if !inputs.absent_varyings.is_empty() {
        h.bytes(b"absent-varyings")
            .bytes(format!("{:?}", inputs.absent_varyings).as_bytes());
    }
    // Appended only when set.
    if let Some(c) = inputs.viewport_compensation {
        h.bytes(b"viewport-compensation").u64(u64::from(c.depth));
        // Appended only when set.
        if c.point_guard {
            h.bytes(b"point-guard");
        }
        // Appended only when set.
        if c.mirrors {
            h.bytes(b"viewport-mirrors");
        }
        // Appended only when set.
        if c.vertex_clamp {
            h.bytes(b"viewport-vertex-clamp");
        }
    }
    // Appended only when set.
    if inputs.folded_groups {
        h.bytes(b"folded-groups");
    }
    // Appended only when set, in its derived `Debug` form (an ordered map of integers and plain
    // enums, a total encoding as `inputs_encoding`'s).
    if !inputs.zeroed_colors.is_empty() {
        h.bytes(b"zeroed-colors")
            .bytes(format!("{:?}", inputs.zeroed_colors).as_bytes());
    }
    if !inputs.dropped_colors.is_empty() {
        h.bytes(b"dropped-colors")
            .bytes(format!("{:?}", inputs.dropped_colors).as_bytes());
    }
    // Appended only when set.
    if let Some(bytes) = inputs.kept_imageblock_bytes {
        h.bytes(b"kept-imageblock-bytes")
            .bytes(&bytes.to_le_bytes());
    }
    // Appended only when set.
    if inputs.depth_clamp_range {
        h.bytes(b"depth-clamp-range");
    }
    // Appended only when set.
    if inputs.rotated_weights {
        h.bytes(b"rotated-weights");
    }
    // Appended only when set.
    if inputs.addressed_function_tables {
        h.bytes(b"addressed-function-tables");
    }
    // Appended only when set, in its derived `Debug` form, as `dropped_colors` is.
    if !inputs.unattached_colors.is_empty() {
        h.bytes(b"unattached-colors")
            .bytes(format!("{:?}", inputs.unattached_colors).as_bytes());
    }
    // Appended only when set; masks alike at every place of a quad as the one mask.
    match inputs.sample_mask {
        Some(m) if crate::pipeline::private_state::uniform(&m) => {
            h.bytes(b"sample-mask").bytes(&m[0].to_le_bytes());
        }
        Some(m) => {
            h.bytes(b"sample-masks");
            for mask in m {
                h.bytes(&mask.to_le_bytes());
            }
        }
        None => {}
    }
    // Appended only when set.
    if let Some(function) = inputs.alpha_test {
        h.bytes(b"alpha-test")
            .bytes(&function.as_raw().to_le_bytes());
    }
    // Appended only when set.
    if inputs.depth_compare_clamp_mask != 0 {
        h.bytes(b"depth-compare-clamp")
            .bytes(&inputs.depth_compare_clamp_mask.to_le_bytes());
    }
    // Appended only when set.
    if inputs.read_only_tests {
        h.bytes(b"read-only-tests");
    }
    // Appended only when set.
    if inputs.shader_tests {
        h.bytes(b"shader-tests");
    }
    // Appended only when set.
    if inputs.shader_stencil {
        h.bytes(b"shader-stencil");
    }
    // Appended only when set.
    if inputs.window_depth_varying {
        h.bytes(b"window-depth-varying");
    }
    // Appended only when set.
    if inputs.threadgroup_size_matches_tile_size {
        h.bytes(b"threadgroup-size-matches-tile-size");
    }
    // Appended only when set, as `storage_view_slots`: a texture whose map is the identity keeps
    // every key.
    if !inputs.address_maps.is_empty() {
        h.bytes(b"address-maps")
            .bytes(format!("{:?}", inputs.address_maps).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.cubic_samplers.is_empty() {
        h.bytes(b"cubic-samplers")
            .bytes(format!("{:?}", inputs.cubic_samplers).as_bytes());
    }
    // Appended only when set; the indexed form adds its index type.
    if let Some(form) = inputs.captured_positions {
        h.bytes(b"captured-positions");
        if let Some(ty) = form.index_type() {
            h.bytes(b"indexed").u64(u64::from(ty.as_raw()));
        }
    }
    // Appended only when set, in its derived `Debug` form, as `dropped_colors` is: the shadowed
    // variant of a function whose fetch inputs already hold the color (a blend's) is otherwise the
    // unshadowed one's key, and a cache would hand it the unshadowed artifact.
    if !inputs.extended_shadows.is_empty() {
        h.bytes(b"extended-shadows")
            .bytes(format!("{:?}", inputs.extended_shadows).as_bytes());
    }
    // Appended only when set: the default mode keeps every key.
    if inputs.texture_write_rounding != crate::metal::TextureWriteRoundingMode::TowardZero {
        h.bytes(b"texture-write-rounding-nearest-even");
    }
    // Appended only when set, as `storage_view_slots`.
    if !inputs.cross_class.is_empty() {
        h.bytes(b"cross-class")
            .bytes(format!("{:?}", inputs.cross_class).as_bytes());
    }
    // Appended only when set, as `storage_view_slots`: a function bound to no anisotropic
    // sampler keeps every key.
    if !inputs.anisotropic_samplers.is_empty() {
        h.bytes(b"anisotropic-samplers")
            .bytes(format!("{:?}", inputs.anisotropic_samplers).as_bytes());
    }
    // Appended only when set, in its derived `Debug` form (an ordered set of byte keys): a
    // fragment function whose vertex function writes no varying it interpolates the same at
    // every vertex keeps its key.
    if !inputs.uniform_varyings.is_empty() {
        h.bytes(b"uniform-varyings")
            .bytes(format!("{:?}", inputs.uniform_varyings).as_bytes());
    }
    // Appended only when set, as `uniform_varyings` (an ordered set of attribute indices): a
    // vertex function no per-instance attribute makes write a varying alike keeps its key.
    if !inputs.instanced_attributes.is_empty() {
        h.bytes(b"instanced-attributes")
            .bytes(format!("{:?}", inputs.instanced_attributes).as_bytes());
    }
    // Appended only when set: a fragment function of any other pipeline or profile keeps its key.
    if inputs.depth16_codes {
        h.bytes(b"depth16-codes");
    }
    if let Some(raises) = inputs.depth16_bias_raises {
        h.bytes(b"depth16-bias-raises").u64(u64::from(raises));
    }
    // Appended only when set, in its derived `Debug` form (integers), as `dropped_colors` is.
    if let Some(group) = &inputs.color_group {
        h.bytes(b"color-group")
            .bytes(format!("{group:?}").as_bytes());
    }
    // Appended only when set: a mesh function whose own payload arena holds its object
    // function's keeps its key.
    if let Some(bytes) = inputs.object_payload {
        h.bytes(b"object-payload").u64(u64::from(bytes));
    }
    // Appended only when present, so a stage that links nothing keeps its key: the tag's length
    // prefix keeps the encoding unambiguous.
    if !linked.is_empty() {
        h.bytes(b"linked-functions").u64(linked.len() as u64);
        for l in linked {
            match l.list {
                LinkedList::Functions => h.u64(0),
                LinkedList::PrivateFunctions => h.u64(1),
                LinkedList::BinaryFunctions => h.u64(2),
                LinkedList::Group(name) => h.u64(3).bytes(name.as_bytes()),
            };
            h.bytes(&l.air_hash).bytes(l.name.as_bytes());
            constants(&mut h, l.constants);
        }
    }
    // Appended only when present, as above.
    if !dynamic.libraries.is_empty() || dynamic.unloaded.is_some() {
        h.bytes(b"dynamic-libraries")
            .u64(dynamic.libraries.len() as u64);
        for (name, hash) in &dynamic.libraries {
            h.bytes(name.as_bytes()).bytes(*hash);
        }
        if let Some(name) = dynamic.unloaded {
            h.bytes(b"unloaded").bytes(name.as_bytes());
        }
    }
    FunctionCompileKey::new(h.finish())
}

/// The `PipelineKey` of a pipeline of `kind` for the profile of fingerprint `profile`, with its
/// stages' keys in stage order and `fields`, the encoding of the descriptor fields its artifact
/// depends on.
pub fn pipeline_key(
    kind: &[u8],
    profile: &Hash256,
    stages: &[FunctionCompileKey],
    fields: &str,
) -> PipelineKey {
    let mut h = KeyHasher::new(b"metal2vulkan/pipeline-key/v1");
    h.bytes(kind).bytes(profile).u64(stages.len() as u64);
    for s in stages {
        h.bytes(s.digest());
    }
    h.bytes(fields.as_bytes());
    PipelineKey::new(h.finish())
}

/// The key of mesh render pipeline descriptor `d` for the profile of fingerprint `profile`
/// (`Device::mesh_descriptor_key`): each stage by `stage`'s kind and digest, then every other
/// field of `d`, as [`render_descriptor_key`] writes a render descriptor's, and each named here
/// so that a field added to the descriptor does not compile until the key reads it.
pub(crate) fn mesh_descriptor_key(
    profile: &Hash256,
    d: &MeshPipelineDescriptor,
    stage: impl Fn(&StageSource) -> (u64, Hash256),
) -> PipelineKey {
    let MeshPipelineDescriptor {
        object,
        mesh,
        fragment,
        color_attachments,
        depth_format,
        stencil_format,
        raster_sample_count,
        rasterization_enabled,
        alpha_to_coverage,
        alpha_to_one,
        support_indirect_command_buffers,
        texture_write_rounding_mode,
    } = d;
    let mut h = KeyHasher::new(b"metal2vulkan/mesh-descriptor-key/v1");
    h.bytes(profile);
    for source in [object.as_ref(), Some(mesh), fragment.as_ref()] {
        match source {
            Some(source) => {
                let (kind, digest) = stage(source);
                h.u64(kind).bytes(&digest);
            }
            // As `render_descriptor_key` writes a descriptor without a fragment stage: a stage a
            // descriptor does not have is told apart from every stage it could have.
            None => {
                h.u64(u64::MAX);
            }
        }
    }
    let fields = [
        format!("{color_attachments:?}"),
        format!("{depth_format:?} {stencil_format:?}"),
        format!("{raster_sample_count} {alpha_to_coverage} {alpha_to_one}"),
        format!("{rasterization_enabled} {support_indirect_command_buffers}"),
        format!("{texture_write_rounding_mode:?}"),
    ];
    h.u64(fields.len() as u64);
    for field in &fields {
        h.bytes(field.as_bytes());
    }
    PipelineKey::new(h.finish())
}

/// The key of render pipeline descriptor `d` for the profile of fingerprint `profile`
/// (`Device::render_descriptor_key`): each stage by `stage`'s kind and digest, then every other
/// field of `d`, each named here so that a field added to the descriptor does not compile until
/// the key reads it. A field is written in its derived `Debug` form, which no float of these
/// types takes (the private state's one float is written by its bits, so `-0.0` and each NaN
/// payload are told apart).
pub(crate) fn render_descriptor_key(
    profile: &Hash256,
    d: &RenderPipelineDescriptor,
    stage: impl Fn(&StageSource) -> (u64, Hash256),
) -> PipelineKey {
    let RenderPipelineDescriptor {
        vertex,
        fragment,
        vertex_descriptor,
        color_attachments,
        depth_format,
        stencil_format,
        raster_sample_count,
        alpha_to_coverage,
        alpha_to_one,
        rasterization_enabled,
        input_primitive_topology,
        max_vertex_amplification_count,
        tessellation,
        support_indirect_command_buffers,
        max_vertex_call_stack_depth,
        max_fragment_call_stack_depth,
        support_adding_vertex_binary_functions,
        support_adding_fragment_binary_functions,
        vertex_buffers,
        fragment_buffers,
        texture_write_rounding_mode,
        private,
    } = d;
    let PrivateRenderState {
        sample_mask,
        sample_coverage,
        sample_coverage_invert,
        alpha_test_enabled,
        alpha_test_function,
        logic_operation_enabled,
        logic_operation,
        clip_distance_enable_mask,
        point_smooth_enabled,
        point_coord_lower_left,
        point_size_output_vs,
        two_side_enabled,
        vertex_depth_compare_clamp_mask,
        fragment_depth_compare_clamp_mask,
        depth_stencil_write_disabled,
        needs_custom_border_color_samplers,
        vertex_amplification_mode,
        explicit_visibility_group_id,
    } = private;
    let mut h = KeyHasher::new(b"metal2vulkan/render-descriptor-key/v1");
    h.bytes(profile);
    let (kind, digest) = stage(vertex);
    h.u64(kind).bytes(&digest);
    match fragment {
        Some(fragment) => {
            let (kind, digest) = stage(fragment);
            h.u64(kind).bytes(&digest);
        }
        None => {
            h.u64(u64::MAX);
        }
    }
    let fields = [
        format!("{vertex_descriptor:?}"),
        format!("{color_attachments:?}"),
        format!("{depth_format:?} {stencil_format:?}"),
        format!("{raster_sample_count} {alpha_to_coverage} {alpha_to_one}"),
        format!("{rasterization_enabled} {input_primitive_topology:?}"),
        format!("{max_vertex_amplification_count} {tessellation:?}"),
        format!("{support_indirect_command_buffers}"),
        format!("{max_vertex_call_stack_depth} {max_fragment_call_stack_depth}"),
        format!(
            "{support_adding_vertex_binary_functions} {support_adding_fragment_binary_functions}"
        ),
        format!("{vertex_buffers:?} {fragment_buffers:?}"),
        format!("{texture_write_rounding_mode:?}"),
        format!(
            "{sample_mask} {} {sample_coverage_invert} {alpha_test_enabled} \
             {alpha_test_function} {logic_operation_enabled} {logic_operation} \
             {clip_distance_enable_mask} {point_smooth_enabled} {point_coord_lower_left} \
             {point_size_output_vs} {two_side_enabled} {vertex_depth_compare_clamp_mask} \
             {fragment_depth_compare_clamp_mask} {depth_stencil_write_disabled} \
             {needs_custom_border_color_samplers}{}",
            sample_coverage.to_bits(),
            // Appended only past Metal's defaults, so every other descriptor keeps its key.
            match (*vertex_amplification_mode, *explicit_visibility_group_id) {
                (0, u32::MAX) => String::new(),
                (mode, group) => format!(" {mode} {group}"),
            }
        ),
    ];
    h.u64(fields.len() as u64);
    for field in &fields {
        h.bytes(field.as_bytes());
    }
    PipelineKey::new(h.finish())
}
