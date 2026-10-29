//! `BindState` and `BindShape`: the Metal-shaped binder input.
//!
//! The consumer keeps its own resource handles; the binder sees facts about what is bound, keyed
//! by Metal slot, and returns instructions that reference slots and consumer tokens. A slot absent
//! from a table is unbound. [`BindShape`] is the hashable projection that can change a plan;
//! offsets, addresses, and lengths only change per-draw bytes ([`BindState::shape`]).

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::bind::ResidentRange;
use crate::device::profile::Limits;
use crate::metal;
use crate::pipeline::artifact::RuntimeDependency;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::ViewportCompensation;
use crate::pipeline::sym::{SlotClass, Stage};

/// Everything bound for one draw or dispatch.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BindState {
    /// The argument tables of each Metal stage.
    pub stages: BTreeMap<Stage, StageTable>,
    /// `useResource` / `useResources` / `useHeap(s)`, and every other live allocation the guest
    /// task may reach (a superset by contract).
    ///
    /// **Textures resident through `useHeap`**: a `Heap` entry makes the heap's address range
    /// resident for buffer pointers; it names no texture. The consumer lists every texture the
    /// guest allocated from a used heap as its own `Texture` entry (its facts and resource id, the
    /// heap's usage and stages). A handle table holds exactly the textures listed here and those
    /// bound at a slot, so a resource id it lacks misses: its reads and queries answer the
    /// neutral's, its writes are dropped, and `is_null` answers true. A texture's entry carries
    /// whether its pixel format has an alpha: a heap read the shader bounds-checks reads Metal's
    /// out-of-bounds alpha from it, so a texture's facts must be the resident texture's.
    pub used_resources: Vec<UsedResource>,
    /// The render encoder's vertex amplification (`setVertexAmplificationCount:viewMappings:`);
    /// Metal's default of a count of 1 without view mappings otherwise.
    #[cfg_attr(feature = "serde", serde(default))]
    pub vertex_amplification: VertexAmplification,
    /// The compute encoder's imageblock size (`setImageblockWidth:height:`); 0 by 0 when the
    /// encoder never set it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub imageblock_size: ImageblockSize,
    /// The render pass's `renderTargetArrayLength` (`MTLRenderPassDescriptor`): the number of
    /// layers a vertex function's `[[render_target_array_index]]` can select. Metal's default, 0,
    /// is a pass that is not layered; 0 and 1 both render every primitive into one slice, so a
    /// vertex function that chooses its slice runs on the variant that does not write `Layer`
    /// (`VariantKey::dropped_vertex_layer`). In a longer pass the variant that writes `Layer`
    /// stores 0 for an index at or past the length, which each draw writes to its push bytes
    /// (`PushField::RenderTargetArrayLength`). 0 in a compute encoder.
    #[cfg_attr(feature = "serde", serde(default))]
    pub render_target_array_length: u32,
    /// The compute encoder's stage-in region (`setStageInRegion:`): the origin a kernel's
    /// `[[stage_in]]` attributes stepping by thread position start from, and its
    /// `[[grid_origin]]` and `[[grid_size]]`. `None` when the encoder never set it: what Metal
    /// reads then is not established, so a dispatch of a kernel that reads the region is
    /// `BindError::MissingStageInRegion`. Only a dispatch's bytes depend on it, so it is not part
    /// of the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub stage_in_region: Option<StageInRegion>,
    /// The render encoder's tessellation factor buffer and scale; Metal's defaults (no buffer,
    /// scale 1) when the encoder never set them. Read by patch draws only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub tessellation_factors: TessellationFactors,
    /// The render area a tile dispatch covers, in pixels (`MTLRenderPassDescriptor`'s
    /// `renderTargetWidth` and `renderTargetHeight`): a tile function stages the pixels of its
    /// attachments inside it. `[0, 0]` outside a tile dispatch. With `imageblock_size` set to the
    /// pass's tile size, a tile dispatch is `Dispatch::Threadgroups` of `ceil(render_area / tile
    /// size)` groups of the `dispatchThreadsPerTile:` size.
    #[cfg_attr(feature = "serde", serde(default))]
    pub render_area: [u32; 2],
    /// Every live sampler state the guest task made, with or without `supportArgumentBuffers`
    /// (Metal's functions sample through either's `gpuResourceID`), in any order: the samplers an
    /// argument buffer's sampler handle can name. The consumer keeps this registry at sampler
    /// creation and release, as it lists every live allocation in `used_resources`. It is derived
    /// from object lifetimes, never from argument-buffer memory: Metal keeps a sampler alive while
    /// a command reads it and never changes a sampler state, so it holds every id a buffer can
    /// validly hold however the buffer was written. A function's sampler handle names the sampler
    /// of its id here, or misses and reads the neutral sampler. Only the shape of the list enters a
    /// plan ([`ArgumentSamplersShape`]); the ids and descriptors are per draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub argument_samplers: Vec<ArgumentSampler>,
    /// The render encoder's depth/stencil state (`setDepthStencilState:`); Metal's default
    /// descriptor (no depth or stencil test) when the encoder never set it. Read only by a draw
    /// whose pipeline runs a shader-side stencil (`GraphicsState::shadow_stencil`), which takes
    /// its stencil state as push data; every other pipeline takes it as Vulkan state
    /// (`metal::state::depth_stencil_state`). Only a draw's bytes depend on it, so it is not part
    /// of the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_stencil: metal::state::DepthStencilDescriptor,
    /// The draw's primitive type (`drawPrimitives:`'s or `drawIndexedPrimitives:`'s
    /// `primitiveType`), which a draw of a pipeline whose vertex function writes the primitive ID
    /// reads (`LoweringInputs::primitive_id_varying`): its `PushField::VerticesPerPrimitive` holds
    /// the type's vertices per primitive, and an indexed strip, which Metal restarts at the index
    /// type's all-ones index, does not run de-indexed. `None` where the consumer does not say;
    /// such a draw is `BindError::MissingPrimitiveType`. Only a draw's bytes depend on it, so it
    /// is not part of the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_type: Option<metal::PrimitiveType>,
    /// The index type of a direct indexed draw of a pipeline whose vertex function writes the
    /// primitive ID's varying (`LoweringInputs::primitive_id_varying`): its primitives share
    /// vertices, so it runs the variant that draws its indices as vertices from 0
    /// (`VariantKey::primitive_id_indices`). Part of the shape
    /// (`BindShape::primitive_id_indices`); `None` for every other draw, whose plan otherwise
    /// asks for it (`BindError::NeedsVariant`) or refuses the draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_id_indices: Option<metal::IndexType>,
    /// The draw is of Metal's private quads or quad strip, which the consumer lists as two
    /// triangles a quad, each quad's adjacent (`bind::RewriteProgram::Quads`, `QuadStrip`): a
    /// pipeline whose fragment function reads Vulkan's `PrimitiveId` runs the variant that reads
    /// the quad's, that value shifted right by one, as Metal counts one primitive a quad
    /// (`VariantKey::quad_primitive_ids`). Part of the shape (`BindShape::quad_primitive_ids`);
    /// `false` for every other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub quad_primitive_ids: bool,
    /// The draw is of one of Metal's private primitive types, drawn in the pulled list form
    /// (`VariantKey::private_list`): without indices, each position of the list the consumer
    /// writes (Metal's base vertex in the word before its first index) run once, from the list's
    /// first index as first vertex. Part of the shape (`BindShape::private_list`); `false` for
    /// every other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub private_list: bool,
    /// Where the heap arrays (set 1) hold the command's heap members and argument samplers
    /// ([`HeapPlacement`]). Only the instantiated tables and the elements the consumer writes
    /// ([`BindingPlan::heap_elements`]) depend on it, so it is not part of the shape.
    ///
    /// [`BindingPlan::heap_elements`]: crate::bind::BindingPlan::heap_elements
    #[cfg_attr(feature = "serde", serde(default))]
    pub heap_placement: HeapPlacement,
    /// The descriptor base alignment of the device's uniform views
    /// ([`view_base_alignment_of`](Self::view_base_alignment_of)), which the runtime sets: the one
    /// alignment above the vertex fetch's that the bind offset class of a buffer at a slot of
    /// [`StageTable::at_bind_views`] tells apart, as a uniform view's range starts at its bind
    /// offset only at a multiple of it (`SpecKey::RangeAtBindOffset`). 0, the
    /// default, tells none apart: every view is then read at its residual, exact on every device,
    /// as it is under any other value the plan does not find its own.
    #[cfg_attr(feature = "serde", serde(default))]
    pub view_base_alignment: u32,
    /// The render encoder's viewport count (`setViewports:count:`): a vertex function's
    /// `[[viewport_array_index]]` at or past it draws through viewport 0, as Metal's does
    /// (`PushField::ViewportCount`). Metal's default is one viewport; 0, the default here, and 1
    /// both draw every index through viewport 0. Only a draw's bytes depend on it, so it is not
    /// part of the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub viewport_count: u32,
    /// Every live visible-function table the command's pipeline state made, in any order: the
    /// tables an argument buffer's `visible_function_table` member can name. The consumer keeps
    /// this registry at table creation and release, as it keeps `argument_samplers`: Metal reads a
    /// table through an argument buffer without `useResource`, so every live table of the state is
    /// listed, resident or not. A function's table names the table of its id here, or misses and
    /// reads the null table. Only the capacity class of the list enters a plan
    /// ([`FunctionTablesShape`]); the ids and sizes are per command, and each table's words are the
    /// consumer's (`WriteSource::FunctionTable`, or `WriteSource::FunctionTableSnapshot` on a
    /// device without the function-table array's features, whose directory places each table by the
    /// sizes before it).
    #[cfg_attr(feature = "serde", serde(default))]
    pub function_tables: Vec<ArgumentFunctionTable>,
    /// The render encoder's tile threadgroup memory: the render pass's `threadgroupMemoryLength`
    /// and each argument `setThreadgroupMemoryLength:offset:atIndex:` placed in it, which a tile
    /// function's `[[threadgroup(n)]]` arguments read (`RuntimeFact::TileThreadgroupMemory`). Empty
    /// outside a render encoder.
    #[cfg_attr(feature = "serde", serde(default))]
    pub tile_threadgroup: TileThreadgroupMemory,
    /// The draw's viewport compensation (`PushField::ViewportCompensation`): `None` where the
    /// draw's viewport is Vulkan's, as it is unless it has a negative width or a depth range past
    /// `[0, 1]` the device does not take. Its variant is part of the shape
    /// (`BindShape::viewport_compensation`), its values only of the draw's bytes.
    #[cfg_attr(feature = "serde", serde(default))]
    pub viewport_compensation: Option<CompensatedViewport>,
    /// The color attachments of the draw's render pass its render pipeline has no pixel format
    /// for, the pass's pixel format of each by color index: plain Metal stores zero bytes into
    /// each at every sample the draw covers (Metal's validation stops the draw). Its variant is
    /// part of the shape (`BindShape::zeroed_colors`); empty for every other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub zeroed_colors: BTreeMap<u32, metal::PixelFormat>,
    /// The color attachments the draw's render pipeline has a pixel format for and its render
    /// pass has none, by color index: plain Metal drops what the fragment function writes there.
    /// Its variant is part of the shape (`BindShape::dropped_colors`); empty for every other
    /// draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub dropped_colors: BTreeSet<u32>,
    /// The bytes of each cell of the tile function's explicit imageblock standing alone that the
    /// dispatch's render pass keeps, where they are fewer than the cell holds: plain Metal reads
    /// the block's words past them as zero and drops their writes (Metal's validation stops the
    /// state's setting). Its variant is part of the shape (`BindShape::kept_imageblock_bytes`);
    /// `None` for every other dispatch and for draws.
    #[cfg_attr(feature = "serde", serde(default))]
    pub kept_imageblock_bytes: Option<u32>,
    /// The draw's depth clamp range `[low, high]` where its fragment function clamps the depth it
    /// writes itself (`PushField::DepthClampRange`): a draw in depth clamp mode through a range
    /// above 0, whose Vulkan depth clamp holds 0. Its variant is part of the shape
    /// (`BindShape::depth_clamp_range`), its values only of the draw's bytes; `None` for every
    /// other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_clamp_range: Option<[f32; 2]>,
    /// Whether the draw draws triangles with Metal's last vertex first, `(c, a, b)` for Metal's
    /// `(a, b, c)`: a list the consumer draws for triangles under Apple's last provoking vertex
    /// (`setProvokingVertexMode:` 1) on a device that takes flat inputs from the first vertex.
    /// A pipeline whose fragment function reads `[[barycentric_coord]]` runs the variant that takes
    /// Metal's weights back from the drawn triangle's (`BindShape::rotated_weights`,
    /// `VariantKey::rotated_weights`). Part of the shape; `false` for every other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_weights: bool,
    /// Whether a mesh draw of an emulated mesh function's pipeline draws each line or triangle
    /// from its last index first: a draw under Apple's last provoking vertex
    /// (`setProvokingVertexMode:` 1) on a device that takes flat inputs from the first vertex.
    /// Its pipeline runs the variant whose synthesized vertex stage reads its corners so
    /// (`BindShape::mesh_last_first`, `VariantKey::mesh_last_first`). Part of the shape; `false`
    /// for every other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mesh_last_first: bool,
    /// Allocations a pointer may reach that the consumer lists apart from `used_resources`, at
    /// their own lengths: the resident set holds them beside the bound buffers and the used
    /// resources, and a replay's extents count them (the runtime's live buffers of their own,
    /// for a replay whose resident root table it does not point at its own set). Only the
    /// resident sets of the root tables depend on them, so they are not part of the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub resident_ranges: Vec<ResidentRange>,
    /// The color attachments the draw's render pass instance leaves out of its subpass, read and
    /// written as storage images in the pixel interlock by its other draws, that the draw's
    /// render pipeline does not itself read or write so: a fragment function that writes one of
    /// them through its first output runs the variant that writes it through the interlocked
    /// storage form, so the draw records in that instance. Its variant is part of the shape
    /// (`BindShape::interlocked_colors`); empty for every other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub interlocked_colors: BTreeSet<u32>,
    /// The extended-range color attachments whose `half` shadow the draw's render pass instance
    /// holds (Metal's tile value of each texel), by color index: a fragment function that writes or
    /// reads one in the interlocked storage form runs the variant that reads and writes the shadow
    /// beside the codes (`VariantKey::extended_shadows`), and the consumer binds the instance's
    /// shadow where the plan writes `WriteSource::ExtendedShadow`. The colors the function writes
    /// this way the instance leaves out of its subpass, as it does `interlocked_colors`. Its
    /// variant is part of the shape (`BindShape::extended_shadows`); empty for every other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub extended_shadows: BTreeSet<u32>,
    /// The fragment texture slots the consumer reads through a snapshot, by Metal index: a draw
    /// that reads a texture its render pass renders, where the consumer draws it twice, first
    /// taking each fragment's texel into the slot's snapshot with every color write mask zero,
    /// then with the pipeline's masks reading the snapshot. The plan keeps a slot only where the
    /// draw's pipeline can (the module docs of `bind::graphics`) and runs the variant that reads
    /// it so (`VariantKey::snapshot_textures`); `BindingPlan::variant`'s `snapshot_textures` says
    /// which it kept, and the consumer serves every other read as it would without it. Part of the
    /// shape (`BindShape::snapshot_textures`); empty for every other draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub snapshot_textures: BTreeSet<u32>,
    /// What the draw's depth and stencil tests can do to its fragments ([`FragmentTests::of`]),
    /// which decides whether a fragment function that stores a color through the interlocked
    /// storage form before the tests (`RuntimeFact::TestedStorageWrites`) stores what Metal
    /// stores: its plan runs it as compiled, asks for the variant that declares early tests, or
    /// refuses the draw. Its default, [`FragmentTests::Any`], refuses such a draw; the consumer
    /// states it for a draw of such a pipeline, and may leave it for every other, whose plan does
    /// not read it. Part of the shape (`BindShape::fragment_tests`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub fragment_tests: FragmentTests,
    /// The render encoder's blend color (`setBlendColorRed:green:blue:alpha:`; Metal's default
    /// `[0, 0, 0, 0]` where the encoder never set it), red first: a fragment function that
    /// computes a blend in the shader under a blend that reads it takes it as push data
    /// (`PushField::HalfBlendColor`). Only a draw's bytes depend on it, so it is not part of the
    /// shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub blend_color: [f32; 4],
    /// The render encoder's alpha test reference (`setAlphaTestReferenceValue:`, Apple SPI);
    /// `None` where the encoder never set it, which Metal's test compares as 1 (each encoder
    /// starts there: a reference set in an earlier encoder is not kept). A fragment function that
    /// runs the pipeline's private alpha test takes its code as push data
    /// (`PushField::AlphaTestReference`). Only a draw's bytes depend on it, so it is not part of
    /// the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub alpha_test_reference: Option<f32>,
    /// Viewport 0's depth range `[near, far]` as the encoder set it (`setViewport:`; Metal's
    /// default viewport has `[0, 1]`), which a fragment function that runs the draw's depth test
    /// itself clamps its depth to in depth clamp mode (`PushField::ShaderTestsState`). Read only
    /// by a draw of such a pipeline (`GraphicsState::shader_tests`), whose consumer sets it;
    /// only the draw's bytes depend on it, so it is not part of the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_range: [f32; 2],
    /// The encoder's depth clip mode is `MTLDepthClipModeClamp` (`setDepthClipMode:`), under
    /// which such a function clamps its depth to the viewport's range, where in clip mode it
    /// clamps to `[0, 1]` (`PushField::ShaderTestsState`). Read only by a draw of such a
    /// pipeline; not part of the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_clip_mode_clamp: bool,
    /// The encoder's stencil references, the front face's then the back face's
    /// (`setStencilReferenceValue:` sets both, `setStencilFrontReferenceValue:backReferenceValue:`
    /// each; Metal's default is 0), which a fragment function that runs the draw's stencil test
    /// itself compares and replaces with (`PushField::ShaderStencilState`). Read only by a draw of
    /// such a pipeline (`ShaderTestsUse::stencil`), whose consumer sets it; only the draw's bytes
    /// depend on it, so it is not part of the shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub stencil_references: [u32; 2],
}

/// What a draw's depth and stencil tests can do to its fragments (`BindState::fragment_tests`).
/// Metal tests a fragment after its function runs, unless the function declares
/// `[[early_fragment_tests]]`: a function that writes a color attachment through the interlocked
/// storage form stores inside the shader, so where it discards, and so cannot be tested early
/// whatever the draw's state, whether its store is Metal's is the draw's tests' to decide.
///
/// Where the tests drop fragments, such a function runs them itself where it can:
/// [`FragmentTests::Depth`] and [`FragmentTests::Stencil`] name the tests it runs, and a
/// [`FragmentTests::Drop`] draw of a function that may not declare early tests runs both.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FragmentTests {
    /// The tests may drop a fragment and the encoder counts the fragments that pass them (a
    /// visibility result mode is set); or the draw does not state them. Such a draw of a function
    /// that stores before the tests is refused `NotImplemented(FramebufferFetch)`: a store before
    /// them would keep what they drop, and early tests would count what the function then
    /// discards. (A draw whose tests write but count nothing is [`FragmentTests::Depth`] or
    /// [`FragmentTests::Stencil`].)
    #[default]
    Any,
    /// The tests may drop a fragment, but write neither the depth nor the stencil attachment, and
    /// the encoder counts nothing: early tests drop the fragments Metal's later tests drop and
    /// leave the attachments as Metal's do, so such a draw runs the variant that declares them
    /// (`VariantKey::read_only_tests`). (A function that may not declare them, since it writes
    /// other memory or returns depth, runs the tests itself as under [`FragmentTests::Stencil`].)
    Drop,
    /// No test can drop a fragment: a store before the tests is the store Metal makes after
    /// them, so such a draw runs the pipeline as compiled.
    Pass,
    /// Only the depth test can drop a fragment, and it writes the depth attachment where
    /// `writes`; the stencil test can neither drop nor write, and the encoder counts nothing.
    /// Such a draw of a function that stores before the tests runs the variant whose function
    /// runs the depth test itself (`VariantKey::shader_tests`) where the device ships it, or, where
    /// it writes nothing and the function has one, the variant that declares early tests, as
    /// [`FragmentTests::Drop`].
    Depth { writes: bool },
    /// The stencil test can drop a fragment or write the stencil attachment, and some test
    /// writes its attachment (the stencil test's operations, or the depth test's writes; where
    /// none does, the tests are [`FragmentTests::Drop`]); the encoder counts nothing. Such a draw
    /// of a function that stores before the tests runs the variant whose function runs the
    /// stencil and depth tests itself (`VariantKey::shader_stencil`), where the pass has a depth
    /// and a stencil attachment and the device ships the form; a pass with a stencil attachment
    /// alone is refused at it.
    Stencil,
}

impl FragmentTests {
    /// The tests of `depth_stencil` in a pass with a depth attachment where `depth` and a
    /// stencil attachment where `stencil`, the encoder counting the fragments that pass them
    /// where `counted`: a test of an attachment the pass lacks does nothing. Depth drops a
    /// fragment unless it compares `Always`, and writes where its writes are enabled; a stencil
    /// face drops one unless it compares `Always`, and writes where its write mask has a bit set
    /// and an operation that can apply is not `Keep` (the stencil failure operation where it
    /// can fail, the depth failure operation where depth can, and the pass operation).
    pub fn of(
        depth_stencil: &metal::state::DepthStencilDescriptor,
        depth: bool,
        stencil: bool,
        counted: bool,
    ) -> FragmentTests {
        use metal::{CompareFunction, StencilOperation};
        let depth_drops = depth && depth_stencil.depth_compare_function != CompareFunction::Always;
        let faces = [
            &depth_stencil.front_face_stencil,
            &depth_stencil.back_face_stencil,
        ];
        let stencil_drops = stencil
            && faces
                .iter()
                .any(|f| f.stencil_compare_function != CompareFunction::Always);
        if !depth_drops && !stencil_drops {
            return FragmentTests::Pass;
        }
        let depth_writes = depth && depth_stencil.depth_write_enabled;
        let stencil_writes = stencil
            && faces.iter().any(|f| {
                let fails = f.stencil_compare_function != CompareFunction::Always;
                let applies = [
                    (fails, f.stencil_failure_operation),
                    (depth_drops, f.depth_failure_operation),
                    (true, f.depth_stencil_pass_operation),
                ];
                f.write_mask != 0
                    && applies
                        .iter()
                        .any(|&(can, op)| can && op != StencilOperation::Keep)
            });
        if !stencil_drops && !stencil_writes && !counted {
            return FragmentTests::Depth {
                writes: depth_writes,
            };
        }
        match (counted, depth_writes || stencil_writes) {
            (true, _) => FragmentTests::Any,
            (false, true) => FragmentTests::Stencil,
            (false, false) => FragmentTests::Drop,
        }
    }
}

/// A draw's viewport compensation (`BindState::viewport_compensation`): the variant its pipeline's
/// vertex function compiles under, and the `[sx, a, b, clip]` it applies to the position, `(sx x,
/// y, a w + b z, w)`, clipping at `z` and `w - z` where `clip` is 1. Under
/// `ViewportCompensation::point_guard`, `guard` is the `[gx, gy]` it scales `x` and `y` by
/// (`PushField::PointGuard`). Under `ViewportCompensation::mirrors`, `mirrors` has bit `i` set
/// where viewport `i` is mirrored (`PushField::ViewportMirrors`). Under
/// `ViewportCompensation::vertex_clamp`, `vertex_clamp` is the `[lo, hi]` the function clamps each
/// vertex's normalized depth `z / w` to (`PushField::VertexDepthClamp`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompensatedViewport {
    pub compensation: ViewportCompensation,
    pub values: [f32; 4],
    #[cfg_attr(feature = "serde", serde(default = "unit_guard"))]
    pub guard: [f32; 2],
    #[cfg_attr(feature = "serde", serde(default))]
    pub mirrors: u32,
    #[cfg_attr(
        feature = "serde",
        serde(default = "unclamped", with = "vertex_clamp_bounds")
    )]
    pub vertex_clamp: [f32; 2],
}

/// The point guard band's identity, `[1, 1]`.
#[cfg(feature = "serde")]
fn unit_guard() -> [f32; 2] {
    [1.0, 1.0]
}

/// The vertex depth clamp's identity, `[-inf, +inf]`: no vertex's depth clamped.
const UNCLAMPED: [f32; 2] = [f32::NEG_INFINITY, f32::INFINITY];

/// [`UNCLAMPED`], serde's default.
#[cfg(feature = "serde")]
fn unclamped() -> [f32; 2] {
    UNCLAMPED
}

/// serde of [`CompensatedViewport::vertex_clamp`]: each bound, or none where that side is
/// unbounded (`-inf` below, `+inf` above, as clip mode's `[0, +inf]` and [`UNCLAMPED`] are),
/// since a format without infinities cannot hold one (`serde_json` writes `null`, which reads back
/// as no `f32`).
#[cfg(feature = "serde")]
mod vertex_clamp_bounds {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(
        bounds: &[f32; 2],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let [lo, hi] = *bounds;
        [
            (lo != f32::NEG_INFINITY).then_some(lo),
            (hi != f32::INFINITY).then_some(hi),
        ]
        .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<[f32; 2], D::Error> {
        let [lo, hi] = <[Option<f32>; 2]>::deserialize(deserializer)?;
        Ok([lo.unwrap_or(f32::NEG_INFINITY), hi.unwrap_or(f32::INFINITY)])
    }
}

impl CompensatedViewport {
    /// The compensation `values` under the variant `compensation`, scaling no point (a `guard`
    /// of `[1, 1]`).
    pub fn new(compensation: ViewportCompensation, values: [f32; 4]) -> CompensatedViewport {
        CompensatedViewport {
            compensation,
            values,
            guard: [1.0, 1.0],
            mirrors: 0,
            vertex_clamp: UNCLAMPED,
        }
    }

    /// This compensation, scaling a point's `x` and `y` by `guard`.
    pub fn with_guard(self, guard: [f32; 2]) -> CompensatedViewport {
        CompensatedViewport { guard, ..self }
    }

    /// This compensation, bit `i` of `mirrors` set where viewport `i` is mirrored.
    pub fn with_mirrors(self, mirrors: u32) -> CompensatedViewport {
        CompensatedViewport { mirrors, ..self }
    }

    /// This compensation, clamping each vertex's normalized depth `z / w` to `vertex_clamp`'s
    /// `[lo, hi]` under `ViewportCompensation::vertex_clamp`.
    pub fn with_vertex_clamp(self, vertex_clamp: [f32; 2]) -> CompensatedViewport {
        CompensatedViewport {
            vertex_clamp,
            ..self
        }
    }
}

/// Where a command's heap arrays (set 1) hold its heap members and its argument samplers. A plan
/// writes member `i` of every image array, and listed argument sampler `i` of every sampler array,
/// at its planned element `i + 1` (element 0 is each array's neutral); the command's handle and
/// sampler tables map each resource ID to the element that holds the resource.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HeapPlacement {
    /// Each at its planned element: every command writes a heap set of its own.
    #[default]
    Planned,
    /// Each at the element its nonzero resource ID names, and the tables map each ID to itself;
    /// an argument sampler with an [`element`](ArgumentSampler::element) at that element, and
    /// its table entry maps its ID there: commands whose IDs name the same resources write the
    /// same element alike, so they can share one heap set. For a consumer whose IDs (and
    /// sampler elements) are dense and small, as the runtime's are: an element at or past the
    /// heap's capacity is refused where the consumer places the writes.
    ResourceId,
}

/// One live sampler state an argument buffer can name (`BindState::argument_samplers`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArgumentSampler {
    pub desc: SamplerDescriptor,
    /// The guest `gpuResourceID`; 0 is the null handle's, which never names a sampler.
    pub resource_id: u64,
    /// Under [`HeapPlacement::ResourceId`], the element of the set-1 sampler arrays that holds
    /// the state, which its table entry maps its resource ID to; `None`, the resource ID
    /// itself. A consumer whose sampler IDs can lie past the arrays (a state placed at an ID)
    /// gives each live state a dense element of its own. Ignored under [`HeapPlacement::Planned`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub element: Option<u32>,
}

impl ArgumentSampler {
    /// The sampler state of `desc` whose `gpuResourceID` is `resource_id`.
    pub fn new(desc: SamplerDescriptor, resource_id: u64) -> ArgumentSampler {
        ArgumentSampler {
            desc,
            resource_id,
            element: None,
        }
    }

    /// This state at `element` of the sampler arrays under [`HeapPlacement::ResourceId`].
    pub fn at_element(self, element: u32) -> ArgumentSampler {
        ArgumentSampler {
            element: Some(element),
            ..self
        }
    }

    /// The number that places the state under [`HeapPlacement::ResourceId`]: its element, else
    /// its resource ID.
    pub(crate) fn placed_at(&self) -> u64 {
        self.element.map_or(self.resource_id, u64::from)
    }
}

/// One live visible-function table an argument buffer can name (`BindState::function_tables`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArgumentFunctionTable {
    /// The table's `gpuResourceID`; 0 is the null table's, which never names a table.
    pub resource_id: u64,
    /// The table's size (`functionCount`): where a function-table snapshot places the tables
    /// after it ([`FunctionTableSnapshot`](crate::bind::FunctionTableSnapshot)).
    #[cfg_attr(feature = "serde", serde(default))]
    pub size: u32,
    /// The device address of the table's words (`FunctionTableWords`), which a pipeline compiled
    /// to read tables through their addresses reads (`VariantKey::addressed_function_tables`);
    /// 0 where the consumer has none, which such a pipeline's dispatch refuses.
    #[cfg_attr(feature = "serde", serde(default))]
    pub address: u64,
}

impl ArgumentFunctionTable {
    /// The table of `size` entries whose `gpuResourceID` is `resource_id`, without a device
    /// address.
    pub fn new(resource_id: u64, size: u32) -> ArgumentFunctionTable {
        ArgumentFunctionTable {
            resource_id,
            size,
            address: 0,
        }
    }

    /// The same table, its words at device address `address`.
    pub fn at_address(self, address: u64) -> ArgumentFunctionTable {
        ArgumentFunctionTable { address, ..self }
    }
}

/// A render encoder's tessellation factor state: `setTessellationFactorBuffer:offset:
/// instanceStride:` and `setTessellationFactorScale:`. A patch draw reads each patch's factors
/// (`MTLTriangleTessellationFactorsHalf` or `MTLQuadTessellationFactorsHalf`) from the buffer as
/// the pipeline's `tessellationFactorStepFunction` addresses them. The default is Metal's: no
/// buffer, whose patches the binder feeds zero factors (culled), and a scale of 1.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationFactors {
    /// The factor buffer; `None` for `nil`.
    pub buffer: Option<TessellationFactorBuffer>,
    /// `setTessellationFactorScale:`, applied when the pipeline enables
    /// `tessellationFactorScaleEnabled` (Metal converts it to `half` first).
    pub scale: f32,
}

impl TessellationFactors {
    /// The factor buffer `buffer` (`None` for `nil`) with scale `scale`.
    pub fn new(buffer: Option<TessellationFactorBuffer>, scale: f32) -> TessellationFactors {
        TessellationFactors { buffer, scale }
    }
}

impl Default for TessellationFactors {
    fn default() -> TessellationFactors {
        TessellationFactors::new(None, 1.0)
    }
}

/// The buffer of `setTessellationFactorBuffer:offset:instanceStride:`. Metal's validation requires
/// the offset and the instance stride to be multiples of 4; plain Metal reads the factors from the
/// offset rounded down to a multiple of 2, and steps the instances by the stride rounded down to a
/// multiple of 4, as the binder does.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationFactorBuffer {
    /// The allocation length (the consumer pads the backing `VkBuffer` to 16 bytes).
    pub length: u64,
    /// `offset`: the byte offset of the factors in the allocation.
    pub offset: u64,
    /// `instanceStride`: the byte stride between two instances' factors.
    pub instance_stride: u64,
}

impl TessellationFactorBuffer {
    /// The factors of an allocation of `length` bytes from `offset`, `instance_stride` bytes per
    /// instance.
    pub fn new(length: u64, offset: u64, instance_stride: u64) -> TessellationFactorBuffer {
        TessellationFactorBuffer {
            length,
            offset,
            instance_stride,
        }
    }
}

impl BindState {
    /// The [`view_base_alignment`](Self::view_base_alignment) of a device of `limits`: the larger
    /// of its `minUniformBufferOffsetAlignment` and 16, the descriptor base alignment of a Metal
    /// buffer slot's uniform view; 0 (none told apart) past `u32`.
    pub fn view_base_alignment_of(limits: &Limits) -> u32 {
        let alignment =
            crate::memory::views::base_alignment(limits.min_uniform_buffer_offset_alignment);
        u32::try_from(alignment).unwrap_or(0)
    }

    /// The projection of this state that can change a plan: per stage, which slots are bound and
    /// with what kind of resource, whether a buffer has a device address, its per-bind stride, its
    /// length class and bind-offset alignment class (the largest power of two up to 4 that
    /// divides the offset, or [`view_base_alignment`](Self::view_base_alignment) where that
    /// divides it at a slot of [`StageTable::at_bind_views`]), texture types and formats, sampler
    /// modes,
    /// threadgroup lengths, the vertex amplification, the render target array length, whether a
    /// tessellation factor buffer is set, a tile dispatch's render area, the argument samplers'
    /// capacity class and shapes ([`ArgumentSamplersShape::of`]), and the tile threadgroup memory
    /// ([`TileThreadgroupMemory`]). Offsets, addresses, lengths, and resource ids are otherwise
    /// left out: they only change per-draw bytes. A stage whose tables are all empty is left out,
    /// so an empty table and an absent one have the same shape.
    pub fn shape(&self) -> BindShape {
        let stages = self
            .stages
            .iter()
            .map(|(stage, t)| (*stage, t.shape(self.view_base_alignment)))
            .filter(|(_, s)| *s != StageShape::default())
            .collect();
        BindShape {
            stages,
            used: self.used_shape(),
            vertex_amplification: self.vertex_amplification.clone(),
            imageblock_size: self.imageblock_size,
            render_target_array_length: self.render_target_array_length,
            tessellation_factor_buffer: self.tessellation_factors.buffer.is_some(),
            render_area: self.render_area,
            argument_samplers: ArgumentSamplersShape::of(&self.argument_samplers),
            function_tables: FunctionTablesShape::of(&self.function_tables),
            tile_threadgroup: self.tile_threadgroup.clone(),
            viewport_compensation: self.viewport_compensation.map(|c| c.compensation),
            zeroed_colors: self.zeroed_colors.clone(),
            dropped_colors: self.dropped_colors.clone(),
            kept_imageblock_bytes: self.kept_imageblock_bytes,
            depth_clamp_range: self.depth_clamp_range.is_some(),
            rotated_weights: self.rotated_weights,
            mesh_last_first: self.mesh_last_first,
            interlocked_colors: self.interlocked_colors.clone(),
            fragment_tests: self.fragment_tests,
            extended_shadows: self.extended_shadows.clone(),
            snapshot_textures: self.snapshot_textures.clone(),
            primitive_id_indices: self.primitive_id_indices,
            quad_primitive_ids: self.quad_primitive_ids,
            private_list: self.private_list,
        }
    }

    /// Whether `shape` is this state's shape: `self.shape() == *shape`, answered without making
    /// the shape, so a consumer that keeps the shapes it made can find a state's among them
    /// without allocating. Its used resources' shape alone is made, where it lists any and every
    /// other field matches; [`position_of_shape`](Self::position_of_shape) makes it once for
    /// many shapes.
    pub fn has_shape(&self, shape: &BindShape) -> bool {
        self.is_shape(shape, &mut None)
    }

    /// The position among `shapes` of the first that is this state's shape (of which
    /// [`has_shape`](Self::has_shape) answers true), found without making the shape: its used
    /// resources' shape is made at most once, however many shapes it is compared with, and only
    /// where it lists any and a shape's every other field matches.
    pub fn position_of_shape<'s>(
        &self,
        shapes: impl IntoIterator<Item = &'s BindShape>,
    ) -> Option<usize> {
        let mut used = None;
        shapes
            .into_iter()
            .position(|shape| self.is_shape(shape, &mut used))
    }

    /// Whether `shape` is this state's shape ([`has_shape`](Self::has_shape)), its used
    /// resources' shape taken from `made`, or made into it where it is needed first.
    fn is_shape(&self, shape: &BindShape, made: &mut Option<BTreeMap<u32, UsedShape>>) -> bool {
        let BindShape {
            stages,
            used,
            vertex_amplification,
            imageblock_size,
            render_target_array_length,
            tessellation_factor_buffer,
            render_area,
            argument_samplers,
            function_tables,
            tile_threadgroup,
            viewport_compensation,
            zeroed_colors,
            dropped_colors,
            kept_imageblock_bytes,
            depth_clamp_range,
            rotated_weights,
            mesh_last_first,
            interlocked_colors,
            fragment_tests,
            extended_shadows,
            snapshot_textures,
            primitive_id_indices,
            quad_primitive_ids,
            private_list,
        } = shape;
        let tables = self.stages.iter().filter(|(_, t)| !t.has_empty_shape());
        tables.clone().count() == stages.len()
            && tables
                .zip(stages)
                .all(|((stage, t), (of, s))| {
                    stage == of && t.has_shape(s, self.view_base_alignment)
                })
            && self.vertex_amplification == *vertex_amplification
            && self.imageblock_size == *imageblock_size
            && self.render_target_array_length == *render_target_array_length
            && self.tessellation_factors.buffer.is_some() == *tessellation_factor_buffer
            && self.render_area == *render_area
            && argument_samplers.is_of(&self.argument_samplers)
            && FunctionTablesShape::of(&self.function_tables) == *function_tables
            && self.tile_threadgroup == *tile_threadgroup
            && self.viewport_compensation.map(|c| c.compensation) == *viewport_compensation
            && self.zeroed_colors == *zeroed_colors
            && self.dropped_colors == *dropped_colors
            && self.kept_imageblock_bytes == *kept_imageblock_bytes
            && self.depth_clamp_range.is_some() == *depth_clamp_range
            && self.rotated_weights == *rotated_weights
            && self.mesh_last_first == *mesh_last_first
            && self.interlocked_colors == *interlocked_colors
            && self.fragment_tests == *fragment_tests
            && self.extended_shadows == *extended_shadows
            && self.snapshot_textures == *snapshot_textures
            && self.primitive_id_indices == *primitive_id_indices
            && self.quad_primitive_ids == *quad_primitive_ids
            && self.private_list == *private_list
            // Last: the one field compared through a shape made of it.
            && match self.used_resources.is_empty() {
                true => used.is_empty(),
                false => *made.get_or_insert_with(|| self.used_shape()) == *used,
            }
    }

    /// `used_resources` by token ([`BindShape::used`]): one token names one resource, and listing
    /// it twice unions its usage and stages.
    fn used_shape(&self) -> BTreeMap<u32, UsedShape> {
        let mut used: BTreeMap<u32, UsedShape> = BTreeMap::new();
        for u in &self.used_resources {
            let class = match u.class {
                ResourceClass::Buffer => UsedClass::Buffer,
                ResourceClass::Texture(facts) => UsedClass::Texture(facts.shape()),
                ResourceClass::Heap => UsedClass::Heap,
            };
            let e = used.entry(u.token).or_insert(UsedShape {
                class,
                usage: metal::ResourceUsage::empty(),
                stages: metal::RenderStages::empty(),
            });
            e.usage |= u.usage;
            e.stages |= u.stages;
        }
        used
    }
}

/// A compute encoder's imageblock size in pixels (`setImageblockWidth:height:`): a kernel's
/// explicit imageblock holds `width · height` cells for each threadgroup of a dispatch.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageblockSize {
    pub width: u32,
    pub height: u32,
}

impl ImageblockSize {
    /// The size `width` by `height`.
    pub fn new(width: u32, height: u32) -> ImageblockSize {
        ImageblockSize { width, height }
    }
}

/// A render encoder's tile threadgroup memory: each tile of the render pass holds `length` bytes,
/// the pass's `threadgroupMemoryLength`, which the pass keeps across its tile dispatches and starts
/// at zero; a tile function's threadgroup argument `n` is `arguments[n].length` bytes at
/// `arguments[n].offset` in its tile's memory, which the binder places where plain Metal begins it
/// (`bind::plan`). Arguments placed at overlapping offsets share their bytes, as Metal's do.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TileThreadgroupMemory {
    /// The render pass's `threadgroupMemoryLength`: each tile's bytes, before Metal's rounding to
    /// 16.
    pub length: u32,
    /// The encoder's `setThreadgroupMemoryLength:offset:atIndex:` by index.
    pub arguments: BTreeMap<u32, TileThreadgroupArgument>,
}

impl TileThreadgroupMemory {
    /// The tile memory of a pass whose `threadgroupMemoryLength` is `length`, with no argument
    /// set.
    pub fn new(length: u32) -> TileThreadgroupMemory {
        TileThreadgroupMemory {
            length,
            arguments: BTreeMap::new(),
        }
    }
}

/// One tile threadgroup argument (`setThreadgroupMemoryLength:offset:atIndex:`): `length` bytes at
/// byte `offset` of the tile's threadgroup memory.
///
/// Metal's API validation wants both to be multiples of 16, which plain Metal does not: it takes
/// every length and every offset. The length takes no part in the layout and bounds no access, as
/// plain Metal's does not, and the binder places the argument where plain Metal begins it, which is
/// not always `offset` itself (`bind::plan`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TileThreadgroupArgument {
    pub length: u32,
    pub offset: u32,
}

impl TileThreadgroupArgument {
    /// `length` bytes at `offset`.
    pub fn new(length: u32, offset: u32) -> TileThreadgroupArgument {
        TileThreadgroupArgument { length, offset }
    }
}

/// A compute encoder's stage-in region (`setStageInRegion:`, `MTLRegion`), in threads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageInRegion {
    /// The region's origin (`MTLRegion.origin`): `[[grid_origin]]`.
    pub origin: [u32; 3],
    /// The region's size (`MTLRegion.size`): `[[grid_size]]`.
    pub size: [u32; 3],
}

impl StageInRegion {
    /// The region of `size` threads from `origin`.
    pub fn new(origin: [u32; 3], size: [u32; 3]) -> StageInRegion {
        StageInRegion { origin, size }
    }
}

/// A render encoder's vertex amplification (`setVertexAmplificationCount:viewMappings:`): each
/// vertex of a draw is processed `count` times, amplification `a` offset by view mapping `a` when
/// the encoder gave mappings (`viewMappings` of `nil` is empty). The default is Metal's: a count of
/// 1 without mappings.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexAmplification {
    pub count: u32,
    /// `MTLVertexAmplificationViewMapping`s by amplification id; empty for `nil`.
    pub view_mappings: Vec<ViewMapping>,
    /// Draw a count above 1 one amplification at a time even where Vulkan multiview would draw it
    /// as views (`BindingPlan::variant`'s `amplification_draws`): the consumer sets it while a
    /// visibility result mode is set, since a query in a multiview render pass instance takes one
    /// query per view, while draws of one amplification each add to the one query, as Metal counts
    /// the samples of every amplification (a Metal probe on the M3 Max and an M4). Not set by
    /// default.
    #[cfg_attr(feature = "serde", serde(default))]
    pub one_at_a_time: bool,
}

impl VertexAmplification {
    /// A count of `count` with `view_mappings` (empty for `nil`), drawn as views where they
    /// express it.
    pub fn new(count: u32, view_mappings: Vec<ViewMapping>) -> VertexAmplification {
        VertexAmplification {
            count,
            view_mappings,
            one_at_a_time: false,
        }
    }

    /// The view mapping amplification `a` takes under a pipeline whose
    /// `maxVertexAmplificationCount` is `max`: its mapping (a `nil` one's offsets are 0), and
    /// none under a pipeline of maximum count 1, which Metal applies no mapping to (Metal probes on
    /// the M3 Max and an M4: a count of 1 whose mapping offsets the slice or the viewport draws
    /// into the first layer through viewport 0 there, and moves under a pipeline of maximum count
    /// 2).
    pub fn mapping(&self, a: u32, max: u32) -> ViewMapping {
        match max > 1 {
            true => self
                .view_mappings
                .get(a as usize)
                .copied()
                .unwrap_or_default(),
            false => ViewMapping::default(),
        }
    }

    /// The layer of a pass of `renderTargetArrayLength` `length` that amplification `a` renders
    /// into under a pipeline whose `maxVertexAmplificationCount` is `max`, before any slice a
    /// vertex function chooses: its mapping's render target offset ([`Self::mapping`]) where that
    /// is below the pass's layers (`length`, at least 1), and the first layer where it is not (a
    /// Metal probe on an M4, the rule for an index).
    pub fn layer(&self, a: u32, length: u32, max: u32) -> u32 {
        let offset = self.mapping(a, max).render_target_array_index_offset;
        match offset < length.max(1) {
            true => offset,
            false => 0,
        }
    }
}

impl Default for VertexAmplification {
    fn default() -> VertexAmplification {
        VertexAmplification::new(1, Vec::new())
    }
}

/// `MTLVertexAmplificationViewMapping`: the offsets Metal adds to an amplification's viewport and
/// render target array indices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ViewMapping {
    pub viewport_array_index_offset: u32,
    pub render_target_array_index_offset: u32,
}

impl ViewMapping {
    /// The mapping of `viewportArrayIndexOffset` and `renderTargetArrayIndexOffset`.
    pub fn new(viewport_array_index_offset: u32, render_target_array_index_offset: u32) -> Self {
        ViewMapping {
            viewport_array_index_offset,
            render_target_array_index_offset,
        }
    }
}

impl StageTable {
    /// The shape of this stage's tables ([`BindState::shape`]), the bind offsets of its buffers at
    /// [`at_bind_views`](Self::at_bind_views) classed under the uniform views' base alignment
    /// `view_base_alignment` ([`BindState::view_base_alignment`]).
    pub fn shape(&self, view_base_alignment: u32) -> StageShape {
        let buffer = |(i, b): (&u32, &BoundBuffer)| {
            (*i, b.shape(self.class_alignment(*i, view_base_alignment)))
        };
        StageShape {
            buffers: self.buffers.iter().map(buffer).collect(),
            textures: self.textures.iter().map(|(i, t)| (*i, t.shape())).collect(),
            samplers: self.samplers.iter().map(|(i, s)| (*i, s.shape())).collect(),
            threadgroup_lengths: self.threadgroup_lengths.clone(),
        }
    }

    /// Whether its shape is the empty one, which [`BindState::shape`] leaves out.
    fn has_empty_shape(&self) -> bool {
        self.buffers.is_empty()
            && self.textures.is_empty()
            && self.samplers.is_empty()
            && self.threadgroup_lengths.is_empty()
    }

    /// Whether `shape` is its shape ([`BindState::has_shape`]), under the uniform views' base
    /// alignment `view_base_alignment`.
    fn has_shape(&self, shape: &StageShape, view_base_alignment: u32) -> bool {
        let StageShape {
            buffers,
            textures,
            samplers,
            threadgroup_lengths,
        } = shape;
        let buffer_shapes = (self.buffers.iter())
            .map(|(i, b)| (*i, b.shape(self.class_alignment(*i, view_base_alignment))));
        let texture_shapes = self.textures.iter().map(|(i, t)| (*i, t.shape()));
        let sampler_shapes = self.samplers.iter().map(|(i, s)| (*i, s.shape()));
        buffer_shapes.eq(buffers.iter().map(|(i, b)| (*i, *b)))
            && texture_shapes.eq(textures.iter().map(|(i, t)| (*i, *t)))
            && sampler_shapes.eq(samplers.iter().map(|(i, s)| (*i, *s)))
            && self.threadgroup_lengths == *threadgroup_lengths
    }

    /// The alignment above the vertex fetch's that the bind offset class of a buffer at index
    /// `index` tells apart ([`BoundBuffer::shape`]): `view_base_alignment` at a slot of
    /// [`at_bind_views`](Self::at_bind_views), else 0 (none).
    pub fn class_alignment(&self, index: u32, view_base_alignment: u32) -> u32 {
        let viewed = index < u64::BITS && (self.at_bind_views >> index) & 1 == 1;
        if viewed {
            view_base_alignment
        } else {
            0
        }
    }
}

impl BoundBuffer {
    /// The shape of this binding, its bind offset classed under the uniform views' base alignment
    /// `view_base_alignment` (`BufferShape::Buffer`'s `offset_alignment`).
    pub fn shape(&self, view_base_alignment: u32) -> BufferShape {
        match *self {
            BoundBuffer::Buffer {
                length,
                offset,
                device_address,
                attribute_stride,
                ..
            } => BufferShape::Buffer {
                attribute_stride,
                device_address: device_address.is_some(),
                length_class: length_class(length),
                offset_alignment: offset_alignment(offset, view_base_alignment),
            },
            BoundBuffer::Bytes {
                attribute_stride, ..
            } => BufferShape::Bytes { attribute_stride },
        }
    }
}

/// The alignment class of bind offset `offset` (a buffer's,
/// [`BufferShape::Buffer::offset_alignment`]): `view_base_alignment` where that divides it (the
/// uniform views' descriptor base alignment, [`BindState::view_base_alignment`]: a view's range
/// then starts at the offset, `SpecKey::RangeAtBindOffset`), else the largest power of two up to
/// [`MAX_VERTEX_ALIGNMENT`](metal::vertex::MAX_VERTEX_ALIGNMENT) that divides it (whether
/// fixed-function vertex input can fetch every attribute format at a buffer bound there). These
/// are the only predicates a plan reads of a bind offset, so offsets moving between multiples of
/// 4 that the base alignment does not divide, or between those it divides, keep one shape; a
/// `view_base_alignment` of 4 or less (0, the default) tells only the vertex fetch's apart.
pub fn offset_alignment(offset: u64, view_base_alignment: u32) -> u32 {
    if view_base_alignment > metal::vertex::MAX_VERTEX_ALIGNMENT
        && offset.is_multiple_of(u64::from(view_base_alignment))
    {
        return view_base_alignment;
    }
    let mut a = metal::vertex::MAX_VERTEX_ALIGNMENT;
    while a > 1 && !offset.is_multiple_of(u64::from(a)) {
        a /= 2;
    }
    a
}

impl BoundTexture {
    /// The shape of this binding.
    pub fn shape(&self) -> TextureShape {
        match *self {
            BoundTexture::Texture { facts, .. } => facts.shape(),
            BoundTexture::Buffer {
                pixel_format,
                swizzle,
                ..
            } => TextureShape::Buffer {
                pixel_format,
                swizzle,
            },
        }
    }
}

impl TextureFacts {
    /// The facts of an image texture that can change a plan (its view facts): the type, the pixel
    /// format, the sample count, the usage, whether it is sparse, its level count, which decides a
    /// storage slot's level views and pushed level count, its swizzle, its address-mode map, its
    /// [`offset_base_axes`](Self::offset_base_axes), and whether its planes are read in place
    /// ([`ycbcr_in_place`](Self::ycbcr_in_place), whose words the slot's push field carries, not
    /// the shape) and through which device conversion where they are held as an image. Its array length only changes the view's subresource range, which is every
    /// layer.
    ///
    /// The level count is stated whatever the usage: plain Metal writes a texture made without
    /// `ShaderWrite` at every level, through every write form, as one made with it.
    /// A plan is keyed by [`BindShape::for_plan`], which keeps it only at the slots whose levels
    /// the pipeline reads, so textures a function only samples do not split plans by their level
    /// counts.
    pub fn shape(&self) -> TextureShape {
        TextureShape::Texture {
            texture_type: self.texture_type,
            pixel_format: self.pixel_format,
            sample_count: self.sample_count,
            usage: self.usage,
            mipmap_level_count: self.mipmap_level_count,
            sparse: self.sparse,
            swizzle: self.swizzle,
            address_modes: self.address_modes,
            offset_base_axes: self.offset_base_axes(),
            ycbcr_in_place: self.ycbcr_in_place.is_some(),
            ycbcr_planes_sampler: self.ycbcr_in_place.and_then(|p| p.planes_sampler()),
        }
    }
}

impl BoundSampler {
    /// The shape of this binding: its descriptor's [`SamplerDescriptor::shape`] and its
    /// [`SamplerWord`](metal::sampler::SamplerWord), which the slot's specialization reads.
    pub fn shape(&self) -> BoundSamplerShape {
        BoundSamplerShape {
            shape: self.desc.shape(),
            word: metal::sampler::SamplerWord::of(&self.desc),
            min_level: metal::sampler::query_min_level(&self.desc),
            degree: metal::sampler::anisotropy_degree(&self.desc),
        }
    }
}

impl SamplerDescriptor {
    /// The facts of this descriptor that can change a plan wherever the sampler is bound: the
    /// reduction mode and comparison, whether each value the Vulkan sampler cannot take as is is
    /// present, whether its border is opaque black, and whether it filters linearly, so a plan
    /// refuses at plan time what `bind::sampler_desc` would refuse per bind and what the sampler's
    /// textures' views cannot read. A slot's sampler also specializes its pipeline by its word
    /// ([`BoundSampler::shape`]); an argument buffer's carries its word in the sampler table.
    pub fn shape(&self) -> SamplerShape {
        use metal::{SamplerAddressMode as A, SamplerBorderColor as B};
        let modes = [
            self.s_address_mode,
            self.t_address_mode,
            self.r_address_mode,
        ];
        // `ClampToZero` reads the border color as `ClampToBorderColor` does.
        let bordered = modes
            .iter()
            .any(|m| matches!(m, A::ClampToZero | A::ClampToBorderColor));
        let (lo, hi) = (self.lod_min_clamp, self.lod_max_clamp);
        SamplerShape {
            reduction: self.reduction_mode,
            compare: self.compare_function != metal::CompareFunction::Never,
            anisotropic: self.max_anisotropy > 1,
            mirror_clamp_to_edge: modes.contains(&A::MirrorClampToEdge),
            valid_lod_clamps: !(lo.is_nan() || hi.is_nan() || lo > hi),
            opaque_black_border: bordered && self.border_color == B::OpaqueBlack,
            linear: self.min_filter == metal::SamplerMinMagFilter::Linear
                || self.mag_filter == metal::SamplerMinMagFilter::Linear
                || self.mip_filter == metal::SamplerMipFilter::Linear,
            cubic: self.min_filter == metal::SamplerMinMagFilter::Cubic
                || self.mag_filter == metal::SamplerMinMagFilter::Cubic,
        }
    }
}

/// One Metal stage's argument tables (`buffers[31]`, `textures[128]`, `samplers[16]`,
/// `threadgroup_lengths[31]`), sparse by index.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageTable {
    pub buffers: BTreeMap<u32, BoundBuffer>,
    pub textures: BTreeMap<u32, BoundTexture>,
    pub samplers: BTreeMap<u32, BoundSampler>,
    /// `setThreadgroupMemoryLength:atIndex:`, in bytes as the guest set them.
    pub threadgroup_lengths: BTreeMap<u32, u32>,
    /// The acceleration structures bound at buffer slots (`setAccelerationStructure:
    /// atBufferIndex:`), by buffer index: what the consumer knows of the Metal structure it holds.
    /// A function that reads a structure's instance count reads the count stated here
    /// (`PushField::InstanceCount`), 0 for a slot absent from this table. A structure the function
    /// intersects is still the consumer's to build and bind (`WriteSource::AccelerationStructure`);
    /// only the counts are read from this table, per dispatch or draw, so it is not part of the
    /// shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub acceleration_structures: BTreeMap<u32, BoundAccelerationStructure>,
    /// The buffer slots (bit `i` for index `i`) whose bind offset class tells
    /// [`BindState::view_base_alignment`] apart, which the runtime sets: the
    /// slots its function reads through a uniform view, whose range a plan starts at the bind
    /// offset where that alignment divides it (`SpecKey::RangeAtBindOffset`). No plan reads that
    /// predicate of another slot, so its offsets moving among multiples of 4 keep one shape. A slot
    /// left out (none, the default) reads its view at its residual, exact at every offset: a
    /// runtime leaves out a slot it finds bound at moving offsets, so that its draws keep one shape.
    #[cfg_attr(feature = "serde", serde(default))]
    pub at_bind_views: u64,
}

/// What the consumer knows of a Metal acceleration structure bound at a buffer slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundAccelerationStructure {
    /// The number of instances of an instance structure as it holds them when the command that
    /// reads it runs: the `instanceCount` of the `MTLInstanceAccelerationStructureDescriptor` it
    /// was last built with, which is what Metal's `get_instance_count()` reads. The binder pushes
    /// it when the command is recorded, so it must be known then: for a structure built from an
    /// indirect descriptor (its count in `instanceCountBuffer`), only once the consumer has waited
    /// for that build and read the count back (docs/API.md); without that wait it is stale. 0 for
    /// a primitive structure.
    pub instance_count: u32,
}

impl BoundAccelerationStructure {
    /// A structure of `instance_count` instances.
    pub fn new(instance_count: u32) -> BoundAccelerationStructure {
        BoundAccelerationStructure { instance_count }
    }
}

/// What is bound at a buffer slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundBuffer {
    /// A buffer at an offset.
    Buffer {
        /// The allocation length (the consumer pads the backing `VkBuffer` to 16 bytes).
        length: u64,
        offset: u64,
        /// The Metal `gpuAddress` the guest sees.
        gpu_address: u64,
        /// The host Vulkan device address, when the buffer has one.
        device_address: Option<u64>,
        /// `setVertexBuffer:offset:attributeStride:`.
        attribute_stride: Option<u32>,
    },
    /// `set*Bytes`: inline bytes the consumer uploads.
    Bytes {
        len: u32,
        /// `setVertexBytes:length:attributeStride:atIndex:` (and a compute encoder's
        /// `setBytes:length:attributeStride:atIndex:`): the stride a layout of
        /// `MTLBufferLayoutStrideDynamic` reads the bytes at; `None` for the plain setters, which
        /// such a layout reads at stride 0, as plain Metal reads them.
        /// `MTLAttributeStrideStatic` is `Some(u32::MAX)`, its low 32 bits: an M4 read element 0
        /// of inline bytes bound so and zeros past it, where a buffer bound so reads stride 0;
        /// the binder refuses a layout that reads inline bytes at that stride
        /// ([`BindError::MissingVertexStride`](crate::error::BindError::MissingVertexStride), and
        /// a kernel's stage-in `NotImplemented(KernelStageIn)`).
        #[cfg_attr(feature = "serde", serde(default))]
        attribute_stride: Option<u32>,
    },
}

/// What is bound at a texture slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundTexture {
    /// An image texture.
    Texture {
        facts: TextureFacts,
        /// The guest `gpuResourceID`.
        resource_id: Option<u64>,
    },
    /// A texture buffer: a texel view of a buffer range.
    Buffer {
        pixel_format: metal::PixelFormat,
        offset: u64,
        length: u64,
        bytes_per_row: u64,
        resource_id: Option<u64>,
        /// `MTLTexture.swizzle`: the source of each channel a function's reads of it return.
        #[cfg_attr(feature = "serde", serde(default))]
        swizzle: metal::TextureSwizzleChannels,
    },
}

/// The view facts of an image texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureFacts {
    pub texture_type: metal::TextureType,
    pub pixel_format: metal::PixelFormat,
    pub sample_count: u32,
    pub usage: metal::TextureUsage,
    pub array_length: u32,
    pub mipmap_level_count: u32,
    /// `MTLTexture.isSparse`: the texture was made from a sparse heap, so regions of it may be
    /// unmapped. A function that observes the residency of its accesses of the texture
    /// (`sparse_color::resident()`, `RuntimeFact::SparseTexture`) answers a texture that is not
    /// sparse on every device, and a sparse one only where the module has its sparse form
    /// (`shaderResourceResidency`); the plan refuses it elsewhere.
    #[cfg_attr(feature = "serde", serde(default))]
    pub sparse: bool,
    /// `MTLTexture.swizzle`: the source of each channel the texture's reads and samples return,
    /// border texels included. A sample emulated by fetches reads its border taps through it, and
    /// an opaque-black border the device replaces is checked against it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub swizzle: metal::TextureSwizzleChannels,
    /// The texture's address-mode map (`metal::sampler::AddressModeMap`, the runtime's
    /// extension): a slot a constexpr sampler samples takes it through the pipeline variant
    /// (`VariantKey::address_maps`) where it is not the identity.
    #[cfg_attr(feature = "serde", serde(default))]
    pub address_modes: metal::sampler::AddressModeMap,
    /// The level of the texture whose levels a view presents at which the view's first level
    /// lies (`MTLTexture.parentRelativeLevel`): 0 for a texture that is not a view, and for a
    /// view that starts at its parent's base level.
    #[cfg_attr(feature = "serde", serde(default))]
    pub parent_relative_level: u32,
    /// The base-level size, `[width, height, depth]`, of the texture whose levels a view presents
    /// (`MTLTexture.parentTexture`'s, the texture whose image holds the view's texels), where
    /// [`parent_relative_level`](Self::parent_relative_level) is not 0; unread where it is.
    /// Metal's texel-offset rule reads it, not the view's own: a slot whose function places an
    /// offset by that rule specializes on the axes where the two differ in being powers of two
    /// (`SpecKey::OffsetBaseAxes`, [`TextureFacts::offset_base_axes`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub parent_base_size: [u32; 3],
    /// The texture's two planes, where a function reads them in place
    /// ([`YCbCrInPlace`](crate::bind::YCbCrInPlace), the runtime's extension): a two-plane YCbCr
    /// texture of a converted 4:2:0 format whose planes the consumer views as one texel buffer.
    /// A slot whose function reads it only as the in-place form reads
    /// (`RuntimeFact::YCbCrInPlace`) takes its in-place variant (`VariantKey::ycbcr_in_place`),
    /// which binds that texel buffer ([`WriteSource::YCbCrInPlace`](crate::bind::WriteSource))
    /// and pushes these words (`PushField::YCbCrInPlace`); any other slot binds the texture's
    /// image. `None` for every other texture.
    #[cfg_attr(feature = "serde", serde(default))]
    pub ycbcr_in_place: Option<crate::bind::YCbCrInPlace>,
}

impl TextureFacts {
    /// A single-sample, single-level, single-layer texture.
    pub fn new(texture_type: metal::TextureType, pixel_format: metal::PixelFormat) -> Self {
        TextureFacts {
            texture_type,
            pixel_format,
            sample_count: 1,
            usage: metal::TextureUsage::SHADER_READ,
            array_length: 1,
            mipmap_level_count: 1,
            sparse: false,
            swizzle: metal::TextureSwizzleChannels::DEFAULT,
            address_modes: metal::sampler::AddressModeMap::IDENTITY,
            parent_relative_level: 0,
            parent_base_size: [0; 3],
            ycbcr_in_place: None,
        }
    }

    /// The `SpecKey::OffsetBaseAxes` of a slot bound to this texture: bit `a` for each plane axis
    /// `a` of its type (a cube's and a texture buffer's none, which take no texel offset) on which
    /// it is a view whose first level lies above its parent's base level
    /// ([`parent_relative_level`](Self::parent_relative_level)), whose parent's base-level size
    /// ([`parent_base_size`](Self::parent_base_size)) is not a power of two where its own,
    /// `max(1, parent >> parent_relative_level)`, is, and some level of which has size 2 or 4.
    /// Metal drops a texel offset on a repeating or mirrored axis whose level size lies in
    /// `[2, 7]` and whose base-level size, the parent's, is not a power of two: such an axis's
    /// levels, powers of two, drop it at sizes 2 and 4 under that parent and keep it under the
    /// view's own base. On every other axis the view's own base-level size decides as the parent's
    /// does, so the bit is 0 and the texture shares its pipelines.
    pub fn offset_base_axes(&self) -> u8 {
        use metal::TextureType as T;
        let level = self.parent_relative_level;
        let axes = match self.texture_type {
            _ if level == 0 => 0,
            T::Type1D | T::Type1DArray => 1,
            T::Type2D | T::Type2DArray | T::Type2DMultisample | T::Type2DMultisampleArray => 2,
            T::Type3D => 3,
            _ => 0,
        };
        let last = self.mipmap_level_count.max(1) - 1;
        let mut bits = 0;
        for (a, &parent) in self.parent_base_size.iter().enumerate().take(axes) {
            let own = parent.checked_shr(level).unwrap_or(0).max(1);
            let smallest = own.checked_shr(last).unwrap_or(0).max(1);
            let small = own >= 2 && smallest <= 4;
            if parent != 0 && !parent.is_power_of_two() && own.is_power_of_two() && small {
                bits |= 1 << a;
            }
        }
        bits
    }
}

/// What is bound at a sampler slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundSampler {
    pub desc: SamplerDescriptor,
    /// The guest `gpuResourceID`.
    pub resource_id: Option<u64>,
}

/// The facts of an `MTLSamplerDescriptor`. [`sampler_desc`](crate::bind::sampler_desc) maps a
/// bound one to a `vk::SamplerDesc`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerDescriptor {
    pub min_filter: metal::SamplerMinMagFilter,
    pub mag_filter: metal::SamplerMinMagFilter,
    pub mip_filter: metal::SamplerMipFilter,
    pub s_address_mode: metal::SamplerAddressMode,
    pub t_address_mode: metal::SamplerAddressMode,
    pub r_address_mode: metal::SamplerAddressMode,
    pub border_color: metal::SamplerBorderColor,
    pub normalized_coordinates: bool,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub lod_average: bool,
    pub max_anisotropy: u32,
    /// `Never` (the Metal default) on a sampler used without comparison.
    pub compare_function: metal::CompareFunction,
    pub reduction_mode: metal::SamplerReductionMode,
}

impl SamplerDescriptor {
    /// `MTLSamplerDescriptor`'s defaults: nearest filtering without mipmaps, clamp to edge,
    /// transparent black borders, normalized coordinates, LOD clamps `[0, FLT_MAX]`, no
    /// anisotropy, no comparison, and weighted-average reduction. The fields are public; set the
    /// ones that differ.
    pub fn new() -> SamplerDescriptor {
        SamplerDescriptor {
            min_filter: metal::SamplerMinMagFilter::Nearest,
            mag_filter: metal::SamplerMinMagFilter::Nearest,
            mip_filter: metal::SamplerMipFilter::NotMipmapped,
            s_address_mode: metal::SamplerAddressMode::ClampToEdge,
            t_address_mode: metal::SamplerAddressMode::ClampToEdge,
            r_address_mode: metal::SamplerAddressMode::ClampToEdge,
            border_color: metal::SamplerBorderColor::TransparentBlack,
            normalized_coordinates: true,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            lod_average: false,
            max_anisotropy: 1,
            compare_function: metal::CompareFunction::Never,
            reduction_mode: metal::SamplerReductionMode::WeightedAverage,
        }
    }
}

impl Default for SamplerDescriptor {
    fn default() -> SamplerDescriptor {
        SamplerDescriptor::new()
    }
}

impl BoundSampler {
    /// A sampler of `desc` without a resource id.
    pub fn new(desc: SamplerDescriptor) -> BoundSampler {
        BoundSampler {
            desc,
            resource_id: None,
        }
    }
}

/// A resource made resident with `useResource`/`useHeap`, or otherwise live in the guest task.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UsedResource {
    pub class: ResourceClass,
    /// The Metal `gpuAddress` (buffers and heaps).
    pub gpu_address: Option<u64>,
    pub length: u64,
    /// The host Vulkan device address, when it has one.
    pub device_address: Option<u64>,
    pub usage: metal::ResourceUsage,
    /// Render stages; empty in compute encoders.
    pub stages: metal::RenderStages,
    /// The consumer's id for the resource; plans refer to it (`UseTarget::Used`,
    /// `HeapSource::Used`).
    pub token: u32,
    /// The guest `gpuResourceID`.
    pub resource_id: Option<u64>,
}

impl UsedResource {
    /// A used resource of `class` with the consumer's `token`: no addresses, length 0, no usage
    /// or stages, no resource id. The fields are public; set the ones that apply.
    pub fn new(class: ResourceClass, token: u32) -> UsedResource {
        UsedResource {
            class,
            gpu_address: None,
            length: 0,
            device_address: None,
            usage: metal::ResourceUsage::default(),
            stages: metal::RenderStages::default(),
            token,
            resource_id: None,
        }
    }
}

/// What kind of resource a [`UsedResource`] is.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceClass {
    Buffer,
    Texture(TextureFacts),
    Heap,
}

/// The length class of an allocation of `length` bytes: the bit length of its padded length, so
/// `padded(length) < 2^class`. A plan compares `2^class - 1` with a descriptor range limit to know
/// that every range of the buffer fits it without the exact length, which is per-draw data.
pub fn length_class(length: u64) -> u8 {
    let padded = u128::from(length).next_multiple_of(16);
    (u128::BITS - padded.leading_zeros()) as u8
}

/// The projection of a [`BindState`] that can change a [`BindingPlan`](super::plan::BindingPlan):
/// which slots are bound and with what kind of resource, texture types and formats, sampler modes,
/// threadgroup lengths, inline bytes, per-bind strides, buffer length and bind-offset alignment
/// classes, the used resources' classes, usages, and stages by consumer token, the vertex
/// amplification, the imageblock size, the render pass's render target array length, whether a
/// tessellation factor buffer is set, a tile dispatch's render area, and the tile threadgroup
/// memory. Hashable; plans memoize on `(PipelineKey, BindShape)`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BindShape {
    pub stages: BTreeMap<Stage, StageShape>,
    /// `BindState.used_resources` by consumer token: what a plan lists in its `uses` for the
    /// resources a function can reach through the root or handle table.
    pub used: BTreeMap<u32, UsedShape>,
    /// `BindState.vertex_amplification`: a render plan's views and amplification count.
    #[cfg_attr(feature = "serde", serde(default))]
    pub vertex_amplification: VertexAmplification,
    /// `BindState.imageblock_size`: a compute plan's imageblock specialization.
    #[cfg_attr(feature = "serde", serde(default))]
    pub imageblock_size: ImageblockSize,
    /// `BindState.render_target_array_length`: whether a vertex function's render target slice
    /// renders into one layer (0 or 1) or selects among layers.
    #[cfg_attr(feature = "serde", serde(default))]
    pub render_target_array_length: u32,
    /// The encoder set a tessellation factor buffer (`BindState.tessellation_factors`): a patch
    /// draw's control stage reads it, or a zero buffer.
    #[cfg_attr(feature = "serde", serde(default))]
    pub tessellation_factor_buffer: bool,
    /// `BindState.render_area`: a tile dispatch's render area specialization.
    #[cfg_attr(feature = "serde", serde(default))]
    pub render_area: [u32; 2],
    /// `BindState.argument_samplers`: the sampler arrays' and tables' length class and the
    /// samplers' shapes.
    #[cfg_attr(feature = "serde", serde(default))]
    pub argument_samplers: ArgumentSamplersShape,
    /// `BindState.function_tables`: the function-table array's and directories' length class.
    #[cfg_attr(feature = "serde", serde(default))]
    pub function_tables: FunctionTablesShape,
    /// `BindState.tile_threadgroup`: a tile function's threadgroup offsets and tile memory
    /// specialization.
    #[cfg_attr(feature = "serde", serde(default))]
    pub tile_threadgroup: TileThreadgroupMemory,
    /// `BindState.viewport_compensation`'s variant: the vertex function a draw through a viewport
    /// Vulkan's cannot be runs (`VariantKey::viewport_compensation`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub viewport_compensation: Option<ViewportCompensation>,
    /// `BindState.zeroed_colors`: the variant whose pipeline has those attachments and whose
    /// fragment function writes zero to them (`VariantKey::zeroed_colors`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub zeroed_colors: BTreeMap<u32, metal::PixelFormat>,
    /// `BindState.dropped_colors`: the variant whose fragment function keeps no output at them
    /// (`VariantKey::dropped_colors`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub dropped_colors: BTreeSet<u32>,
    /// `BindState.kept_imageblock_bytes`: the variant whose tile function reads its explicit
    /// imageblock's words past them as zero and writes them nowhere
    /// (`VariantKey::kept_imageblock_bytes`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub kept_imageblock_bytes: Option<u32>,
    /// Whether `BindState.depth_clamp_range` is set: the variant whose fragment function clamps
    /// the depth it writes to the draw's range itself (`VariantKey::depth_clamp_range`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_clamp_range: bool,
    /// `BindState.rotated_weights`: the variant whose fragment function takes Metal's barycentric
    /// weights back from triangles drawn with Metal's last vertex first
    /// (`VariantKey::rotated_weights`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_weights: bool,
    /// `BindState.mesh_last_first`: the variant whose emulated mesh pipeline's synthesized
    /// vertex stage draws each line or triangle from its last index first
    /// (`VariantKey::mesh_last_first`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub mesh_last_first: bool,
    /// `BindState.interlocked_colors`: the variant whose fragment function writes those it writes
    /// through the interlocked storage form (`VariantKey::interlocked_colors`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub interlocked_colors: BTreeSet<u32>,
    /// `BindState.fragment_tests`: whether a fragment function that stores before the tests runs
    /// as compiled, runs the variant that declares early tests (`VariantKey::read_only_tests`),
    /// or is refused.
    #[cfg_attr(feature = "serde", serde(default))]
    pub fragment_tests: FragmentTests,
    /// `BindState.extended_shadows`: the variant whose fragment function reads and writes those
    /// it reads or writes in the storage form against their shadows
    /// (`VariantKey::extended_shadows`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub extended_shadows: BTreeSet<u32>,
    /// `BindState.snapshot_textures`: the variant whose fragment function reads those of them
    /// the pipeline can read through a snapshot so (`VariantKey::snapshot_textures`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub snapshot_textures: BTreeSet<u32>,
    /// `BindState.primitive_id_indices`: the primitive ID's de-indexed variant of an indexed
    /// draw's index type (`VariantKey::primitive_id_indices`), where the pipeline writes the ID.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_id_indices: Option<metal::IndexType>,
    /// `BindState.quad_primitive_ids`: a draw of Metal's private quads or quad strip, whose
    /// pipeline's fragment function reads `[[primitive_id]]` from Vulkan's `PrimitiveId`, runs the
    /// variant that reads the quad's (`VariantKey::quad_primitive_ids`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub quad_primitive_ids: bool,
    /// `BindState.private_list`: a draw of one of Metal's private primitive types in the pulled
    /// list form runs the variant that draws it so (`VariantKey::private_list`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub private_list: bool,
}

impl BindShape {
    /// The shape a plan of a pipeline whose runtime dependencies are `dependencies` is planned
    /// and keyed by: this shape with each bound texture's level count 1 but at the slots whose
    /// levels the pipeline reads (`RuntimeFact::StorageTextureLevels` and
    /// `StorageTextureLevelViews`), and each used texture's 1 (a heap array views level 0 alone).
    /// The plan reads a texture's level count only at those slots, so it is the same plan, and
    /// bound textures of other level counts that a function only samples, or writes at level 0,
    /// share it. Borrowed where nothing changes.
    pub fn for_plan(&self, dependencies: &[RuntimeDependency]) -> Cow<'_, BindShape> {
        let reads_levels = |stage: Stage, index: u32| {
            dependencies.iter().any(|d| match d.fact {
                RuntimeFact::StorageTextureLevels { slot }
                | RuntimeFact::StorageTextureLevelViews { slot } => {
                    slot.stage == stage && slot.class == SlotClass::Texture && slot.index == index
                }
                _ => false,
            })
        };
        let unread = |stage: Stage, index: u32, shape: &TextureShape| {
            matches!(shape, TextureShape::Texture { mipmap_level_count, .. } if *mipmap_level_count != 1)
                && !reads_levels(stage, index)
        };
        let used = |shape: &UsedShape| {
            matches!(
                shape.class,
                UsedClass::Texture(TextureShape::Texture { mipmap_level_count, .. })
                    if mipmap_level_count != 1
            )
        };
        let changes = self.stages.iter().any(|(stage, table)| {
            table
                .textures
                .iter()
                .any(|(index, shape)| unread(*stage, *index, shape))
        }) || self.used.values().any(used);
        if !changes {
            return Cow::Borrowed(self);
        }
        let one_level = |shape: &mut TextureShape| *shape = shape.at_one_level();
        let mut projected = self.clone();
        for (stage, table) in &mut projected.stages {
            for (index, shape) in &mut table.textures {
                if unread(*stage, *index, shape) {
                    one_level(shape);
                }
            }
        }
        for shape in projected.used.values_mut() {
            if let UsedClass::Texture(texture) = &mut shape.class {
                one_level(texture);
            }
        }
        Cow::Owned(projected)
    }
}

/// The projection of `BindState::argument_samplers` that can change a plan: the capacity class of
/// the list's length, and the distinct shapes of its samplers. A plan holds
/// [`ArgumentSamplersShape::entries`] entries in each sampler table and sampler array, so a
/// sampler made or released changes the shape (and needs a new plan) only when the count leaves
/// its class or the last sampler of a shape goes or a new shape comes; every other change is
/// per-draw bytes and the per-draw resolution of `WriteSource::ArgumentSampler`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArgumentSamplersShape {
    /// The capacity class: the list's length plus the null entry is at most
    /// `MIN_ENTRIES << class` (the least such class).
    pub class: u8,
    /// The distinct shapes of the listed samplers: every one is checked for every use of a plan's
    /// sampler arrays (`metal::sampler::check_argument_sampler`), and an opaque-black border meets
    /// every member a sampled heap array holds.
    pub shapes: std::collections::BTreeSet<SamplerShape>,
}

impl ArgumentSamplersShape {
    /// The least entries of a class: a sampler table never holds fewer.
    pub const MIN_ENTRIES: u64 = 16;

    /// The shape of the argument samplers `samplers`.
    pub fn of(samplers: &[ArgumentSampler]) -> ArgumentSamplersShape {
        ArgumentSamplersShape {
            class: capacity_class(samplers.len()),
            shapes: samplers.iter().map(|s| s.desc.shape()).collect(),
        }
    }

    /// Whether it is the shape of `samplers` (`ArgumentSamplersShape::of(samplers) == *self`),
    /// answered without making that shape for up to [`SHAPES_AT_ONCE`](Self::SHAPES_AT_ONCE)
    /// samplers: each sampler's shape is among `shapes`, and each of `shapes` is some sampler's.
    /// More are answered by making it.
    fn is_of(&self, samplers: &[ArgumentSampler]) -> bool {
        if samplers.len() > Self::SHAPES_AT_ONCE {
            return Self::of(samplers) == *self;
        }
        let mut seen = [None; Self::SHAPES_AT_ONCE];
        for (shape, sampler) in seen.iter_mut().zip(samplers) {
            *shape = Some(sampler.desc.shape());
        }
        let seen = &seen[..samplers.len()];
        self.class == capacity_class(samplers.len())
            && seen
                .iter()
                .flatten()
                .all(|shape| self.shapes.contains(shape))
            && self.shapes.iter().all(|shape| seen.contains(&Some(*shape)))
    }

    /// The most samplers [`is_of`](Self::is_of) answers for without making their shape.
    const SHAPES_AT_ONCE: usize = 64;

    /// The entries of the class (the null entry included): `MIN_ENTRIES << class`.
    pub fn entries(&self) -> u64 {
        Self::MIN_ENTRIES << self.class
    }

    /// The least count of samplers plus the null entry a list of this class has: one more than
    /// the previous class's entries.
    pub fn least_count(&self) -> u64 {
        least_count(self.class)
    }
}

/// The least capacity class (`MIN_ENTRIES << class` entries) that holds `len` listed objects
/// plus the null entry.
fn capacity_class(len: usize) -> u8 {
    let count = len as u64 + 1;
    let mut class = 0u8;
    while (ArgumentSamplersShape::MIN_ENTRIES << class) < count {
        class += 1;
    }
    class
}

/// The least count of listed objects plus the null entry of capacity class `class`: one more
/// than the previous class's entries.
fn least_count(class: u8) -> u64 {
    match class {
        0 => 1,
        c => (ArgumentSamplersShape::MIN_ENTRIES << (c - 1)) + 1,
    }
}

/// The projection of `BindState::function_tables` that can change a plan: the capacity class of the
/// list's length, as [`ArgumentSamplersShape`]'s. A plan holds [`FunctionTablesShape::entries`]
/// entries in each function-table directory and the function-table array, so a table made or
/// released changes the shape only when the count leaves its class.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionTablesShape {
    /// The capacity class: the list's length plus the null entry is at most
    /// `ArgumentSamplersShape::MIN_ENTRIES << class` (the least such class).
    pub class: u8,
    /// The listed tables' bytes as the function-table snapshot holds them (the null record,
    /// then each table's word count and words), at least any one table's words: 0 where they are
    /// fewer than `2^27`, Vulkan's least `maxStorageBufferRange`, which every device reaches, else
    /// their bit length, so that no listed table, nor the snapshot, passes a range of
    /// `2^bytes_class - 1` or more ([`FunctionTablesShape::may_pass`]). A table made or released
    /// changes it only when those bytes leave their power of two past 128 MiB.
    #[cfg_attr(feature = "serde", serde(default))]
    pub bytes_class: u8,
}

impl FunctionTablesShape {
    /// The bit length of Vulkan's least `maxStorageBufferRange`, `2^27` bytes: tables of fewer
    /// bytes fit every device's ([`FunctionTablesShape::bytes_class`] 0).
    const EVERY_RANGE_BITS: u32 = 27;

    /// The shape of the tables `tables`.
    pub fn of(tables: &[ArgumentFunctionTable]) -> FunctionTablesShape {
        let bytes = tables.iter().fold(4u64, |bytes, t| {
            bytes.saturating_add(4 * (u64::from(t.size) + 3))
        });
        let bits = u64::BITS - bytes.leading_zeros();
        FunctionTablesShape {
            class: capacity_class(tables.len()),
            bytes_class: match bits <= Self::EVERY_RANGE_BITS {
                true => 0,
                false => bits as u8,
            },
        }
    }

    /// Whether a listed table's words, or their snapshot, may pass `max_storage_buffer_range`
    /// bytes: whether `2^bytes_class - 1` does, never for class 0.
    pub fn may_pass(&self, max_storage_buffer_range: u32) -> bool {
        if self.bytes_class == 0 {
            return false;
        }
        let most = 1u128
            .checked_shl(u32::from(self.bytes_class))
            .map_or(u128::MAX, |b| b - 1);
        most > u128::from(max_storage_buffer_range)
    }

    /// The entries of the class (the null entry included).
    pub fn entries(&self) -> u64 {
        ArgumentSamplersShape::MIN_ENTRIES << self.class
    }

    /// The least count of tables plus the null entry a list of this class has.
    pub fn least_count(&self) -> u64 {
        least_count(self.class)
    }
}

/// The shape of one used resource: what a plan's `uses` states about it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UsedShape {
    pub class: UsedClass,
    pub usage: metal::ResourceUsage,
    /// Render stages; empty in compute encoders.
    pub stages: metal::RenderStages,
}

/// The kind of a used resource; a texture with its view facts (a heap member's element and
/// views depend on them).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UsedClass {
    Buffer,
    Texture(TextureShape),
    Heap,
}

/// The shape of one stage's argument tables.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageShape {
    pub buffers: BTreeMap<u32, BufferShape>,
    pub textures: BTreeMap<u32, TextureShape>,
    pub samplers: BTreeMap<u32, BoundSamplerShape>,
    pub threadgroup_lengths: BTreeMap<u32, u32>,
}

/// The shape of a bound buffer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BufferShape {
    Buffer {
        /// The per-bind stride (`setVertexBuffer:offset:attributeStride:`): whether fixed-function
        /// vertex input can fetch with it decides the pulled-vertex-buffer variant.
        attribute_stride: Option<u32>,
        device_address: bool,
        /// The length class of the allocation, as [`BindState::shape`] computes it.
        length_class: u8,
        /// The alignment class of the bind offset, as [`BindState::shape`] computes it: the
        /// uniform views' base alignment ([`BindState::view_base_alignment`]) where the slot
        /// tells it apart ([`StageTable::at_bind_views`]) and it divides the offset, else the
        /// largest power of two up to 4 that divides it.
        offset_alignment: u32,
    },
    /// Inline bytes (`set*Bytes`).
    Bytes {
        /// The per-bind stride ([`BoundBuffer::Bytes`]), which decides the pulled-vertex-buffer
        /// variant as a buffer's does.
        #[cfg_attr(feature = "serde", serde(default))]
        attribute_stride: Option<u32>,
    },
}

/// The level count of a texture shape stated without one.
#[cfg(feature = "serde")]
fn one_level() -> u32 {
    1
}

/// The shape of a bound texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureShape {
    Texture {
        texture_type: metal::TextureType,
        pixel_format: metal::PixelFormat,
        sample_count: u32,
        /// Which views the texture admits: a heap writes a sampled view only of a
        /// `ShaderRead` texture and a storage view only of a `ShaderWrite` or `BlockWritesOnly`
        /// one.
        usage: metal::TextureUsage,
        /// The texture's `mipmapLevelCount` ([`TextureFacts::shape`]; 1 at a slot whose levels
        /// the pipeline does not read, in [`BindShape::for_plan`]): a storage slot's level views
        /// view each level, and a neutral past the last.
        #[cfg_attr(feature = "serde", serde(default = "one_level"))]
        mipmap_level_count: u32,
        /// The texture is sparse (`MTLTexture.isSparse`, [`TextureFacts::sparse`]): it selects
        /// the sparse form of a slot whose residency the function observes, or refuses the plan
        /// where the module has none.
        #[cfg_attr(feature = "serde", serde(default))]
        sparse: bool,
        /// The texture's swizzle ([`TextureFacts::swizzle`]): the border taps of a sample emulated
        /// by fetches read through it, and an opaque-black border the device replaces is checked
        /// against it.
        #[cfg_attr(feature = "serde", serde(default))]
        swizzle: metal::TextureSwizzleChannels,
        /// The texture's address-mode map ([`TextureFacts::address_modes`]): a slot a constexpr
        /// sampler samples compiles under it where it is not the identity
        /// (`VariantKey::address_maps`).
        #[cfg_attr(feature = "serde", serde(default))]
        address_modes: metal::sampler::AddressModeMap,
        /// The plane axes on which the base-level size Metal's texel-offset rule reads, a view's
        /// parent's, is not a power of two where the texture's own is
        /// ([`TextureFacts::offset_base_axes`]): a slot whose function places an offset by that
        /// rule specializes on them (`SpecKey::OffsetBaseAxes`).
        #[cfg_attr(feature = "serde", serde(default))]
        offset_base_axes: u8,
        /// The texture's planes are read in place where a slot's function reads them so
        /// ([`TextureFacts::ycbcr_in_place`]): such a slot takes its in-place variant
        /// (`VariantKey::ycbcr_in_place`).
        #[cfg_attr(feature = "serde", serde(default))]
        ycbcr_in_place: bool,
        /// The device's conversion of the texture's planes, where the consumer holds them as an
        /// image it samples ([`YCbCrInPlace::planes_sampler`](crate::bind::YCbCrInPlace)): a slot
        /// whose every sample the device's sampler takes samples that image
        /// (`VariantKey::ycbcr_planes_sampled`).
        #[cfg_attr(feature = "serde", serde(default))]
        ycbcr_planes_sampler: Option<crate::bind::YCbCrPlanesSampler>,
    },
    Buffer {
        pixel_format: metal::PixelFormat,
        /// The texture buffer's swizzle (`BoundTexture::Buffer`'s): a slot whose reads apply it
        /// specializes on it (`SpecKey::TextureSwizzle`).
        #[cfg_attr(feature = "serde", serde(default))]
        swizzle: metal::TextureSwizzleChannels,
    },
}

impl TextureShape {
    /// The bound texture's pixel format.
    pub(crate) fn pixel_format(&self) -> metal::PixelFormat {
        match *self {
            TextureShape::Texture { pixel_format, .. }
            | TextureShape::Buffer { pixel_format, .. } => pixel_format,
        }
    }

    /// The `SpecKey::OutOfBoundsAlpha` of a slot bound to this texture: 1 where an
    /// out-of-bounds read of its pixel format reads alpha 1. `None` for a placeholder pixel format,
    /// which has no sampled type, so no declared texture is compatible with it and the slot reads
    /// its neutral.
    pub(crate) fn out_of_bounds_alpha(&self) -> Option<u64> {
        crate::metal::format::out_of_bounds_alpha_one(self.pixel_format()).map(u64::from)
    }

    /// The bound texture's swizzle: a texture's or a texture buffer's.
    pub(crate) fn swizzle(&self) -> metal::TextureSwizzleChannels {
        match *self {
            TextureShape::Texture { swizzle, .. } | TextureShape::Buffer { swizzle, .. } => swizzle,
        }
    }

    /// The `SpecKey::OffsetBaseAxes` of a slot bound to this texture
    /// ([`TextureFacts::offset_base_axes`]); a texture buffer's 0.
    pub(crate) fn offset_base_axes(&self) -> u64 {
        match *self {
            TextureShape::Texture {
                offset_base_axes, ..
            } => u64::from(offset_base_axes),
            TextureShape::Buffer { .. } => 0,
        }
    }

    /// The `SpecKey::ConstantLanes` of a slot bound to this texture: where the texture is sparse,
    /// the lanes its format and swizzle make constants (`metal::sampler::constant_lanes`), which
    /// the function sets after a sampled read whose texels a block that holds no mapped tile may
    /// answer the device's value for; 0 elsewhere, where the device reads those lanes as Metal
    /// does, and for a format whose channels are not stated (a depth or stencil format).
    pub(crate) fn constant_lanes(&self) -> u64 {
        match *self {
            TextureShape::Texture {
                pixel_format,
                sparse: true,
                swizzle,
                ..
            } => crate::metal::format::channels(pixel_format).map_or(0, |present| {
                u64::from(crate::metal::sampler::constant_lanes(present, swizzle))
            }),
            TextureShape::Texture { .. } | TextureShape::Buffer { .. } => 0,
        }
    }

    /// Whether the bound texture is sparse (`MTLTexture.isSparse`); a texture buffer never is.
    pub(crate) fn sparse(&self) -> bool {
        matches!(self, TextureShape::Texture { sparse: true, .. })
    }

    /// This shape with a level count of 1: [`BindShape::for_plan`]'s projection of a texture
    /// whose levels the pipeline does not read, and what a heap member's shape is checked by
    /// (a heap array views level 0 alone).
    pub(crate) fn at_one_level(mut self) -> TextureShape {
        if let TextureShape::Texture {
            mipmap_level_count, ..
        } = &mut self
        {
            *mipmap_level_count = 1;
        }
        self
    }
}

/// The shape of a sampler bound to a slot: its descriptor's [`SamplerShape`], its state word, which
/// the slot's `SpecKey::SamplerState` specialization masks by the fields its function reads, and
/// the level its `SpecKey::SamplerMinLevel` specialization states.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundSamplerShape {
    pub shape: SamplerShape,
    pub word: metal::sampler::SamplerWord,
    /// `metal::sampler::query_min_level` of the descriptor: the level, in sixty-fourths, that
    /// Metal's `calculate_clamped_lod` answers outside a fragment function before the texture's
    /// last level bounds it.
    pub min_level: u32,
    /// `metal::sampler::anisotropy_degree` of the descriptor: `log2` of the degree Metal filters
    /// with, which the slot's `SpecKey::SamplerAnisotropy` specialization states.
    #[cfg_attr(feature = "serde", serde(default))]
    pub degree: u32,
}

/// The shape of a sampler: the modes that are runtime facts, and the descriptor values that decide
/// whether Vulkan can express it (checked at plan time by `metal::sampler::check_sampler`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerShape {
    pub reduction: metal::SamplerReductionMode,
    pub compare: bool,
    /// `max_anisotropy > 1` (`samplerAnisotropy`).
    pub anisotropic: bool,
    /// Some axis uses `MirrorClampToEdge` (`samplerMirrorClampToEdge`).
    pub mirror_clamp_to_edge: bool,
    /// The LOD clamps are numbers and `lod_min_clamp <= lod_max_clamp`.
    pub valid_lod_clamps: bool,
    /// The border the sampler replaces texels with is opaque black: whether it reads exactly
    /// through a texture's view depends on the view (`metal::sampler::check_border_view`).
    pub opaque_black_border: bool,
    /// Some filter (minification, magnification, or mip) is linear: whether the sampler needs
    /// the view format's filter features depends on its use and the view
    /// (`metal::sampler::vk_filtering`), which a plan checks where the sampler meets textures
    /// through separate descriptors.
    pub linear: bool,
    /// The min or the mag filter is `Cubic`: Metal samples through the sampler with its bicubic
    /// filter whatever the level, which a function emulates under its plan's
    /// cubic-sampler variant (`VariantKey::cubic_samplers`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub cubic: bool,
}
