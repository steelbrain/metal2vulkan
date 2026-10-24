//! Metal-shaped pipeline descriptors.
//!
//! The consumer's decoded guest descriptors map field for field onto these; raw `u32` ordinals
//! convert through the `metal::` `TryFrom<u32>` impls, which reject unknown values. Constructors
//! start from Metal's own defaults. Compilation is `Device::compile_*_pipeline`; features not
//! implemented yet (mesh, tile, tessellation) still have their descriptor fields so the refusal is
//! typed. A compute pipeline's stage-input descriptor feeds a kernel's `[[stage_in]]`. Linked
//! functions resolve a function's visible-function references and are what its visible-function
//! tables can hold.

use std::collections::BTreeMap;

use crate::library::compiled::CompiledFunction;
use crate::library::dynamic::DynamicLibrary;
use crate::library::function::{Function, FunctionConstantValues};
use crate::metal;

/// A function plus its constant values and the functions its pipeline links
/// (`MTLFunctionDescriptor` and the stage's `MTLLinkedFunctions`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionDescriptor {
    pub function: Function,
    pub constants: FunctionConstantValues,
    /// The functions the stage links: each visible-function reference of `function` (and of each
    /// linked function) calls the linked function of the name it references.
    pub linked: LinkedFunctions,
    /// The dynamic libraries the stage links: each symbol `function` (or a linked function, or a
    /// dynamic library) imports is the first of them that exports it.
    pub libraries: DynamicLibraries,
}

impl FunctionDescriptor {
    /// `function` with no constant values, no linked functions, and no dynamic libraries.
    pub fn new(function: Function) -> FunctionDescriptor {
        FunctionDescriptor {
            function,
            constants: FunctionConstantValues::defaults(),
            linked: LinkedFunctions::default(),
            libraries: DynamicLibraries::default(),
        }
    }

    /// This descriptor with `linked` as its linked functions.
    pub fn with_linked(mut self, linked: LinkedFunctions) -> FunctionDescriptor {
        self.linked = linked;
        self
    }

    /// This descriptor with `libraries` as its dynamic libraries.
    pub fn with_libraries(mut self, libraries: DynamicLibraries) -> FunctionDescriptor {
        self.libraries = libraries;
        self
    }

    /// This descriptor with a library in `libraries.loaded` for each install name the stage's
    /// search reaches that it lacks, as Metal loads them at pipeline creation: `load` gives the
    /// library at an install name, or `None` where none loads. The install names come in the order
    /// [`DynamicLibraries`] searches them (a loaded library's own after those already reached), and
    /// loading stops at the first that `load` gives none for: compiling then refuses any import
    /// that survives specialization (`NeedsInput(DynamicLibrary)`), as Metal fails such a creation
    /// and makes the others. A library `loaded` already holds is used as it is.
    pub fn with_loaded_dependencies(
        mut self,
        mut load: impl FnMut(&str) -> Option<DynamicLibrary>,
    ) -> FunctionDescriptor {
        loop {
            let unloaded = crate::pipeline::compile::dynamic_order(
                &self.function,
                &self.linked,
                &self.libraries,
            )
            .unloaded
            .map(str::to_owned);
            let Some(name) = unloaded else {
                return self;
            };
            match load(&name) {
                Some(library) => {
                    self.libraries.loaded.insert(name, library);
                }
                None => return self,
            }
        }
    }
}

/// The dynamic libraries a pipeline stage links (`MTLDynamicLibrary`): its preloaded libraries, and
/// the libraries the install names of its libraries load.
///
/// A library imports a symbol (a function or a variable) from dynamic libraries through a
/// declaration its AIR lists in `air.dyld_flat_table`, and names the libraries it was linked
/// against by install name ([`Library::dependencies`](crate::library::Library::dependencies)).
/// Compile resolves each import a specialized stage still uses as Metal resolves it at pipeline
/// creation (probes, an M3 Max and an M4): one search, in order, over `preloaded`, then the
/// libraries `loaded` holds for the install names the stage's library names, breadth first (its own
/// dependencies in link order, then theirs; then those of its linked functions' libraries, then
/// those of the preloaded libraries, an order not measured). The first library that exports the
/// symbol supplies it, and its definition is linked statically and inlined; a library's calls to
/// the functions it defines itself stay its own. An import no library exports is
/// `Error::NeedsInput(DynamicSymbol)`, and with an install name `loaded` lacks (a file Metal could
/// not load) every import is `Error::NeedsInput(DynamicLibrary)`, even one a preloaded library
/// exports, as Metal fails such a creation; each only when a use of an import survives
/// specialization. Every library the search reaches is part of the compile key.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DynamicLibraries {
    /// `preloadedLibraries` (a render stage's `vertexPreloadedLibraries` or
    /// `fragmentPreloadedLibraries`): searched first, in order.
    pub preloaded: Vec<DynamicLibrary>,
    /// By install name, the library loaded from the file it names, as Metal loads it: searched
    /// after `preloaded`, as the install names reach them. The file at an install name may hold
    /// a library of another install name; Metal uses it all the same.
    pub loaded: BTreeMap<String, DynamicLibrary>,
}

impl DynamicLibraries {
    /// No dynamic library.
    pub fn new() -> DynamicLibraries {
        DynamicLibraries::default()
    }

    /// Whether no library is given.
    pub fn is_empty(&self) -> bool {
        self.preloaded.is_empty() && self.loaded.is_empty()
    }

    /// These libraries with `library` added to `preloaded`.
    pub fn with_preloaded(mut self, library: DynamicLibrary) -> DynamicLibraries {
        self.preloaded.push(library);
        self
    }

    /// These libraries with `library` loaded for `install_name`.
    pub fn with_loaded(
        mut self,
        install_name: impl Into<String>,
        library: DynamicLibrary,
    ) -> DynamicLibraries {
        self.loaded.insert(install_name.into(), library);
        self
    }
}

/// `MTLLinkedFunctions`: the functions a pipeline stage links.
///
/// A function calls a `[[visible]]` function it does not define through a visible-function
/// reference (`!air.visible_function_references`), which names it. Compile resolves each
/// reference against the functions of `functions`, `private_functions`, and `binary_functions`
/// by [`FunctionInfo::name`](crate::library::FunctionInfo): the named function (a visible
/// function of its module, [`FunctionStage::Visible`](crate::library::FunctionStage) in a
/// bitcode library) is linked statically and inlined, as are the functions it references in
/// turn. A reference nothing names is `Error::NeedsInput(LinkedFunction)` if a call to it
/// survives specialization. The functions of `functions` and `binary_functions` are also what a
/// visible-function table can hold: each has a [`FunctionHandle`](crate::pipeline::FunctionHandle)
/// ([`LinkedFunctions::function_handle`]), and a call through a table compiles to a switch over
/// those of the call's type. `groups` names the functions a call site through a function pointer
/// may call; the decoder drops the call site's `!air.function_groups`, so no output depends on it.
/// Every field is part of the compile key.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkedFunctions {
    /// `functions`: visible functions the pipeline calls by reference or through a function
    /// table (each has a table handle).
    pub functions: Vec<LinkedFunction>,
    /// `privateFunctions`: visible functions the pipeline calls only by reference.
    pub private_functions: Vec<LinkedFunction>,
    /// `binaryFunctions`: visible functions compiled to a binary. The translator links their AIR
    /// as it links `functions`, and a table can hold them too.
    pub binary_functions: Vec<LinkedFunction>,
    /// `groups`: by call-site group name, the functions a call through a function pointer in
    /// that group may call.
    pub groups: BTreeMap<String, Vec<LinkedFunction>>,
}

impl LinkedFunctions {
    /// No linked functions.
    pub fn new() -> LinkedFunctions {
        LinkedFunctions::default()
    }

    /// Whether no function is linked.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
            && self.private_functions.is_empty()
            && self.binary_functions.is_empty()
            && self.groups.is_empty()
    }

    /// These linked functions with `f` added to `functions`.
    pub fn with_function(mut self, f: impl Into<LinkedFunction>) -> LinkedFunctions {
        self.functions.push(f.into());
        self
    }

    /// These linked functions with `f` added to `private_functions`.
    pub fn with_private_function(mut self, f: impl Into<LinkedFunction>) -> LinkedFunctions {
        self.private_functions.push(f.into());
        self
    }

    /// Every linked function with the list it is in, in field order (`groups` by name).
    pub(crate) fn listed(&self) -> impl Iterator<Item = (LinkedList<'_>, &LinkedFunction)> + '_ {
        self.functions
            .iter()
            .map(|f| (LinkedList::Functions, f))
            .chain(
                self.private_functions
                    .iter()
                    .map(|f| (LinkedList::PrivateFunctions, f)),
            )
            .chain(
                self.binary_functions
                    .iter()
                    .map(|f| (LinkedList::BinaryFunctions, f)),
            )
            .chain(self.groups.iter().flat_map(|(name, fs)| {
                fs.iter()
                    .map(move |f| (LinkedList::Group(name.as_str()), f))
            }))
    }
}

/// Which list of [`LinkedFunctions`] a linked function is in.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinkedList<'a> {
    Functions,
    PrivateFunctions,
    BinaryFunctions,
    /// The call-site group of this name.
    Group(&'a str),
}

impl LinkedList<'_> {
    /// Whether a visible-function reference resolves against this list: every list but a group,
    /// whose functions only a function-pointer call reaches.
    pub fn resolves_references(self) -> bool {
        !matches!(self, LinkedList::Group(_))
    }
}

/// One function of [`LinkedFunctions`]: a library function and the constant values it was
/// specialized with (`newFunctionWithDescriptor:`; none for `newFunctionWithName:`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedFunction {
    pub function: Function,
    pub constants: FunctionConstantValues,
    /// The function was made compiled to a binary (`MTLFunctionOptionCompileToBinary`), as
    /// Metal requires of a function in `binary_functions` and of a binary function added to a
    /// made pipeline state; the runtime checks it, refusing one that is not as Metal does. The
    /// translation links the function's AIR either way, so it is not part of any key.
    pub precompiled: bool,
}

impl LinkedFunction {
    /// `function` with no constant values, not compiled to a binary.
    pub fn new(function: Function) -> LinkedFunction {
        LinkedFunction {
            function,
            constants: FunctionConstantValues::defaults(),
            precompiled: false,
        }
    }

    /// This function specialized with `constants`.
    pub fn with_constants(mut self, constants: FunctionConstantValues) -> LinkedFunction {
        self.constants = constants;
        self
    }

    /// This function, compiled to a binary when `precompiled` (`precompiled`).
    pub fn with_precompiled(mut self, precompiled: bool) -> LinkedFunction {
        self.precompiled = precompiled;
        self
    }
}

impl From<Function> for LinkedFunction {
    fn from(function: Function) -> LinkedFunction {
        LinkedFunction::new(function)
    }
}

/// Where a pipeline stage comes from: a function to compile, or a function the caller already
/// compiled (link and relocate only; no decode, no lowering).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageSource {
    Function(FunctionDescriptor),
    Compiled(CompiledFunction),
}

/// `MTLRenderPipelineDescriptor`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderPipelineDescriptor {
    pub vertex: StageSource,
    /// The fragment function, `None` where none is set. Taken with rasterization disabled too:
    /// Metal links it either way, so its refusals see it (a texture with indirect command buffers,
    /// a depth output without a depth attachment, stage inputs no vertex output writes), and the
    /// runtime then drops it for the compile; one that writes no attached color and reads nothing
    /// kept is made.
    pub fragment: Option<StageSource>,
    pub vertex_descriptor: Option<VertexDescriptor>,
    /// Color attachments by index; `None` for `MTLPixelFormatInvalid`.
    pub color_attachments: [Option<ColorAttachmentDescriptor>; 8],
    pub depth_format: Option<metal::PixelFormat>,
    pub stencil_format: Option<metal::PixelFormat>,
    pub raster_sample_count: u32,
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
    pub rasterization_enabled: bool,
    pub input_primitive_topology: metal::PrimitiveTopologyClass,
    /// `maxVertexAmplificationCount` (default 1): above 1 the pipeline is a multiview pipeline; 0
    /// is an invalid descriptor.
    pub max_vertex_amplification_count: u32,
    /// The tessellation state a post-tessellation vertex function's pipeline draws patches with;
    /// `None` is Metal's defaults. Ignored for any other vertex function.
    pub tessellation: Option<TessellationDescriptor>,
    pub support_indirect_command_buffers: bool,
    /// `maxVertexCallStackDepth`: how deep calls through visible functions may nest in the
    /// vertex function, 1 by default as Metal's. A `Function` vertex stage compiles under it
    /// (`LoweringInputs::max_call_stack_depth`), as a compute pipeline's kernel does under its
    /// `max_call_stack_depth`.
    pub max_vertex_call_stack_depth: u32,
    /// `maxFragmentCallStackDepth`: the same for the fragment function.
    pub max_fragment_call_stack_depth: u32,
    /// `supportAddingVertexBinaryFunctions`: binary functions may be added to the vertex stage
    /// of a state made from this descriptor. The pipeline does not depend on it, so it is not
    /// part of the key.
    pub support_adding_vertex_binary_functions: bool,
    /// `supportAddingFragmentBinaryFunctions`: the same for the fragment stage.
    pub support_adding_fragment_binary_functions: bool,
    /// `vertexBuffers[i].mutability` by buffer index; an index not listed is
    /// [`Mutability::Default`](metal::Mutability). Metal takes it as a promise to its optimizer:
    /// writes to an immutable buffer land, and neither API nor shader validation stops them
    /// (measured on an M3 Max and an M4). Not part of the key.
    pub vertex_buffers: BTreeMap<u32, metal::Mutability>,
    /// `fragmentBuffers[i].mutability` by buffer index, as `vertex_buffers`.
    pub fragment_buffers: BTreeMap<u32, metal::Mutability>,
    /// `textureWriteRoundingMode` (Apple SPI): how the fragment function's `float` outputs to
    /// half-float color attachments round. A `Function` fragment stage compiles under it
    /// (`LoweringInputs::color_writes`).
    pub texture_write_rounding_mode: metal::TextureWriteRoundingMode,
    /// Metal's private fixed-function state of OpenGL on Metal ([`PrivateRenderState`]): Apple
    /// SPI of `MTLRenderPipelineDescriptor`, Metal's defaults until set.
    pub private: PrivateRenderState,
}

/// The private fixed-function properties of Metal's `MTLRenderPipelineDescriptor` that OpenGL on
/// Metal sets (Apple SPI: selectors in no header of the public SDK), each Metal's default until
/// set. What each does on Metal was measured on an Apple M4; a property whose effect the
/// translation has not built yet refuses a non-default value at the pipeline's creation,
/// `NotImplemented` with [`Feature::PrivateRenderState`](crate::error::Feature) naming it
/// ([`PrivateRenderField`](crate::error::PrivateRenderField)).
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct PrivateRenderState {
    /// `sampleMask` (default all ones): ANDed with the coverage of a pipeline with a fragment
    /// function after the function runs, bits past the pipeline's sample count ignored; a
    /// pipeline without one ignores it.
    pub sample_mask: u64,
    /// `sampleCoverage` (default 1): a fraction of each 2x2 quad's samples to keep, dithered
    /// over its window pixels and applied with `sample_mask` after the fragment function runs
    /// (`pipeline::private_state`'s docs); a value other than 1 is refused at a sample count past
    /// 4, where Metal was not measured.
    pub sample_coverage: f32,
    /// `sampleCoverageInvert` (default off): changes nothing Metal draws.
    pub sample_coverage_invert: bool,
    /// `alphaTestEnabled` (default off): the fragment passes where `alpha_test_function` holds
    /// between color 0's alpha and the render encoder's reference, each as an 8-bit unorm code
    /// (`pipeline::private_state`'s docs); one that fails writes nothing.
    pub alpha_test_enabled: bool,
    /// `alphaTestFunction` (default 0, `MTLCompareFunctionNever`), the raw value: read only
    /// with the alpha test on, where a value past `MTLCompareFunctionAlways` is refused.
    pub alpha_test_function: u64,
    /// `logicOperationEnabled` (default off): `logic_operation` applies to the codes of each
    /// normalized, sRGB and integer color attachment, which then blends nothing; a float
    /// attachment ignores it (`pipeline::private_state`'s docs). Copy and no-op draw on every
    /// device; the other operations need the device's `logicOp`, and are refused where an
    /// attachment is sRGB, a float attachment blends, or a format was not measured.
    pub logic_operation_enabled: bool,
    /// `logicOperation` (default 0), the raw value: read only with the logic operation on, by its
    /// low four bits, in Direct3D's order (0 clear, 1 set, 2 copy, 3 copy inverted, 4 no-op,
    /// 5 invert, 6 and, 7 nand, 8 or, 9 nor, 10 xor, 11 equivalent, 12 and reverse, 13 and
    /// inverted, 14 or reverse, 15 or inverted).
    pub logic_operation: u64,
    /// `clipDistanceEnableMask` (default `0xff`): changes nothing Metal draws; every clip
    /// distance the vertex function writes clips.
    pub clip_distance_enable_mask: u8,
    /// `pointSmoothEnabled` (default off; not built: on is refused).
    pub point_smooth_enabled: bool,
    /// `pointCoordLowerLeft` (default off): changes nothing Metal draws; `[[point_coord]]`
    /// keeps its upper-left origin.
    pub point_coord_lower_left: bool,
    /// `pointSizeOutputVS` (default off): changes nothing Metal draws; a point's size is the
    /// vertex function's `[[point_size]]`.
    pub point_size_output_vs: bool,
    /// `twoSideEnabled` (default off): changes nothing Metal draws.
    pub two_side_enabled: bool,
    /// `vertexDepthCompareClampMask` (default 0): changes nothing Metal draws.
    pub vertex_depth_compare_clamp_mask: u32,
    /// `fragmentDepthCompareClampMask` (default 0): bit `s` clamps the reference of a fragment
    /// function's depth comparison through `[[sampler(s)]]` to `[0, 1]`, a NaN taken as 0, and
    /// bit 31 one through a constexpr sampler; a comparison through an argument buffer's sampler
    /// is refused under a mask other than 0.
    pub fragment_depth_compare_clamp_mask: u32,
    /// `depthStencilWriteDisabled` (default off): changes nothing Metal draws.
    pub depth_stencil_write_disabled: bool,
    /// `needsCustomBorderColorSamplers` (default off): changes nothing Metal draws; a sampler's
    /// private custom border color samples transparent black with it or without.
    pub needs_custom_border_color_samplers: bool,
    /// `vertexAmplificationMode` (default 0): 0 and 1 draw alike, the amplification count and view
    /// mappings the encoder sets; Metal's descriptor asserts at 2 or more, which is refused as an
    /// unknown value.
    pub vertex_amplification_mode: u64,
    /// `explicitVisibilityGroupID` (default `u32::MAX`, none): changes nothing Metal draws in a
    /// pipeline whose `maxVertexAmplificationCount` is 1; any other group is refused where the
    /// pipeline amplifies (Metal there draws one copy's color and counts every copy's samples,
    /// a rule not built).
    pub explicit_visibility_group_id: u32,
}

impl PrivateRenderState {
    /// Metal's defaults.
    pub const fn new() -> PrivateRenderState {
        PrivateRenderState {
            sample_mask: u64::MAX,
            sample_coverage: 1.0,
            sample_coverage_invert: false,
            alpha_test_enabled: false,
            alpha_test_function: 0,
            logic_operation_enabled: false,
            logic_operation: 0,
            clip_distance_enable_mask: 0xff,
            point_smooth_enabled: false,
            point_coord_lower_left: false,
            point_size_output_vs: false,
            two_side_enabled: false,
            vertex_depth_compare_clamp_mask: 0,
            fragment_depth_compare_clamp_mask: 0,
            depth_stencil_write_disabled: false,
            needs_custom_border_color_samplers: false,
            vertex_amplification_mode: 0,
            explicit_visibility_group_id: u32::MAX,
        }
    }

    /// The fields as plain bits, the coverage by its `f32` bits: what equality compares.
    fn bits(&self) -> (u64, u32, [bool; 9], [u64; 3], u8, [u32; 3]) {
        (
            self.sample_mask,
            self.sample_coverage.to_bits(),
            [
                self.sample_coverage_invert,
                self.alpha_test_enabled,
                self.logic_operation_enabled,
                self.point_smooth_enabled,
                self.point_coord_lower_left,
                self.point_size_output_vs,
                self.two_side_enabled,
                self.depth_stencil_write_disabled,
                self.needs_custom_border_color_samplers,
            ],
            [
                self.alpha_test_function,
                self.logic_operation,
                self.vertex_amplification_mode,
            ],
            self.clip_distance_enable_mask,
            [
                self.vertex_depth_compare_clamp_mask,
                self.fragment_depth_compare_clamp_mask,
                self.explicit_visibility_group_id,
            ],
        )
    }
}

impl Default for PrivateRenderState {
    fn default() -> PrivateRenderState {
        PrivateRenderState::new()
    }
}

impl PartialEq for PrivateRenderState {
    fn eq(&self, other: &PrivateRenderState) -> bool {
        self.bits() == other.bits()
    }
}

impl Eq for PrivateRenderState {}

impl RenderPipelineDescriptor {
    /// A descriptor with Metal's defaults: one sample, rasterization on, no attachments.
    pub fn new(vertex: StageSource) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            vertex,
            fragment: None,
            vertex_descriptor: None,
            color_attachments: Default::default(),
            depth_format: None,
            stencil_format: None,
            raster_sample_count: 1,
            alpha_to_coverage: false,
            alpha_to_one: false,
            rasterization_enabled: true,
            input_primitive_topology: metal::PrimitiveTopologyClass::Unspecified,
            max_vertex_amplification_count: 1,
            tessellation: None,
            support_indirect_command_buffers: false,
            max_vertex_call_stack_depth: 1,
            max_fragment_call_stack_depth: 1,
            support_adding_vertex_binary_functions: false,
            support_adding_fragment_binary_functions: false,
            vertex_buffers: BTreeMap::new(),
            fragment_buffers: BTreeMap::new(),
            texture_write_rounding_mode: metal::TextureWriteRoundingMode::TowardZero,
            private: PrivateRenderState::new(),
        }
    }
}

/// `MTLComputePipelineDescriptor`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputePipelineDescriptor {
    pub kernel: StageSource,
    /// `stageInputDescriptor`: how the kernel's `[[stage_in]]` attributes are fetched. A kernel
    /// reading a stage-in attribute compiles under it (`LoweringInputs::stage_input`).
    pub stage_input: Option<StageInputDescriptor>,
    /// `None` for Metal's 0 (the device maximum).
    pub max_total_threads_per_threadgroup: Option<u32>,
    /// The guest's promise that interior regions' local sizes are multiples of 32.
    pub threadgroup_size_is_multiple_of_execution_width: bool,
    pub support_indirect_command_buffers: bool,
    /// `maxCallStackDepth`: how deep calls through visible functions may nest, 1 by default as
    /// Metal's. The kernel compiles under it (`LoweringInputs::max_call_stack_depth`), which sizes
    /// the call stack of recursion through visible functions.
    pub max_call_stack_depth: u32,
    /// `requiredThreadsPerThreadgroup`: `None` for Metal's zero size, no requirement. The runtime
    /// refuses a dispatch of another threadgroup size, as Metal's API validation stops one. The
    /// pipeline does not depend on it (the state's `maxTotalThreadsPerThreadgroup` stays the
    /// kernel's, as Metal's does), so it is not part of the key.
    pub required_threads_per_threadgroup: Option<[u32; 3]>,
    /// `supportAddingBinaryFunctions`: binary functions may be added to a state made from this
    /// descriptor. Not part of the key.
    pub support_adding_binary_functions: bool,
    /// `buffers[i].mutability` by buffer index; an index not listed is
    /// [`Mutability::Default`](metal::Mutability). Metal takes it as a promise to its optimizer:
    /// writes to an immutable buffer land, and neither API nor shader validation stops them
    /// (measured on an M3 Max and an M4). Not part of the key.
    pub buffers: BTreeMap<u32, metal::Mutability>,
    /// `needsCustomBorderColorSamplers` (Apple SPI, default off): changes nothing Metal runs, as
    /// the render descriptor's ([`PrivateRenderState::needs_custom_border_color_samplers`]).
    pub needs_custom_border_color_samplers: bool,
}

impl ComputePipelineDescriptor {
    /// A descriptor with Metal's defaults.
    pub fn new(kernel: StageSource) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            kernel,
            stage_input: None,
            max_total_threads_per_threadgroup: None,
            threadgroup_size_is_multiple_of_execution_width: false,
            support_indirect_command_buffers: false,
            max_call_stack_depth: 1,
            required_threads_per_threadgroup: None,
            support_adding_binary_functions: false,
            buffers: BTreeMap::new(),
            needs_custom_border_color_samplers: false,
        }
    }
}

/// `MTLMeshRenderPipelineDescriptor`, compiled by `Device::compile_mesh_pipeline`: an optional
/// object function, the mesh function, and the fragment function a rasterizing pipeline needs, with
/// a render pipeline's fixed-function state.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshPipelineDescriptor {
    pub object: Option<StageSource>,
    pub mesh: StageSource,
    pub fragment: Option<StageSource>,
    pub color_attachments: [Option<ColorAttachmentDescriptor>; 8],
    pub depth_format: Option<metal::PixelFormat>,
    pub stencil_format: Option<metal::PixelFormat>,
    pub raster_sample_count: u32,
    pub rasterization_enabled: bool,
    /// `alphaToCoverageEnabled`, as a render pipeline's.
    pub alpha_to_coverage: bool,
    /// `alphaToOneEnabled`, as a render pipeline's.
    pub alpha_to_one: bool,
    /// `supportIndirectCommandBuffers`: an indirect command buffer's render commands may set the
    /// state, as a render pipeline's (the runtime's; the compile ignores it).
    pub support_indirect_command_buffers: bool,
    /// `textureWriteRoundingMode` (Apple SPI), as a render pipeline's: the fragment function's
    /// `float` outputs to half-float color attachments round by it.
    pub texture_write_rounding_mode: metal::TextureWriteRoundingMode,
}

impl MeshPipelineDescriptor {
    /// A descriptor with Metal's defaults.
    pub fn new(mesh: StageSource) -> MeshPipelineDescriptor {
        MeshPipelineDescriptor {
            object: None,
            mesh,
            fragment: None,
            color_attachments: Default::default(),
            depth_format: None,
            stencil_format: None,
            raster_sample_count: 1,
            rasterization_enabled: true,
            alpha_to_coverage: false,
            alpha_to_one: false,
            support_indirect_command_buffers: false,
            texture_write_rounding_mode: metal::TextureWriteRoundingMode::TowardZero,
        }
    }
}

/// `MTLTileRenderPipelineDescriptor`, compiled by `Device::compile_tile_pipeline`: the tile
/// function and the render pass's color attachment formats by color index, which its implicit
/// imageblock holds.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileRenderPipelineDescriptor {
    pub tile: StageSource,
    pub color_attachment_formats: [Option<metal::PixelFormat>; 8],
    pub raster_sample_count: u32,
    /// `threadgroupSizeMatchesTileSize`. Plain Metal then answers the pass's tile as the tile
    /// function's threadgroup, whatever threads per tile a dispatch runs: its threads and
    /// simdgroups per threadgroup, its position in the grid, the threads per grid, and a thread
    /// index of `simdgroup · 32 + lane`. Metal's API validation stops a dispatch whose threads per
    /// tile are not the tile. The tile function compiles under it
    /// (`LoweringInputs::threadgroup_size_matches_tile_size`), and the dispatch still runs the
    /// threadgroup it gives (`Dispatch::Threadgroups`), which the binder checks against the shapes
    /// whose layout is measured.
    pub threadgroup_size_matches_tile_size: bool,
    /// `maxTotalThreadsPerThreadgroup`: `None` for Metal's 0 (the pipeline's own maximum). It
    /// limits the compiled pipeline's `metal_info.max_total_threads_per_threadgroup`, as a compute
    /// descriptor's does.
    pub max_total_threads_per_threadgroup: Option<u32>,
    /// `maxCallStackDepth`: how deep calls through visible functions may nest in the tile
    /// function, 1 by default as Metal's. A `Function` tile function compiles under it, as a
    /// compute pipeline's kernel does.
    pub max_call_stack_depth: u32,
    /// `requiredThreadsPerThreadgroup`: `None` for Metal's zero size, no requirement. The runtime
    /// refuses a tile dispatch of another threadgroup size, as Metal's API validation stops one.
    /// Not part of the key.
    pub required_threads_per_threadgroup: Option<[u32; 3]>,
    /// `supportAddingBinaryFunctions`: binary functions may be added to the tile stage of a
    /// state made from this descriptor. Not part of the key.
    pub support_adding_binary_functions: bool,
    /// `buffers[i].mutability` by buffer index, as a compute descriptor's
    /// ([`ComputePipelineDescriptor::buffers`]). Not part of the key.
    pub buffers: BTreeMap<u32, metal::Mutability>,
}

impl TileRenderPipelineDescriptor {
    /// A descriptor with Metal's defaults.
    pub fn new(tile: StageSource) -> TileRenderPipelineDescriptor {
        TileRenderPipelineDescriptor {
            tile,
            color_attachment_formats: [None; 8],
            raster_sample_count: 1,
            threadgroup_size_matches_tile_size: false,
            max_total_threads_per_threadgroup: None,
            max_call_stack_depth: 1,
            required_threads_per_threadgroup: None,
            support_adding_binary_functions: false,
            buffers: BTreeMap::new(),
        }
    }
}

/// `MTLRenderPipelineColorAttachmentDescriptor` (an attachment whose pixel format is not
/// `Invalid`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorAttachmentDescriptor {
    pub pixel_format: metal::PixelFormat,
    pub blending_enabled: bool,
    pub source_rgb_blend_factor: metal::BlendFactor,
    pub destination_rgb_blend_factor: metal::BlendFactor,
    pub rgb_blend_operation: metal::BlendOperation,
    pub source_alpha_blend_factor: metal::BlendFactor,
    pub destination_alpha_blend_factor: metal::BlendFactor,
    pub alpha_blend_operation: metal::BlendOperation,
    pub write_mask: metal::ColorWriteMask,
}

impl ColorAttachmentDescriptor {
    /// An attachment of `pixel_format` with Metal's defaults: blending off, `One`/`Zero`/`Add`,
    /// every channel written.
    pub fn new(pixel_format: metal::PixelFormat) -> ColorAttachmentDescriptor {
        ColorAttachmentDescriptor {
            pixel_format,
            blending_enabled: false,
            source_rgb_blend_factor: metal::BlendFactor::One,
            destination_rgb_blend_factor: metal::BlendFactor::Zero,
            rgb_blend_operation: metal::BlendOperation::Add,
            source_alpha_blend_factor: metal::BlendFactor::One,
            destination_alpha_blend_factor: metal::BlendFactor::Zero,
            alpha_blend_operation: metal::BlendOperation::Add,
            write_mask: metal::ColorWriteMask::ALL,
        }
    }

    /// The attachment with Metal 4's three specialization placeholders replaced by what a
    /// **classic** `MTLRenderPipelineDescriptor` blends with where one is set: each slot's own
    /// default.
    ///
    /// Measured on an Apple M3 Max, macOS 27.0.1, one run a shape, drawing one triangle over a
    /// known destination through a classic descriptor whose other five blend fields were `One`,
    /// `One` and `Add` (so a slot that fell back to its default read differently from one that
    /// kept the probe's value):
    ///
    /// - `MTLBlendFactorUnspecialized` (19) in a **source** slot blends as
    ///   [`One`](metal::BlendFactor::One), in a **destination** slot as
    ///   [`Zero`](metal::BlendFactor::Zero), and `MTLBlendOperationUnspecialized` (5) as
    ///   [`Add`](metal::BlendOperation::Add) — each the slot's own default;
    /// - the setter keeps the placeholder: the descriptor reads 19 and 5 back, so what changes is
    ///   what the blend does with it, not what the descriptor holds;
    /// - `newRenderPipelineStateWithDescriptor:error:` answers a state with no error, and the
    ///   draw stores what those defaults compute. The placeholders are accepted **specifically**:
    ///   the next value up ends plain Metal's process on an assertion (`Invalid blend factor` for
    ///   a factor of 20, `func is not a valid MTLBlendOperation` for an operation of 6), with no
    ///   API validation layer enabled.
    ///
    /// `MTLColorWriteMaskUnspecialized` is the third placeholder and is not here: it is one bit of
    /// a mask rather than a slot's whole value, and
    /// [`color_write_mask`](crate::metal::state::color_write_mask) holds its own measurement.
    ///
    /// The resolution happens where the blend is **interpreted**, never where the descriptor is
    /// built, because Metal's own getter answers the placeholder back and a binary archive's key
    /// is the raw word.
    #[must_use]
    pub fn specialized(&self) -> ColorAttachmentDescriptor {
        fn factor(held: metal::BlendFactor, default: metal::BlendFactor) -> metal::BlendFactor {
            match held {
                metal::BlendFactor::Unspecialized => default,
                other => other,
            }
        }
        fn operation(held: metal::BlendOperation) -> metal::BlendOperation {
            match held {
                metal::BlendOperation::Unspecialized => metal::BlendOperation::Add,
                other => other,
            }
        }
        ColorAttachmentDescriptor {
            source_rgb_blend_factor: factor(self.source_rgb_blend_factor, metal::BlendFactor::One),
            destination_rgb_blend_factor: factor(
                self.destination_rgb_blend_factor,
                metal::BlendFactor::Zero,
            ),
            rgb_blend_operation: operation(self.rgb_blend_operation),
            source_alpha_blend_factor: factor(
                self.source_alpha_blend_factor,
                metal::BlendFactor::One,
            ),
            destination_alpha_blend_factor: factor(
                self.destination_alpha_blend_factor,
                metal::BlendFactor::Zero,
            ),
            alpha_blend_operation: operation(self.alpha_blend_operation),
            ..*self
        }
    }
}

/// `MTLVertexDescriptor`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexDescriptor {
    /// By buffer index.
    pub layouts: BTreeMap<u32, VertexBufferLayoutDescriptor>,
    /// By attribute index.
    pub attributes: BTreeMap<u32, VertexAttributeDescriptor>,
}

impl VertexDescriptor {
    /// An empty vertex descriptor.
    pub fn new() -> VertexDescriptor {
        VertexDescriptor::default()
    }

    /// That the descriptor's layouts and set attributes agree as Metal requires, or the first
    /// buffer where they do not as `InvalidDescriptor(VertexLayout)`. Metal stops pipeline creation
    /// at an assertion from `newSerializedDescriptor`, with and without API validation, for:
    ///
    /// - a set attribute whose buffer has no layout ("Attribute at index N references a buffer at
    ///   index M that has no stride"), whether the vertex function reads it, declares it unread,
    ///   or does not declare it; checked first, in attribute order;
    /// - a layout of a nonzero or dynamic stride whose buffer no set attribute names ("None of
    ///   the attributes set bufferIndex to M, but MTLVertexDescriptor set buffer
    ///   layout[M].stride(S)"), whatever its step; a stride of 0 passes; then, in buffer order.
    ///
    /// An attribute of format `Invalid` is unset, whatever buffer it names.
    pub(crate) fn check_layouts(&self) -> Result<(), crate::error::Error> {
        let set = self
            .attributes
            .values()
            .filter(|a| a.format != metal::VertexFormat::Invalid)
            .map(|a| a.buffer_index);
        let strides = self.layouts.iter().map(|(&b, l)| (b, l.stride));
        disagreeing(set, strides).map_or(Ok(()), |layout| {
            Err(crate::error::Error::InvalidDescriptor(
                crate::error::DescriptorError::VertexLayout { layout },
            ))
        })
    }
}

/// The first buffer where a descriptor's set attributes (`attributes`, each one's buffer, in
/// attribute order) and its layouts (`layouts`, each one's buffer and stride, in buffer order;
/// `None` the dynamic stride) disagree as Metal stops at ([`VertexDescriptor::check_layouts`]):
/// an attribute's buffer without a layout, else a layout of a nonzero or dynamic stride that no
/// attribute names.
fn disagreeing(
    attributes: impl Iterator<Item = u32>,
    layouts: impl Iterator<Item = (u32, Option<u32>)>,
) -> Option<u32> {
    let attributes: Vec<u32> = attributes.collect();
    let strides: BTreeMap<u32, Option<u32>> = layouts.collect();
    attributes
        .iter()
        .copied()
        .find(|buffer| !strides.contains_key(buffer))
        .or_else(|| {
            strides
                .iter()
                .find(|&(buffer, &stride)| stride != Some(0) && !attributes.contains(buffer))
                .map(|(&buffer, _)| buffer)
        })
}

/// `MTLVertexBufferLayoutDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexBufferLayoutDescriptor {
    /// Bytes between elements; `MTLBufferLayoutStrideDynamic` is `None` (set per bind).
    pub stride: Option<u32>,
    pub step_function: metal::VertexStepFunction,
    pub step_rate: u32,
}

impl VertexBufferLayoutDescriptor {
    /// A per-vertex layout with this stride.
    pub fn new(stride: Option<u32>) -> VertexBufferLayoutDescriptor {
        VertexBufferLayoutDescriptor {
            stride,
            step_function: metal::VertexStepFunction::PerVertex,
            step_rate: 1,
        }
    }
}

/// `MTLVertexAttributeDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexAttributeDescriptor {
    pub format: metal::VertexFormat,
    pub offset: u32,
    pub buffer_index: u32,
}

impl VertexAttributeDescriptor {
    /// An attribute of `format` at `offset` in buffer `buffer_index`.
    pub fn new(format: metal::VertexFormat, offset: u32, buffer_index: u32) -> Self {
        VertexAttributeDescriptor {
            format,
            offset,
            buffer_index,
        }
    }
}

/// `MTLStageInputOutputDescriptor`: how a kernel fetches its `[[stage_in]]` attributes.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageInputDescriptor {
    /// By buffer index.
    pub layouts: BTreeMap<u32, StageInputLayoutDescriptor>,
    /// By attribute index.
    pub attributes: BTreeMap<u32, StageInputAttributeDescriptor>,
    pub index_type: metal::IndexType,
    pub index_buffer_index: u32,
}

impl StageInputDescriptor {
    /// No layouts or attributes, 16-bit indices from buffer 0 (Metal's defaults).
    pub fn new() -> StageInputDescriptor {
        StageInputDescriptor {
            layouts: BTreeMap::new(),
            attributes: BTreeMap::new(),
            index_type: metal::IndexType::UInt16,
            index_buffer_index: 0,
        }
    }

    /// That the descriptor's layouts and set attributes agree as Metal requires, or the first
    /// buffer where they do not as `InvalidDescriptor(StageInputLayout)`: the rules of
    /// [`VertexDescriptor::check_layouts`], which Metal's assertions state alike for a kernel's
    /// attributes (read, declared unread or not declared) and its layouts. Then, in buffer order,
    /// each layout but Metal's unset one whose step Metal's pipeline creation stops at an assertion
    /// for, with and without API validation: a vertex step function
    /// (`InvalidDescriptor(StageInputStepFunction)`, "invalid stepFunction value"), and a
    /// constant step at a step rate other than 0 (`InvalidDescriptor(StageInputStepRate)`,
    /// "stepRate(1) must be zero if stepFunction is MTLStepFunctionConstant").
    pub(crate) fn check_layouts(&self) -> Result<(), crate::error::Error> {
        use crate::error::{DescriptorError, Error};
        let set = self
            .attributes
            .values()
            .filter(|a| a.format != metal::AttributeFormat::Invalid)
            .map(|a| a.buffer_index);
        let strides = self.layouts.iter().map(|(&b, l)| (b, l.stride));
        if let Some(layout) = disagreeing(set, strides) {
            return Err(Error::InvalidDescriptor(
                DescriptorError::StageInputLayout { layout },
            ));
        }
        for (&layout, l) in self.layouts.iter().filter(|(_, l)| !l.unset()) {
            if vertex_step(l.step_function) {
                return Err(Error::InvalidDescriptor(
                    DescriptorError::StageInputStepFunction {
                        layout,
                        step_function: l.step_function,
                    },
                ));
            }
            if l.step_function == metal::StepFunction::Constant && l.step_rate != 0 {
                return Err(Error::InvalidDescriptor(
                    DescriptorError::StageInputStepRate {
                        layout,
                        step_rate: l.step_rate,
                    },
                ));
            }
        }
        Ok(())
    }
}

impl StageInputDescriptor {
    /// The descriptor as a fetch reads it: only the layouts an attribute names; a constant step's
    /// stride as `Some(0)` and step rate as 0 (it reads one element, at its offset, whatever the
    /// stride, static or per bind); the index type and index buffer at Metal's defaults unless a
    /// kept layout steps through the index buffer. Descriptors that fetch every attribute alike are
    /// equal.
    pub fn canonical(&self) -> StageInputDescriptor {
        use crate::metal::StepFunction as S;
        let mut out = StageInputDescriptor::new();
        out.attributes = self.attributes.clone();
        for a in self.attributes.values() {
            if let Some(l) = self.layouts.get(&a.buffer_index) {
                let mut l = *l;
                if l.step_function == S::Constant {
                    l.stride = Some(0);
                    l.step_rate = 0;
                }
                out.layouts.insert(a.buffer_index, l);
            }
        }
        let indexed = out.layouts.values().any(|l| {
            matches!(
                l.step_function,
                S::ThreadPositionInGridXIndexed | S::ThreadPositionInGridYIndexed
            )
        });
        if indexed {
            out.index_type = self.index_type;
            out.index_buffer_index = self.index_buffer_index;
        }
        out
    }
}

impl Default for StageInputDescriptor {
    fn default() -> StageInputDescriptor {
        StageInputDescriptor::new()
    }
}

/// `MTLBufferLayoutDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageInputLayoutDescriptor {
    /// Bytes between elements; `MTLBufferLayoutStrideDynamic` is `None`: the stride the encoder
    /// binds the buffer with (`setBuffer:offset:attributeStride:atIndex:` and its forms), 0 for
    /// a buffer bound without one, as Metal reads it.
    pub stride: Option<u32>,
    pub step_function: metal::StepFunction,
    pub step_rate: u32,
}

impl StageInputLayoutDescriptor {
    /// A layout with this stride, stepping per vertex at a rate of 1, as a fresh
    /// `MTLStageInputOutputDescriptor`'s layouts do (stride 0, `MTLStepFunctionPerVertex`, step
    /// rate 1, measured on an M3 Max and an M4): `new(Some(0))` is Metal's unset layout. A kernel
    /// fetches through no vertex step function, so set `step_function` (a thread position in the
    /// grid, or `Constant` at a step rate of 0) for a layout an attribute reads.
    pub fn new(stride: Option<u32>) -> StageInputLayoutDescriptor {
        StageInputLayoutDescriptor {
            stride,
            step_function: metal::StepFunction::PerVertex,
            step_rate: 1,
        }
    }

    /// Whether the layout is Metal's unset one (stride 0, per vertex, step rate 1), which
    /// Metal's pipeline creation passes over: every other layout's step function is checked
    /// ([`StageInputDescriptor::check_layouts`]).
    fn unset(&self) -> bool {
        *self == StageInputLayoutDescriptor::new(Some(0))
    }
}

/// Whether `step` is one of a vertex stage's step functions (per vertex, instance, patch or
/// control point), none of which a compute pipeline's stage input takes.
pub(crate) fn vertex_step(step: metal::StepFunction) -> bool {
    use metal::StepFunction as S;
    matches!(
        step,
        S::PerVertex | S::PerInstance | S::PerPatch | S::PerPatchControlPoint
    )
}

/// `MTLAttributeDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageInputAttributeDescriptor {
    pub format: metal::AttributeFormat,
    pub offset: u32,
    pub buffer_index: u32,
}

impl StageInputAttributeDescriptor {
    /// An attribute of `format` at `offset` in buffer `buffer_index`.
    pub fn new(format: metal::AttributeFormat, offset: u32, buffer_index: u32) -> Self {
        StageInputAttributeDescriptor {
            format,
            offset,
            buffer_index,
        }
    }
}

/// The tessellation fields of `MTLRenderPipelineDescriptor`: the synthesized control stage's factor
/// rules and modes. A control-point index type other than `None` adds the control-point index
/// buffer to the control stage's slots and carries the width in its state; both widths are exact.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationDescriptor {
    pub max_tessellation_factor: u32,
    pub partition_mode: metal::TessellationPartitionMode,
    pub factor_format: metal::TessellationFactorFormat,
    pub factor_step_function: metal::TessellationFactorStepFunction,
    pub output_winding_order: metal::Winding,
    pub control_point_index_type: metal::TessellationControlPointIndexType,
    pub factor_scale_enabled: bool,
}

impl Default for TessellationDescriptor {
    /// Metal's defaults: factor 16, pow2 partitioning, half factors, constant step, clockwise.
    fn default() -> TessellationDescriptor {
        TessellationDescriptor {
            max_tessellation_factor: 16,
            partition_mode: metal::TessellationPartitionMode::Pow2,
            factor_format: metal::TessellationFactorFormat::Half,
            factor_step_function: metal::TessellationFactorStepFunction::Constant,
            output_winding_order: metal::Winding::Clockwise,
            control_point_index_type: metal::TessellationControlPointIndexType::None,
            factor_scale_enabled: false,
        }
    }
}
