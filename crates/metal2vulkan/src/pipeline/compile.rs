//! Per-function compilation under `catch_unwind`.
//!
//! [`compile_stage`] runs the whole per-function pipeline: decode → container cross-check →
//! specialize → analyze → select → (size check) → structurize → emit → P9 → verify → serialize.
//! It keeps nothing but the
//! [`RelocatableModule`](crate::spirv::serialize::RelocatableModule) and plain metadata: decoded
//! AIR, the plan, and the typed module are dropped before it returns. Every entry point here runs
//! under `catch_unwind`, so a panic anywhere in the core is `Error::Internal(Panic)`, never an
//! unwind into the caller. The crate keeps no memo; the caller caches the result under its
//! [`FunctionCompileKey`].
//!
//! The public surface (`Device::compile_function`, `function_compile_key`,
//! `compile_compute_pipeline`, `compile_render_pipeline`, and the pipeline keys) is [`api`]; the
//! key derivation is [`key`]; holding a pipeline to the device's compiler word bound is `fit`.

pub mod api;
pub(crate) mod color_group;
mod entry;
mod fit;
pub mod key;
mod numbered;
mod partition;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::air::abi::{AbiInt, AirAbi, OutputRole, ParamRole, Stage as AbiStage};
use crate::air::container::{self, LibraryFunction};
use crate::air::ir::Value;
use crate::air::{self, ConstValue, LinkedCode, LinkedModule, LinkedName, SpecRequest};
use crate::bind::argument_buffer::{self, EntryLayout};
use crate::bind::{ArgumentBufferLayout, ArgumentLayoutError};
use crate::budget::{Meter, SPIRV_WORDS};
use crate::cache::Hash256;
use crate::cache::VariantKey;
use crate::device::DeviceProfile;
use crate::error::{
    DescriptorError, DiagName, Error, Feature, InternalError, InternalKind, Unsupported,
};
use crate::library::function::FunctionSource;
use crate::library::{
    CompiledFunction, DynamicLibrary, Function, FunctionArtifact, FunctionCompileKey,
    FunctionConstantValues, KernelFacts,
};
use crate::lower::dispatch::LocalSizeForm;
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::artifact::{PrecisionNote, StrategyCensus, StrategyUse};
use crate::pipeline::compile::key::{DynamicKey, LinkedKey};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, DynamicLibraries, FunctionDescriptor, LinkedFunctions,
    MeshPipelineDescriptor, RenderPipelineDescriptor, TessellationDescriptor, VertexDescriptor,
};
use crate::pipeline::interface::{SlotInterface, StageInterface};
use crate::pipeline::sym::VaryingKey;
use crate::reflection;
use crate::spirv::ir::TypedModule;

pub use entry::{EntryDescriptor, EntryPlan, StandaloneModule};

/// Which entry of a bitcode module to compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntryRef<'a> {
    /// The entry point with this name.
    Name(&'a str),
    /// The `i`-th entry point of the module's AIR metadata (`!air.kernel`, `!air.vertex`,
    /// `!air.fragment`, in that order).
    Index(u32),
}

/// Where a compile's lowering inputs come from.
#[derive(Clone, Copy, Debug)]
pub enum InputsFrom<'a> {
    /// The caller's (`Device::compile_function`, validation, tests).
    Given(&'a LowerInputs),
    /// The vertex stage of a render pipeline under a variant key: the vertex-fetch projection of
    /// this vertex descriptor over the attributes the specialized entry reads with the key's
    /// pulled vertex buffers (`link::vertex::vertex_fetch_projection_for`), the key's other
    /// vertex-stage inputs (`VariantKey::lowering_inputs`), the pipeline's
    /// `maxVertexCallStackDepth` (the `u32`), whether the pipeline's fragment function reads
    /// `[[primitive_id]]` where the device carries it through a varying
    /// ([`reads_primitive_id_varying`]), and its tessellation state (`None` for Metal's
    /// defaults), which a post-tessellation vertex function bakes in on a device without
    /// `tessellationShader` (`LoweringInputs::tessellation`). A plain vertex function writes the
    /// primitive ID's varying where the device cannot declare a fragment `PrimitiveId`, and a
    /// post-tessellation one the device emulates on any device
    /// (`LoweringInputs::primitive_id_varying`).
    Vertex(
        Option<&'a VertexDescriptor>,
        &'a VariantKey,
        u32,
        bool,
        Option<&'a TessellationDescriptor>,
    ),
    /// The fragment stage of this render pipeline under a variant key: the key's fragment-stage
    /// inputs, the attachments among the pipeline's color attachments that the specialized entry
    /// writes and whose pixel format is a signed integer
    /// (`LoweringInputs::signed_color_attachments`), whether the pipeline rasterizes more than one
    /// sample (`LoweringInputs::multisampled`), on a device that orders framebuffer fetch only
    /// through the pixel interlock the state of the attachments the entry reads
    /// (`LoweringInputs::framebuffer_fetch`), and, for each index the pipeline's vertex function
    /// writes (the [`RawIndices`]), whether the entry reads it
    /// (`LoweringInputs::viewport_index_varying`;
    /// `LoweringInputs::render_target_array_index_varying`). The pipeline's
    /// `maxFragmentCallStackDepth` is the descriptor's. In a mesh render pipeline the mesh function
    /// stands in the vertex function's place, and the keys it writes per primitive (the set) are
    /// those the entry reads per primitive (`LoweringInputs::primitive_varyings`); a render
    /// pipeline's set is empty. Last, the user varyings the vertex or mesh function writes: the
    /// entry's inputs the pipeline's linking drops of other keys are absent
    /// (`LoweringInputs::absent_varyings`); `None` drops none. Then the user varyings the vertex
    /// function writes the same at every vertex of a primitive: the entry reads those it
    /// interpolates flat (`LoweringInputs::uniform_varyings`); a mesh render pipeline's set is
    /// empty.
    Fragment(
        FragmentPass<'a>,
        &'a VariantKey,
        RawIndices,
        &'a BTreeSet<VaryingKey>,
        Option<&'a BTreeSet<VaryingKey>>,
        &'a BTreeSet<VaryingKey>,
    ),
}

/// No varying written per primitive: a render pipeline's fragment stage's
/// ([`InputsFrom::Fragment`]). Also no varying written the same at every vertex of a primitive:
/// a mesh render pipeline's.
pub(crate) static NO_PRIMITIVE_VARYINGS: BTreeSet<VaryingKey> = BTreeSet::new();

/// The pipeline state a fragment stage compiles under ([`InputsFrom::Fragment`]): its render
/// pipeline's, or its mesh render pipeline's, which has no vertex amplification and no call stack
/// depth of its own (Metal's default, 1).
#[derive(Clone, Copy, Debug)]
pub struct FragmentPass<'a> {
    pub(crate) color_attachments: &'a [Option<ColorAttachmentDescriptor>; 8],
    /// The pipeline has a depth or a stencil attachment.
    pub(crate) depth_or_stencil: bool,
    /// The pipeline's depth attachment is `Depth16Unorm` (`LoweringInputs::depth16_codes`).
    pub(crate) depth16: bool,
    pub(crate) raster_sample_count: u32,
    pub(crate) rasterization_enabled: bool,
    pub(crate) alpha_to_coverage: bool,
    pub(crate) max_vertex_amplification_count: u32,
    pub(crate) max_call_stack_depth: u32,
    /// The pipeline's `textureWriteRoundingMode`.
    pub(crate) texture_write_rounding_mode: crate::metal::TextureWriteRoundingMode,
    /// The pipeline's private masks the function applies itself (`LoweringInputs::sample_mask`);
    /// `None` from a descriptor, set by the compile layer where they differ between a quad's
    /// pixels or the function compiled without them could tell Vulkan's mask from Metal's.
    pub(crate) sample_mask: Option<[u32; 4]>,
    /// The pipeline's private alpha test (`LoweringInputs::alpha_test`).
    pub(crate) alpha_test: Option<crate::metal::CompareFunction>,
    /// The pipeline's private `fragmentDepthCompareClampMask`
    /// (`LoweringInputs::depth_compare_clamp_mask`).
    pub(crate) depth_compare_clamp_mask: u32,
    /// The group of the pipeline's color attachments the function writes
    /// (`LoweringInputs::color_group`), whose variant's descriptor `color_attachments` is
    /// (`color_group::descriptor`); `None` from a descriptor, set by the compile layer.
    pub(crate) color_group: Option<&'a crate::pipeline::lower_inputs::ColorGroup>,
}

impl<'a> From<&'a RenderPipelineDescriptor> for FragmentPass<'a> {
    fn from(d: &'a RenderPipelineDescriptor) -> FragmentPass<'a> {
        FragmentPass {
            color_attachments: &d.color_attachments,
            depth_or_stencil: d.depth_format.is_some() || d.stencil_format.is_some(),
            depth16: d.depth_format == Some(crate::metal::PixelFormat::Depth16Unorm),
            raster_sample_count: d.raster_sample_count,
            rasterization_enabled: d.rasterization_enabled,
            alpha_to_coverage: d.alpha_to_coverage,
            max_vertex_amplification_count: d.max_vertex_amplification_count,
            max_call_stack_depth: d.max_fragment_call_stack_depth,
            texture_write_rounding_mode: d.texture_write_rounding_mode,
            sample_mask: None,
            // Refused at the pipeline's creation where the test is not built.
            alpha_test: crate::pipeline::private_state::alpha_test(&d.private)
                .ok()
                .flatten(),
            depth_compare_clamp_mask: d.private.fragment_depth_compare_clamp_mask,
            color_group: None,
        }
    }
}

impl<'a> From<&'a MeshPipelineDescriptor> for FragmentPass<'a> {
    fn from(d: &'a MeshPipelineDescriptor) -> FragmentPass<'a> {
        FragmentPass {
            color_attachments: &d.color_attachments,
            depth_or_stencil: d.depth_format.is_some() || d.stencil_format.is_some(),
            depth16: d.depth_format == Some(crate::metal::PixelFormat::Depth16Unorm),
            raster_sample_count: d.raster_sample_count,
            rasterization_enabled: d.rasterization_enabled,
            alpha_to_coverage: d.alpha_to_coverage,
            max_vertex_amplification_count: 1,
            max_call_stack_depth: 1,
            texture_write_rounding_mode: d.texture_write_rounding_mode,
            sample_mask: None,
            alpha_test: None,
            depth_compare_clamp_mask: 0,
            color_group: None,
        }
    }
}

/// The indices a render pipeline's vertex function writes, each of which it also writes as
/// written to a raw varying that a fragment function of the pipeline reads in its place:
/// `[[viewport_array_index]]` and `[[render_target_array_index]]`; and the primitive ID's varying,
/// which a fragment function reads as its `[[primitive_id]]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawIndices {
    /// `[[viewport_array_index]]` (`IoRole::ViewportIndex`).
    pub viewport: bool,
    /// `[[render_target_array_index]]` (`IoRole::RenderTargetArrayIndex`).
    pub layer: bool,
    /// The primitive ID (`IoRole::PrimitiveId`).
    pub primitive_id: bool,
}

impl<'a> InputsFrom<'a> {
    /// The `maxCallStackDepth` the stage compiles under: the given inputs', or the render
    /// pipeline's for its stage. 0 and 1 are Metal's default.
    fn call_stack_depth(&self) -> u32 {
        match self {
            InputsFrom::Given(i) => i.max_call_stack_depth.unwrap_or(1),
            InputsFrom::Vertex(_, _, depth, _, _) => (*depth).max(1),
            InputsFrom::Fragment(pass, ..) => pass.max_call_stack_depth.max(1),
        }
    }

    /// The group of its pipeline's color attachments a fragment stage compiles for
    /// (`LoweringInputs::color_group`): the given inputs', or the render pipeline variant's.
    fn color_group(&self) -> Option<&'a crate::pipeline::lower_inputs::ColorGroup> {
        match self {
            InputsFrom::Given(i) => i.color_group.as_ref(),
            InputsFrom::Vertex(..) => None,
            InputsFrom::Fragment(pass, ..) => pass.color_group,
        }
    }
}

/// The specialized entry `spec` as a compile under `inputs` lowers it on `p`: renumbered for its
/// color group (`color_group::entry`) where it compiles for one.
fn grouped_entry(
    spec: air::Entry,
    inputs: InputsFrom<'_>,
    p: &DeviceProfile,
) -> Result<air::Entry, Error> {
    match inputs.color_group() {
        Some(g) => color_group::entry(spec, g, p),
        None => Ok(spec),
    }
}

/// The lowering input of a `maxCallStackDepth` `depth`: `None` at Metal's default, 1 (or 0).
fn call_stack_input(depth: u32) -> Option<u32> {
    (depth > 1).then_some(depth)
}

/// What a decode learns of a stage before it is compiled ([`stage_inputs`]).
#[derive(Clone, Debug)]
pub struct StageInputs {
    /// The lowering inputs a compile of the stage uses.
    pub inputs: LowerInputs,
    /// The indices the specialized entry writes (a vertex function's outputs), which a fragment
    /// function of its pipeline reads from their raw varyings
    /// (`LoweringInputs::viewport_index_varying`;
    /// `LoweringInputs::render_target_array_index_varying`).
    pub writes: RawIndices,
    /// The user varyings the specialized entry writes (a vertex function's outputs), against
    /// which a fragment function of its pipeline finds its absent inputs
    /// (`LoweringInputs::absent_varyings`).
    pub varyings: BTreeSet<VaryingKey>,
    /// The user varyings the specialized entry writes the same at every vertex of a primitive (a
    /// vertex function's, `StageIo::uniform_outputs`), which a fragment function of its pipeline
    /// reads flat where it interpolates them (`LoweringInputs::uniform_varyings`).
    pub uniform: BTreeSet<VaryingKey>,
}

/// One linked function as a compile takes it: its key (the list it is in, its name, its AIR hash,
/// its constant values) and its bitcode module.
#[derive(Clone, Copy, Debug)]
pub struct LinkedSource<'a> {
    pub(crate) key: LinkedKey<'a>,
    pub(crate) bitcode: &'a [u8],
}

impl<'a> LinkedSource<'a> {
    /// The function `name` of `functions` (`MTLLinkedFunctions.functions`), defined in the module
    /// `bitcode` (hashed here), specialized with `constants`.
    pub fn new(
        name: &'a str,
        bitcode: &'a [u8],
        constants: &'a FunctionConstantValues,
    ) -> LinkedSource<'a> {
        LinkedSource {
            key: LinkedKey {
                list: crate::pipeline::descriptor::LinkedList::Functions,
                name,
                air_hash: container::sha256(bitcode),
                constants,
            },
            bitcode,
        }
    }
}

/// The key of each function of `linked`, in field order.
pub(crate) fn linked_keys(linked: &LinkedFunctions) -> Vec<LinkedKey<'_>> {
    linked
        .listed()
        .map(|(list, f)| {
            let info = f.function.info();
            LinkedKey {
                list,
                name: info.name,
                air_hash: info.air_hash,
                constants: &f.constants,
            }
        })
        .collect()
}

/// The sources of `linked`, in field order.
fn linked_sources(linked: &LinkedFunctions) -> Result<Vec<LinkedSource<'_>>, Error> {
    linked
        .listed()
        .zip(linked_keys(linked))
        .map(|((_, f), key)| {
            Ok(LinkedSource {
                key,
                bitcode: f.function.bitcode()?,
            })
        })
        .collect()
}

/// The dynamic libraries a stage's imports are searched in, in order, and the first install name of
/// the search that no library was loaded for.
#[derive(Clone, Debug, Default)]
pub(crate) struct DynamicOrder<'a> {
    /// The preloaded libraries, then the loaded ones the install names reach.
    pub libraries: Vec<&'a DynamicLibrary>,
    /// The first install name `DynamicLibraries::loaded` lacks: every import is refused with it.
    pub unloaded: Option<&'a str>,
}

impl DynamicOrder<'_> {
    /// What the compile key reads of the order: each library's install name and content hash,
    /// and the install name not loaded.
    pub(crate) fn key(&self) -> DynamicKey<'_> {
        DynamicKey {
            libraries: self
                .libraries
                .iter()
                .map(|l| (l.install_name(), l.library().content_hash()))
                .collect(),
            unloaded: self.unloaded,
        }
    }
}

/// The search order of the imports of function `f`, linking `linked` and `libraries`: the preloaded
/// libraries in order, then breadth first by install name the libraries `loaded` holds, from the
/// install names `f`'s library depends on in link order, then those of each linked function's
/// library, then those of each preloaded library (each name once, each loaded library's own names
/// after the names already queued), up to the first install name with no loaded library. Container
/// metadata only: nothing is decoded.
pub(crate) fn dynamic_order<'a>(
    f: &'a Function,
    linked: &'a LinkedFunctions,
    libraries: &'a DynamicLibraries,
) -> DynamicOrder<'a> {
    let mut order: Vec<&DynamicLibrary> = libraries.preloaded.iter().collect();
    let mut queue: VecDeque<&str> = f.library().dependencies().collect();
    for (_, l) in linked.listed() {
        queue.extend(l.function.library().dependencies());
    }
    for p in &libraries.preloaded {
        queue.extend(p.library().dependencies());
    }
    let mut seen = BTreeSet::new();
    while let Some(name) = queue.pop_front() {
        if !seen.insert(name) {
            continue;
        }
        match libraries.loaded.get(name) {
            Some(library) => {
                order.push(library);
                queue.extend(library.library().dependencies());
            }
            None => {
                return DynamicOrder {
                    libraries: order,
                    unloaded: Some(name),
                }
            }
        }
    }
    DynamicOrder {
        libraries: order,
        unloaded: None,
    }
}

/// The dynamic libraries of `order`, each decoded, when the stage's module `module` or a module
/// of `linked` imports a symbol from a dynamic library; else none, as nothing is searched. A
/// library whose module does not decode is recorded rather than returned: canonicalization refuses
/// it only when a search reaches it. Budget stops, cancellation, and internal errors return at
/// once.
fn dynamic_links(
    module: &air::ir::Module,
    linked: &[LinkedModule],
    order: &DynamicOrder<'_>,
    m: &mut Meter,
) -> Result<air::DynamicLinks, Error> {
    let imports = air::canon::imports_dynamically(module)
        || linked.iter().any(|l| {
            l.code
                .as_ref()
                .is_ok_and(|c| air::canon::imports_dynamically(&c.module))
        });
    if !imports {
        return Ok(air::DynamicLinks::default());
    }
    let mut libraries = Vec::with_capacity(order.libraries.len());
    for library in &order.libraries {
        m.tick(1)?;
        let code = match air::decode(library.module(), m) {
            Ok(air::Decoded { module, abi, .. }) => Ok(LinkedCode { module, abi }),
            Err(
                e @ (air::error::Error::Budget(_)
                | air::error::Error::Cancelled
                | air::error::Error::Internal(_)),
            ) => return Err(e.into()),
            Err(e) => Err(e),
        };
        m.charge(std::mem::size_of::<air::DynamicModule>() + library.install_name().len())?;
        libraries.push(air::DynamicModule {
            install_name: library.install_name().as_bytes().into(),
            code,
        });
    }
    let unloaded = order.unloaded.map(|name| {
        air::error::Error::NeedsInput(air::error::MissingInput::DynamicLibrary {
            install_name: DiagName::new(name),
        })
    });
    Ok(air::DynamicLinks {
        libraries,
        unloaded,
    })
}

/// Specialize `decoded` for `req`, linking the request's functions and dynamic libraries.
fn specialize_request(
    decoded: air::Decoded,
    req: &SpecRequest,
    r: &Request<'_>,
    m: &mut Meter,
) -> Result<air::Entry, Error> {
    let linked = linked_modules(&decoded.abi, r.linked, m)?;
    let dynamic = dynamic_links(&decoded.module, &linked, &r.dynamic, m)?;
    Ok(air::specialize_dynamic(
        decoded,
        req,
        (&linked, &dynamic),
        m,
    )?)
}

/// Run `f` under `catch_unwind`: a panic is `Internal(Panic)`.
pub fn guarded<T>(f: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".to_owned());
            Err(Error::Internal(InternalError::new(
                InternalKind::Panic,
                detail,
            )))
        }
    }
}

/// Compile function `f` of its library with `consts`, linking `linked`, for profile `p`, as the
/// opaque [`CompiledFunction`], from inputs given or derived (`inputs`). No memo inside; the result
/// is keyed by `FunctionCompileKey` for the caller's cache. Its artifact carries the provenance a
/// pipeline checks and the kernel facts its Metal info needs.
pub fn compile_stage(
    f: &Function,
    consts: &FunctionConstantValues,
    links: (&LinkedFunctions, &DynamicLibraries),
    p: &DeviceProfile,
    inputs: InputsFrom<'_>,
    m: &mut Meter,
) -> Result<CompiledFunction, Error> {
    guarded(|| {
        with_request(f, consts, links, inputs, |r| {
            compile_one(r, p, m, SPIRV_WORDS).map(CompiledFunction::new)
        })
    })
}

/// [`compile_stage`] of stage `fd`, reading its specialized entry from `entries` (the one a check
/// or key of the same creation made, else one made there now): the same compile, which decodes
/// and specializes nothing a creation already did.
pub(crate) fn compile_stage_from(
    fd: &FunctionDescriptor,
    p: &DeviceProfile,
    inputs: InputsFrom<'_>,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<CompiledFunction, Error> {
    guarded(|| {
        with_request_kept(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            inputs,
            Some((entries, fd)),
            |r| compile_one(r, p, m, SPIRV_WORDS).map(CompiledFunction::new),
        )
    })
}

/// The lowering inputs a compile of stage `fd` under `inputs` uses, and the indices its entry
/// writes ([`RawIndices`]): decode and specialize only (the entry `entries` holds, or one made
/// there now), for a key before compiling (a vertex stage's projection depends on the attributes
/// it reads, a fragment stage's inputs on which indices the vertex stage writes).
pub fn stage_inputs(
    fd: &FunctionDescriptor,
    p: &DeviceProfile,
    inputs: InputsFrom<'_>,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<StageInputs, Error> {
    guarded(|| {
        let specialized = entries.specialized(fd, inputs.call_stack_depth(), m)?;
        // A color group's variant reads its own renumbering of the entry the creation shares.
        let grouped = match inputs.color_group() {
            Some(_) => Some(grouped_entry(specialized.spec.clone(), inputs, p)?),
            None => None,
        };
        let spec = grouped.as_ref().unwrap_or(&specialized.spec);
        let inputs = lower_inputs(spec, inputs, p, m)?;
        let uniform = crate::lower::stage_io::uniform_varyings(spec, &inputs, m)?;
        let writes = raw_indices_written(spec, &inputs);
        Ok(StageInputs {
            inputs,
            writes,
            varyings: varyings_written(spec),
            uniform,
        })
    })
}

/// The layout of the argument buffer function `f` takes at buffer `index`, with the function
/// constants `consts` (`newArgumentEncoderWithBufferIndex:`): decode only, reading the entry's
/// `air.struct_type_info` as it stands; and where a function constant decides it (a parameter at
/// the index that one gates, or an index or count one states), specialize with `consts` and no
/// linked functions first, as Metal's function has none.
pub(crate) fn argument_buffer_layout(
    f: &Function,
    consts: &FunctionConstantValues,
    index: u32,
    m: &mut Meter,
) -> Result<ArgumentBufferLayout, ArgumentLayoutError> {
    let (linked, libraries) = (LinkedFunctions::default(), DynamicLibraries::default());
    let inputs = LowerInputs::new();
    let layout = guarded(|| {
        with_request(
            f,
            consts,
            (&linked, &libraries),
            InputsFrom::Given(&inputs),
            |r| {
                let mut decoded = air::decode(r.bitcode, m)?;
                if let Some(c) = r.container {
                    decoded.abi.check_container(&decoded.module, c, m)?;
                }
                let req = spec_request(&decoded, r.entry, r.consts)?;
                let decoded_layout = {
                    let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
                    argument_buffer::entry_layout(&ep.params, &decoded.abi.structs, index, m)
                };
                match decoded_layout {
                    EntryLayout::Done(layout) => Ok(layout),
                    EntryLayout::Specialize => {
                        let spec = air::specialize_linked(decoded, &req, &[], m)?;
                        match argument_buffer::entry_layout(
                            &spec.ep.params,
                            &spec.structs,
                            index,
                            m,
                        ) {
                            EntryLayout::Done(layout) => Ok(layout),
                            EntryLayout::Specialize => Err(Error::Internal(InternalError::new(
                                InternalKind::Invariant,
                                "a specialized entry left its argument buffer to specialization",
                            ))),
                        }
                    }
                }
            },
        )
    });
    match layout {
        Ok(layout) => layout,
        Err(error) => Err(ArgumentLayoutError::Compile(error)),
    }
}

/// What a pipeline's reflection states for its stage `fd`, its calls nesting `depth` deep through
/// visible functions: the entry's bindings and the attributes it reads, from the entry its
/// constants and linked functions specialize where they decide a binding
/// (`reflection::needs_resolution`) or which arguments it reads
/// (`reflection::specialization_decides_reads`), else from the decoded entry.
pub(crate) fn stage_reflection(
    fd: &FunctionDescriptor,
    depth: u32,
    m: &mut Meter,
) -> Result<reflection::StageReflection, Error> {
    let mut inputs = LowerInputs::new();
    inputs.max_call_stack_depth = call_stack_input(depth);
    guarded(|| {
        with_request(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            InputsFrom::Given(&inputs),
            |r| {
                let mut decoded = air::decode(r.bitcode, m)?;
                if let Some(c) = r.container {
                    decoded.abi.check_container(&decoded.module, c, m)?;
                }
                let mut req = spec_request(&decoded, r.entry, r.consts)?;
                req.call_stack_depth = r.inputs.call_stack_depth();
                let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
                if !reflection::needs_resolution(ep, &decoded.abi.structs)
                    && !reflection::specialization_decides_reads(&decoded.abi)
                {
                    return Ok(reflection::StageReflection {
                        bindings: reflection::bindings(
                            ep,
                            &decoded.abi.structs,
                            reflection::Channel::Pipeline,
                            &BTreeSet::new(),
                        )?,
                        attributes: reflection::read_attributes(ep)?,
                    });
                }
                let before = reflection::UnspecializedReads::of(&decoded.module, &decoded.abi, ep);
                let spec = specialize_request(decoded, &req, r, m)?;
                let read = reflection::specialized_reads(&spec, &before)?;
                Ok(reflection::StageReflection {
                    bindings: reflection::bindings(
                        &read,
                        &spec.structs,
                        reflection::Channel::Pipeline,
                        &spec.function_tables,
                    )?,
                    attributes: reflection::read_attributes(&read)?,
                })
            },
        )
    })
}

/// What stage `fd`, its calls nesting `depth` deep through visible functions, may store through
/// each slot its arguments bind: of the entry [`stage_reflection`] reads, its parameters' roles,
/// declared accesses and slots alone (`reflection::entry_slot_writes`). It derives no type
/// description, so an argument whose reflection Metal's answer has not been measured for does not
/// refuse it: only the decode and the specialization can.
pub(crate) fn stage_slot_writes(
    fd: &FunctionDescriptor,
    depth: u32,
    m: &mut Meter,
) -> Result<Vec<reflection::SlotWrite>, Error> {
    let mut inputs = LowerInputs::new();
    inputs.max_call_stack_depth = call_stack_input(depth);
    guarded(|| {
        with_request(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            InputsFrom::Given(&inputs),
            |r| {
                let mut decoded = air::decode(r.bitcode, m)?;
                if let Some(c) = r.container {
                    decoded.abi.check_container(&decoded.module, c, m)?;
                }
                let mut req = spec_request(&decoded, r.entry, r.consts)?;
                req.call_stack_depth = r.inputs.call_stack_depth();
                let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
                if !reflection::needs_resolution(ep, &decoded.abi.structs)
                    && !reflection::specialization_decides_reads(&decoded.abi)
                {
                    return Ok(reflection::entry_slot_writes(ep));
                }
                let before = reflection::UnspecializedReads::of(&decoded.module, &decoded.abi, ep);
                let spec = specialize_request(decoded, &req, r, m)?;
                match reflection::specialized_reads(&spec, &before) {
                    Ok(read) => Ok(reflection::entry_slot_writes(&read)),
                    // Refused over control-point inputs a function constant gates, which bind no
                    // argument-table slot (`kept_reflection` says what that costs).
                    Err(Error::Unsupported(u)) if u.feature == Feature::Reflection => {
                        Ok(reflection::entry_slot_writes(&spec.ep))
                    }
                    Err(error) => Err(error),
                }
            },
        )
    })
}

/// Whether stage `fd`, its calls nesting `depth` deep through visible functions, uses a direct
/// texture or sampler argument (`[[texture(n)]]`, `[[sampler(n)]]`, a texture array among them)
/// that a pipeline's reflection lists as used: of the entry [`stage_reflection`] reads, its
/// arguments of those roles that are not `air.arg_unused`. It derives no binding's type, so an
/// argument whose reflection Metal's answer has not been measured for decides nothing here: only
/// the roles and the use do.
pub(crate) fn stage_uses_direct_arguments(
    fd: &FunctionDescriptor,
    depth: u32,
    m: &mut Meter,
) -> Result<bool, Error> {
    let direct = |ep: &air::abi::EntryPoint| {
        ep.params.iter().any(|p| {
            !p.unused && matches!(p.role, ParamRole::Texture(_) | ParamRole::Sampler { .. })
        })
    };
    let mut inputs = LowerInputs::new();
    inputs.max_call_stack_depth = call_stack_input(depth);
    guarded(|| {
        with_request(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            InputsFrom::Given(&inputs),
            |r| {
                let mut decoded = air::decode(r.bitcode, m)?;
                if let Some(c) = r.container {
                    decoded.abi.check_container(&decoded.module, c, m)?;
                }
                let mut req = spec_request(&decoded, r.entry, r.consts)?;
                req.call_stack_depth = r.inputs.call_stack_depth();
                let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
                if !reflection::needs_resolution(ep, &decoded.abi.structs)
                    && !reflection::specialization_decides_reads(&decoded.abi)
                {
                    return Ok(direct(ep));
                }
                let before = reflection::UnspecializedReads::of(&decoded.module, &decoded.abi, ep);
                let spec = specialize_request(decoded, &req, r, m)?;
                let read = reflection::specialized_reads(&spec, &before)?;
                Ok(direct(&read))
            },
        )
    })
}

/// The specialized entries of the stages of one pipeline state's creation: each stage's function,
/// with its calls nesting a given depth, is decoded and specialized at most once while the object
/// lives, and every check, key and compile of the creation that is given the object reads that one
/// entry. Specializing a large function takes far longer than decoding it (16 ms of an 18 ms decode
/// and specialization, measured), and a render state's creation asks for each stage's entry up to
/// three times before its compile: its descriptor's linked-use checks
/// (`RenderPipelineDescriptor::unwritten_fragment_inputs` and
/// `fragment_writes_depth_unattached`) and its key (`Device::render_variant_key`).
///
/// It holds each entry until it is dropped: a creation makes one, gives it to each of its steps,
/// and drops it with the creation, so no entry outlives the state's making. Entries are told apart
/// by their function descriptor (the function, its constants, its linked functions and dynamic
/// libraries) and depth, so one object given to steps of different descriptors stays exact.
#[derive(Default)]
pub struct StageEntries {
    made: std::sync::Mutex<Vec<StageEntry>>,
}

/// One specialized entry of [`StageEntries`], with what tells it apart.
struct StageEntry {
    function: FunctionDescriptor,
    depth: u32,
    specialized: Arc<Specialized>,
}

impl std::fmt::Debug for StageEntries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let made = self.made.lock().map_or(0, |made| made.len());
        f.debug_struct("StageEntries").field("made", &made).finish()
    }
}

impl StageEntries {
    /// No entry made yet.
    pub fn new() -> StageEntries {
        StageEntries::default()
    }

    /// The entry of stage `fd`, its calls nesting `depth` deep through visible functions (0 is
    /// taken as 1, Metal's default), specialized as a compile specializes it before lowering
    /// (its constants bound, its linked functions resolved, its calls inlined, SCCP run), with
    /// what a compile takes beside it (The AIR hash, the entry name, what its code read before, so
    /// a compile of the stage reads it too, [`specialize_one`]): the one made before, else made now
    /// under `m`. A refusal is not kept: the next ask makes it again.
    fn specialized(
        &self,
        fd: &FunctionDescriptor,
        depth: u32,
        m: &mut Meter,
    ) -> Result<Arc<Specialized>, Error> {
        let depth = depth.max(1);
        let lock = || self.made.lock().unwrap_or_else(|e| e.into_inner());
        let found = lock()
            .iter()
            .find(|e| e.depth == depth && e.function == *fd)
            .map(|e| Arc::clone(&e.specialized));
        if let Some(specialized) = found {
            return Ok(specialized);
        }
        let mut inputs = LowerInputs::new();
        inputs.max_call_stack_depth = call_stack_input(depth);
        let specialized = Arc::new(with_request(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            InputsFrom::Given(&inputs),
            |r| specialize_one(r, m),
        )?);
        lock().push(StageEntry {
            function: fd.clone(),
            depth,
            specialized: Arc::clone(&specialized),
        });
        Ok(specialized)
    }
}

/// The entry of stage `fd`, its calls nesting `depth` deep through visible functions, specialized
/// as a compile specializes it before lowering ([`StageEntries`] holds it), given to `f`: what a
/// pipeline's linked-use questions read (`reflection::linked`).
pub(crate) fn stage_entry<T>(
    fd: &FunctionDescriptor,
    depth: u32,
    entries: &StageEntries,
    m: &mut Meter,
    f: impl FnOnce(&air::Entry, &mut Meter) -> Result<T, Error>,
) -> Result<T, Error> {
    guarded(|| {
        let specialized = entries.specialized(fd, depth, m)?;
        f(&specialized.spec, m)
    })
}

/// Whether the entry of stage `fd` returns a value: its function's return type is not `void`.
/// It decodes the AIR, and specializes nothing.
pub(crate) fn stage_returns_value(fd: &FunctionDescriptor, m: &mut Meter) -> Result<bool, Error> {
    let inputs = LowerInputs::new();
    guarded(|| {
        with_request(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            InputsFrom::Given(&inputs),
            |r| {
                let mut decoded = air::decode(r.bitcode, m)?;
                if let Some(c) = r.container {
                    decoded.abi.check_container(&decoded.module, c, m)?;
                }
                let req = spec_request(&decoded, r.entry, r.consts)?;
                let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
                let types = decoded.module.types();
                match decoded.module.func(ep.func).and_then(|f| types.get(f.ty)) {
                    Some(air::ir::Type::Function { ret, .. }) => {
                        Ok(!matches!(types.get(*ret), Some(air::ir::Type::Void)))
                    }
                    _ => Err(Error::Internal(InternalError::new(
                        InternalKind::Invariant,
                        "an entry whose type is not a function",
                    ))),
                }
            },
        )
    })
}

/// What `MTLFunction` states about function `f` made with the constant values `consts`
/// (`newFunctionWithName:constantValues:error:`): the entry's properties from its AIR, after
/// specialization where a constant decides one of its inputs, so an attribute a constant gates
/// off is left out, as Metal leaves it out, and a stage input the specialized code never reads is
/// inactive.
pub(crate) fn specialized_properties(
    f: &Function,
    consts: &FunctionConstantValues,
    m: &mut Meter,
) -> Result<reflection::FunctionProperties, Error> {
    let (linked, libraries) = (LinkedFunctions::default(), DynamicLibraries::default());
    let inputs = LowerInputs::new();
    guarded(|| {
        with_request(
            f,
            consts,
            (&linked, &libraries),
            InputsFrom::Given(&inputs),
            |r| {
                let mut decoded = air::decode(r.bitcode, m)?;
                if let Some(c) = r.container {
                    decoded.abi.check_container(&decoded.module, c, m)?;
                }
                let req = spec_request(&decoded, r.entry, r.consts)?;
                let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
                if !reflection::needs_resolution(ep, &decoded.abi.structs)
                    && !reflection::specialization_decides_reads(&decoded.abi)
                {
                    return reflection::entry_properties(ep);
                }
                let before = reflection::UnspecializedReads::of(&decoded.module, &decoded.abi, ep);
                let spec = air::specialize_linked(decoded, &req, &[], m)?;
                reflection::entry_properties(&reflection::specialized_reads(&spec, &before)?)
            },
        )
    })
}

/// The user varyings the specialized entry `spec` writes (a vertex function's outputs).
fn varyings_written(spec: &air::Entry) -> BTreeSet<VaryingKey> {
    spec.ep
        .outputs
        .iter()
        .filter_map(|o| match &o.role {
            OutputRole::VertexOutput(key) => Some(VaryingKey::user(&key.0)),
            _ => None,
        })
        .collect()
}

/// The keys of the specialized fragment entry `spec`'s inputs it interpolates (`center`,
/// `centroid` or `sample`, with or without perspective) whose type is `float` or `half` or a
/// vector of either: those a uniform vertex output lets it read flat
/// (`LoweringInputs::uniform_varyings`). An integer or `bool` input is flat already, and a
/// `bfloat` one's interpolation by Metal is not measured.
fn interpolated_float_inputs(spec: &air::Entry) -> impl Iterator<Item = VaryingKey> + '_ {
    use crate::air::abi::Interpolation as AirInterp;
    use crate::air::ir::Type;
    let types = spec.module.types();
    let params = spec
        .module
        .func(spec.func)
        .and_then(|f| match types.get(f.ty) {
            Some(Type::Function { params, .. }) => Some(params.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let float = move |ty| match types.get(ty) {
        Some(Type::Vector { elem, .. }) => {
            matches!(types.get(*elem), Some(Type::Float | Type::Half))
        }
        Some(Type::Float | Type::Half) => true,
        _ => false,
    };
    spec.ep.params.iter().filter_map(move |b| match &b.role {
        ParamRole::FragmentInput(v)
            if matches!(v.interpolation, AirInterp::Smooth { .. })
                && params.get(b.param as usize).is_some_and(|&t| float(t)) =>
        {
            Some(VaryingKey::user(&v.key.0))
        }
        _ => None,
    })
}

/// The indices the specialized entry `spec` writes under `inputs`: a vertex function compiled
/// under `LoweringInputs::primitive_id_varying` writes the primitive ID's varying.
fn raw_indices_written(spec: &air::Entry, inputs: &LowerInputs) -> RawIndices {
    let writes = |f: fn(&OutputRole) -> bool| spec.ep.outputs.iter().any(|o| f(&o.role));
    RawIndices {
        viewport: writes(|r| matches!(r, OutputRole::ViewportArrayIndex)),
        layer: writes(|r| matches!(r, OutputRole::RenderTargetArrayIndex)),
        primitive_id: inputs.primitive_id_varying
            && matches!(spec.ep.stage, AbiStage::Vertex { .. }),
    }
}

/// The indices the specialized entry `spec` reads. The varying keys the specialized fragment entry
/// `spec` declares inputs of under `inputs`: its user varyings, and the raw indices it reads from
/// their varyings (`LoweringInputs::viewport_index_varying`,
/// `LoweringInputs::render_target_array_index_varying`).
fn varyings_read<'a>(
    spec: &'a air::Entry,
    inputs: &LowerInputs,
) -> impl Iterator<Item = VaryingKey> + 'a {
    let raw = [
        (inputs.viewport_index_varying, VaryingKey::viewport_index()),
        (
            inputs.render_target_array_index_varying,
            VaryingKey::render_target_array_index(),
        ),
        (inputs.primitive_id_varying, VaryingKey::primitive_id()),
    ];
    spec.ep
        .params
        .iter()
        .filter_map(|b| match &b.role {
            ParamRole::FragmentInput(v) => Some(VaryingKey::user(&v.key.0)),
            _ => None,
        })
        .chain(raw.into_iter().filter_map(|(read, k)| read.then_some(k)))
}

fn raw_indices_read(spec: &air::Entry) -> RawIndices {
    use crate::air::abi::Builtin;
    let reads = |b: Builtin| {
        spec.ep
            .params
            .iter()
            .any(|p| p.role == ParamRole::Builtin(b))
    };
    RawIndices {
        viewport: reads(Builtin::ViewportArrayIndex),
        layer: reads(Builtin::RenderTargetArrayIndex),
        primitive_id: reads(Builtin::PrimitiveId),
    }
}

/// Whether a fragment function compiled for `p` reads `[[primitive_id]]` from a varying
/// (`IoRole::PrimitiveId`): `p` enables none of the features that let a fragment stage declare
/// `PrimitiveId` (`geometryShader`, `tessellationShader`).
pub(crate) fn primitive_id_varying_on(p: &DeviceProfile) -> bool {
    let enabled = &p.fields().enabled.features;
    !crate::spirv::requirements::builtin_enablers(
        crate::spv::BuiltIn::PrimitiveId,
        crate::spirv::ir::IoDir::Input,
    )
    .iter()
    .any(|(_, f)| enabled.contains(f))
}

/// Whether a render pipeline's fragment stage `fragment`, its calls nesting `depth` deep, reads
/// `[[primitive_id]]`, which its vertex function then writes to its varying where the device
/// carries it that way (`LoweringInputs::primitive_id_varying`): a `Function` stage whose
/// specialized entry (the one `entries` holds, else one made there now under `m`) declares the
/// parameter, or a `Compiled` stage whose interface reads the varying or Vulkan's `PrimitiveId`
/// (link refuses the second behind a vertex function that writes the varying). Called only on a
/// device without a fragment `PrimitiveId` ([`primitive_id_varying_on`]) or without
/// `tessellationShader` (`lower::tessellation::emulates`), so no other decodes anything.
pub(crate) fn reads_primitive_id_varying(
    fragment: Option<&crate::pipeline::descriptor::StageSource>,
    depth: u32,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<bool, Error> {
    use crate::pipeline::descriptor::StageSource;
    Ok(match fragment {
        Some(StageSource::Function(fd)) => entries
            .specialized(fd, depth, m)?
            .spec
            .ep
            .params
            .iter()
            .any(|p| p.role == ParamRole::Builtin(crate::air::abi::Builtin::PrimitiveId)),
        Some(StageSource::Compiled(c)) => {
            c.artifact()
                .interface
                .io
                .inputs
                .iter()
                .any(|v| v.role == crate::pipeline::stage_io::IoRole::PrimitiveId)
                || c.artifact()
                    .interface
                    .io
                    .builtins
                    .contains(&crate::pipeline::stage_io::StageBuiltin::PrimitiveId)
        }
        None => false,
    })
}

/// Whether render pipelines on `p` may carry `[[primitive_id]]` through a varying, so that a
/// pipeline's fragment entry is read for it ([`reads_primitive_id_varying`]): `p` cannot declare a
/// fragment `PrimitiveId` ([`primitive_id_varying_on`]), or it emulates post-tessellation vertex
/// functions (`lower::tessellation::emulates`), whose patch's position the fragment reads.
pub(crate) fn primitive_id_carried_on(p: &DeviceProfile) -> bool {
    primitive_id_varying_on(p) || crate::lower::tessellation::emulates(p)
}

/// Run `g` on the compile request of function `f`. A visible function is not a stage: it is
/// `EntryNotFound`, as for a module that does not declare the entry.
fn with_request<T>(
    f: &Function,
    consts: &FunctionConstantValues,
    links: (&LinkedFunctions, &DynamicLibraries),
    inputs: InputsFrom<'_>,
    g: impl FnOnce(&Request<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    with_request_kept(f, consts, links, inputs, None, g)
}

/// [`with_request`], the request reading its specialized entry from `kept` where given
/// (`Request::kept`).
fn with_request_kept<'k, T>(
    f: &Function,
    consts: &FunctionConstantValues,
    (linked, libraries): (&LinkedFunctions, &DynamicLibraries),
    inputs: InputsFrom<'_>,
    kept: Option<(&'k StageEntries, &'k FunctionDescriptor)>,
    g: impl FnOnce(&Request<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    let source = f.source()?;
    let dynamic = dynamic_order(f, linked, libraries);
    let linked = linked_sources(linked)?;
    let r = match &source {
        FunctionSource::Metallib(lf) => Request {
            bitcode: lf.bitcode,
            entry: EntryRef::Name(lf.name),
            container: Some(lf.as_ref()),
            consts,
            linked: &linked,
            dynamic,
            inputs,
            kept,
        },
        FunctionSource::Module { bitcode, index, .. } => Request {
            bitcode,
            entry: EntryRef::Index(*index),
            container: None,
            consts,
            linked: &linked,
            dynamic,
            inputs,
            kept,
        },
        FunctionSource::Visible => {
            let entries = f
                .library()
                .functions()
                .filter(|i| {
                    !matches!(
                        i.stage,
                        crate::library::FunctionStage::Visible
                            | crate::library::FunctionStage::Exported
                    )
                })
                .count();
            return Err(Error::InvalidDescriptor(DescriptorError::EntryNotFound {
                requested: DiagName::new(f.info().name),
                available: u32::try_from(entries).unwrap_or(u32::MAX),
            }));
        }
    };
    g(&r)
}

/// Compile entry `entry` of one wrapped or raw bitcode module (`bitcode`), refusing an estimate
/// above `word_cap` SPIR-V words before emission and a module above it after serialization
/// (`budget::SPIRV_WORDS` in production).
#[cfg(test)]
pub(crate) fn compile_bitcode(
    bitcode: &[u8],
    entry: EntryRef<'_>,
    consts: &FunctionConstantValues,
    p: &DeviceProfile,
    inputs: &LowerInputs,
    m: &mut Meter,
    word_cap: u64,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode_linked(bitcode, entry, (consts, &[]), p, inputs, m, word_cap)
}

/// [`compile_bitcode`] with the function-constant values `consts`, resolving the module's
/// visible-function references against `linked`.
#[cfg(test)]
pub(crate) fn compile_bitcode_linked(
    bitcode: &[u8],
    entry: EntryRef<'_>,
    (consts, linked): (&FunctionConstantValues, &[LinkedSource<'_>]),
    p: &DeviceProfile,
    inputs: &LowerInputs,
    m: &mut Meter,
    word_cap: u64,
) -> Result<Arc<FunctionArtifact>, Error> {
    guarded(|| {
        let r = Request::bitcode(bitcode, entry, consts, linked, inputs);
        compile_one(&r, p, m, word_cap)
    })
}

/// `compile_bitcode` (no function constants, no linked functions), also returning the plan and
/// the finished typed module the artifact's words were serialized from: the module after P9 and
/// the verifier, whose serialization the artifact holds. For the CPU
/// interpreter, which then runs the module of the very compile whose words were linked and
/// validated rather than a second compile of the same source. The kept module and plan are not
/// charged to `m`; never on a product path. The crate's fuzz executor (`tests::fuzz_interp`) runs
/// it.
#[cfg(test)]
pub(crate) fn compile_bitcode_kept(
    bitcode: &[u8],
    entry: EntryRef<'_>,
    p: &DeviceProfile,
    inputs: &LowerInputs,
    m: &mut Meter,
    word_cap: u64,
) -> Result<(CompiledFunction, Plan, TypedModule), Error> {
    guarded(|| {
        let consts = FunctionConstantValues::defaults();
        let r = Request::bitcode(bitcode, entry, &consts, &[], inputs);
        compile_kept(&r, p, m, word_cap, TypedModule::clone)
            .map(|(artifact, plan, module)| (CompiledFunction::new(artifact), plan, module))
    })
}

/// The stage `fd` compiled for `p` as [`compile_stage`] compiles it (its linked functions and
/// dynamic libraries included), returning the finished typed module its words were serialized
/// from, for the CPU interpreter.
#[cfg(test)]
pub(crate) fn compile_stage_kept(
    fd: &FunctionDescriptor,
    p: &DeviceProfile,
    m: &mut Meter,
) -> Result<TypedModule, Error> {
    let inputs = LowerInputs::new();
    guarded(|| {
        with_request(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            InputsFrom::Given(&inputs),
            |r| compile_kept(r, p, m, SPIRV_WORDS, TypedModule::clone).map(|(_, _, t)| t),
        )
    })
}

/// The [`Plan`] of entry `entry` of `bitcode`: everything [`compile_bitcode`] does before the
/// size check (decode, specialize, analyze, select), under `catch_unwind`. For tests and
/// measurement of the plan's own claims, such as `SizeEstimate` being an upper bound.
#[cfg(test)]
pub(crate) fn plan_bitcode(
    bitcode: &[u8],
    entry: EntryRef<'_>,
    consts: &FunctionConstantValues,
    p: &DeviceProfile,
    inputs: &LowerInputs,
    m: &mut Meter,
) -> Result<Plan, Error> {
    guarded(|| {
        let r = Request::bitcode(bitcode, entry, consts, &[], inputs);
        plan_one(&r, p, m).map(|planned| planned.plan)
    })
}

/// One per-function compile request.
struct Request<'a> {
    /// A wrapped or raw bitcode module.
    bitcode: &'a [u8],
    entry: EntryRef<'a>,
    /// A metallib function's container tags, cross-checked against the decoded AIR metadata.
    container: Option<&'a LibraryFunction<'a>>,
    consts: &'a FunctionConstantValues,
    /// The functions the stage links.
    linked: &'a [LinkedSource<'a>],
    /// The dynamic libraries the stage's imports are searched in.
    dynamic: DynamicOrder<'a>,
    inputs: InputsFrom<'a>,
    /// A creation's specialized entries, with the stage's function descriptor: the compile reads
    /// its specialized entry there, the one a check or key of the creation made, else one made
    /// there now ([`specialize_one`]). `None` specializes for the compile alone.
    kept: Option<(&'a StageEntries, &'a FunctionDescriptor)>,
}

impl<'a> Request<'a> {
    /// Entry `entry` of a bare module with the caller's inputs.
    #[cfg(test)]
    fn bitcode(
        bitcode: &'a [u8],
        entry: EntryRef<'a>,
        consts: &'a FunctionConstantValues,
        linked: &'a [LinkedSource<'a>],
        inputs: &'a LowerInputs,
    ) -> Request<'a> {
        Request {
            bitcode,
            entry,
            container: None,
            consts,
            linked,
            dynamic: DynamicOrder::default(),
            inputs: InputsFrom::Given(inputs),
            kept: None,
        }
    }
}

/// `consts` by index against `abi`'s declared constants: a value supplied by name takes the
/// index of the constant of that name (`setConstantValue:type:withName:`, an API contract, not a
/// heuristic). A value for a constant the module does not declare, by index or by name, is
/// ignored, as `newFunctionWithName:constantValues:` ignores it ("values that do not correspond
/// to a function constant in the named function are ignored", Metal Programming Guide, Function
/// Specialization).
fn constant_values(
    abi: &AirAbi,
    consts: &FunctionConstantValues,
) -> Result<BTreeMap<u32, ConstValue>, crate::air::error::Error> {
    use crate::air::error::{Error, RequestError};
    let mut out = BTreeMap::new();
    for (index, v) in &consts.by_index {
        if !abi.function_constants.iter().any(|c| c.index == *index) {
            continue;
        }
        let value = ConstValue {
            ty: v.ty,
            bytes: v.bytes.clone(),
        };
        if out.insert(*index, value).is_some() {
            return Err(Error::InvalidRequest(
                RequestError::DuplicateFunctionConstant { index: *index },
            ));
        }
    }
    for (name, v) in &consts.by_name {
        let Some(decl) = abi
            .function_constants
            .iter()
            .find(|c| c.name.to_string() == *name)
        else {
            continue;
        };
        let value = ConstValue {
            ty: v.ty,
            bytes: v.bytes.clone(),
        };
        if out.insert(decl.index, value).is_some() {
            return Err(Error::InvalidRequest(
                RequestError::DuplicateFunctionConstant { index: decl.index },
            ));
        }
    }
    Ok(out)
}

/// The linked functions the references of the module with ABI `abi` reach, by module: the function
/// of each referenced name, then the functions their own references name, in breadth-first order.
/// When an entry of the module takes a visible-function table or an intersection function table, or
/// a buffer it takes holds a visible-function table, every function a table can hold (`functions`
/// and `binary_functions`) is reached too, before the references, and each listed function carries
/// its table word. A name that no function of `linked` has is left to canonicalization (a call to
/// it may be dead); a function nothing references (or, without tables, nothing but a table could
/// call) is listed only where it is one a table can hold and its module decodes and has constexpr
/// samplers, which Metal numbers, and is otherwise never decoded. Each distinct module (bitcode
/// hash) is decoded once for all its functions. Every function of a referenced name is listed (the
/// same function listed twice, with the same constants, is one), so canonicalization sees two
/// different ones as ambiguous. Each function takes the values of the constants its module
/// declares, the others ignored (`constant_values`). A module that does not decode, or values it
/// refuses (two for one constant), are recorded rather than returned: canonicalization refuses them
/// only at a surviving call. Budget stops, cancellation, and internal errors return at once.
pub(crate) fn linked_modules(
    abi: &AirAbi,
    linked: &[LinkedSource<'_>],
    m: &mut Meter,
) -> Result<Vec<LinkedModule>, Error> {
    if linked.is_empty() {
        return Ok(Vec::new());
    }
    let keys: Vec<LinkedKey<'_>> = linked.iter().map(|s| s.key).collect();
    let handles = crate::pipeline::table::handles(&keys);
    let mut out: Vec<(Hash256, LinkedModule)> = Vec::new();
    let mut listed: Vec<(Hash256, &str, &FunctionConstantValues)> = Vec::new();
    let mut seen: BTreeSet<Box<[u8]>> = BTreeSet::new();
    let mut queue: VecDeque<Box<[u8]>> = VecDeque::new();
    let tables = abi.entries.iter().any(|e| {
        abi.reaches_function_tables(e)
            || e.params
                .iter()
                .any(|p| matches!(p.role, ParamRole::IntersectionTable(_)))
    });
    if tables {
        for (s, h) in linked.iter().zip(&handles) {
            if h.is_some() {
                m.charge(s.key.name.len())?;
                queue.push_back(s.key.name.as_bytes().into());
            }
        }
    }
    let refs = |abi: &AirAbi, queue: &mut VecDeque<Box<[u8]>>, m: &mut Meter| {
        for (_, symbol) in abi.visible_function_stubs() {
            m.charge(symbol.0.len())?;
            queue.push_back(symbol.0.clone());
        }
        Ok::<(), Error>(())
    };
    refs(abi, &mut queue, m)?;
    while let Some(name) = queue.pop_front() {
        m.tick(1)?;
        if seen.contains(&name) {
            continue;
        }
        m.charge(name.len())?;
        seen.insert(name.clone());
        for (s, handle) in linked.iter().zip(&handles) {
            m.tick(1)?;
            let key = &s.key;
            if !key.list.resolves_references() || key.name.as_bytes() != &*name {
                continue;
            }
            // The same function listed twice (in two lists, or twice in one) is one function.
            if listed
                .iter()
                .any(|(h, n, c)| *h == key.air_hash && *n == key.name && *c == key.constants)
            {
                continue;
            }
            m.charge(std::mem::size_of::<(Hash256, &str, &FunctionConstantValues)>())?;
            listed.push((key.air_hash, key.name, key.constants));
            let at = match out.iter().position(|(h, _)| *h == key.air_hash) {
                Some(at) => at,
                None => {
                    let code = match air::decode(s.bitcode, m) {
                        Ok(air::Decoded { module, abi, .. }) => {
                            refs(&abi, &mut queue, m)?;
                            Ok(LinkedCode { module, abi })
                        }
                        Err(
                            e @ (air::error::Error::Budget(_)
                            | air::error::Error::Cancelled
                            | air::error::Error::Internal(_)),
                        ) => return Err(e.into()),
                        Err(e) => Err(e),
                    };
                    let functions = Vec::new();
                    out.push((key.air_hash, LinkedModule { functions, code }));
                    out.len() - 1
                }
            };
            let (_, l) = out.get_mut(at).ok_or_else(|| {
                Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    "a linked module out of range",
                ))
            })?;
            let constants = match &l.code {
                Ok(code) => constant_values(&code.abi, key.constants),
                // Refused for the module's code first.
                Err(_) => Ok(BTreeMap::new()),
            };
            m.charge(std::mem::size_of::<LinkedName>())?;
            l.functions.push(LinkedName {
                name: name.clone(),
                constants,
                handle: handle.map(|h| h.word()),
            });
        }
    }
    // Metal's pipeline numbers the constexpr samplers of a function of `functions` nothing names:
    // one a table can hold that nothing reached is listed too, with its table word, where its
    // module decodes and has constexpr samplers, so that canonicalization, which imports no such
    // function, numbers its samplers as a group of its own (`Entry::linked_sampler_groups`).
    for (s, handle) in linked.iter().zip(&handles) {
        m.tick(1)?;
        let key = &s.key;
        let reached = listed
            .iter()
            .any(|(h, n, c)| *h == key.air_hash && *n == key.name && *c == key.constants);
        if handle.is_none() || reached {
            continue;
        }
        let at = match out.iter().position(|(h, _)| *h == key.air_hash) {
            Some(at) => at,
            None => match air::decode(s.bitcode, m) {
                Ok(air::Decoded { module, abi, .. }) => {
                    if abi.static_samplers.is_empty() {
                        continue;
                    }
                    let code = Ok(LinkedCode { module, abi });
                    let functions = Vec::new();
                    out.push((key.air_hash, LinkedModule { functions, code }));
                    out.len() - 1
                }
                Err(
                    e @ (air::error::Error::Budget(_)
                    | air::error::Error::Cancelled
                    | air::error::Error::Internal(_)),
                ) => return Err(e.into()),
                Err(_) => continue,
            },
        };
        let Some((_, l)) = out.get_mut(at) else {
            continue;
        };
        let constants = match &l.code {
            Ok(code) if !code.abi.static_samplers.is_empty() => {
                constant_values(&code.abi, key.constants)
            }
            _ => continue,
        };
        m.charge(std::mem::size_of::<(Hash256, &str, &FunctionConstantValues)>())?;
        listed.push((key.air_hash, key.name, key.constants));
        m.charge(std::mem::size_of::<LinkedName>() + key.name.len())?;
        l.functions.push(LinkedName {
            name: key.name.as_bytes().into(),
            constants,
            handle: handle.map(|h| h.word()),
        });
    }
    Ok(out.into_iter().map(|(_, l)| l).collect())
}

/// The `SpecRequest` of `consts` against the decoded ABI's declared constants.
pub(crate) fn spec_request(
    decoded: &air::Decoded,
    entry: EntryRef<'_>,
    consts: &FunctionConstantValues,
) -> Result<SpecRequest, Error> {
    let name = match entry {
        EntryRef::Name(n) => n.to_owned(),
        EntryRef::Index(i) => decoded
            .abi
            .entries
            .get(i as usize)
            .map(|e| e.name.to_string())
            .ok_or_else(|| {
                Error::InvalidDescriptor(DescriptorError::EntryNotFound {
                    requested: DiagName::new(format!("#{i}")),
                    available: u32::try_from(decoded.abi.entries.len()).unwrap_or(u32::MAX),
                })
            })?,
    };
    Ok(SpecRequest {
        entry: name,
        constants: constant_values(&decoded.abi, consts)?,
        call_stack_depth: 1,
    })
}

/// The Metal attribute indices the specialized vertex entry `spec` reads, with the numeric class
/// each is declared with: its `[[attribute(n)]]` parameters that survived function-constant
/// specialization and that an instruction uses (the parameters `lower::stage_io` plans). Empty
/// for any other stage.
pub(crate) fn attributes_read(
    spec: &air::Entry,
) -> std::collections::BTreeMap<u32, crate::pipeline::stage_io::IoNumeric> {
    use crate::pipeline::stage_io::IoNumeric;
    let mut out = std::collections::BTreeMap::new();
    if !matches!(spec.ep.stage, AbiStage::Vertex { .. }) {
        return out;
    }
    let Some(func) = spec.module.func(spec.func) else {
        return out;
    };
    let Some(body) = func.body.as_ref() else {
        return out;
    };
    let mut used = BTreeSet::new();
    for inst in body.insts.values() {
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                used.insert(a.raw());
            }
        });
    }
    for b in &spec.ep.params {
        if let ParamRole::VertexInput { location } = &b.role {
            if let AbiInt::Const(index) = location.index {
                if used.contains(&b.param) {
                    // The class the attribute is declared with: float for float forms, else the
                    // held unsigned integer (as `lower::stage_io` declares it). A type stage IO
                    // cannot hold is refused when the entry lowers.
                    let types = spec.module.types();
                    let float = match types.get(func.ty) {
                        Some(air::ir::Type::Function { params, .. }) => params
                            .get(b.param as usize)
                            .and_then(|t| crate::lower::arith::legal::form_of(types, *t).ok())
                            .is_some_and(|f| f.is_float()),
                        _ => false,
                    };
                    out.insert(
                        index,
                        if float {
                            IoNumeric::Float
                        } else {
                            IoNumeric::Uint
                        },
                    );
                }
            }
        }
    }
    out
}

/// The color attachments the specialized fragment entry `spec` reads (`[[color(n)]]`) whose
/// pixel format in `colors` is `RGBA8Unorm_sRGB` or `BGRA8Unorm_sRGB`
/// (`LoweringInputs::srgb_fetches`). Empty for any other stage.
pub(crate) fn srgb_fetches(
    spec: &air::Entry,
    colors: &[Option<ColorAttachmentDescriptor>],
) -> BTreeSet<u32> {
    if !matches!(spec.ep.stage, AbiStage::Fragment { .. }) {
        return BTreeSet::new();
    }
    spec.ep
        .params
        .iter()
        .filter_map(|b| match b.role {
            ParamRole::ColorInput {
                index: AbiInt::Const(color),
                ..
            } => Some(color),
            _ => None,
        })
        .filter(|&c| {
            colors
                .get(c as usize)
                .copied()
                .flatten()
                .is_some_and(|a| crate::pipeline::color_writes::srgb_fetched(a.pixel_format))
        })
        .collect()
}

/// The color attachments the specialized fragment entry `spec` writes (an output of either
/// dual-source index) whose pixel format in `colors` is a signed integer. Empty for any other
/// stage.
pub(crate) fn signed_colors_written(
    spec: &air::Entry,
    colors: &[Option<ColorAttachmentDescriptor>],
) -> BTreeSet<u32> {
    use crate::pipeline::sym::SampledType;
    if !matches!(spec.ep.stage, AbiStage::Fragment { .. }) {
        return BTreeSet::new();
    }
    spec.ep
        .outputs
        .iter()
        .filter_map(|o| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(a),
                ..
            } => Some(a),
            _ => None,
        })
        .filter(|a| {
            colors
                .get(*a as usize)
                .copied()
                .flatten()
                .and_then(|c| crate::metal::format::sampled_type(c.pixel_format))
                == Some(SampledType::Sint)
        })
        .collect()
}

/// The colors the specialized fragment entry `spec` writes (either dual-source index) that a
/// render pipeline whose color attachments are `colors` has no attachment for
/// (`LoweringInputs::unattached_colors`), but color 0 under alpha to coverage
/// (`alpha_to_coverage`), whose alpha Metal's coverage reads with or without the attachment.
pub(crate) fn unattached_colors_written(
    spec: &air::Entry,
    colors: &[Option<ColorAttachmentDescriptor>],
    alpha_to_coverage: bool,
) -> BTreeSet<u32> {
    if !matches!(spec.ep.stage, AbiStage::Fragment { .. }) {
        return BTreeSet::new();
    }
    spec.ep
        .outputs
        .iter()
        .filter_map(|o| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(a),
                ..
            } => Some(a),
            _ => None,
        })
        .filter(|&a| !(alpha_to_coverage && a == 0))
        .filter(|&a| colors.get(a as usize).copied().flatten().is_none())
        .collect()
}

/// The framebuffer-fetch state of pass `d` for the specialized fragment entry `spec` on `p`
/// (`LoweringInputs::framebuffer_fetch`): on a device that orders fetch only through the pixel
/// interlock, the state of every attachment the entry declares a `[[color(n)]]` input of a constant
/// index for, of every extended-range attachment it writes that the pass blends (whose blend the
/// function computes from the texel it reads), and of every attachment variant `v`'s instance keeps
/// out of its subpass that the entry writes through its first output
/// (`VariantKey::interlocked_colors`); `None` on another device or for an entry that declares none
/// of them. On a device that orders fetch through rasterization-order input attachments, the state
/// of every extended-range attachment the entry writes that the pass blends (whose blend the
/// function computes over its input attachment, `lower::fragment`'s `input_blend`), and of every
/// word attachment it writes under a mask of some of the word's channels (whose word the function
/// merges into its input attachment's, `color_writes::merged_word`). Where the
/// profile asks for Metal's 16-bit float blend of a `half4` output (`DevicePolicy::half_blend`) on
/// a device that orders fetch through the interlock, also every such attachment the entry writes,
/// those it names for that alone in `FramebufferFetchInputs::half_blends`.
pub(crate) fn fetch_inputs(
    spec: &air::Entry,
    d: FragmentPass<'_>,
    v: &VariantKey,
    p: &DeviceProfile,
) -> Option<crate::pipeline::lower_inputs::FramebufferFetchInputs> {
    let interlock = crate::lower::fragment::interlock_orders(p);
    let inputs = crate::lower::fragment::inputs_order(p);
    if !matches!(spec.ep.stage, AbiStage::Fragment { .. }) || !(interlock || inputs) {
        return None;
    }
    // Each 16-bit float attachment a `half` output of a module without fast math blends into,
    // where the profile asks for Metal's blend there, which the function computes in the storage
    // form where it is exact.
    let half_blends: BTreeSet<u32> = match crate::lower::fragment::blends_half_in_shader(p) {
        true => d
            .color_attachments
            .iter()
            .zip(0u32..)
            .filter_map(|(c, color)| c.as_ref().map(|c| (c, color)))
            .filter(|&(c, color)| {
                crate::lower::fragment::half_blend_attachment(
                    c.pixel_format,
                    c.blending_enabled,
                    c.write_mask,
                    crate::pipeline::lower_inputs::BlendEquation::of(c),
                ) && crate::lower::fragment::half_output(spec, color)
            })
            .map(|(_, color)| color)
            .collect(),
        false => BTreeSet::new(),
    };
    let blended = |color: u32| {
        d.color_attachments
            .get(color as usize)
            .copied()
            .flatten()
            .is_some_and(|c| {
                c.blending_enabled
                    && crate::metal::format::ExtendedRange::of(c.pixel_format).is_some()
            })
    };
    // A word attachment whose writes the function blends over, or merges into, the word it reads.
    let stored_word = |color: u32| {
        d.color_attachments
            .get(color as usize)
            .copied()
            .flatten()
            .and_then(|c| crate::pipeline::color_writes::stored_word(&c, p))
            .is_some()
    };
    // A word attachment whose writes the function merges into the word its input attachment
    // holds.
    let merged_word = |color: u32| {
        d.color_attachments
            .get(color as usize)
            .copied()
            .flatten()
            .and_then(|c| {
                crate::pipeline::color_writes::merged_word(color, &c, d.alpha_to_coverage, p)
            })
            .is_some()
    };
    let read: BTreeSet<u32> = spec
        .ep
        .params
        .iter()
        .filter_map(|b| match b.role {
            ParamRole::ColorInput {
                index: AbiInt::Const(color),
                ..
            } if interlock => Some(color),
            _ => None,
        })
        .chain(spec.ep.outputs.iter().filter_map(|o| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } if blended(color)
                || (interlock && stored_word(color))
                || (inputs && merged_word(color)) =>
            {
                Some(color)
            }
            // A color the draw's instance keeps out of its subpass, written through the storage
            // form as a fetched one is.
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } if interlock && v.interlocked_colors.contains(&color) => Some(color),
            // An extended-range color whose `half` shadow the draw's instance holds, written
            // through the storage form against it.
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } if interlock && v.extended_shadows.contains(&color) => Some(color),
            _ => None,
        }))
        .collect();
    // Those the pipeline names for that blend alone, which the function leaves to the device
    // where the storage form is not exact.
    let optional: BTreeSet<u32> = half_blends.difference(&read).copied().collect();
    let read: BTreeSet<u32> = read.union(&half_blends).copied().collect();
    if read.is_empty() {
        return None;
    }
    // The blend over an input attachment depends on the attachments' state alone: its write is
    // the fixed function's, which the tests, coverage and views take as any other's.
    Some(match interlock {
        true => crate::pipeline::lower_inputs::FramebufferFetchInputs {
            half_blends: optional,
            ..crate::pipeline::lower_inputs::FramebufferFetchInputs::of(
                read,
                d.color_attachments,
                d.depth_or_stencil,
                d.alpha_to_coverage,
                d.max_vertex_amplification_count,
            )
        },
        false => crate::pipeline::lower_inputs::FramebufferFetchInputs::of(
            read,
            d.color_attachments,
            false,
            false,
            1,
        ),
    })
}

/// A fragment stage's lowering inputs under variant `v` on `p` that need no decoded entry: the
/// key's, whether pass `d` rasterizes more than one sample (`LoweringInputs::multisampled`), and
/// how its color attachments take float outputs (`LoweringInputs::color_writes`).
pub(crate) fn fragment_inputs(
    d: FragmentPass<'_>,
    v: &VariantKey,
    p: &DeviceProfile,
) -> LowerInputs {
    let mut out = v.lowering_inputs(crate::pipeline::sym::Stage::Fragment);
    out.multisampled = d.raster_sample_count > 1 && d.rasterization_enabled;
    out.max_call_stack_depth = call_stack_input(d.max_call_stack_depth);
    // Every attachment's, whether the function writes it or not: the map needs no decode, and the
    // function records the entries of those it writes.
    out.color_writes = crate::pipeline::color_writes::color_writes(
        0..d.color_attachments.len() as u32,
        d.color_attachments,
        (d.alpha_to_coverage, d.texture_write_rounding_mode),
        p,
    );
    out.extended_range_colors = crate::pipeline::color_writes::extended_range_colors(
        d.color_attachments,
        d.texture_write_rounding_mode,
    );
    out.word_colors =
        crate::pipeline::color_writes::word_colors(d.color_attachments, d.alpha_to_coverage, p);
    out.rotated_colors = crate::pipeline::color_writes::rotated_colors(d.color_attachments, p);
    out.alpha_in_red_colors =
        crate::pipeline::color_writes::alpha_in_red_colors(d.color_attachments, p);
    out.rotated_factors = crate::pipeline::color_writes::rotated_factors(d.color_attachments, p);
    out.rotated_blends = crate::pipeline::color_writes::rotated_blends(d.color_attachments, p);
    out.sample_mask = d.sample_mask;
    out.alpha_test = d.alpha_test;
    out.depth_compare_clamp_mask = d.depth_compare_clamp_mask;
    out.half_output_lanes = crate::pipeline::color_writes::half_output_lanes(d.color_attachments);
    out.texture_write_rounding = d.texture_write_rounding_mode;
    out.depth16_codes = depth16_codes(d, v, p);
    out.depth16_bias_raises = v.depth16_bias_raises.filter(|_| out.depth16_codes);
    out
}

/// Whether a fragment stage of pass `d` under variant `v` on `p` stores Metal's `Depth16Unorm`
/// codes itself (`LoweringInputs::depth16_codes`): the depth it renders against is
/// `Depth16Unorm` (`VariantKey::depth16_pass`, else the pipeline's depth attachment), and
/// [`depth16_stage_form`] holds, for a biased draw under `VariantKey::depth16_biased`.
pub(crate) fn depth16_codes(d: FragmentPass<'_>, v: &VariantKey, p: &DeviceProfile) -> bool {
    v.depth16_pass.unwrap_or(d.depth16)
        && depth16_stage_form(
            d.rasterization_enabled,
            d.raster_sample_count,
            v.depth16_biased,
            p,
        )
}

/// Whether a pipeline that rasterizes (`rasterizes`) `samples` samples per pixel on `p` takes
/// Metal's codes from its fragment stage where it renders against a `Depth16Unorm` depth, for a
/// draw that biases it where `biased`: it rasterizes one sample, and the profile lacks
/// `DriverDeviation::MetalDepth16Stores`, or for a biased draw
/// `DriverDeviation::MetalDepth16BiasedStores`. At more than one sample the device's depth is
/// each sample's own, which one written depth would replace.
pub(crate) fn depth16_stage_form(
    rasterizes: bool,
    samples: u32,
    biased: bool,
    p: &DeviceProfile,
) -> bool {
    use crate::error::DriverDeviation::{MetalDepth16BiasedStores, MetalDepth16Stores};
    let stores = if biased {
        MetalDepth16BiasedStores
    } else {
        MetalDepth16Stores
    };
    rasterizes && samples <= 1 && !p.takes_driver_deviation(stores)
}

/// The attributes the vertex function `spec` fetches in fixed function under `inputs` from
/// layouts `desc` steps per instance, where taking them so makes it write some varying the same
/// at every vertex of a primitive (`LoweringInputs::instanced_attributes`); else none, and the
/// function keeps the key of any step.
fn instanced_attributes(
    spec: &air::Entry,
    inputs: &LowerInputs,
    desc: Option<&VertexDescriptor>,
    m: &mut Meter,
) -> Result<BTreeSet<u32>, Error> {
    let (Some(projection), Some(desc)) = (&inputs.vertex_fetch, desc) else {
        return Ok(BTreeSet::new());
    };
    let stepped: BTreeSet<u32> = projection
        .attributes
        .iter()
        .filter(|(_, f)| matches!(f, crate::pipeline::lower_inputs::VertexFetch::FixedFunction))
        .filter(|(a, _)| {
            desc.attributes
                .get(a)
                .and_then(|attr| desc.layouts.get(&attr.buffer_index))
                .is_some_and(|l| l.step_function == crate::metal::VertexStepFunction::PerInstance)
        })
        .map(|(a, _)| *a)
        .collect();
    if stepped.is_empty() {
        return Ok(stepped);
    }
    let without = crate::lower::stage_io::uniform_varyings(spec, inputs, m)?;
    let mut with = inputs.clone();
    with.instanced_attributes = stepped;
    let uniform = crate::lower::stage_io::uniform_varyings(spec, &with, m)?;
    Ok(if uniform == without {
        BTreeSet::new()
    } else {
        with.instanced_attributes
    })
}

/// The caller's inputs `i` for the specialized entry `spec`, its object payload kept only where it
/// widens a mesh function's payload arena (`LoweringInputs::object_payload`): any other function,
/// and a mesh function whose own arena holds it, compiles as without it.
fn given_inputs(spec: &air::Entry, i: &LowerInputs) -> LowerInputs {
    let mut out = i.clone();
    out.object_payload = i.object_payload.filter(|&bytes| {
        matches!(spec.ep.stage, AbiStage::Mesh { .. })
            && crate::lower::mesh::payload_size(spec)
                .is_some_and(|own| u64::from(bytes) > crate::memory::arena::payload_bytes(own))
    });
    out
}

/// The lowering inputs of the specialized entry `spec` (see [`InputsFrom`]).
fn lower_inputs(
    spec: &air::Entry,
    inputs: InputsFrom<'_>,
    p: &DeviceProfile,
    m: &mut Meter,
) -> Result<LowerInputs, Error> {
    match inputs {
        InputsFrom::Given(i) => Ok(given_inputs(spec, i)),
        // A post-tessellation vertex function pulls every attribute it reads.
        InputsFrom::Vertex(desc, variant, depth, primitive_id, tessellation)
            if matches!(spec.ep.stage, AbiStage::Vertex { patch: Some(_) }) =>
        {
            let attributes = crate::lower::tessellation::attributes_read(spec);
            let projection =
                crate::pipeline::link::tessellation::patch_fetch_projection(desc, &attributes)?;
            let mut out = variant.lowering_inputs(crate::pipeline::sym::Stage::Vertex);
            out.vertex_fetch = Some(projection);
            out.max_call_stack_depth = call_stack_input(depth);
            // Only an emulated function bakes the tessellation state in, and writes its patch's
            // position for a fragment function that reads `[[primitive_id]]`; one drawn through the
            // device's tessellation stages is refused under the field (`lower::stage_io`).
            if crate::lower::tessellation::emulates(p) {
                out.tessellation = tessellation.copied();
            }
            out.primitive_id_varying = primitive_id;
            Ok(key::canonical_inputs(&out))
        }
        InputsFrom::Vertex(desc, variant, depth, primitive_id, _) => {
            let attributes = attributes_read(spec);
            // The indexed capture form, the primitive ID's de-indexed form and the private list
            // form pull every attribute, whatever the caller's projection would fetch in fixed
            // function.
            let pull_every = variant
                .captured_positions
                .is_some_and(|f| f.index_type().is_some())
                || variant.primitive_id_indices.is_some()
                || variant.private_list;
            let projection = crate::pipeline::link::vertex::vertex_fetch_projection_for(
                desc,
                &attributes,
                p,
                &variant.pulled_vertex_buffers,
                pull_every,
            )?;
            let mut out = variant.lowering_inputs(crate::pipeline::sym::Stage::Vertex);
            out.vertex_fetch = Some(projection);
            out.max_call_stack_depth = call_stack_input(depth);
            // Only on a device that cannot declare a fragment `PrimitiveId`: elsewhere a plain
            // vertex function's fragment function reads Vulkan's, Metal's there.
            out.primitive_id_varying = primitive_id && primitive_id_varying_on(p);
            out.instanced_attributes = instanced_attributes(spec, &out, desc, m)?;
            Ok(key::canonical_inputs(&out))
        }
        InputsFrom::Fragment(d, variant, written, primitive, varyings, uniform) => {
            let mut out = fragment_inputs(d, variant, p);
            out.color_group = d.color_group.cloned();
            out.signed_color_attachments = signed_colors_written(spec, d.color_attachments);
            out.unattached_colors =
                unattached_colors_written(spec, d.color_attachments, d.alpha_to_coverage);
            out.framebuffer_fetch = fetch_inputs(spec, d, variant, p);
            out.srgb_fetches = srgb_fetches(spec, d.color_attachments);
            let read = raw_indices_read(spec);
            out.viewport_index_varying = written.viewport && read.viewport;
            out.render_target_array_index_varying = written.layer && read.layer;
            // Behind a vertex function that writes the primitive ID's varying, on a device that
            // declares a fragment `PrimitiveId` (behind an emulated post-tessellation vertex
            // function): elsewhere the fragment function reads the varying without the field.
            out.primitive_id_varying =
                written.primitive_id && read.primitive_id && !primitive_id_varying_on(p);
            out.primitive_varyings = varyings_read(spec, &out)
                .filter(|k| primitive.contains(k))
                .collect();
            // Only where an input is unwritten, which would not link otherwise: no analysis
            // runs for a pipeline whose vertex function writes every input.
            let unwritten = |varyings: &BTreeSet<VaryingKey>| {
                spec.ep.params.iter().any(|p| match &p.role {
                    ParamRole::FragmentInput(v) => !varyings.contains(&VaryingKey::user(&v.key.0)),
                    _ => false,
                })
            };
            if let Some(varyings) = varyings.filter(|v| unwritten(v)) {
                out.absent_varyings =
                    crate::reflection::dropped_fragment_inputs(spec, d.color_attachments, m)?
                        .into_iter()
                        .map(|key| VaryingKey::user(&key.0))
                        .filter(|key| !varyings.contains(key))
                        .collect();
            }
            // The interpolated `float` and `half` inputs the vertex function writes the same at
            // every vertex of a primitive: none is absent or read per primitive.
            if !uniform.is_empty() {
                out.uniform_varyings = interpolated_float_inputs(spec)
                    .filter(|k| {
                        uniform.contains(k)
                            && !out.absent_varyings.contains(k)
                            && !out.primitive_varyings.contains(k)
                    })
                    .collect();
            }
            Ok(key::canonical_inputs(&out))
        }
    }
}

/// What planning produced: the key, the specialized entry, what its code read before
/// specialization (for its kept reflection), its plan, and the kernel facts.
struct Planned {
    key: FunctionCompileKey,
    spec: air::Entry,
    before: reflection::UnspecializedReads,
    plan: Plan,
    kernel: KernelFacts,
}

/// What [`specialize_one`] makes: the AIR hash, the entry name, the specialized entry, and what
/// its code read before specialization.
#[derive(Clone)]
struct Specialized {
    air_hash: Hash256,
    entry: String,
    spec: air::Entry,
    before: reflection::UnspecializedReads,
}

/// Decode, cross-check a metallib function's container tags, and specialize: the AIR hash, the
/// entry name, the specialized entry, and what its code read before.
fn specialize_one(r: &Request<'_>, m: &mut Meter) -> Result<Specialized, Error> {
    #[cfg(test)]
    tests::inject::panic_if_armed();
    // The creation's entry of this stage: the same request's, made by a check or key of the
    // creation, else made there now.
    if let Some((entries, fd)) = r.kept {
        let kept = entries.specialized(fd, r.inputs.call_stack_depth(), m)?;
        return Ok(Specialized::clone(&kept));
    }
    let air_hash = container::sha256(r.bitcode);
    let mut decoded = air::decode(r.bitcode, m)?;
    if let Some(f) = r.container {
        decoded.abi.check_container(&decoded.module, f, m)?;
    }
    let mut req = spec_request(&decoded, r.entry, r.consts)?;
    // A stage compiles under its pipeline's `maxCallStackDepth` for it.
    req.call_stack_depth = r.inputs.call_stack_depth();
    // Only a module that declares function constants needs them; specialization selects the
    // entry the same way.
    let before = if reflection::specialization_decides_reads(&decoded.abi) {
        let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
        reflection::UnspecializedReads::of(&decoded.module, &decoded.abi, ep)
    } else {
        reflection::UnspecializedReads::default()
    };
    let spec = specialize_request(decoded, &req, r, m)?;
    Ok(Specialized {
        air_hash,
        entry: req.entry,
        spec,
        before,
    })
}

/// [`specialize_one`], then Metal's grouping of multiply chains (`lower::arith::regroup`) and of
/// fast-math sums (`lower::arith::reassoc`), and the order Apple's simdgroup lockstep gives
/// threadgroup accesses made explicit for `p`: the entry selection reads, whose instruction indices
/// a refusal's location names. [`plan_one`] and `DeviceProfile::specialized_entry` both build it
/// here, so a tool that resolves a refusal's location reads the compile's own entry.
fn ordered_entry(r: &Request<'_>, p: &DeviceProfile, m: &mut Meter) -> Result<Specialized, Error> {
    let Specialized {
        air_hash,
        entry,
        spec,
        before,
    } = specialize_one(r, m)?;
    // Metal's pairing of a multiply chain's lane-uniform factors, before lockstep's barriers name
    // their instructions.
    let spec = crate::lower::arith::regroup::apply(spec, m)?;
    // Metal's grouping of each fast-math sum whose grouping the rank pins.
    let spec = crate::lower::arith::reassoc::apply(spec, m)?;
    let spec = crate::lower::simd::lockstep::order(spec, p, m)?;
    Ok(Specialized {
        air_hash,
        entry,
        spec,
        before,
    })
}

/// Entry `entry` of the bare module `bitcode` as [`ordered_entry`] builds it, with the function
/// constants `consts`, the `linked` functions, and the lowering inputs `inputs` (a kernel's
/// `maxCallStackDepth`), under `catch_unwind`: the entry a compile of the same request selects
/// from (`pipeline::testing::specialized_entry`).
#[cfg(test)]
pub(crate) fn specialized_bitcode_entry(
    bitcode: &[u8],
    entry: EntryRef<'_>,
    (consts, linked, inputs): (&FunctionConstantValues, &[LinkedSource<'_>], &LowerInputs),
    p: &DeviceProfile,
    m: &mut Meter,
) -> Result<air::Entry, Error> {
    guarded(|| {
        let r = Request::bitcode(bitcode, entry, consts, linked, inputs);
        ordered_entry(&r, p, m).map(|specialized| specialized.spec)
    })
}

/// [`ordered_entry`], then analyze and select.
fn plan_one(r: &Request<'_>, p: &DeviceProfile, m: &mut Meter) -> Result<Planned, Error> {
    let Specialized {
        air_hash,
        entry,
        spec,
        before,
    } = ordered_entry(r, p, m)?;
    let spec = grouped_entry(spec, r.inputs, p)?;
    let inputs = lower_inputs(&spec, r.inputs, p, m)?;
    // A flat variant runs its grid in threadgroups of its own: only a kernel that does not observe
    // its partition computes the same there.
    let partition_oblivious = partition::oblivious(&spec);
    if inputs.flat_grid && !partition_oblivious {
        return Err(Unsupported::not_representable(Feature::GpuEncodedIndirectCommands).into());
    }
    // A folded variant runs one threadgroup dispatch of its own threadgroup size: neither a flat
    // grid nor a device replay's bindings.
    if inputs.folded_groups && (inputs.flat_grid || inputs.replay_bindings) {
        return Err(Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into());
    }
    let linked: Vec<LinkedKey<'_>> = r.linked.iter().map(|s| s.key).collect();
    let key = key::function_key(
        &air_hash,
        &entry,
        r.consts,
        &p.fingerprint(),
        &inputs,
        (&linked, &r.dynamic.key()),
    );
    let memory = crate::memory::plan::analyze_memory(&spec, p, m)?;
    let plan = crate::lower::select::select(&spec, &memory, p, &inputs, m)?;
    let kernel = KernelFacts {
        max_total_threads: spec.ep.stage.max_total_threads(),
        flat_local_size: plan
            .dispatch
            .as_ref()
            .is_some_and(|d| d.local_size == LocalSizeForm::Flat),
        // The static part: the objects, and with `[[threadgroup(n)]]` regions the 16-byte-aligned
        // offset where the first region starts. With typed threadgroup objects (no regions), the
        // static part of Metal's own layout, which holds them too.
        static_threadgroup_bytes: if memory.typed_shared.is_empty() {
            crate::memory::arena::TgShape::of(&memory.tg_arena)
                .static_bytes()
                .unwrap_or(u64::MAX)
        } else {
            memory.tg_metal_static
        },
        partition_oblivious,
    };
    Ok(Planned {
        key,
        spec,
        before,
        plan,
        kernel,
    })
}

fn compile_one(
    r: &Request<'_>,
    p: &DeviceProfile,
    m: &mut Meter,
    word_cap: u64,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_kept(r, p, m, word_cap, |_| ()).map(|(artifact, _, ())| artifact)
}

/// [`compile_one`], also returning the plan and what `keep` takes of the finished typed module
/// before it is serialized ([`compile_bitcode_kept`] keeps a copy).
fn compile_kept<K>(
    r: &Request<'_>,
    p: &DeviceProfile,
    m: &mut Meter,
    word_cap: u64,
    keep: impl FnOnce(&TypedModule) -> K,
) -> Result<(Arc<FunctionArtifact>, Plan, K), Error> {
    let Planned {
        key,
        spec,
        before,
        plan,
        kernel,
    } = plan_one(r, p, m)?;
    plan.estimate.check(word_cap)?;
    let crate::lower::emit::Emitted {
        mut module,
        merged,
        loop_free,
    } = crate::lower::emit::emit_module(&spec, &plan, m)?;
    // Every driver compiles each repeated region of the entry again; above the measured size,
    // repeated regions are written once, as a loop. With the loops final, the lowering's offers for
    // where no loop holds them are weighed: a `half` decoded by the device's widening outside every
    // loop.
    let held = std::mem::size_of_val(loop_free.as_slice());
    m.charge(held)?;
    let taken = crate::spirv::roll::roll_entry(&mut module, m)
        .and_then(|_| crate::spirv::dead::take_outside_loops(&mut module, &loop_free, m));
    m.release(held);
    drop(loop_free);
    let decoded_natively = taken? > 0;
    // On a driver whose compiler loses a simdgroup's storage writes after an `optnone` call only
    // its helper invocations make, those invocations take another arm where that is exact.
    let helper_guard = crate::spirv::helper_branches::guard_for(&mut module, p, m)?;
    let function_tables = spec.function_tables.clone();
    let reaches_command_buffers = spec.may_encode_indirect_commands();
    let command_writes = spec.command_writes();
    let writes_textures = spec.may_write_textures();
    // A binary function's and a function-pointer group's numbering is not measured.
    let unmeasured = r.linked.iter().any(|s| {
        matches!(
            s.key.list,
            crate::pipeline::descriptor::LinkedList::BinaryFunctions
                | crate::pipeline::descriptor::LinkedList::Group(_)
        )
    });
    let (numbered_samplers, numbering_differs) = numbered::samplers(&spec, unmeasured);
    // The slot writes are kept whatever the reflection could describe, so a pipeline of this
    // function always answers what it may store through each of its slots.
    let KeptAnswers {
        reflection,
        slot_writes,
    } = kept_reflection(&spec, &before)?;
    // Decoded AIR is not needed past emission.
    drop(spec);
    // The function's own side effects; the `ExecutionStatus` buffer's words (the resident-miss
    // count's) are the product's.
    let writes_memory = crate::spirv::requirements::writes_memory(&module)?;
    // A fragment function's texture slots read only at the fragment's own pixel.
    let mut own_pixel_textures = crate::spirv::requirements::own_pixel_textures(&module, m)?;
    // Those a draw can read through a snapshot an earlier draw of the same primitives took, of a
    // function that writes no memory. A variant compiled to read them so states its own: its only
    // texel access of each is the entry's copy, at the fragment's own pixel.
    let snapshots: Vec<u32> = plan.families.fragment.snapshot.keys().copied().collect();
    let snapshot_textures = match (snapshots.is_empty(), writes_memory) {
        (false, _) => {
            if !crate::spirv::requirements::snapshot_copies_alone(&module, &snapshots, m)? {
                return Err(Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    "a snapshot variant that reads its texture outside the copy",
                )));
            }
            own_pixel_textures.extend(&snapshots);
            own_pixel_textures.sort_unstable();
            own_pixel_textures.dedup();
            snapshots
        }
        (true, true) => Vec::new(),
        (true, false) => {
            crate::spirv::requirements::snapshot_textures(&module, &own_pixel_textures, m)?
        }
    };
    // A `half` rounded through `OpQuantizeToF16` (`QuantizeForm::Device`) because the device's
    // is measured to round as Metal does: the module requires that fact. Elsewhere a `half` rounds
    // by integer arithmetic, exact on every device, and no module holds a quantize.
    let mut precision = plan.precision.clone();
    let specialized = plan.specialized.clone();
    let device_quantize = plan.families.arith.quantize
        == crate::spirv::builder::QuantizeForm::Device
        && crate::lower::arith::narrow::quantizes(&module);
    // A `half` rounded through the device's conversion to 16 bits and back, behind a barrier
    // Metal's compiler cannot fold it through, with no quantize, because that conversion is
    // Metal's: the module requires that fact.
    let converts = plan.families.arith.quantize == crate::spirv::builder::QuantizeForm::Converted
        && crate::lower::arith::narrow::rounds_by_conversion(&module, m)?;
    // A `half` rounded through the device's conversion and its widening, each decorated
    // `NoContraction`, under the float controls that make it Metal's, computed by the device's
    // 16-bit arithmetic, which they make Metal's too, or read from its bits by the device's
    // widening, which they make exact but for a NaN's bits: the entry declares them, and so
    // requires the device's float-controls properties for them, and no fact. A module that widens
    // so leaves a NaN's bits to the device, as the profile does (`NanMinMaxPolicy::Native`).
    if plan.families.arith.quantize == crate::spirv::builder::QuantizeForm::Controlled {
        let widens = crate::lower::arith::narrow::widens_bits(&module, m)?;
        if widens
            || crate::lower::arith::narrow::computes_in_16_bits(&module)
            || crate::lower::arith::narrow::rounds_by_conversion(&module, m)?
        {
            let modes = crate::lower::arith::narrow::controlled_modes(p).unwrap_or_default();
            module.declare_modes(&modes)?;
        }
        if widens && !precision.contains(&PrecisionNote::NativeNanEncoding) {
            precision.push(PrecisionNote::NativeNanEncoding);
            precision.sort_unstable();
        }
    }
    // Native half rounding: a module that rounds through the device's conversion or quantize, or
    // decodes a `half` by the device's widening, where the profile does not take the deviation
    // making it Metal's records the kind, and requires no fact.
    let native_half = if device_quantize || converts || decoded_natively {
        crate::lower::arith::narrow::native_rounding(p, plan.families.arith.quantize)
    } else {
        None
    };
    if let Some(note) = native_half {
        precision.push(note);
        precision.sort_unstable();
    }
    // A driver release past its deviations' measurement: every module relies on them.
    if !p.fields().deviations_beyond.is_empty() {
        precision.push(PrecisionNote::BeyondMeasuredDeviation);
        precision.sort_unstable();
    }
    if helper_guard.exposed {
        precision.push(PrecisionNote::HelperBranchDropsSimdgroupStores);
        precision.sort_unstable();
    }
    if numbering_differs {
        precision.push(PrecisionNote::LinkedSamplerNumbering);
        precision.sort_unstable();
    }
    // A sample left to the sampler of a device that samples natively: the device's answer.
    let sampled = plan.families.sampler.metal_sampler || plan.families.sampler.device_sampling;
    if sampled
        && p.fields()
            .enabled
            .properties
            .contains(&crate::device::requirements::VkProperty::DeviceSampling)
        && !precision.contains(&PrecisionNote::DeviceSampling)
    {
        precision.push(PrecisionNote::DeviceSampling);
        precision.sort_unstable();
    }
    crate::spirv::verify::verify(&module, p, m)?;
    let kept = keep(&module);
    let mut symbols =
        crate::pipeline::link::contract::stage_symbols(&module, &plan.families.stage_io.interface)?;
    // The bits of each specialization constant the module never reads.
    crate::spirv::spec_bits::mark_unread(&module, &mut symbols.spec, &specialized, m)?;
    // The serializer's id tables live while it writes.
    let ids = crate::spirv::serialize::id_table_bytes(&module, p);
    m.charge(ids)?;
    let module = crate::spirv::serialize::serialize_for(module, p)?;
    m.release(ids);
    m.charge(module.footprint())?; // The retained words and relocations.

    // The estimate refused early; the cap holds on what was emitted.
    crate::lower::estimate::check_emitted(module.words.len(), word_cap)?;
    let interface = stage_interface(
        &plan,
        symbols,
        writes_memory,
        &merged,
        &function_tables,
        (reaches_command_buffers, command_writes),
        (writes_textures, numbered_samplers),
        (own_pixel_textures, snapshot_textures),
    );
    let mut requirements = module.requirements.clone();
    // Pairs only the device's simdgroup lockstep orders.
    if plan.families.simd.device_lockstep != 0 {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::SimdgroupLockstep);
    }
    // A filtered `half` sample left to the device's filter because it is Metal's. On a device that
    // samples natively, every sample left to the device's sampler requires that instead, and the
    // stage records it.
    let sampling = &plan.families.sampler;
    if sampling.metal_sampler || sampling.device_sampling {
        use crate::device::requirements::VkProperty;
        let native = p
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::DeviceSampling);
        if native {
            requirements.properties.insert(VkProperty::DeviceSampling);
        } else if sampling.metal_sampler {
            requirements.properties.insert(VkProperty::MetalSampler);
        }
    }
    // An 8-bit unsigned normalized linear sample left to the device's filter because it lands
    // within one code of Metal's at Metal's footprint step.
    if plan.families.sampler.native_filter {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::Unorm8LinearFilter);
    }
    // A `half` fused multiply-add left to the device's 16-bit `Fma` because it is Metal's.
    if plan.families.arith.f16_fma {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalHalfFma);
    }
    // Metal's `float` and `half` minimum and maximum family left to the device's `NMin`, `NMax`
    // and `NClamp` because they are Metal's.
    if plan.families.arith.device_min_max {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalMinMax);
    }
    // `half` `+ - * /` left to the device's 16-bit arithmetic because it is Metal's, and a `half`
    // load's decode left to the device's widening because that measurement makes it Metal's.
    if plan.families.arith.f16_arith || !plan.families.arith.widened.is_empty() {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalHalfArithmetic);
    }
    // A fragment's simdgroup lane read from its subgroup lane because the device's fragment
    // subgroups are Metal's simdgroups.
    if plan.families.simd.metal_lane {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalFragmentSimdgroups);
    }
    // A fragment's quads run whole without `RequireFullQuadsKHR` because the device runs its
    // helpers as Metal does.
    if plan.families.simd.metal_full_quads {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalFullQuads);
    }
    // `float` denormals left to the device because it reads them as Metal does.
    if plan.families.arith.device_denormals {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalFloatDenormals);
    }
    // A `half` rounding left to the device's `OpQuantizeToF16` because it rounds as Metal does.
    if device_quantize && native_half.is_none() {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalHalfQuantize);
    }
    // A `half` rounding left to the device's conversion because it is Metal's.
    if converts && native_half.is_none() {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalHalfConversion);
    }
    // Center varyings read as they are as their pixel form, because the device reads them at
    // the pixel center as Metal does.
    if plan.families.stage_io.center_varyings_by_device {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalCenterVaryings);
    }
    // Varyings the device interpolates, with no note, because it interpolates as Metal does.
    if plan.families.stage_io.interpolation_by_metal {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalInterpolation);
    }
    // An index stored to its builtin although it differs among a primitive's vertices, because
    // the device takes the first vertex's as Metal does.
    if plan.families.stage_io.first_vertex_indices_by_device {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::MetalFirstVertexIndices);
    }
    // Color channels written as computed, because the device converts them as Metal does.
    requirements
        .properties
        .extend(plan.families.stage_io.color_properties.iter().copied());
    // Half-float storage writes left to the device's conversion, or their NaN to its store of
    // `0x7fc00000`, because the device stores Metal's half.
    requirements
        .properties
        .extend(plan.families.texture.storage_properties.iter().copied());
    // Depth comparisons the device makes, with no note, because its depth images keep values
    // outside `[0, 1]` as Metal's do.
    if plan.families.texture.kept_depth_range {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::DepthImageKeepsValuesOutsideUnitRange);
    }
    // The set-1 heap's binding flags need their features beside the module's.
    requirements
        .features
        .extend(crate::lower::texture::heap_layout_features(&plan));
    requirements
        .features
        .extend(crate::lower::sampler::heap_layout_features(&plan));
    // Set 1's arrays are written while command buffers reading other elements are pending
    // where the profile has the feature (`link::heap_unused_while_pending`).
    let heap = interface
        .descriptors
        .iter()
        .any(|d| matches!(d.slot.kind, crate::pipeline::sym::ResourceKind::Heap(_)));
    if heap && crate::pipeline::link::heap_unused_while_pending(p) {
        requirements.features.insert(
            crate::device::requirements::VkFeature::DescriptorBindingUpdateUnusedWhilePending,
        );
    }
    // On a portability-subset device interpolation functions also need the subset's member
    // (`VUID-RuntimeSpirv-shaderSampleRateInterpolationFunctions-06325`); lowering emits them
    // only where the device has it.
    {
        use crate::device::requirements::{SpirvCapability, VkExtension, VkFeature};
        let interpolates = requirements
            .spirv_capabilities
            .contains(&SpirvCapability::of(
                crate::spv::Capability::InterpolationFunction,
            ));
        if interpolates
            && p.fields()
                .enabled
                .extensions
                .contains(&VkExtension::KhrPortabilitySubset)
        {
            requirements
                .features
                .insert(VkFeature::ShaderSampleRateInterpolationFunctions);
        }
    }
    let artifact = Arc::new(FunctionArtifact {
        key,
        module,
        interface,
        facts: plan.facts.clone(),
        subgroup: plan.subgroup,
        // Implementation-defined where a tangled instruction runs without maximal reconvergence
        // (`lower::simd` chooses the entry's reconvergence mode).
        reconvergence: crate::lower::simd::reconvergence(&plan),
        precision,
        strategies: emitted_census(&plan, helper_guard),
        blob: plan.families.access.blob.as_ref().map(|b| b.bytes.clone()),
        requirements,
        variant_bits: 0,
        profile: p.fingerprint(),
        kernel,
        reflection,
        slot_writes,
        specialized,
    });
    Ok((artifact, plan, kept))
}

/// What a pipeline's reflection states for the compiled entry `spec`, kept in its artifact for a
/// `StageSource::Compiled` stage, and what the stage may store through each slot its arguments
/// bind, both from the same specialized entry and what its code read before specialization
/// (`before`), as [`stage_reflection`] and [`stage_slot_writes`] answer them from a function.
///
/// The reflection is `None` where a binding's answer is unmeasured (`Feature::Reflection`). The
/// slot writes are always kept: they read the entry's roles, declared accesses and slots alone.
fn kept_reflection(
    spec: &air::Entry,
    before: &reflection::UnspecializedReads,
) -> Result<KeptAnswers, Error> {
    let read = match reflection::specialized_reads(spec, before) {
        Ok(read) => read,
        // Refused over control-point inputs a function constant gates. Those bind no
        // argument-table slot, so the entry's own parameters still state every slot's access;
        // what the specialized read would have added is which arguments the constants leave
        // unread.
        Err(Error::Unsupported(u)) if u.feature == Feature::Reflection => {
            return Ok(KeptAnswers {
                reflection: None,
                slot_writes: reflection::entry_slot_writes(&spec.ep),
            })
        }
        Err(error) => return Err(error),
    };
    let slot_writes = reflection::entry_slot_writes(&read);
    let answer = (|| {
        Ok(reflection::StageReflection {
            bindings: reflection::bindings(
                &read,
                &spec.structs,
                reflection::Channel::Pipeline,
                &spec.function_tables,
            )?,
            attributes: reflection::read_attributes(&read)?,
        })
    })();
    match answer {
        Ok(stage) => Ok(KeptAnswers {
            reflection: Some(stage),
            slot_writes,
        }),
        Err(Error::Unsupported(u)) if u.feature == Feature::Reflection => Ok(KeptAnswers {
            reflection: None,
            slot_writes,
        }),
        Err(error) => Err(error),
    }
}

/// What [`kept_reflection`] keeps of one compiled entry for its artifact.
struct KeptAnswers {
    reflection: Option<reflection::StageReflection>,
    slot_writes: Vec<reflection::SlotWrite>,
}

fn census(plan: &Plan) -> StrategyCensus {
    let mut c = StrategyCensus::default();
    for (name, (uses, words)) in plan.census() {
        c.entries.insert(name, StrategyUse { uses, words });
    }
    c
}

/// [`census`] of the emitted module: the plan's, and the selections whose helper invocations
/// take another arm (`spirv::helper_branches`), each with its helper query and one or two
/// logical operations (about 12 words).
fn emitted_census(
    plan: &Plan,
    helper_guard: crate::spirv::helper_branches::Guarded,
) -> StrategyCensus {
    let mut c = census(plan);
    if helper_guard.selections > 0 {
        c.entries.insert(
            crate::spirv::helper_branches::STRATEGY.to_owned(),
            StrategyUse {
                uses: helper_guard.selections,
                words: u64::from(helper_guard.selections) * 12,
            },
        );
    }
    c
}

/// The stage interface the artifact carries: the slots, the push placement, the module's link
/// contract (every descriptor binding and specialization constant it declares), and whether it
/// writes memory. A buffer slot's access is the function's, with what the accesses through loaded
/// or converted pointers may do to a slot whose address escapes (`memory::plan::SlotPlan::wide`),
/// and a read where emission merged a store's partial word into the slot (`merged`): the sub-word
/// store's atomics read the word they modify. A buffer slot at an index of `function_tables` holds
/// a visible-function table read as its words. `reaches_command_buffers` is the canonical entry's
/// `Entry::may_encode_indirect_commands`, `command_writes` its `Entry::command_writes`,
/// `writes_textures` its `Entry::may_write_textures`, and
/// `numbered_samplers` the samplers a pipeline state numbers for it (`numbered::samplers`).
fn stage_interface(
    plan: &Plan,
    symbols: crate::pipeline::link::contract::StageSymbols,
    writes_memory: bool,
    merged: &std::collections::BTreeSet<crate::pipeline::sym::SlotRef>,
    function_tables: &std::collections::BTreeSet<u32>,
    (reaches_command_buffers, command_writes): (bool, crate::pipeline::interface::CommandWrites),
    (writes_textures, numbered_samplers): (bool, Vec<crate::pipeline::interface::NumberedSampler>),
    (own_pixel_textures, snapshot_textures): (Vec<u32>, Vec<u32>),
) -> StageInterface {
    use crate::pipeline::interface::{BufferKind, BufferSlot};
    use crate::pipeline::sym::{PushField, SlotClass};
    let mut slots = std::collections::BTreeMap::new();
    for c in plan.slots.iter().flatten() {
        let mut access = c.access;
        access.read |= merged.contains(&c.slot);
        let table = c.slot.class == SlotClass::Buffer && function_tables.contains(&c.slot.index);
        slots.insert(
            c.slot,
            SlotInterface::Buffer(BufferSlot {
                access,
                window: c.window,
                form: c.form,
                kind: if table {
                    BufferKind::VisibleFunctionTable
                } else {
                    BufferKind::Buffer
                },
            }),
        );
    }
    for s in plan.families.texture.slots.values() {
        if let Some(t) = s.interface() {
            slots.insert(s.slot, SlotInterface::Texture(t));
        }
    }
    for s in plan.families.sampler.slots.values() {
        slots.insert(s.slot, SlotInterface::Sampler(s.interface()));
    }
    // Pulled vertex attributes' source buffers with slots of their own; a shared one is its
    // argument's slot above.
    slots.extend(crate::lower::access::pulled::slot_interfaces(
        &plan.families.access,
    ));
    // Intersection function tables, read-only word slots of the ray tracing family.
    slots.extend(crate::lower::raytracing::slot_interfaces(plan));
    StageInterface {
        stage: plan.stage,
        slots,
        threadgroup: Vec::new(),
        push: plan.push.clone(),
        descriptors: symbols.descriptors,
        spec: symbols.spec,
        // Every reader of the root table reads its address from this push field.
        needs_pointer_translation: plan
            .push
            .fields
            .iter()
            .any(|f| f.field == PushField::RootTableAddress),
        // A texture handle loaded from an argument buffer reads the handle table.
        needs_handle_translation: plan.families.access.handle_table.is_some(),
        io: plan.families.stage_io.interface.clone(),
        tessellation: crate::lower::tessellation::interface(plan),
        writes_memory,
        mesh_records: crate::lower::mesh::records(&plan.families.mesh),
        emulated_payload: crate::lower::mesh::emulated_payload(&plan.families.mesh),
        payload: plan.families.mesh.payload,
        reaches_command_buffers,
        command_writes,
        writes_textures,
        numbered_samplers,
        own_pixel_textures,
        snapshot_textures,
        private_bytes: plan.private_bytes,
    }
}
