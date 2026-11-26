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
//! key derivation is [`key`].

pub mod api;
mod entry;
mod images;
pub mod key;
mod numbered;
pub(crate) mod partition;
mod stage_bound;
#[cfg(test)]
mod tests;
mod unbound;
pub(crate) mod weights;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use self::partition::Partition;
use crate::air::abi::{AbiInt, AirAbi, OutputRole, ParamRole, Stage as AbiStage};
use crate::air::container::{self, LibraryFunction};
use crate::air::{self, ConstValue, LinkedCode, LinkedModule, LinkedName, SpecRequest};
use crate::bind::argument_buffer::{self, EntryLayout};
use crate::bind::{ArgumentBufferLayout, ArgumentLayoutError};
use crate::budget::{Meter, Spent, SPIRV_WORDS};
use crate::cache::Hash256;
use crate::cache::VariantKey;
use crate::device::DeviceProfile;
use crate::error::{
    DescriptorError, DiagName, Error, Feature, InternalError, InternalKind, MetalAnswer,
    Unsupported,
};
use crate::library::function::FunctionSource;
use crate::library::{
    CompiledFunction, DynamicLibrary, Function, FunctionArtifact, FunctionCompileKey,
    FunctionConstantValues, KernelFacts,
};
use crate::lower::dispatch::LocalSizeForm;
use crate::lower::select::{LowerInputs, Plan};
use crate::lower::simd::lockstep::{self, OrderKey};
use crate::pipeline::artifact::{PrecisionNote, Reconvergence, StrategyCensus, StrategyUse};
use crate::pipeline::compile::key::{DynamicKey, LinkedKey};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, DynamicLibraries, FunctionDescriptor, LinkedFunction,
    LinkedFunctions, LinkedList, MeshPipelineDescriptor, RenderPipelineDescriptor,
    VertexDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{SlotInterface, StageInterface};
use crate::pipeline::sym::{SlotClass, SlotRef, Stage, VaryingKey};
use crate::reflection;
use crate::spirv::ir::TypedModule;

pub use entry::{EntryDescriptor, EntryPlan, StandaloneModule};
pub(crate) use stage_bound::StageBounds;
pub use stage_bound::COMPILED_STAGE_WEIGHT;

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
    /// this vertex descriptor over the attributes the specialized entry reads
    /// (`link::vertex::vertex_fetch_projection_for`), the key's vertex-stage inputs (`VariantKey::lowering_inputs`), the pipeline's
    /// `maxVertexCallStackDepth` (the `u32`), whether the pipeline's fragment function reads
    /// `[[primitive_id]]` where the device carries it through a varying
    /// ([`reads_primitive_id_varying`]). A vertex function writes the primitive ID's varying where
    /// the device cannot declare a fragment `PrimitiveId` (`LoweringInputs::primitive_id_varying`),
    /// and whether the pipeline state is made for indirect command buffers
    /// (`supportIndirectCommandBuffers`): a vertex function then pulls every attribute it reads
    /// and tests its command against the execution range (`LoweringInputs::indirect_range`).
    Vertex(
        Option<&'a VertexDescriptor>,
        &'a VariantKey,
        u32,
        bool,
        bool,
    ),
    /// The fragment stage of this render pipeline under a variant key: the key's fragment-stage
    /// inputs, the attachments among the pipeline's color attachments that the specialized entry
    /// writes and whose pixel format is a signed integer
    /// (`LoweringInputs::signed_color_attachments`), whether the pipeline rasterizes more than one
    /// sample (`LoweringInputs::multisampled`), and, for each index the pipeline's vertex function
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
/// pipeline's, or its mesh render pipeline's, which has no call stack depth of its own (Metal's
/// default, 1).
#[derive(Clone, Copy, Debug)]
pub struct FragmentPass<'a> {
    pub(crate) color_attachments: &'a [Option<ColorAttachmentDescriptor>; 8],
    pub(crate) raster_sample_count: u32,
    pub(crate) rasterization_enabled: bool,
    pub(crate) alpha_to_coverage: bool,
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
    /// Where the function's push fields start on the floor (`LoweringInputs::push_base`): 0 from
    /// a render descriptor, set by the compile layer from the vertex function it compiled; past
    /// the object and mesh functions' windows for a mesh pipeline's.
    pub(crate) push_base: u32,
}

impl<'a> From<&'a RenderPipelineDescriptor> for FragmentPass<'a> {
    fn from(d: &'a RenderPipelineDescriptor) -> FragmentPass<'a> {
        FragmentPass {
            color_attachments: &d.color_attachments,
            raster_sample_count: d.raster_sample_count,
            rasterization_enabled: d.rasterization_enabled,
            alpha_to_coverage: d.alpha_to_coverage,
            max_call_stack_depth: d.max_fragment_call_stack_depth,
            texture_write_rounding_mode: d.texture_write_rounding_mode,
            sample_mask: None,
            // Refused at the pipeline's creation where the test is not built.
            alpha_test: crate::pipeline::private_state::alpha_test(&d.private)
                .ok()
                .flatten(),
            depth_compare_clamp_mask: d.private.fragment_depth_compare_clamp_mask,
            push_base: 0,
        }
    }
}

impl<'a> From<&'a MeshPipelineDescriptor> for FragmentPass<'a> {
    fn from(d: &'a MeshPipelineDescriptor) -> FragmentPass<'a> {
        FragmentPass {
            color_attachments: &d.color_attachments,
            raster_sample_count: d.raster_sample_count,
            rasterization_enabled: d.rasterization_enabled,
            alpha_to_coverage: d.alpha_to_coverage,
            max_call_stack_depth: 1,
            texture_write_rounding_mode: d.texture_write_rounding_mode,
            sample_mask: None,
            alpha_test: None,
            depth_compare_clamp_mask: 0,
            push_base: crate::lower::push::MESH_END,
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
            InputsFrom::Vertex(_, _, depth, ..) => (*depth).max(1),
            InputsFrom::Fragment(pass, ..) => pass.max_call_stack_depth.max(1),
        }
    }

    /// Whether the stage compiles as the kernel of a compute pipeline of OpenCL mode
    /// (`LoweringInputs::open_cl_mode`): the given inputs' say; a render pipeline's stage never
    /// does.
    fn open_cl_mode(&self) -> bool {
        match self {
            InputsFrom::Given(i) => i.open_cl_mode,
            InputsFrom::Vertex(..) | InputsFrom::Fragment(..) => false,
        }
    }

    /// The kernel's texture slots of OpenCL mode's 1D images bound to a 1D texture
    /// (`LoweringInputs::open_cl_lines`): the given inputs'; a render pipeline's stage has none.
    fn open_cl_lines(&self) -> &'a BTreeSet<SlotRef> {
        static NONE: BTreeSet<SlotRef> = BTreeSet::new();
        match self {
            InputsFrom::Given(i) => {
                let given: &'a LowerInputs = i;
                &given.open_cl_lines
            }
            InputsFrom::Vertex(..) | InputsFrom::Fragment(..) => &NONE,
        }
    }
}

/// Where `inputs` are a render pipeline's fragment stage's, each `[[color(n)]]` input the
/// specialized fragment entry `spec` declares has an attachment with a pixel format in the
/// pipeline, whether or not the body reads it: plain Metal refuses the pipeline otherwise, with the
/// read dead after specialization as with it live ("Shaders reads from a color attachment whose
/// pixel format is MTLPixelFormatInvalid",
/// `metal2vulkan-conformance/probes/dead_ordered_and_unattached_reads.m`, an M3 Max under macOS
/// 27.0.1). `InvalidDescriptor(ColorAttachment)` at the first index without one. An input its
/// constants gate is no parameter of the specialized entry. Checked by both a stage's keys
/// ([`stage_inputs`]) and its compile.
fn color_inputs_attached(spec: &air::Entry, inputs: InputsFrom<'_>) -> Result<(), Error> {
    let InputsFrom::Fragment(pass, ..) = inputs else {
        return Ok(());
    };
    for b in &spec.ep.params {
        let ParamRole::ColorInput {
            index: AbiInt::Const(color),
            ..
        } = b.role
        else {
            continue;
        };
        let attached = pass
            .color_attachments
            .get(color as usize)
            .is_some_and(Option::is_some);
        if !attached {
            return Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                index: color,
                metal: MetalAnswer::NilWithError,
            }));
        }
    }
    Ok(())
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

/// One linked function as a compile takes it: its key (the list it is in, its name, its
/// specialized name, its AIR hash, its constant values) and its bitcode module.
#[derive(Clone, Copy, Debug)]
pub struct LinkedSource<'a> {
    pub(crate) key: LinkedKey<'a>,
    pub(crate) bitcode: &'a [u8],
}

impl<'a> LinkedSource<'a> {
    /// The function `name` of `functions` (`MTLLinkedFunctions.functions`), defined in the module
    /// `bitcode` (hashed here), specialized with `constants`, linked under its own name; made
    /// specialized only where `constants` holds a value ([`LinkedFunction::specialized`]).
    pub fn new(
        name: &'a str,
        bitcode: &'a [u8],
        constants: &'a FunctionConstantValues,
    ) -> LinkedSource<'a> {
        LinkedSource {
            key: LinkedKey {
                list: LinkedList::Functions,
                name,
                specialized_name: None,
                specialized: !constants.is_empty(),
                air_hash: container::sha256(bitcode),
                constants,
            },
            bitcode,
        }
    }

    /// This function made under the specialized name `name` (`specializedName`), which it links
    /// under in place of its own ([`LinkedFunction::name`]): specialized too.
    pub fn with_specialized_name(mut self, name: &'a str) -> LinkedSource<'a> {
        self.key.specialized_name = Some(name).filter(|&n| n != self.key.name);
        self.key.specialized = true;
        self
    }

    /// This function, made specialized when `specialized` ([`LinkedFunction::specialized`]).
    pub fn with_specialized(mut self, specialized: bool) -> LinkedSource<'a> {
        self.key.specialized = specialized;
        self
    }
}

/// The key of `f`, listed in `list`.
pub(crate) fn linked_key<'a>(list: LinkedList<'a>, f: &'a LinkedFunction) -> LinkedKey<'a> {
    let info = f.function.info();
    LinkedKey {
        list,
        name: info.name,
        specialized_name: f.other_name(),
        specialized: f.specialized,
        air_hash: info.air_hash,
        constants: &f.constants,
    }
}

/// The key of each function of `linked`, in field order.
pub(crate) fn linked_keys(linked: &LinkedFunctions) -> Vec<LinkedKey<'_>> {
    linked
        .listed()
        .map(|(list, f)| linked_key(list, f))
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
        with_request_shared(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            inputs,
            Shared::Entry(entries, fd),
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
        let open_cl = (inputs.open_cl_mode(), inputs.open_cl_lines());
        let specialized = entries.specialized(fd, inputs.call_stack_depth(), open_cl, m)?;
        color_inputs_attached(&specialized.spec, inputs)?;
        let spec = &specialized.spec;
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

/// Where the fragment function's push fields start behind vertex stage `fd` compiled under
/// `inputs` on the floor ([`CompiledFunction::fragment_push_base`] of that compile): the end of the
/// push range its plan places, from a selection alone. The entry is decoded and specialized as
/// [`stage_inputs`] does (the one `entries` holds, or one made there now), analyzed and selected;
/// nothing is emitted. A vertex stage is selected once ([`stage_bound`]'s second selection is a
/// kernel's alone), so the plan is the compile's.
pub(crate) fn vertex_push_end(
    fd: &FunctionDescriptor,
    p: &DeviceProfile,
    inputs: InputsFrom<'_>,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<u32, Error> {
    guarded(|| {
        with_request_shared(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            inputs,
            Shared::Entry(entries, fd),
            |r| {
                m.weighted(weights::REST, |m| {
                    let analyzed = analyze_one(r, p, m)?;
                    let widest = crate::lower::simd::WidestPin::Weighed;
                    let (plan, _) = analyzed.select(p, widest, m)?;
                    let tessellated = crate::lower::tessellation::interface(&plan).is_some();
                    Ok(crate::lower::push::fragment_base(&plan.push, tessellated))
                })
            },
        )
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

/// Entry point `ep` of `module` as its pipeline binds it: where `open_cl_mode`, a kernel as a
/// pipeline of OpenCL mode binds it (`air::canon::opencl::entry`), its hidden buffers the
/// parameters past the function's own, as specialization appends them; else `ep` itself.
fn bound_entry<'a>(
    module: &air::ir::Module,
    ep: &'a air::abi::EntryPoint,
    open_cl_mode: bool,
) -> Result<std::borrow::Cow<'a, air::abi::EntryPoint>, Error> {
    if !open_cl_mode || !matches!(ep.stage, AbiStage::Kernel { .. }) {
        return Ok(std::borrow::Cow::Borrowed(ep));
    }
    let params = module.func(ep.func).map(|f| f.params.len());
    let first = params.and_then(|n| u32::try_from(n).ok()).ok_or_else(|| {
        Error::Internal(InternalError::new(
            InternalKind::Invariant,
            "the entry's function is not in its module",
        ))
    })?;
    // The reflection states each 1D image as the 2D texture Apple's OpenCL binds for it, whatever
    // a variant binds.
    let (ep, _) = air::canon::opencl::entry(ep.clone(), first, &BTreeSet::new())?;
    Ok(std::borrow::Cow::Owned(ep))
}

/// What a pipeline's reflection states for its stage `fd`, its calls nesting `depth` deep through
/// visible functions (a kernel of OpenCL mode where `open_cl_mode`): the entry's bindings and the attributes it reads, from the entry its
/// constants and linked functions specialize where they decide a binding
/// (`reflection::needs_resolution`) or which arguments it reads
/// (`reflection::specialization_decides_reads`), else from the decoded entry.
pub(crate) fn stage_reflection(
    fd: &FunctionDescriptor,
    depth: u32,
    open_cl_mode: bool,
    m: &mut Meter,
) -> Result<reflection::StageReflection, Error> {
    let mut inputs = LowerInputs::new();
    inputs.max_call_stack_depth = call_stack_input(depth);
    inputs.open_cl_mode = open_cl_mode;
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
                req.open_cl_mode = r.inputs.open_cl_mode();
                let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
                if !reflection::needs_resolution(ep, &decoded.abi.structs)
                    && !reflection::specialization_decides_reads(&decoded.abi)
                {
                    let ep = bound_entry(&decoded.module, ep, req.open_cl_mode)?;
                    return Ok(reflection::StageReflection {
                        bindings: reflection::bindings(
                            &ep,
                            &decoded.abi.structs,
                            reflection::Channel::Pipeline,
                            &BTreeSet::new(),
                        )?,
                        attributes: reflection::read_attributes(&ep)?,
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

/// What stage `fd`, its calls nesting `depth` deep through visible functions (a kernel of OpenCL
/// mode where `open_cl_mode`), may store through each slot its arguments bind: of the entry [`stage_reflection`] reads, its parameters' roles,
/// declared accesses and slots alone (`reflection::entry_slot_writes`). It derives no type
/// description, so an argument whose reflection Metal's answer has not been measured for does not
/// refuse it: only the decode and the specialization can.
pub(crate) fn stage_slot_writes(
    fd: &FunctionDescriptor,
    depth: u32,
    open_cl_mode: bool,
    m: &mut Meter,
) -> Result<Vec<reflection::SlotWrite>, Error> {
    let mut inputs = LowerInputs::new();
    inputs.max_call_stack_depth = call_stack_input(depth);
    inputs.open_cl_mode = open_cl_mode;
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
                req.open_cl_mode = r.inputs.open_cl_mode();
                let ep = air::canon::select(&decoded.module, &decoded.abi, &req.entry)?;
                if !reflection::needs_resolution(ep, &decoded.abi.structs)
                    && !reflection::specialization_decides_reads(&decoded.abi)
                {
                    let ep = bound_entry(&decoded.module, ep, req.open_cl_mode)?;
                    return Ok(reflection::entry_slot_writes(&ep));
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

/// The specialized entries of pipeline states' stages: each stage's function, with its calls
/// nesting a given depth, is decoded and specialized at most once while the object holds its
/// entry, and every check, key and compile that is given the object reads that one entry.
/// Specializing a large function takes far longer than decoding it (16 ms of an 18 ms decode
/// and specialization, measured), and a render state's creation asks for each stage's entry up to
/// three times before its compile: its descriptor's linked-use checks
/// (`RenderPipelineDescriptor::unwritten_fragment_inputs` and
/// `fragment_writes_depth_unattached`) and its key (`Device::render_variant_key`).
///
/// **One object serves many creations.** Nothing in an entry depends on the pipeline state it was
/// asked for: it is the function, its constants, its linked functions and its dynamic libraries
/// alone (the device's profile does not reach it either, which is why a consumer may share one
/// object across its devices). So a consumer that keeps one object and gives it to every creation
/// specializes each stage once however many states it makes of it -- a sweep of pipeline states
/// over one vertex-and-fragment pair decodes and specializes its two functions once rather than
/// twice a state. Entries are told apart by their function descriptor and depth, so one object
/// given to steps of different descriptors stays exact.
///
/// **The plain answers too.** A creation also reads answers of a stage's AIR that need no
/// specialization and so are not worth keeping a specialized entry for: whether its entry returns
/// a value, which Metal's creation refuses with rasterization disabled
/// ([`RenderPipelineDescriptor::vertex_returns_value`]). Each is a function of the stage's
/// descriptor alone, so the object keeps it beside the entries and a sweep of states decodes for
/// it once.
///
/// **The module decodes too.** Decoding a module reads its bitcode and nothing else: which entry
/// of it is being compiled, and with which constants and linked functions, reaches no part of the
/// decode. So the object keeps the decode of each module it reads, by the module's AIR hash (the
/// function's `FunctionInfo::air_hash`, which its library states without decoding), and every
/// later ask over the same bytes copies that decode instead of reading the bitstream, the bitcode
/// records and the AIR metadata ABI again. A library of six entries decodes once rather than six
/// times, and the same function asked for at two call-stack depths, with two sets of constants,
/// or for a plain answer and then an entry, decodes once rather than twice. The same work happens
/// at the same time for the same inputs: the whole module is decoded, with every entry's argument
/// metadata, by the first ask that needs it.
///
/// **Charged as made.** An ask that finds what an earlier ask made is charged what making it
/// charged (`Spent`: its work units, its ledger peak and the ledger bytes it holds), though
/// nothing is made again. So the object spares time and never budget: a compile's answer under
/// its budget, a refusal at a cap included, is the same whether its entry, its decode or its
/// answer was made or found, and the same as a compile's that shares nothing
/// (`Device::compile_function`).
///
/// **Bounded.** It keeps at most 32 specialized entries and at most 64 MiB of them (each entry
/// counted as the ledger bytes specializing it held), at most 32 plain answers, and at most 4
/// module decodes and 32 MiB of them (each counted as the ledger bytes decoding it held); past
/// any of those, the least recently used go, so an object that outlives one creation never grows
/// without bound. A refusal is not kept: the next ask makes it again.
#[derive(Default)]
pub struct StageEntries {
    kept: std::sync::Mutex<Kept>,
}

/// The most entries one [`StageEntries`] keeps, whatever their sizes. One creation asks for at
/// most three (its stages), so this is room for the stages of many states' worth of functions,
/// while the search stays a scan of a handful.
pub(crate) const KEPT_ENTRIES: usize = 32;

/// The most bytes of specialized entries one [`StageEntries`] keeps: 64 MiB, each entry counted
/// as the ledger bytes specializing it held. One translation attempt may use 500 MiB, so
/// an object shared between creations holds a fraction of one attempt's room.
pub(crate) const KEPT_ENTRY_BYTES: u64 = 64 << 20;

/// The most module decodes one [`StageEntries`] keeps, whatever their sizes: a render creation's
/// three stages and one more, which is the window a sweep of states, or a library's entries
/// compiled in turn, asks over.
const KEPT_DECODES: usize = 4;

/// The most bytes of module decodes one [`StageEntries`] keeps: 32 MiB, each decode counted as
/// the ledger bytes decoding it held. Half the entries' room, and a sixteenth of one
/// translation attempt's 500 MiB.
pub(crate) const KEPT_DECODE_BYTES: u64 = 32 << 20;

/// What a [`StageEntries`] holds: its specialized entries, its plain answers and its module
/// decodes, each least recently used first, the bytes the entries and the decodes hold, and the
/// totals its counts report.
#[derive(Default)]
struct Kept {
    entries: Vec<StageEntry>,
    /// The answers read of a stage's AIR without specializing it.
    answers: Vec<StageAnswer>,
    /// The module decodes every ask over the same bitcode copies instead of reading again.
    decodes: Vec<KeptDecode>,
    bytes: u64,
    made: u64,
    found: u64,
    /// The decodes read of a module's bitcode, and the asks that copied one already read.
    decodes_read: u64,
    decodes_copied: u64,
    decode_bytes: u64,
    /// The entries ordered ([`StageEntries::ordered`]), and the asks that found one ordered
    /// before.
    orders_made: u64,
    orders_found: u64,
}

/// One module decode of [`StageEntries`], with what tells it apart: the bitcode's own AIR hash,
/// which decides the decode completely.
struct KeptDecode {
    /// SHA-256 of the module's bitcode: the function's AIR hash, which its library states without
    /// decoding anything (a metallib's verified `HASH`, or a module's own hash, which every entry
    /// of it carries).
    air_hash: Hash256,
    /// What decoding it charged, which every ask that copies it is charged; the byte bound counts
    /// the bytes it holds.
    spent: Spent,
    decoded: Arc<air::Decoded>,
}

/// One plain answer of [`StageEntries`], read of a stage's AIR without specializing it, with what
/// tells it apart: the stage's descriptor alone decides it.
struct StageAnswer {
    /// As a [`StageEntry`]'s: compared before the descriptor.
    air_hash: Hash256,
    function: FunctionDescriptor,
    /// Whether the entry returns a value ([`stage_returns_value`]).
    returns_value: bool,
    /// What reading it charged, which every ask that finds it is charged.
    spent: Spent,
}

/// One specialized entry of [`StageEntries`], with what tells it apart.
struct StageEntry {
    /// The function's AIR hash, which tells entries apart without comparing their descriptors:
    /// the library's own table states it, so a scan reads it without decoding anything.
    air_hash: Hash256,
    function: FunctionDescriptor,
    depth: u32,
    /// Specialized as the kernel of a compute pipeline of OpenCL mode.
    open_cl_mode: bool,
    /// With it, the slots of its 1D images bound to a 1D texture (`LoweringInputs::open_cl_lines`).
    open_cl_lines: BTreeSet<SlotRef>,
    /// What specializing it charged, which every ask that finds it is charged; the byte bound
    /// counts the bytes it holds.
    spent: Spent,
    specialized: Arc<Specialized>,
    /// The entry ordered for each key asked of it, least recently used first, at most
    /// [`KEPT_ORDERS`]; the byte bound counts the bytes each holds.
    ordered: Vec<KeptOrder>,
}

/// One ordered entry of a [`StageEntry`]: the specialized entry ordered for `key`
/// ([`lockstep::order`]), with what ordering it charged, which every ask that finds it is charged
/// beside what specializing it charged.
struct KeptOrder {
    key: OrderKey,
    spent: Spent,
    ordered: Arc<Specialized>,
}

/// The most ordered entries one [`StageEntry`] keeps. A device's profile fixes the whole key, so a
/// creation asks for one.
const KEPT_ORDERS: usize = 1;

impl std::fmt::Debug for StageEntries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let made = self.kept.lock().map_or(0, |kept| kept.entries.len());
        f.debug_struct("StageEntries").field("made", &made).finish()
    }
}

impl StageEntries {
    /// No entry made yet.
    pub fn new() -> StageEntries {
        StageEntries::default()
    }

    /// How many times it has read a stage's AIR -- to specialize an entry, or alone for a
    /// plain answer -- and how many asks found what an earlier one read, over its whole life
    /// (neither ever falls). A consumer that gives one object to every pipeline creation reads
    /// the entries it spared in the second.
    pub fn counts(&self) -> (u64, u64) {
        let kept = self.kept.lock();
        kept.as_ref().map_or((0, 0), |k| (k.made, k.found))
    }

    /// What it holds now: its specialized entries, its module decodes, and the bytes both hold
    /// (each counted as the ledger bytes making it held, as its bounds count them). Each falls as
    /// the least recently used go.
    pub fn held(&self) -> (u64, u64, u64) {
        let kept = self.kept.lock();
        kept.as_ref().map_or((0, 0, 0), |k| {
            (
                k.entries.len() as u64,
                k.decodes.len() as u64,
                k.bytes + k.decode_bytes,
            )
        })
    }

    /// How many module decodes it has read of bitcode, and how many asks copied a decode an
    /// earlier ask had read, over its whole life (neither ever falls). Every entry it specializes
    /// and every plain answer it reads needs the decode of its module, so the entries of one
    /// library, and the descriptors and depths of one function, read one decode between them.
    #[cfg(test)]
    pub(crate) fn decodes(&self) -> (u64, u64) {
        let kept = self.kept.lock();
        kept.as_ref()
            .map_or((0, 0), |k| (k.decodes_read, k.decodes_copied))
    }

    /// The decode of the module whose AIR hash is `air_hash`, whose bitcode is `bitcode`: a copy
    /// of the one an earlier ask read, charged to `m` what reading it charged, else one read now
    /// under `m` and kept. A refusal is not kept: the next ask reads it again.
    fn module(
        &self,
        air_hash: &Hash256,
        bitcode: &[u8],
        m: &mut Meter,
    ) -> Result<air::Decoded, Error> {
        let lock = || self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let found = lock().decode(air_hash);
        if let Some((decoded, spent)) = found {
            m.spend(spent)?;
            return Ok(air::Decoded::clone(&decoded));
        }
        let (decoded, spent) = m.measured(|m| air::decode(bitcode, m))?;
        let decoded = Arc::new(decoded);
        let mine = air::Decoded::clone(&decoded);
        let mut kept = lock();
        // Another caller may have decoded the same module meanwhile: the bytes decide the decode,
        // so its decode is this one and the two share what is kept.
        if let Some((theirs, _)) = kept.decode(air_hash) {
            return Ok(air::Decoded::clone(&theirs));
        }
        kept.decodes_read += 1;
        kept.decode_bytes = kept.decode_bytes.saturating_add(spent.held());
        kept.decodes.push(KeptDecode {
            air_hash: *air_hash,
            spent,
            decoded,
        });
        kept.evict();
        Ok(mine)
    }

    /// Whether the entry of stage `fd` returns a value: the answer an earlier ask read, charged to
    /// `m` what reading it charged, else read now under `m` by `read`, which decodes the stage's
    /// AIR and specializes nothing ([`stage_returns_value`]). The stage's descriptor alone decides
    /// it. A refusal is not kept.
    fn returns_value(
        &self,
        fd: &FunctionDescriptor,
        m: &mut Meter,
        read: impl FnOnce(&StageEntries, &mut Meter) -> Result<bool, Error>,
    ) -> Result<bool, Error> {
        // The library's table states it without decoding (`Function::info`).
        let air_hash = fd.function.info().air_hash;
        let lock = || self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let found = lock().answer(&air_hash, fd);
        if let Some((returns, spent)) = found {
            m.spend(spent)?;
            return Ok(returns);
        }
        let (returns_value, spent) = m.measured(|m| read(self, m))?;
        let mut kept = lock();
        // As a specialized entry's: another caller may have read it meanwhile, and its answer is
        // this one.
        if let Some((theirs, _)) = kept.answer(&air_hash, fd) {
            return Ok(theirs);
        }
        kept.made += 1;
        kept.answers.push(StageAnswer {
            air_hash,
            function: fd.clone(),
            returns_value,
            spent,
        });
        kept.evict();
        Ok(returns_value)
    }

    /// The entry of stage `fd`, its calls nesting `depth` deep through visible functions (0 is
    /// taken as 1, Metal's default), as the kernel of a pipeline of OpenCL mode where `open_cl`
    /// says so, its 1D images bound to a 1D texture at the slots it names, specialized as a
    /// compile specializes it before lowering
    /// (its constants bound, its linked functions resolved, its calls inlined, SCCP run), with
    /// what a compile takes beside it (The AIR hash, the entry name, what its code read before, so
    /// a compile of the stage reads it too, [`specialize_one`]): the one made before, charged to
    /// `m` what making it charged, else made now under `m`. A refusal is not kept: the next ask
    /// makes it again.
    fn specialized(
        &self,
        fd: &FunctionDescriptor,
        depth: u32,
        (open_cl_mode, open_cl_lines): (bool, &BTreeSet<SlotRef>),
        m: &mut Meter,
    ) -> Result<Arc<Specialized>, Error> {
        let depth = depth.max(1);
        // The library's table states it without decoding (`Function::info`).
        let air_hash = fd.function.info().air_hash;
        let lock = || self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let open_cl = (open_cl_mode, open_cl_lines);
        let found = lock().take(&air_hash, fd, depth, open_cl);
        if let Some((specialized, spent)) = found {
            m.spend(spent)?;
            return Ok(specialized);
        }
        let mut inputs = LowerInputs::new();
        inputs.max_call_stack_depth = call_stack_input(depth);
        inputs.open_cl_mode = open_cl_mode;
        inputs.open_cl_lines = open_cl_lines.clone();
        let (specialized, spent) = m.measured(|m| {
            with_request_shared(
                &fd.function,
                &fd.constants,
                (&fd.linked, &fd.libraries),
                InputsFrom::Given(&inputs),
                Shared::Decode(self),
                |r| specialize_one(r, m),
            )
        })?;
        let specialized = Arc::new(specialized);
        let mut kept = lock();
        // Another caller may have specialized the same stage meanwhile: its entry is this one, so
        // the two share one.
        if let Some((theirs, _)) = kept.take(&air_hash, fd, depth, open_cl) {
            return Ok(theirs);
        }
        kept.made += 1;
        kept.bytes = kept.bytes.saturating_add(spent.held());
        kept.entries.push(StageEntry {
            air_hash,
            function: fd.clone(),
            depth,
            open_cl_mode,
            open_cl_lines: inputs.open_cl_lines,
            spent,
            specialized: Arc::clone(&specialized),
            ordered: Vec::new(),
        });
        kept.evict();
        Ok(specialized)
    }

    /// The entry of stage `fd` as [`StageEntries::specialized`] gives it, ordered for `key`
    /// ([`lockstep::order`], under [`weights::LOCKSTEP`]): the one ordered before, charged to `m`
    /// what specializing and ordering it charged, else ordered now under `m` and kept beside the
    /// specialized entry. A creation's check of the partition its kernel observes
    /// ([`kernel_observes_partition`]) and its compile ask for the same key, so the compile orders
    /// nothing again. A refusal is not kept: the next ask orders it again.
    fn ordered(
        &self,
        fd: &FunctionDescriptor,
        depth: u32,
        open_cl: (bool, &BTreeSet<SlotRef>),
        key: OrderKey,
        m: &mut Meter,
    ) -> Result<Arc<Specialized>, Error> {
        let depth = depth.max(1);
        // The library's table states it without decoding (`Function::info`).
        let air_hash = fd.function.info().air_hash;
        let lock = || self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let found = lock().take_ordered(&air_hash, fd, depth, open_cl, key);
        if let Some((ordered, specializing, ordering)) = found {
            m.spend(specializing)?;
            m.spend(ordering)?;
            return Ok(ordered);
        }
        let specialized = self.specialized(fd, depth, open_cl, m)?;
        let (ordered, spent) =
            m.measured(|m| order_specialized(Specialized::clone(&specialized), key, m))?;
        let ordered = Arc::new(ordered);
        let mut kept = lock();
        // Another caller may have ordered the same entry meanwhile: its order is this one, so the
        // two share one.
        if let Some((theirs, ..)) = kept.take_ordered(&air_hash, fd, depth, open_cl, key) {
            return Ok(theirs);
        }
        kept.keep_ordered(
            &air_hash,
            fd,
            depth,
            open_cl,
            (key, spent, Arc::clone(&ordered)),
        );
        kept.evict();
        Ok(ordered)
    }

    /// How many entries it has ordered, and how many asks found one ordered before, over its
    /// whole life (neither ever falls).
    #[cfg(test)]
    pub(crate) fn orders(&self) -> (u64, u64) {
        let kept = self.kept.lock();
        kept.as_ref()
            .map_or((0, 0), |k| (k.orders_made, k.orders_found))
    }
}

impl Kept {
    /// The entry of `fd` at `depth` whose function's AIR hash is `air_hash`, made the most
    /// recently used, with what making it charged: `None` where none is held. The hash is
    /// compared first, so a scan reads a function descriptor only where the hashes agree.
    fn take(
        &mut self,
        air_hash: &Hash256,
        fd: &FunctionDescriptor,
        depth: u32,
        open_cl: (bool, &BTreeSet<SlotRef>),
    ) -> Option<(Arc<Specialized>, Spent)> {
        let entry = self.used(air_hash, fd, depth, open_cl)?;
        let found = (Arc::clone(&entry.specialized), entry.spent);
        self.found += 1;
        Some(found)
    }

    /// The entry of `fd` at `depth` whose function's AIR hash is `air_hash`, made the most
    /// recently used: `None` where none is held. The hash is compared first, so a scan reads a
    /// function descriptor only where the hashes agree.
    fn used(
        &mut self,
        air_hash: &Hash256,
        fd: &FunctionDescriptor,
        depth: u32,
        (open_cl_mode, open_cl_lines): (bool, &BTreeSet<SlotRef>),
    ) -> Option<&mut StageEntry> {
        let at = self.entries.iter().position(|e| {
            e.depth == depth
                && e.open_cl_mode == open_cl_mode
                && e.open_cl_lines == *open_cl_lines
                && e.air_hash == *air_hash
                && e.function == *fd
        })?;
        let entry = self.entries.remove(at);
        self.entries.push(entry);
        self.entries.last_mut()
    }

    /// The entry of `fd` at `depth` ordered for `key`, both made the most recently used, with
    /// what specializing it and what ordering it charged, counted found as an entry and as an
    /// order: `None` where either is not held.
    fn take_ordered(
        &mut self,
        air_hash: &Hash256,
        fd: &FunctionDescriptor,
        depth: u32,
        open_cl: (bool, &BTreeSet<SlotRef>),
        key: OrderKey,
    ) -> Option<(Arc<Specialized>, Spent, Spent)> {
        let entry = self.used(air_hash, fd, depth, open_cl)?;
        let at = entry.ordered.iter().position(|o| o.key == key)?;
        let order = entry.ordered.remove(at);
        let found = (Arc::clone(&order.ordered), entry.spent, order.spent);
        entry.ordered.push(order);
        // The ask found the specialized entry too, as `take` counts it.
        self.found += 1;
        self.orders_found += 1;
        Some(found)
    }

    /// Keeps `ordered`, the entry of `fd` at `depth` ordered for its key with what ordering it
    /// charged, beside that entry, dropping the entry's least recently used order past
    /// [`KEPT_ORDERS`]. Nothing where the entry itself is no longer held.
    fn keep_ordered(
        &mut self,
        air_hash: &Hash256,
        fd: &FunctionDescriptor,
        depth: u32,
        open_cl: (bool, &BTreeSet<SlotRef>),
        (key, spent, ordered): (OrderKey, Spent, Arc<Specialized>),
    ) {
        let Some(entry) = self.used(air_hash, fd, depth, open_cl) else {
            return;
        };
        let mut bytes = spent.held();
        entry.ordered.push(KeptOrder {
            key,
            spent,
            ordered,
        });
        let mut freed = 0;
        while entry.ordered.len() > KEPT_ORDERS {
            freed += entry.ordered.remove(0).spent.held();
        }
        bytes = bytes.saturating_sub(freed);
        self.orders_made += 1;
        self.bytes = self.bytes.saturating_add(bytes);
    }

    /// The plain answer of `fd` whose function's AIR hash is `air_hash`, made the most recently
    /// used, with what reading it charged: `None` where none is held. The hash is compared first,
    /// as [`Kept::take`] compares it.
    fn answer(&mut self, air_hash: &Hash256, fd: &FunctionDescriptor) -> Option<(bool, Spent)> {
        let at = self
            .answers
            .iter()
            .position(|a| a.air_hash == *air_hash && a.function == *fd)?;
        let answer = self.answers.remove(at);
        let found = (answer.returns_value, answer.spent);
        self.answers.push(answer);
        self.found += 1;
        Some(found)
    }

    /// The decode of the module whose AIR hash is `air_hash`, made the most recently used, with
    /// what reading it charged: `None` where none is held. The hash decides the decode
    /// completely, so nothing else is compared.
    fn decode(&mut self, air_hash: &Hash256) -> Option<(Arc<air::Decoded>, Spent)> {
        let at = self.decodes.iter().position(|d| d.air_hash == *air_hash)?;
        let decode = self.decodes.remove(at);
        let found = (Arc::clone(&decode.decoded), decode.spent);
        self.decodes.push(decode);
        self.decodes_copied += 1;
        Some(found)
    }

    /// Drop the least recently used entries, answers and decodes until the bounds hold, never the
    /// last one of any: a bound below one would make the object useless to the creation that is
    /// asking now.
    fn evict(&mut self) {
        while self.entries.len() > 1
            && (self.entries.len() > KEPT_ENTRIES || self.bytes > KEPT_ENTRY_BYTES)
        {
            let gone = self.entries.remove(0);
            let ordered: u64 = gone.ordered.iter().map(|o| o.spent.held()).sum();
            self.bytes = self
                .bytes
                .saturating_sub(gone.spent.held().saturating_add(ordered));
        }
        while self.answers.len() > KEPT_ENTRIES.max(1) {
            self.answers.remove(0);
        }
        while self.decodes.len() > 1
            && (self.decodes.len() > KEPT_DECODES || self.decode_bytes > KEPT_DECODE_BYTES)
        {
            let gone = self.decodes.remove(0);
            self.decode_bytes = self.decode_bytes.saturating_sub(gone.spent.held());
        }
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
        // A render or mesh pipeline's stage, never a kernel of OpenCL mode.
        let specialized = entries.specialized(fd, depth, (false, &BTreeSet::new()), m)?;
        f(&specialized.spec, m)
    })
}

/// Whether the entry of stage `fd` returns a value: its function's return type is not `void`.
/// It decodes the AIR, and specializes nothing; `entries` keeps the answer, which the stage's
/// descriptor alone decides, so however many states are made of one function it decodes once.
pub(crate) fn stage_returns_value(
    fd: &FunctionDescriptor,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<bool, Error> {
    entries.returns_value(fd, m, |shared, m| stage_returns_value_read(fd, shared, m))
}

/// [`FunctionDescriptor::needs_specialization`] under `m`: a module function's constants read
/// through `entries`' module decodes, which a compile of the stage through the same object copies.
pub(crate) fn stage_needs_specialization(
    fd: &FunctionDescriptor,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<bool, Error> {
    if fd.specialized || !fd.constants.is_empty() {
        return Ok(false);
    }
    let air_hash = fd.function.info().air_hash;
    let constants = fd
        .function
        .constants_dictionary_with(|bitcode, m| entries.module(&air_hash, bitcode, m), m)?;
    Ok(!constants.is_empty())
}

/// [`stage_returns_value`]'s own read of the AIR, which [`StageEntries`] keeps the answer of. It
/// reads the module's decode from `shared`, the same object, so the answer and the stage's entry
/// decode the module once between them.
fn stage_returns_value_read(
    fd: &FunctionDescriptor,
    shared: &StageEntries,
    m: &mut Meter,
) -> Result<bool, Error> {
    let inputs = LowerInputs::new();
    guarded(|| {
        with_request_shared(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            InputsFrom::Given(&inputs),
            Shared::Decode(shared),
            |r| {
                let mut decoded = decoded_module(r, m)?;
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
/// device without a fragment `PrimitiveId` ([`primitive_id_varying_on`]), so no other decodes
/// anything.
pub(crate) fn reads_primitive_id_varying(
    fragment: Option<&crate::pipeline::descriptor::StageSource>,
    depth: u32,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<bool, Error> {
    use crate::pipeline::descriptor::StageSource;
    Ok(match fragment {
        Some(StageSource::Function(fd)) => entries
            .specialized(fd, depth, (false, &BTreeSet::new()), m)?
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

/// Run `g` on the compile request of function `f`. A visible function is not a stage: it is
/// `EntryNotFound`, as for a module that does not declare the entry.
fn with_request<T>(
    f: &Function,
    consts: &FunctionConstantValues,
    links: (&LinkedFunctions, &DynamicLibraries),
    inputs: InputsFrom<'_>,
    g: impl FnOnce(&Request<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    with_request_shared(f, consts, links, inputs, Shared::Nothing, g)
}

/// [`with_request`], the request reading from what the creation `shared` holds
/// (`Request::shared`).
fn with_request_shared<'k, T>(
    f: &Function,
    consts: &FunctionConstantValues,
    (linked, libraries): (&LinkedFunctions, &DynamicLibraries),
    inputs: InputsFrom<'_>,
    shared: Shared<'k>,
    g: impl FnOnce(&Request<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    let source = f.source()?;
    let air_hash = f.info().air_hash;
    let dynamic = dynamic_order(f, linked, libraries);
    let linked = linked_sources(linked)?;
    let r = match &source {
        FunctionSource::Metallib(lf) => Request {
            bitcode: lf.bitcode,
            air_hash,
            entry: EntryRef::Name(lf.name),
            container: Some(lf.as_ref()),
            consts,
            linked: &linked,
            dynamic,
            inputs,
            shared,
        },
        FunctionSource::Module { bitcode, index, .. } => Request {
            bitcode,
            air_hash,
            entry: EntryRef::Index(*index),
            container: None,
            consts,
            linked: &linked,
            dynamic,
            inputs,
            shared,
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
                            | crate::library::FunctionStage::Extern
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

/// Compile entry `entry` of one wrapped or raw bitcode module (`bitcode`), refusing a module
/// above `word_cap` SPIR-V words after serialization (`budget::SPIRV_WORDS` in production).
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

/// [`compile_bitcode_kept`] under the stage bounds `bounds` in place of
/// [`StageBounds::MEASURED`], so a test reaches each bound with a small module.
#[cfg(test)]
pub(crate) fn compile_bitcode_bounded(
    bitcode: &[u8],
    entry: EntryRef<'_>,
    p: &DeviceProfile,
    m: &mut Meter,
    bounds: StageBounds,
) -> Result<(CompiledFunction, Plan, TypedModule), Error> {
    guarded(|| {
        let consts = FunctionConstantValues::defaults();
        let inputs = LowerInputs::new();
        let r = Request::bitcode(bitcode, entry, &consts, &[], &inputs);
        compile_bounded(&r, p, m, (SPIRV_WORDS, bounds), TypedModule::clone)
            .map(|(artifact, plan, module)| (CompiledFunction::new(artifact), plan, module))
    })
}

/// One plan of `bitcode`'s entry with default lowering inputs, keeping the widest mode's pin or
/// weighing it as `widest` says, and its module emitted and weighed as the stage bound makes each
/// plan it compares (`stage_bound`), with that weight.
#[cfg(test)]
pub(crate) fn emit_bitcode_as(
    bitcode: &[u8],
    entry: EntryRef<'_>,
    p: &DeviceProfile,
    widest: crate::lower::simd::WidestPin,
    m: &mut Meter,
) -> Result<(Plan, TypedModule, u64), Error> {
    guarded(|| {
        let consts = FunctionConstantValues::defaults();
        let inputs = LowerInputs::new();
        let r = Request::bitcode(bitcode, entry, &consts, &[], &inputs);
        stage_bound::emit_as(&r, p, widest, m)
    })
}

/// Stage `fd` compiled with default lowering inputs as [`compile_stage`] compiles it, or as
/// [`compile_stage_from`] compiles it reading `entries` where given, under the stage bounds
/// `bounds` in place of [`StageBounds::MEASURED`]: so a test reaches each bound with a small
/// module through either path.
#[cfg(test)]
pub(crate) fn compile_stage_bounded(
    fd: &FunctionDescriptor,
    p: &DeviceProfile,
    entries: Option<&StageEntries>,
    m: &mut Meter,
    bounds: StageBounds,
) -> Result<CompiledFunction, Error> {
    let inputs = LowerInputs::new();
    let shared = entries.map_or(Shared::Nothing, |e| Shared::Entry(e, fd));
    guarded(|| {
        with_request_shared(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            InputsFrom::Given(&inputs),
            shared,
            |r| {
                compile_bounded(r, p, m, (SPIRV_WORDS, bounds), |_| ())
                    .map(|(artifact, _, ())| CompiledFunction::new(artifact))
            },
        )
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
/// measurement of the plan's own claims.
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
        let widest = crate::lower::simd::WidestPin::Weighed;
        plan_one(&r, p, widest, m).map(|planned| planned.plan)
    })
}

/// What a creation shares with one compile request.
#[derive(Clone, Copy, Default)]
enum Shared<'a> {
    /// Nothing: the request decodes and specializes for itself alone.
    #[default]
    Nothing,
    /// The creation's specialized entries, with the stage's function descriptor: the compile reads
    /// its specialized entry there, the one a check or key of the creation made, else one made
    /// there now ([`specialize_one`]).
    Entry(&'a StageEntries, &'a FunctionDescriptor),
    /// The creation's module decodes: the request specializes here, reading the module's decode
    /// from the creation's object, so the entries of one module decode it once
    /// ([`StageEntries::module`]).
    Decode(&'a StageEntries),
}

/// One per-function compile request.
struct Request<'a> {
    /// A wrapped or raw bitcode module.
    bitcode: &'a [u8],
    /// SHA-256 of `bitcode`: the function's AIR hash, which its library states without decoding
    /// (`FunctionInfo::air_hash`, a metallib's verified `HASH` or a module's own hash) and which
    /// the compile records and keys its decode by, rather than hashing the module again.
    air_hash: Hash256,
    entry: EntryRef<'a>,
    /// A metallib function's container tags, cross-checked against the decoded AIR metadata.
    container: Option<&'a LibraryFunction<'a>>,
    consts: &'a FunctionConstantValues,
    /// The functions the stage links.
    linked: &'a [LinkedSource<'a>],
    /// The dynamic libraries the stage's imports are searched in.
    dynamic: DynamicOrder<'a>,
    inputs: InputsFrom<'a>,
    /// What the creation shares with this request.
    shared: Shared<'a>,
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
            air_hash: container::sha256(bitcode),
            entry,
            container: None,
            consts,
            linked,
            dynamic: DynamicOrder::default(),
            inputs: InputsFrom::Given(inputs),
            shared: Shared::Nothing,
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
/// A reference reaches the functions of its name that `functions` and `private_functions` list,
/// never one only `binary_functions` lists, as Metal resolves no reference against
/// `binaryFunctions` ([`LinkedList::resolves_references`]). When an entry of the module takes a
/// visible-function table or an intersection function table, or a buffer it takes holds a
/// visible-function table, every function a table can hold (`functions` and `binary_functions`) is
/// reached too, itself rather than by its name, before the references, and each listed function
/// carries its table word. A name that no function of `linked` has is left to canonicalization (a
/// call to it may be dead); a function nothing references (or, without tables, nothing but a table
/// could call) is listed only where it is one a table can hold and its module decodes and has
/// constexpr samplers, which Metal numbers, and is otherwise never decoded. Each distinct module
/// (bitcode hash) is decoded once for all its functions. Every function a referenced name reaches
/// is listed (the same function listed twice, with the same constants, is one), so
/// canonicalization sees two different ones as ambiguous; each listed function says whether a
/// reference resolves to it (`LinkedName::resolves_references`). Each function takes the values of
/// the constants its module
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
    let mut listed: Vec<LinkedKey<'_>> = Vec::new();
    let mut seen: BTreeSet<Box<[u8]>> = BTreeSet::new();
    let mut queue: VecDeque<Reach> = VecDeque::new();
    let tables = abi.entries.iter().any(|e| {
        abi.reaches_function_tables(e)
            || e.params
                .iter()
                .any(|p| matches!(p.role, ParamRole::IntersectionTable(_)))
    });
    if tables {
        for (at, h) in handles.iter().enumerate() {
            if h.is_some() {
                m.charge(std::mem::size_of::<Reach>())?;
                queue.push_back(Reach::Table(at));
            }
        }
    }
    let refs = |abi: &AirAbi, queue: &mut VecDeque<Reach>, m: &mut Meter| {
        for (_, symbol) in abi.visible_function_stubs() {
            m.charge(std::mem::size_of::<Reach>() + symbol.0.len())?;
            queue.push_back(Reach::Name(symbol.0.clone()));
        }
        Ok::<(), Error>(())
    };
    refs(abi, &mut queue, m)?;
    while let Some(reach) = queue.pop_front() {
        m.tick(1)?;
        if let Reach::Name(name) = &reach {
            if seen.contains(name) {
                continue;
            }
            m.charge(name.len())?;
            seen.insert(name.clone());
        }
        for (at, (s, handle)) in linked.iter().zip(&handles).enumerate() {
            m.tick(1)?;
            let key = &s.key;
            let reached = match &reach {
                Reach::Table(table) => at == *table,
                Reach::Name(name) => {
                    key.list.resolves_references() && key.link_name().as_bytes() == &**name
                }
            };
            if !reached {
                continue;
            }
            // The same function listed twice (in two lists, or twice in one) is one function.
            if listed.iter().any(|l| l.same_function(key)) {
                continue;
            }
            m.charge(std::mem::size_of::<LinkedKey<'_>>())?;
            listed.push(*key);
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
            let link_name = key.link_name();
            m.charge(std::mem::size_of::<LinkedName>() + link_name.len() + key.name.len())?;
            l.functions.push(LinkedName {
                name: link_name.as_bytes().into(),
                function: key.name.as_bytes().into(),
                specialized: key.specialized,
                constants,
                handle: handle.map(|h| h.word()),
                resolves_references: resolves_references(linked, key, m)?,
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
        let reached = listed.iter().any(|l| l.same_function(key));
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
        m.charge(std::mem::size_of::<LinkedKey<'_>>())?;
        listed.push(*key);
        let link_name = key.link_name();
        m.charge(std::mem::size_of::<LinkedName>() + link_name.len() + key.name.len())?;
        l.functions.push(LinkedName {
            name: link_name.as_bytes().into(),
            function: key.name.as_bytes().into(),
            specialized: key.specialized,
            constants,
            handle: handle.map(|h| h.word()),
            resolves_references: resolves_references(linked, key, m)?,
        });
    }
    Ok(out.into_iter().map(|(_, l)| l).collect())
}

/// What reaches a linked function in [`linked_modules`]: a visible-function reference to a name,
/// or a table that can hold the linked function at this index.
enum Reach {
    Name(Box<[u8]>),
    Table(usize),
}

/// Whether a visible-function reference to `key`'s name resolves to it: `linked` lists the same
/// function in a list a reference resolves against (`functions`, `private_functions`).
fn resolves_references(
    linked: &[LinkedSource<'_>],
    key: &LinkedKey<'_>,
    m: &mut Meter,
) -> Result<bool, Error> {
    for s in linked {
        m.tick(1)?;
        if s.key.list.resolves_references() && s.key.same_function(key) {
            return Ok(true);
        }
    }
    Ok(false)
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
        open_cl_mode: false,
        open_cl_lines: BTreeSet::new(),
    })
}

/// The parameters of the specialized entry `spec` an instruction of its body uses, as the lowering
/// reads a use (`lower::select`: an argument any instruction names, a call's operand included). A
/// parameter the function's constants leave unread (its reads folded away by specialization and
/// deleted with what nothing observes) is not among them.
fn params_used(spec: &air::Entry) -> BTreeSet<u32> {
    let body = spec.module.func(spec.func).and_then(|f| f.body.as_ref());
    body.map(crate::lower::users::params_used)
        .unwrap_or_default()
}

/// The Metal attribute indices the specialized vertex entry `spec` reads: its `[[attribute(n)]]`
/// parameters that survived function-constant specialization and that an instruction uses (the
/// parameters `lower::stage_io` plans), each with the numeric class it is declared with. Empty for
/// any other stage. An attribute whose value reaches only an output the constants gate off is not
/// among them, as Metal reports it inactive: specialization inserts `undef` in that output's field.
pub(crate) fn declared_attributes(
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
    let used = params_used(spec);
    for b in &spec.ep.params {
        if let ParamRole::VertexInput { location } = &b.role {
            if let AbiInt::Const(index) = location.index {
                if used.contains(&b.param) {
                    // The class the attribute is declared with: float for float forms, else the
                    // held unsigned integer (as `lower::stage_io` declares it). A type stage IO
                    // cannot hold is refused when the entry lowers.
                    let types = spec.module.types();
                    let form = match types.get(func.ty) {
                        Some(air::ir::Type::Function { params, .. }) => params
                            .get(b.param as usize)
                            .and_then(|t| crate::lower::arith::legal::form_of(types, *t).ok()),
                        _ => None,
                    };
                    let float = form.is_some_and(|f| f.is_float());
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

/// The Metal `[[stage_in]]` attribute indices the specialized kernel entry `spec` reads: its
/// `air.stage_in` parameters that survived function-constant specialization and that an
/// instruction uses, which are the parameters `lower::stage_io` plans and so the only attributes
/// the module fetches (`lower::stage_io::stage_in::param`, one `StageInFetch::of` a parameter).
/// The use is read as the lowering reads it (`lower::select`: an argument any instruction of the
/// entry's body names, a call's operand included), so the set is the lowering's own and not a
/// narrower reading of it. Empty for any other stage, which declares no such parameter.
pub(crate) fn stage_in_attributes(spec: &air::Entry) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let used = params_used(spec);
    for b in &spec.ep.params {
        if let ParamRole::StageIn { location } = &b.role {
            if let AbiInt::Const(index) = location.index {
                if used.contains(&b.param) {
                    out.insert(index);
                }
            }
        }
    }
    out
}

/// `inputs` with a kernel's stage-input descriptor narrowed to the attributes the entry of stage
/// `fd` reads (`StageInputDescriptor::narrowed_to` of [`stage_in_attributes`]): the inputs a
/// compile of `fd` under `inputs` emits its module from, stated so that the key the compile is
/// keyed under names them and nothing else. It decodes and specializes only (the entry `entries`
/// holds, or one made there now), as a key before compiling does.
///
/// `inputs` without a stage-input descriptor is its own answer, decoding nothing: a kernel that
/// states none, and every other stage, keep the inputs they were given.
pub(crate) fn kernel_inputs_read(
    fd: &FunctionDescriptor,
    inputs: &LowerInputs,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<LowerInputs, Error> {
    guarded(|| {
        let Some(given) = inputs.stage_input.as_ref() else {
            return Ok(inputs.clone());
        };
        let depth = inputs.max_call_stack_depth.unwrap_or(1);
        let open_cl = (inputs.open_cl_mode, &inputs.open_cl_lines);
        let specialized = entries.specialized(fd, depth, open_cl, m)?;
        let read = stage_in_attributes(&specialized.spec);
        let mut out = inputs.clone();
        out.stage_input = Some(given.narrowed_to(&read));
        Ok(out)
    })
}

/// Whether the kernel of stage `fd` observes its threadgroup partition under `inputs`, as its
/// compile decides it (`KernelFacts::partition_oblivious`, which
/// `CompiledFunction::observes_threadgroup_partition` answers): of the entry the compile selects
/// from, specialized (the entry `entries` holds, or one made there now) and ordered for the
/// device of `p` where its profile takes the order (`lower::simd::lockstep::order`, whose barriers
/// observe it), lowering nothing.
/// Any other stage observes it.
pub(crate) fn kernel_observes_partition(
    fd: &FunctionDescriptor,
    p: &DeviceProfile,
    inputs: InputsFrom<'_>,
    entries: &StageEntries,
    m: &mut Meter,
) -> Result<bool, Error> {
    guarded(|| {
        with_request_shared(
            &fd.function,
            &fd.constants,
            (&fd.linked, &fd.libraries),
            inputs,
            Shared::Entry(entries, fd),
            |r| {
                let ordered = ordered_entry(r, p, m)?;
                Ok(partition::partition(&ordered.spec) != Partition::Oblivious)
            },
        )
    })
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

/// The colors the specialized fragment entry `spec` reads (`[[color(n)]]`) that a render pipeline
/// whose color attachments are `colors` gives a signed-integer pixel format, under the identity
/// domains, where an integer read fetches the attachment's own view through an `int` image
/// (`lower::fragment::color`'s `texture_class`); none elsewhere, where it reads the attachment's
/// bits through a `uint` input attachment.
fn signed_colors_fetched(
    spec: &air::Entry,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    use crate::pipeline::sym::SampledType;
    if !crate::lower::access::identity(p) || !matches!(spec.ep.stage, AbiStage::Fragment { .. }) {
        return BTreeSet::new();
    }
    spec.ep
        .params
        .iter()
        .filter_map(|param| match param.role {
            ParamRole::ColorInput {
                index: AbiInt::Const(a),
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
    out.texture_write_rounding = d.texture_write_rounding_mode;
    // The floor's one render range moves the fragment function's fields.
    out.push_base = d.push_base;
    out
}

/// The attributes the vertex function `spec` fetches in fixed function under `inputs` from
/// layouts `desc` steps per instance or at a constant step, where taking them so makes it write
/// some varying the same at every vertex of a primitive
/// (`LoweringInputs::instanced_attributes`); else none, and the function keeps the key of any
/// step.
///
/// A constant step is in that set because it is the stronger statement of the same thing: the
/// fetch reads the layout's first element for every vertex and every instance, so a varying it
/// makes is alike at every vertex of a primitive as surely as a per-instance one's is. It needs
/// saying here only since the step became a fixed-function binding of stride 0
/// (`metal::vertex::fetch_as`): a pulled constant-step fetch reads element 0 in the shader, where the
/// uniformity is already plain, so this listing is what keeps the two fetch paths writing the
/// same varyings flat.
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
        .filter(|(_, f)| {
            use crate::pipeline::lower_inputs::VertexFetch;
            matches!(
                f,
                VertexFetch::FixedFunction | VertexFetch::SignedFixedFunction
            )
        })
        .filter(|(a, _)| {
            desc.attributes
                .get(a)
                .and_then(|attr| desc.layouts.get(&attr.buffer_index))
                .is_some_and(|l| {
                    matches!(
                        l.step_function,
                        crate::metal::VertexStepFunction::PerInstance
                            | crate::metal::VertexStepFunction::Constant
                    )
                })
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
/// widens an object or mesh function's payload arena (`LoweringInputs::object_payload`): any other
/// function, and one whose own arena holds it, compiles as without it. OpenCL mode is kept for a
/// kernel alone (`LoweringInputs::open_cl_mode`).
fn given_inputs(spec: &air::Entry, i: &LowerInputs) -> LowerInputs {
    let mut out = i.clone();
    out.open_cl_mode = i.open_cl_mode && matches!(spec.ep.stage, AbiStage::Kernel { .. });
    if !out.open_cl_mode {
        out.open_cl_lines = BTreeSet::new();
    }
    out.object_payload = i.object_payload.filter(|&bytes| {
        matches!(
            spec.ep.stage,
            AbiStage::Mesh { .. } | AbiStage::Object { .. }
        ) && crate::lower::mesh::payload_size(spec)
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
        InputsFrom::Vertex(desc, variant, depth, primitive_id, _)
            if matches!(spec.ep.stage, AbiStage::Vertex { patch: Some(_) }) =>
        {
            let attributes = crate::lower::tessellation::attributes_read(spec);
            let projection =
                crate::pipeline::link::tessellation::patch_fetch_projection(desc, &attributes)?;
            let mut out = variant.lowering_inputs(crate::pipeline::sym::Stage::Vertex);
            out.vertex_fetch = Some(projection);
            out.max_call_stack_depth = call_stack_input(depth);
            // A fragment function that reads `[[primitive_id]]` behind it is refused under the
            // field (`lower::stage_io`).
            out.primitive_id_varying = primitive_id;
            Ok(key::canonical_inputs(&out))
        }
        InputsFrom::Vertex(desc, variant, depth, primitive_id, indirect) => {
            let attributes = declared_attributes(spec);
            // The primitive ID's de-indexed form, the private list form and a state made for
            // indirect command buffers pull every attribute, whatever the caller's projection
            // would fetch in fixed function: a command binds its vertex buffers by address alone.
            let pull_every =
                variant.primitive_id_indices.is_some() || variant.private_list || indirect;
            let projection = crate::pipeline::link::vertex::vertex_fetch_projection_for(
                desc,
                &attributes,
                p,
                pull_every,
            )?;
            let mut out = variant.lowering_inputs(crate::pipeline::sym::Stage::Vertex);
            out.vertex_fetch = Some(projection);
            out.max_call_stack_depth = call_stack_input(depth);
            // Only on a device that cannot declare a fragment `PrimitiveId`: elsewhere a plain
            // vertex function's fragment function reads Vulkan's, Metal's there.
            out.primitive_id_varying = primitive_id && primitive_id_varying_on(p);
            out.indirect_range = indirect;
            out.instanced_attributes = instanced_attributes(spec, &out, desc, m)?;
            Ok(key::canonical_inputs(&out))
        }
        InputsFrom::Fragment(d, variant, written, primitive, varyings, uniform) => {
            let mut out = fragment_inputs(d, variant, p);
            out.signed_color_attachments = signed_colors_written(spec, d.color_attachments);
            out.signed_color_attachments.extend(signed_colors_fetched(
                spec,
                d.color_attachments,
                p,
            ));
            out.unattached_colors =
                unattached_colors_written(spec, d.color_attachments, d.alpha_to_coverage);
            let read = raw_indices_read(spec);
            out.viewport_index_varying = written.viewport && read.viewport;
            out.render_target_array_index_varying = written.layer && read.layer;
            // Behind a vertex function that stores its region's draw index as `ViewportIndex`,
            // a fragment function that reads `[[viewport_array_index]]` reads 0 in its place.
            out.region_viewports = variant.region_viewports && read.viewport;
            // Behind a vertex function that writes the primitive ID's varying, on a device that
            // declares a fragment `PrimitiveId` (one compiled under the field): elsewhere the
            // fragment function reads the varying without the field.
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
    unbound: unbound::Unbound,
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
/// entry name, the specialized entry, and what its code read before. Its work weighs
/// [`weights::SPECIALIZE`] whoever asks.
fn specialize_one(r: &Request<'_>, m: &mut Meter) -> Result<Specialized, Error> {
    m.weighted(weights::SPECIALIZE, |m| specialize_unweighed(r, m))
}

/// [`specialize_one`]'s work.
fn specialize_unweighed(r: &Request<'_>, m: &mut Meter) -> Result<Specialized, Error> {
    #[cfg(test)]
    tests::inject::panic_if_armed();
    // The creation's entry of this stage: the same request's, made by a check or key of the
    // creation, else made there now.
    if let Shared::Entry(entries, fd) = r.shared {
        let open_cl = (r.inputs.open_cl_mode(), r.inputs.open_cl_lines());
        let kept = entries.specialized(fd, r.inputs.call_stack_depth(), open_cl, m)?;
        return Ok(Specialized::clone(&kept));
    }
    let mut decoded = decoded_module(r, m)?;
    if let Some(f) = r.container {
        decoded.abi.check_container(&decoded.module, f, m)?;
    }
    let mut req = spec_request(&decoded, r.entry, r.consts)?;
    // A stage compiles under its pipeline's `maxCallStackDepth` for it, and a kernel under its
    // pipeline's OpenCL mode.
    req.call_stack_depth = r.inputs.call_stack_depth();
    req.open_cl_mode = r.inputs.open_cl_mode();
    // And a kernel's 1D images bound to a 1D texture, by texture index, which stay 1D.
    req.open_cl_lines = (r.inputs.open_cl_lines().iter())
        .filter(|s| s.stage == Stage::Kernel && s.class == SlotClass::Texture)
        .map(|s| s.index)
        .collect();
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
        air_hash: r.air_hash,
        entry: req.entry,
        spec,
        before,
    })
}

/// The decode of the request's module: a copy of the one the creation's object holds (read there
/// now where it holds none), else one read for this compile alone.
///
/// Decoding reads the bitcode and nothing else, so every entry of one module, and every
/// descriptor and depth of one function, decodes the same bytes to the same module. A request
/// that shares nothing ([`Shared::Nothing`], which is what [`compile_stage`] and every
/// reflection read of a lone function make) decodes as it always did and keeps nothing.
fn decoded_module(r: &Request<'_>, m: &mut Meter) -> Result<air::Decoded, Error> {
    match r.shared {
        Shared::Decode(entries) => entries.module(&r.air_hash, r.bitcode, m),
        Shared::Entry(..) | Shared::Nothing => Ok(air::decode(r.bitcode, m)?),
    }
}

/// [`specialize_one`], then the order Apple's simdgroup lockstep gives threadgroup accesses made
/// explicit for `p` where its profile takes one ([`OrderKey::of`]; under the identity domains the
/// specialized entry with its pairs found and none ordered): the entry selection reads, whose instruction indices a refusal's
/// location names. [`plan_one`] and `DeviceProfile::specialized_entry` both build it
/// here, so a tool that resolves a refusal's location reads the compile's own entry.
///
/// A request that shares the creation's entries orders the entry there once for the creation
/// ([`StageEntries::ordered`]): the check of the partition its kernel observes and its compile
/// read one ordered entry.
fn ordered_entry(r: &Request<'_>, p: &DeviceProfile, m: &mut Meter) -> Result<Specialized, Error> {
    let key = OrderKey::of(p);
    if let Shared::Entry(entries, fd) = r.shared {
        let open_cl = (r.inputs.open_cl_mode(), r.inputs.open_cl_lines());
        let depth = r.inputs.call_stack_depth();
        let kept = entries.ordered(fd, depth, open_cl, key, m)?;
        return Ok(Specialized::clone(&kept));
    }
    let specialized = specialize_one(r, m)?;
    order_specialized(specialized, key, m)
}

/// The specialized entry `specialized` ordered for `key` ([`lockstep::order`]), its work weighing
/// [`weights::LOCKSTEP`].
fn order_specialized(
    specialized: Specialized,
    key: OrderKey,
    m: &mut Meter,
) -> Result<Specialized, Error> {
    let Specialized {
        air_hash,
        entry,
        spec,
        before,
    } = specialized;
    let spec = m.weighted(weights::LOCKSTEP, |m| lockstep::order(spec, key, m))?;
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
fn plan_one(
    r: &Request<'_>,
    p: &DeviceProfile,
    widest: crate::lower::simd::WidestPin,
    m: &mut Meter,
) -> Result<Planned, Error> {
    let analyzed = analyze_one(r, p, m)?;
    let (plan, kernel) = analyzed.select(p, widest, m)?;
    Ok(analyzed.planned(plan, kernel))
}

/// What [`plan_one`] reads of a request before it selects: the ordered entry, what its code read
/// before specialization, the inputs it lowers under, its key, its memory plan, the partition
/// it observes, whether it is image-only, and what it reads and writes through anything but its bound slots. A plan made again past the stage bound (`stage_bound`) selects from it again,
/// without decoding, specializing, ordering or analyzing the entry again.
struct Analyzed {
    key: FunctionCompileKey,
    spec: air::Entry,
    before: reflection::UnspecializedReads,
    inputs: LowerInputs,
    memory: crate::memory::plan::MemoryPlan,
    observed: Partition,
    image_only: bool,
    unbound: unbound::Unbound,
}

impl Analyzed {
    /// The plan of the analyzed entry, keeping the widest mode's pin or weighing it as `widest`
    /// says, and the kernel facts it gives.
    fn select(
        &self,
        p: &DeviceProfile,
        widest: crate::lower::simd::WidestPin,
        m: &mut Meter,
    ) -> Result<(Plan, KernelFacts), Error> {
        let (spec, memory, observed) = (&self.spec, &self.memory, self.observed);
        let plan = m.weighted(weights::SELECT, |m| {
            crate::lower::select::select(spec, memory, p, (&self.inputs, widest), m)
        })?;
        let kernel = KernelFacts {
            max_total_threads: spec.ep.stage.max_total_threads(),
            work_group_size: spec.ep.stage.work_group_size(),
            flat_local_size: plan
                .dispatch
                .as_ref()
                .is_some_and(|d| d.local_size == LocalSizeForm::Flat),
            // The static part: the objects, and with `[[threadgroup(n)]]` regions the
            // 16-byte-aligned offset where the first region starts. With typed threadgroup
            // objects (no regions), the static part of Metal's own layout, which holds them too.
            static_threadgroup_bytes: if memory.typed_shared.is_empty() {
                crate::memory::arena::TgShape::of(&memory.tg_arena)
                    .static_bytes()
                    .unwrap_or(u64::MAX)
            } else {
                memory.tg_metal_static
            },
            partition_oblivious: observed == Partition::Oblivious,
            partition_simdgroups: observed == Partition::Simdgroups,
            image_only: self.image_only,
        };
        Ok((plan, kernel))
    }

    /// The plan `plan` of the analyzed entry, with its kernel facts `kernel`.
    fn planned(self, plan: Plan, kernel: KernelFacts) -> Planned {
        Planned {
            key: self.key,
            spec: self.spec,
            before: self.before,
            plan,
            kernel,
            unbound: self.unbound,
        }
    }
}

/// [`plan_one`]'s reads before it selects ([`Analyzed`]).
fn analyze_one(r: &Request<'_>, p: &DeviceProfile, m: &mut Meter) -> Result<Analyzed, Error> {
    let Specialized {
        air_hash,
        entry,
        spec,
        before,
    } = ordered_entry(r, p, m)?;
    color_inputs_attached(&spec, r.inputs)?;
    let inputs = lower_inputs(&spec, r.inputs, p, m)?;
    // A partial threadgroups variant runs a direct `dispatchThreads` on subgroups, whose
    // invocations past the grid it walks: none of the other forms, and no emulated simdgroups. It
    // is refused first, so the variant asked with another form is refused as itself.
    if inputs.partial_threadgroups
        && (inputs.flat_grid
            || inputs.folded_groups
            || inputs.replay_bindings
            || inputs.emulated_simdgroups)
    {
        return Err(Unsupported::not_implemented(Feature::PartialThreadgroups).into());
    }
    // A flat variant runs its grid in threadgroups of its own: only a kernel that does not observe
    // its partition computes the same there; one of the call's threadgroups, also a kernel that
    // observes it through its simdgroups alone.
    let observed = partition::partition(&spec);
    let image_only = images::image_only(&spec);
    let flat_runs = match inputs.call_threadgroups {
        false => observed == Partition::Oblivious,
        true => observed <= Partition::Simdgroups,
    };
    if inputs.flat_grid && !flat_runs {
        return Err(Unsupported::not_representable(Feature::GpuEncodedIndirectCommands).into());
    }
    // A folded variant runs one threadgroup dispatch of its own threadgroup size: neither a flat
    // grid nor a device replay's bindings. The call's threadgroups are a flat grid's alone, and
    // not a device replay's, whose commands' threadgroups its preprocess reads.
    if inputs.folded_groups && (inputs.flat_grid || inputs.replay_bindings)
        || inputs.call_threadgroups && (!inputs.flat_grid || inputs.replay_bindings)
    {
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
    let memory = m.weighted(crate::pipeline::compile::weights::MEMORY, |m| {
        crate::memory::plan::analyze_memory(&spec, p, m)
    })?;
    let unbound = unbound::unbound(&spec, &memory);
    Ok(Analyzed {
        key,
        spec,
        before,
        inputs,
        memory,
        observed,
        image_only,
        unbound,
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
/// before it is serialized (`compile_bitcode_kept` keeps a copy).
fn compile_kept<K>(
    r: &Request<'_>,
    p: &DeviceProfile,
    m: &mut Meter,
    word_cap: u64,
    keep: impl FnOnce(&TypedModule) -> K,
) -> Result<(Arc<FunctionArtifact>, Plan, K), Error> {
    compile_bounded(r, p, m, (word_cap, StageBounds::MEASURED), keep)
}

/// [`compile_kept`] under the stage bounds `bounds` where the profile has their defects or bounds
/// every stage (`stage_bound`; [`StageBounds::MEASURED`] in production). The work of no stage
/// with a weight of its own weighs [`weights::REST`].
fn compile_bounded<K>(
    r: &Request<'_>,
    p: &DeviceProfile,
    m: &mut Meter,
    bounds: (u64, StageBounds),
    keep: impl FnOnce(&TypedModule) -> K,
) -> Result<(Arc<FunctionArtifact>, Plan, K), Error> {
    m.weighted(weights::REST, |m| lower_bounded(r, p, m, bounds, keep))
}

/// [`compile_bounded`]'s lowering.
fn lower_bounded<K>(
    r: &Request<'_>,
    p: &DeviceProfile,
    m: &mut Meter,
    (word_cap, bounds): (u64, StageBounds),
    keep: impl FnOnce(&TypedModule) -> K,
) -> Result<(Arc<FunctionArtifact>, Plan, K), Error> {
    let (
        Planned {
            key,
            spec,
            before,
            plan,
            kernel,
            unbound,
        },
        crate::lower::emit::Emitted {
            mut module,
            merged,
            held,
        },
        entry_weight,
    ) = stage_bound::emit(r, p, bounds, m)?;
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
    // A stencil output drawn by bits runs the function once more for each bit, so its memory
    // writes would be made again.
    if writes_memory && plan.facts.contains(&RuntimeFact::StencilBitDraws) {
        return Err(crate::lower::stage_io::stencil_export_refused());
    }
    // A fragment function's texture slots read only at the fragment's own pixel.
    let own_pixel_textures = crate::spirv::requirements::own_pixel_textures(&module, m)?;
    // The levels of its bound view each texture slot's reads reach.
    let level_reads = crate::spirv::requirements::level_reads(&module, m)?;
    // A `half` rounded through `OpQuantizeToF16` (`QuantizeForm::Device`): the module records the
    // native note on every profile, a result below the smallest normal half the device's.
    let mut precision = plan.precision.clone();
    let specialized = plan.specialized.clone();
    let quantize = plan.families.arith.quantize;
    // Whether the module computes `half` under the 16-bit float controls
    // (`lower::arith::narrow::half_controls`), whatever its rounding form.
    let controls = plan.families.arith.controls;
    let device_quantize = quantize == crate::spirv::builder::QuantizeForm::Device
        && crate::lower::arith::narrow::quantizes(&module);
    // A `half` rounded through the device's conversion to 16 bits and back, with no quantize: the
    // module records the native note.
    let converts = quantize == crate::spirv::builder::QuantizeForm::Converted
        && crate::lower::arith::narrow::rounds_by_conversion(&module, m)?;
    // A `half` read from its bits by the device's widening, or encoded to them by its narrowing
    // (every profile: `OpFConvert` with `shaderFloat16`, `UnpackHalf2x16` and `PackHalf2x16`
    // without): a NaN's bits are the device's, recorded. Under the float controls its denormals
    // are Metal's, as a single conversion's, also on a driver that folds the round trip; elsewhere
    // the device may flush them, within the native conversion's note.
    let widens = crate::lower::arith::narrow::widens_bits(&module, m)?
        || crate::lower::arith::narrow::encodes_bits(&module, m)?;
    let native_widening = widens && !controls;
    // A `half` rounded through the conversion and its widening under the float controls: Metal's
    // `half(float)` by the Vulkan spec, but a driver's compiler may fold the pair into the value it
    // started from or fuse the `float` operation computing the value into the narrowing and round
    // its exact result once. The module records the native note on every device, a static note of
    // the rounding.
    let controlled = quantize == crate::spirv::builder::QuantizeForm::Controlled;
    let fused = controlled && crate::lower::arith::narrow::rounds_by_conversion(&module, m)?;
    // A `half` rounded through the device's conversion and its widening, each decorated
    // `NoContraction`, under the float controls that make it Metal's, computed by the device's
    // 16-bit arithmetic, which they make Metal's too, or read from its bits by the device's
    // widening, which they make exact but for a NaN's bits: the entry declares them, and so
    // requires the device's float-controls properties for them, and no fact. A driver may fold
    // the round trip whatever the entry declares (NVIDIA's does): the pair's rounding is then the
    // device's, the static note recorded below, and the rest is Metal's under them.
    let declares_controls =
        controls && (widens || crate::lower::arith::narrow::computes_in_16_bits(&module) || fused);
    if declares_controls {
        let modes = crate::lower::arith::narrow::controlled_modes(p).unwrap_or_default();
        module.declare_modes(&modes)?;
    }
    // An entry whose tangled instructions all stand where their invocations reach them together
    // takes its reconvergence mode only where the module holds one in a loop a back edge re-enters
    // from inside a selection.
    let reconvergence = crate::lower::simd::reentry::settle(&plan.families.simd, &mut module)?;
    // Native half rounding and widening: a module that rounds through the device's quantize or
    // conversion, or widens a `half` by the device where that is not controlled to be Metal's,
    // records each kind, and requires no fact.
    let native_half = [
        (widens, PrecisionNote::NativeNanEncoding),
        (device_quantize, PrecisionNote::NativeHalfQuantize),
        (
            converts || native_widening || fused,
            PrecisionNote::NativeHalfConversion,
        ),
    ];
    for (recorded, note) in native_half {
        if recorded && !precision.contains(&note) {
            precision.push(note);
            precision.sort_unstable();
        }
    }
    if numbering_differs {
        precision.push(PrecisionNote::LinkedSamplerNumbering);
        precision.sort_unstable();
    }
    // A sample left to the device's sampler: the device's answer.
    if plan.families.sampler.device_sampling && !precision.contains(&PrecisionNote::DeviceSampling)
    {
        precision.push(PrecisionNote::DeviceSampling);
        precision.sort_unstable();
    }
    let stats = crate::pipeline::artifact::ModuleStats {
        entry_weight,
        ..crate::spirv::verify::verify(&module, p, m)?
    };
    let kept = keep(&module);
    let mut symbols =
        crate::pipeline::link::contract::stage_symbols(&module, &plan.families.stage_io.interface)?;
    // The bits of each specialization constant the module never reads.
    crate::spirv::spec_bits::mark_unread(&module, &mut symbols.spec, &specialized, m)?;
    // The serializer's id tables live while it writes.
    let ids = crate::spirv::serialize::id_table_bytes(&module);
    m.charge(ids)?;
    let module = crate::spirv::serialize::serialize(module)?;
    // The typed module is gone with its serialization, and its charge with it.
    m.release(ids.saturating_add(held));
    m.charge(module.footprint())?; // The retained words and relocations.

    // The cap holds on what was emitted.
    if u64::try_from(module.words.len()).map_or(true, |w| w > word_cap) {
        return Err(Error::LimitExceeded(crate::error::Limit::SpirvWords));
    }
    let interface = stage_interface(
        &plan,
        symbols,
        writes_memory,
        &merged,
        &function_tables,
        (reaches_command_buffers, command_writes),
        (writes_textures, numbered_samplers),
        (own_pixel_textures, level_reads),
        unbound,
    );
    let mut requirements = module.requirements.clone();
    // Every sample, gather and comparison is the device's own sampler's, which the stage records
    // and requires.
    if plan.families.sampler.device_sampling {
        requirements
            .properties
            .insert(crate::device::requirements::VkProperty::DeviceSampling);
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
    // Implementation-defined where a tangled instruction runs without maximal reconvergence
    // (`lower::simd` chooses the entry's reconvergence mode), which the stage's note records.
    if reconvergence == Reconvergence::ImplementationDefined
        && !precision.contains(&PrecisionNote::DeviceReconvergence)
    {
        precision.push(PrecisionNote::DeviceReconvergence);
        precision.sort_unstable();
    }
    let artifact = Arc::new(FunctionArtifact {
        key,
        module,
        interface,
        facts: plan.facts.clone(),
        subgroup: plan.subgroup,
        reconvergence,
        precision,
        strategies: census(&plan),
        stats,
        // The module's own blob (the identity domains) is in its words: the artifact carries a
        // bound blob's bytes alone.
        blob: (plan.families.access.blob.as_ref())
            .filter(|b| b.home == crate::memory::blob::BlobHome::Bound)
            .map(|b| b.bytes.clone()),
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

/// The stage interface the artifact carries: the slots, the push placement, the module's link
/// contract (every descriptor binding and specialization constant it declares), and whether it
/// writes memory. A buffer slot's access is the function's, with what the accesses through loaded
/// or converted pointers may do to a slot whose address escapes (`memory::plan::SlotPlan::wide`),
/// and a read where emission merged a store's partial word into the slot (`merged`): the sub-word
/// store's atomics read the word they modify. A buffer slot at an index of `function_tables` holds
/// a visible-function table read as its words. `reaches_command_buffers` is the canonical entry's
/// `Entry::may_encode_indirect_commands`, `command_writes` its `Entry::command_writes`,
/// `writes_textures` its `Entry::may_write_textures`,
/// `numbered_samplers` the samplers a pipeline state numbers for it (`numbered::samplers`), and
/// `unbound` what it reads and writes through anything but its bound slots (`unbound`).
fn stage_interface(
    plan: &Plan,
    symbols: crate::pipeline::link::contract::StageSymbols,
    writes_memory: bool,
    merged: &std::collections::BTreeSet<crate::pipeline::sym::SlotRef>,
    function_tables: &std::collections::BTreeSet<u32>,
    (reaches_command_buffers, command_writes): (bool, crate::pipeline::interface::CommandWrites),
    (writes_textures, numbered_samplers): (bool, Vec<crate::pipeline::interface::NumberedSampler>),
    (own_pixel_textures, level_reads): (Vec<u32>, Vec<crate::pipeline::interface::LevelReads>),
    unbound: unbound::Unbound,
) -> StageInterface {
    use crate::pipeline::interface::{BufferKind, BufferSlot};
    use crate::pipeline::sym::PushField;
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
        payload: plan.families.mesh.payload,
        reaches_command_buffers,
        command_writes,
        writes_textures,
        reads_unbound_resources: unbound.reads,
        writes_unbound_resources: unbound.writes,
        unbound_storage_textures: unbound.storage_textures,
        numbered_samplers,
        own_pixel_textures,
        level_reads,
        private_bytes: plan.private_bytes,
        sampler_arrays: crate::lower::sampler::sampler_arrays(plan),
    }
}
