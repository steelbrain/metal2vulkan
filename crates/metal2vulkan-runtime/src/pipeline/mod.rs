//! `ComputePipelineState` (`MTLComputePipelineState`),
//! `Device::new_compute_pipeline_state_with_function` and `Device::new_compute_pipeline_state` (its
//! descriptor form, with the kernel's linked functions and the descriptor's other fields); render
//! pipeline states are `render.rs`, their `VkPipeline`s `graphics.rs`, and a tile pipeline's state
//! (`tile.rs`) is a render pipeline state over a compute state of its tile function, which the
//! compile layer compiles as a kernel.
//!
//! - **Creation.** A state compiles its kernel under the empty variant key and the budget of its
//!   [`PipelineOptions`] (`options.rs`), as Metal compiles at creation, and makes the variant's
//!   shader module and layouts: what Metal's pipeline creation refuses, this refuses
//!   (`Compile`), and the state answers Metal's queries from that compile. Where that compile
//!   needs storage textures' formats (a device without `shaderStorageImageReadWithoutFormat`
//!   declares the format of a texture a function reads), it takes stand-ins for them
//!   (`stand_in`): that compile answers the queries, and is planned but never dispatched. A
//!   function no stand-in compiles for is refused with the first compile's reason
//!   (`NotImplemented(Textures)` for a format it needs).
//! - **Variants.** A dispatch's bind shape can ask for another variant (`BindingPlan::variant`,
//!   and `dispatch_variant` for a threadgroup with a partial simdgroup): the state compiles it
//!   under the standard budget when a dispatch first needs it, plans it again, and dispatches the
//!   variant the plan settles on. A plan of the compiled variant that asks for yet another one is
//!   an internal error. A dispatch in the large-buffer form ([`BufferForms::Large`], an indirect
//!   command's the GPU may write) runs on the variant with every slot a large buffer moves to its
//!   device address there, whatever buffer is bound: its plan may ask for fewer such slots, never
//!   for another variant.
//! - **The memo, archives and the disk cache.** A variant's translated function and compiled
//!   pipeline are looked up by their keys before compiling (`sources.rs`): in the device's memo
//!   (`memo.rs`), then in the creation's binary archives, then in the shader cache on disk
//!   (`shader_cache`), and kept after. A hit skips the translation, and the budget with it. Under
//!   `FAIL_ON_BINARY_ARCHIVE_MISS`, a creation that lists archives none of which holds its archive
//!   key is refused first (`binary_archive`). A pipeline the memo knows a live variant of is that
//!   variant: a state made again shares its objects, plans and `VkPipeline`s, and is a state of its
//!   own, as Metal's are.
//! - **Imageblocks.** `imageblock_memory_length_for_dimensions` answers an M4's rounding, never
//!   less than what a dispatch of that imageblock size counts against the device's threadgroup
//!   memory.
//! - **Memos.** Each state keeps its variants, each variant its plans by bind shape and its
//!   `VkPipeline`s by region specialization, every memo bounded, the least recently used going
//!   first (`memo::Lru`). Command buffers retain what their commands use, so dropping a memo's
//!   entry never destroys an object a pending command uses.

mod additions;
mod attachment_formats;
pub(super) mod budget;
pub(super) mod content_keyed;
pub(super) mod creation_log;
mod dispatch_making;
pub(super) mod flight;
mod graphics;
mod indirect_arguments;
pub(super) mod kept;
pub(super) mod libraries;
mod linked_fragment;
pub(super) mod making;
pub(super) mod memo;
pub(super) mod objects;
mod options;
mod piece_holds;
pub(super) mod pool;
mod predicted;
mod raster_samples;
pub(super) mod reaper;
mod render;
pub(super) mod shared_layouts;
mod sources;
pub(super) mod stand_in;
mod texture_access;
mod tile;
mod unrasterized;

pub use additions::{
    BinaryFunctionProblem, BinaryFunctionStage, RenderPipelineFunctionsDescriptor,
    RequiredThreadsProblem,
};
pub(super) use creation_log::enabled as creation_log_enabled;
pub(super) use dispatch_making::RegionPipeline;
pub(super) use graphics::{
    color_write_masks, keyed_assembly, DepthRange, DrawKey, FixedState, GraphicsKey,
};
pub(super) use making::Pending;
pub(super) use objects::ShaderModules;
pub(super) use objects::{create_set_layout, set_needs, RawPipeline, VariantObjects};
pub use options::{PipelineOption, PipelineOptions};
/// The form a driver's facts choose, for the test that holds every committed profile to it
/// (`tests::pipeline::every_committed_driver_reads_its_uniform_views_at_their_residuals`).
#[cfg(test)]
pub(super) use render::views_read_at_bind;
pub(super) use render::DrawDynamic;
pub(super) use render::DrawPipeline;
pub use render::RenderPipelineState;
pub(super) use render::RenderProgram;
pub(super) use render::RenderVariant;
pub(super) use render::Settled;
pub(super) use render::WeakRenderPipelineState;
pub(super) use render::{reads_views_at_bind, residual_form};
pub(super) use render::{GraphicsState, Kind as RenderStateKind, StateShared as RenderShared};
pub(super) use tile::TileState;

use super::binary_archive;
use super::device::{device_error, lock};
use super::error::{internal, invalid, ArchivedPipeline, Error, Problem, RuntimeGap};
use super::families::GpuFamily;
use super::library::Function;
use super::values::Size;
use super::Device;
use crate::argument_table::AtBindViews;
use crate::binary_archive::Used;
use crate::degradation::{CompiledPipeline, Occurrence};
use crate::released_programs::KeptProgram;
use crate::residency::handle_stages;
use crate::resource_ids::{
    function_name, number, pipeline_placement, Holder, IdHold, IdPlacement, IdSpace, Numbered,
    NumberedStage, ResourceId, StateKind,
};
use crate::unique_identifier::UniqueIdentifier;
use memo::{Lru, Remembered};
use metal2vulkan::bind::{
    BindShape, BindingPlan, Dispatch, DispatchCommands, DispatchRegion, IndirectRef,
};
use metal2vulkan::cache::{SpecializationKey, VariantKey};
use metal2vulkan::error::{Feature, Needs, RuntimeInput};
use metal2vulkan::metal::RenderStages;
use metal2vulkan::pipeline::{
    CommandWrites, ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, LinkedFunction,
    LinkedFunctions, PipelineInterface, PipelineLayout, RuntimeFact, SpecKey, Stage,
    StageInputDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan::reflection::ComputePipelineReflection;
use sources::Sources;
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// `imageblockMemoryLengthForDimensions:` for an explicit imageblock of `cell_bytes` a cell (the
/// compile's), none without an imageblock or at a width or height of 0. Otherwise each side rounds
/// up to a power of two and the depth is ignored, as Metal does on an M3 and an M4 (its API takes
/// only powers of two). Metal rounds the cell up to 4 bytes on an M3 and to 8 on an M4: this
/// answers the M4's, raised to what the binder reserves on the device (the cells rounded up to 16
/// bytes) where that is more, which is only a single cell whose rounded size is an odd multiple
/// of 8. So the answer is never less than what a dispatch of that size counts against the device's
/// threadgroup memory (docs/API.md §10).
pub(super) fn imageblock_memory_length(cell_bytes: Option<u32>, dimensions: Size) -> u64 {
    let Some(cell) = cell_bytes else {
        return 0;
    };
    if dimensions.width == 0 || dimensions.height == 0 {
        return 0;
    }
    let side = |n: u64| n.checked_next_power_of_two().unwrap_or(u64::MAX);
    let cells = side(dimensions.width).saturating_mul(side(dimensions.height));
    let rounded = |bytes: u64, to: u64| bytes.checked_next_multiple_of(to).unwrap_or(u64::MAX);
    let answered = cells.saturating_mul(rounded(u64::from(cell), 8));
    let reserved = rounded(cells.saturating_mul(u64::from(cell)), 16);
    answered.max(reserved)
}

/// The value of each of a pipeline's specialization `constants` (its id, its meaning and the
/// module's default) where `key` specializes the pipeline, by its meaning, in id order, one per
/// id: the value `key` gives it, else the default, each as its low bytes, as
/// `SpecConstant::default` holds a default. An entry of `key` for an id no constant has
/// specializes nothing and is left out; one whose bytes `key`'s data does not hold leaves the
/// default.
pub(super) fn specialization_values(
    constants: impl IntoIterator<Item = (u32, SpecKey, u64)>,
    key: &SpecializationKey,
) -> Vec<(SpecKey, u64)> {
    let mut values = BTreeMap::new();
    for (id, meaning, default) in constants {
        values.entry(id).or_insert((meaning, default));
    }
    for entry in &key.entries {
        let Some((_, value)) = values.get_mut(&entry.constant_id) else {
            continue;
        };
        let at = entry.offset as usize;
        let Some(bytes) = key.data.get(at..at + entry.size as usize) else {
            continue;
        };
        let mut low = [0u8; 8];
        for (to, from) in low.iter_mut().zip(bytes) {
            *to = *from;
        }
        *value = u64::from_le_bytes(low);
    }
    values.into_values().collect()
}

/// The most plans a variant keeps.
const PLANS: usize = 64;
/// The most `VkPipeline`s a variant keeps.
const PIPELINES: usize = 32;
/// The most variants besides the first a state keeps.
const VARIANTS: usize = 8;

/// A Metal compute pipeline state (`MTLComputePipelineState`). Cheap to clone (a retain); `Send +
/// Sync`.
#[derive(Clone)]
pub struct ComputePipelineState(Arc<StateShared>);

/// A compute pipeline state named without holding it ([`ComputePipelineState::downgrade`]).
#[derive(Clone)]
pub(crate) struct WeakComputePipelineState(Weak<StateShared>);

impl WeakComputePipelineState {
    /// The state, while the program holds it.
    pub(crate) fn upgrade(&self) -> Option<ComputePipelineState> {
        self.0.upgrade().map(ComputePipelineState)
    }
}

/// What a compute state runs, held without its ID: an indirect command keeps the program of the
/// state set on it past the program's last release of the state, as Metal's runs it, while the
/// state's ID goes back (`ComputePipelineState::program`).
#[derive(Clone)]
pub(crate) struct ComputeProgram(Arc<Program>);

impl ComputeProgram {
    /// A state that runs the program, with no ID (`gpu_resource_id` 0): the one a command runs
    /// once the program released the state set on it. Equal to that state.
    pub(crate) fn state(&self) -> ComputePipelineState {
        ComputePipelineState(Arc::new(StateShared {
            id: None,
            identifier: None,
            _samplers: Vec::new(),
            program: Arc::clone(&self.0),
        }))
    }
}

/// A compute state: its ID, which goes back at its last handle's release, and what it runs.
pub(super) struct StateShared {
    /// The state's ID, held while it lives: a state made with `supportIndirectCommandBuffers`
    /// has one of the pipeline states' space, any other none.
    id: Option<IdHold>,
    /// Its `uniqueIdentifier` (`unique_identifier.rs`), held while it lives; none for the
    /// stand-in of a released state's program (`ComputeProgram::state`).
    identifier: Option<UniqueIdentifier>,
    /// The sampler IDs the state numbered as it was made (`resource_ids/numbered.rs`), held
    /// while it lives.
    _samplers: Vec<Arc<Numbered>>,
    program: Arc<Program>,
}

impl std::ops::Deref for StateShared {
    type Target = Program;
    fn deref(&self) -> &Program {
        &self.program
    }
}

/// The program's release of a state made with `supportIndirectCommandBuffers`: each buffer whose
/// records the GPU may write keeps what it runs (`released_programs.rs`).
impl Drop for StateShared {
    fn drop(&mut self) {
        let Some(id) = self.id.as_ref().filter(|id| id.id().0 != 0) else {
            return;
        };
        let program = &self.program;
        program
            .device
            .shared()
            .keepers
            .released(id.id().0, id.serial(), || {
                KeptProgram::Compute(ComputeProgram(Arc::clone(program)))
            });
    }
}

/// What a compute state runs and answers, but its ID.
pub(super) struct Program {
    device: Device,
    /// The kernel: its function with the constant values it was made with, and the functions it
    /// links.
    function: FunctionDescriptor,
    /// The descriptor the state was made from, for its fields other than the kernel (the stage
    /// input descriptor, the threadgroup limit, and the rest), which every variant compiles under.
    descriptor: Kernel,
    /// That descriptor with its kernel or tile function as its creation compiled it (`base`'s,
    /// `StageSource::Compiled`), whose compile kept what the pipeline's reflection states of it
    /// ([`ComputePipelineState::compiled_reflection`]).
    compiled: Kernel,
    /// The variant compiled at creation: under the empty key, or under stand-ins for the storage
    /// textures' formats it needs (`stand_in`).
    base: Arc<Variant>,
    /// `base` was compiled under stand-ins: it answers Metal's queries and plans dispatches,
    /// which run on the variant their plans name, never on it.
    stand_in: bool,
    /// The other variants dispatches needed.
    others: Mutex<Lru<VariantKey, Arc<Variant>>>,
    /// The compile layer refused the kernel a flat variant ([`ComputePipelineState::settle_flat`]),
    /// so its indirect dispatches take the host step.
    flat_refused: AtomicBool,
    /// The state has no flat device-replay variant whose plan binds no descriptor
    /// ([`ComputePipelineState::replay_variant`]), so an execution of an indirect command
    /// buffer that may run it takes the host step.
    replay_refused: AtomicBool,
    /// The compile layer refused the kernel a folded variant
    /// ([`ComputePipelineState::settle_folded`]), so its indirect threadgroup dispatches take the
    /// host step where no flat variant runs them.
    folded_refused: AtomicBool,
    /// How many of the kernel's last `binary_functions` were added without its descriptor's
    /// `support_adding_binary_functions`
    /// ([`ComputePipelineState::new_compute_pipeline_state_with_additional_binary_functions`]):
    /// their handles call the kernel's last other callable function (`function_table.rs`).
    added_unsupported: usize,
    /// `label`: its descriptor's (`PipelineOptions::label`).
    label: Option<String>,
    /// What its pool pieces hold (`piece_holds.rs`): its dispatches' makings
    /// (`dispatch_making.rs`), kept here so a pool thread never holds the device's last
    /// reference.
    pieces: Arc<piece_holds::PieceHolds<dispatch_making::RegionHeld>>,
    /// Its buffer slots whose bind offset class tells the device's view base alignment apart
    /// (`argument_table::AtBindViews`), of its base's interface.
    at_bind_views: AtBindViews,
    /// What its last direct dispatch that planned planned ([`Planned`]).
    planned: Mutex<Option<Arc<Planned>>>,
}

/// What a direct dispatch of a compute state planned (`compute::settle_dispatch`): the variant
/// `dispatch` of a bind state of `shape` runs on in `forms`, and its plan for `shape`, as
/// [`ComputePipelineState::settle`] answers them, a function of those three alone; and the
/// `VkPipeline`s of its regions of those keys, where all were found made. A later dispatch of the
/// same dispatch and forms whose bind state has the shape takes them, from any encoder, rather
/// than settling them again (`WorkCounts::dispatches_planned_as_before`).
pub(crate) struct Planned {
    pub(crate) shape: Arc<BindShape>,
    pub(crate) dispatch: Dispatch,
    pub(crate) forms: BufferForms,
    pub(crate) variant: Arc<Variant>,
    pub(crate) plan: Arc<BindingPlan>,
    /// Each region's key and its made `VkPipeline`, where all were found made, each its key's own
    /// (none a residual form's, `dispatch_making.rs`).
    pub(crate) made: OnceLock<(Vec<RegionKey>, Vec<Arc<RawPipeline>>)>,
}

impl Planned {
    /// The made `VkPipeline`s of `commands`' regions, where they are as many, each of the key of
    /// one found made.
    pub(crate) fn made_for(&self, commands: &DispatchCommands) -> Option<&Vec<Arc<RawPipeline>>> {
        let (keys, made) = self.made.get()?;
        let same = keys.len() == commands.regions.len()
            && keys
                .iter()
                .zip(&commands.regions)
                .all(|(key, region)| key.is_of(region));
        same.then_some(made)
    }
}

/// The program's release returns at once: a piece not started stops, and a piece running gives
/// back to the device's reaper (`piece_holds.rs`), as a render state's.
impl Drop for Program {
    fn drop(&mut self) {
        self.pieces.close(&self.device.shared().reaper);
    }
}

impl std::fmt::Debug for ComputePipelineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipelineState")
            .field("function", &self.0.function.function.info().name)
            .finish()
    }
}

impl PartialEq for ComputePipelineState {
    /// The same state object (Metal's identity: one made state and its clones, and the state a
    /// command runs once the program released it, `ComputeProgram::state`), not an equal
    /// descriptor: two states made from one descriptor are two states.
    fn eq(&self, other: &ComputePipelineState) -> bool {
        Arc::ptr_eq(&self.0.program, &other.0.program)
    }
}

impl Eq for ComputePipelineState {}

/// How many times a compute creation has answered `supportIndirectCommandBuffers`'s
/// direct-argument check from the **made** descriptor's AIR rather than from the kernel its own
/// compile produced ([`direct_arguments_from_air`]).
///
/// The check is a count and not a time. A made descriptor's kernel is a `Function` stage, so
/// asking it decodes the kernel's AIR -- a second decode of what the compile that just finished
/// already kept. Asking the compiled kernel decodes nothing
/// (`ComputePipelineDescriptor::uses_direct_arguments`), and this counts the creations that had to
/// fall back because a compile kept no answer: an ordinary creation must add none
/// (`tests::direct_arguments_from_compiled`). A test reads it as a difference across the creation
/// it makes, and nextest gives each test its own process, so no other test's creations are in it.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
static DIRECT_ARGUMENTS_FROM_AIR: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Count one fall back to the made descriptor's AIR ([`DIRECT_ARGUMENTS_FROM_AIR`]).
fn note_direct_arguments_from_air() {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    DIRECT_ARGUMENTS_FROM_AIR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// [`DIRECT_ARGUMENTS_FROM_AIR`].
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn direct_arguments_from_air() -> u64 {
    DIRECT_ARGUMENTS_FROM_AIR.load(std::sync::atomic::Ordering::Relaxed)
}

/// What a compute state compiles: a compute descriptor's kernel, or a tile descriptor's tile
/// function (a tile pipeline's state, `tile.rs`), with the descriptor's other fields.
#[derive(Clone)]
pub(super) enum Kernel {
    Compute(ComputePipelineDescriptor),
    Tile(TileRenderPipelineDescriptor),
}

impl Kernel {
    /// The kernel or tile function.
    fn stage(&self) -> &StageSource {
        match self {
            Kernel::Compute(descriptor) => &descriptor.kernel,
            Kernel::Tile(descriptor) => &descriptor.tile,
        }
    }

    /// The descriptor with `stage` its kernel or tile function.
    fn with_stage(&self, stage: StageSource) -> Kernel {
        match self {
            Kernel::Compute(descriptor) => {
                let mut descriptor = descriptor.clone();
                descriptor.kernel = stage;
                Kernel::Compute(descriptor)
            }
            Kernel::Tile(descriptor) => {
                let mut descriptor = descriptor.clone();
                descriptor.tile = stage;
                Kernel::Tile(descriptor)
            }
        }
    }

    /// The lowering inputs of the function's variant `key`: a compute descriptor's stage input
    /// descriptor, or a tile descriptor's color attachment formats by index and its
    /// `threadgroupSizeMatchesTileSize`, and either's `maxCallStackDepth` past Metal's default, 1,
    /// as the compile layer compiles a `Function` stage of each.
    fn inputs(&self, key: &VariantKey) -> metal2vulkan::pipeline::LoweringInputs {
        let mut inputs = key.lowering_inputs(Stage::Kernel);
        match self {
            Kernel::Compute(descriptor) => {
                inputs.stage_input = descriptor.stage_input.clone();
                inputs.max_call_stack_depth =
                    Some(descriptor.max_call_stack_depth).filter(|&k| k > 1);
            }
            Kernel::Tile(descriptor) => {
                inputs.tile_attachments = Some(
                    (0u32..)
                        .zip(&descriptor.color_attachment_formats)
                        .filter_map(|(index, format)| Some((index, (*format)?)))
                        .collect(),
                );
                inputs.threadgroup_size_matches_tile_size =
                    descriptor.threadgroup_size_matches_tile_size;
                inputs.max_call_stack_depth =
                    Some(descriptor.max_call_stack_depth).filter(|&k| k > 1);
            }
        }
        inputs
    }

    /// `pipeline`, this descriptor's compile with its kernel or tile function compiled under
    /// `inputs` ([`Kernel::inputs`]), held to the device's compiler word bound
    /// (`fit_compute_pipeline`, `fit_tile_pipeline`): translated again from the function this
    /// descriptor holds, with the exact forms dropped, where its module passes the bound.
    fn fit(
        &self,
        compile: &metal2vulkan::device::Device,
        inputs: &metal2vulkan::pipeline::LoweringInputs,
        budget: &metal2vulkan::Budget,
        pipeline: ComputePipeline,
    ) -> Result<ComputePipeline, metal2vulkan::error::Error> {
        match self {
            Kernel::Compute(descriptor) => {
                compile.fit_compute_pipeline(descriptor, inputs, budget, pipeline)
            }
            Kernel::Tile(descriptor) => {
                compile.fit_tile_pipeline(descriptor, inputs, budget, pipeline)
            }
        }
    }
}

/// One compiled variant of a state.
pub(super) struct Variant {
    key: VariantKey,
    /// Shared with the execution records of the dispatches that ran it (`execution.rs`).
    pipeline: Arc<ComputePipeline>,
    objects: Arc<VariantObjects>,
    plans: Mutex<Lru<BindShape, Arc<BindingPlan>>>,
    pipelines: making::Making<RegionKey, RawPipeline>,
}

/// What a region's `VkPipeline` is made from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct RegionKey {
    specialization: SpecializationKey,
    stage_flags: metal2vulkan::vk::PipelineShaderStageCreateFlags,
    required_subgroup_size: Option<u32>,
}

impl Device {
    /// `newComputePipelineStateWithFunction:options:reflection:error:` with no reflection:
    /// [`Device::new_compute_pipeline_state`] of a descriptor with `function` its kernel and
    /// every other field Metal's default, under `options` (`PipelineOptions::default()` is
    /// Metal's `newComputePipelineStateWithFunction:error:`). A function of another device is
    /// refused (`Invalid(ForeignObject)`).
    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &Function,
        options: &PipelineOptions,
    ) -> Result<ComputePipelineState, Error> {
        if !function.device().same(self) {
            return Err(invalid(
                "new_compute_pipeline_state_with_function",
                Problem::ForeignObject,
            ));
        }
        self.new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            options,
        )
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` with no reflection:
    /// compile `descriptor` into a pipeline state under `options`. Its kernel is a function's
    /// descriptor ([`Function::function_descriptor`]), with the functions it links
    /// (`FunctionDescriptor::with_linked` of [`Function::linked_function`]s,
    /// `MTLComputePipelineDescriptor.linkedFunctions`); every variant compiles under the
    /// descriptor's other fields. A kernel the compile layer already compiled
    /// (`StageSource::Compiled`) is refused (`Invalid(CompiledKernel)`): the state compiles its
    /// kernel again for each variant a dispatch needs. So is a name `functions` or
    /// `binary_functions` lists twice, or both lists hold for two different functions
    /// (`Invalid(DuplicateLinkedFunction)`); one function listed in both is linked once, with one
    /// handle, as plain Metal takes it where its API validation stops it. So is an archive of
    /// another device (`Invalid(ForeignObject)`). Under `FAIL_ON_BINARY_ARCHIVE_MISS`, a pipeline
    /// no listed archive holds is `BinaryArchiveMiss`. What the compile refuses is `Compile` or
    /// `Unsupported`: constant values that do not fit the function's constants, and a call to a
    /// visible function no linked function supplies, among it. A kernel that sets a render state on
    /// a render command is `Unsupported{IndirectRenderState}` with `Needs::MetalGpuFamily` of
    /// Apple10 on a device that does not claim Apple10, as Metal's compiler refuses it on an M4 and
    /// an M3 Max, and `NotImplemented(IndirectRenderCommands)` on one that does.
    ///
    /// The symbols the kernel imports come from its dynamic libraries: those its descriptor
    /// preloads (`FunctionDescriptor::libraries`, `preloadedLibraries`, of
    /// [`DynamicLibrary::preloaded_library`](crate::DynamicLibrary::preloaded_library)), then
    /// those its libraries' install names name, which the creation loads from their files as
    /// Metal does. An import that survives specialization and that no library supplies is
    /// `Compile(NeedsInput(DynamicSymbol))`, and `Compile(NeedsInput(DynamicLibrary))` where an
    /// install name loaded nothing, as Metal's creation fails ("Undefined symbols").
    pub fn new_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<ComputePipelineState, Error> {
        ComputePipelineState::create(self, descriptor, options, None)
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` with reflection:
    /// [`Device::new_compute_pipeline_state`], and the pipeline's reflection when `options` asks
    /// for it (any word but `NONE`, which Metal answers alike), `None`
    /// when it does not, as Metal leaves `reflection` nil. The reflection is
    /// [`ComputePipelineDescriptor::reflection`]'s, under the creation's budget; what it refuses
    /// refuses the call (`Unsupported`, `Compile`).
    pub fn new_compute_pipeline_state_with_reflection(
        &self,
        descriptor: &ComputePipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<(ComputePipelineState, Option<ComputePipelineReflection>), Error> {
        let state = self.new_compute_pipeline_state(descriptor, options)?;
        // The descriptor the state was made from, with the dynamic libraries its creation loaded
        // (`dynamic_library.rs`).
        let made = match &state.0.descriptor {
            Kernel::Compute(made) => made,
            Kernel::Tile(_) => return Err(internal("a compute state of a tile descriptor")),
        };
        let reflection = match options.options.reflects() {
            true => Some(state.kernel_reflection(made, &options.budget())?),
            false => None,
        };
        Ok((state, reflection))
    }

    /// `newComputePipelineStateWithFunction:options:reflection:error:` with reflection:
    /// [`Device::new_compute_pipeline_state_with_reflection`] of a descriptor with `function` its
    /// kernel and every other field Metal's default. A function of another device is refused
    /// (`Invalid(ForeignObject)`).
    pub fn new_compute_pipeline_state_with_function_and_reflection(
        &self,
        function: &Function,
        options: &PipelineOptions,
    ) -> Result<(ComputePipelineState, Option<ComputePipelineReflection>), Error> {
        if !function.device().same(self) {
            return Err(invalid(
                "new_compute_pipeline_state_with_function_and_reflection",
                Problem::ForeignObject,
            ));
        }
        self.new_compute_pipeline_state_with_reflection(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            options,
        )
    }
}

impl ComputePipelineState {
    /// The state `descriptor` describes, made on `device` under `options`; a binary archive's
    /// add notes what the creation used in `used`. The kernel's dynamic libraries are loaded
    /// first (`dynamic_library.rs`), and the state keeps the descriptor with them.
    pub(crate) fn create(
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
        options: &PipelineOptions,
        used: Option<&Used>,
    ) -> Result<ComputePipelineState, Error> {
        ComputePipelineState::create_adding(device, descriptor, options, used, 0)
    }

    /// [`create`](Self::create), of a descriptor whose kernel's last `added_unsupported`
    /// `binary_functions` were added without its `support_adding_binary_functions`.
    pub(super) fn create_adding(
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
        options: &PipelineOptions,
        used: Option<&Used>,
        added_unsupported: usize,
    ) -> Result<ComputePipelineState, Error> {
        let mut descriptor = descriptor.clone();
        descriptor.kernel = crate::dynamic_library::stage_with_dependencies(&descriptor.kernel);
        ComputePipelineState::of_kernel(
            device,
            Kernel::Compute(descriptor),
            options,
            used,
            "new_compute_pipeline_state",
            added_unsupported,
        )
    }

    /// The state of `kernel`, a compute or tile descriptor, made on `device` under `options` in
    /// `call`; a binary archive's add notes what the creation used in `used`. Its function's
    /// checks and its archive key are its descriptor kind's. The kernel's last
    /// `added_unsupported` `binary_functions` were added without its support flag.
    pub(super) fn of_kernel(
        device: &Device,
        kernel: Kernel,
        options: &PipelineOptions,
        used: Option<&Used>,
        call: &'static str,
        added_unsupported: usize,
    ) -> Result<ComputePipelineState, Error> {
        let StageSource::Function(function) = kernel.stage() else {
            return Err(invalid(call, Problem::CompiledKernel));
        };
        if let Some(name) = duplicate_linked_name(&function.linked) {
            return Err(invalid(call, Problem::DuplicateLinkedFunction { name }));
        }
        additions::check_kernel(device, &kernel, function, call)?;
        // A tile pipeline's descriptor has no `supportIndirectCommandBuffers`, and no
        // `resourceIndex`.
        let indirect = match &kernel {
            Kernel::Compute(descriptor) => descriptor.support_indirect_command_buffers,
            Kernel::Tile(_) => false,
        };
        let placement = if indirect {
            pipeline_placement(call, options.placement)?
        } else {
            IdPlacement::Any
        };
        match &kernel {
            Kernel::Compute(descriptor) => binary_archive::check(
                device,
                options,
                call,
                ArchivedPipeline::Compute,
                binary_archive::compute_missing(descriptor),
            ),
            Kernel::Tile(descriptor) => binary_archive::check(
                device,
                options,
                call,
                ArchivedPipeline::Tile,
                binary_archive::tile_missing(descriptor),
            ),
        }?;
        let sources = Sources::creation(device, options, used);
        let profile = device.compile_device().profile();
        let ((found, compiled), key) = stand_in::compile_with_stand_ins(profile, |key| {
            Variant::compiled(&sources, function, &kernel, key)
        })
        .map_err(|error| compile_error(device, error))?;
        // Metal answers nil with an `NSError` for a compute state that supports indirect command
        // buffers whose kernel uses a direct texture or sampler (`indirect_arguments.rs`).
        // Asked of the kernel the compile above produced, which answers from what that compile
        // kept and decodes nothing; the made descriptor's kernel would decode its AIR again, for
        // an answer the compile that just finished already holds. Where the compile kept none the
        // made descriptor answers, as `kernel_reflection` falls back for a reflection.
        if let (true, Kernel::Compute(made), Kernel::Compute(descriptor)) =
            (indirect, &kernel, &compiled)
        {
            let budget = options.budget();
            let uses = match descriptor.uses_direct_arguments(&budget) {
                Err(metal2vulkan::error::Error::Unsupported(u))
                    if u.feature == Feature::Reflection =>
                {
                    note_direct_arguments_from_air();
                    made.uses_direct_arguments(&budget)
                }
                answer => answer,
            };
            indirect_arguments::check_compute(call, uses)?;
        }
        let stand_in = !key.storage_images.is_empty();
        let base = Variant::of(device, key, found)?;
        // The device's neutrals the base's plan for nothing bound may bind (an unbound texture,
        // sampler or texel slot's), made here, where the creation may wait, so that no dispatch's
        // encoding makes one (`neutral.rs`; `RenderPipelineState::make_neutrals`). One that fails
        // is left to the dispatch that binds it. And room kept in the memory the encodings place
        // their own resources in (`Device::keep_room`).
        if !crate::exit_guard::in_host_step() {
            if let Ok(plan) = base.plan(device, &BindShape::default()) {
                let _ = crate::binding::make_neutrals(device, &plan);
            }
            device.keep_room();
        }
        // Its first dispatch's predicted `VkPipeline`, made where Metal compiles
        // (`dispatch_making.rs`). A stand-in compile is never dispatched.
        if let (Kernel::Compute(descriptor), false) = (&kernel, stand_in) {
            base.make_at_creation(device, descriptor);
        }
        device.degrade_absent_reads(&base.shared_pipeline().dependencies, |_| {
            function.function.info().name.to_string()
        });
        // The kernel's or tile function's samplers, numbered as Metal's creation numbers them.
        let numbered = &base.shared_pipeline().interface.numbered_samplers;
        let samplers = number(
            &device.shared().resource_ids,
            &[NumberedStage {
                function: Some(function_name(&function.function)),
                device: false,
                samplers: numbered
                    .first()
                    .map_or(&[][..], |(_, samplers)| samplers.as_slice()),
            }],
        );
        let identifier =
            UniqueIdentifier::state(&device.shared().identifiers, StateKind::Compute, indirect)?;
        let at_bind_views = AtBindViews::new(device, &base.shared_pipeline().interface);
        Ok(ComputePipelineState(Arc::new_cyclic(|state| {
            // The state holds its ID's slot, which lists it among the states an indirect
            // command's record can name (`indirect_states`).
            let id = indirect.then(|| {
                let id = IdHold::take(&device.shared().resource_ids, IdSpace::Pipeline, placement);
                id.hold_state(
                    Holder::Pipeline(Weak::clone(state)),
                    StateKind::Compute,
                    identifier.value(),
                );
                id
            });
            let program = Program {
                device: device.clone(),
                function: function.clone(),
                descriptor: kernel.clone(),
                compiled,
                base,
                stand_in,
                others: Mutex::new(Lru::counted(VARIANTS)),
                flat_refused: AtomicBool::new(false),
                replay_refused: AtomicBool::new(false),
                folded_refused: AtomicBool::new(false),
                added_unsupported,
                label: options.label.clone(),
                pieces: piece_holds::PieceHolds::new(),
                at_bind_views,
                planned: Mutex::new(None),
            };
            StateShared {
                id,
                identifier: Some(identifier),
                _samplers: samplers,
                program: Arc::new(program),
            }
        })))
    }
}

/// The first name `linked` repeats where Metal's pipeline creation fails or was not measured: a
/// name `functions` lists twice ("error: duplicated symbol", without validation too), or
/// `binary_functions` twice, and a name both lists hold for two different functions (another
/// AIR, name or constant values), in that order. Metal's API validation stops every repeat ("All
/// linked functions must have unique names. Function 'add_one' is a duplicate symbol in
/// binaryFunctions, symbol first found in functions"), but plain Metal takes one function listed
/// in both lists, with one handle (a probe, an M3 and an M4), as the compile layer links it once.
/// Whether validation also counts `private_functions` and `groups` was not measured, so they are
/// left to it.
pub(super) fn duplicate_linked_name(linked: &LinkedFunctions) -> Option<String> {
    fn name(f: &LinkedFunction) -> &str {
        f.function.info().name
    }
    fn repeated(list: &[LinkedFunction]) -> Option<&str> {
        let mut seen = HashSet::new();
        list.iter().map(name).find(|n| !seen.insert(*n))
    }
    let same = |a: &LinkedFunction, b: &LinkedFunction| {
        name(a) == name(b)
            && a.function.info().air_hash == b.function.info().air_hash
            && a.constants == b.constants
    };
    repeated(&linked.functions)
        .or_else(|| repeated(&linked.binary_functions))
        .or_else(|| {
            linked
                .binary_functions
                .iter()
                .find(|b| {
                    linked
                        .functions
                        .iter()
                        .any(|f| name(f) == name(b) && !same(f, b))
                })
                .map(name)
        })
        .map(str::to_owned)
}

impl ComputePipelineState {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// The kernel's name.
    pub(super) fn function_name(&self) -> &str {
        self.0.function.function.info().name
    }

    /// The functions the kernel links (`MTLComputePipelineDescriptor.linkedFunctions`).
    pub(super) fn linked_functions(&self) -> &metal2vulkan::pipeline::LinkedFunctions {
        &self.0.function.linked
    }

    /// How many of the kernel's last `binary_functions` were added without its descriptor's
    /// `support_adding_binary_functions`.
    pub(super) fn added_unsupported(&self) -> usize {
        self.0.added_unsupported
    }

    /// The compute descriptor's stage-input descriptor (`stageInputDescriptor`), which the
    /// kernel's `[[stage_in]]` attributes are fetched by; `None` for none, and for a tile
    /// function.
    pub(crate) fn stage_input(&self) -> Option<&StageInputDescriptor> {
        match &self.0.descriptor {
            Kernel::Compute(descriptor) => descriptor.stage_input.as_ref(),
            Kernel::Tile(_) => None,
        }
    }

    /// Whether the state is a tile pipeline's tile function, not a compute pipeline's kernel.
    pub(super) fn is_tile(&self) -> bool {
        matches!(self.0.descriptor, Kernel::Tile(_))
    }

    /// The slots the kernel reads, as its base variant states them: which buffer slot is a
    /// visible-function table comes from the AIR, so every variant states it alike.
    pub(crate) fn interface(&self) -> &PipelineInterface {
        &self.0.base.pipeline().interface
    }

    /// Its buffer slots whose bind offset class tells the device's view base alignment apart
    /// (`argument_table::AtBindViews`).
    pub(crate) fn at_bind_views(&self) -> &AtBindViews {
        &self.0.at_bind_views
    }

    /// The layout of the kernel's base variant: which texture slot each sampler samples comes
    /// from the AIR, so every variant pairs them alike.
    pub(crate) fn base_layout(&self) -> &PipelineLayout {
        &self.0.base.pipeline().layout
    }

    /// `maxTotalThreadsPerThreadgroup`.
    pub fn max_total_threads_per_threadgroup(&self) -> u32 {
        self.0
            .base
            .pipeline
            .metal_info
            .max_total_threads_per_threadgroup
    }

    /// `maxTotalThreadsPerThreadgroup` on Apple's GPUs, before this device's limits
    /// (`ComputePipelineMetalInfo::metal_max_total_threads_per_threadgroup`): past it a dispatch
    /// runs nothing, as plain Metal's does.
    pub(crate) fn metal_max_total_threads_per_threadgroup(&self) -> u32 {
        self.0
            .base
            .pipeline
            .metal_info
            .metal_max_total_threads_per_threadgroup
    }

    /// `threadExecutionWidth` (32).
    pub fn thread_execution_width(&self) -> u32 {
        self.0.base.pipeline.metal_info.thread_execution_width
    }

    /// `staticThreadgroupMemoryLength`.
    pub fn static_threadgroup_memory_length(&self) -> u32 {
        self.0
            .base
            .pipeline
            .metal_info
            .static_threadgroup_memory_length
    }

    /// Whether the kernel fetches `[[stage_in]]` attributes at the stride the encoder binds
    /// buffer `index` with: a layout of `MTLBufferLayoutStrideDynamic`
    /// (`RuntimeFact::StageInStride`).
    pub(super) fn reads_stage_in_stride(&self, index: u32) -> bool {
        self.0
            .base
            .pipeline
            .dependencies
            .iter()
            .any(|d| d.fact == RuntimeFact::StageInStride { layout: index })
    }

    /// The size of the explicit imageblock a tile function aliases onto its implicit one
    /// (`RuntimeFact::AliasedImageblock`); `None` without one.
    pub(super) fn aliased_imageblock_bytes(&self) -> Option<u32> {
        self.0
            .base
            .pipeline
            .dependencies
            .iter()
            .find_map(|d| match d.fact {
                RuntimeFact::AliasedImageblock { data_size } => Some(data_size),
                _ => None,
            })
    }

    /// The color attachments a tile function reaches through its implicit imageblock, staged as
    /// storage images (`RuntimeFact::TileAttachmentFormat`), by index.
    pub(super) fn tile_attachments_reached(&self) -> impl Iterator<Item = u32> + '_ {
        self.0
            .base
            .pipeline
            .dependencies
            .iter()
            .filter_map(|d| match d.fact {
                RuntimeFact::TileAttachmentFormat { color, .. } => Some(color),
                _ => None,
            })
    }

    /// The size and alignment of a tile function's explicit imageblock that stands alone, whose
    /// cells are the pass's imageblock storage (`RuntimeFact::TileImageblock`); `None` without
    /// one.
    pub(super) fn tile_imageblock(&self) -> Option<(u32, u32)> {
        self.0
            .base
            .pipeline
            .dependencies
            .iter()
            .find_map(|d| match d.fact {
                RuntimeFact::TileImageblock {
                    data_size,
                    alignment,
                } => Some((data_size, alignment)),
                _ => None,
            })
    }

    /// `imageblockMemoryLengthForDimensions:` (`imageblock_memory_length`): 0 for a kernel
    /// without an explicit imageblock.
    pub fn imageblock_memory_length_for_dimensions(&self, dimensions: Size) -> u64 {
        imageblock_memory_length(self.imageblock_cell_bytes(), dimensions)
    }

    /// The bytes of a cell of the kernel's explicit imageblock, or of a tile function's implicit
    /// one; `None` without one.
    pub(super) fn imageblock_cell_bytes(&self) -> Option<u32> {
        self.0
            .base
            .pipeline
            .dependencies
            .iter()
            .find_map(|d| match d.fact {
                RuntimeFact::ImageblockSize { cell_bytes } => Some(cell_bytes),
                _ => None,
            })
    }

    /// Whether the kernel reads texture handles from memory (its layout has a handle table): only
    /// then does a dispatch reach a resident heap's textures.
    pub(super) fn reads_texture_handles(&self) -> bool {
        !handle_stages(&self.0.base.pipeline.layout, RenderStages::TILE).is_empty()
    }

    /// The pipeline's reflection, as its creation's out-parameter answers it where the options ask
    /// for one ([`Device::new_compute_pipeline_state_with_reflection`]): a runtime extension, at
    /// any time after the creation. Metal's `reflection` property (macOS 26) answers nil for a
    /// state an `MTLDevice` made; this is the creation's answer. The stage input descriptor's
    /// buffers, then the kernel's arguments, of the entry its constants specialized: read from what
    /// the kernel's compile at the creation kept, with no decode of its AIR. Refused as the
    /// creation's reflection is: an argument whose answer Metal's has not been measured for
    /// (`Unsupported`, `Feature::Reflection`, naming its type, which the kernel's AIR is decoded
    /// again to name).
    pub fn compiled_reflection(&self) -> Result<ComputePipelineReflection, Error> {
        match &self.0.descriptor {
            Kernel::Compute(made) => {
                self.kernel_reflection(made, &metal2vulkan::Budget::standard())
            }
            Kernel::Tile(_) => Err(internal("a compute state of a tile descriptor")),
        }
    }

    /// The reflection of the pipeline of `made`, the descriptor the state was made from (with its
    /// dynamic libraries), from its compiled kernel ([`Program::compiled`]); where that kept no
    /// answer, `made`'s, decoded under `budget`, which names the argument it refuses.
    fn kernel_reflection(
        &self,
        made: &ComputePipelineDescriptor,
        budget: &metal2vulkan::Budget,
    ) -> Result<ComputePipelineReflection, Error> {
        let Kernel::Compute(compiled) = &self.0.compiled else {
            return Err(internal("a compute state of a tile descriptor"));
        };
        match compiled.reflection(budget) {
            Err(metal2vulkan::error::Error::Unsupported(u)) if u.feature == Feature::Reflection => {
                made.reflection(budget).map_err(device_error)
            }
            answer => answer.map_err(device_error),
        }
    }

    /// The tile pipeline's reflection of a tile pipeline's state ([`RenderPipelineState::compiled_reflection`]), as its
    /// compute pipeline's [`compiled_reflection`](Self::compiled_reflection).
    pub(super) fn tile_reflection(
        &self,
        budget: &metal2vulkan::Budget,
    ) -> Result<metal2vulkan::reflection::RenderPipelineReflection, Error> {
        let (Kernel::Tile(made), Kernel::Tile(compiled)) = (&self.0.descriptor, &self.0.compiled)
        else {
            return Err(internal("a tile state of a compute descriptor"));
        };
        match compiled.reflection(budget) {
            Err(metal2vulkan::error::Error::Unsupported(u)) if u.feature == Feature::Reflection => {
                made.reflection(budget).map_err(device_error)
            }
            answer => answer.map_err(device_error),
        }
    }

    /// Whether the kernel may encode, reset or copy indirect commands: a runtime extension, with
    /// no Metal counterpart (`MTLComputePipelineReflection` does not state it). It is the
    /// compiled pipeline's `PipelineInterface::reaches_command_buffers`, read from the function's
    /// AIR, of the function its constants specialized: `true` where it takes a `command_buffer`
    /// (an argument, or a member of an argument buffer it takes at any depth of its structs), and
    /// where its body calls a command operation on any pointer, one it casts from other bytes
    /// included, or makes a call the translator cannot see into. Only a state that answers `true`
    /// can encode, reset or copy indirect commands, so `false` is proven; one may answer `true`
    /// and never do so (a command buffer it takes and does not touch).
    pub fn can_encode_indirect_commands(&self) -> bool {
        self.interface().reaches_command_buffers
    }

    /// What the kernel may write into the indirect command buffers it reaches: a runtime
    /// extension, with no Metal counterpart. It is the compiled pipeline's
    /// `PipelineInterface::command_writes`, read from the function's AIR, of the function its
    /// constants specialized ([`CommandWrites`]: the indirect-command operations its body may
    /// call, and the primitive types its draws may pass, every type where a draw's is not a
    /// constant). A clear bit is proven: the kernel writes no such record field. Empty where
    /// [`can_encode_indirect_commands`](Self::can_encode_indirect_commands) answers `false`.
    pub fn indirect_command_writes(&self) -> CommandWrites {
        self.interface().command_writes
    }

    /// Whether the kernel may write a texture: a runtime extension, with no Metal counterpart
    /// (`MTLComputePipelineReflection` does not state it). It is read from the function's AIR,
    /// of the function its constants specialized (the compiled pipeline's
    /// `PipelineInterface::texture_writing_stages`, from `StageInterface::writes_textures`):
    /// `true` where it takes a texture declared `access::write` or `access::read_write` (an
    /// argument, or a member of an argument buffer it takes at any depth of its structs), and
    /// where its body calls a texture operation that stores a texel (a `write`, an imageblock
    /// slice written to a texture, or an atomic other than a load) through any handle, an
    /// argument buffer's, a heap's or a resource ID included, or makes a call the translator
    /// cannot see into. Stores to the bytes of a buffer a texture views are not texture writes.
    /// `false` is proven: the kernel writes no texture; one may answer `true` and never write
    /// (a writable texture it takes and does not touch). A tile pipeline's state answers its
    /// tile function's as
    /// [`RenderPipelineState::tile_writes_textures`](crate::RenderPipelineState::tile_writes_textures).
    pub fn writes_textures(&self) -> bool {
        !self.interface().texture_writing_stages.is_empty()
    }

    /// `supportIndirectCommandBuffers` (the descriptor's; false for a state made from a
    /// function).
    pub fn support_indirect_command_buffers(&self) -> bool {
        self.0
            .base
            .pipeline
            .metal_info
            .support_indirect_command_buffers
    }

    /// `gpuResourceID`: for a state made with `supportIndirectCommandBuffers`, its placement's ID
    /// (`PipelineOptions::placement`), else the lowest ID free when it was made, from 1, of the
    /// pipeline states' own space, given back when the state goes; 0 for any other state, as Metal
    /// answers (a probe; Metal's render states share the space, and the runtime's join it). It is
    /// what an indirect command's record names the state by.
    pub fn gpu_resource_id(&self) -> ResourceId {
        self.0.id.as_ref().map_or(ResourceId(0), IdHold::id)
    }

    /// The serial of the state's hold of its ID (`IdHold::serial`), for a state that has one:
    /// what tells whether the state was the ID's last holder.
    pub(crate) fn id_serial(&self) -> Option<u64> {
        self.0.id.as_ref().map(IdHold::serial)
    }

    /// `resourceIndex`, Apple's private property: [`gpu_resource_id`](Self::gpu_resource_id)'s
    /// value, 0 for a state made without `supportIndirectCommandBuffers`, as Metal answers (equal
    /// to `gpuResourceID` for compute and render states made with and
    /// without it).
    pub fn resource_index(&self) -> u64 {
        self.gpu_resource_id().0
    }

    /// `uniqueIdentifier`, Apple's private property: a nonzero value the state answers for its
    /// life, no other live state's. A state made with `supportIndirectCommandBuffers` answers 2^36
    /// plus an offset in the device's allocation of such compute states, the last released taken
    /// again first, as Metal's does, at offsets that are not Metal's; any other a value of one
    /// process-wide count past them, where Metal's is an address that differs from run to run.
    /// No state's is measured to be Metal's number: the first answer announces, once a device,
    /// [`Degradation::PlainStateUniqueIdentifier`](crate::Degradation::PlainStateUniqueIdentifier)
    /// for a plain state and
    /// [`Degradation::UnmeasuredUniqueIdentifier`](crate::Degradation::UnmeasuredUniqueIdentifier)
    /// for one with indirect command buffer support. Such a compute state's first identifier is one
    /// step into Metal's range, `2^36 + 0xc0`, no workload's first.
    pub fn unique_identifier(&self) -> u64 {
        match &self.0.identifier {
            Some(identifier) => identifier.answer(self.device(), "a compute pipeline state"),
            None => 0,
        }
    }

    /// Its `uniqueIdentifier` as the runtime holds it, announcing nothing: what an indirect
    /// command that names the state answers.
    pub(crate) fn identifier_value(&self) -> u64 {
        self.0
            .identifier
            .as_ref()
            .map_or(0, UniqueIdentifier::value)
    }

    /// `label`: the label of the descriptor the state was made with
    /// (`PipelineOptions::label`), `None` for nil; a state made with additional binary functions
    /// keeps its original's, as Metal's does.
    pub fn label(&self) -> Option<&str> {
        self.0.label.as_deref()
    }

    /// The states of `device` made with `supportIndirectCommandBuffers` that live now, by ID:
    /// every state an indirect command's record can name.
    pub(super) fn indirect_states(device: &Device) -> Vec<ComputePipelineState> {
        IndirectStates::list(device).live()
    }

    /// The state, named without holding it, as an indirect command names the state set on it:
    /// Metal's command keeps no reference, so the program's last release frees the state's ID.
    pub(crate) fn downgrade(&self) -> WeakComputePipelineState {
        WeakComputePipelineState(Arc::downgrade(&self.0))
    }

    /// What the state runs, held without its ID (`ComputeProgram`).
    pub(crate) fn program(&self) -> ComputeProgram {
        ComputeProgram(Arc::clone(&self.0.program))
    }

    /// What its last direct dispatch that planned planned ([`Planned`]).
    pub(super) fn planned(&self) -> Option<Arc<Planned>> {
        lock(&self.0.planned).clone()
    }

    /// Keep `planned`, a direct dispatch's, in place of what the last planned.
    pub(super) fn keep_planned(&self, planned: Arc<Planned>) {
        *lock(&self.0.planned) = Some(planned);
    }

    /// The variant `dispatch` of a bind state of `shape` runs on in `forms`, and its plan for
    /// `shape`: the first variant's plan, or that of the variant it asks for, compiled when first
    /// needed.
    pub(super) fn settle(
        &self,
        shape: &BindShape,
        dispatch: &Dispatch,
        forms: BufferForms,
    ) -> Result<(Arc<Variant>, Arc<BindingPlan>), Error> {
        let base = Arc::clone(&self.0.base);
        let plan = base.plan(&self.0.device, shape)?;
        let wanted = match forms {
            BufferForms::Bound => plan.dispatch_variant(dispatch).map_err(Error::Bind)?,
            BufferForms::Large => plan
                .dispatch_variant(dispatch)
                .map_err(Error::Bind)?
                .large_buffer_form(&self.0.base.pipeline().dependencies, shape),
            BufferForms::Replay | BufferForms::Merged => {
                return Err(internal("a dispatch in a draw replay's forms"))
            }
        };
        // A stand-in compile is never dispatched: a plan whose bound formats are the stand-ins'
        // takes a variant of its own (the same compile, from the shader cache).
        if wanted == base.key && !self.0.stand_in {
            return Ok((base, plan));
        }
        let variant = self.variant(&wanted)?;
        let plan = variant.plan(&self.0.device, shape)?;
        let settled = plan.dispatch_variant(dispatch).map_err(Error::Bind)?;
        let runs = match forms {
            BufferForms::Bound | BufferForms::Replay | BufferForms::Merged => {
                settled == variant.key
            }
            BufferForms::Large => variant.key.covers(&settled),
        };
        if !runs {
            return Err(internal(
                "the plan of a compiled variant asks for another variant",
            ));
        }
        Ok((variant, plan))
    }

    /// Make the variants a dispatch of a bind state of `shape` in `forms` may run on, whatever
    /// its threadgroup size, each with its plan for `shape`: its plan's variant in that form, and
    /// that variant with emulated simdgroups where a threadgroup that is not whole simdgroups
    /// asks for it. In the large-buffer form every other bind shape of the same bound textures and
    /// samplers settles on one of them; in the bound form, a dispatch of `shape` whose size its
    /// host step reads (`indirect_dispatch.rs`) does.
    pub(super) fn prepare(
        &self,
        shape: &BindShape,
        forms: BufferForms,
    ) -> Result<Vec<(Arc<Variant>, Arc<BindingPlan>)>, Error> {
        let device = &self.0.device;
        let plan = self.0.base.plan(device, shape)?;
        let mut key = match forms {
            BufferForms::Bound => plan.variant.clone(),
            BufferForms::Large => plan
                .variant
                .large_buffer_form(&self.0.base.pipeline().dependencies, shape),
            BufferForms::Replay | BufferForms::Merged => {
                return Err(internal("a dispatch in a draw replay's forms"))
            }
        };
        let mut keys = vec![key.clone()];
        // One thread: a threadgroup no tier runs as whole simdgroups.
        let partial = Dispatch::Threadgroups {
            groups: [1, 1, 1],
            threads_per_group: [1, 1, 1],
        };
        if plan
            .dispatch_variant(&partial)
            .map_err(Error::Bind)?
            .emulated_simdgroups
        {
            key.emulated_simdgroups = true;
            keys.push(key);
        }
        let mut prepared = Vec::with_capacity(keys.len());
        for key in &keys {
            let variant = self.keyed(key)?;
            let plan = variant.plan(device, shape)?;
            prepared.push((variant, plan));
        }
        Ok(prepared)
    }

    /// The flat device-replay variant of the state's kernel (`VariantKey::flat_grid` and
    /// `replay_bindings`), compiled when first needed, and its plan for a bind state of `shape`
    /// (a replay's: no kernel buffer, and the resident set): what an execution of commands a
    /// kernel writes runs on the GPU alone (`gpu_execute.rs`). `None` where the compile
    /// layer refuses the kernel one, `Unsupported{GpuEncodedIndirectCommands}` (as
    /// [`settle_flat`](Self::settle_flat)'s), or its plan binds a descriptor, which no record can
    /// supply on that path: the state remembers either, which no bind shape changes.
    pub(super) fn replay_variant(
        &self,
        shape: &BindShape,
    ) -> Result<Option<(Arc<Variant>, Arc<BindingPlan>)>, Error> {
        if self.0.flat_refused.load(Ordering::Relaxed)
            || self.0.replay_refused.load(Ordering::Relaxed)
        {
            return Ok(None);
        }
        let mut key = VariantKey::new();
        key.flat_grid = true;
        key.replay_bindings = true;
        let variant = match self.variant(&key) {
            Ok(variant) => variant,
            Err(Error::Unsupported(unsupported))
                if unsupported.feature == Feature::GpuEncodedIndirectCommands =>
            {
                self.0.replay_refused.store(true, Ordering::Relaxed);
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let plan = variant.plan(&self.0.device, shape)?;
        // The plan states the flat grid, not the bindings a replay compiles (`VariantKey`).
        let mut planned = plan.variant.clone();
        planned.replay_bindings = true;
        if planned != key {
            return Err(internal(
                "the plan of a flat replay variant asks for another variant",
            ));
        }
        if !plan.descriptors.is_empty() {
            self.0.replay_refused.store(true, Ordering::Relaxed);
            return Ok(None);
        }
        Ok(Some((variant, plan)))
    }

    /// The flat variant (`VariantKey::flat_grid`) an indirect dispatch of a bind state of `shape`
    /// runs on in threadgroups of `local` threads, with its plan for `shape`: the variant the first
    /// plan names for that dispatch, with the flat grid, compiled when first needed
    /// (`indirect_dispatch.rs`). `None` where the compile layer refuses the kernel one,
    /// `Unsupported{GpuEncodedIndirectCommands}`: the kernel observes its threadgroup partition
    /// (`KernelFacts::partition_oblivious`), or the device reaches no memory by device address. The
    /// state remembers the refusal, which no bind shape changes.
    pub(super) fn settle_flat(
        &self,
        shape: &BindShape,
        local: [u32; 3],
    ) -> Result<Option<(Arc<Variant>, Arc<BindingPlan>)>, Error> {
        if self.0.flat_refused.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let dispatch = flat_dispatch(local);
        let plan = self.0.base.plan(&self.0.device, shape)?;
        let mut key = plan.dispatch_variant(&dispatch).map_err(Error::Bind)?;
        key.flat_grid = true;
        let variant = match self.variant(&key) {
            Ok(variant) => variant,
            Err(Error::Unsupported(unsupported))
                if unsupported.feature == Feature::GpuEncodedIndirectCommands =>
            {
                self.0.flat_refused.store(true, Ordering::Relaxed);
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let plan = variant.plan(&self.0.device, shape)?;
        if plan.dispatch_variant(&dispatch).map_err(Error::Bind)? != key {
            return Err(internal(
                "the plan of a flat variant asks for another variant",
            ));
        }
        Ok(Some((variant, plan)))
    }

    /// The folded variant (`VariantKey::folded_groups`) an indirect threadgroup dispatch of a bind
    /// state of `shape`, in threadgroups of `local` threads, runs on, with its plan for `shape`:
    /// the variant the first plan names for that dispatch (emulated simdgroups where its
    /// threadgroup asks for them), with the folded groups, compiled when first needed
    /// (`indirect_dispatch.rs`). `None` where the compile layer refuses the kernel one,
    /// `Unsupported{GpuEncodedIndirectCommands}`: the device reaches no memory by device address,
    /// or the kernel reads a stage-input region. The state remembers the refusal, which no bind
    /// shape or threadgroup size changes.
    pub(super) fn settle_folded(
        &self,
        shape: &BindShape,
        local: [u32; 3],
    ) -> Result<Option<(Arc<Variant>, Arc<BindingPlan>)>, Error> {
        if self.0.folded_refused.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let dispatch = flat_dispatch(local);
        let plan = self.0.base.plan(&self.0.device, shape)?;
        let mut key = plan.dispatch_variant(&dispatch).map_err(Error::Bind)?;
        key.folded_groups = true;
        let variant = match self.variant(&key) {
            Ok(variant) => variant,
            Err(Error::Unsupported(unsupported))
                if unsupported.feature == Feature::GpuEncodedIndirectCommands =>
            {
                self.0.folded_refused.store(true, Ordering::Relaxed);
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let plan = variant.plan(&self.0.device, shape)?;
        if plan.dispatch_variant(&dispatch).map_err(Error::Bind)? != key {
            return Err(internal(
                "the plan of a folded variant asks for another variant",
            ));
        }
        Ok(Some((variant, plan)))
    }

    /// The variant of `key`: the first one where it is that one's (and not a stand-in's), else
    /// compiled when first needed.
    fn keyed(&self, key: &VariantKey) -> Result<Arc<Variant>, Error> {
        if *key == self.0.base.key && !self.0.stand_in {
            return Ok(Arc::clone(&self.0.base));
        }
        self.variant(key)
    }

    /// The variant of `key`, compiled when first needed.
    fn variant(&self, key: &VariantKey) -> Result<Arc<Variant>, Error> {
        let mut others = lock(&self.0.others);
        if let Some(found) = others.get(key) {
            return Ok(Arc::clone(found));
        }
        let variant = Variant::compile(
            &self.0.device,
            &self.0.function,
            &self.0.descriptor,
            key.clone(),
        )?;
        others.insert(key.clone(), Arc::clone(&variant), 1);
        Ok(variant)
    }
}

/// The dispatch a flat variant runs in threadgroups of `local` threads: one indirect region at
/// the group counts its preprocess writes (`BindingPlan::instantiate_dispatch`); a folded variant's
/// likewise, `local` the call's threadgroup size.
pub(super) fn flat_dispatch(local: [u32; 3]) -> Dispatch {
    Dispatch::ThreadgroupsIndirect {
        indirect: IndirectRef::new(0, 0),
        threads_per_group: local,
    }
}

/// The buffer forms a dispatch's variant takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferForms {
    /// Each slot's as its bound buffer asks: a large buffer's slot at its device address, every
    /// other slot a descriptor's.
    Bound,
    /// The large-buffer form (`VariantKey::large_buffer_form`, the product's): every slot a large
    /// buffer would move to its device address at it, whatever buffer is bound, so no buffer a
    /// command binds asks for another variant. The same results for every access inside a slot's
    /// window; outside it, the large buffer's containment: accesses before the bind offset's
    /// aligned block within the buffer are contained, and an out-of-bounds load reads 0 under
    /// `robustBufferAccess` too. On a device without device addresses, as `Bound`.
    Large,
    /// A draw replay's (`render/executions/gpu.rs`): the vertex and fragment functions read every
    /// buffer through the slot table of the replayed command (`VariantKey::replay_bindings`), so
    /// no buffer is bound. A render pipeline's alone.
    Replay,
    /// A draw replay merged into one draw's: `Replay`, the vertex function finding its command
    /// in the merged draw's header (`VariantKey::merged_draws`). A render pipeline's alone.
    Merged,
}

/// The states of a device made with `supportIndirectCommandBuffers` as they were at a moment,
/// held weakly: those a later host step lets a record name, so a state made since is named by none.
pub(super) struct IndirectStates(Vec<Weak<StateShared>>);

impl IndirectStates {
    /// The states of `device` made with `supportIndirectCommandBuffers` now, by ID.
    pub(super) fn list(device: &Device) -> IndirectStates {
        IndirectStates(lock(&device.shared().resource_ids).pipelines())
    }

    /// Those of them that still live. Upgraded with the IDs' lock free: a state's last handle,
    /// dropped, takes it.
    pub(super) fn live(&self) -> Vec<ComputePipelineState> {
        self.0
            .iter()
            .filter_map(Weak::upgrade)
            .map(ComputePipelineState)
            .collect()
    }
}

impl Variant {
    /// The variant of `function`, the kernel or tile function of `descriptor`, under `key` a
    /// dispatch needs: compiled under the standard budget, or found, and its objects made.
    fn compile(
        device: &Device,
        function: &FunctionDescriptor,
        descriptor: &Kernel,
        key: VariantKey,
    ) -> Result<Arc<Variant>, Error> {
        let (found, _) = Variant::compiled(&Sources::later(device), function, descriptor, &key)
            .map_err(|error| compile_error(device, error))?;
        Variant::of(device, key, found)
    }

    /// The pipeline of `descriptor` with `function` its kernel or tile function, under `key`:
    /// the variant of it the memo knows live, or the pipeline `sources` give, each stage found or
    /// compiled under one budget. The function compiles under the descriptor's stage input
    /// descriptor or color attachment formats ([`Kernel::inputs`]), and the pipeline under its
    /// other fields, held to the device's compiler word bound from `descriptor`, which holds the
    /// function ([`Kernel::fit`]); with `descriptor` holding the function as compiled. A refusal is
    /// the compile layer's.
    fn compiled(
        sources: &Sources<'_>,
        function: &FunctionDescriptor,
        descriptor: &Kernel,
        key: &VariantKey,
    ) -> Result<(Remembered<ComputePipeline, Variant>, Kernel), metal2vulkan::error::Error> {
        let compile = sources.device().compile_device();
        let budget = sources.budget();
        let inputs = descriptor.inputs(key);
        let function_key = compile.function_compile_key(function, &inputs);
        let compiled = sources.function(function, &inputs, (&function_key, None), &budget)?;
        let compiled = descriptor.with_stage(StageSource::Compiled(compiled));
        let found = sources.kernel(&compiled, (descriptor, &inputs), &budget)?;
        Ok((found, compiled))
    }

    /// The variant of `key` of what the memo or a compile `found`: the live variant, or one made
    /// of the pipeline, its objects made, which the memo then knows live.
    fn of(
        device: &Device,
        key: VariantKey,
        found: Remembered<ComputePipeline, Variant>,
    ) -> Result<Arc<Variant>, Error> {
        let pipeline = match found {
            Remembered::Live(variant) if variant.key == key => return Ok(variant),
            Remembered::Live(variant) => Arc::clone(&variant.pipeline),
            Remembered::Compiled(pipeline) => pipeline,
        };
        // Before the variant's shader modules, as a render variant's.
        device.degrade_pipeline(CompiledPipeline::Kernel(&pipeline), Occurrence::Made);
        let (objects, kept) = VariantObjects::kept_or_create(
            device,
            kept::Compiled::Compute(pipeline.key),
            &pipeline.layout,
            std::slice::from_ref(&pipeline.stage),
            CompiledPipeline::Kernel(&pipeline),
        )?;
        let variant = Arc::new(Variant {
            key,
            pipeline,
            objects: Arc::new(objects),
            plans: Mutex::new(Lru::counted(PLANS)),
            pipelines: making::Making::counted(PIPELINES),
        });
        for (place, made) in kept {
            if let kept::Place::Region(region) = place {
                variant.pipelines.seed(region, made);
            }
        }
        device.shared().shader_cache.memo.live_compute(&variant);
        Ok(variant)
    }

    /// The compiled pipeline.
    pub(super) fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// The compiled pipeline, shared (an execution record's).
    pub(super) fn shared_pipeline(&self) -> &Arc<ComputePipeline> {
        &self.pipeline
    }

    /// The variant's layouts and module.
    pub(super) fn objects(&self) -> &Arc<VariantObjects> {
        &self.objects
    }

    /// `commands`' regions, each specialization its residual form where `device` reads no uniform
    /// view at its bind offset alone (`render/residual.rs`): what every region of a dispatch or a
    /// tile dispatch of the variant takes before its key is looked up.
    pub(crate) fn keyed_regions(&self, device: &Device, commands: &mut DispatchCommands) {
        if reads_views_at_bind(device) {
            return;
        }
        let constants = &self.pipeline.stage.specialization.constants;
        for region in &mut commands.regions {
            if let Some(residual) = residual_form(constants.iter(), &region.specialization) {
                region.specialization = residual;
            }
        }
    }

    /// Whether the variant is the kernel's emulated-simdgroups one.
    pub(super) fn emulates_simdgroups(&self) -> bool {
        self.key.emulated_simdgroups
    }

    /// The plan for `shape`, from the memo or planned now, both keyed by the shape's projection
    /// for the pipeline (`BindShape::for_plan`: a bound texture's level count only where the
    /// pipeline reads it).
    fn plan(&self, device: &Device, shape: &BindShape) -> Result<Arc<BindingPlan>, Error> {
        let shape = shape.for_plan(&self.pipeline.dependencies);
        let shape = shape.as_ref();
        if let Some(plan) = lock(&self.plans).get(shape) {
            return Ok(Arc::clone(plan));
        }
        let plan = Arc::new(
            self.pipeline
                .plan(device.compile_device(), shape)
                .map_err(Error::Bind)?,
        );
        lock(&self.plans).insert(shape.clone(), Arc::clone(&plan), 1);
        Ok(plan)
    }

    /// The `VkPipeline` of `region`, from the memo or made now, outside the memo's lock
    /// (`making.rs`), for a dispatch that binds `formats` and waits for it: one an execution of
    /// commands or a host step records, or one of an encoder encoded again (an encoder's own
    /// dispatch or tile dispatch hands its making to the pool, `dispatch_making.rs`).
    pub(super) fn region_pipeline(
        &self,
        region: &DispatchRegion,
        formats: crate::degradation::BoundFormats<'_>,
    ) -> Result<Arc<RawPipeline>, Error> {
        let key = RegionKey::of(region);
        let made = self.pipelines.get_or_make(&key, || {
            let made = self
                .objects
                .pipeline(&key, graphics::Notices::Now(Some(formats)))?;
            Ok(made.kept_at(&self.objects, kept::Place::Region(key.clone())))
        })?;
        self.first_use(&made, &key, formats);
        Ok(made)
    }

    /// The value of each of the kernel's specialization constants where `key` specializes it
    /// ([`specialization_values`]): what a dispatch's failed making names
    /// (`FailedPipeline::specialization`).
    pub(super) fn constants(&self, key: &RegionKey) -> Vec<(SpecKey, u64)> {
        let constants = self
            .pipeline
            .stage
            .specialization
            .constants
            .iter()
            .map(|constant| (constant.id, constant.key, constant.default));
        specialization_values(constants, &key.specialization)
    }

    /// Its key (`FailedPipeline::variant`).
    pub(super) fn key(&self) -> &VariantKey {
        &self.key
    }
}

/// The devices whose Metal refuses the render pipeline of a fragment function reading a global
/// `thread_index_in_quadgroup` ("Encountered unlowered function call to
/// air.get_thread_index_in_quadgroup.i32", the M4 adding "Verify device or GPU architecture
/// supports this feature").
const GLOBAL_BUILTIN_IN_FRAGMENT_REFUSED_ON: &[&str] =
    &["Apple M3 Max, macOS 26.7", "Apple M4, macOS 27.0.1"];

/// The runtime error of a compile refusal on `device`: a storage texture's format the creation
/// compile needs and no stand-in gives (`stand_in`) is not built (`NotImplemented(Textures)`); a
/// render state a function sets on an indirect render command (`IndirectRenderState`) is Metal's
/// own refusal on a device that does not claim Apple10, as Metal's compiler refuses it on an M4 and
/// an M3 Max (`Needs::MetalGpuFamily`), and not built on one that does
/// (`NotImplemented(IndirectRenderCommands)`, as the setters of `indirect_command_buffer.rs`); a
/// fragment function's global built-in Metal does not lower (`GlobalBuiltinInFragment`) is Metal's
/// own refusal on a device the measured M3 Max and M4 stand for, one that does not claim Apple10
/// (`Needs::MetalRefuses`, naming them), and stays `NotImplemented` on one that does, which no
/// measurement covers; the rest as the compile layer's device reports it.
fn compile_error(device: &Device, error: metal2vulkan::error::Error) -> Error {
    match error {
        metal2vulkan::error::Error::NeedsInput(RuntimeInput::StorageTextureFormat { .. }) => {
            Error::NotImplemented(RuntimeGap::Textures)
        }
        metal2vulkan::error::Error::Unsupported(mut unsupported)
            if unsupported.feature == Feature::IndirectRenderState =>
        {
            let apple10 = GpuFamily::Apple10.raw();
            if device.supports_family(apple10) {
                return Error::NotImplemented(RuntimeGap::IndirectRenderCommands);
            }
            unsupported.needs = Needs::MetalGpuFamily(apple10);
            Error::Unsupported(unsupported)
        }
        metal2vulkan::error::Error::Unsupported(mut unsupported)
            if unsupported.feature == Feature::GlobalBuiltinInFragment
                && !device.supports_family(GpuFamily::Apple10.raw()) =>
        {
            unsupported.needs = Needs::MetalRefuses(GLOBAL_BUILTIN_IN_FRAGMENT_REFUSED_ON);
            Error::Unsupported(unsupported)
        }
        other => device_error(other),
    }
}
