//! `RenderPipelineState` (`MTLRenderPipelineState`) and `Device::new_render_pipeline_state`. A
//! render pipeline state is one that draws (a `GraphicsState`, this module) or a tile pipeline's
//! (`tile.rs`), as Metal's one protocol is both.
//!
//! - **Creation.** A state compiles its descriptor under the empty variant key and the budget of
//!   its `PipelineOptions`, as Metal compiles at creation, and makes the variant's shader
//!   modules and layouts: what Metal's pipeline creation refuses, this refuses (`Compile`). A
//!   storage texture's format the creation compile needs takes a stand-in, as a compute state's
//!   does (`stand_in`): that compile is planned, never drawn.
//! - **Variants.** A draw's bind shape can ask for another variant (`BindingPlan::variant`: a
//!   vertex buffer fixed-function input cannot fetch as bound, a dropped `Layer` output, storage
//!   textures' formats): the state compiles it under the standard budget when a draw first needs
//!   it.
//! - **The memo, archives and the disk cache.** The linked pipeline, and when it is not found
//!   each `Function` stage's translation, are looked up by their keys (`RenderPipelineKeys`) as a
//!   compute state's are (`sources.rs`): in the device's memo, the creation's binary archives, then
//!   the shader cache on disk, before compiling, and kept after. A pipeline the memo knows a live
//!   variant of is that variant, as a compute state's is.
//! - **Memos.** Each state keeps its variants, each variant its plans by bind shape and its
//!   `VkPipeline`s by graphics key (`graphics.rs`), every memo bounded, the least recently used
//!   going first. Command buffers retain what their draws use, so dropping a memo's entry never
//!   destroys an object a pending command uses.
//! - **The factor-scaling twin**. A tessellating state keeps, once an indirect command's patch draw
//!   first scales its factors, a state of its descriptor with the factor scale enabled and without
//!   `supportIndirectCommandBuffers` (which Metal refuses together), that the command's draw runs
//!   on with the command's scale.
//! - **Neutrals**. A state an indirect command buffer's commands may draw (made with
//!   `supportIndirectCommandBuffers`) makes, at its creation, the device's neutrals its compiled
//!   variants' plans may bind with nothing bound (`binding::make_neutrals`): an execution's host
//!   step may draw it, though it was made after the commit, and a neutral made before is one the
//!   step, a serial point of the command buffers' stream, need not make (`neutral.rs`).
//!
//! The descriptor names the compile layer's functions ([`Function::function_descriptor`] gives
//! one of a runtime function).
//!
//! [`Function::function_descriptor`]: crate::Function::function_descriptor

use super::creation_log::{self, StateLog};
use super::graphics::{DrawKey, GraphicsKey, Notices};
use super::kept::{Compiled, Place};
use super::libraries::Linked;
use super::making::{Begun, Making, Pending, Ticket};
use super::memo::{At, Lru, Remembered};
use super::objects::{words_digest, RawPipeline, VariantObjects};
use super::piece_holds::PieceHolds;
use super::pool::Joined;
use super::sources::Sources;
use super::texture_access::{TextureAccessMemo, TextureAccesses};
use super::tile::TileState;
use super::BufferForms;
use super::NoteEntries;
use super::{compile_error, stand_in, PipelineOptions, PIPELINES, PLANS, VARIANTS};
use crate::argument_table::AtBindViews;
use crate::binary_archive::{self, Used};
use crate::color_reformat::Routing;
use crate::degradation::{BoundFormats, CompiledPipeline, Degradation, Occurrence};
use crate::device::{device_error, lock};
use crate::error::{internal, invalid, ArchivedPipeline, Error, Problem, RuntimeGap};
use crate::released_programs::KeptProgram;
use crate::render_pass::MAX_COLOR_ATTACHMENTS;
use crate::render_target::RenderPassObject;
use crate::residency::{every_render_stage, handle_stages};
use crate::resource_ids::{
    function_name, number, pipeline_placement, Holder, IdHold, IdPlacement, IdSpace, Numbered,
    NumberedStage, ResourceId, StateKind,
};
use crate::shader_cache::VariantStages;
use crate::unique_identifier::UniqueIdentifier;
use crate::values::Size;
use crate::Device;
use metal2vulkan::bind::{BindShape, BindState, BindingPlan, FragmentTests};
use metal2vulkan::cache::{Hash256, PipelineKey, SpecializationKey, VariantKey};
use metal2vulkan::device::PipelineWaitPolicy;
use metal2vulkan::library::{CompiledFunction, FunctionCompileKey};
use metal2vulkan::metal::{BindingAccess, PixelFormat, PrimitiveTopologyClass, RenderStages};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, CommandWrites, MeshPipelineDescriptor, PipelineInterface,
    RenderPipeline, RenderPipelineDescriptor, RuntimeFact, SpecKey, StageEntries, StageModule,
    StageSource, TessellationDescriptor,
};
use metal2vulkan::reflection::{LinkedUseAnswers, RenderPipelineReflection};
use std::any::Any;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;

mod residual;

/// The general form of a key, which `tests::general_form` reads from IDs and meanings alone.
#[cfg(test)]
pub(crate) use residual::general_form;
#[cfg(test)]
pub(crate) use residual::views_read_at_bind;
pub(crate) use residual::{reads_views_at_bind, residual_form};

/// A Metal render pipeline state (`MTLRenderPipelineState`): one that draws, or a tile pipeline's
/// ([`Device::new_tile_render_pipeline_state`]), which dispatches per tile. Cheap to clone (a
/// retain); `Send + Sync`.
#[derive(Clone)]
pub struct RenderPipelineState(Kind);

/// What a render pipeline state is.
#[derive(Clone)]
pub(crate) enum Kind {
    /// A state that draws: a vertex function, and a fragment function where it rasterizes.
    Graphics(GraphicsState),
    /// A tile pipeline's state (`tile.rs`).
    Tile(TileState),
}

/// A render pipeline state that draws.
#[derive(Clone)]
pub(crate) struct GraphicsState(Arc<StateShared>);

/// A drawing state named without holding it ([`RenderPipelineState::downgrade`]).
#[derive(Clone)]
pub(crate) struct WeakRenderPipelineState(Weak<StateShared>);

impl WeakRenderPipelineState {
    /// The state, while the program holds it.
    pub(crate) fn upgrade(&self) -> Option<RenderPipelineState> {
        self.0
            .upgrade()
            .map(|state| RenderPipelineState(Kind::Graphics(GraphicsState(state))))
    }
}

/// What a drawing state runs, held without its ID: an indirect render command keeps the program
/// of the state set on it past the program's last release of the state, as Metal's runs it,
/// while the state's ID goes back (`RenderPipelineState::program`).
#[derive(Clone)]
pub(crate) struct RenderProgram(Arc<Program>);

impl RenderProgram {
    /// A state that runs the program, with no ID (`gpu_resource_id` 0): the one a command runs
    /// once the program released the state set on it. Equal to that state.
    pub(crate) fn state(&self) -> RenderPipelineState {
        RenderPipelineState(Kind::Graphics(GraphicsState(Arc::new(StateShared {
            id: None,
            identifier: None,
            _samplers: Vec::new(),
            program: Arc::clone(&self.0),
        }))))
    }
}

/// A drawing state: its ID, which goes back at its last handle's release, and what it runs.
pub(crate) struct StateShared {
    /// The state's ID, held while it lives: a state made with `supportIndirectCommandBuffers` has
    /// one of the pipeline states' space, which compute states share; any other none.
    id: Option<IdHold>,
    /// Its `uniqueIdentifier` (`unique_identifier.rs`), held while it lives; none for the
    /// stand-in of a released state's program (`RenderProgram::state`).
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
                KeptProgram::Render(RenderProgram(Arc::clone(program)))
            });
    }
}

/// What a drawing state runs and answers, but its ID.
pub(crate) struct Program {
    device: Device,
    descriptor: StateDescriptor,
    /// For a twin of a color group (`GraphicsState::color_group`), the descriptor its group's
    /// passes draw it as (`GraphicsState::descriptor`): the group's color attachments alone,
    /// attachment `colors()[l]` at index `l`, as its variants' outputs and the group's projected
    /// pass are numbered. Its variants compile from `descriptor`, under the group's key.
    drawn: Option<StateDescriptor>,
    /// The variant compiled at creation: under the empty key, or under stand-ins for the storage
    /// textures' formats it needs (`stand_in`).
    base: Arc<RenderVariant>,
    /// `base` was compiled under stand-ins: it plans draws, which run on the variant their plans
    /// name, never on it.
    stand_in: bool,
    /// The other variants draws needed.
    others: Mutex<Lru<VariantKey, Arc<RenderVariant>>>,
    /// The variant in the large-buffer form, compiled at creation for a state that supports
    /// indirect command buffers ([`large_form`]), held while the state lives: what a host step's
    /// draw of a command the GPU may write runs on, which its commit then only looks up.
    large: Option<Arc<RenderVariant>>,
    /// The state's factor-scaling twin (`GraphicsState::factor_scaling`), made when an indirect
    /// command first scales its patches' factors.
    scaling: Mutex<Option<GraphicsState>>,
    /// How many of the vertex function's, then the fragment function's, last `binary_functions`
    /// were added without the stage's support flag (`support_adding_vertex_binary_functions`,
    /// `support_adding_fragment_binary_functions`).
    added_unsupported: [usize; 2],
    /// The state's single-sample twin (`GraphicsState::single_sample`), made when a pass of one
    /// sample first takes the state.
    single_sample: Mutex<Option<GraphicsState>>,
    /// The state's twins of its color groups (`GraphicsState::color_group`), in group order, made
    /// at its creation; none for a state the device draws whole.
    color_groups: OnceLock<Vec<GraphicsState>>,
    /// `label`: its descriptor's (`PipelineOptions::label`).
    label: Option<String>,
    /// Its functions' texture slots and their accesses (`texture_access.rs`), read when first
    /// asked.
    texture_accesses: TextureAccessMemo,
    /// The key of the descriptor it draws as ([`GraphicsState::descriptor_key`]), built when
    /// something first looks an answer up by it.
    ///
    /// Building it digests every field of the descriptor, each in its derived `Debug` form, so it
    /// allocates a dozen strings a build; `descriptor()` is fixed for the state's life, so the
    /// key is, and `compiled_reflection` built it again on every read to find a reflection the
    /// shader cache had already kept. Held in a `OnceLock` behind
    /// [`GraphicsState::descriptor_key`], which is the only way to it: a second build of one
    /// state's key cannot be written through this path, so nothing has to check that it is not.
    descriptor_key: OnceLock<PipelineKey>,
    /// The key of the descriptor its variants compile from ([`GraphicsState::compiled_key`]),
    /// built when a draw first compiles a variant of its own.
    ///
    /// Unset for a state drawn whole, which compiles from the descriptor it draws as: that
    /// state's one key is [`Program::descriptor_key`], and `compiled_key` reads it. A twin of a
    /// color group compiles from the descriptor it was made from and draws as its group's colors
    /// alone (`drawn`), two descriptors the key tells apart, so it holds a key of each: a
    /// compile is keyed by the descriptor it compiles, a question by the descriptor it asks of.
    compiled_key: OnceLock<PipelineKey>,
    /// Its part of the pipeline creation log, while the log is on (`creation_log.rs`).
    log: Option<StateLog>,
    /// What its last [`SETTLED`] settles of other shapes settled, in which buffer forms, the
    /// latest used first (`GraphicsState::settle_shape`).
    settled: Mutex<VecDeque<(BufferForms, Settled)>>,
    /// The `VkPipeline`s its last draws found made in their variants' memos, the latest found
    /// first, at most [`LAST_PIPELINES`] (`RenderVariant::begin_graphics_pipeline`).
    last_pipelines: Mutex<VecDeque<LastPipeline>>,
    /// Its routing into the pass colors it was last set in a pass of (`GraphicsState::routed`).
    routed: Mutex<Option<(PassColors, Result<Arc<Routing>, u32>)>>,
    /// What its pool pieces hold (`piece_holds.rs`): its draws' makings, kept here so a pool
    /// thread never holds the device's last reference.
    pieces: Arc<PieceHolds<PieceHeld>>,
    /// Its buffer slots whose bind offset class tells the device's view base alignment apart
    /// (`argument_table::AtBindViews`), of its base's interface.
    at_bind_views: AtBindViews,
}

/// The state's release returns at once: it lets go of what its pool pieces held on its own
/// thread, a piece not started stops, and a piece running gives back to the device's reaper
/// (`piece_holds.rs`, `reaper.rs`).
impl Drop for Program {
    fn drop(&mut self) {
        self.pieces.close(&self.device.shared().reaper);
    }
}

/// What a state keeps for one of its pool pieces (`piece_holds.rs`): each holds the device.
pub(crate) enum PieceHeld {
    /// A draw's making: its variant, the pass it is made against, and its making's ticket.
    Draw(
        Arc<RenderVariant>,
        Arc<RenderPassObject>,
        Box<Ticket<GraphicsKey, RawPipeline>>,
    ),
    /// What a draw's making gives back once settled, held until let go of: its variant and its
    /// pass.
    Drawn {
        _variant: Arc<RenderVariant>,
        _pass: Arc<RenderPassObject>,
    },
}

/// A pass's color formats, by index: what a state's routing is found for
/// ([`GraphicsState::routed`]).
pub(crate) type PassColors = [Option<PixelFormat>; MAX_COLOR_ATTACHMENTS];

/// A `VkPipeline` a state's draw found made in its variant's memo: the variant (held weakly, so
/// the state holds it no longer than before, and its address stays its own while held), the key,
/// and where the memo's entry lies.
struct LastPipeline {
    variant: Weak<RenderVariant>,
    key: GraphicsKey,
    at: At,
}

/// What a draw of a bind shape settled ([`GraphicsState::settle_shape`]): the shape, which the
/// draw's execution record shares, and the variant and plan [`settle`](GraphicsState::settle)
/// answers for it.
#[derive(Clone)]
pub(crate) struct Settled {
    pub(crate) shape: Arc<BindShape>,
    pub(crate) variant: Arc<RenderVariant>,
    pub(crate) plan: Arc<BindingPlan>,
}

/// The most settles a state keeps (`GraphicsState::settle_shape`). A window server's draws of one
/// state take a few shapes in turn: a pass's first draw binds nothing else, and a later one finds
/// what the draws of other states left bound (their bytes), which is of its shape.
const SETTLED: usize = 4;

/// The most `VkPipeline`s a state keeps of those its draws found made
/// (`RenderVariant::begin_graphics_pipeline`). A window server's draws of one state take a few
/// keys in turn, as they take a few shapes ([`SETTLED`]): a pass's first draw finds its own bind
/// shape, whose plan's specialization another draw of the pass does not share.
const LAST_PIPELINES: usize = 4;

impl std::fmt::Debug for RenderPipelineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Kind::Graphics(state) => f
                .debug_struct("RenderPipelineState")
                .field("key", &state.0.base.pipeline.key)
                .finish(),
            Kind::Tile(state) => f
                .debug_struct("RenderPipelineState")
                .field("tile", &state.kernel())
                .finish(),
        }
    }
}

impl PartialEq for RenderPipelineState {
    /// The same state object (Metal's identity: one made state and its clones), not an equal
    /// descriptor: two states made from one descriptor are two states.
    fn eq(&self, other: &RenderPipelineState) -> bool {
        match (&self.0, &other.0) {
            (Kind::Graphics(a), Kind::Graphics(b)) => a.is(b),
            (Kind::Tile(a), Kind::Tile(b)) => a.is(b),
            (Kind::Graphics(_), Kind::Tile(_)) | (Kind::Tile(_), Kind::Graphics(_)) => false,
        }
    }
}

impl Eq for RenderPipelineState {}

/// What a drawing state was made from: a render pipeline's descriptor, or a mesh render
/// pipeline's (`Device::new_mesh_render_pipeline_state`).
#[derive(Clone)]
pub(crate) enum StateDescriptor {
    Vertex(RenderPipelineDescriptor),
    Mesh(MeshPipelineDescriptor),
}

impl StateDescriptor {
    /// Its functions' names, in pipeline order (`stage_name`).
    fn function_names(&self) -> Vec<String> {
        let stages: Vec<&StageSource> = match self {
            StateDescriptor::Vertex(descriptor) => std::iter::once(&descriptor.vertex)
                .chain(&descriptor.fragment)
                .collect(),
            StateDescriptor::Mesh(descriptor) => descriptor
                .object
                .iter()
                .chain(std::iter::once(&descriptor.mesh))
                .chain(&descriptor.fragment)
                .collect(),
        };
        stages.into_iter().map(stage_name).collect()
    }

    /// A draw of a state of this descriptor at color `index`, in words for a notice: its
    /// fragment function's name and the color's blend state (operation and factors for color
    /// and alpha, or unblended) and write mask, as the descriptor states them.
    pub(crate) fn draw_at(&self, index: u32) -> String {
        let fragment = match self {
            StateDescriptor::Vertex(descriptor) => descriptor.fragment.as_ref(),
            StateDescriptor::Mesh(descriptor) => descriptor.fragment.as_ref(),
        };
        let function = match fragment {
            Some(source) => format!("fragment function `{}`", stage_name(source)),
            None => "no fragment function".to_owned(),
        };
        let color = self
            .color_attachments()
            .get(index as usize)
            .copied()
            .flatten();
        let state = match color {
            Some(a) if a.blending_enabled => format!(
                "color {index} blended: color {:?} {:?} {:?}, alpha {:?} {:?} {:?}, write mask \
                 {:?}",
                a.source_rgb_blend_factor,
                a.rgb_blend_operation,
                a.destination_rgb_blend_factor,
                a.source_alpha_blend_factor,
                a.alpha_blend_operation,
                a.destination_alpha_blend_factor,
                a.write_mask
            ),
            Some(a) => format!("color {index} unblended, write mask {:?}", a.write_mask),
            None => format!("no color {index}"),
        };
        format!("a draw of {function} ({state})")
    }

    /// A render pipeline's descriptor; `None` for a mesh render pipeline's.
    pub(crate) fn vertex(&self) -> Option<&RenderPipelineDescriptor> {
        match self {
            StateDescriptor::Vertex(descriptor) => Some(descriptor),
            StateDescriptor::Mesh(_) => None,
        }
    }

    /// The color attachments' pixel formats by index, `None` where the descriptor has none.
    pub(crate) fn color_formats(&self) -> [Option<PixelFormat>; 8] {
        self.color_attachments().map(|a| a.map(|a| a.pixel_format))
    }

    /// The color attachments by index, `None` where the descriptor has none.
    pub(crate) fn color_attachments(&self) -> [Option<ColorAttachmentDescriptor>; 8] {
        match self {
            StateDescriptor::Vertex(descriptor) => descriptor.color_attachments,
            StateDescriptor::Mesh(descriptor) => descriptor.color_attachments,
        }
    }

    /// `rasterSampleCount`.
    pub(crate) fn raster_sample_count(&self) -> u32 {
        match self {
            StateDescriptor::Vertex(descriptor) => descriptor.raster_sample_count,
            StateDescriptor::Mesh(descriptor) => descriptor.raster_sample_count,
        }
    }

    /// `supportIndirectCommandBuffers`: a render or mesh render pipeline's descriptor's.
    pub(crate) fn support_indirect_command_buffers(&self) -> bool {
        match self {
            StateDescriptor::Vertex(descriptor) => descriptor.support_indirect_command_buffers,
            StateDescriptor::Mesh(descriptor) => descriptor.support_indirect_command_buffers,
        }
    }

    /// The function of `stage`, where the descriptor has one.
    pub(crate) fn stage(&self, stage: metal2vulkan::pipeline::Stage) -> Option<&StageSource> {
        use metal2vulkan::pipeline::Stage;
        match (self, stage) {
            (StateDescriptor::Vertex(d), Stage::Vertex) => Some(&d.vertex),
            (StateDescriptor::Vertex(d), Stage::Fragment) => d.fragment.as_ref(),
            (StateDescriptor::Mesh(d), Stage::Object) => d.object.as_ref(),
            (StateDescriptor::Mesh(d), Stage::Mesh) => Some(&d.mesh),
            (StateDescriptor::Mesh(d), Stage::Fragment) => d.fragment.as_ref(),
            _ => None,
        }
    }

    /// Number the samplers of a state of this descriptor on `device`, as Metal's creation
    /// numbers them (`resource_ids/numbered.rs`): the device's own sampler where it has a
    /// fragment function, then its functions' samplers (`interface`'s, the compiled pipeline's)
    /// in the reverse of its stages, fragment, vertex or mesh, object.
    fn number_samplers(
        &self,
        device: &Device,
        interface: &PipelineInterface,
    ) -> Vec<Arc<Numbered>> {
        use metal2vulkan::pipeline::Stage;
        let stages: Vec<NumberedStage<'_>> =
            [Stage::Fragment, Stage::Vertex, Stage::Mesh, Stage::Object]
                .into_iter()
                .filter_map(|stage| {
                    let source = self.stage(stage)?;
                    let samplers = interface
                        .numbered_samplers
                        .iter()
                        .find(|(of, _)| *of == stage)
                        .map_or(&[][..], |(_, samplers)| samplers.as_slice());
                    Some(NumberedStage {
                        function: match source {
                            StageSource::Function(function) => {
                                Some(function_name(&function.function))
                            }
                            StageSource::Compiled(_) => None,
                        },
                        device: stage == Stage::Fragment,
                        samplers,
                    })
                })
                .collect();
        number(&device.shared().resource_ids, &stages)
    }
}

/// One compiled variant of a render state.
pub(crate) struct RenderVariant {
    key: VariantKey,
    /// Shared with the execution records of the draws that ran it (`execution.rs`).
    pipeline: Arc<RenderPipeline>,
    objects: Arc<VariantObjects>,
    plans: Mutex<Lru<BindShape, Arc<BindingPlan>>>,
    pipelines: Making<GraphicsKey, RawPipeline>,
    /// An emulated mesh pipeline's compute pipelines, by mesh draw specialization.
    mesh_pipelines: Making<SpecializationKey, MeshPipelines>,
    /// The render stages whose functions read texture handles, of its layout
    /// ([`GraphicsState::texture_handle_stages`]): worked out as it is made, not at each draw.
    handle_stages: RenderStages,
    /// What its draws take of its dynamic states ([`DrawDynamic`]), worked out at its first draw.
    draw_dynamic: OnceLock<DrawDynamic>,
    /// Each stage's SPIR-V as the pipeline creation log names it (`creation_log::spirv_hashes`),
    /// where the log is on: worked out beside its objects as it is made
    /// ([`RenderVariant::logged_spirv`]).
    logged_spirv: OnceLock<Box<[u64]>>,
}

/// A render creation's reflection, worked out beside it ([`GraphicsState::create_reflecting`]).
pub(crate) enum Reflected {
    /// The one the shader cache keeps for the descriptor's key.
    Kept(RenderPipelineReflection),
    /// Worked out now for the descriptor of the key, for the caller to keep where it answers.
    Made(
        PipelineKey,
        Result<RenderPipelineReflection, metal2vulkan::error::Error>,
    ),
}

/// What every draw of a variant takes of its dynamic states, a function of its pipeline and its
/// device alone (`render.rs`, `DrawDynamic::of`): worked out at its first draw and kept.
pub(crate) struct DrawDynamic {
    /// The dynamic states its draws record: its own, but for those of a part it lacks.
    pub(crate) states: Vec<ash::vk::DynamicState>,
    /// Whether the encoder records each of them on the device.
    pub(crate) recorded: bool,
}

/// A draw's `VkPipeline` as the draw finds it ([`RenderVariant::begin_graphics_pipeline`]): made,
/// or being made.
pub(crate) enum DrawPipeline {
    Made(Arc<RawPipeline>),
    Making(Pending<RawPipeline>),
}

/// The compute `VkPipeline`s of an emulated mesh pipeline under one mesh draw's specialization:
/// its object function's, where it has one, and its mesh function's.
#[derive(Clone)]
pub(crate) struct MeshPipelines {
    pub(crate) object: Option<Arc<RawPipeline>>,
    pub(crate) mesh: Arc<RawPipeline>,
}

impl Device {
    /// `newRenderPipelineStateWithDescriptor:options:reflection:error:` with no reflection:
    /// compile `descriptor` into a pipeline state under `options` (`PipelineOptions::default()`
    /// is Metal's `newRenderPipelineStateWithDescriptor:error:`). An archive of another device
    /// is refused (`Invalid(ForeignObject)`), and a pipeline no listed archive holds under
    /// `FAIL_ON_BINARY_ARCHIVE_MISS` is `BinaryArchiveMiss`. What the compile refuses is
    /// `Compile` or `Unsupported` (a storage texture's format no stand-in gives is
    /// `NotImplemented(Textures)`). A vertex
    /// function that writes `[[point_size]]` under the line or triangle input primitive topology
    /// class is `Invalid` (`PointSizeTopology`), as Metal's creation answers an error for it
    /// (measured on an M3 Max and an M4). A framebuffer fetch no ordered form serves on the device
    /// is `Unsupported{FramebufferFetch}`, naming what each form needs. Each stage's imports come
    /// from its own dynamic libraries (`vertexPreloadedLibraries` in the vertex stage's
    /// function descriptor, `fragmentPreloadedLibraries` in the fragment's, then those its
    /// libraries' install names name), as a compute state's do
    /// ([`Device::new_compute_pipeline_state`]). A fragment function that reads a program-scope
    /// `thread_index_in_quadgroup` is `Unsupported{GlobalBuiltinInFragment}` with
    /// `Needs::MetalRefuses` (an M3 Max and an M4) on a device that does not claim Apple10, as
    /// Metal's creation refuses it there, and `NotImplemented` on one that does.
    pub fn new_render_pipeline_state(
        &self,
        descriptor: &RenderPipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<RenderPipelineState, Error> {
        GraphicsState::create(self, descriptor, options, None)
            .map(|state| RenderPipelineState(Kind::Graphics(state)))
    }

    /// `newRenderPipelineStateWithDescriptor:options:reflection:error:` with reflection:
    /// [`Device::new_render_pipeline_state`], and the pipeline's reflection when `options` asks
    /// for it (any word but `NONE`, which Metal answers alike), `None`
    /// when it does not, as Metal leaves `reflection` nil. The reflection is
    /// [`RenderPipelineDescriptor::reflection`]'s, under the creation's budget; what it refuses
    /// refuses the call (`Unsupported`, `Compile`).
    pub fn new_render_pipeline_state_with_reflection(
        &self,
        descriptor: &RenderPipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<(RenderPipelineState, Option<RenderPipelineReflection>), Error> {
        let reflects = options.options.reflects();
        let (state, reflected) =
            GraphicsState::create_reflecting(self, descriptor, options, (None, [0, 0]), reflects)?;
        // The descriptor the state was made from, with the dynamic libraries its creation loaded
        // (`dynamic_library.rs`).
        let reflection = match (reflects, state.descriptor().vertex()) {
            (true, Some(descriptor)) => {
                let compile = self.compile_device();
                let cache = &self.shared().shader_cache;
                Some(match reflected {
                    Some(Reflected::Kept(reflection)) => reflection,
                    Some(Reflected::Made(key, made)) => {
                        cache.keep_render_reflection(compile, &key, made.map_err(device_error)?)
                    }
                    // Where no thread could be made: under the state's own key, the one its
                    // creation built for this descriptor.
                    None => cache
                        .render_reflection(compile, state.descriptor_key(), || {
                            descriptor.reflection(&options.budget())
                        })
                        .map_err(device_error)?,
                })
            }
            (true, None) => return Err(internal("a render state made of a mesh descriptor")),
            (false, _) => None,
        };
        Ok((RenderPipelineState(Kind::Graphics(state)), reflection))
    }

    /// `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:` with no reflection:
    /// compile the mesh render pipeline `descriptor` (an object function where it has one, a
    /// mesh function, and a fragment function where it rasterizes) into a pipeline state under
    /// `options`, as the compile layer's `Device::compile_mesh_pipeline` compiles it. Its draws
    /// are the encoder's mesh draws ([`RenderCommandEncoder::draw_mesh_threadgroups`]). A device
    /// without `VK_EXT_mesh_shader`, or without its stage's feature (`meshShader`, `taskShader`
    /// for an object function), is `Unsupported{MeshShading}` naming them: no emulation draws a
    /// mesh function's primitives in Metal's order on it. What the compile refuses otherwise is
    /// `Compile` or `Unsupported`, as [`Device::new_render_pipeline_state`] says, and so is an
    /// archive of another device; no binary archive holds a mesh render pipeline, so
    /// `FAIL_ON_BINARY_ARCHIVE_MISS` with archives listed is `BinaryArchiveMiss`. Each stage's
    /// imports come from its own dynamic libraries, as a render state's do.
    ///
    /// [`RenderCommandEncoder::draw_mesh_threadgroups`]: crate::RenderCommandEncoder::draw_mesh_threadgroups
    pub fn new_mesh_render_pipeline_state(
        &self,
        descriptor: &MeshPipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<RenderPipelineState, Error> {
        GraphicsState::create_mesh(self, descriptor, options)
            .map(|state| RenderPipelineState(Kind::Graphics(state)))
    }

    /// `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:` with reflection:
    /// [`Device::new_mesh_render_pipeline_state`], and the pipeline's reflection when `options`
    /// asks for it (any word but `NONE`, which Metal answers alike),
    /// `None` when it does not, as Metal leaves `reflection` nil. The reflection is
    /// `MeshPipelineDescriptor::reflection`'s (its object, mesh and fragment bindings) of the
    /// descriptor the state was made from, with the dynamic libraries its creation loaded, under
    /// the creation's budget; what it refuses refuses the call (`Unsupported`, `Compile`). It
    /// comes from the shader cache under the descriptor's key
    /// (`Device::mesh_descriptor_key`), so every state of one descriptor shares one answer rather
    /// than decoding the three stages' AIR again.
    pub fn new_mesh_render_pipeline_state_with_reflection(
        &self,
        descriptor: &MeshPipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<(RenderPipelineState, Option<RenderPipelineReflection>), Error> {
        let state = GraphicsState::create_mesh(self, descriptor, options)?;
        let reflection = match (options.options.reflects(), state.descriptor()) {
            // Through the shader cache under the descriptor's key, as a drawing pipeline's
            // reflection goes: it decodes the object, mesh and fragment functions' AIR, and
            // Metal answers every state of one descriptor from the library it compiled once.
            (true, StateDescriptor::Mesh(loaded)) => {
                let compile = self.compile_device();
                let cache = &self.shared().shader_cache;
                Some(
                    cache
                        .render_reflection(compile, state.descriptor_key(), || {
                            loaded.reflection(&options.budget())
                        })
                        .map_err(device_error)?,
                )
            }
            (true, StateDescriptor::Vertex(_)) => {
                return Err(internal("a mesh state made of a render descriptor"))
            }
            (false, _) => None,
        };
        Ok((RenderPipelineState(Kind::Graphics(state)), reflection))
    }
}

impl RenderPipelineState {
    /// The state of a tile pipeline (`tile.rs`).
    pub(super) fn of_tile(state: TileState) -> RenderPipelineState {
        RenderPipelineState(Kind::Tile(state))
    }

    /// The state that draws `state`.
    pub(super) fn of_graphics(state: GraphicsState) -> RenderPipelineState {
        RenderPipelineState(Kind::Graphics(state))
    }

    /// What the state is.
    pub(crate) fn kind(&self) -> &Kind {
        &self.0
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        match &self.0 {
            Kind::Graphics(state) => &state.0.device,
            Kind::Tile(state) => state.kernel().device(),
        }
    }

    /// `supportIndirectCommandBuffers`: a drawing state's descriptor's; false for a tile
    /// pipeline's, whose descriptor has no such field.
    pub fn support_indirect_command_buffers(&self) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state.0.descriptor.support_indirect_command_buffers(),
            Kind::Tile(state) => state.kernel().support_indirect_command_buffers(),
        }
    }

    /// Whether the state's fragment function reads a flat input: a runtime extension, with no
    /// Metal counterpart (`MTLRenderPipelineReflection` does not state it). It is the compiled
    /// pipeline's `GraphicsState::flat_fragment_inputs`, from the translated fragment stage's
    /// inputs: a `[[flat]]` varying, or the `[[render_target_array_index]]` or
    /// `[[viewport_array_index]]` the vertex function wrote, each of which every fragment of a
    /// primitive reads from one vertex of it. `false` for a state without a fragment function,
    /// for one whose fragment function reads only smooth inputs or none, and for a tile
    /// pipeline's, whose function reads no varying. A consumer that draws a primitive's vertices
    /// in another order changes what only such a state shows, or one whose fragment function
    /// reads `[[barycentric_coord]]`, which this does not count
    /// ([`reads_barycentric_coord`](Self::reads_barycentric_coord) states it).
    pub fn has_flat_fragment_inputs(&self) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state.base().graphics.flat_fragment_inputs,
            Kind::Tile(_) => false,
        }
    }

    /// Whether the state's fragment function reads `[[barycentric_coord]]`: a runtime extension,
    /// with no Metal counterpart (`MTLRenderPipelineReflection` does not state it). It is the
    /// compiled pipeline's `GraphicsState::reads_barycentric_coord`, from the translated fragment
    /// stage's builtins, or its varyings on a device without `fragmentShaderBarycentric`: the
    /// weights perspective-correct or not, of any width. Metal reads them at the pixel center
    /// only, so these are all its forms. `false` for a state without a
    /// fragment function, for one whose fragment function reads no barycentric weights, and for
    /// a tile pipeline's, whose function reads none. A primitive's weights follow the order its
    /// vertices are drawn in: a consumer that draws them in another order changes what a draw
    /// shows only where this or [`has_flat_fragment_inputs`](Self::has_flat_fragment_inputs)
    /// answers `true`.
    pub fn reads_barycentric_coord(&self) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state.base().graphics.reads_barycentric_coord,
            Kind::Tile(_) => false,
        }
    }

    /// Whether the state's vertex function writes memory: a runtime extension, with no Metal
    /// counterpart (`MTLRenderPipelineReflection` does not state it). It is the compiled
    /// pipeline's `GraphicsState::vertex_writes_memory`, from the translated vertex stage: a
    /// store to a buffer or a texture, or an atomic update of one, that another invocation or a
    /// later command can read (`StageInterface::writes_memory`); a post-tessellation vertex
    /// function's, and for a mesh pipeline's state, its mesh function's. `false` for a vertex
    /// function that only reads memory and returns its outputs, and for a tile pipeline's state,
    /// which has no vertex function. Every run of the vertex stage repeats its writes, so a
    /// consumer that draws a primitive's vertices another number of times (a re-indexed list in
    /// place of a strip or a fan) changes what a draw writes only where this answers `true`.
    pub fn vertex_writes_memory(&self) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state.base().graphics.vertex_writes_memory,
            Kind::Tile(_) => false,
        }
    }

    /// Whether the state's vertex function writes `[[point_size]]`: a runtime extension, with no
    /// Metal counterpart (`MTLRenderPipelineReflection` does not state it). It is the compiled
    /// pipeline's `GraphicsState::writes_point_size`, from the translated vertex stage's
    /// built-in outputs (its `PointSize` output, not the 1.0 the module writes for a function
    /// without one); for a mesh pipeline's state, its mesh function's. `false` for a vertex
    /// function without a point-size output, for a post-tessellation vertex function, whose point
    /// size sizes nothing and is dropped, and for a tile pipeline's state. Metal refuses a state
    /// whose vertex function writes it under the line or triangle input primitive topology
    /// class (`Problem::PointSizeTopology`), and draws a point of size 1 only under the point
    /// class for one that does not (docs/API.md section 4).
    pub fn writes_point_size(&self) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state.base().graphics.writes_point_size,
            Kind::Tile(_) => false,
        }
    }

    /// Whether the state's vertex function writes `[[render_target_array_index]]`: a runtime
    /// extension, with no Metal counterpart (`MTLRenderPipelineReflection` does not state it).
    /// It is decided from the translated vertex stage's outputs, whose roles come from the
    /// function's AIR output metadata: a vertex function that writes the index depends on the
    /// pass's array length (`RuntimeFact::RenderTargetArrayLength`) in every variant, whether or
    /// not the variant stores the index (a pass of one layer runs the one that does not). A
    /// post-tessellation vertex function's counts, and for a mesh pipeline's state, its mesh
    /// function's. `false` for a tile pipeline's state. Metal routes each primitive of a pipeline
    /// that writes this or [`writes_viewport_array_index`](Self::writes_viewport_array_index) by
    /// its provoking vertex's index (measured on an M3 Max by the reims-vgpu consumer), so a
    /// consumer that draws a primitive's vertices in another order changes where such a draw lands
    /// whether or not the fragment function reads the index.
    pub fn writes_render_target_array_index(&self) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state.vertex_chooses_slice(),
            Kind::Tile(_) => false,
        }
    }

    /// Whether the state's vertex function writes `[[viewport_array_index]]`: a runtime
    /// extension, with no Metal counterpart (`MTLRenderPipelineReflection` does not state it).
    /// It is the compiled pipeline's `GraphicsState::writes_viewport_index`, from the translated
    /// vertex stage's built-in outputs, whose roles come from the function's AIR output metadata:
    /// a post-tessellation vertex function's counts, and for a mesh pipeline's state, its mesh
    /// function's. `false` for a tile pipeline's state. As for
    /// [`writes_render_target_array_index`](Self::writes_render_target_array_index), Metal routes
    /// each primitive by its provoking vertex's index. `true` on a device without `multiViewport`
    /// too, where a draw has one viewport and the function stores no index: there a draw while
    /// the encoder holds more than one viewport is refused, `Unsupported(LayeredRendering)` needing
    /// `multiViewport`.
    pub fn writes_viewport_array_index(&self) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state.base().graphics.writes_viewport_index,
            Kind::Tile(_) => false,
        }
    }

    /// Whether a function of the state may encode, reset or copy indirect commands: a runtime
    /// extension, with no Metal counterpart (`MTLRenderPipelineReflection` does not state it). It
    /// is the compiled pipeline's `PipelineInterface::reaches_command_buffers` (a tile
    /// pipeline's, its tile function's), read from the functions' AIR, of the functions their
    /// constants specialized: `true` where a function takes a `command_buffer` (an argument, or a
    /// member of an argument buffer it takes at any depth of its structs), and where a body calls
    /// a command operation on any pointer, one it casts from other bytes included, or makes a
    /// call the translator cannot see into. Only a state that answers `true` can encode, reset or
    /// copy indirect commands, so `false` is proven; one may answer `true` and never do so (a
    /// command buffer a function takes and does not touch).
    pub fn can_encode_indirect_commands(&self) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state.base().interface.reaches_command_buffers,
            Kind::Tile(state) => state.kernel().can_encode_indirect_commands(),
        }
    }

    /// What the state's functions may write into the indirect command buffers they reach: a
    /// runtime extension, with no Metal counterpart. It is the compiled pipeline's
    /// `PipelineInterface::command_writes` (a tile pipeline's, its tile function's), read from
    /// the functions' AIR, of the functions their constants specialized ([`CommandWrites`]: the
    /// indirect-command operations their bodies may call, and the primitive types their draws
    /// may pass, every type where a draw's is not a constant). A clear bit is proven: no function
    /// of the state writes such a record field. Empty where
    /// [`can_encode_indirect_commands`](Self::can_encode_indirect_commands) answers `false`.
    pub fn indirect_command_writes(&self) -> CommandWrites {
        match &self.0 {
            Kind::Graphics(state) => state.base().interface.command_writes,
            Kind::Tile(state) => state.kernel().indirect_command_writes(),
        }
    }

    /// Whether the state's vertex function may write a texture: a runtime extension, with no
    /// Metal counterpart (`MTLRenderPipelineReflection` does not state it). It is read from the
    /// function's AIR, of the function its constants specialized (the compiled pipeline's
    /// `PipelineInterface::texture_writing_stages`, from `StageInterface::writes_textures`):
    /// `true` where it takes a texture declared `access::write` or `access::read_write` (an
    /// argument, or a member of an argument buffer it takes at any depth of its structs), and
    /// where its body calls a texture operation that stores a texel (a `write`, an imageblock
    /// slice written to a texture, or an atomic other than a load) through any handle, an
    /// argument buffer's, a heap's or a resource ID included, or makes a call the translator
    /// cannot see into. Color attachment outputs, imageblock and tile stores, and stores to the
    /// bytes of a buffer a texture views are not texture writes. `false` is proven: the
    /// function writes no texture; one may answer `true` and never write (a writable texture it
    /// takes and does not touch). A post-tessellation vertex function's counts; `false` for a
    /// mesh pipeline's state, which has none (see
    /// [`object_writes_textures`](Self::object_writes_textures) and
    /// [`mesh_writes_textures`](Self::mesh_writes_textures)), and for a tile pipeline's state.
    pub fn vertex_writes_textures(&self) -> bool {
        self.stage_writes_textures(metal2vulkan::pipeline::Stage::Vertex)
    }

    /// Whether the state's fragment function may write a texture: a runtime extension, decided
    /// as [`vertex_writes_textures`](Self::vertex_writes_textures) decides the vertex
    /// function's. `false` for a state without a fragment function and for a tile pipeline's
    /// state.
    pub fn fragment_writes_textures(&self) -> bool {
        self.stage_writes_textures(metal2vulkan::pipeline::Stage::Fragment)
    }

    /// Whether a mesh pipeline's state's object function may write a texture: a runtime
    /// extension, decided as [`vertex_writes_textures`](Self::vertex_writes_textures) decides
    /// the vertex function's. `false` for a state without an object function.
    pub fn object_writes_textures(&self) -> bool {
        self.stage_writes_textures(metal2vulkan::pipeline::Stage::Object)
    }

    /// Whether a mesh pipeline's state's mesh function may write a texture: a runtime
    /// extension, decided as [`vertex_writes_textures`](Self::vertex_writes_textures) decides
    /// the vertex function's. `false` for any other state.
    pub fn mesh_writes_textures(&self) -> bool {
        self.stage_writes_textures(metal2vulkan::pipeline::Stage::Mesh)
    }

    /// Whether a tile pipeline's state's tile function may write a texture: a runtime
    /// extension, decided as [`vertex_writes_textures`](Self::vertex_writes_textures) decides
    /// the vertex function's (its tile function's
    /// [`ComputePipelineState::writes_textures`](crate::ComputePipelineState::writes_textures)).
    /// `false` for any other state.
    pub fn tile_writes_textures(&self) -> bool {
        match &self.0 {
            Kind::Graphics(_) => false,
            Kind::Tile(state) => state.kernel().writes_textures(),
        }
    }

    /// The color attachments the state's fragment function may write, by color index, sorted:
    /// a runtime extension, with no Metal counterpart (`MTLRenderPipelineReflection` does not
    /// state it). It is the compiled pipeline's `GraphicsState::color_outputs`, decided from the
    /// fragment stage's outputs (its mask of them). Those come from the AIR's output metadata of the function its
    /// constants specialized, never from names. An index counts where a first output
    /// (`[[color(n)]]`, index 0) names it, whether the function writes it as an output or, for an
    /// attachment it reads in the interlocked storage form, as a storage image.
    ///
    /// An index not listed is one the function never writes: no output names it, a function
    /// constant gates its output off, or only a dual-source second output
    /// (`[[color(n), index(1)]]`) names it, or the descriptor has no attachment there, where
    /// Metal discards the write (but color 0 under alpha to coverage, whose alpha the coverage
    /// reads). One listed may still go unwritten, since an output the function leaves undefined
    /// counts. Empty for a state
    /// without a fragment function, and for a tile pipeline's state, whose tile function writes
    /// the attachments through the imageblock rather than as color outputs. A mesh pipeline's
    /// state answers for its fragment function.
    pub fn fragment_color_outputs(&self) -> Vec<u32> {
        let mask = match &self.0 {
            Kind::Graphics(state) => state.base().graphics.color_outputs,
            Kind::Tile(_) => 0,
        };
        (0..u32::BITS).filter(|n| mask & (1 << n) != 0).collect()
    }

    /// How the state's function of `stages` accesses texture slot `index`: a runtime extension.
    /// For one stage it is the `access` that the pipeline's reflection states for the texture
    /// argument covering the slot, an array's covering each element's slot. That is the
    /// `MTLTextureBinding` of the `reflection:` out-parameter of Metal's creation call
    /// ([`Device::new_render_pipeline_state_with_reflection`] and the mesh and tile forms), which
    /// the runtime reads from the descriptor the state was made from the first time it is asked,
    /// and keeps. Metal's own `reflection` property of a state made from a device is nil. The
    /// access is the declared one, measured on an M3 Max and an M4 (`reflection`'s rules in the
    /// compile layer, a fragment function's slots by our probe):
    /// - `ReadOnly` for a texture declared `access::read`, or sampled;
    /// - `WriteOnly` for `access::write`;
    /// - `ReadWrite` for `access::read_write`, whether or not the function both reads and writes
    ///   it.
    ///
    /// `None` where no texture argument of the stage covers the slot. That includes one a
    /// function constant gates off and one the function never references (Metal's reflection
    /// lists that one with `isUsed` false; the function neither reads nor writes it). Where
    /// `stages` names several stages, their accesses join: one access where they agree,
    /// `ReadWrite` where they differ. A tile pipeline's state answers for `TILE`, a mesh
    /// pipeline's for `OBJECT`, `MESH` and `FRAGMENT`.
    ///
    /// Only the slot is answered. A function may also reach textures through an argument buffer,
    /// a heap or a resource ID, which no slot's answer covers;
    /// [`vertex_writes_textures`](Self::vertex_writes_textures) and its siblings cover those
    /// too.
    ///
    /// The answers come from the pipeline's slot writes
    /// (`metal2vulkan::reflection::SlotWrite`), not from its reflection: a reflection must give
    /// Metal's `MTLType` for every binding and refuses the whole function where Metal's answer
    /// for one argument's *type* has not been measured, which refused every slot of the state,
    /// read-only ones included. A state that compiles answers here.
    ///
    /// Refused for one slot alone (`Unsupported` with `Feature::Reflection`) where an argument
    /// that may cover it leaves it undetermined: a slot only a function constant states, a
    /// payload in a stage Metal's access for one is unmeasured in, or an argument whose role only
    /// an argument-buffer member takes. Every other slot still answers, and a slot a determined
    /// argument covers answers whatever else the state holds. The decode's own errors are
    /// `Compile`. No refusal is remembered: only a successful set of answers is kept, so a later
    /// call is answered again rather than handed the first failure.
    pub fn texture_access(
        &self,
        stages: RenderStages,
        index: u64,
    ) -> Result<Option<BindingAccess>, Error> {
        let accesses = match &self.0 {
            Kind::Graphics(state) => state.texture_accesses()?,
            Kind::Tile(state) => state.texture_accesses()?,
        };
        accesses.at(stages, index)
    }

    /// The pipeline's reflection, as its creation's out-parameter answers it where the options
    /// ask for one ([`Device::new_render_pipeline_state_with_reflection`] and the mesh and tile
    /// forms): a runtime extension, at any time after the creation. Metal's `reflection` property
    /// (macOS 26) answers nil for a state an `MTLDevice` made; this is the creation's answer, each
    /// stage's bindings of the entry its constants specialized. A drawing state's is the one the
    /// device's shader cache keeps for its descriptor's key, which the creation's out-parameter
    /// keeps too: its functions' AIR is decoded only where neither the memo nor the cache's
    /// directory holds it. A tile pipeline's is read from what its tile function's compile at the
    /// creation kept, with no decode. A mesh pipeline's decodes its functions' AIR. Refused as the
    /// creation's reflection is (`Unsupported` with `Feature::Reflection` for an argument whose
    /// answer Metal's has not been measured for, `Compile` for the decode's errors).
    pub fn compiled_reflection(&self) -> Result<RenderPipelineReflection, Error> {
        let budget = metal2vulkan::Budget::standard();
        match &self.0 {
            Kind::Graphics(state) => state.reflection(&budget),
            Kind::Tile(state) => state.kernel().tile_reflection(&budget),
        }
    }

    /// Whether the graphics state's function of `stage` may write a texture; `false` for a tile
    /// pipeline's state.
    fn stage_writes_textures(&self, stage: metal2vulkan::pipeline::Stage) -> bool {
        match &self.0 {
            Kind::Graphics(state) => state
                .base()
                .interface
                .texture_writing_stages
                .contains(&stage),
            Kind::Tile(_) => false,
        }
    }

    /// `gpuResourceID`: for a drawing state made with `supportIndirectCommandBuffers`, its
    /// placement's ID (`PipelineOptions::placement`), else the lowest ID free when it was made,
    /// from 1, of the space compute states made so take theirs from (the probes: Metal's compute
    /// and render states share one dense space), given back when the state goes; 0 for any other
    /// state, a tile pipeline's included, as Metal answers. It is what an indirect render command's
    /// record names the state by.
    pub fn gpu_resource_id(&self) -> ResourceId {
        match &self.0 {
            Kind::Graphics(state) => state.0.id.as_ref().map_or(ResourceId(0), IdHold::id),
            Kind::Tile(_) => ResourceId(0),
        }
    }

    /// The serial of a drawing state's hold of its ID (`ComputePipelineState::id_serial`), for
    /// a state that has one.
    pub(crate) fn id_serial(&self) -> Option<u64> {
        match &self.0 {
            Kind::Graphics(state) => state.0.id.as_ref().map(IdHold::serial),
            Kind::Tile(_) => None,
        }
    }

    /// `resourceIndex`, Apple's private property: [`gpu_resource_id`](Self::gpu_resource_id)'s
    /// value, 0 for a state made without `supportIndirectCommandBuffers` (a tile pipeline's
    /// included), as Metal answers (equal to `gpuResourceID` for compute and
    /// render states made with and without it).
    pub fn resource_index(&self) -> u64 {
        self.gpu_resource_id().0
    }

    /// `uniqueIdentifier`, Apple's private property: a nonzero value the state answers for its
    /// life, no other live state's. A drawing state made with `supportIndirectCommandBuffers`
    /// answers 2^36 plus an offset in the device's allocation of such render states (apart from
    /// compute states'), the last released taken again first, as Metal's does, at offsets that
    /// are not Metal's; any other, a tile pipeline's included, a value of one process-wide count
    /// past them, where Metal's is an address that differs from run to run. No state's is measured
    /// to be Metal's number: the first answer announces, once a device,
    /// [`Degradation::PlainStateUniqueIdentifier`](crate::Degradation::PlainStateUniqueIdentifier)
    /// for a plain state (a tile state's included) and
    /// [`Degradation::UnmeasuredUniqueIdentifier`](crate::Degradation::UnmeasuredUniqueIdentifier)
    /// for one with indirect command buffer support.
    pub fn unique_identifier(&self) -> u64 {
        match &self.0 {
            Kind::Graphics(state) => match &state.0.identifier {
                Some(identifier) => identifier.answer(self.device(), "a render pipeline state"),
                None => 0,
            },
            Kind::Tile(tile) => tile.kernel().unique_identifier(),
        }
    }

    /// Its `uniqueIdentifier` as the runtime holds it, announcing nothing: what an indirect
    /// command that names the state answers.
    pub(crate) fn identifier_value(&self) -> u64 {
        match &self.0 {
            Kind::Graphics(state) => state
                .0
                .identifier
                .as_ref()
                .map_or(0, UniqueIdentifier::value),
            Kind::Tile(tile) => tile.kernel().identifier_value(),
        }
    }

    /// `label`: the label of the descriptor the state was made with (a render, mesh or tile
    /// pipeline's, `PipelineOptions::label`), `None` for nil; a state made with additional
    /// binary functions keeps its original's, as Metal's does.
    pub fn label(&self) -> Option<&str> {
        match &self.0 {
            Kind::Graphics(state) => state.0.label.as_deref(),
            Kind::Tile(state) => state.kernel().label(),
        }
    }

    /// The state a patch draw an indirect command scales runs on: a drawing state's
    /// factor-scaling twin (`GraphicsState::factor_scaling`); `None` for a state that does not
    /// tessellate, a tile pipeline's included.
    pub(crate) fn factor_scaling(&self) -> Result<Option<RenderPipelineState>, Error> {
        match &self.0 {
            Kind::Graphics(state) => Ok(state
                .factor_scaling()?
                .map(|twin| RenderPipelineState(Kind::Graphics(twin)))),
            Kind::Tile(_) => Ok(None),
        }
    }

    /// A drawing state's single-sample twin (`GraphicsState::single_sample`); `None` for a mesh
    /// render pipeline's state and a tile pipeline's.
    pub(crate) fn single_sample(&self) -> Result<Option<RenderPipelineState>, Error> {
        match &self.0 {
            Kind::Graphics(state) => Ok(state
                .single_sample()?
                .map(|twin| RenderPipelineState(Kind::Graphics(twin)))),
            Kind::Tile(_) => Ok(None),
        }
    }

    /// A drawing state's twin of color group `group` (`GraphicsState::color_group`); `None` for
    /// a tile pipeline's state and a group the state does not have.
    pub(crate) fn color_group(&self, group: u32) -> Option<RenderPipelineState> {
        match &self.0 {
            Kind::Graphics(state) => state
                .color_group(group)
                .map(|twin| RenderPipelineState(Kind::Graphics(twin))),
            Kind::Tile(_) => None,
        }
    }

    /// A drawing state, named without holding it, as an indirect render command names the state
    /// set on it ([`ComputePipelineState::downgrade`](super::ComputePipelineState::downgrade));
    /// `None` for a tile pipeline's.
    pub(crate) fn downgrade(&self) -> Option<WeakRenderPipelineState> {
        match &self.0 {
            Kind::Graphics(state) => Some(WeakRenderPipelineState(Arc::downgrade(&state.0))),
            Kind::Tile(_) => None,
        }
    }

    /// What a drawing state runs, held without its ID; `None` for a tile pipeline's.
    pub(crate) fn program(&self) -> Option<RenderProgram> {
        match &self.0 {
            Kind::Graphics(state) => Some(RenderProgram(Arc::clone(&state.0.program))),
            Kind::Tile(_) => None,
        }
    }

    /// The drawing states of `device` made with `supportIndirectCommandBuffers` that live now, by
    /// ID: every state an indirect render command's record can name.
    pub(crate) fn indirect_states(device: &Device) -> Vec<RenderPipelineState> {
        let held = lock(&device.shared().resource_ids).render_pipelines();
        // Upgraded with the IDs' lock free: a state's last handle, dropped, takes it.
        held.iter()
            .filter_map(Weak::upgrade)
            .map(|state| RenderPipelineState(Kind::Graphics(GraphicsState(state))))
            .collect()
    }

    /// `maxTotalThreadsPerThreadgroup`: 0 for a drawing state, as Metal answers (measured on an
    /// M3 Max and an M4); a tile pipeline's compile's (`tile.rs`).
    pub fn max_total_threads_per_threadgroup(&self) -> u32 {
        match &self.0 {
            Kind::Graphics(_) => 0,
            Kind::Tile(state) => state.kernel().max_total_threads_per_threadgroup(),
        }
    }

    /// `threadgroupSizeMatchesTileSize`: a tile pipeline's descriptor's; false for a drawing
    /// state.
    pub fn threadgroup_size_matches_tile_size(&self) -> bool {
        match &self.0 {
            Kind::Graphics(_) => false,
            Kind::Tile(state) => state.descriptor().threadgroup_size_matches_tile_size,
        }
    }

    /// `imageblockSampleLength` (measured on an M3 Max and an M4): a drawing state's color
    /// attachments' texels as
    /// [`imageblock_sample_length`](metal2vulkan::metal::format::imageblock_sample_length) packs them,
    /// depth and stencil excluded, after its fragment function's explicit imageblock where it
    /// declares one, as
    /// [`explicit_imageblock_sample_length`](metal2vulkan::metal::format::explicit_imageblock_sample_length)
    /// places them (measured on an M3 Max); a tile pipeline's as `tile.rs` says. A tile state
    /// whose descriptor gives a color format without a texel size (one no pass has) is not
    /// measured (`NotImplemented(Render)`).
    pub fn imageblock_sample_length(&self) -> Result<u64, Error> {
        match &self.0 {
            Kind::Graphics(state) => {
                let formats = state.0.descriptor.color_formats();
                let colors = metal2vulkan::metal::format::imageblock_sample_length(&formats)
                    .ok_or_else(|| internal("a color attachment format without a texel size"))?;
                let length = match &state.base().graphics.fragment_imageblock {
                    Some(ib) => metal2vulkan::metal::format::explicit_imageblock_sample_length(
                        ib.data_size,
                        ib.alignment,
                        colors,
                    )
                    .ok_or_else(|| internal("an imageblock sample past 4 GiB"))?,
                    None => colors,
                };
                Ok(u64::from(length))
            }
            Kind::Tile(state) => state
                .imageblock_sample_length()
                .ok_or(Error::NotImplemented(RuntimeGap::Render)),
        }
    }

    /// `imageblockMemoryLengthForDimensions:` (measured on an M3 Max and an M4): the sample length
    /// times the dimensions' width and height, its depth ignored, and for a drawing state times its
    /// raster sample count; neither rounded. Refused as
    /// [`imageblock_sample_length`](Self::imageblock_sample_length) is.
    pub fn imageblock_memory_length_for_dimensions(&self, dimensions: Size) -> Result<u64, Error> {
        let samples = match &self.0 {
            Kind::Graphics(state) => u64::from(state.0.descriptor.raster_sample_count()),
            Kind::Tile(_) => 1,
        };
        Ok(self
            .imageblock_sample_length()?
            .saturating_mul(dimensions.width)
            .saturating_mul(dimensions.height)
            .saturating_mul(samples))
    }
}

impl GraphicsState {
    /// `new_render_pipeline_state` of `descriptor` under `options` on `device`, noting what it
    /// uses in `used` (a binary archive's add). Each stage's dynamic libraries are loaded first
    /// (`dynamic_library.rs`), and the state keeps the descriptor with them.
    pub(crate) fn create(
        device: &Device,
        descriptor: &RenderPipelineDescriptor,
        options: &PipelineOptions,
        used: Option<&Used>,
    ) -> Result<GraphicsState, Error> {
        GraphicsState::create_adding(device, descriptor, options, used, [0, 0])
    }

    /// [`create`](Self::create), of a descriptor whose vertex function's, then fragment
    /// function's, last `added_unsupported` `binary_functions` were added without the stage's
    /// support flag.
    pub(super) fn create_adding(
        device: &Device,
        descriptor: &RenderPipelineDescriptor,
        options: &PipelineOptions,
        used: Option<&Used>,
        added_unsupported: [usize; 2],
    ) -> Result<GraphicsState, Error> {
        GraphicsState::create_reflecting(
            device,
            descriptor,
            options,
            (used, added_unsupported),
            false,
        )
        .map(|(state, _)| state)
    }

    /// [`create_adding`](Self::create_adding), and where `reflect`, the reflection of the
    /// descriptor the state is made from, worked out beside the creation: the one the shader
    /// cache keeps for its key, looked up once the creation's checks passed, else
    /// `RenderPipelineDescriptor::reflection`'s answer, worked out on a thread of its own while
    /// the creation translates and makes, joined before the creation returns. The caller keeps
    /// what was worked out ([`Reflected::Made`]). `None` where `reflect` is false, or where no
    /// thread could be made for it (the caller then works it out, as before).
    pub(super) fn create_reflecting(
        device: &Device,
        descriptor: &RenderPipelineDescriptor,
        options: &PipelineOptions,
        (used, added_unsupported): (Option<&Used>, [usize; 2]),
        reflect: bool,
    ) -> Result<(GraphicsState, Option<Reflected>), Error> {
        let _noted = NoteEntries(device);
        let mut loaded = descriptor.clone();
        loaded.vertex = crate::dynamic_library::stage_with_dependencies(&loaded.vertex);
        loaded.fragment = loaded
            .fragment
            .as_ref()
            .map(crate::dynamic_library::stage_with_dependencies);
        let descriptor = &loaded;
        // Metal's descriptor validation stops an attachment in a format Metal describes none
        // of, first (`attachment_formats.rs`).
        super::attachment_formats::check(
            "new_render_pipeline_state",
            &descriptor.color_attachments,
            descriptor.depth_format,
            descriptor.stencil_format,
        )?;
        // Metal answers nil for a `rasterSampleCount` an Apple GPU does not take, whatever the
        // device rasterizes (`raster_samples.rs`).
        super::raster_samples::check("new_render_pipeline_state", descriptor.raster_sample_count)?;
        let scaled = descriptor
            .tessellation
            .as_ref()
            .is_some_and(|t| t.factor_scale_enabled);
        if scaled && descriptor.support_indirect_command_buffers {
            return Err(invalid(
                "new_render_pipeline_state",
                Problem::IndirectScaledTessellation,
            ));
        }
        super::additions::check_render(descriptor, "new_render_pipeline_state")?;
        // Metal answers nil with an `NSError` for a state with rasterization disabled whose vertex
        // function returns a value (`unrasterized.rs`), before it compiles anything.
        // Each answer below is the one a creation of a descriptor of the same key worked out, where
        // one kept it in memory or in the cache directory (`shader_cache/creation.rs`), else worked
        // out now; kept where this creation worked one out and every question has its answer.
        let compile = device.compile_device();
        let cache = &device.shared().shader_cache;
        let descriptor_key = render_descriptor_key(device, descriptor);
        let known = cache
            .linked_answers(compile, &descriptor_key)
            .unwrap_or_default();
        let mut answers = LinkedUseAnswers::default();
        let mut learned = false;
        // Every check, and the key below, read the device's stage entries: a state of functions an
        // earlier creation read reads its entries and its plain answers rather than decoding and
        // specializing them again (`pipeline/memo.rs`).
        let entries = cache.memo.stage_entries();
        if !descriptor.rasterization_enabled {
            let returns = match known.vertex_returns_value {
                Some(returns) => Ok(returns),
                None => {
                    learned = true;
                    device.shared().counts.linked_use_question_asked();
                    descriptor.vertex_returns_value(&options.budget(), entries)
                }
            };
            answers.vertex_returns_value = returns.as_ref().ok().copied();
            super::unrasterized::check("new_render_pipeline_state", returns)?;
        }
        // Metal links the fragment function whether or not the pipeline rasterizes, and answers
        // nil with an `NSError`, in this order, for a fragment function writing depth without a
        // depth attachment, a function using a direct texture or sampler with indirect command
        // buffers (`indirect_arguments.rs`), and stage inputs the linking keeps that no vertex
        // output writes (`linked_fragment.rs`), before it compiles anything. The compile then
        // drops a fragment function nothing rasterizes.
        let writes = match known.fragment_writes_depth_unattached {
            Some(writes) => Ok(writes),
            None => {
                learned = true;
                device.shared().counts.linked_use_question_asked();
                descriptor.fragment_writes_depth_unattached(&options.budget(), entries)
            }
        };
        answers.fragment_writes_depth_unattached = writes.as_ref().ok().copied();
        super::linked_fragment::check_depth("new_render_pipeline_state", writes)?;
        if descriptor.support_indirect_command_buffers {
            let uses = match known.stages_using_direct_arguments {
                Some(uses) => Ok(uses),
                None => {
                    learned = true;
                    device.shared().counts.linked_use_question_asked();
                    descriptor.stages_using_direct_arguments(&options.budget(), entries)
                }
            };
            answers.stages_using_direct_arguments = uses.as_ref().ok().cloned();
            super::indirect_arguments::check_render("new_render_pipeline_state", uses)?;
        }
        // A post-tessellation vertex function's too.
        let unwritten = match known.unwritten_fragment_inputs {
            Some(unwritten) => Ok(unwritten),
            None => {
                learned = true;
                device.shared().counts.linked_use_question_asked();
                descriptor.unwritten_fragment_inputs(&options.budget(), entries)
            }
        };
        answers.unwritten_fragment_inputs = unwritten.as_ref().ok().cloned();
        super::linked_fragment::check_inputs("new_render_pipeline_state", "vertex", unwritten)?;
        if learned && answers_every_question(&answers, descriptor) {
            cache.keep_linked_answers(compile, &descriptor_key, answers);
        }
        let rasterized;
        // The descriptor the state is made from, and its key: the one the checks above asked
        // their questions under, where it is the same descriptor. Dropping the fragment stage
        // makes another descriptor, which the key reads as one (a stage no descriptor names is
        // written apart from any stage), so that creation builds the one other key it needs.
        let (descriptor, made_key) =
            match descriptor.fragment.is_some() && !descriptor.rasterization_enabled {
                true => {
                    let mut unrasterized = descriptor.clone();
                    unrasterized.fragment = None;
                    rasterized = unrasterized;
                    let key = render_descriptor_key(device, &rasterized);
                    (&rasterized, key)
                }
                false => (descriptor, descriptor_key),
            };
        // The reflection of the descriptor the state is made from: kept, else worked out on a
        // thread of its own beside the rest of the creation, which it shares nothing with. Under
        // the key the compile and the state read, which the creation built once.
        let reflection_key = reflect.then_some(made_key);
        let kept = reflection_key
            .as_ref()
            .map(|key| cache.kept_render_reflection(compile, key));
        std::thread::scope(|scope| {
            // A thread that cannot be made leaves the reflection to the caller, as before.
            let making = match (&kept, &reflection_key) {
                (Some(None), Some(key)) => {
                    let budget = options.budget();
                    crate::exit_guard::spawn_scoped(
                        scope,
                        std::thread::Builder::new().name("metal2vulkan-reflection".into()),
                        move || descriptor.reflection(&budget),
                    )
                    .ok()
                    .map(|making| (*key, making))
                }
                _ => None,
            };
            let state = GraphicsState::create_checked(
                device,
                (descriptor, options, used, added_unsupported),
                entries,
                &made_key,
            )?;
            let reflected = match (kept, making) {
                (Some(Some(reflection)), _) => Some(Reflected::Kept(reflection)),
                (_, Some((key, making))) => Some(Reflected::Made(
                    key,
                    making
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
                )),
                _ => None,
            };
            Ok((state, reflected))
        })
    }

    /// The state of `descriptor`, the rasterized descriptor of a render creation whose checks
    /// passed, with `entries`, the device's specialized stage entries, which its checks read and
    /// filled, and `descriptor_key`, the key of `descriptor` its creation built: compiled under
    /// stand-ins where it needs them, its variant's objects made, and the pipelines Metal
    /// compiles at its creation made (`create_reflecting`). Every compile of the creation looks
    /// its pipeline up under the key given, and the state keeps it, so that one creation builds
    /// it once (`render_descriptor_key`).
    fn create_checked(
        device: &Device,
        (descriptor, options, used, added_unsupported): (
            &RenderPipelineDescriptor,
            &PipelineOptions,
            Option<&Used>,
            [usize; 2],
        ),
        entries: &StageEntries,
        descriptor_key: &PipelineKey,
    ) -> Result<GraphicsState, Error> {
        let placement = if descriptor.support_indirect_command_buffers {
            pipeline_placement("new_render_pipeline_state", options.placement)?
        } else {
            IdPlacement::Any
        };
        binary_archive::check(
            device,
            options,
            "new_render_pipeline_state",
            ArchivedPipeline::Render,
            binary_archive::render_missing(descriptor),
        )?;
        let sources = Sources::creation(device, options, used);
        let profile = device.compile_device().profile();
        let (found, key) = stand_in::compile_with_stand_ins(profile, |key| {
            RenderVariant::compiled(&sources, descriptor, key, entries, descriptor_key)
        })
        .map_err(|error| compile_error(device, error))?;
        let sized_class = matches!(
            descriptor.input_primitive_topology,
            PrimitiveTopologyClass::Line | PrimitiveTopologyClass::Triangle
        );
        let graphics = match &found.remembered {
            Remembered::Live(variant) => &variant.pipeline.graphics,
            Remembered::Compiled(pipeline) => &pipeline.graphics,
        };
        if sized_class && graphics.writes_point_size {
            return Err(invalid(
                "new_render_pipeline_state",
                Problem::PointSizeTopology,
            ));
        }
        let stand_in = !key.storage_images.is_empty();
        let base = RenderVariant::of(device, key, found)?;
        if binary_archive::combines_stages(device, options, descriptor) {
            device.degrade(Degradation::CombinedArchiveStages, || {
                (
                    format!(
                        "the render pipeline state of vertex function {} and fragment function {}",
                        stage_name(&descriptor.vertex),
                        descriptor
                            .fragment
                            .as_ref()
                            .map_or("none".into(), stage_name)
                    ),
                    "the listed binary archives hold its vertex stage and its fragment stage \
                     from two adds, not one"
                        .to_owned(),
                )
            });
        }
        let large = large_form(
            device,
            descriptor,
            (&base, entries),
            stand_in,
            descriptor_key,
        );
        // Each color group's variant, where Metal compiles: a refusal of any group's refuses the
        // creation, as Metal's creation fails (`GraphicsState::color_group`). The first group's
        // is the base.
        let group_bases = group_bases(
            &sources,
            descriptor,
            entries,
            (&base, stand_in),
            descriptor_key,
        )?;
        let grouped = base.pipeline.graphics.color_group.clone().map(|group| {
            let mut twin = descriptor.clone();
            // A twin is no state an indirect command names: it takes no ID.
            twin.support_indirect_command_buffers = false;
            (twin, group)
        });
        let descriptor = StateDescriptor::Vertex(descriptor.clone());
        device.degrade_absent_reads(&base.shared_pipeline().dependencies, |stage| {
            descriptor.stage(stage).map_or("none".into(), stage_name)
        });
        device.degrade_synthesized_geometry(&base.shared_pipeline().dependencies, || {
            let name = |stage| descriptor.stage(stage).map_or("none".into(), stage_name);
            format!(
                "the render pipeline state of vertex function {} and fragment function {}",
                name(metal2vulkan::pipeline::Stage::Vertex),
                name(metal2vulkan::pipeline::Stage::Fragment)
            )
        });
        let state = GraphicsState::of(
            device,
            (descriptor, None),
            (base, large),
            (stand_in, placement, options.label.clone()),
            added_unsupported,
            Some(*descriptor_key),
        )?;
        state.make_neutrals();
        // Its first draw's predicted `VkPipeline`, made where Metal compiles (`predicted.rs`):
        // a state drawn in color groups draws as its twins, which predict their own.
        if grouped.is_none() {
            state.make_at_creation();
        }
        // The pipelines its tested draws' depth stand-in copies by, made where Metal compiles.
        state.make_stand_in_copies();
        if let Some((twin, grouping)) = grouped {
            let mut twins = Vec::with_capacity(group_bases.len());
            for (group, (base, stand_in)) in (0..).zip(group_bases) {
                // The group's attachments at their locations (`Program::drawn`).
                let mut of_group = grouping.clone();
                of_group.group = group;
                let mut drawn = twin.clone();
                drawn.color_attachments = Default::default();
                for (slot, &index) in drawn.color_attachments.iter_mut().zip(of_group.colors()) {
                    let color = twin
                        .color_attachments
                        .get(index as usize)
                        .cloned()
                        .flatten();
                    *slot = color;
                }
                let made = GraphicsState::of(
                    device,
                    (
                        StateDescriptor::Vertex(twin.clone()),
                        Some(StateDescriptor::Vertex(drawn)),
                    ),
                    (base, None),
                    (stand_in, IdPlacement::Any, options.label.clone()),
                    added_unsupported,
                    None,
                )?;
                made.make_neutrals();
                made.make_at_creation();
                made.make_stand_in_copies();
                twins.push(made);
            }
            state
                .0
                .color_groups
                .set(twins)
                .map_err(|_| internal("a state's color groups made twice"))?;
        }
        Ok(state)
    }

    /// `new_mesh_render_pipeline_state` of `descriptor` under `options` on `device`
    /// ([`Device::new_mesh_render_pipeline_state`]). Each stage's dynamic libraries are loaded
    /// first, and the state keeps the descriptor with them.
    pub(crate) fn create_mesh(
        device: &Device,
        descriptor: &MeshPipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<GraphicsState, Error> {
        const CALL: &str = "new_mesh_render_pipeline_state";
        let _noted = NoteEntries(device);
        // Metal's descriptor validation stops an attachment in a format Metal describes none
        // of, first (`attachment_formats.rs`).
        super::attachment_formats::check(
            CALL,
            &descriptor.color_attachments,
            descriptor.depth_format,
            descriptor.stencil_format,
        )?;
        super::raster_samples::check(CALL, descriptor.raster_sample_count)?;
        // A state an indirect command buffer's commands may set takes its ID where the options
        // place it, as a render state's does.
        let placement = if descriptor.support_indirect_command_buffers {
            pipeline_placement(CALL, options.placement)?
        } else {
            IdPlacement::Any
        };
        let mut loaded = descriptor.clone();
        loaded.object = loaded
            .object
            .as_ref()
            .map(crate::dynamic_library::stage_with_dependencies);
        loaded.mesh = crate::dynamic_library::stage_with_dependencies(&loaded.mesh);
        loaded.fragment = loaded
            .fragment
            .as_ref()
            .map(crate::dynamic_library::stage_with_dependencies);
        let descriptor = &loaded;
        // No binary archive holds a mesh render pipeline (`BinaryArchive` adds render, compute
        // and tile pipelines): under `FAIL_ON_BINARY_ARCHIVE_MISS` every listed archive misses
        // its mesh function.
        binary_archive::check(device, options, CALL, ArchivedPipeline::Render, |_, _| {
            Some(&descriptor.mesh)
        })?;
        let budget = options.budget();
        // As a render state's, in Metal's order: a fragment function writing
        // depth without a depth attachment, the indirect command buffers' refusal, then the
        // fragment function's stage inputs the linking keeps that the mesh function does not
        // write (`linked_fragment.rs`), before the compile.
        // The device's specialized stage entries, as a drawing state's creation reads them.
        let entries = device.shared().shader_cache.memo.stage_entries();
        let counts = &device.shared().counts;
        counts.linked_use_question_asked();
        super::linked_fragment::check_depth(
            CALL,
            descriptor.fragment_writes_depth_unattached(&budget, entries),
        )?;
        if descriptor.support_indirect_command_buffers {
            counts.linked_use_question_asked();
            super::indirect_arguments::check_render(
                CALL,
                descriptor.stages_using_direct_arguments(&budget, entries),
            )?;
        }
        counts.linked_use_question_asked();
        super::linked_fragment::check_inputs(
            CALL,
            "mesh",
            descriptor.unwritten_fragment_inputs(&budget, entries),
        )?;
        let compile = device.compile_device();
        let (pipeline, key) = stand_in::compile_with_stand_ins(compile.profile(), |key| {
            compile.compile_mesh_variant(descriptor, key, &budget)
        })
        .map_err(|error| compile_error(device, error))?;
        let stand_in = !key.storage_images.is_empty();
        let base = RenderVariant::of(
            device,
            key,
            CompiledVariant::found(Remembered::Compiled(Arc::new(pipeline))),
        )?;
        let descriptor = StateDescriptor::Mesh(descriptor.clone());
        device.degrade_absent_reads(&base.shared_pipeline().dependencies, |stage| {
            descriptor.stage(stage).map_or("none".into(), stage_name)
        });
        let state = GraphicsState::of(
            device,
            (descriptor, None),
            (base, None),
            (stand_in, placement, options.label.clone()),
            [0, 0],
            None,
        )?;
        state.make_neutrals();
        Ok(state)
    }

    /// Make the device's neutrals the plans of its compiled variants, its base and its
    /// large-buffer form, may bind (`binding::make_neutrals`), at the state's creation, which
    /// may wait, so that neither a draw's encoding nor a host step, a serial point of the command
    /// buffers' stream that may draw it (an execution's, of a state made after the commit
    /// included, `render/executions.rs`), makes one: a neutral image takes memory of its own, and
    /// NVIDIA's `vkAllocateMemory` waited 2.37 s for other processes' GPU work inside an encoding.
    /// Each plan is the one for nothing bound, which writes a neutral at every slot a bind could
    /// leave unbound, each of the kind its function declares, the kind a bound texture the slot
    /// cannot take is replaced with too; another variant of the state, of the same functions,
    /// declares the same textures, samplers and heap arrays. The commit makes those of the
    /// variants it settles (`RenderCommandEncoder::settle_draw_of`), a shader-side stencil's
    /// neutral among them; a host step's draw of a state made after the commit on a variant whose
    /// plan asks for a neutral neither made makes it (`neutral.rs`). Nothing where the state is
    /// made in a host step (a twin: the state it is a twin of, of the same functions, made them).
    /// A plan or a neutral
    /// that fails is left to the draw that binds it, which meets it as it is then. Then room kept
    /// in the memory the encodings place their own resources in (`Device::keep_room`).
    fn make_neutrals(&self) {
        if crate::exit_guard::in_host_step() {
            return;
        }
        let device = &self.0.device;
        for variant in std::iter::once(&self.0.base).chain(&self.0.large) {
            if let Ok(plan) = variant.plan(device, &BindShape::default()) {
                let _ = crate::binding::make_neutrals(device, &plan);
            }
        }
        device.keep_room();
    }

    /// The state of `descriptor` whose creation compiled `base`, under stand-ins where
    /// `stand_in`, and `large`, its large-buffer form's variant where it made one, its ID placed
    /// by `placement` where its descriptor supports indirect command buffers, with
    /// `added_unsupported` (`StateShared::added_unsupported`); drawn as `drawn` where given
    /// (`Program::drawn`).
    ///
    /// `key` is the key of the descriptor it draws as, where the caller holds it: a creation
    /// built that key to look its compiles up under, so the state keeps it rather than building
    /// it again the first time something reads it ([`GraphicsState::descriptor_key`]). `None`
    /// where the caller's key is of another descriptor -- a twin of a color group draws as its
    /// group's attachments alone, and the mesh creation builds no key -- and the state builds its
    /// own when it is first asked.
    fn of(
        device: &Device,
        (descriptor, drawn): (StateDescriptor, Option<StateDescriptor>),
        (base, large): (Arc<RenderVariant>, Option<Arc<RenderVariant>>),
        (stand_in, placement, label): (bool, IdPlacement, Option<String>),
        added_unsupported: [usize; 2],
        key: Option<PipelineKey>,
    ) -> Result<GraphicsState, Error> {
        let identifier = UniqueIdentifier::state(
            &device.shared().identifiers,
            StateKind::Render,
            descriptor.support_indirect_command_buffers(),
        )?;
        let samplers = descriptor.number_samplers(device, &base.shared_pipeline().interface);
        let at_bind_views = AtBindViews::new(device, &base.shared_pipeline().interface);
        let log = StateLog::new(label.as_deref(), &descriptor.function_names());
        Ok(GraphicsState(Arc::new_cyclic(|state| {
            // The state holds its ID's slot, which lists it among the states an indirect render
            // command's record can name (`indirect_states`).
            let id = descriptor.support_indirect_command_buffers().then(|| {
                let id = IdHold::take(&device.shared().resource_ids, IdSpace::Pipeline, placement);
                id.hold_state(
                    Holder::RenderPipeline(Weak::clone(state)),
                    StateKind::Render,
                    identifier.value(),
                );
                id
            });
            let program = Program {
                device: device.clone(),
                descriptor,
                drawn,
                base,
                stand_in,
                others: Mutex::new(Lru::counted(VARIANTS)),
                large,
                scaling: Mutex::new(None),
                added_unsupported,
                single_sample: Mutex::new(None),
                color_groups: OnceLock::new(),
                log,
                label,
                texture_accesses: TextureAccessMemo::new(),
                descriptor_key: key.map_or_else(OnceLock::new, OnceLock::from),
                compiled_key: OnceLock::new(),
                settled: Mutex::new(VecDeque::with_capacity(SETTLED)),
                last_pipelines: Mutex::new(VecDeque::with_capacity(LAST_PIPELINES)),
                routed: Mutex::new(None),
                pieces: PieceHolds::new(),
                at_bind_views,
            };
            StateShared {
                id,
                identifier: Some(identifier),
                _samplers: samplers,
                program: Arc::new(program),
            }
        })))
    }

    /// The descriptor the state draws as: the one it was made from, but for a twin of a color
    /// group, whose colors are its group's alone (`Program::drawn`).
    pub(crate) fn descriptor(&self) -> &StateDescriptor {
        self.0.drawn.as_ref().unwrap_or(&self.0.descriptor)
    }

    /// Its functions' texture slots and their accesses, from its slot writes the first time.
    ///
    /// Not from its reflection: that refuses the whole state where Metal's answer for one
    /// argument's type has not been measured, which took every slot's access with it
    /// (`pipeline::texture_access`). The slot writes read the functions' roles, declared accesses
    /// and slots alone, so a state that compiles answers.
    fn texture_accesses(&self) -> Result<&TextureAccesses, Error> {
        TextureAccesses::remembered(&self.0.texture_accesses, |budget| {
            match self.descriptor() {
                StateDescriptor::Vertex(descriptor) => descriptor.slot_writes(budget),
                StateDescriptor::Mesh(descriptor) => descriptor.slot_writes(budget),
            }
            .map_err(device_error)
        })
    }

    /// The pipeline's reflection ([`RenderPipelineState::compiled_reflection`]) of the descriptor
    /// it draws as (with the dynamic libraries its creation loaded), as its creation's
    /// out-parameter answers it: the shader cache's for the descriptor's key, worked out under
    /// `budget` and kept where it holds none, for a mesh pipeline as for a drawing one -- so every
    /// state of one descriptor, and every read of one state's, share the one answer, where Metal
    /// answers each from the library it compiled once.
    pub(crate) fn reflection(
        &self,
        budget: &metal2vulkan::Budget,
    ) -> Result<RenderPipelineReflection, Error> {
        let device = &self.0.device;
        let compile = device.compile_device();
        let key = self.descriptor_key();
        let cache = &device.shared().shader_cache;
        let work = || match self.descriptor() {
            StateDescriptor::Vertex(descriptor) => descriptor.reflection(budget),
            StateDescriptor::Mesh(descriptor) => descriptor.reflection(budget),
        };
        (cache.render_reflection(compile, key, work)).map_err(device_error)
    }

    /// The key of the descriptor the state draws as, built once and held
    /// ([`Program::descriptor_key`]).
    pub(crate) fn descriptor_key(&self) -> &PipelineKey {
        self.0.descriptor_key.get_or_init(|| {
            let device = &self.0.device;
            match self.descriptor() {
                StateDescriptor::Vertex(descriptor) => render_descriptor_key(device, descriptor),
                StateDescriptor::Mesh(descriptor) => mesh_descriptor_key(device, descriptor),
            }
        })
    }

    /// The key of the descriptor the state's variants compile from, built once and held
    /// ([`Program::compiled_key`]): what a draw's compile of another variant looks its pipeline
    /// up under, as its creation's compiles looked theirs up under the key the creation built.
    ///
    /// Not [`descriptor_key`](Self::descriptor_key) for a twin of a color group, which draws as
    /// its group's color attachments alone and compiles from the descriptor it was made from:
    /// its two descriptors differ in their color attachments, and in
    /// `supportIndirectCommandBuffers`, which the twin's descriptor clears -- both fields the
    /// key reads. Keying a compile by the descriptor it does not compile would look it up in
    /// another descriptor's entries, so each key stays with its own descriptor.
    pub(crate) fn compiled_key(&self) -> &PipelineKey {
        // A state drawn whole compiles from the descriptor it draws as: one descriptor, one key.
        if self.0.drawn.is_none() {
            return self.descriptor_key();
        }
        self.0.compiled_key.get_or_init(|| {
            let device = &self.0.device;
            match &self.0.descriptor {
                StateDescriptor::Vertex(descriptor) => render_descriptor_key(device, descriptor),
                StateDescriptor::Mesh(descriptor) => mesh_descriptor_key(device, descriptor),
            }
        })
    }

    /// The state a patch draw an indirect command scales runs on (`None` for a state that does
    /// not tessellate, whose draw refuses patches): its twin, of this state's descriptor with
    /// `tessellationFactorScaleEnabled` set and without `supportIndirectCommandBuffers`, made
    /// with the default options when a command first asks for it and kept while the state lives.
    /// Its control stage multiplies the factors by the encoder's scale, which the execution sets
    /// to the command's: Metal multiplies them by a kernel's scale whatever the pipeline, while a
    /// state made with `supportIndirectCommandBuffers` never enables the scale. What making it
    /// refuses refuses the draw.
    pub(crate) fn factor_scaling(&self) -> Result<Option<GraphicsState>, Error> {
        if crate::tessellation::patches(&self.base().graphics).is_none() {
            return Ok(None);
        }
        let Some(descriptor) = self.0.descriptor.vertex() else {
            return Err(internal("a tessellating mesh render pipeline state"));
        };
        let mut held = lock(&self.0.scaling);
        if let Some(twin) = held.as_ref() {
            return Ok(Some(twin.clone()));
        }
        let mut descriptor = descriptor.clone();
        descriptor.support_indirect_command_buffers = false;
        descriptor
            .tessellation
            .get_or_insert_with(TessellationDescriptor::default)
            .factor_scale_enabled = true;
        let twin = GraphicsState::create(
            &self.0.device,
            &descriptor,
            &PipelineOptions::default(),
            None,
        )?;
        *held = Some(twin.clone());
        Ok(Some(twin))
    }

    /// The state a pass of one sample draws this one as where the state rasterizes more samples
    /// (`None` for a mesh render pipeline's state, unmeasured): its twin, of this state's
    /// descriptor with `rasterSampleCount` 1, made with the default options when such a pass
    /// first takes the state and kept while the state lives. Plain Metal draws a 2- or 4-sample
    /// state in a 1-sample pass as a 1-sample one but for alpha to coverage and the fragment
    /// function's sample count, which the encoder refuses (`set_render_pipeline_state`). What
    /// making it refuses refuses the call.
    pub(crate) fn single_sample(&self) -> Result<Option<GraphicsState>, Error> {
        let Some(descriptor) = self.0.descriptor.vertex() else {
            return Ok(None);
        };
        let mut held = lock(&self.0.single_sample);
        if let Some(twin) = held.as_ref() {
            return Ok(Some(twin.clone()));
        }
        let mut descriptor = descriptor.clone();
        descriptor.raster_sample_count = 1;
        let twin = GraphicsState::create_adding(
            &self.0.device,
            &descriptor,
            &PipelineOptions::default(),
            None,
            self.0.added_unsupported,
        )?;
        *held = Some(twin.clone());
        Ok(Some(twin))
    }

    /// The state a pass of color group `group` draws this one as, where the device draws its
    /// color attachments in groups (`metal2vulkan::pipeline::ColorGroup`, the base's
    /// `GraphicsState::color_group`): its twin of that group, of this state's descriptor, made at
    /// its creation, drawn as the group's colors alone (`Program::drawn`), whose every variant
    /// draws that group's attachments (`VariantKey::color_group`), the first group's on this
    /// state's base. `None` for a group the state does not have, and for a state the device draws
    /// whole.
    pub(crate) fn color_group(&self, group: u32) -> Option<GraphicsState> {
        self.0
            .color_groups
            .get()?
            .get(usize::try_from(group).ok()?)
            .cloned()
    }

    /// A function of the state reads its raster sample count (`get_num_samples()`, the
    /// `RuntimeFact::SampleCount` it records): of a state of one sample, only that call does.
    pub(crate) fn reads_sample_count(&self) -> bool {
        self.base()
            .dependencies
            .iter()
            .any(|d| matches!(d.fact, RuntimeFact::SampleCount))
    }

    /// `device`.
    pub(crate) fn device(&self) -> &Device {
        &self.0.device
    }

    /// Whether `other` is this state (a clone of it, or the state a command runs once the
    /// program released it, `RenderProgram::state`), not another state alike.
    pub(crate) fn is(&self, other: &GraphicsState) -> bool {
        Arc::ptr_eq(&self.0.program, &other.0.program)
    }

    /// How many of `stage`'s function's last `binary_functions` were added without the stage's
    /// support flag (`StateShared::added_unsupported`): none for another stage.
    pub(crate) fn added_unsupported(&self, stage: metal2vulkan::pipeline::Stage) -> usize {
        match stage {
            metal2vulkan::pipeline::Stage::Vertex => self.0.added_unsupported[0],
            metal2vulkan::pipeline::Stage::Fragment => self.0.added_unsupported[1],
            _ => 0,
        }
    }

    /// The functions `stage`'s function links (`vertexLinkedFunctions`,
    /// `fragmentLinkedFunctions`, and a mesh render pipeline's `objectLinkedFunctions` and
    /// `meshLinkedFunctions`): its descriptor's; `None` for a stage the state has no function
    /// of, or whose function the compile layer already compiled (`StageSource::Compiled`), which
    /// links nothing more.
    pub(crate) fn linked_functions(
        &self,
        stage: metal2vulkan::pipeline::Stage,
    ) -> Option<&metal2vulkan::pipeline::LinkedFunctions> {
        match self.0.descriptor.stage(stage)? {
            StageSource::Function(function) => Some(&function.linked),
            StageSource::Compiled(_) => None,
        }
    }

    /// The pipeline compiled at creation, which answers what every variant shares (its graphics
    /// state's formats, samples and topology class).
    pub(crate) fn base(&self) -> &RenderPipeline {
        &self.0.base.pipeline
    }

    /// Its buffer slots whose bind offset class tells the device's view base alignment apart
    /// (`argument_table::AtBindViews`).
    pub(crate) fn at_bind_views(&self) -> &AtBindViews {
        &self.0.at_bind_views
    }

    /// The render stages whose functions read texture handles from memory (a handle table): only
    /// there does a draw reach a resident heap's textures. A kernel stage of the pipeline, which no
    /// render stage names, is taken as every one.
    /// Whether `other` is this state: the same object, not an equal one.
    pub(crate) fn same(&self, other: &GraphicsState) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Another reference to the state, which a command buffer holds it by
    /// (`EncoderCore::retain_shared`).
    pub(crate) fn held(&self) -> Arc<dyn Any + Send + Sync> {
        self.0.clone()
    }

    /// Its routing into a pass of the colors `pass` (`RenderCommandEncoder::routing`), as `route`
    /// finds it: a function of the state and the pass's colors alone, so it is found once for the
    /// latest pass colors and kept, and a pass of other colors finds it again.
    pub(crate) fn routed(
        &self,
        pass: &PassColors,
        route: impl FnOnce() -> Result<Routing, u32>,
    ) -> Result<Arc<Routing>, u32> {
        let mut routed = lock(&self.0.routed);
        if let Some((_, routing)) = routed.as_ref().filter(|(colors, _)| colors == pass) {
            return routing.clone();
        }
        let routing = route().map(Arc::new);
        *routed = Some((*pass, routing.clone()));
        routing
    }

    pub(crate) fn texture_handle_stages(&self) -> RenderStages {
        self.0.base.handle_stages
    }

    /// Whether the vertex function writes `[[render_target_array_index]]`: it depends on the
    /// pass's array length (`RuntimeFact::RenderTargetArrayLength`) in every variant, whether or
    /// not the variant stores the index.
    pub(crate) fn vertex_chooses_slice(&self) -> bool {
        self.base()
            .dependencies
            .iter()
            .any(|d| matches!(d.fact, RuntimeFact::RenderTargetArrayLength { .. }))
    }

    /// Whether the fragment function reads its threadgroup arguments, its tile's threadgroup
    /// memory, which the pass keeps (`RuntimeFact::TileThreadgroupMemory`): its draws bind the
    /// pass's buffer of every tile's (`render/tile_memory.rs`).
    pub(crate) fn reads_tile_memory(&self) -> bool {
        self.base()
            .dependencies
            .iter()
            .any(|d| matches!(d.fact, RuntimeFact::TileThreadgroupMemory { .. }))
    }

    /// The variant a draw of a bind state of `shape` runs on in `forms`, and its plan for
    /// `shape`: the first variant's plan, or that of the variant it names, compiled when first
    /// needed. In the large-buffer form (an execution's host step, `render/executions.rs`) every
    /// bind shape of the same bound textures and samplers settles on one variant, whatever buffers
    /// it binds, as a compute command's does.
    pub(crate) fn settle(
        &self,
        shape: &BindShape,
        forms: BufferForms,
    ) -> Result<(Arc<RenderVariant>, Arc<BindingPlan>), Error> {
        let base = Arc::clone(&self.0.base);
        let plan = base.plan(&self.0.device, shape)?;
        let wanted = match forms {
            BufferForms::Bound => plan.variant.clone(),
            BufferForms::Large => plan
                .variant
                .large_buffer_form(&base.pipeline.dependencies, shape),
            BufferForms::Replay | BufferForms::Merged => {
                let mut key = plan.variant.clone();
                key.replay_bindings = true;
                key.merged_draws = forms == BufferForms::Merged;
                key
            }
        };
        // A stand-in compile is never drawn: a plan whose bound formats are the stand-ins' takes
        // a variant of its own (the same compile, from the shader cache).
        if wanted == base.key && !self.0.stand_in {
            return Ok((base, plan));
        }
        let variant = self.variant(&wanted)?;
        let plan = variant.plan(&self.0.device, shape)?;
        let runs = match forms {
            BufferForms::Bound | BufferForms::Replay | BufferForms::Merged => {
                plan.variant == variant.key
            }
            BufferForms::Large => variant.key.covers(&plan.variant),
        };
        if !runs {
            return Err(internal(
                "the plan of a compiled variant asks for another variant",
            ));
        }
        Ok((variant, plan))
    }

    /// The shape of `state` among those of what the state settled lately
    /// ([`settle_shape`](Self::settle_shape)), compared with the state in place
    /// (`BindState::position_of_shape`, which makes the shape of the resources it lists at most
    /// once for all of them): what a draw of `state` settles where it has one, and finds there,
    /// without its shape made (`live_bind.rs`).
    pub(crate) fn kept_shape(&self, state: &BindState) -> Option<Arc<BindShape>> {
        let settled = lock(&self.0.settled);
        let at = state.position_of_shape(settled.iter().map(|(_, last)| &*last.shape))?;
        Some(Arc::clone(&settled[at].1.shape))
    }

    /// What a draw of bind shape `shape` runs on in `forms` ([`settle`](Self::settle)'s answer),
    /// through what the state settled last for its latest [`SETTLED`] shapes: a draw of an equal
    /// shape in the same forms (the very one, which its encoder kept, `live_bind.rs`, looked for
    /// first by identity, or an equal one, compared whole where none is the very one) settles as
    /// it did, not planned again; any other settles, and is kept in place of the
    /// one used least lately. Exact: what `settle` answers is a function of the shape and `forms`
    /// alone. A refused settle keeps nothing, and is answered as `settle` refuses it.
    pub(crate) fn settle_shape(
        &self,
        shape: &Arc<BindShape>,
        forms: BufferForms,
    ) -> Result<Settled, Error> {
        {
            let mut kept = lock(&self.0.settled);
            // The very shape first, as a draw's encoder keeps the one it found here; else an
            // equal one, compared whole.
            let found = (kept.iter())
                .position(|(kept, last)| *kept == forms && Arc::ptr_eq(&last.shape, shape))
                .or_else(|| {
                    (kept.iter()).position(|(kept, last)| *kept == forms && last.shape == *shape)
                });
            if let Some(entry) = found.and_then(|at| kept.remove(at)) {
                let last = entry.1.clone();
                kept.push_front(entry);
                return Ok(last);
            }
        }
        self.0.device.shared().counts.draw_settled();
        let (variant, plan) = self.settle(shape, forms)?;
        let settled = Settled {
            shape: Arc::clone(shape),
            variant,
            plan,
        };
        let mut kept = lock(&self.0.settled);
        if kept.len() == SETTLED {
            kept.pop_back();
        }
        kept.push_front((forms, settled.clone()));
        Ok(settled)
    }

    /// Make the pipelines the depth stand-in's copies take (`depth_stand_in::make_copies`), where
    /// a draw whose depth test alone can drop fragments runs the variant whose fragment function
    /// tests the depth itself (`VariantKey::shader_tests`): the binder's plan of the base for
    /// such a draw (`FragmentTests::Depth`) says so. That variant is a draw's, made by its first
    /// draw (`predicted.rs`); the copies' pipelines depend only on the depth attachment's format
    /// and samples, so its creation makes them, where Metal compiles.
    pub(crate) fn make_stand_in_copies(&self) {
        let graphics = &self.base().graphics;
        let Some(format) = graphics.depth_format else {
            return;
        };
        let mut shape = BindShape::default();
        shape.fragment_tests = FragmentTests::Depth { writes: true };
        let tested = self
            .0
            .base
            .plan(&self.0.device, &shape)
            .is_ok_and(|plan| plan.variant.shader_tests);
        if tested {
            let samples = graphics.rasterization_samples.as_raw().max(1);
            crate::depth_stand_in::make_copies(&self.0.device, samples, format);
        }
    }

    /// The variant a creation predicts its first draw's key on (`predicted.rs`): the base of a
    /// render descriptor's state, untessellated and not compiled under stand-ins; `None` for any
    /// other.
    pub(crate) fn predicting_variant(&self) -> Option<&Arc<RenderVariant>> {
        let StateDescriptor::Vertex(_) = &self.0.descriptor else {
            return None;
        };
        // A post-tessellation vertex function's pipeline tessellates where the descriptor's
        // tessellation state is `None` (Metal's defaults) too.
        let untessellated = crate::tessellation::patches(&self.base().graphics).is_none();
        (untessellated && !self.0.stand_in).then_some(&self.0.base)
    }

    /// The base variant and its plan for `shape`, where a draw of that bind shape runs on the
    /// base ([`settle`](Self::settle)'s first answer); `None` where it needs another variant,
    /// which only a draw makes, or where the plan refuses the shape.
    pub(crate) fn base_plan(
        &self,
        shape: &BindShape,
    ) -> Option<(Arc<RenderVariant>, Arc<BindingPlan>)> {
        let base = Arc::clone(&self.0.base);
        let plan = base.plan(&self.0.device, shape).ok()?;
        (plan.variant == base.key && !self.0.stand_in).then_some((base, plan))
    }

    /// The variant of `key`: the large-buffer form's compiled at creation, or another compiled
    /// when first needed.
    fn variant(&self, key: &VariantKey) -> Result<Arc<RenderVariant>, Error> {
        if let Some(large) = self.0.large.as_ref().filter(|large| large.key == *key) {
            return Ok(Arc::clone(large));
        }
        let mut others = lock(&self.0.others);
        if let Some(found) = others.get(key) {
            return Ok(Arc::clone(found));
        }
        let sources = Sources::later(&self.0.device);
        let found = match &self.0.descriptor {
            StateDescriptor::Vertex(descriptor) => RenderVariant::compiled(
                &sources,
                descriptor,
                key,
                self.0.device.shared().shader_cache.memo.stage_entries(),
                self.compiled_key(),
            ),
            // A mesh render pipeline is compiled as it is asked for, past the memo, the archives
            // and the disk cache, which hold the other render pipelines.
            StateDescriptor::Mesh(descriptor) => self
                .0
                .device
                .compile_device()
                .compile_mesh_variant(descriptor, key, &sources.budget())
                .map(|pipeline| CompiledVariant::found(Remembered::Compiled(Arc::new(pipeline)))),
        }
        .map_err(|error| compile_error(&self.0.device, error))?;
        let variant = RenderVariant::of(&self.0.device, key.clone(), found)?;
        others.insert(key.clone(), Arc::clone(&variant), 1);
        Ok(variant)
    }
}

/// The base variant of each color group of the state of `descriptor` whose creation compiled
/// `base` (under stand-ins where `stand_in`), with whether it was compiled under stand-ins
/// (`GraphicsState::color_group`): the first group's is `base`, each later one's compiled as the
/// base was, under the empty key of that group (`VariantKey::color_group`), or under the
/// stand-ins its compile asks for. None where the device draws the pipeline's colors whole. Its
/// stages' specialized entries are the creation's (`entries`), and `descriptor_key` is the key
/// of `descriptor` the creation built.
fn group_bases(
    sources: &Sources<'_>,
    descriptor: &RenderPipelineDescriptor,
    entries: &StageEntries,
    (base, stand_in): (&Arc<RenderVariant>, bool),
    descriptor_key: &PipelineKey,
) -> Result<Vec<(Arc<RenderVariant>, bool)>, Error> {
    let device = sources.device();
    let Some(groups) = base
        .pipeline
        .graphics
        .color_group
        .as_ref()
        .map(|g| g.groups())
    else {
        return Ok(Vec::new());
    };
    let profile = device.compile_device().profile();
    let later = (1..groups).map(|group| {
        let (found, mut key) = stand_in::compile_with_stand_ins(profile, |key| {
            let mut key = key.clone();
            key.color_group = group;
            RenderVariant::compiled(sources, descriptor, &key, entries, descriptor_key)
        })
        .map_err(|error| compile_error(device, error))?;
        key.color_group = group;
        let stand_in = !key.storage_images.is_empty();
        Ok((RenderVariant::of(device, key, found)?, stand_in))
    });
    std::iter::once(Ok((Arc::clone(base), stand_in)))
        .chain(later)
        .collect()
}

/// The variant in the large-buffer form of the state of `descriptor` whose creation compiled
/// `base` (under stand-ins where `stand_in`), compiled now, where Metal compiles: the variant an
/// execution's host step draws a command the GPU may write on (`BufferForms::Large`,
/// `render/executions.rs`) where every buffer the command reaches has a device address (no slot
/// of an empty bind shape is addressless). The commit that settles such a draw then looks it
/// up. `None` for a state that does not support indirect command buffers, that tessellates
/// (whose draws the step compiles), whose base was compiled under stand-ins (its draws' formats
/// unknown until they bind them), or whose large form is its base; and where the compile
/// refuses it, which the draw that asks for it meets as it did. Its stages' specialized entries
/// are the creation's (`entries`), and `descriptor_key` is the key of `descriptor` the creation
/// built.
fn large_form(
    device: &Device,
    descriptor: &RenderPipelineDescriptor,
    (base, entries): (&Arc<RenderVariant>, &StageEntries),
    stand_in: bool,
    descriptor_key: &PipelineKey,
) -> Option<Arc<RenderVariant>> {
    if !descriptor.support_indirect_command_buffers
        || stand_in
        || crate::tessellation::patches(&base.pipeline.graphics).is_some()
    {
        return None;
    }
    let key = base
        .key
        .large_buffer_form(&base.pipeline.dependencies, &BindShape::default());
    if key == base.key {
        return None;
    }
    let found = RenderVariant::compiled(
        &Sources::later(device),
        descriptor,
        &key,
        entries,
        descriptor_key,
    )
    .ok()?;
    RenderVariant::of(device, key, found).ok()
}

/// Whether `answers` holds an answer to every linked-use question a creation of `descriptor`
/// asks (`GraphicsState::create_adding`): whether its vertex function returns a value where it
/// does not rasterize, the stages using a direct argument where it supports indirect command
/// buffers, and always the fragment's depth write and the unwritten fragment inputs.
fn answers_every_question(
    answers: &LinkedUseAnswers,
    descriptor: &RenderPipelineDescriptor,
) -> bool {
    (descriptor.rasterization_enabled || answers.vertex_returns_value.is_some())
        && answers.fragment_writes_depth_unattached.is_some()
        && (!descriptor.support_indirect_command_buffers
            || answers.stages_using_direct_arguments.is_some())
        && answers.unwritten_fragment_inputs.is_some()
}

/// The key of render pipeline descriptor `d` on `device` (`Device::render_descriptor_key`),
/// counted. Building one digests every field of the descriptor in its derived `Debug` form and
/// hashes each stage's compile key, so the runtime builds one per creation and passes it to
/// everything that wants it rather than building it again at each site; every build it makes goes
/// through here or [`mesh_descriptor_key`], so `descriptor_keys_built` is how many a call built
/// (`work_counts.rs`).
fn render_descriptor_key(device: &Device, d: &RenderPipelineDescriptor) -> PipelineKey {
    device.shared().counts.descriptor_key_built();
    device.compile_device().render_descriptor_key(d)
}

/// The key of mesh render pipeline descriptor `d` on `device`
/// (`Device::mesh_descriptor_key`), counted as [`render_descriptor_key`] counts a render
/// descriptor's.
fn mesh_descriptor_key(device: &Device, d: &MeshPipelineDescriptor) -> PipelineKey {
    device.shared().counts.descriptor_key_built();
    device.compile_device().mesh_descriptor_key(d)
}

/// A stage's function name, or the compile key of a stage compiled already.
fn stage_name(source: &StageSource) -> String {
    match source {
        StageSource::Function(descriptor) => descriptor.function.info().name.to_string(),
        StageSource::Compiled(compiled) => format!("{:?}", compiled.key()),
    }
}

/// What a render variant's compile gives back ([`RenderVariant::compiled`]): the pipeline, and
/// the compiled functions the link was given where this creation did the linking.
///
/// The functions are what lets a creation name its stages' modules without hashing their words
/// again ([`stage_words_digest`]). Empty for a pipeline the memo, a binary archive or the disk
/// cache gave whole, and for a mesh pipeline, none of which link from compiled functions here.
struct CompiledVariant {
    remembered: Remembered<RenderPipeline, RenderVariant>,
    functions: Vec<CompiledFunction>,
}

impl CompiledVariant {
    /// What the memo, an archive or the disk cache gave, with no compiled function to ask about
    /// its stages.
    fn found(remembered: Remembered<RenderPipeline, RenderVariant>) -> CompiledVariant {
        CompiledVariant {
            remembered,
            functions: Vec::new(),
        }
    }
}

/// The digest of `stage`'s words ([`words_digest`]): the one a compiled function of the creation
/// holds for the very words the stage took from it, else one taken here.
///
/// **A stage cannot be matched to a function by position.** A pipeline's stages carry the ones
/// the link synthesizes -- a post-tessellation vertex function's vertex and tessellation control
/// stages, the geometry stages a barycentric or layer-selecting function needs, one per topology
/// class -- spliced in among the compiled ones, so stage `i` is not function `i`. What matches
/// them is the identity of the words: a compiled stage's words are the very allocation its
/// function's relocation slot holds, which is what `CompiledFunction::relocated_words_digest`
/// answers for and nothing else. A stage no function claims is a synthesized one, whose words no
/// slot ever held, and it is hashed here as every stage was before.
fn stage_words_digest(functions: &[CompiledFunction], stage: &StageModule) -> Hash256 {
    functions
        .iter()
        .find_map(|function| function.relocated_words_digest(&stage.spirv, words_digest))
        .unwrap_or_else(|| words_digest(&stage.spirv))
}

impl RenderVariant {
    /// `descriptor`'s pipeline under `key`: the variant of it the memo knows live, or the
    /// pipeline `sources` give, else each `Function` stage's translation (found, or compiled)
    /// linked, all under one budget. A refusal is the compile layer's.
    ///
    /// `descriptor_key` is `descriptor`'s own key (`render_descriptor_key`), which the caller
    /// built or holds: a creation compiles its base, its large-buffer form and each of its color
    /// groups' bases, all of one descriptor, so it builds that key once and every one of them
    /// looks its pipeline up under it; a state holds the key of each descriptor its later
    /// compiles are of (`GraphicsState::compiled_key`), so a draw builds none.
    fn compiled(
        sources: &Sources<'_>,
        descriptor: &RenderPipelineDescriptor,
        key: &VariantKey,
        entries: &StageEntries,
        descriptor_key: &PipelineKey,
    ) -> Result<CompiledVariant, metal2vulkan::error::Error> {
        let compile = sources.device().compile_device();
        let budget = sources.budget();
        // The pipeline a creation of a descriptor of the same key learned for `key`, found
        // without asking its keys again; else its stages' functions, found by the keys that
        // creation learned, linked with no stage decoded; else the keys, kept for the next one.
        let cache = &sources.device().shared().shader_cache;
        let mut asked = None;
        if let Some(kept) = cache.variant_pipeline(compile, descriptor_key, key) {
            if let Some(found) = sources.render(&kept.pipeline) {
                return Ok(CompiledVariant::found(found));
            }
            if let Some(linked) = Self::kept_stages(sources, descriptor, &kept) {
                return Self::link(sources, linked, key, None);
            }
            asked = Some(kept.pipeline);
        }
        let keys = compile.render_variant_key(descriptor, key, &budget, entries)?;
        let stages = VariantStages {
            pipeline: keys.pipeline,
            vertex: keys.vertex,
            fragment: keys.fragment,
        };
        let unrecorded = cache.remember_variant_pipeline(descriptor_key, key, &stages);
        let record = unrecorded.then_some((descriptor_key, &stages));
        // The key a record gave is the key this creation computes, so asking for its pipeline
        // again would repeat a lookup that has just missed. The keys are compared rather than
        // assumed equal: a record of another key is one this build does not recognize, and it is
        // looked up as any other.
        if asked != Some(keys.pipeline) {
            if let Some(found) = sources.render(&keys.pipeline) {
                if unrecorded {
                    cache.record_variant_pipeline(compile, descriptor_key, key, &stages);
                }
                return Ok(CompiledVariant::found(found));
            }
        }
        let mut linked = descriptor.clone();
        if let (StageSource::Function(function), Some(inputs)) =
            (&descriptor.vertex, &keys.vertex_inputs)
        {
            linked.vertex = StageSource::Compiled(sources.function(
                function,
                inputs,
                (&keys.vertex, Some(entries)),
                &budget,
            )?);
        }
        if let (Some(StageSource::Function(function)), Some(inputs), Some(fragment)) =
            (&descriptor.fragment, &keys.fragment_inputs, &keys.fragment)
        {
            linked.fragment = Some(StageSource::Compiled(sources.function(
                function,
                inputs,
                (fragment, Some(entries)),
                &budget,
            )?));
        }
        Self::link(sources, linked, key, record)
    }

    /// `descriptor` with each `Function` stage the function of the compile key `kept` names for
    /// it, found by that key alone ([`Sources::found_function`]): the descriptor a creation that
    /// learned those keys links. `None` where one is not found, or where `kept` names a stage the
    /// descriptor does not have or none for one it has: the creation learns the keys again.
    fn kept_stages(
        sources: &Sources<'_>,
        descriptor: &RenderPipelineDescriptor,
        kept: &VariantStages,
    ) -> Option<RenderPipelineDescriptor> {
        let mut linked = descriptor.clone();
        if let StageSource::Function(_) = &descriptor.vertex {
            linked.vertex = StageSource::Compiled(sources.found_function(&kept.vertex)?);
        }
        match (&descriptor.fragment, &kept.fragment) {
            (Some(StageSource::Function(_)), Some(fragment)) => {
                linked.fragment = Some(StageSource::Compiled(sources.found_function(fragment)?));
            }
            (Some(StageSource::Compiled(_)), _) | (None, None) => {}
            (Some(StageSource::Function(_)), None) | (None, Some(_)) => return None,
        }
        Some(linked)
    }

    /// The pipeline under `key` linked from `linked`, a descriptor with its stages compiled,
    /// kept as [`Sources::linked_render`] keeps it; and where `record` names
    /// the descriptor's key and the variant's keys, the directory's record of them
    /// (`ShaderCache::record_variant_pipeline`), which a later process finds what it names by:
    /// the pipeline's artifact where the directory took it, else the stages' functions, which
    /// the directory then holds.
    fn link(
        sources: &Sources<'_>,
        linked: RenderPipelineDescriptor,
        key: &VariantKey,
        record: Option<(&PipelineKey, &VariantStages)>,
    ) -> Result<CompiledVariant, metal2vulkan::error::Error> {
        let compile = sources.device().compile_device();
        let budget = sources.budget();
        let pipeline = compile.compile_render_variant(&linked, key, &budget)?;
        // The stages the link was given, whichever way they arrived: the ones this creation
        // compiled or found, and the ones the descriptor already carried compiled.
        let functions: Vec<CompiledFunction> = [Some(&linked.vertex), linked.fragment.as_ref()]
            .into_iter()
            .flatten()
            .filter_map(|source| match source {
                StageSource::Compiled(compiled) => Some(compiled.clone()),
                StageSource::Function(_) => None,
            })
            .collect();
        let stage_keys: Vec<FunctionCompileKey> = functions.iter().map(|f| *f.key()).collect();
        let pipeline = sources.linked_render(pipeline, &stage_keys);
        if let Some((descriptor_key, stages)) = record {
            let cache = &sources.device().shared().shader_cache;
            cache.record_variant_pipeline(compile, descriptor_key, key, stages);
        }
        Ok(CompiledVariant {
            remembered: Remembered::Compiled(pipeline),
            functions,
        })
    }

    /// The variant of `key` of what the memo or a compile `found`: the live variant, or one made
    /// of the pipeline, its objects made, which the memo then knows live.
    fn of(
        device: &Device,
        key: VariantKey,
        compiled: CompiledVariant,
    ) -> Result<Arc<RenderVariant>, Error> {
        let CompiledVariant {
            remembered,
            functions,
        } = compiled;
        let logged_spirv = OnceLock::new();
        let pipeline = match remembered {
            Remembered::Live(variant) if variant.key == key => return Ok(variant),
            Remembered::Live(variant) => {
                if let Some(logged) = variant.logged_spirv.get() {
                    let _ = logged_spirv.set(logged.clone());
                }
                Arc::clone(&variant.pipeline)
            }
            Remembered::Compiled(pipeline) => pipeline,
        };
        // Before the variant's shader modules, whose creation the layer reports.
        device.degrade_pipeline(CompiledPipeline::Render(&pipeline), Occurrence::Made);
        // Each stage's words' digest, where this creation linked the pipeline itself: the
        // digest its compiled function holds beside the very words the stage took, so a sweep of
        // states over one function pair hashes those words once rather than once a creation
        // ([`stage_words_digest`]). A pipeline the memo, an archive or the disk gave has no
        // compiled function to ask, and its objects hash their stages as they always did.
        let digests: Option<Vec<Hash256>> = (!functions.is_empty()).then(|| {
            pipeline
                .stages
                .iter()
                .map(|stage| stage_words_digest(&functions, stage))
                .collect()
        });
        // Where the creation log is on, the stages' SPIR-V it names hashed on a thread of its
        // own while the objects are made: the hash its first line reads, off the creation's path.
        let (objects, kept) = std::thread::scope(|scope| {
            if creation_log::enabled() && logged_spirv.get().is_none() {
                // A thread that cannot be made leaves the hash to the log's first line.
                let _ = crate::exit_guard::spawn_scoped(
                    scope,
                    std::thread::Builder::new().name("metal2vulkan-log-hash".into()),
                    || {
                        let _ = logged_spirv.set(creation_log::spirv_hashes(&pipeline));
                    },
                );
            }
            VariantObjects::kept_or_create(
                device,
                Compiled::Render(pipeline.key),
                &pipeline.layout,
                &pipeline.stages,
                CompiledPipeline::Render(&pipeline),
                digests.as_deref(),
            )
        })?;
        let stages = handle_stages(&pipeline.layout, every_render_stage());
        let variant = Arc::new(RenderVariant {
            key,
            pipeline,
            objects: Arc::new(objects),
            plans: Mutex::new(Lru::counted(PLANS)),
            pipelines: Making::counted(PIPELINES),
            mesh_pipelines: Making::counted(PIPELINES),
            handle_stages: stages,
            draw_dynamic: OnceLock::new(),
            logged_spirv,
        });
        for (place, made) in kept {
            if let Place::Graphics(graphics) = place {
                variant.pipelines.seed(*graphics, made);
            }
        }
        device.shared().shader_cache.memo.live_render(&variant);
        Ok(variant)
    }

    /// What its draws take of its dynamic states: `work`'s answer for its pipeline, worked out at
    /// its first draw.
    pub(crate) fn draw_dynamic(
        &self,
        work: impl FnOnce(&RenderPipeline) -> DrawDynamic,
    ) -> &DrawDynamic {
        self.draw_dynamic.get_or_init(|| work(&self.pipeline))
    }

    /// The compiled pipeline.
    pub(crate) fn pipeline(&self) -> &RenderPipeline {
        &self.pipeline
    }

    /// Each stage's SPIR-V as the pipeline creation log names it: the hashes worked out as the
    /// variant was made, else worked out now (`creation_log::spirv_hashes`).
    pub(crate) fn logged_spirv(&self) -> &[u64] {
        self.logged_spirv
            .get_or_init(|| creation_log::spirv_hashes(&self.pipeline))
    }

    /// Its variant key.
    pub(crate) fn key(&self) -> &VariantKey {
        &self.key
    }

    /// The value of each specialization constant of its pipeline's stages under `key`, a
    /// `VkPipeline`'s specialization, by its meaning, in id order: the value `key` gives it, else
    /// the module's default, each as its low bytes (`SpecConstant::default`).
    pub(crate) fn constants(&self, key: &SpecializationKey) -> Vec<(SpecKey, u64)> {
        let constants = self
            .pipeline
            .stages
            .iter()
            .flat_map(|module| &module.specialization.constants)
            .map(|constant| (constant.id, constant.key, constant.default));
        super::specialization_values(constants, key)
    }

    /// The compiled pipeline, shared (an execution record's).
    pub(crate) fn shared_pipeline(&self) -> &Arc<RenderPipeline> {
        &self.pipeline
    }

    /// The variant's layouts and modules.
    pub(crate) fn objects(&self) -> &Arc<VariantObjects> {
        &self.objects
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
        let mut plan = self
            .pipeline
            .plan(device.compile_device(), shape)
            .map_err(Error::Bind)?;
        // The binder's key names no color group (`BindingPlan::variant`): a plan of a group's
        // variant asks for a variant of that group, as every draw of the group's pass draws it
        // (`render/color_groups.rs`).
        plan.variant.color_group = self
            .pipeline
            .graphics
            .color_group
            .as_ref()
            .map_or(0, |g| g.group);
        // Where the device makes no key that reads a uniform view at its bind offset, every key
        // is its residual form, which reads the same bytes at every bind offset
        // (`render/residual.rs`).
        if !reads_views_at_bind(device) {
            if let Some(residual) = self.residual_form(&plan.specialization) {
                plan.specialization = residual;
            }
        }
        let plan = Arc::new(plan);
        lock(&self.plans).insert(shape.clone(), Arc::clone(&plan), 1);
        Ok(plan)
    }

    /// The `VkPipeline` of `key`, from the memo or made now against `render_pass`, a pass of
    /// `key.pass`; made once, outside the memo's lock (`making.rs`), for a draw of `state` that
    /// binds `formats`, whose part of the pipeline creation log logs it. Where it is not made and
    /// its general form's is, the draw takes that one, and `key`'s is made on the device's pool
    /// (`render/residual.rs`). A lookup by key (`WorkCounts::graphics_pipelines_looked_up`).
    pub(crate) fn graphics_pipeline(
        self: &Arc<Self>,
        key: &GraphicsKey,
        render_pass: &Arc<RenderPassObject>,
        (state, formats): (&GraphicsState, crate::degradation::BoundFormats<'_>),
    ) -> Result<Arc<RawPipeline>, Error> {
        self.objects
            .device
            .shared()
            .counts
            .graphics_pipeline_looked_up();
        self.made_or_make(key, render_pass, (state, formats))
            .map(|(made, _)| made)
    }

    /// The `VkPipeline` of `key` for an execution's settle at commit, which records nothing
    /// (`render/executions.rs`), against `render_pass`: from the memo, the making in flight for
    /// it, or begun now as a draw's is ([`begin_graphics_pipeline`](Self::begin_graphics_pipeline):
    /// linked here where its shaders' libraries are made, else handed to `device`'s pool), so the
    /// commit compiles nothing on the program's thread; the host step's draw of the key waits for
    /// it until its command buffer's bound at most (`render.rs`). Under
    /// `PipelineWaitPolicy::Exact` made now ([`graphics_pipeline`](Self::graphics_pipeline)).
    pub(crate) fn settle_graphics_pipeline(
        self: &Arc<Self>,
        device: &Device,
        key: &GraphicsKey,
        render_pass: &Arc<RenderPassObject>,
        (state, formats): (&GraphicsState, crate::degradation::BoundFormats<'_>),
    ) -> Result<DrawPipeline, Error> {
        if device.compile_device().policy().pipeline_waits == PipelineWaitPolicy::Exact {
            return self
                .graphics_pipeline(key, render_pass, (state, formats))
                .map(DrawPipeline::Made);
        }
        device.shared().counts.graphics_pipeline_looked_up();
        Ok(match self.pipelines.begin_at(key) {
            (Begun::Made(made), _) => DrawPipeline::Made(made),
            (Begun::Pending(pending), _) => DrawPipeline::Making(pending),
            (Begun::Yours(ticket), _) => {
                self.make_drawn(device, key, render_pass, (state, formats), ticket)
            }
        })
    }

    /// [`graphics_pipeline`](Self::graphics_pipeline)'s finding or making, uncounted, with
    /// whether it is `key`'s own `VkPipeline`, not its general form's.
    fn made_or_make(
        self: &Arc<Self>,
        key: &GraphicsKey,
        render_pass: &Arc<RenderPassObject>,
        (state, formats): (&GraphicsState, crate::degradation::BoundFormats<'_>),
    ) -> Result<(Arc<RawPipeline>, bool), Error> {
        if let Some(drawn) = self.general_drawn(key, render_pass, (state, formats)) {
            return Ok(drawn);
        }
        let made = self.pipelines.get_or_make(key, || {
            let make = || {
                self.objects.graphics_pipeline(
                    &self.pipeline,
                    key,
                    (&**render_pass, Notices::Now(Some(formats))),
                    None,
                )
            };
            match &state.0.log {
                None => make(),
                Some(log) => log.made(
                    &state.0.device,
                    (&self.key, &self.pipeline, key, self.logged_spirv()),
                    make,
                ),
            }
        })?;
        Ok((made, true))
    }

    /// The libraries of `key`'s parts (`libraries.rs`), claimed on this thread: the device's
    /// where this is a program's call. `None` where the device makes its pipelines whole, where
    /// `key`'s pipeline is made whole, or where its state cannot be described (its making then
    /// says why).
    pub(crate) fn claim_libraries(&self, key: &GraphicsKey) -> Option<Linked> {
        let shared = self.objects.device.shared();
        let libraries = shared.libraries.as_ref()?;
        let content = self.objects.graphics_content(&self.pipeline, key).ok()?;
        content
            .links()
            .then(|| libraries.claim(shared, &content.names(&self.objects, &self.pipeline, key)))
    }

    /// The `VkPipeline`s of `keys`, each against its pass (a pass of the key's `pass`), begun at
    /// the creation of `state`, before any draw. What the program or host sees is decided on this
    /// thread, in `keys`' order: each key's memo entry begun before (a key made already, or being
    /// made by another caller, is not this creation's), and its creation log line numbered
    /// before. Under `PipelineWaitPolicy::Bounded` (the default) each making is handed to the
    /// device's pool, kept by the state's piece holds as a draw's is ([`pooled`](Self::pooled)),
    /// and the creation waits for them the policy's `pipeline_creation_wait` (4.5 s by default) at
    /// most, one deadline for all of them, as Metal compiles in its creation: a driver's cold
    /// compile (seconds on MoltenVK with a cold Metal compiler service) holds the thread creating
    /// the state, which a host may create inside a command buffer's encoding, no longer; a draw
    /// that finds the making in flight takes it (`render/pending.rs`), its command buffer waiting a
    /// bound for it in turn (`WorkCounts::creations_left_making`). Under
    /// `PipelineWaitPolicy::Exact` they are made before the creation returns: the first on this
    /// thread, the others on the device's pool, joined (`Pool::join_all`, the owner's HITCH rules 1
    /// and 3). Each piece settles its own key's memo entry as its pipeline is made, a failed one
    /// not kept, and the join returns once every piece settled. The concurrent part only calls the
    /// driver and settles its entries. The makings of `ahead`'s keys (a predicted key whose general
    /// form is among `keys`, `render/residual.rs`) are handed to the pool right after, under either
    /// policy, at the rank nothing waits for, and not waited for. Their specialized notices wait
    /// for their first draws ([`first_use`](Self::first_use)). Each key is its variant's.
    pub(crate) fn make_at_creation(
        keys: Vec<(Arc<RenderVariant>, GraphicsKey, RenderPassObject)>,
        ahead: Vec<(Arc<RenderVariant>, GraphicsKey, RenderPassObject)>,
        state: &GraphicsState,
    ) {
        let device = &state.0.device;
        let joined = device.compile_device().policy().pipeline_waits == PipelineWaitPolicy::Exact;
        let mut handed = Vec::new();
        let mut pending = Vec::with_capacity(keys.len());
        let mut pieces: Vec<Joined<()>> = Vec::with_capacity(keys.len());
        for (variant, key, pass) in keys {
            let Begun::Yours(ticket) = variant.pipelines.begin(&key) else {
                continue;
            };
            let line = state.0.log.as_ref().map(|log| {
                log.begin((
                    &variant.key,
                    &variant.pipeline,
                    &key,
                    variant.logged_spirv(),
                ))
                .at_creation()
            });
            // Its libraries claimed here, on the creating thread, where the device links them:
            // the device's, found by every later key and state that shares a part.
            let claimed = variant.claim_libraries(&key);
            if !joined {
                handed.push(variant.pooled(
                    state,
                    (key, Arc::new(pass), ticket),
                    (line, claimed),
                    true,
                ));
                continue;
            }
            pending.push(ticket.pending());
            // The piece owns the variant (its modules, layouts and immutable samplers, and the
            // device with its pipeline cache), the key (the specialization's bytes), the key's
            // pass, which the making borrows, and its making's ticket: each lives until the
            // making returned, wherever the piece runs and whether or not it runs.
            pieces.push(Box::new(move || {
                let make = || {
                    variant.objects.graphics_pipeline(
                        &variant.pipeline,
                        &key,
                        (&pass, Notices::FirstDraw),
                        claimed,
                    )
                };
                let made = match line {
                    None => make(),
                    Some(line) => line.made(&variant.objects.device, make),
                };
                // Settled as it is made, so a waiter for this key waits for it alone, not for
                // the creation's whole set. A failed making is not kept: the draw of the key
                // meets it as it would have. Dropped unrun (the process exiting), the ticket
                // fails the making, as a pool's does.
                if made.is_ok() {
                    variant
                        .objects
                        .device
                        .shared()
                        .counts
                        .creation_pipeline_made(crate::work_counts::Predicted::Render);
                }
                drop(ticket.settle(made));
            }));
        }
        // Those made ahead of the draws that take them, right after: handed to the pool under
        // either wait, at the rank nothing waits for until a draw does, and not waited for here.
        for (variant, key, pass) in ahead {
            let Begun::Yours(ticket) = variant.pipelines.begin(&key) else {
                continue;
            };
            let line = state.0.log.as_ref().map(|log| {
                log.begin((
                    &variant.key,
                    &variant.pipeline,
                    &key,
                    variant.logged_spirv(),
                ))
                .ahead_at_creation()
            });
            let claimed = variant.claim_libraries(&key);
            drop(variant.pooled(state, (key, Arc::new(pass), ticket), (line, claimed), true));
        }
        // Those handed to the pool, waited for a bound at most: a making still running then goes
        // on for the draws that take it.
        let deadline = Instant::now() + device.compile_device().policy().pipeline_creation_wait;
        for making in &handed {
            if making.wait_until(deadline).is_none() {
                device.shared().counts.creation_left_making();
            }
        }
        // The full join: every key begun here is settled before the creation returns.
        if pieces.is_empty() {
            return;
        }
        drop(device.shared().pipeline_pool.join_all(
            pieces,
            true,
            // Each key handed to the pool waits there: a draw that finds it in flight, its
            // commit, or a host step hastens it through its place.
            |index, place| {
                if let Some(pending) = pending.get(index) {
                    pending.queued_at(place.clone());
                }
            },
        ));
    }

    /// A draw that binds `formats` takes `made`, its `VkPipeline` of `key`: where `made` was made
    /// at its state's creation and this is its first draw, the notices its making would have
    /// announced are announced now, naming what the draw binds, through the device's
    /// once-per-kind path.
    pub(crate) fn first_use(
        &self,
        made: &RawPipeline,
        specialization: &SpecializationKey,
        formats: BoundFormats<'_>,
    ) {
        if made.first_drawn() {
            self.objects
                .device
                .shared()
                .counts
                .creation_pipeline_drawn(crate::work_counts::Predicted::Render);
            self.objects.device.degrade_specialized(
                &self.objects.specialized,
                specialization,
                Some(formats),
                Occurrence::Made,
            );
        }
    }

    /// The `VkPipeline` of `key` for a draw that records it later: from the memo (from the entry
    /// the state's last draw found made, where that draw was of this variant and an equal key), the
    /// making in flight for it, or made on `device`'s pool (`pool.rs`) against `compatible`, a pass
    /// of `key.pass`, held until it is made, for a draw of `state` that binds `formats` (taken for
    /// the pool where a specialized note tests them, `VariantObjects::tested_formats`), whose part
    /// of the pipeline creation log logs it as [`graphics_pipeline`](Self::graphics_pipeline)
    /// does. The draw goes on meanwhile; its command buffer's batch is submitted once it is made
    /// (`render/pending.rs`). Where it is not made and its general form's is, the draw takes that
    /// one, and nothing waits for `key`'s making (`render/residual.rs`). With whether it is
    /// `key`'s own `VkPipeline`, made: not a general form, nor being made, which a render encoder
    /// keeps no draw's shape with (`render/draw_shape.rs`).
    pub(crate) fn begin_graphics_pipeline(
        self: &Arc<Self>,
        device: &Device,
        key: &DrawKey<'_>,
        compatible: &Arc<RenderPassObject>,
        (state, formats): (&GraphicsState, crate::degradation::BoundFormats<'_>),
    ) -> (DrawPipeline, bool) {
        if let Some(made) = self.last_made(state, key) {
            return (DrawPipeline::Made(made), true);
        }
        device.shared().counts.graphics_pipeline_looked_up();
        let key = &key.key();
        match self.pipelines.begin_at(key) {
            (Begun::Made(made), at) => {
                if let Some(at) = at {
                    self.keep_last(state, key, at);
                }
                (DrawPipeline::Made(made), true)
            }
            // A draw of a key another caller makes takes the key's general form where it is made
            // (`render/residual.rs`).
            (Begun::Pending(pending), _) => match self.general_made(key) {
                Some(drawn) => {
                    device.shared().counts.residual_form_drawn();
                    (DrawPipeline::Made(drawn), false)
                }
                None => (DrawPipeline::Making(pending), false),
            },
            (Begun::Yours(ticket), _) => {
                match self.make_drawn(device, key, compatible, (state, formats), ticket) {
                    // Handed to the pool: the draw takes the key's general form where it is made,
                    // and nothing waits for the making (`render/residual.rs`).
                    DrawPipeline::Making(pending) => match self.general_made(key) {
                        Some(drawn) => {
                            device.shared().counts.residual_form_making();
                            device.shared().counts.residual_form_drawn();
                            (DrawPipeline::Made(drawn), false)
                        }
                        None => (DrawPipeline::Making(pending), false),
                    },
                    made => (made, true),
                }
            }
        }
    }

    /// The `VkPipeline` of a draw of this variant and `key` that records now
    /// ([`graphics_pipeline`](Self::graphics_pipeline)'s: a draw of an execution's replay, or of
    /// a pass of custom sample positions), finding first, as
    /// [`begin_graphics_pipeline`](Self::begin_graphics_pipeline) does, an entry `state`'s last
    /// draws found made ([`last_made`](Self::last_made)), and keeping where one it looks up lies.
    /// A key found neither way is made now, or, where its general form's is made, drawn with that
    /// one while its own is made on the pool ([`general_drawn`](Self::general_drawn)). With
    /// whether it is `key`'s own `VkPipeline`, not its general form's, as
    /// [`begin_graphics_pipeline`](Self::begin_graphics_pipeline)'s.
    pub(crate) fn graphics_pipeline_now(
        self: &Arc<Self>,
        device: &Device,
        key: &DrawKey<'_>,
        render_pass: &Arc<RenderPassObject>,
        (state, formats): (&GraphicsState, crate::degradation::BoundFormats<'_>),
    ) -> Result<(Arc<RawPipeline>, bool), Error> {
        if let Some(made) = self.last_made(state, key) {
            return Ok((made, true));
        }
        device.shared().counts.graphics_pipeline_looked_up();
        let key = &key.key();
        if let Some((made, at)) = self.pipelines.made_with_at(key) {
            self.keep_last(state, key, at);
            return Ok((made, true));
        }
        self.made_or_make(key, render_pass, (state, formats))
    }

    /// The `VkPipeline` of a draw of this variant and `key` where `state`'s last draws found it
    /// made, and its entry stays: the draw uses it as a lookup of the key would, the key compared,
    /// not hashed, nor made. An entry that went is forgotten here.
    fn last_made(
        self: &Arc<Self>,
        state: &GraphicsState,
        key: &DrawKey<'_>,
    ) -> Option<Arc<RawPipeline>> {
        let ours = |last: &LastPipeline| std::ptr::eq(last.variant.as_ptr(), Arc::as_ptr(self));
        let at = lock(&state.0.last_pipelines)
            .iter()
            .find(|last| ours(last) && key.is(&last.key))
            .map(|last| last.at)?;
        let made = self.pipelines.made_at(at);
        if made.is_none() {
            lock(&state.0.last_pipelines).retain(|last| !(ours(last) && last.at == at));
        }
        made
    }

    /// Keep that a draw of `state` found `key`'s `VkPipeline` made at `at` of this variant's memo,
    /// the latest first, at most [`LAST_PIPELINES`].
    fn keep_last(self: &Arc<Self>, state: &GraphicsState, key: &GraphicsKey, at: At) {
        let mut kept = lock(&state.0.last_pipelines);
        if kept.len() == LAST_PIPELINES {
            kept.pop_back();
        }
        kept.push_front(LastPipeline {
            variant: Arc::downgrade(self),
            key: key.clone(),
            at,
        });
    }

    /// The making of `key`, this caller's (`ticket`), for a draw of `state` that binds `formats`
    /// ([`begin_graphics_pipeline`](Self::begin_graphics_pipeline)): linked here, against
    /// `compatible`, where its shaders' libraries are made; else handed to `device`'s pool, the
    /// draw going on meanwhile.
    fn make_drawn(
        self: &Arc<Self>,
        device: &Device,
        key: &GraphicsKey,
        compatible: &Arc<RenderPassObject>,
        (state, formats): (&GraphicsState, crate::degradation::BoundFormats<'_>),
        ticket: Ticket<GraphicsKey, RawPipeline>,
    ) -> DrawPipeline {
        // Its libraries claimed on the draw's thread, the device's where it links them.
        // Where its shaders' libraries are made it is linked here, in milliseconds, and
        // the draw takes it: no making in the pool, and nothing waits.
        let claimed = self.claim_libraries(key);
        if let Some(linked) = claimed.as_ref().filter(|linked| linked.shaders_made()) {
            let line = state
                .0
                .log
                .as_ref()
                .map(|log| log.begin((&self.key, &self.pipeline, key, self.logged_spirv())));
            let make = || {
                self.objects.graphics_pipeline(
                    &self.pipeline,
                    key,
                    (&**compatible, Notices::Now(Some(formats))),
                    Some(linked.clone()),
                )
            };
            let made = match line {
                None => make(),
                Some(line) => line.made(&self.objects.device, make),
            };
            let pending = ticket.pending();
            return match ticket.settle(made) {
                Ok(made) => DrawPipeline::Made(made),
                // The draw's batch meets the failure as it meets a pool's.
                Err(_) => DrawPipeline::Making(pending),
            };
        }
        device.shared().counts.pipeline_pooled();
        // Its specialized notices now, on the draw's thread in the guest's order: the
        // pool decides nothing the guest or host sees.
        self.objects.device.degrade_specialized(
            &self.objects.specialized,
            &key.specialization,
            Some(formats),
            Occurrence::Made,
        );
        // Its creation log line numbered here, on the draw's thread in the guest's order.
        let line = state
            .0
            .log
            .as_ref()
            .map(|log| log.begin((&self.key, &self.pipeline, key, self.logged_spirv())));
        let handed = (key.clone(), Arc::clone(compatible), ticket);
        DrawPipeline::Making(self.pooled(state, handed, (line, claimed), false))
    }

    /// Hand the making of `key`, `ticket`'s, to the device's pool, against `pass`, a pass of
    /// `key.pass`, kept by `state`'s piece holds (the state's release stops a piece not started,
    /// failing the making), logged on `line`, linked from `claimed` where the device links
    /// libraries: a draw's making, whose specialized notices the draw announced; or, where
    /// `at_creation`, a state creation's predicted key's, whose notices wait for its first draw
    /// ([`first_use`](Self::first_use)), counted made. The making, to look at later, queued at
    /// its place there.
    fn pooled(
        self: &Arc<Self>,
        state: &GraphicsState,
        (key, pass, ticket): (
            GraphicsKey,
            Arc<RenderPassObject>,
            Ticket<GraphicsKey, RawPipeline>,
        ),
        (line, claimed): (Option<creation_log::Line>, Option<Linked>),
        at_creation: bool,
    ) -> Pending<RawPipeline> {
        let pending = ticket.pending();
        let keeping =
            (state.0.pieces).keep(PieceHeld::Draw(Arc::clone(self), pass, Box::new(ticket)));
        let place = (self.objects.device.shared().pipeline_pool).run(Box::new(move || {
            // Where the state let go meanwhile, the piece stops: its ticket went with the
            // state, failing the making.
            let Some((lease, held)) = keeping.start() else {
                return;
            };
            let (variant, compatible, ticket) = match held {
                PieceHeld::Draw(variant, compatible, ticket) => (variant, compatible, *ticket),
                other => return lease.give_back(other, None),
            };
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            super::pool::held::mid_making();
            let notices = match at_creation {
                true => Notices::FirstDraw,
                false => Notices::Announced,
            };
            let make = || {
                variant.objects.graphics_pipeline(
                    &variant.pipeline,
                    &key,
                    (&*compatible, notices),
                    claimed,
                )
            };
            let made = match line {
                None => make(),
                Some(line) => line.made(&variant.objects.device, make),
            };
            if at_creation && made.is_ok() {
                let counts = &variant.objects.device.shared().counts;
                counts.creation_pipeline_made(crate::work_counts::Predicted::Render);
            }
            // Its callers take it from the making: from the settle, the batches that wait
            // for it may run and complete while this piece still holds the variant and the
            // pass, and the device with them, until it gave them back. A trail spans that
            // (`trails.rs`).
            let trail = variant.objects.device.shared().trails.open();
            drop(ticket.settle(made));
            lease.give_back(
                PieceHeld::Drawn {
                    _variant: variant,
                    _pass: compatible,
                },
                Some(trail),
            );
        }));
        pending.queued_at(place);
        pending
    }

    /// The compute `VkPipeline`s of the variant's emulated object and mesh functions under a
    /// mesh draw's `specialization` (its `COMPUTE` stages, the object function's first where it
    /// has one), from the memo or made now, outside the memo's lock (`making.rs`), for a draw
    /// that binds `formats`.
    pub(crate) fn mesh_pipelines(
        &self,
        specialization: &SpecializationKey,
        formats: crate::degradation::BoundFormats<'_>,
    ) -> Result<MeshPipelines, Error> {
        let made = self.mesh_pipelines.get_or_make(specialization, || {
            self.make_mesh_pipelines(specialization, formats)
        })?;
        Ok(MeshPipelines::clone(&made))
    }

    /// The compute `VkPipeline`s of [`mesh_pipelines`](Self::mesh_pipelines), made now.
    fn make_mesh_pipelines(
        &self,
        specialization: &SpecializationKey,
        formats: crate::degradation::BoundFormats<'_>,
    ) -> Result<MeshPipelines, Error> {
        let compute: Vec<(usize, &metal2vulkan::pipeline::StageModule)> = self
            .pipeline
            .stages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.stage == metal2vulkan::vk::ShaderStage::COMPUTE)
            .collect();
        let make = |&(index, module): &(usize, &metal2vulkan::pipeline::StageModule)| {
            self.objects
                .emulated_pipeline(index, module, (specialization, formats))
                .map(Arc::new)
        };
        let made = match compute.as_slice() {
            [mesh] => MeshPipelines {
                object: None,
                mesh: make(mesh)?,
            },
            [object, mesh] => MeshPipelines {
                object: Some(make(object)?),
                mesh: make(mesh)?,
            },
            _ => {
                return Err(internal(
                    "an emulated mesh pipeline of other compute stages",
                ))
            }
        };
        Ok(made)
    }
}
