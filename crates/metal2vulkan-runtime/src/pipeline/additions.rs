//! The pipeline fields: `requiredThreadsPerThreadgroup`, binary functions (a stage's
//! `binaryFunctions` and the binary functions added to a made state), and what the runtime checks
//! of them.
//!
//! Measured on an M3 Max (macOS 26.7) and an M4 (macOS 27.0.1), which agree:
//! - **Required threads per threadgroup.** Creation stops, with or without API validation, a
//!   size with an axis of 0, one of more threads than the device's limit, and one of more threads
//!   than the descriptor's `maxTotalThreadsPerThreadgroup`; it fails (an error, not a stop) where
//!   that maximum is above the size's threads. The state's `maxTotalThreadsPerThreadgroup` stays
//!   the kernel's own. API validation stops a direct, threads or indirect dispatch, and a tile
//!   dispatch, whose threadgroup differs on any axis; without it Metal runs the size given, and
//!   so does the runtime. An indirect command buffer's dispatch is not checked, and runs the size
//!   it was encoded with.
//! - **Binary functions.** A function in a stage's `binaryFunctions`, or among the functions added
//!   to a made state, must be compiled to a binary (`MTLFunctionOptionCompileToBinary`): API
//!   validation stops another, and creation fails without validation. Adding to a stage of a
//!   state made without its support flag (`supportAddingBinaryFunctions`, or the vertex and
//!   fragment forms) is an API validation stop; without validation Metal adds them. The added
//!   state has the base state's handles and one for each added function; an addition the stage
//!   already links keeps its handle, and adding nothing makes a state alike.

use super::render::{GraphicsState, Kind, RenderPipelineState, StateDescriptor};
use super::tile::TileState;
use super::{ComputePipelineState, Kernel, PipelineOptions};
use crate::error::{invalid, Error, Problem};
use crate::library::Function;
use crate::values::Size;
use crate::Device;
use metal2vulkan::pipeline::{
    FunctionDescriptor, LinkedFunction, LinkedFunctions, MeshPipelineDescriptor,
    RenderPipelineDescriptor, Stage, StageSource,
};

/// A `requiredThreadsPerThreadgroup` Metal stops at a pipeline's creation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequiredThreadsProblem {
    /// A size with an axis of 0, though not every axis (all three 0 is no requirement): "must
    /// not be 0", at creation, with or without API validation.
    ///
    /// Basis: metal-documented (Metal's creation fails "must not be 0", with or without API
    /// validation).
    EmptyAxis {
        /// The size given.
        required: [u32; 3],
    },
    /// A size of more threads than the device's limit (its `maxTotalThreadsPerThreadgroup`
    /// answer): "must be <= 1024 (device limit)", at creation, with or without API validation.
    ///
    /// Basis: metal-documented (Metal's creation fails "must be <= 1024 (device limit)", with or
    /// without API validation).
    OverDeviceLimit {
        /// The size's threads.
        threads: u64,
        /// The device's limit.
        limit: u32,
    },
    /// A size of more threads than the descriptor's `maxTotalThreadsPerThreadgroup`: "must be <=
    /// maxTotalThreadsPerThreadgroup (32)", at creation, with or without API validation.
    ///
    /// Basis: metal-documented (Metal's creation fails "must be <= maxTotalThreadsPerThreadgroup
    /// (32)", with or without API validation).
    OverMaxTotal {
        /// The size's threads.
        threads: u64,
        /// The descriptor's `maxTotalThreadsPerThreadgroup`.
        max: u32,
    },
    /// A descriptor's `maxTotalThreadsPerThreadgroup` above the size's threads: Metal's creation
    /// fails, with or without API validation ("Kernel specified max total threads per
    /// threadgroup (64) must match compute pipeline specified max total threads per threadgroup
    /// (256)").
    ///
    /// Basis: metal-documented (Metal's creation fails "Kernel specified max total threads per
    /// threadgroup (64) must match compute pipeline specified max total threads per threadgroup
    /// (256)", with or without API validation).
    MaxTotalDiffers {
        /// The size's threads.
        threads: u64,
        /// The descriptor's `maxTotalThreadsPerThreadgroup`.
        max: u32,
    },
}

/// The stage of a state that binary functions are added to.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryFunctionStage {
    /// A compute state's kernel.
    Kernel,
    /// A render state's vertex function.
    Vertex,
    /// A render state's fragment function.
    Fragment,
    /// A tile pipeline's tile function.
    Tile,
}

/// Binary functions Metal stops.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryFunctionProblem {
    /// Functions added to any stage of a mesh state, whose descriptor has no support flag:
    /// Metal's API validation stops functions added to a stage without its flag ("Pipeline
    /// state is adding fragment functions but it was created with
    /// supportAddingFragmentBinaryFunctions=NO"), and plain Metal's process ends at the call for
    /// a mesh state, whether the functions are added to its fragment stage or to the vertex or
    /// tile stage it lacks (an M4 under macOS 27.0.1: `mesh_frag_two`,
    /// `mesh_frag_one_add_two`, `mesh_vertex_add`, `mesh_tile_add`). Plain Metal's answer is
    /// the runtime's for a compute state's kernel and a drawing or tile state's own stages made
    /// without their flags, and for a stage a drawing or tile state lacks.
    ///
    /// Basis: metal-measured (an M4, macOS 27.0.1: plain Metal's process ends at the call for a
    /// mesh state — cases `mesh_frag_two`, `mesh_frag_one_add_two`, `mesh_vertex_add` and
    /// `mesh_tile_add`).
    NotSupported {
        /// The stage.
        stage: BinaryFunctionStage,
    },
    /// A function not compiled to a binary in a stage's `binary_functions` ("Function 'f' was
    /// found in binaryFunctions array but is not precompiled.") or among the functions added
    /// ("Function 'f' was found in Fragment functions array but is not precompiled."): Metal's
    /// API validation stops it, and its creation fails without validation.
    ///
    /// Basis: metal-documented (Metal's validation says "Function 'f' was found in binaryFunctions
    /// array but is not precompiled.", and its creation fails without validation).
    NotPrecompiled {
        /// The function's name.
        name: String,
    },
    /// Functions added to a stage the compile layer already compiled (`StageSource::Compiled`),
    /// whose linked functions are fixed: Metal's stages are functions, so it has no such call.
    ///
    /// Basis: ours (this crate's already-compiled stage: Metal's descriptor takes a function, so it
    /// has no such call).
    CompiledStage {
        /// The stage.
        stage: BinaryFunctionStage,
    },
}

/// `MTLRenderPipelineFunctionsDescriptor`: the binary functions
/// [`RenderPipelineState::new_render_pipeline_state_with_additional_binary_functions`] adds to
/// each stage.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct RenderPipelineFunctionsDescriptor {
    /// `vertexAdditionalBinaryFunctions`.
    pub vertex_additional_binary_functions: Vec<Function>,
    /// `fragmentAdditionalBinaryFunctions`.
    pub fragment_additional_binary_functions: Vec<Function>,
    /// `tileAdditionalBinaryFunctions`.
    pub tile_additional_binary_functions: Vec<Function>,
}

impl RenderPipelineFunctionsDescriptor {
    /// A descriptor that adds nothing.
    pub fn new() -> RenderPipelineFunctionsDescriptor {
        RenderPipelineFunctionsDescriptor::default()
    }
}

/// `required`, with Metal's all-zero size as none.
pub(super) fn required_size(required: Option<[u32; 3]>) -> Option<[u32; 3]> {
    required.filter(|axes| *axes != [0; 3])
}

impl Kernel {
    /// The descriptor's `requiredThreadsPerThreadgroup` and `maxTotalThreadsPerThreadgroup`.
    fn threadgroup_fields(&self) -> (Option<[u32; 3]>, Option<u32>) {
        match self {
            Kernel::Compute(d) => (
                d.required_threads_per_threadgroup,
                d.max_total_threads_per_threadgroup,
            ),
            Kernel::Tile(d) => (
                d.required_threads_per_threadgroup,
                d.max_total_threads_per_threadgroup,
            ),
        }
    }
}

/// Check, before compiling, a compute or tile creation of `kernel` (whose function is `function`)
/// in `call` on `device`: its required threads per threadgroup against the device's limit and its
/// own maximum, and that its `binary_functions` are precompiled, as Metal does (the module docs).
pub(super) fn check_kernel(
    device: &Device,
    kernel: &Kernel,
    function: &FunctionDescriptor,
    call: &'static str,
) -> Result<(), Error> {
    let (required, max) = kernel.threadgroup_fields();
    if let Some(required) = required_size(required) {
        let problem = |p| invalid(call, Problem::RequiredThreads(p));
        if required.contains(&0) {
            return Err(problem(RequiredThreadsProblem::EmptyAxis { required }));
        }
        let threads: u64 = required.iter().map(|&axis| u64::from(axis)).product();
        let limit = device
            .compile_device()
            .metal_properties()
            .max_total_threads_per_threadgroup;
        if threads > u64::from(limit) {
            return Err(problem(RequiredThreadsProblem::OverDeviceLimit {
                threads,
                limit,
            }));
        }
        match max.filter(|&max| max > 0) {
            Some(max) if threads > u64::from(max) => {
                return Err(problem(RequiredThreadsProblem::OverMaxTotal {
                    threads,
                    max,
                }));
            }
            Some(max) if u64::from(max) > threads => {
                return Err(problem(RequiredThreadsProblem::MaxTotalDiffers {
                    threads,
                    max,
                }));
            }
            _ => {}
        }
    }
    check_precompiled(call, &function.linked)
}

/// Check, before compiling, that a render creation's stages link only precompiled
/// `binary_functions`, as a compute creation's must.
pub(super) fn check_render(
    descriptor: &RenderPipelineDescriptor,
    call: &'static str,
) -> Result<(), Error> {
    for stage in std::iter::once(&descriptor.vertex).chain(&descriptor.fragment) {
        if let StageSource::Function(function) = stage {
            check_precompiled(call, &function.linked)?;
        }
    }
    Ok(())
}

/// The first of `linked`'s `binary_functions` that is not precompiled refuses `call`.
fn check_precompiled(call: &'static str, linked: &LinkedFunctions) -> Result<(), Error> {
    match linked.binary_functions.iter().find(|f| !f.precompiled) {
        Some(f) => Err(not_precompiled(call, f)),
        None => Ok(()),
    }
}

fn not_precompiled(call: &'static str, f: &LinkedFunction) -> Error {
    invalid(
        call,
        Problem::BinaryFunctions(BinaryFunctionProblem::NotPrecompiled {
            name: f.name().to_owned(),
        }),
    )
}

/// `stage` of a state, with `functions` added to its `binary_functions`, in `call` on `device`;
/// `supported` is the stage's support flag. Each function must be of `device` and precompiled;
/// one the stage's `functions` or `binary_functions` already link (the same name it links under,
/// library function, AIR, constant values and specialization) keeps its entry and handle.
fn added(
    device: &Device,
    call: &'static str,
    stage: (&StageSource, BinaryFunctionStage),
    supported: bool,
    functions: &[Function],
) -> Result<StageSource, Error> {
    let (source, which) = stage;
    if functions.is_empty() {
        return Ok(source.clone());
    }
    if !supported {
        return Err(invalid(
            call,
            Problem::BinaryFunctions(BinaryFunctionProblem::NotSupported { stage: which }),
        ));
    }
    let StageSource::Function(descriptor) = source else {
        return Err(invalid(
            call,
            Problem::BinaryFunctions(BinaryFunctionProblem::CompiledStage { stage: which }),
        ));
    };
    let mut descriptor = descriptor.clone();
    for function in functions {
        if !function.device().same(device) {
            return Err(invalid(call, Problem::ForeignObject));
        }
        let linked = function.linked_function();
        if !linked.precompiled {
            return Err(not_precompiled(call, &linked));
        }
        let same = |f: &LinkedFunction| {
            f.name() == linked.name()
                && f.function.info().name == linked.function.info().name
                && f.function.info().air_hash == linked.function.info().air_hash
                && f.constants == linked.constants
                && f.specialized == linked.specialized
        };
        let held = descriptor.linked.functions.iter().any(same)
            || descriptor.linked.binary_functions.iter().any(same);
        if !held {
            descriptor.linked.binary_functions.push(linked);
        }
    }
    Ok(StageSource::Function(descriptor))
}

/// A stage's support flag, for a stage the state lacks.
const ABSENT: bool = false;

/// How many `binary_functions` `stage`'s function links: none for a compiled stage.
fn binary_count(stage: &StageSource) -> usize {
    match stage {
        StageSource::Function(descriptor) => descriptor.linked.binary_functions.len(),
        _ => 0,
    }
}

/// [`added`] of `stage`, whose support flag is `supported`, linking `functions` as with the flag,
/// and how many functions it newly links without the flag: none where it is set. Plain Metal
/// makes the state without the flag, where its API validation stops it, and its handle of such
/// a function calls the stage's last other callable function, which a table of the state takes it
/// as (`function_table.rs`).
fn added_counting(
    device: &Device,
    call: &'static str,
    stage: (&StageSource, BinaryFunctionStage),
    supported: bool,
    functions: &[Function],
) -> Result<(StageSource, usize), Error> {
    let before = binary_count(stage.0);
    let source = added(device, call, stage, true, functions)?;
    let unsupported = match supported {
        true => 0,
        false => binary_count(&source) - before,
    };
    Ok((source, unsupported))
}

/// `functions` added in `call` to a stage a drawing or tile state lacks, which plain Metal makes
/// the state without, giving no handle of them, where its API validation stops them: refused
/// (`Invalid`) only as a stage's are, for a function of another device or one not compiled to a
/// binary.
fn ignored(device: &Device, call: &'static str, functions: &[Function]) -> Result<(), Error> {
    for function in functions {
        if !function.device().same(device) {
            return Err(invalid(call, Problem::ForeignObject));
        }
        let linked = function.linked_function();
        if !linked.precompiled {
            return Err(not_precompiled(call, &linked));
        }
    }
    Ok(())
}

impl ComputePipelineState {
    /// `newComputePipelineStateWithAdditionalBinaryFunctions:error:`: a new state of this one's
    /// descriptor with `functions` added to its kernel's `binary_functions`, made as
    /// [`Device::new_compute_pipeline_state`] makes one with the default options and this state's
    /// label. It has this state's function handles and one for each added function; adding nothing
    /// makes a state alike. On a state made without `support_adding_binary_functions`, which
    /// Metal's API validation stops, an added function's handle calls the kernel's last other
    /// callable function, and none where it has none (an entry that reads null), as plain Metal's
    /// does. Refused (`Invalid`), as Metal's API validation refuses it: a function not made
    /// compiled to a binary (`FunctionOptions::COMPILE_TO_BINARY`), which Metal's creation also
    /// fails on without validation (`BinaryFunctions`); and a function of another device
    /// (`ForeignObject`). Otherwise refused as the creation is.
    pub fn new_compute_pipeline_state_with_additional_binary_functions(
        &self,
        functions: &[Function],
    ) -> Result<ComputePipelineState, Error> {
        const CALL: &str = "new_compute_pipeline_state_with_additional_binary_functions";
        let device = self.device().clone();
        let Kernel::Compute(descriptor) = &self.0.descriptor else {
            return Err(invalid(
                CALL,
                Problem::BinaryFunctions(BinaryFunctionProblem::NotSupported {
                    stage: BinaryFunctionStage::Kernel,
                }),
            ));
        };
        let mut descriptor = descriptor.clone();
        let unsupported;
        (descriptor.kernel, unsupported) = added_counting(
            &device,
            CALL,
            (&descriptor.kernel, BinaryFunctionStage::Kernel),
            descriptor.support_adding_binary_functions,
            functions,
        )?;
        ComputePipelineState::create_adding(
            &device,
            &descriptor,
            &labelled(self.label()),
            None,
            self.added_unsupported() + unsupported,
        )
    }

    /// `requiredThreadsPerThreadgroup`: the descriptor's, a zero size where it requires none.
    pub fn required_threads_per_threadgroup(&self) -> Size {
        size(self.required_threads())
    }

    /// The threadgroup size every dispatch of the state must have, if the descriptor requires
    /// one.
    pub(crate) fn required_threads(&self) -> Option<[u32; 3]> {
        required_size(self.0.descriptor.threadgroup_fields().0)
    }
}

/// The default options with `label`: a state made with additional binary functions keeps its
/// original's label, as Metal's does.
fn labelled(label: Option<&str>) -> PipelineOptions {
    PipelineOptions {
        label: label.map(str::to_owned),
        ..PipelineOptions::default()
    }
}

/// `required` as Metal's `MTLSize`, zero where none.
fn size(required: Option<[u32; 3]>) -> Size {
    let [width, height, depth] = required.unwrap_or_default().map(u64::from);
    Size::new(width, height, depth)
}

/// `new_render_pipeline_state_with_additional_binary_functions` of a mesh render pipeline's state
/// of `descriptor`, in `call`: its descriptor supports adding binary functions to none of the
/// stages the call names, so any function listed is refused as one added to a stage without its
/// support flag, where plain Metal's process ends; adding none makes a state alike, with the
/// default options and the original's `label`.
fn mesh_with_binary_functions(
    device: &Device,
    call: &'static str,
    (descriptor, label): (&MeshPipelineDescriptor, Option<&str>),
    functions: &RenderPipelineFunctionsDescriptor,
) -> Result<RenderPipelineState, Error> {
    for (list, stage) in [
        (
            &functions.vertex_additional_binary_functions,
            BinaryFunctionStage::Vertex,
        ),
        (
            &functions.fragment_additional_binary_functions,
            BinaryFunctionStage::Fragment,
        ),
        (
            &functions.tile_additional_binary_functions,
            BinaryFunctionStage::Tile,
        ),
    ] {
        added(device, call, (&descriptor.mesh, stage), ABSENT, list)?;
    }
    GraphicsState::create_mesh(device, descriptor, &labelled(label))
        .map(RenderPipelineState::of_graphics)
}

impl RenderPipelineState {
    /// `newRenderPipelineStateWithAdditionalBinaryFunctions:error:`: a new state of this one's
    /// descriptor with `functions`' lists added to its stages' `binary_functions`, made as
    /// [`Device::new_render_pipeline_state`] or [`Device::new_tile_render_pipeline_state`] makes
    /// one with the default options and this state's label. It has this state's function handles
    /// and one for each added function. Functions added to a stage the state lacks (a drawing
    /// state's tile stage or missing fragment stage, a tile state's vertex and fragment stages) are
    /// left out, with no handle, as plain Metal leaves them where its API validation stops them; on
    /// a stage made without its support flag (`support_adding_vertex_binary_functions`,
    /// `support_adding_fragment_binary_functions`, a tile descriptor's
    /// `support_adding_binary_functions`), which its API validation also stops, an added function's
    /// handle calls the stage's last other callable function, and none where it has none (an entry
    /// that reads null), as plain Metal's does. Refused (`Invalid`), as Metal's API validation
    /// refuses it: functions added to a mesh state (`BinaryFunctions(NotSupported)`; plain Metal's
    /// process ends there), and a function not made compiled to a binary
    /// (`BinaryFunctions(NotPrecompiled)`); a function of another device (`ForeignObject`); and
    /// functions added to a stage the compile layer already compiled
    /// (`BinaryFunctions(CompiledStage)`). Otherwise refused as the creation is.
    pub fn new_render_pipeline_state_with_additional_binary_functions(
        &self,
        functions: &RenderPipelineFunctionsDescriptor,
    ) -> Result<RenderPipelineState, Error> {
        const CALL: &str = "new_render_pipeline_state_with_additional_binary_functions";
        let device = self.device().clone();
        match self.kind() {
            Kind::Graphics(state) => {
                let descriptor = match state.descriptor() {
                    StateDescriptor::Vertex(descriptor) => descriptor,
                    StateDescriptor::Mesh(descriptor) => {
                        return mesh_with_binary_functions(
                            &device,
                            CALL,
                            (descriptor, self.label()),
                            functions,
                        )
                    }
                };
                ignored(&device, CALL, &functions.tile_additional_binary_functions)?;
                let mut descriptor = descriptor.clone();
                let vertex;
                (descriptor.vertex, vertex) = added_counting(
                    &device,
                    CALL,
                    (&descriptor.vertex, BinaryFunctionStage::Vertex),
                    descriptor.support_adding_vertex_binary_functions,
                    &functions.vertex_additional_binary_functions,
                )?;
                let mut fragment = 0;
                if let Some(source) = &descriptor.fragment {
                    let (added, unsupported) = added_counting(
                        &device,
                        CALL,
                        (source, BinaryFunctionStage::Fragment),
                        descriptor.support_adding_fragment_binary_functions,
                        &functions.fragment_additional_binary_functions,
                    )?;
                    descriptor.fragment = Some(added);
                    fragment = unsupported;
                } else {
                    ignored(
                        &device,
                        CALL,
                        &functions.fragment_additional_binary_functions,
                    )?;
                }
                GraphicsState::create_adding(
                    &device,
                    &descriptor,
                    &labelled(self.label()),
                    None,
                    [
                        state.added_unsupported(Stage::Vertex) + vertex,
                        state.added_unsupported(Stage::Fragment) + fragment,
                    ],
                )
                .map(RenderPipelineState::of_graphics)
            }
            Kind::Tile(state) => {
                let mut descriptor = state.descriptor().clone();
                for list in [
                    &functions.vertex_additional_binary_functions,
                    &functions.fragment_additional_binary_functions,
                ] {
                    ignored(&device, CALL, list)?;
                }
                let unsupported;
                (descriptor.tile, unsupported) = added_counting(
                    &device,
                    CALL,
                    (&descriptor.tile, BinaryFunctionStage::Tile),
                    descriptor.support_adding_binary_functions,
                    &functions.tile_additional_binary_functions,
                )?;
                TileState::create_adding(
                    &device,
                    &descriptor,
                    &labelled(self.label()),
                    None,
                    state.kernel().added_unsupported() + unsupported,
                )
                .map(RenderPipelineState::of_tile)
            }
        }
    }

    /// `requiredThreadsPerTileThreadgroup`: a tile pipeline's descriptor's
    /// `required_threads_per_threadgroup`, a zero size where it requires none and for a drawing
    /// state.
    pub fn required_threads_per_tile_threadgroup(&self) -> Size {
        match self.kind() {
            Kind::Graphics(_) => size(None),
            Kind::Tile(state) => size(state.kernel().required_threads()),
        }
    }
}
