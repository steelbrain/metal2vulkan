use crate::c3::reference::{admit_recording, MetalIdentity, References};
use crate::case::AuthoredCase;
use crate::check::admission::{
    admit, AdmittedCase, ExecutionRefused, ExecutionRequest, ExecutorError,
};
use crate::check::loop_free::Classification;
use crate::check::{CheckedCase, CheckedShader, FuelScope, LoopClass};
use crate::hash::sha256_bytes;
use crate::literal::LiteralResources;
use crate::observation::{CaseTiming, MetalObservation, MetalStatus, ReferenceGap};
use crate::oracle_fuel::{OracleBound, DEFAULT_INVOCATION_FUEL};
use crate::source::SourceRow;
use crate::store::CorpusStore;
use base64::Engine as _;
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// The Metal oracle's ABI: how it compiles, binds, runs and reads back a case on Metal. Every Metal
/// observation records it, and a row recorded under another value is not current evidence for its
/// case.
///
/// Bump it whenever what the oracle compiles, binds, submits, reads back, or records changes. A
/// change no recorded row can observe (a case kind no earlier oracle ran, or one it refused) needs
/// no bump; its commit says why. The commit that bumps it says what changed; this comment keeps no
/// history.
pub const ORACLE_ABI: &str = "metal-literal-resources-v49";

/// The case's own module: the AIR the oracle compiles for it and how it bounds its execution.
#[derive(Clone)]
#[cfg_attr(
    not(target_os = "macos"),
    expect(
        dead_code,
        reason = "only the macOS Metal platform reads the source; elsewhere the stub refuses"
    )
)]
pub(crate) struct OracleSource<'a> {
    /// The checked row (its entry and label).
    pub(crate) row: &'a SourceRow,
    /// The text facts about the row's entry are read from ([`CheckedShader::entry_ll`]); the oracle
    /// compiles `air`.
    pub(crate) entry_ll: &'a str,
    /// The AIR bitcode the oracle compiles for the row ([`ShaderAir`]).
    pub(crate) air: Arc<[u8]>,
    pub(crate) bound: OracleBound,
    /// The generated stage the case draws with: the checker's, which the engine candidate draws
    /// with too.
    pub(crate) companion: Option<&'a crate::companion::Companion>,
}

/// A pipeline template and the AIR the oracle compiles for it. Templates run as compiled, so only a
/// template whose compiled module is loop-free gets here.
#[cfg_attr(
    not(target_os = "macos"),
    expect(
        dead_code,
        reason = "only the macOS Metal platform reads templates; elsewhere the stub refuses"
    )
)]
pub(crate) struct OracleTemplate<'a> {
    pub(crate) shader: &'a CheckedShader<Arc<SourceRow>>,
    pub(crate) air: Vec<u8>,
}

/// The AIR bitcode the oracle compiles for one row.
pub(crate) struct ShaderAir {
    pub(crate) bitcode: Vec<u8>,
    /// Whether `bitcode` is the row's retained blob, the very module the checker classified,
    /// rather than a public fixture compiled on this host.
    pub(crate) checked: bool,
}

/// What the oracle compiles for one checked case.
struct OracleModules<'a> {
    primary: OracleSource<'a>,
    templates: BTreeMap<String, OracleTemplate<'a>>,
}
pub const QUALIFICATION_RUNS: usize = 3;

/// `corpus-metal`'s exit status for a qualification that recorded a reference gap: the device
/// cannot run the case, so the row it wrote is no evidence a candidate compares against. The Metal
/// batch reads it with the row.
pub const REFERENCE_GAP_EXIT: i32 = 4;

/// Why one Metal run returned no output.
#[derive(Debug)]
pub(crate) enum MetalRunError {
    /// Metal's pipeline creation named an Apple GPU family the device does not support: the
    /// reference device cannot run the case.
    #[cfg_attr(
        not(target_os = "macos"),
        expect(
            dead_code,
            reason = "only the macOS Metal platform creates pipelines; elsewhere the stub refuses"
        )
    )]
    ReferenceGap(ReferenceGap),
    Failed(String),
}

impl From<String> for MetalRunError {
    fn from(error: String) -> Self {
        Self::Failed(error)
    }
}

impl From<&str> for MetalRunError {
    fn from(error: &str) -> Self {
        Self::Failed(error.into())
    }
}

/// What one Metal run answered: the selected output bytes, or that the device cannot run the
/// case.
type RunAnswer = Result<Vec<u8>, ReferenceGap>;

/// What a case's [`QUALIFICATION_RUNS`] executions answered, and what each of them cost in time.
type Runs = (Vec<RunAnswer>, Vec<std::time::Duration>);

/// Admit `case` for the Metal oracle, then qualify it. A refusal (a case that is not mechanically
/// loop-free) returns before any device opens.
pub fn qualify(
    root: &Path,
    case: AuthoredCase,
    environment_id: &str,
) -> Result<MetalObservation, ExecutorError> {
    let admitted = admit(root, case, ExecutionRequest::MetalOracle)?;
    qualify_case(root, admitted, environment_id)
}

/// Execute one admitted case on the Metal device [`QUALIFICATION_RUNS`] times and record the
/// qualified observation. Metal has no CPU device, so this always runs on the GPU. A module the
/// oracle compiles that fuel does not bound is refused before any device opens.
pub fn qualify_case(
    root: &Path,
    admitted: AdmittedCase,
    environment_id: &str,
) -> Result<MetalObservation, ExecutorError> {
    if environment_id.trim().is_empty() {
        return Err("environment_id must not be empty".into());
    }
    // A reference environment's rows come from hosts of its declared identity.
    let environment = platform::environment()?;
    let references = References::load(&References::committed_path())?;
    admit_recording(&references, environment_id, &platform::identity()?)?;
    let checked = admitted.into_checked_for(ExecutionRequest::MetalOracle)?;
    crate::executor_contract::require_case(&checked.primary.case, "Metal executor")?;
    let modules = oracle_modules(&checked)?;
    let resources = LiteralResources::prepare(&checked.primary.case)?;
    let initial = resources.selected_initial(&checked.primary.case.output)?;
    // Each execution loads the same libraries: each is built once, for the first.
    let (answers, took) = platform::reusing_tool_results(|| -> Result<Runs, String> {
        let mut answers = Vec::with_capacity(QUALIFICATION_RUNS);
        // What each execution cost, the gap-answering ones too: a reference gap is a row, and a
        // row records what its case cost.
        let mut took = Vec::with_capacity(QUALIFICATION_RUNS);
        for run in 0..QUALIFICATION_RUNS {
            let clock = crate::budgets::CreationClock::start();
            let executed = platform::execute(
                &checked.primary.case,
                modules.primary.clone(),
                &resources,
                &checked.primary.reflection,
                &checked.primary.interface,
                &checked.primary.linked_functions,
                &modules.templates,
            );
            took.push(clock.read());
            let output = match executed {
                Ok(output) => output,
                Err(MetalRunError::ReferenceGap(gap)) => {
                    answers.push(Err(gap));
                    continue;
                }
                Err(MetalRunError::Failed(error)) => return Err(error),
            };
            // What Metal returned in each declared out-of-bounds read, reported (never recorded)
            // so a run measures Metal's result there.
            for line in crate::executor_contract::out_of_bounds::undefined_reads(
                &checked.primary.case,
                &output,
            ) {
                eprintln!(
                    "metal-out-of-bounds-read case={} run={run} {line}",
                    checked.primary.case.case_id
                );
            }
            // Metal leaves an out-of-bounds read undefined: the oracle records zeros there.
            answers.push(Ok(crate::executor_contract::out_of_bounds::metal_recorded(
                &checked.primary.case,
                output,
            )));
        }
        Ok((answers, took))
    })?;
    let (output, status) = qualify_answers(&initial, answers)?;
    if matches!(status, MetalStatus::Qualified) {
        // An answer that leaves the selected output as authored is evidence only when the case
        // declares that path.
        crate::no_effect::judge(&checked.primary.case, &initial, &output)?;
    }
    let observation = MetalObservation {
        case_id: checked.primary.case.case_id.clone(),
        air_sha256: checked.primary.case.air_sha256.clone(),
        input_sha256: checked.primary.input_sha256,
        metal_output_sha256: sha256_bytes(&output),
        output_b64: base64::engine::general_purpose::STANDARD.encode(output),
        environment_id: environment_id.into(),
        environment,
        oracle_abi: ORACLE_ABI.into(),
        timing: CaseTiming::of(took),
        status,
    };
    CorpusStore::new(root).upsert_metal(observation.clone())?;
    Ok(observation)
}

/// This host's Metal identity: what every row it qualifies records, and the environment ID it
/// produces.
pub fn identity() -> Result<MetalIdentity, String> {
    platform::identity()
}

/// What the oracle would hand to Metal for one case, built without opening a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepared {
    pub bound: OracleBound,
    /// What the fuel transform instrumented, under [`OracleBound::Fuel`].
    pub fuel: Option<crate::oracle_fuel::FuelReport>,
    /// The native buffer slot the fuel status buffer binds at, under [`OracleBound::Fuel`].
    pub status_slot: Option<u32>,
    /// The size of the case's own packaged metallib.
    pub metallib_bytes: u64,
    /// Every metallib built: the case's own, each pipeline template's, and each distinct linked
    /// module's.
    pub libraries: usize,
    /// One SHA-256 over the SHA-256 of each of those metallibs, in build order.
    pub library_sha256: String,
    /// One SHA-256 over the reflection facts the oracle binds the case by: for the entry and each
    /// pipeline template, its implicit imageblock attachments, its fragment imageblock, and the
    /// declared size of each authored buffer's binding.
    pub binding_sha256: String,
}

/// Admit `case` for the Metal oracle and do the CPU-side part of one execution: choose the
/// bound of every module the oracle compiles, instrument the case's own AIR under fuel, and verify
/// and package it, each pipeline template, and each linked module with `metal-opt` and `metallib`.
/// Stops before any device opens and records nothing; it screens a batch of cases for tool-side
/// failures before any of them reaches the GPU.
pub fn prepare(root: &Path, case: AuthoredCase) -> Result<Prepared, ExecutorError> {
    let admitted = admit(root, case, ExecutionRequest::MetalOracle)?;
    let checked = admitted.into_checked_for(ExecutionRequest::MetalOracle)?;
    crate::executor_contract::require_case(&checked.primary.case, "Metal executor")?;
    let modules = oracle_modules(&checked)?;
    let library = platform::prepare(
        &checked.primary.case,
        modules.primary,
        &checked.primary.interface,
        &checked.primary.linked_functions,
        &modules.templates,
    )?;
    Ok(Prepared {
        bound: library.bound,
        fuel: library.fuel,
        status_slot: library.status_slot,
        metallib_bytes: library.metallib_bytes,
        libraries: library.library_sha256.len(),
        library_sha256: sha256_bytes(library.library_sha256.join("\n").as_bytes()),
        binding_sha256: binding_digest(&checked),
    })
}

/// One SHA-256 over the reflection facts the oracle binds `checked` by, for its entry and then
/// each pipeline template in id order: the implicit imageblock attachments (whose presence makes a
/// kernel a tile function), the fragment imageblock, and the declared size of each authored
/// buffer's binding (to which the oracle pads the buffer). Two builds that print the same digest
/// bind every case alike.
fn binding_digest<S>(checked: &CheckedCase<S>) -> String {
    let shaders = std::iter::once(("", &checked.primary.case, &checked.primary.reflection)).chain(
        checked
            .pipeline_templates
            .iter()
            .map(|(id, template)| (id.as_str(), &template.case, &template.reflection)),
    );
    let mut text = String::new();
    for (id, case, reflection) in shaders {
        let attachments = reflection
            .implicit_imageblock_attachments
            .iter()
            .map(|attachment| {
                format!(
                    "{:?}",
                    (
                        attachment.attachment,
                        attachment.data_rate,
                        attachment.binding,
                        attachment.format,
                        attachment.access,
                    )
                )
            })
            .collect::<Vec<_>>();
        let declared_sizes = case
            .buffers
            .iter()
            .map(|buffer| {
                format!(
                    "{}={:?}",
                    buffer.binding,
                    crate::executor_contract::reflected_buffer_declared_size(
                        reflection,
                        buffer.binding
                    )
                )
            })
            .collect::<Vec<_>>();
        text.push_str(&format!(
            "shader {id:?} attachments [{}] fragment_imageblock {:?} declared_sizes [{}]\n",
            attachments.join(", "),
            reflection.fragment_imageblock,
            declared_sizes.join(", ")
        ));
    }
    sha256_bytes(text.as_bytes())
}

/// What one compile probe measured.
#[derive(Debug)]
pub struct CompileProbe {
    /// How long compute pipeline creation took.
    pub seconds: f64,
    /// What the fuel transform instrumented, for an instrumented probe.
    pub fuel: Option<crate::oracle_fuel::FuelReport>,
}

/// Time compute pipeline creation for `case`'s own module, as compiled (`loops` `None`) or
/// fuel-instrumented with only the selected loops checked: admit and check the case as [`prepare`]
/// does, then open the device and create the pipeline, running nothing. A diagnostic: it records
/// nothing, and the oracle itself always checks every loop.
pub fn probe_compile(
    root: &Path,
    case: AuthoredCase,
    loops: Option<crate::oracle_fuel::LoopSelection>,
) -> Result<CompileProbe, ExecutorError> {
    let admitted = admit(root, case, ExecutionRequest::MetalOracle)?;
    let checked = admitted.into_checked_for(ExecutionRequest::MetalOracle)?;
    crate::executor_contract::require_case(&checked.primary.case, "Metal executor")?;
    let modules = oracle_modules(&checked)?;
    let resources = LiteralResources::prepare(&checked.primary.case)?;
    Ok(platform::probe_compile(
        &checked.primary.case,
        modules.primary,
        &checked.primary.interface,
        &checked.primary.linked_functions,
        &resources.function_constants,
        loops,
    )?)
}

/// The modules the oracle compiles for `checked` and how it bounds them, decided before any device
/// opens: [`bound_modules`] over the AIR this host compiles for each row.
fn oracle_modules(checked: &CheckedCase) -> Result<OracleModules<'_>, ExecutorError> {
    bound_modules(checked, platform::shader_air)
}

/// The modules the oracle compiles for `checked`, with `air_of` yielding the AIR it compiles for
/// a row, and how it bounds them. The bound follows those bytes, classified by the checker's own
/// rules over the same linked modules ([`classify_air`]). The case's own module runs as compiled
/// when loop-free and fuel-instrumented when fuel covers every cycle; a pipeline template runs as
/// compiled, so only a loop-free one runs. Anything else is refused.
fn bound_modules<S: Borrow<SourceRow>>(
    checked: &CheckedCase<S>,
    air_of: impl Fn(&SourceRow) -> Result<ShaderAir, String>,
) -> Result<OracleModules<'_>, ExecutorError> {
    let invocation_fuel = checked.primary.case.fuel.unwrap_or(DEFAULT_INVOCATION_FUEL);
    let (air, classification) = compiled(&checked.primary, &air_of)?;
    let primary = OracleSource {
        row: checked.primary.source.borrow(),
        entry_ll: checked.primary.entry_ll(),
        air: air.into(),
        bound: primary_bound(classification, invocation_fuel)?,
        companion: checked.primary.companion.as_ref(),
    };
    let templates = checked
        .pipeline_templates
        .iter()
        .map(|(id, shader)| {
            let (air, classification) = compiled(shader, &air_of)?;
            template_runs_as_compiled(classification.class)?;
            Ok((id.clone(), OracleTemplate { shader, air }))
        })
        .collect::<Result<_, ExecutorError>>()?;
    Ok(OracleModules { primary, templates })
}

/// The AIR `air_of` yields for `shader`'s row and its classification.
fn compiled<S: Borrow<SourceRow>>(
    shader: &CheckedShader<S>,
    air_of: impl Fn(&SourceRow) -> Result<ShaderAir, String>,
) -> Result<(Vec<u8>, Classification), String> {
    let air = air_of(shader.source.borrow())?;
    let classification = classify_air(shader, &air)?;
    Ok((air.bitcode, classification))
}

/// The classification of `air`, the module the oracle compiles for `shader`. A retained blob is
/// the module the checker classified, so its classification stands. A public fixture is compiled
/// on this host apart from the row the checker assembled, and this host's compiler may keep a
/// loop the committed text folded away (`kernel_instance_as_world_space`), or fold one it kept, so
/// its compiled module is classified again, with the same linked modules. The oracle binds by
/// the checked row's interface, so a compiled module that binds otherwise fails
/// ([`binding::binding_difference`]).
fn classify_air<S: Borrow<SourceRow>>(
    shader: &CheckedShader<S>,
    air: &ShaderAir,
) -> Result<Classification, String> {
    if air.checked {
        return Ok(Classification {
            class: shader.loop_class.clone(),
            fuel: shader.fuel_scope.clone(),
        });
    }
    let (classification, interface) = shader.classify_compiled(&air.bitcode)?;
    if let Some(difference) = interface
        .as_ref()
        .and_then(|compiled| binding::binding_difference(&shader.interface, compiled))
    {
        return Err(format!(
            "the module compiled from {} does not bind as its checked row: {difference}",
            shader.source.borrow().label
        ));
    }
    Ok(classification)
}

/// How the oracle bounds the case's own compiled module: as compiled when loop-free,
/// fuel-instrumented when fuel covers every cycle, else refused.
fn primary_bound(
    classification: Classification,
    invocation_fuel: u32,
) -> Result<OracleBound, ExecutionRefused> {
    match classification {
        Classification {
            class: LoopClass::LoopFree,
            ..
        } => Ok(OracleBound::LoopFree),
        Classification {
            fuel: FuelScope::Covered,
            ..
        } => Ok(OracleBound::Fuel { invocation_fuel }),
        Classification {
            fuel: FuelScope::Uncovered(evidence),
            ..
        } => Err(ExecutionRefused::NotFuelBounded(evidence)),
    }
}

/// A pipeline template runs as compiled: a compiled template module that is not loop-free is
/// refused.
fn template_runs_as_compiled(class: LoopClass) -> Result<(), ExecutionRefused> {
    match class {
        LoopClass::LoopFree => Ok(()),
        LoopClass::Cyclic(evidence) => Err(ExecutionRefused::NotFuelBounded(evidence)),
    }
}

/// What [`QUALIFICATION_RUNS`] runs qualify: the output every run returned, or the reference gap
/// every run reported (no output). Runs that disagree, an output in some and a gap in others or
/// two gaps, qualify nothing.
fn qualify_answers(
    initial: &[u8],
    answers: Vec<RunAnswer>,
) -> Result<(Vec<u8>, MetalStatus), String> {
    let mut outputs = Vec::new();
    let mut gaps = Vec::new();
    for answer in answers {
        match answer {
            Ok(output) => outputs.push(output),
            Err(gap) => gaps.push(gap),
        }
    }
    let runs = gaps.len();
    let mut gaps = gaps.into_iter();
    let Some(gap) = gaps.next() else {
        return Ok((qualify_outputs(initial, outputs)?, MetalStatus::Qualified));
    };
    if !outputs.is_empty() {
        return Err(format!(
            "Metal qualification is nondeterministic: {} of {QUALIFICATION_RUNS} runs returned \
             output and the others could not create the pipeline ({})",
            outputs.len(),
            gap.message
        ));
    }
    if runs != QUALIFICATION_RUNS {
        return Err(format!(
            "Metal qualification requires exactly {QUALIFICATION_RUNS} runs, got {runs}"
        ));
    }
    if let Some(other) = gaps.find(|other| *other != gap) {
        return Err(format!(
            "Metal qualification is nondeterministic: its runs reported different reference \
             gaps ({:?} and {:?})",
            gap.message, other.message
        ));
    }
    gap.validate()?;
    Ok((Vec::new(), MetalStatus::ReferenceGap(gap)))
}

fn qualify_outputs(initial: &[u8], outputs: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    if outputs.len() != QUALIFICATION_RUNS {
        return Err(format!(
            "Metal qualification requires exactly {QUALIFICATION_RUNS} runs, got {}",
            outputs.len()
        ));
    }
    if outputs.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(format!(
            "Metal qualification is nondeterministic across {QUALIFICATION_RUNS} runs"
        ));
    }
    let output = outputs
        .into_iter()
        .next()
        .ok_or_else(|| "qualification performed no runs".to_string())?;
    if output.len() != initial.len() {
        return Err(format!(
            "Metal returned {} selected bytes, expected {}",
            output.len(),
            initial.len()
        ));
    }
    Ok(output)
}

/// Build the native descriptor from the same literal checked by candidate record replay.
/// This creates a configuration object only; device allocation and submission are separate.
#[cfg(target_os = "macos")]
pub fn indirect_command_descriptor(
    resource: &crate::case::indirect_commands::CommandBufferResource,
) -> Result<objc2::rc::Retained<objc2_metal::MTLIndirectCommandBufferDescriptor>, String> {
    use crate::case::indirect_commands::{CommandDescriptor, ComputeDispatch};
    use objc2_metal::{MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType};
    resource.record_storage_bytes()?;
    let descriptor = MTLIndirectCommandBufferDescriptor::new();
    descriptor.setInheritPipelineState(resource.inherit_pipeline_state);
    descriptor.setInheritBuffers(resource.inherit_buffers);
    descriptor.setMaxVertexBufferBindCount(0);
    descriptor.setMaxFragmentBufferBindCount(0);
    descriptor.setMaxKernelBufferBindCount(0);
    let (dynamic_stride, threadgroup_count) = match resource.descriptor {
        CommandDescriptor::Compute {
            dispatch,
            max_buffer_bind_count,
            max_threadgroup_memory_bind_count,
            support_dynamic_attribute_stride,
        } => {
            descriptor.setCommandTypes(match dispatch {
                ComputeDispatch::Threadgroups => MTLIndirectCommandType::ConcurrentDispatch,
                ComputeDispatch::Threads => MTLIndirectCommandType::ConcurrentDispatchThreads,
                ComputeDispatch::Both => {
                    MTLIndirectCommandType::ConcurrentDispatch
                        | MTLIndirectCommandType::ConcurrentDispatchThreads
                }
            });
            descriptor.setMaxKernelBufferBindCount(max_buffer_bind_count as usize);
            (
                support_dynamic_attribute_stride,
                max_threadgroup_memory_bind_count,
            )
        }
        CommandDescriptor::Render {
            draws,
            max_vertex_buffer_bind_count,
            max_fragment_buffer_bind_count,
            max_object_buffer_bind_count,
            max_mesh_buffer_bind_count,
            support_dynamic_attribute_stride,
        } => {
            let types = [
                (draws.primitives, MTLIndirectCommandType::Draw),
                (draws.indexed, MTLIndirectCommandType::DrawIndexed),
                (draws.patches, MTLIndirectCommandType::DrawPatches),
                (
                    draws.indexed_patches,
                    MTLIndirectCommandType::DrawIndexedPatches,
                ),
                (
                    draws.mesh_threadgroups,
                    MTLIndirectCommandType::DrawMeshThreadgroups,
                ),
                (draws.mesh_threads, MTLIndirectCommandType::DrawMeshThreads),
            ]
            .into_iter()
            .filter(|&(allowed, _)| allowed)
            .fold(MTLIndirectCommandType::empty(), |types, (_, kind)| {
                types | kind
            });
            descriptor.setCommandTypes(types);
            descriptor.setMaxVertexBufferBindCount(max_vertex_buffer_bind_count as usize);
            descriptor.setMaxFragmentBufferBindCount(max_fragment_buffer_bind_count as usize);
            unsafe {
                descriptor.setMaxObjectBufferBindCount(max_object_buffer_bind_count as usize);
                descriptor.setMaxMeshBufferBindCount(max_mesh_buffer_bind_count as usize);
            }
            (support_dynamic_attribute_stride, 0)
        }
    };
    // Literal validation bounds this count to the 31 kernel threadgroup-memory slots.
    unsafe { descriptor.setMaxKernelThreadgroupMemoryBindCount(threadgroup_count as usize) };
    descriptor.setSupportDynamicAttributeStride(dynamic_stride);
    Ok(descriptor)
}

mod binding;

pub mod command_guard;
#[cfg(target_os = "macos")]
mod platform;
/// The system Metal device, behind the hang tripwire check: the parity harness's Metal side
/// (`parity::metal`) and the crate's binaries open it here.
#[cfg(target_os = "macos")]
pub use platform::system_device;
/// The libraries the parity harness's Metal side uses (`parity::metal`).
#[cfg(all(test, m2v_gpu, target_os = "macos"))]
pub(crate) use platform::{load_library, shader_air};

#[cfg(test)]
mod tests;

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;
    use crate::source::SourceRow;

    pub fn execute(
        _case: &AuthoredCase,
        _source: OracleSource<'_>,
        _resources: &LiteralResources,
        _reflection: &crate::air::reflect::ShaderInterface,
        _interface: &crate::check::interface::AirInterface,
        _function_tables: &crate::library_module::ResolvedLinkedFunctions,
        _templates: &BTreeMap<String, OracleTemplate<'_>>,
    ) -> Result<Vec<u8>, MetalRunError> {
        Err("Metal qualification requires macOS".into())
    }

    pub fn shader_air(_row: &SourceRow) -> Result<ShaderAir, String> {
        Err("Metal qualification requires macOS".into())
    }

    pub fn reusing_tool_results<R>(body: impl FnOnce() -> R) -> R {
        body()
    }

    pub fn probe_compile(
        _case: &AuthoredCase,
        _source: OracleSource<'_>,
        _interface: &crate::check::interface::AirInterface,
        _function_tables: &crate::library_module::ResolvedLinkedFunctions,
        _constants: &[crate::literal::LiteralFunctionConstant],
        _loops: Option<crate::oracle_fuel::LoopSelection>,
    ) -> Result<CompileProbe, String> {
        Err("Metal qualification requires macOS".into())
    }

    pub fn environment() -> Result<serde_json::Value, String> {
        Err("Metal qualification requires macOS".into())
    }

    pub fn identity() -> Result<MetalIdentity, String> {
        Err("Metal qualification requires macOS".into())
    }

    pub struct PreparedLibrary {
        pub bound: OracleBound,
        pub fuel: Option<crate::oracle_fuel::FuelReport>,
        pub status_slot: Option<u32>,
        pub metallib_bytes: u64,
        pub library_sha256: Vec<String>,
    }

    pub fn prepare(
        _case: &AuthoredCase,
        _source: OracleSource<'_>,
        _interface: &crate::check::interface::AirInterface,
        _linked: &crate::library_module::ResolvedLinkedFunctions,
        _templates: &BTreeMap<String, OracleTemplate<'_>>,
    ) -> Result<PreparedLibrary, String> {
        Err("Metal qualification requires macOS".into())
    }
}
