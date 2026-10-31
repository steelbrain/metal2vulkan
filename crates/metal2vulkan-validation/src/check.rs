use crate::air::reflect::ShaderInterface;
use crate::case::{AuthoredCase, ExecutionSafety, ResourceRole, Stage, TextureFormat};
use crate::library_module::ResolvedLinkedFunctions;
use crate::source::{find_case_source, SourceRow};
use base64::Engine as _;
use std::borrow::Borrow;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Arc;

pub mod admission;
pub mod caps;
pub mod cfg_fact;
mod conformance;
pub mod grid_reach;
pub mod interface;
pub mod loop_free;
pub mod safety_census;
pub mod table_index;

pub use cfg_fact::has_cfg_cycle;
pub use loop_free::{CycleEvidence, FuelScope, LoopClass};

/// A shader checked against an owned or caller-borrowed exact source row.
pub struct CheckedShader<S> {
    pub case: AuthoredCase,
    pub source: S,
    /// The Metal-ABI facts of the entry, from the AIR input layer: what the Metal oracle binds.
    pub interface: interface::AirInterface,
    /// The entry's declared interface ([`crate::air::reflect`], the old engine's `inspect` as
    /// validation keeps it): the executor-contract vocabulary the census shares (the observation
    /// requirements included, which read the declared type names), and the families
    /// [`interface::AirInterface`] does not model yet (imageblocks, tessellation). A vertex
    /// observation's companion is derived from the AIR ([`CheckedShader::companion`]).
    pub reflection: ShaderInterface,
    /// The generated stage the case draws with (a fragment case's passthrough vertex, an observed
    /// vertex case's observer fragment), derived from the entry's AIR: both executors draw exactly
    /// this companion.
    pub companion: Option<crate::companion::Companion>,
    /// Every tooling requirement of the case against its source
    /// (`executor_contract::checked_requirements`). The checker accepts the case only when the
    /// Metal oracle and some candidate run it; each executor refuses the requirements that stop it.
    pub requirements: std::collections::BTreeSet<crate::requirement::ToolingRequirement>,
    pub linked_functions: ResolvedLinkedFunctions,
    pub input_sha256: String,
    /// Mechanical loop-freedom of the entry module together with every linked module it calls.
    pub loop_class: LoopClass,
    /// Whether the Metal oracle's fuel transform bounds every cycle of those modules.
    pub fuel_scope: FuelScope,
    /// The case entry's projection of its AIR when the AIR declares several entries, `None` when
    /// the AIR is the entry's own module: what [`CheckedShader::entry_ll`] reads.
    pub entry_projection: Option<String>,
}

impl<S: Borrow<SourceRow>> CheckedShader<S> {
    /// The text facts about the case's entry are read from: its AIR when the AIR declares one
    /// entry, and otherwise the entry's projection of it ([`crate::air::entry_text`]). The checker
    /// derives the case's reflection and requirements from it, and an executor reads its facts from
    /// it; every executor still compiles or translates the AIR itself, selecting the case's entry.
    pub fn entry_ll(&self) -> &str {
        self.entry_projection
            .as_deref()
            .unwrap_or(&self.source.borrow().air_ll)
    }

    /// Classify `bitcode`, the module an executor compiles for this shader's entry in place of
    /// its checked row, together with the linked modules the checker classified the row with:
    /// the same rules and the same executed set, only the entry module's bytes differ. The
    /// compiled entry's [`interface::AirInterface`] for the case's constants comes with it, so the
    /// executor can hold the compiled module to the interface it binds by; `None` when the
    /// module does not decode, which the classification refuses. An entry that decodes but whose
    /// interface does not derive is an error.
    pub(crate) fn classify_compiled(
        &self,
        bitcode: &[u8],
    ) -> Result<(loop_free::Classification, Option<interface::AirInterface>), String> {
        let row = self.source.borrow();
        let (classification, entry) = loop_free::classify_compiled_with_fuel(
            &self.case.entry,
            &row.label,
            bitcode,
            // `executed_modules` yields the entry module first.
            executed_modules(row, &self.linked_functions).skip(1),
        );
        let interface = match entry {
            Ok(module) => Some(
                air_interface(&self.case, Ok(module))
                    .map_err(|error| format!("the module compiled from {}: {error}", row.label))?
                    .facts,
            ),
            Err(_) => None,
        };
        Ok((classification, interface))
    }
}

/// One primary shader and its deduplicated, transitively checked pipeline templates.
pub struct CheckedCase<S = SourceRow> {
    pub primary: CheckedShader<S>,
    pub pipeline_templates: BTreeMap<String, CheckedShader<Arc<SourceRow>>>,
}

impl<S> CheckedCase<S> {
    /// Mechanical loop-freedom of everything this case executes: the primary shader with its
    /// linked modules and every transitively referenced pipeline template.
    pub fn loop_class(&self) -> &LoopClass {
        std::iter::once(&self.primary.loop_class)
            .chain(
                self.pipeline_templates
                    .values()
                    .map(|template| &template.loop_class),
            )
            .find(|class| !class.is_loop_free())
            .unwrap_or(&self.primary.loop_class)
    }

    /// Every tooling requirement of what this case executes: the primary shader's and each
    /// pipeline template's ([`CheckedShader::requirements`]). Each executor refuses the ones that
    /// stop it.
    pub fn requirements(
        &self,
    ) -> std::collections::BTreeSet<crate::requirement::ToolingRequirement> {
        std::iter::once(&self.primary.requirements)
            .chain(
                self.pipeline_templates
                    .values()
                    .map(|template| &template.requirements),
            )
            .flatten()
            .copied()
            .collect()
    }

    /// What the Metal oracle's fuel transform bounds of everything this case executes. It
    /// instruments only the primary shader; a pipeline template runs as compiled, so a cyclic
    /// template is uncovered.
    pub fn fuel_scope(&self) -> FuelScope {
        for template in self.pipeline_templates.values() {
            if let Some(evidence) = template.loop_class.evidence() {
                return FuelScope::Uncovered(evidence.clone());
            }
        }
        self.primary.fuel_scope.clone()
    }

    /// Every metadata value the AIR layer does not evaluate for this case, primary shader first,
    /// each naming the check it leaves out: empty when every check is complete.
    pub fn unresolved(&self) -> Vec<String> {
        let primary = self
            .primary
            .interface
            .unresolved
            .iter()
            .map(|value| value.to_string());
        let templates = self.pipeline_templates.iter().flat_map(|(id, template)| {
            template
                .interface
                .unresolved
                .iter()
                .map(move |value| format!("pipeline template {id}: {value}"))
        });
        primary.chain(templates).collect()
    }
}

/// An installed case whose manifest no longer checks against its AIR.
pub struct StaleCase {
    pub case_id: String,
    pub name: String,
    pub errors: Vec<String>,
}

/// An installed case the checker accepts with some checks left out: metadata values the AIR
/// layer does not evaluate ([`CheckedCase::unresolved`]).
pub struct PartialCase {
    pub case_id: String,
    pub name: String,
    pub unresolved: Vec<String>,
}

/// What re-checking the installed cases found.
pub struct Recheck {
    pub total: usize,
    /// The cases the checker now rejects.
    pub stale: Vec<StaleCase>,
    /// The accepted cases whose checks are partial.
    pub partial: Vec<PartialCase>,
}

/// Re-check every installed case against the CURRENT checker, newest semantics.
///
/// An authored case is evidence about the bytes this translator emits today. A product change that
/// moves a descriptor slot, or that starts requiring a resource, turns that evidence into a manifest
/// the shared checker no longer accepts -- with its Metal and candidate observations still committed
/// beside it, describing a run nobody can reproduce. Nothing on the ordinary path re-reads the store
/// for that: `corpus-index --check` verifies the index against the shards, not the shards against
/// the product.
///
/// `e84ea3fa` is the case in point. Reading `air.location_index` positionally moved 41 authored
/// texture arrays off the Metal slot their manifests declare; its full-corpus A/B measured 0
/// translation status changes, which is the wrong instrument for this, and the store went 133
/// commits with 60 stale cases in it.
pub fn recheck_installed_cases(
    root: &Path,
    store: &crate::store::CorpusStore,
) -> Result<Recheck, String> {
    let cases = store.read_all_cases()?;
    let mut recheck = Recheck {
        total: cases.len(),
        stale: Vec::new(),
        partial: Vec::new(),
    };
    for case in cases {
        let case_id = case.case_id.clone();
        let name = case.name.clone();
        match check_case(root, case) {
            Ok(checked) => {
                let unresolved = checked.unresolved();
                if !unresolved.is_empty() {
                    recheck.partial.push(PartialCase {
                        case_id,
                        name,
                        unresolved,
                    });
                }
            }
            Err(errors) => recheck.stale.push(StaleCase {
                case_id,
                name,
                errors,
            }),
        }
    }
    Ok(recheck)
}

pub fn check_case(root: &Path, case: AuthoredCase) -> Result<CheckedCase, Vec<String>> {
    checked_execution(root, check_shader(root, case)?)
}

fn check_shader(root: &Path, case: AuthoredCase) -> Result<CheckedShader<SourceRow>, Vec<String>> {
    // A case naming its public fixture is checked against that fixture first, so an edited
    // fixture is named rather than reported as an AIR no shard holds.
    if let Some(path) = &case.public_source {
        crate::source::check_public_source(path, &case.air_sha256).map_err(|error| vec![error])?;
    }
    let source = required_source(root, &case.air_sha256, &case.entry)?;
    check_shader_against_source(root, case, source)
}

/// The row of entry `entry` of the module `air_sha256`.
fn required_source(root: &Path, air_sha256: &str, entry: &str) -> Result<SourceRow, Vec<String>> {
    find_case_source(root, air_sha256, entry)
        .map_err(|error| vec![error])?
        .ok_or_else(|| {
            vec![format!(
                "AIR {air_sha256} entry {entry:?} does not exist in aligned source shards or \
                 public fixtures"
            )]
        })
}

/// Check an authored case against an already selected exact AIR row.
/// The primary source stays borrowed; only selected dependency rows are loaded.
pub fn check_case_against_source<'a>(
    root: &Path,
    case: AuthoredCase,
    source: &'a SourceRow,
) -> Result<CheckedCase<&'a SourceRow>, Vec<String>> {
    checked_execution(root, check_shader_against_source(root, case, source)?)
}

fn checked_execution<S>(
    root: &Path,
    primary: CheckedShader<S>,
) -> Result<CheckedCase<S>, Vec<String>> {
    let pipeline_templates = resolve_pipeline_templates(root, &primary.case, &primary.interface)?;
    check_render_replay_templates(&primary.case, &primary.interface, &pipeline_templates)?;
    check_replayed_strides(&primary.case, &pipeline_templates)?;
    Ok(CheckedCase {
        primary,
        pipeline_templates,
    })
}

fn check_case_identity(case: &AuthoredCase) -> (Vec<String>, String) {
    let mut errors = case.validate_literal_resources().err().unwrap_or_default();
    errors.extend(caps::cap_errors(case));
    match case.computed_case_id() {
        Ok(computed) if computed != case.case_id => errors.push(format!(
            "case_id mismatch: manifest={} computed={computed}",
            case.case_id
        )),
        Err(error) => errors.push(error),
        _ => {}
    }
    let input_sha256 = match case.computed_input_sha256() {
        Ok(digest) => digest,
        Err(error) => {
            errors.push(error);
            String::new()
        }
    };
    (errors, input_sha256)
}

fn check_shader_against_source<S: Borrow<SourceRow>>(
    root: &Path,
    case: AuthoredCase,
    selected: S,
) -> Result<CheckedShader<S>, Vec<String>> {
    let (mut errors, input_sha256) = check_case_identity(&case);
    let source = selected.borrow();
    if source.air_sha256 != case.air_sha256 {
        errors.push(format!(
            "AIR hash mismatch: case={} source={}",
            case.air_sha256, source.air_sha256
        ));
    }
    if let Some(path) = &case.public_source {
        if crate::source::public_source_path(&source.label).as_deref() != Some(path.as_str()) {
            errors.push(format!(
                "public_source {path} is not the checked AIR's source {}",
                source.label
            ));
        }
    }
    if source.stage != case.stage.metadata_label() {
        errors.push(format!(
            "stage mismatch: manifest={:?} AIR metadata={}",
            case.stage, source.stage
        ));
    }
    if source.entry != case.entry {
        errors.push(format!(
            "entry mismatch: manifest={:?} AIR metadata={:?}",
            case.entry, source.entry
        ));
    }
    // Facts about the case's entry are read from its projection of the AIR.
    let entry_projection = match crate::air::entry_text(&source.air_ll, &case.entry) {
        Ok(std::borrow::Cow::Borrowed(_)) => None,
        Ok(std::borrow::Cow::Owned(projection)) => Some(projection),
        Err(error) => {
            errors.push(format!(
                "the AIR has no projection onto the entry {:?}: {error}",
                case.entry
            ));
            return Err(errors);
        }
    };
    let entry_ll = entry_projection.as_deref().unwrap_or(&source.air_ll);
    // A layered pass (the function reads its render target slice) draws into one-slice array
    // targets, of which no executor makes a multisampled one.
    if case.raster_sample_count.is_some()
        && crate::executor_contract::render_target_array::render_target_array_length(
            case.stage, entry_ll,
        ) > 0
    {
        errors.push(
            "a multisampled render pass of a function that reads its render target slice: a \
             layered multisampled pass is not modelled"
                .into(),
        );
    }
    let linked_functions = match crate::library_module::resolve_linked_functions(root, &case) {
        Ok(tables) => tables,
        Err(table_errors) => {
            errors.extend(table_errors);
            ResolvedLinkedFunctions::default()
        }
    };
    validate_visible_reference_closure(entry_ll, &linked_functions, &mut errors);
    let (
        loop_free::Classification {
            class: loop_class,
            fuel: fuel_scope,
        },
        decoded_modules,
    ) = loop_free::decode_and_classify(&case.entry, executed_modules(source, &linked_functions));
    validate_execution_safety(case.execution_safety, &loop_class, &mut errors);

    let (entry_module, linked_modules) = match decoded_modules {
        Ok(mut modules) => {
            let linked = modules.split_off(1);
            let (_, entry) = modules.remove(0);
            (Ok(entry), linked)
        }
        Err(evidence) => (Err(evidence), Vec::new()),
    };
    let interface = match air_interface(&case, entry_module) {
        Ok(interface) => interface,
        Err(error) => {
            errors.push(error);
            return Err(errors);
        }
    };
    conformance::validate(&case, &interface.facts, &mut errors);
    {
        let executed = std::iter::once((loop_free::ModuleRole::Entry, &interface.decoded))
            .chain(linked_modules.iter().map(|(role, module)| (*role, module)))
            .collect::<Vec<_>>();
        errors.extend(table_index::past_the_end(
            &case.entry,
            &executed,
            &interface.facts,
        ));
    }
    if let Some(error) = interface.omitted_constants {
        errors.push(error);
    }
    let companion = interface.companion.unwrap_or_else(|error| {
        errors.push(error);
        None
    });
    let interface = interface.facts;

    let reflection = match reflect(entry_ll, &case) {
        Ok(reflection) => reflection,
        Err(error) => {
            errors.push(format!("reflection failed: {error}"));
            return Err(errors);
        }
    };
    // Where the coordinate-mapping calls read their maps is the placement's one decision, which
    // classification shares.
    conformance::validate_rate_maps(
        &case,
        &interface,
        &crate::executor_contract::rate_map::rate_map_placement(entry_ll, &reflection),
        &mut errors,
    );
    let requirements = crate::executor_contract::require_reflection(
        &case,
        entry_ll,
        &reflection,
        "shared executor",
    )
    .unwrap_or_else(|error| {
        errors.push(error);
        Default::default()
    });
    let imageblock = validate_reflected_families(&case, &reflection, &mut errors);
    let mut partner_targets =
        validate_mesh_partners(&case, &source.air_ll, &reflection, &mut errors);
    // An object or mesh case's generated observer fragment, in its fragment partner's place,
    // writes render target 0.
    if matches!(
        &companion,
        Some(crate::companion::Companion::MeshStages(stages)) if stages.fragment.is_some()
    ) {
        partner_targets.insert(0);
    }
    let partner_writes = matches!(
        case.output,
        crate::case::OutputSelection::RenderTarget { index, .. }
            if partner_targets.contains(&index)
    );
    if !conformance::selected_output_is_writable(&case, &interface, &imageblock) && !partner_writes
    {
        errors.push(OUTPUT_NOT_WRITABLE.into());
    }
    if errors.is_empty() {
        Ok(CheckedShader {
            case,
            source: selected,
            interface,
            reflection,
            companion,
            requirements,
            linked_functions,
            input_sha256,
            loop_class,
            fuel_scope,
            entry_projection,
        })
    } else {
        Err(errors)
    }
}

/// The entry's [`interface::AirInterface`], whether the constants the case omits are proven
/// unused, and the companion the case draws with.
struct CheckedInterface {
    facts: interface::AirInterface,
    /// The entry module the facts were read from, for the scans over the executed modules.
    decoded: loop_free::DecodedModule,
    omitted_constants: Option<String>,
    companion: Result<Option<crate::companion::Companion>, String>,
}

fn air_interface(
    case: &AuthoredCase,
    entry_module: Result<loop_free::DecodedModule, CycleEvidence>,
) -> Result<CheckedInterface, String> {
    let decoded =
        entry_module.map_err(|evidence| format!("the AIR does not decode: {evidence}"))?;
    let (abi, constants, facts) = entry_interface(case, &decoded)?;
    let omitted_constants = facts
        .function_constants
        .keys()
        .any(|index| !constants.contains_key(index))
        .then(|| omitted_constants_error(decoded.module(), abi, &case.entry, constants))
        .flatten();
    let companion = crate::companion::derive(case, decoded.module(), abi, &facts);
    Ok(CheckedInterface {
        facts,
        decoded,
        omitted_constants,
        companion,
    })
}

/// The AIR-layer interface of `case`'s entry in its decoded entry module under the case's
/// function constants, with the metadata ABI and the constants it was derived from. The checker
/// and the vertex descriptor of a case its executors draw with share it.
pub(crate) fn entry_interface<'m>(
    case: &AuthoredCase,
    decoded: &'m loop_free::DecodedModule,
) -> Result<
    (
        &'m metal2vulkan_air::abi::AirAbi,
        BTreeMap<u32, metal2vulkan_air::canon::ConstValue>,
        interface::AirInterface,
    ),
    String,
> {
    let abi = decoded
        .abi()
        .map_err(|error| format!("the AIR metadata ABI does not decode: {error}"))?;
    let constants = case_constants(case)?;
    let facts = interface::AirInterface::new(decoded.module(), abi, &case.entry, &constants)?;
    Ok((abi, constants, facts))
}

/// The case's function constants as the AIR layer binds them.
pub(crate) fn case_constants(
    case: &AuthoredCase,
) -> Result<BTreeMap<u32, metal2vulkan_air::canon::ConstValue>, String> {
    crate::literal::function_constants(case)?
        .into_iter()
        .map(|constant| {
            let ty = constant
                .scalar_type
                .metal_data_type(constant.lanes)
                .ok_or_else(|| {
                    format!(
                        "function constant {} of type {:?}x{} has no Metal data type",
                        constant.index, constant.scalar_type, constant.lanes
                    )
                })?;
            Ok((
                constant.index,
                metal2vulkan_air::canon::ConstValue {
                    ty,
                    bytes: constant.bytes.into(),
                },
            ))
        })
        .collect()
}

/// An omitted function constant is undefined in the pipeline. The case is valid only when no
/// value use of it survives specialization: the AIR layer's canonicalization names each undefined
/// constant whose value the specialized body reads (`Entry::undefined_reads`). A refusal leaves it
/// unproven.
fn omitted_constants_error(
    module: &metal2vulkan_air::ir::Module,
    abi: &metal2vulkan_air::abi::AirAbi,
    entry: &str,
    constants: BTreeMap<u32, metal2vulkan_air::canon::ConstValue>,
) -> Option<String> {
    let request = metal2vulkan_air::canon::SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_string(),
        constants,
    };
    // Deterministic caps and no deadline: the verdict must not depend on host load.
    let mut meter = metal2vulkan_air::budget::Meter::new(&metal2vulkan::Budget::default());
    match metal2vulkan_air::canon::specialize(module.clone(), abi, &request, &mut meter)
        .map_err(metal2vulkan::error::Error::from)
    {
        Ok(entry) => entry.undefined_reads.first().map(|index| {
            format!(
                "function constant {index} is omitted, but a value use of it survives \
                 specialization"
            )
        }),
        Err(error) => Some(format!(
            "cannot validate omitted function-constant values: {error}"
        )),
    }
}

/// Bound retained dependency state independently of the worker's byte/time guards.
const MAX_PIPELINE_TEMPLATES: usize = 256;

/// Which pipeline state an argument-buffer member holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineClass {
    Compute,
    Render,
}

/// The family of pipeline a command buffer's commands run: compute commands a compute pipeline,
/// render commands a render one.
fn command_family(
    resource: &crate::case::indirect_commands::CommandBufferResource,
) -> PipelineClass {
    if resource.is_render() {
        PipelineClass::Render
    } else {
        PipelineClass::Compute
    }
}

/// The draws a render template runs: a vertex function's, of its draw's topology class, a
/// post-tessellation vertex function's patches, of its patch type and control-point count, or a
/// mesh render pipeline's mesh draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TemplateDraws {
    Primitives(metal2vulkan::metal::PrimitiveTopologyClass),
    Patches {
        domain: crate::air::reflect::PatchDomain,
        control_points: u32,
    },
    Mesh,
}

/// The render templates an executed render command of `case` may select: each render pipeline
/// state's (`kinds` names each token's family) and each render execution's parent. Each must be a
/// rasterization-disabled vertex case, whose vertex inputs a command's vertex buffers feed at the
/// template's slots (its stride a dynamic layout's), a rasterization-disabled post-tessellation
/// one without control points or patch inputs, or an object or mesh case whose mesh draw does not
/// rasterize, whose draw writes only memory: the execution's render pass has no attachment a case
/// authors, so it is the Metal oracle's 1x1 scaffolding, which a rasterization-disabled template's
/// pipeline names and no draw writes. A template the host sets on a command, or an execution's
/// parent, runs draws its command buffer holds: a mesh template a buffer's that allows mesh
/// draws, and another a buffer's that allows draws of another kind (Metal's process ends in the
/// host's setter of a pipeline whose kind the buffer's command types hold no draw of: a mesh
/// pipeline on a buffer of no mesh draw, a vertex pipeline on a buffer of mesh draws alone). And
/// each host-encoded draw is one the pipeline it runs draws where the literal states it (the
/// parent's when the commands inherit it, else the command's own): a draw of primitives or an
/// indexed draw of the topology class of a vertex template, which Metal compiles for its case's
/// class, a draw of another class being no draw it defines; a patch draw of a post-tessellation
/// template, stating the template's control points, whose factors (a triangle patch's 8 bytes, a
/// quad patch's 12, at each patch's position) lie inside its factor buffer. No host-encoded draw is
/// a mesh draw.
fn check_render_replay_templates(
    case: &AuthoredCase,
    interface: &interface::AirInterface,
    resolved: &BTreeMap<String, CheckedShader<Arc<SourceRow>>>,
) -> Result<(), Vec<String>> {
    use crate::air::reflect::PatchDomain;
    use crate::case::indirect_commands::HostAction;
    use metal2vulkan_air::abi::ParamRole;
    let executes_render = case.command_buffers.iter().any(|resource| {
        resource.execute.is_some() && command_family(resource) == PipelineClass::Render
    });
    if !executes_render {
        return Ok(());
    }
    let render_states = case.pipeline_states.iter().filter(|resource| {
        interface
            .member_at(resource.buffer_binding, resource.field_offset)
            .is_some_and(|(_, member)| matches!(member.role, ParamRole::RenderPipelineState(_)))
    });
    let mut tokens = BTreeMap::new();
    for (token, case_id) in render_states
        .map(|resource| (resource.token, &resource.case_id))
        .chain(case.command_buffers.iter().filter_map(|resource| {
            let template = resource.execute.as_ref()?.pipeline.as_ref()?;
            (command_family(resource) == PipelineClass::Render)
                .then_some((template.token, &template.case_id))
        }))
    {
        let template = resolved
            .get(case_id)
            .ok_or_else(|| vec![format!("pipeline template {case_id} was not resolved")])?;
        if matches!(template.case.stage, Stage::Object | Stage::Mesh) {
            if template
                .case
                .mesh_draw
                .as_ref()
                .is_none_or(|draw| draw.rasterization)
            {
                return Err(vec![format!(
                    "pipeline template {case_id}, which an executed render command may select, \
                     is an object or mesh case whose mesh draw rasterizes"
                )]);
            }
            tokens.insert(token, TemplateDraws::Mesh);
            continue;
        }
        let unfetched = template
            .case
            .tessellation
            .as_ref()
            .is_none_or(|draw| draw.control_points.is_empty() && draw.patch_inputs.is_empty());
        if !template.case.is_rasterization_disabled_vertex() || !unfetched {
            return Err(vec![format!(
                "pipeline template {case_id}, which an executed render command may select, is \
                 not a rasterization-disabled vertex case without control points or patch inputs, \
                 or an object or mesh case that does not rasterize"
            )]);
        }
        let draws = match &template.reflection.tessellation {
            Some(patches) => TemplateDraws::Patches {
                domain: patches.domain,
                control_points: patches.control_point_count,
            },
            None => {
                let draw = template
                    .case
                    .draw
                    .as_ref()
                    .ok_or_else(|| vec![format!("pipeline template {case_id} has no draw")])?;
                TemplateDraws::Primitives(draw.primitive.topology_class())
            }
        };
        tokens.insert(token, draws);
    }
    let lengths = case.buffer_lengths();
    for resource in &case.command_buffers {
        let crate::case::indirect_commands::CommandDescriptor::Render { draws: held, .. } =
            resource.descriptor
        else {
            continue;
        };
        let parent = resource
            .execute
            .as_ref()
            .and_then(|execution| execution.pipeline.as_ref())
            .map(|template| template.token);
        // Each template the host sets on this buffer's commands, or its parent, of a kind the
        // buffer holds draws of.
        let other = held.primitives || held.indexed || held.patches || held.indexed_patches;
        for token in resource
            .commands
            .iter()
            .filter_map(|command| command.pipeline)
            .chain(parent)
        {
            let Some(draws) = tokens.get(&token) else {
                continue;
            };
            let holds = match draws {
                TemplateDraws::Mesh => held.mesh(),
                TemplateDraws::Primitives(_) | TemplateDraws::Patches { .. } => other,
            };
            if !holds {
                return Err(vec![format!(
                    "command buffer ({}, {}) runs pipeline {token}, a template whose draws its \
                     command types hold none of",
                    resource.buffer_binding, resource.field_offset
                )]);
            }
        }
        for command in &resource.commands {
            let Some(action) = command.action else {
                continue;
            };
            let token = if resource.inherit_pipeline_state {
                parent
            } else {
                command.pipeline
            };
            let Some(draws) = token.and_then(|token| tokens.get(&token)) else {
                continue;
            };
            let what = format!(
                "host command {} of command buffer ({}, {})",
                command.index, resource.buffer_binding, resource.field_offset
            );
            let runs = match (action, *draws) {
                (
                    HostAction::DrawPrimitives { primitive, .. }
                    | HostAction::DrawIndexedPrimitives { primitive, .. },
                    TemplateDraws::Primitives(class),
                ) => class == primitive.topology_class(),
                (
                    HostAction::DrawPatches {
                        control_points,
                        patch_start,
                        patch_count,
                        factor_buffer,
                        factor_buffer_offset,
                        ..
                    },
                    TemplateDraws::Patches {
                        domain,
                        control_points: pipeline_points,
                    },
                ) => {
                    let size = match domain {
                        PatchDomain::Triangle => 8,
                        PatchDomain::Quad => 12,
                        PatchDomain::Isoline => {
                            return Err(vec![format!("{what}: Metal has no isoline patches")])
                        }
                    };
                    let end = u64::from(factor_buffer_offset)
                        + size * (u64::from(patch_start) + u64::from(patch_count));
                    // The literal checked the factor buffer is a case buffer.
                    if lengths
                        .get(&factor_buffer)
                        .is_none_or(|&length| end > length as u64)
                    {
                        return Err(vec![format!(
                            "{what}: its patches' factors end at byte {end}, past factor buffer \
                             {factor_buffer}"
                        )]);
                    }
                    control_points == pipeline_points
                }
                _ => false,
            };
            if !runs {
                return Err(vec![format!(
                    "{what} draws what the pipeline it runs does not draw"
                )]);
            }
        }
    }
    Ok(())
}

/// Each executed host command of `case` that draws or dispatches, on a command buffer whose
/// descriptor supports dynamic attribute strides, binds every buffer its pipeline's layout of
/// `MTLBufferLayoutStrideDynamic` reads (the template's dynamic stage or vertex inputs, at their
/// slots) with the stride form: Metal's fetch reads one bound without a stride there at stride
/// 2^32 - 1, which the runtime's 32-bit fetch cannot place, so it refuses such a command at the
/// commit (`RuntimeGap::IndirectCommandUnstridedBuffer`). Without that support the records hold
/// no stride word, Metal reads stride 0 and the runtime gives it, so nothing is required. A
/// command that inherits its buffers, or runs nothing, binds none of its own.
fn check_replayed_strides(
    case: &AuthoredCase,
    resolved: &BTreeMap<String, CheckedShader<Arc<SourceRow>>>,
) -> Result<(), Vec<String>> {
    use crate::case::indirect_commands::CommandDescriptor;
    use crate::executor_contract::{buffer_slots, stage_inputs};
    let states: BTreeMap<u64, &String> = case
        .pipeline_states
        .iter()
        .map(|resource| (resource.token, &resource.case_id))
        .collect();
    for resource in &case.command_buffers {
        let Some(execution) = &resource.execute else {
            continue;
        };
        if resource.inherit_buffers {
            continue;
        }
        // Without dynamic attribute strides a record holds no stride word, so Metal reads stride
        // 0 and so does the runtime: nothing to require.
        let dynamic_strides = match resource.descriptor {
            CommandDescriptor::Compute {
                support_dynamic_attribute_stride,
                ..
            }
            | CommandDescriptor::Render {
                support_dynamic_attribute_stride,
                ..
            } => support_dynamic_attribute_stride,
        };
        if !dynamic_strides {
            continue;
        }
        let [start, end] = resource.execution_range;
        for command in &resource.commands {
            if command.action.is_none() || !(start..end).contains(&command.index) {
                continue;
            }
            let case_id = if resource.inherit_pipeline_state {
                execution
                    .pipeline
                    .as_ref()
                    .map(|template| &template.case_id)
            } else {
                command
                    .pipeline
                    .and_then(|token| states.get(&token).copied())
            };
            let Some(template) = case_id.and_then(|id| resolved.get(id)) else {
                continue;
            };
            let (dynamic, bindings) = match command_family(resource) {
                PipelineClass::Compute => {
                    let fetches = stage_inputs::authored(
                        &template.case.kernel_stage_inputs,
                        &template.interface,
                    )
                    .map_err(|error| vec![error])?;
                    let slots = stage_inputs::bound_strides(&fetches)
                        .into_keys()
                        .collect::<Vec<_>>();
                    (slots, &command.buffers)
                }
                PipelineClass::Render => {
                    let slots =
                        buffer_slots::vertex_input_slots(&template.case, &template.interface)
                            .map_err(|error| vec![error])?;
                    let dynamic = template
                        .case
                        .vertex_inputs
                        .iter()
                        .filter(|input| input.dynamic_stride)
                        .filter_map(|input| slots.get(&input.location).copied())
                        .collect::<Vec<_>>();
                    (dynamic, &command.vertex_buffers)
                }
            };
            let unstrided = bindings.iter().find(|binding| {
                binding.attribute_stride.is_none() && dynamic.contains(&binding.index)
            });
            if let Some(binding) = unstrided {
                return Err(vec![format!(
                    "host command {} of command buffer ({}, {}) binds buffer index {} without a \
                     stride, which its pipeline's dynamic layout reads: Metal's fetch reads it at \
                     stride 2^32 - 1 there and the runtime refuses the command",
                    command.index, resource.buffer_binding, resource.field_offset, binding.index
                )]);
            }
        }
    }
    Ok(())
}

fn resolve_pipeline_templates(
    root: &Path,
    case: &AuthoredCase,
    interface: &interface::AirInterface,
) -> Result<BTreeMap<String, CheckedShader<Arc<SourceRow>>>, Vec<String>> {
    use metal2vulkan_air::abi::ParamRole;

    let mut pending = BTreeMap::new();
    let mut identities = BTreeMap::new();
    let mut resolved = BTreeMap::new();
    let mut sources = BTreeMap::new();
    let mut enqueue =
        |case: &AuthoredCase,
         interface: &interface::AirInterface,
         pending: &mut BTreeMap<String, (String, PipelineClass)>| {
            let mut templates = Vec::new();
            let mut kinds = BTreeMap::new();
            for resource in &case.pipeline_states {
                // The stage class the argument-buffer member's handle type admits: a compute
                // pipeline state takes a kernel, a render pipeline state a vertex or fragment.
                let kind = interface
                    .member_at(resource.buffer_binding, resource.field_offset)
                    .and_then(|(_, member)| match member.role {
                        ParamRole::ComputePipelineState(_) => Some(PipelineClass::Compute),
                        ParamRole::RenderPipelineState(_) => Some(PipelineClass::Render),
                        _ => None,
                    });
                let Some(kind) = kind else {
                    return Err(vec![format!(
                        "pipeline-state field ({}, {}) is no pipeline-state member in AIR",
                        resource.buffer_binding, resource.field_offset
                    )]);
                };
                kinds.insert(resource.token, kind);
                templates.push((&resource.air_sha256, &resource.case_id, kind));
            }
            // An execution's parent pipeline is a template of its commands' family: a compute
            // template for compute commands, a render one for render commands.
            for resource in &case.command_buffers {
                if let Some(template) = resource
                    .execute
                    .as_ref()
                    .and_then(|execution| execution.pipeline.as_ref())
                {
                    let kind = command_family(resource);
                    kinds.insert(template.token, kind);
                    templates.push((&template.air_sha256, &template.case_id, kind));
                }
            }
            // A host-encoded command sets a pipeline of its own family.
            for resource in &case.command_buffers {
                for command in &resource.commands {
                    let Some(token) = command.pipeline else {
                        continue;
                    };
                    if kinds
                        .get(&token)
                        .is_some_and(|kind| *kind != command_family(resource))
                    {
                        return Err(vec![format!(
                            "host command {} of command buffer ({}, {}) sets pipeline {token}, \
                             a template of the other command family",
                            command.index, resource.buffer_binding, resource.field_offset,
                        )]);
                    }
                }
            }
            for (air_sha256, case_id, kind) in templates {
                let identity = (air_sha256.clone(), kind);
                if let Some(previous) = identities.get(case_id) {
                    if previous != &identity {
                        return Err(vec![format!("pipeline template {case_id} has conflicting source identities or compute/render uses")]);
                    }
                } else {
                    if identities.len() == MAX_PIPELINE_TEMPLATES {
                        return Err(vec![format!("pipeline template dependency graph exceeds {MAX_PIPELINE_TEMPLATES} distinct templates")]);
                    }
                    identities.insert(case_id.clone(), identity);
                    pending.insert(case_id.clone(), (air_sha256.clone(), kind));
                }
            }
            Ok(())
        };
    enqueue(case, interface, &mut pending)?;
    let store = crate::store::CorpusStore::new(root);
    while let Some((id, (air_sha256, kind))) = pending.pop_first() {
        let template = crate::case::indirect_commands::resolve_template(&store, &air_sha256, &id)
            .map_err(|error| vec![error])?;
        let expected = match template.stage {
            Stage::Kernel => PipelineClass::Compute,
            Stage::Vertex | Stage::Fragment | Stage::Object | Stage::Mesh => PipelineClass::Render,
        };
        if kind != expected {
            return Err(vec![format!(
                "pipeline template {id} stage {:?} disagrees with its AIR handle kind",
                template.stage
            )]);
        }
        let source = match sources.entry((template.air_sha256.clone(), template.entry.clone())) {
            std::collections::btree_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
            std::collections::btree_map::Entry::Vacant(entry) => {
                Arc::clone(entry.insert(Arc::new(required_source(
                    root,
                    &template.air_sha256,
                    &template.entry,
                )?)))
            }
        };
        let checked = check_shader_against_source(root, template, source)?;
        enqueue(&checked.case, &checked.interface, &mut pending)?;
        resolved.insert(id, checked);
    }
    Ok(resolved)
}

fn validate_visible_reference_closure(
    entry_ll: &str,
    linked: &ResolvedLinkedFunctions,
    errors: &mut Vec<String>,
) {
    let authored = linked
        .references
        .iter()
        .map(
            |reference| crate::air::visible_references::AuthoredReference {
                symbol: &reference.function,
                module_ll: &reference.module.air_ll,
            },
        )
        .collect::<Vec<_>>();
    let modules = std::iter::once(entry_ll).chain(linked.visible.iter().flat_map(|table| {
        table
            .entries
            .iter()
            .map(|entry| entry.module.air_ll.as_str())
    }));
    let modules = modules.chain(linked.intersection.iter().flat_map(|table| {
        table.entries.iter().filter_map(|entry| match entry {
            crate::library_module::ResolvedIntersectionFunctionEntry::Linked(entry) => {
                Some(entry.module.air_ll.as_str())
            }
            crate::library_module::ResolvedIntersectionFunctionEntry::OpaqueTriangle { .. } => None,
        })
    }));
    let mut seen = HashSet::new();
    for module in modules {
        if seen.insert(module) {
            if let Err(error) = crate::air::visible_references::validate_closure(module, &authored)
            {
                errors.push(error);
            }
        }
    }
}

/// The entry module followed by every distinct linked module whose functions it can execute:
/// direct visible-function references, visible-function table entries, and linked intersection
/// functions.
fn executed_modules<'a>(
    entry: &'a SourceRow,
    linked: &'a ResolvedLinkedFunctions,
) -> impl Iterator<Item = (loop_free::ModuleRole, loop_free::ModuleSource<'a>)> {
    use crate::library_module::LibraryModuleRow;
    use loop_free::{ModuleRole, ModuleSource};
    let library = |role, module: &'a LibraryModuleRow| {
        (
            role,
            ModuleSource {
                label: &module.label,
                air_ll: &module.air_ll,
                // A public linked module retains no blob: its text is assembled.
                blob_b64: (!module.is_public()).then_some(module.blob_b64.as_str()),
            },
        )
    };
    let references = linked
        .references
        .iter()
        .map(move |reference| library(ModuleRole::Linked, &reference.module));
    let visible = linked.visible.iter().flat_map(move |table| {
        table
            .entries
            .iter()
            .map(move |entry| library(ModuleRole::Linked, &entry.module))
    });
    let intersection = linked.intersection.iter().flat_map(move |table| {
        table.entries.iter().filter_map(move |entry| match entry {
            crate::library_module::ResolvedIntersectionFunctionEntry::Linked(entry) => {
                Some(library(ModuleRole::Intersection, &entry.module))
            }
            crate::library_module::ResolvedIntersectionFunctionEntry::OpaqueTriangle { .. } => None,
        })
    });
    let entry = ModuleSource {
        label: &entry.label,
        air_ll: &entry.air_ll,
        blob_b64: entry.blob_b64.as_deref(),
    };
    // A module linked in two roles keeps each: its functions are then targets of both kinds.
    let mut seen = HashSet::new();
    std::iter::once((ModuleRole::Entry, entry))
        .chain(references)
        .chain(visible)
        .chain(intersection)
        .filter(move |(role, module)| seen.insert((*role, module.air_ll, module.blob_b64)))
}

/// The author's `execution_safety` is a declaration the mechanical classifier verifies.
fn validate_execution_safety(safety: ExecutionSafety, class: &LoopClass, errors: &mut Vec<String>) {
    if let (ExecutionSafety::LoopFree, LoopClass::Cyclic(evidence)) = (safety, class) {
        errors.push(format!(
            "execution_safety=loop_free, but the AIR is not mechanically loop-free ({evidence}); \
             use execution_safety=authored_bounded and identify the finite input/function-constant \
             bound in rationale"
        ));
    }
}

/// The case entry's declared interface, for the case's stage, entry and function constants.
fn reflect(air_ll: &str, case: &AuthoredCase) -> Result<ShaderInterface, String> {
    let function_constants = crate::literal::function_constants(case)?
        .into_iter()
        .map(|constant| (constant.index, constant.bytes))
        .collect::<Vec<_>>();
    crate::air::reflect::inspect(air_ll, case.stage, &case.entry, &function_constants)
}

/// The partners of an object or mesh case's mesh draw ([`crate::case::MeshDraw`]), checked
/// against the case's own module `module_ll` (whose entry's reflection is `reflection`): each is an
/// entry of the module at the stage the draw names it for, loop-free (no executor bounds a
/// partner's loops), reflected under the case's function constants, and binding no resource (the
/// case authors only its own entry's). The pipeline's stages agree as Metal links them: each
/// threadgroup within its function's `[[max_total_threads_per_threadgroup]]`, a mesh function that
/// reads a payload under an object function whose payload holds as many bytes, and every user input
/// of the fragment function a field the mesh function writes under the same user key. Returns the
/// render-target locations the fragment partner writes: a render-target output selects one.
fn validate_mesh_partners(
    case: &AuthoredCase,
    module_ll: &str,
    reflection: &ShaderInterface,
    errors: &mut Vec<String>,
) -> HashSet<u32> {
    use crate::air::reflect::{MeshFieldRole, MeshStageInterface};
    let Some(draw) = &case.mesh_draw else {
        return HashSet::new();
    };
    let entries = match crate::air::module_entries(module_ll) {
        Ok(entries) => entries,
        Err(error) => {
            errors.push(format!("the module's entries do not read: {error}"));
            return HashSet::new();
        }
    };
    let constants = match crate::literal::function_constants(case) {
        Ok(constants) => constants
            .into_iter()
            .map(|constant| (constant.index, constant.bytes))
            .collect::<Vec<_>>(),
        Err(error) => {
            errors.push(error);
            return HashSet::new();
        }
    };
    let mut partners = BTreeMap::new();
    let mut fragment_text = None;
    for (stage, name) in draw.partners() {
        let field = format!("mesh_draw.{}", stage.entry_list());
        if !entries
            .iter()
            .any(|entry| entry.name == name && entry.stage == stage.metadata_label())
        {
            errors.push(format!(
                "{field} names {name:?}, which the module declares as no {} function",
                stage.entry_list()
            ));
            continue;
        }
        let text = match crate::air::entry_text(module_ll, name) {
            Ok(text) => text.into_owned(),
            Err(error) => {
                errors.push(format!("{field} {name:?} has no projection: {error}"));
                continue;
            }
        };
        if has_cfg_cycle(&text) {
            errors.push(format!(
                "{field} {name:?} has a loop: a partner must be loop-free, since no executor \
                 bounds a partner's loops"
            ));
        }
        let partner = match crate::air::reflect::inspect(&text, stage, name, &constants) {
            Ok(partner) => partner,
            Err(error) => {
                errors.push(format!("{field} {name:?} does not reflect: {error}"));
                continue;
            }
        };
        if !partner.bindings.is_empty() || !partner.argument_buffer_fields.is_empty() {
            errors.push(format!(
                "{field} {name:?} binds resources: a partner binds none, since a case authors \
                 only its own entry's"
            ));
        }
        if stage == Stage::Fragment {
            fragment_text = Some(text);
        }
        partners.insert(stage, partner);
    }
    let own = |stage: Stage| -> Option<&MeshStageInterface> {
        if stage == case.stage {
            reflection.mesh_stage.as_ref()
        } else {
            partners
                .get(&stage)
                .and_then(|partner| partner.mesh_stage.as_ref())
        }
    };
    for (stage, threads) in [
        (Stage::Object, draw.threads_per_object_threadgroup),
        (Stage::Mesh, draw.threads_per_mesh_threadgroup),
    ] {
        let Some(max) = own(stage).and_then(|own| own.max_total_threads) else {
            continue;
        };
        let total = threads.iter().map(|axis| u64::from(*axis)).product::<u64>();
        if total > u64::from(max) {
            errors.push(format!(
                "mesh_draw's {} threadgroup {threads:?} holds {total} threads, past the {} \
                 function's max_total_threads_per_threadgroup {max}",
                stage.entry_list(),
                stage.entry_list()
            ));
        }
    }
    let payload = |stage| own(stage).and_then(|own| own.payload.as_ref());
    if let (Some(_), Some(read)) = (own(Stage::Object), payload(Stage::Mesh)) {
        match payload(Stage::Object) {
            None => errors.push(
                "the mesh function reads a payload, and the object function declares none".into(),
            ),
            Some(written) if read.size.unwrap_or(0) > written.size.unwrap_or(0) => {
                errors.push(format!(
                    "the mesh function reads a {}-byte payload, and the object function's holds \
                     {} bytes",
                    read.size.unwrap_or(0),
                    written.size.unwrap_or(0)
                ))
            }
            Some(_) => {}
        }
    }
    if let (Some(text), Some(outputs)) = (
        &fragment_text,
        own(Stage::Mesh).and_then(|own| own.outputs.as_ref()),
    ) {
        let keys = outputs
            .vertex
            .iter()
            .chain(&outputs.primitive)
            .filter_map(|field| match &field.role {
                MeshFieldRole::VertexData { key, .. }
                | MeshFieldRole::PrimitiveData { key, .. } => Some(key.as_str()),
                _ => None,
            })
            .collect::<HashSet<_>>();
        let (fragment, _) = crate::air::meta::parse_air_fragment_meta(text);
        let mut inputs = fragment
            .map(|fragment| {
                fragment
                    .varying_user_semantics
                    .into_values()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        inputs.sort();
        for input in inputs {
            if !keys.contains(input.as_str()) {
                errors.push(format!(
                    "the fragment function reads {input:?}, which the mesh function writes no \
                     field of"
                ));
            }
        }
    }
    partners
        .get(&Stage::Fragment)
        .map(|fragment| {
            fragment
                .render_targets
                .iter()
                .map(|target| target.location)
                .collect()
        })
        .unwrap_or_default()
}

const OUTPUT_NOT_WRITABLE: &str =
    "selected output is nothing the AIR declares or performs a write to";

/// The families the checker reads from the declared reflection: tessellation and the kernel and
/// fragment imageblocks. (The vertex observation is checked where its companion is derived from the
/// AIR, `companion::derive`.) Returns the writable imageblock outputs for the output rule.
fn validate_reflected_families(
    case: &AuthoredCase,
    reflection: &ShaderInterface,
    errors: &mut Vec<String>,
) -> conformance::ImageblockOutputs {
    use crate::air::reflect::ResourceAccess;
    validate_tessellation(case, reflection, errors);
    match (
        reflection.imageblock_layouts.is_empty(),
        case.imageblock.as_ref(),
    ) {
        (true, Some(_)) => errors.push(
            "manifest declares an imageblock, but reflection has no imageblock layout"
                .into(),
        ),
        (false, None) => errors.push(format!(
            "reflection requires {} imageblock layout(s), but the manifest declares no imageblock dimensions",
            reflection.imageblock_layouts.len()
        )),
        _ => {}
    }
    validate_implicit_imageblock_attachments(case, reflection, errors);
    validate_fragment_imageblock(case, reflection, errors);

    let writable = |access: ResourceAccess| {
        !matches!(
            access,
            ResourceAccess::Unused | ResourceAccess::ReadOnly | ResourceAccess::Sampled
        )
    };
    let attachments = reflection
        .implicit_imageblock_attachments
        .iter()
        .filter(|attachment| writable(attachment.access))
        .map(|attachment| attachment.attachment)
        .collect::<HashSet<_>>();
    let fragment_members = reflection
        .fragment_imageblock
        .iter()
        .flat_map(|imageblock| imageblock.members.iter())
        .filter(|member| member.binding.is_some() && writable(member.access))
        .map(|member| member.semantic.clone())
        .collect::<HashSet<_>>();
    conformance::ImageblockOutputs {
        writes: !attachments.is_empty() || !fragment_members.is_empty(),
        attachments,
        fragment_members,
    }
}

fn validate_tessellation(
    case: &AuthoredCase,
    reflection: &ShaderInterface,
    errors: &mut Vec<String>,
) {
    let (Some(authored), Some(interface)) = (&case.tessellation, &reflection.tessellation) else {
        if case.tessellation.is_some() != reflection.tessellation.is_some() {
            errors.push("manifest and AIR tessellation execution do not match".into());
        }
        return;
    };
    let (edge_count, inside_count) = match interface.domain {
        crate::air::reflect::PatchDomain::Triangle => (3, 1),
        crate::air::reflect::PatchDomain::Quad => (4, 2),
        crate::air::reflect::PatchDomain::Isoline => (2, 0),
    };
    for (patch, factors) in authored.factors.iter().enumerate() {
        if factors.edge_f16.len() != edge_count || factors.inside_f16.len() != inside_count {
            errors.push(format!(
                "tessellation patch {patch} requires {edge_count} edge and {inside_count} inside factors for {:?}",
                interface.domain
            ));
        }
    }
    // An out-of-bounds target's buffer holds fewer elements than the draw fetches, by design; its
    // reach past it covers the rest (`check::conformance`).
    let targets = case
        .out_of_bounds
        .iter()
        .flat_map(|declaration| &declaration.targets)
        .filter_map(|target| match target.resource {
            crate::case::out_of_bounds::OutOfBoundsResource::AttributeBuffer { location } => {
                Some(location)
            }
            _ => None,
        })
        .collect::<HashSet<_>>();
    validate_tessellation_attributes(
        "control-point",
        &authored.control_points,
        &interface.control_point_attributes,
        (
            authored.factors.len(),
            interface.control_point_count as usize,
        ),
        &targets,
        errors,
    );
    validate_tessellation_attributes(
        "patch",
        &authored.patch_inputs,
        &interface.patch_attributes,
        (authored.factors.len(), 1),
        &targets,
        errors,
    );
}

/// Each authored tessellation input of `label` against its reflected attribute: the locations
/// agree, each format reads its AIR type, and each buffer holds the `(patches, records per
/// patch)` records the draw fetches unless its location is one of `targets`.
fn validate_tessellation_attributes(
    label: &str,
    authored: &[crate::case::AttributeInput],
    reflected: &[crate::air::reflect::TessellationAttribute],
    (patch_count, records_per_patch): (usize, usize),
    targets: &HashSet<u32>,
    errors: &mut Vec<String>,
) {
    let authored_locations = authored
        .iter()
        .map(|input| input.location)
        .collect::<HashSet<_>>();
    let reflected_locations = reflected
        .iter()
        .map(|input| input.location)
        .collect::<HashSet<_>>();
    report_set_difference(
        &format!("tessellation {label} input"),
        &reflected_locations,
        &authored_locations,
        errors,
    );
    for input in authored {
        let expected = reflected
            .iter()
            .find(|reflected| reflected.location == input.location)
            .and_then(|reflected| reflected.type_name.as_deref());
        if let Some(expected) = expected {
            if input.format.air_type_name() != expected {
                errors.push(format!(
                    "tessellation {label} input {} format {:?} is incompatible with AIR type {expected}",
                    input.location, input.format
                ));
            }
        }
        if targets.contains(&input.location) {
            // An out-of-bounds read of a target is judged by the format's zero, which holds only
            // when the format has the lanes the shader reads, so no fetch fills a component: its
            // AIR type must be known.
            if expected.is_none() {
                errors.push(format!(
                    "out_of_bounds target attribute buffer {}: the AIR states no type for the \
                     tessellation {label} input, so its format's lanes cannot be checked against \
                     the shader's",
                    input.location
                ));
            }
            continue;
        }
        let required = patch_count
            .checked_mul(records_per_patch)
            .and_then(|records| records.checked_mul(input.stride as usize));
        if let (Some(required), Ok(bytes)) = (
            required,
            base64::engine::general_purpose::STANDARD.decode(&input.bytes_b64),
        ) {
            if bytes.len() < required {
                errors.push(format!(
                    "tessellation {label} input {} has {} bytes, execution requires at least {required}",
                    input.location,
                    bytes.len()
                ));
            }
        }
    }
}

fn validate_implicit_imageblock_attachments(
    case: &AuthoredCase,
    reflection: &ShaderInterface,
    errors: &mut Vec<String>,
) {
    let has_implicit_attachments = !reflection.implicit_imageblock_attachments.is_empty();
    match (
        has_implicit_attachments,
        case.imageblock
            .as_ref()
            .and_then(|imageblock| imageblock.implicit_coverage),
    ) {
        (true, None) => errors.push(
            "implicit imageblock attachments require an authored imageblock.implicit_coverage"
                .into(),
        ),
        (false, Some(_)) => errors.push(
            "imageblock.implicit_coverage is valid only for an implicit imageblock layout".into(),
        ),
        _ => {}
    }
    let reflected = reflection
        .implicit_imageblock_attachments
        .iter()
        .map(|attachment| attachment.attachment)
        .collect::<HashSet<_>>();
    let manifest = case
        .render_targets
        .iter()
        .map(|target| target.index)
        .collect::<HashSet<_>>();
    if case.stage == Stage::Kernel {
        report_set_difference(
            "implicit imageblock attachment",
            &reflected,
            &manifest,
            errors,
        );
    }
    if case.imageblock.is_none() {
        return;
    }
    // A tile dispatch covers the render area, the dispatch's grid, in tiles of the imageblock's
    // dimensions (the Metal oracle's `renderTargetWidth`/`Height` and `tileWidth`/`Height`), so
    // every attachment is the render area's size.
    let area = case.dispatch.as_ref().map(|dispatch| dispatch.grid);
    for attachment in &reflection.implicit_imageblock_attachments {
        let Some(target) = case
            .render_targets
            .iter()
            .find(|target| target.index == attachment.attachment)
        else {
            continue;
        };
        if area.is_some_and(|[w, h, d]| d != 1 || target.dimensions != [w, h]) {
            errors.push(format!(
                "implicit imageblock attachment {} dimensions {:?} must equal the dispatch grid {:?}, the render area",
                attachment.attachment,
                target.dimensions,
                area.unwrap_or_default()
            ));
        }
        let expected = match attachment.format {
            crate::air::reflect::TextureFormat::R16f => TextureFormat::R16Float,
            crate::air::reflect::TextureFormat::Rg16f => TextureFormat::Rg16Float,
            crate::air::reflect::TextureFormat::Rg32f => TextureFormat::Rg32Float,
            crate::air::reflect::TextureFormat::R32f => TextureFormat::R32Float,
            crate::air::reflect::TextureFormat::R32ui => TextureFormat::R32Uint,
            crate::air::reflect::TextureFormat::Rgba16f => TextureFormat::Rgba16Float,
            crate::air::reflect::TextureFormat::Rgba32f => TextureFormat::Rgba32Float,
            other => {
                errors.push(format!(
                    "implicit imageblock attachment {} has unsupported reflected format {other:?}",
                    attachment.attachment
                ));
                continue;
            }
        };
        if target.format != expected {
            errors.push(format!(
                "implicit imageblock attachment {} requires format {expected:?}, got {:?}",
                attachment.attachment, target.format
            ));
        }
    }
}

fn validate_fragment_imageblock(
    case: &AuthoredCase,
    reflection: &ShaderInterface,
    errors: &mut Vec<String>,
) {
    let authored = case.fragment_imageblock.as_ref();
    let reflected = reflection.fragment_imageblock.as_ref();
    match (authored, reflected) {
        (None, None) => return,
        (Some(_), None) => {
            errors.push(
                "manifest declares fragment_imageblock, but reflection has no custom fragment imageblock"
                    .into(),
            );
            return;
        }
        (None, Some(_)) => {
            errors.push(
                "reflection requires a custom fragment imageblock, but the manifest declares none"
                    .into(),
            );
            return;
        }
        (Some(_), Some(_)) => {}
    }
    let authored = authored.expect("matched above");
    let reflected = reflected.expect("matched above");
    let attachment_dimensions = case
        .render_targets
        .first()
        .map(|target| target.dimensions)
        .or_else(|| case.depth_stencil.as_ref().map(|target| target.dimensions));
    if attachment_dimensions.is_some_and(|dimensions| dimensions != authored.dimensions) {
        errors.push(format!(
            "fragment imageblock dimensions {:?} must equal graphics attachment dimensions {:?}",
            authored.dimensions,
            attachment_dimensions.expect("checked above")
        ));
    }
    errors.extend(caps::fragment_imageblock_cap_errors(case, reflected));
    for member in reflected
        .members
        .iter()
        .filter(|member| member.binding.is_some())
    {
        let Some(resource) = authored
            .members
            .iter()
            .find(|resource| resource.semantic == member.semantic)
        else {
            errors.push(format!(
                "reflected fragment imageblock member {} has no authored plane",
                member.semantic
            ));
            continue;
        };
        let expected_role = match member.access {
            crate::air::reflect::ResourceAccess::ReadOnly => ResourceRole::Input,
            crate::air::reflect::ResourceAccess::WriteOnly => ResourceRole::Output,
            crate::air::reflect::ResourceAccess::ReadWrite => ResourceRole::InOut,
            crate::air::reflect::ResourceAccess::Unused => continue,
            other => {
                errors.push(format!(
                    "fragment imageblock member {} has invalid reflected access {other:?}",
                    member.semantic
                ));
                continue;
            }
        };
        if resource.role != expected_role {
            errors.push(format!(
                "fragment imageblock member {} role {:?} must match reflected access {:?}",
                member.semantic, resource.role, member.access
            ));
        }
        let reflected_format =
            crate::case::FragmentImageblockFormat::from_air_type(&member.type_name, member.size);
        if reflected_format.is_none() {
            errors.push(format!(
                "fragment imageblock member {} has unsupported reflected type {} size {}",
                member.semantic, member.type_name, member.size
            ));
        } else if reflected_format != Some(resource.format) {
            errors.push(format!(
                "fragment imageblock member {} format {:?} must match reflected type {} size {}",
                member.semantic, resource.format, member.type_name, member.size
            ));
        }
    }
    for resource in &authored.members {
        if !reflected
            .members
            .iter()
            .any(|member| member.binding.is_some() && member.semantic == resource.semantic)
        {
            errors.push(format!(
                "authored fragment imageblock member {} is not an accessed reflected member",
                resource.semantic
            ));
        }
    }
}

fn report_set_difference<T: Copy + Ord + std::hash::Hash + std::fmt::Debug>(
    kind: &str,
    reflected: &HashSet<T>,
    manifest: &HashSet<T>,
    errors: &mut Vec<String>,
) {
    let mut missing = reflected.difference(manifest).copied().collect::<Vec<_>>();
    let mut extra = manifest.difference(reflected).copied().collect::<Vec<_>>();
    missing.sort_unstable();
    extra.sort_unstable();
    for binding in missing {
        errors.push(format!(
            "reflection requires {kind} {binding:?}, but the manifest does not declare it"
        ));
    }
    for binding in extra {
        errors.push(format!(
            "manifest declares {kind} {binding:?}, but reflection does not"
        ));
    }
}

#[cfg(test)]
mod resource_contract_tests;

#[cfg(test)]
mod tests;
