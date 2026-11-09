use crate::case::AuthoredCase;
use crate::check::admission::{MetalGap, NeedsMetal};
use crate::coverage::ExecutedCensus;
use crate::executor_contract::out_of_bounds::OutOfBoundsRun;
use crate::hash::sha256_bytes;
use crate::observation_contract::ComparisonClass;
use base64::Engine as _;
use metal2vulkan_runtime::Degradation;
use serde::{Deserialize, Serialize};

/// The product's source fingerprint (`validation/product_fingerprint.rs`: the translator's sources,
/// then the runtime crate's, and the dev-tools crate's, which assemble the public cases' bitcode).
/// Every candidate row's translator fingerprint is derived from it
/// (`candidate::engine::engine_fingerprint`), so a row is current only for this product.
pub const TRANSLATOR_FINGERPRINT: &str = env!("METAL2VULKAN_PRODUCT_FINGERPRINT");

/// `VK_LAYER_KHRONOS_profiles`, as an active tool's `layer` names it in a candidate environment's
/// `tools`.
pub const PROFILES_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_profiles";

/// What one row's executions of its case cost in time, as the run that recorded the row measured
/// them on its own device: every Metal row and every candidate row carries one, so a reader of a
/// stored row sees the cost without re-running anything.
///
/// Each execution is read by one [`crate::budgets::CreationClock`], the crate's clock for a device
/// call: its wall time less the time the calling thread waited for a CPU. The clock the
/// pipeline-creation ceiling is checked against is the clock a row records, so the two can never
/// disagree.
///
/// **A timing is not semantics and decides nothing.** No verdict, content hash, `input_sha256`,
/// translator fingerprint, executor ABI or census key reads it; the same case run twice on one
/// device records the same bytes and two different figures. So a row is the row it was before it
/// carried a figure, and two rows that differ only in their figures are one row —
/// [`CaseTiming`]'s equality says exactly that, because an `Eq` that let a duration separate two
/// rows would let a re-measurement read as new evidence.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseTiming {
    /// How many executions of the case the figures cover: [`crate::metal::QUALIFICATION_RUNS`] for
    /// a Metal row, one for a candidate row.
    pub runs: u32,
    /// What those executions took in all, in microseconds.
    pub total_micros: u64,
    /// The least one of them took, in microseconds: the figure the first execution's compiles do
    /// not reach.
    pub least_micros: u64,
}

impl CaseTiming {
    /// The timing of one execution that took `span`: what a candidate run and a refused attempt on
    /// an opened device each record. It cannot fail, so a caller that read a clock has a figure
    /// without an unreachable error path to carry.
    pub fn of_one(span: std::time::Duration) -> CaseTiming {
        let micros = u64::try_from(span.as_micros()).unwrap_or(u64::MAX);
        CaseTiming {
            runs: 1,
            total_micros: micros,
            least_micros: micros,
        }
    }

    /// The timing of `runs` executions that each took the time `took` holds, in the order they
    /// ran. Returns `None` for no executions, which no recorded row has.
    pub fn of(took: impl IntoIterator<Item = std::time::Duration>) -> Option<CaseTiming> {
        let micros =
            |span: std::time::Duration| u64::try_from(span.as_micros()).unwrap_or(u64::MAX);
        let mut timing: Option<CaseTiming> = None;
        for span in took {
            let span = micros(span);
            timing = Some(match timing {
                None => CaseTiming {
                    runs: 1,
                    total_micros: span,
                    least_micros: span,
                },
                Some(before) => CaseTiming {
                    runs: before.runs.saturating_add(1),
                    total_micros: before.total_micros.saturating_add(span),
                    least_micros: before.least_micros.min(span),
                },
            });
        }
        timing
    }
}

/// Every timing equals every other, including one of another number of runs: a figure is measured
/// noise, and [`CaseTiming`]'s documentation says why no comparison may read it.
impl PartialEq for CaseTiming {
    fn eq(&self, _: &CaseTiming) -> bool {
        true
    }
}

impl Eq for CaseTiming {}

/// The timing a test row carries: a row this build records carries a measurement, and a test that
/// builds one takes this fixed figure rather than reading a clock, so no test can turn a duration
/// into evidence. A test of what a row recorded before the figure existed passes `None` itself.
#[cfg(test)]
pub(crate) fn test_timing() -> CaseTiming {
    CaseTiming {
        runs: 1,
        total_micros: 1_234,
        least_micros: 1_234,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetalObservation {
    pub case_id: String,
    pub air_sha256: String,
    pub input_sha256: String,
    pub metal_output_sha256: String,
    pub output_b64: String,
    pub environment_id: String,
    pub environment: serde_json::Value,
    pub oracle_abi: String,
    /// What the row's executions on Metal cost in time ([`CaseTiming`]). Every row this build
    /// records carries one ([`crate::store::CorpusStore::upsert_metal`] refuses one that does
    /// not); a row recorded before the figure existed holds none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<CaseTiming>,
    pub status: MetalStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalStatus {
    Qualified,
    /// The reference device cannot run the case, by Metal's own answer: its pipeline creation
    /// failed naming a GPU family the device does not support. The row records no output, so no
    /// candidate compares against it; C3 and coverage count it apart, and it stands only with the
    /// user's sign-off.
    ReferenceGap(ReferenceGap),
}

/// What a [`MetalStatus::ReferenceGap`] row records: the Apple GPU family Metal's pipeline
/// creation named (`GPUFamily<N>`), which the device's `supportsFamily(MTLGPUFamilyApple<N>)`
/// denied, and Metal's message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceGap {
    pub apple_family: u32,
    pub message: String,
}

impl ReferenceGap {
    /// The family `message` (a pipeline creation's error) names as the one its shader needs:
    /// the number after Metal's `GPUFamily`, when exactly one family is named.
    pub fn named_family(message: &str) -> Option<u32> {
        let mut families = message.match_indices("GPUFamily").map(|(at, word)| {
            message[at + word.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse::<u32>()
                .ok()
        });
        match (families.next(), families.next()) {
            (Some(Some(family)), None) if family > 0 => Some(family),
            _ => None,
        }
    }

    /// A gap whose family is the one its message names.
    pub fn validate(&self) -> Result<(), String> {
        match Self::named_family(&self.message) {
            Some(family) if family == self.apple_family => Ok(()),
            _ => Err(format!(
                "a reference gap of Apple GPU family {} whose message names no such single \
                 family: {:?}",
                self.apple_family, self.message
            )),
        }
    }
}

/// The member of a candidate row's `environment` that holds the codes of the degrade notices its
/// runtime device delivered ([`Degradation::code`]): a run through `metal2vulkan_runtime`.
pub const RUNTIME_DEGRADES: &str = "runtime_degrades";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateObservation {
    pub case_id: String,
    pub air_sha256: String,
    pub input_sha256: String,
    pub golden_output_sha256: String,
    pub spv_sha256: String,
    pub translator_fingerprint: String,
    pub candidate_output_sha256: String,
    pub output_b64: String,
    pub backend: Backend,
    pub environment_id: String,
    pub environment: serde_json::Value,
    pub executor_abi: String,
    /// The class the status was decided under: `Exact` for the old engine, derived from the
    /// executed modules' precision notes and strategy census for the new one.
    pub comparison: ComparisonClass,
    /// What the engine's executed modules selected, and where they ran; absent for the old
    /// engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub census: Option<ExecutedCensus>,
    /// What the row's execution on its Vulkan device cost in time ([`CaseTiming`]), as the device
    /// worker that ran it measured. Every row this build records carries one
    /// ([`crate::store::CorpusStore::upsert_candidate`] refuses one that does not); a row recorded
    /// before the figure existed holds none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<CaseTiming>,
    pub status: CandidateStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Moltenvk,
    Vulkan,
}

impl Backend {
    pub fn directory(self) -> &'static str {
        match self {
            Self::Moltenvk => "moltenvk",
            Self::Vulkan => "vulkan",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Match,
    Mismatch,
}

impl MetalObservation {
    pub fn validate_content(&self) -> Result<(), String> {
        validate_hash("case_id", &self.case_id)?;
        validate_hash("air_sha256", &self.air_sha256)?;
        validate_hash("input_sha256", &self.input_sha256)?;
        validate_hash("metal_output_sha256", &self.metal_output_sha256)?;
        validate_slot_fields(&self.environment_id, &self.oracle_abi)?;
        if !self.environment.is_object() {
            return Err("Metal environment must be a JSON object".into());
        }
        let output = base64::engine::general_purpose::STANDARD
            .decode(&self.output_b64)
            .map_err(|error| format!("invalid Metal output_b64: {error}"))?;
        let computed = sha256_bytes(&output);
        if computed != self.metal_output_sha256 {
            return Err(format!(
                "Metal output hash mismatch: row={} computed={computed}",
                self.metal_output_sha256
            ));
        }
        if let MetalStatus::ReferenceGap(gap) = &self.status {
            gap.validate()?;
            if !output.is_empty() {
                return Err("a Metal reference gap records no output".into());
            }
        }
        Ok(())
    }

    /// The output this row observed: `None` for a reference gap, which records none.
    pub fn observed_output(&self) -> Option<&String> {
        match self.status {
            MetalStatus::Qualified => Some(&self.output_b64),
            MetalStatus::ReferenceGap(_) => None,
        }
    }

    /// The reference gap this row records for `case` under `oracle_abi`: a valid row of the case's
    /// AIR and inputs under that oracle whose status is [`MetalStatus::ReferenceGap`].
    pub fn reference_gap_for(
        &self,
        case: &AuthoredCase,
        oracle_abi: &str,
    ) -> Option<&ReferenceGap> {
        match &self.status {
            MetalStatus::ReferenceGap(gap)
                if self.validate_content().is_ok()
                    && self.case_id == case.case_id
                    && self.air_sha256 == case.air_sha256
                    && case
                        .computed_input_sha256()
                        .is_ok_and(|digest| digest == self.input_sha256)
                    && self.oracle_abi == oracle_abi =>
            {
                Some(gap)
            }
            _ => None,
        }
    }

    pub fn dependency_matches(&self, case: &AuthoredCase, oracle_abi: &str) -> bool {
        self.validate_content().is_ok()
            && self.case_id == case.case_id
            && self.air_sha256 == case.air_sha256
            && case
                .computed_input_sha256()
                .is_ok_and(|digest| digest == self.input_sha256)
            && self.oracle_abi == oracle_abi
            && self.status == MetalStatus::Qualified
    }

    /// `case`'s row among `rows` in `environment_id` when it is current evidence under
    /// `oracle_abi` ([`Self::dependency_matches`]), the only row a candidate compares against.
    /// Otherwise the case needs Metal qualification, and [`NeedsMetal`] says why: a row recorded by
    /// another oracle is never read as current evidence.
    pub fn current(
        rows: impl IntoIterator<Item = Self>,
        case: &AuthoredCase,
        environment_id: &str,
        oracle_abi: &str,
    ) -> Result<Self, NeedsMetal> {
        let needs = |gap| NeedsMetal {
            case_id: case.case_id.clone(),
            environment_id: environment_id.into(),
            gap,
        };
        let row = rows
            .into_iter()
            .find(|row| row.case_id == case.case_id && row.environment_id == environment_id)
            .ok_or_else(|| needs(MetalGap::Missing))?;
        if row.dependency_matches(case, oracle_abi) {
            Ok(row)
        } else if let Some(gap) = row.reference_gap_for(case, oracle_abi) {
            Err(needs(MetalGap::ReferenceGap {
                apple_family: gap.apple_family,
            }))
        } else if row.oracle_abi != oracle_abi {
            Err(needs(MetalGap::OracleAbi {
                recorded: row.oracle_abi,
                current: oracle_abi.into(),
            }))
        } else {
            Err(needs(MetalGap::OtherInputs))
        }
    }
}

pub struct CandidateDependencies<'a> {
    pub case: &'a AuthoredCase,
    pub metal: &'a MetalObservation,
    pub spv_sha256: &'a str,
    pub translator_fingerprint: &'a str,
    pub backend: Backend,
    pub environment_id: &'a str,
    pub executor_abi: &'a str,
}

impl CandidateObservation {
    pub fn validate_content(&self) -> Result<(), String> {
        for (field, value) in [
            ("case_id", self.case_id.as_str()),
            ("air_sha256", self.air_sha256.as_str()),
            ("input_sha256", self.input_sha256.as_str()),
            ("golden_output_sha256", self.golden_output_sha256.as_str()),
            ("spv_sha256", self.spv_sha256.as_str()),
            (
                "candidate_output_sha256",
                self.candidate_output_sha256.as_str(),
            ),
        ] {
            validate_hash(field, value)?;
        }
        validate_hash("translator_fingerprint", &self.translator_fingerprint)?;
        validate_slot_fields(&self.environment_id, &self.executor_abi)?;
        if !self.environment.is_object() {
            return Err("candidate environment must be a JSON object".into());
        }
        self.validate_restriction()?;
        self.validate_leg()?;
        let output = base64::engine::general_purpose::STANDARD
            .decode(&self.output_b64)
            .map_err(|error| format!("invalid candidate output_b64: {error}"))?;
        let computed = sha256_bytes(&output);
        if computed != self.candidate_output_sha256 {
            return Err(format!(
                "candidate output hash mismatch: row={} computed={computed}",
                self.candidate_output_sha256
            ));
        }
        let hashes_match = self.candidate_output_sha256 == self.golden_output_sha256;
        let run = self
            .census
            .as_ref()
            .and_then(|census| census.out_of_bounds.as_ref());
        // Equal bytes match under every class; differing bytes match only under a class that
        // tolerates a difference, or for a run whose out-of-bounds targets may read a value
        // Metal's row records as zero. Equal bytes are a mismatch only for a run that changed a
        // target's guard. `dependency_matches` re-decides each from the Metal bytes.
        let tolerated = self.comparison.tolerates_differences()
            || run.is_some_and(OutOfBoundsRun::tolerates_differences);
        let contradicts = match self.status {
            CandidateStatus::Match => !hashes_match && !tolerated,
            CandidateStatus::Mismatch => hashes_match && !run.is_some_and(OutOfBoundsRun::escapes),
        };
        if contradicts {
            return Err(format!(
                "candidate status {:?} contradicts candidate/golden output hashes under {:?}",
                self.status, self.comparison
            ));
        }
        Ok(())
    }

    /// The row's profiles-layer restriction is the one its run requested. A context the profiles
    /// layer restricts without a request (an environment that stacks the layer, such as
    /// `scripts/linux-gate --profile`) opens and runs, but its row would read as native, so it is
    /// never stored. The layer is active (`tools`) exactly when the environment records a
    /// restriction (`profiles_layer`), and the census names that profile (`restricted_to`).
    /// Rows from before the environment recorded its tools record neither.
    fn validate_restriction(&self) -> Result<(), String> {
        let active = self
            .environment
            .get("tools")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|tools| {
                tools.iter().any(|tool| {
                    tool.get("layer")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|layer| layer.as_bytes() == PROFILES_LAYER.to_bytes())
                })
            });
        let requested = self
            .environment
            .get("profiles_layer")
            .filter(|restriction| !restriction.is_null())
            .map(|restriction| {
                restriction
                    .get("profile")
                    .and_then(serde_json::Value::as_str)
                    .ok_or("the candidate environment's profiles_layer names no profile")
            })
            .transpose()?;
        let restricted_to = self
            .census
            .as_ref()
            .and_then(|census| census.restricted_to.as_deref());
        match (active, requested) {
            (true, None) => Err(format!(
                "{} was active on the device, but the run requested no restriction (an \
                 environment stacks the layer): the row would read as native, so it is not \
                 stored; pass --profiles-layer P to run restricted",
                PROFILES_LAYER.to_string_lossy()
            )),
            (false, Some(profile)) => Err(format!(
                "the candidate environment records a restriction to {profile}, but {} was not \
                 active",
                PROFILES_LAYER.to_string_lossy()
            )),
            _ if restricted_to != requested => Err(format!(
                "the candidate census is restricted to {restricted_to:?}, but the environment \
                 records {requested:?}"
            )),
            _ => Ok(()),
        }
    }

    /// The row's robustness leg is the one its environment records and the one its slot names,
    /// and its modules' containment agrees with it: a run that enabled no robustness feature
    /// compiled no robust encoding, and is stored in a robustness-off slot
    /// (`ENV@engine+robustness-off`), which holds no other row. Rows from before executors ran
    /// a leg record none, in the census or the environment, and are in no robustness-off slot.
    fn validate_leg(&self) -> Result<(), String> {
        let off_slot = crate::candidate::engine::is_robustness_off_slot(&self.environment_id);
        let recorded = self
            .environment
            .get("robustness")
            .filter(|robustness| !robustness.is_null())
            .map(|robustness| {
                robustness
                    .get("leg")
                    .cloned()
                    .and_then(|leg| serde_json::from_value::<crate::coverage::Leg>(leg).ok())
                    .ok_or("the candidate environment's robustness names no leg")
            })
            .transpose()?;
        let Some(census) = &self.census else {
            if off_slot {
                return Err(format!(
                    "the row is stored in robustness-off slot {:?}, but records no census, so no \
                     robustness leg",
                    self.environment_id
                ));
            }
            return Ok(());
        };
        if census.leg != recorded {
            return Err(format!(
                "the candidate census ran in robustness leg {:?}, but the environment records {:?}",
                census.leg, recorded
            ));
        }
        if off_slot != (census.leg == Some(crate::coverage::Leg::RobustnessOff)) {
            return Err(format!(
                "the candidate census ran in robustness leg {:?}, but the row is stored in slot \
                 {:?}",
                census.leg, self.environment_id
            ));
        }
        if !census.leg_holds() {
            return Err(
                "the candidate census ran with no robustness feature enabled, but records a \
                 module compiled under a robust encoding"
                    .into(),
            );
        }
        if let Some(artifact) = &census.artifact {
            artifact.validate()?;
        }
        Ok(())
    }

    /// Whether the recorded status is the verdict of the recorded class on `golden`, the Metal
    /// output, for an observation of `case`: for a case with out-of-bounds targets, the verdict
    /// of their containment property over the recorded census.
    pub fn status_holds(&self, case: &AuthoredCase, golden: &[u8]) -> bool {
        let Ok(output) = base64::engine::general_purpose::STANDARD.decode(&self.output_b64) else {
            return false;
        };
        let matches = crate::executor_contract::out_of_bounds::matches(
            case,
            &self.comparison,
            golden,
            &output,
            self.census
                .as_ref()
                .and_then(|census| census.out_of_bounds.as_ref()),
        );
        (self.status == CandidateStatus::Match) == matches
    }

    /// Whether the recorded class is the one the recorded census derives: a row without a census
    /// compared byte for byte.
    pub fn class_holds(&self) -> bool {
        self.comparison
            == self
                .census
                .as_ref()
                .map_or(ComparisonClass::Exact, ExecutedCensus::comparison_class)
    }

    pub fn dependency_matches(&self, dependencies: &CandidateDependencies<'_>) -> bool {
        self.validate_content().is_ok()
            && self.class_holds()
            && dependencies.metal.validate_content().is_ok()
            && self.case_id == dependencies.case.case_id
            && self.air_sha256 == dependencies.case.air_sha256
            && self.input_sha256 == dependencies.metal.input_sha256
            && self.golden_output_sha256 == dependencies.metal.metal_output_sha256
            && self.spv_sha256 == dependencies.spv_sha256
            && self.translator_fingerprint == dependencies.translator_fingerprint
            && self.backend == dependencies.backend
            && self.environment_id == dependencies.environment_id
            && self.executor_abi == dependencies.executor_abi
            && base64::engine::general_purpose::STANDARD
                .decode(&dependencies.metal.output_b64)
                .is_ok_and(|golden| self.status_holds(dependencies.case, &golden))
    }

    /// The degradations the row's run announced ([`RUNTIME_DEGRADES`]), read through the
    /// runtime's closed vocabulary ([`Degradation::from_code`]): a code it does not know (a
    /// retired kind's) is none of them.
    pub fn announced_degradations(&self) -> impl Iterator<Item = Degradation> + '_ {
        self.environment
            .get(RUNTIME_DEGRADES)
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|code| Degradation::from_code(code.as_str()?))
    }
}

pub(crate) fn validate_hash(field: &str, value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} must be 64 hexadecimal characters"));
    }
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(format!("{field} must use lowercase hexadecimal"));
    }
    Ok(())
}

fn validate_slot_fields(environment_id: &str, abi: &str) -> Result<(), String> {
    if environment_id.trim().is_empty() {
        return Err("environment_id must not be empty".into());
    }
    if abi.trim().is_empty() {
        return Err("executor/oracle ABI must not be empty".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{
        AuthoredCase, BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection,
        ResourceRole, Stage,
    };

    fn case() -> AuthoredCase {
        let mut case = AuthoredCase {
            air_sha256: "11".repeat(32),
            case_id: String::new(),
            name: "case".into(),
            entry: "main".into(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("AAAAAA==".into()),
                rasterization_rate_map: None,
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            command_buffers: vec![],
            pipeline_states: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            argument_buffer_samplers: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            varying_values: Vec::new(),
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
                stage_in_region: None,
            }),
            draw: None,
            tessellation: None,
            mesh_draw: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            fuel: None,
            out_of_bounds: None,
            rationale: None,
            authored_by: None,
            render_pass_rate_map: None,
            raster_sample_count: None,
            public_source: None,
            no_effect: None,
            creation_bound: None,
        };
        case.case_id = case.computed_case_id().unwrap();
        case
    }

    /// A candidate compares only against the case's row in the named environment that is current
    /// evidence; anything else names why the case needs Metal qualification.
    #[test]
    fn only_a_current_metal_row_is_evidence() {
        let case = case();
        let output = [1u8, 0, 0, 0];
        let row = |environment_id: &str, oracle_abi: &str| MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: sha256_bytes(&output),
            output_b64: "AQAAAA==".into(),
            environment_id: environment_id.into(),
            environment: serde_json::json!({"device": "test"}),
            oracle_abi: oracle_abi.into(),
            status: MetalStatus::Qualified,
        };
        let needs = |gap| NeedsMetal {
            case_id: case.case_id.clone(),
            environment_id: "metal-env".into(),
            gap,
        };
        let current = row("metal-env", "oracle-v36");
        assert_eq!(
            MetalObservation::current(
                [row("other-env", "oracle-v36"), current.clone()],
                &case,
                "metal-env",
                "oracle-v36"
            ),
            Ok(current)
        );
        assert_eq!(
            MetalObservation::current(
                [row("other-env", "oracle-v36")],
                &case,
                "metal-env",
                "oracle-v36"
            ),
            Err(needs(MetalGap::Missing))
        );
        assert_eq!(
            MetalObservation::current(
                [row("metal-env", "oracle-v32")],
                &case,
                "metal-env",
                "oracle-v36"
            ),
            Err(needs(MetalGap::OracleAbi {
                recorded: "oracle-v32".into(),
                current: "oracle-v36".into(),
            }))
        );
        let mut other_input = row("metal-env", "oracle-v36");
        other_input.input_sha256 = "22".repeat(32);
        let mut other_air = row("metal-env", "oracle-v36");
        other_air.air_sha256 = "33".repeat(32);
        let mut malformed = row("metal-env", "oracle-v36");
        malformed.output_b64 = "AgAAAA==".into();
        for row in [other_input, other_air, malformed] {
            assert_eq!(
                MetalObservation::current([row], &case, "metal-env", "oracle-v36"),
                Err(needs(MetalGap::OtherInputs))
            );
        }
        assert_eq!(
            needs(MetalGap::Missing).to_string(),
            format!(
                "needs_metal_qualification{{missing}}: case {} has no Metal observation in \
                 environment metal-env; qualify the case on Metal (corpus-metal) before a \
                 candidate runs it",
                case.case_id
            )
        );
    }

    /// A reference gap names exactly one Apple GPU family in Metal's message, records no output,
    /// and is never current evidence: a candidate is told the reference device cannot run the case.
    #[test]
    fn a_reference_gap_records_its_family_and_is_never_evidence() {
        let message =
            "simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later";
        assert_eq!(ReferenceGap::named_family(message), Some(10));
        for unnamed in [
            "no family here",
            "GPUFamily",
            "GPUFamily0 and later",
            "GPUFamilyApple9 and later",
            "GPUFamily9 or GPUFamily10",
            "GPUFamily99999999999",
        ] {
            assert_eq!(ReferenceGap::named_family(unnamed), None, "{unnamed}");
        }
        let case = case();
        let gap = ReferenceGap {
            apple_family: 10,
            message: message.into(),
        };
        let row = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: sha256_bytes(&[]),
            output_b64: String::new(),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({"device": "test"}),
            oracle_abi: "oracle-v47".into(),
            status: MetalStatus::ReferenceGap(gap.clone()),
        };
        assert_eq!(row.validate_content(), Ok(()));
        assert_eq!(row.observed_output(), None);
        assert_eq!(row.reference_gap_for(&case, "oracle-v47"), Some(&gap));
        assert_eq!(row.reference_gap_for(&case, "oracle-v48"), None);
        assert!(!row.dependency_matches(&case, "oracle-v47"));
        let needs =
            MetalObservation::current([row.clone()], &case, "metal-env", "oracle-v47").unwrap_err();
        assert_eq!(needs.gap, MetalGap::ReferenceGap { apple_family: 10 });
        assert!(
            needs.to_string().contains("lacks Apple GPU family 10"),
            "{needs}"
        );

        let mut other_family = row.clone();
        other_family.status = MetalStatus::ReferenceGap(ReferenceGap {
            apple_family: 9,
            ..gap.clone()
        });
        assert!(other_family.validate_content().is_err());
        let mut with_output = row.clone();
        with_output.output_b64 = "AQAAAA==".into();
        with_output.metal_output_sha256 = sha256_bytes(&[1, 0, 0, 0]);
        assert_eq!(
            with_output.validate_content(),
            Err("a Metal reference gap records no output".into())
        );
        let line = serde_json::to_string(&row.status).unwrap();
        assert_eq!(
            line,
            format!(r#"{{"reference_gap":{{"apple_family":10,"message":{message:?}}}}}"#)
        );
        assert_eq!(
            serde_json::to_string(&MetalStatus::Qualified).unwrap(),
            r#""qualified""#
        );
    }

    #[test]
    fn every_candidate_dependency_participates_in_reuse() {
        let case = case();
        let input_sha256 = case.computed_input_sha256().unwrap();
        let output_sha256 = sha256_bytes(&[1, 0, 0, 0]);
        let metal = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: input_sha256.clone(),
            metal_output_sha256: output_sha256.clone(),
            output_b64: "AQAAAA==".into(),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({"device": "test"}),
            oracle_abi: "oracle-v1".into(),
            status: MetalStatus::Qualified,
        };
        assert!(metal.dependency_matches(&case, "oracle-v1"));
        assert!(!metal.dependency_matches(&case, "oracle-v2"));

        let candidate = CandidateObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256,
            golden_output_sha256: metal.metal_output_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: "55".repeat(32),
            candidate_output_sha256: output_sha256,
            output_b64: metal.output_b64.clone(),
            backend: Backend::Vulkan,
            environment_id: "vk-env".into(),
            environment: serde_json::json!({"driver": "test"}),
            executor_abi: "executor-v1".into(),
            comparison: ComparisonClass::Exact,
            census: None,
            status: CandidateStatus::Match,
        };
        let mut missing = serde_json::to_value(&candidate).unwrap();
        missing
            .as_object_mut()
            .unwrap()
            .remove("translator_fingerprint");
        assert!(serde_json::from_value::<CandidateObservation>(missing).is_err());
        let mut empty = candidate.clone();
        empty.translator_fingerprint.clear();
        assert!(empty
            .validate_content()
            .unwrap_err()
            .contains("translator_fingerprint"));
        let spv_sha256 = "33".repeat(32);
        let dependencies = CandidateDependencies {
            case: &case,
            metal: &metal,
            spv_sha256: &spv_sha256,
            translator_fingerprint: &candidate.translator_fingerprint,
            backend: Backend::Vulkan,
            environment_id: "vk-env",
            executor_abi: "executor-v1",
        };
        assert!(candidate.dependency_matches(&dependencies));

        let changed_spv = "44".repeat(32);
        let changed = CandidateDependencies {
            spv_sha256: &changed_spv,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
        let changed_fingerprint = "66".repeat(32);
        let changed = CandidateDependencies {
            translator_fingerprint: &changed_fingerprint,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
        let changed = CandidateDependencies {
            backend: Backend::Moltenvk,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
        let changed = CandidateDependencies {
            environment_id: "new-driver",
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
        let changed = CandidateDependencies {
            executor_abi: "executor-v2",
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));

        let mut changed_metal = metal.clone();
        changed_metal.metal_output_sha256 = sha256_bytes(&[2, 0, 0, 0]);
        changed_metal.output_b64 = "AgAAAA==".into();
        let changed = CandidateDependencies {
            metal: &changed_metal,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));

        let mut changed_case = case.clone();
        changed_case.buffers[0].initial_bytes_b64 = Some("AQAAAA==".into());
        changed_case.case_id = changed_case.computed_case_id().unwrap();
        let changed = CandidateDependencies {
            case: &changed_case,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
    }

    #[test]
    fn observation_payload_hash_and_status_are_checked() {
        let case = case();
        let output_sha256 = sha256_bytes(&[1, 0, 0, 0]);
        let mut row = CandidateObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            golden_output_sha256: output_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: "55".repeat(32),
            candidate_output_sha256: output_sha256,
            output_b64: "AQAAAA==".into(),
            backend: Backend::Vulkan,
            environment_id: "vk-env".into(),
            environment: serde_json::json!({}),
            executor_abi: "executor-v1".into(),
            comparison: ComparisonClass::Exact,
            census: None,
            status: CandidateStatus::Match,
        };
        assert!(row.validate_content().is_ok());
        row.output_b64 = "AgAAAA==".into();
        assert!(row
            .validate_content()
            .unwrap_err()
            .contains("hash mismatch"));
    }

    /// A row run where the profiles layer restricted the device without a request (its context
    /// opens and runs) is never valid, so the store refuses it; a requested restriction is valid
    /// only as the census records it.
    #[test]
    fn a_row_under_an_unrequested_profiles_layer_is_never_valid() {
        let case = case();
        let output_sha256 = sha256_bytes(&[1, 0, 0, 0]);
        let row =
            |environment: serde_json::Value, restricted_to: Option<&str>| CandidateObservation {
                timing: Some(crate::observation::test_timing()),
                case_id: case.case_id.clone(),
                air_sha256: case.air_sha256.clone(),
                input_sha256: case.computed_input_sha256().unwrap(),
                golden_output_sha256: output_sha256.clone(),
                spv_sha256: "33".repeat(32),
                translator_fingerprint: "55".repeat(32),
                candidate_output_sha256: output_sha256.clone(),
                output_b64: "AQAAAA==".into(),
                backend: Backend::Vulkan,
                environment_id: "vk-env".into(),
                environment,
                executor_abi: "executor-v1".into(),
                comparison: ComparisonClass::Exact,
                census: Some(ExecutedCensus {
                    out_of_bounds: None,
                    rate_map_pass: None,
                    device: Some(crate::coverage::ExecutionClass::CpuIcd),
                    product: "77".repeat(32),
                    restricted_to: restricted_to.map(str::to_owned),
                    leg: None,
                    modules: Vec::new(),
                    artifact: None,
                }),
                status: CandidateStatus::Match,
            };
        let validation = serde_json::json!({"layer": "VK_LAYER_KHRONOS_validation"});
        let profiles = serde_json::json!({"layer": "VK_LAYER_KHRONOS_profiles"});
        let restriction = serde_json::json!({"profile": "vk12-floor"});

        // Native rows, with and without recorded tools.
        assert_eq!(row(serde_json::json!({}), None).validate_content(), Ok(()));
        let native = serde_json::json!({"tools": [validation], "profiles_layer": null});
        assert_eq!(row(native, None).validate_content(), Ok(()));
        // The layer active without a request, with or without the key.
        for unrequested in [
            serde_json::json!({"tools": [validation, profiles]}),
            serde_json::json!({"tools": [validation, profiles], "profiles_layer": null}),
        ] {
            let refused = row(unrequested, None).validate_content().unwrap_err();
            assert!(refused.contains("--profiles-layer"), "{refused}");
        }
        // A requested restriction, as recorded and as the census names it.
        let restricted = serde_json::json!({
            "tools": [validation, profiles],
            "profiles_layer": restriction,
        });
        assert_eq!(
            row(restricted.clone(), Some("vk12-floor")).validate_content(),
            Ok(())
        );
        assert!(row(restricted.clone(), None).validate_content().is_err());
        assert!(row(restricted, Some("roadmap-2022"))
            .validate_content()
            .is_err());
        let inactive = serde_json::json!({"tools": [validation], "profiles_layer": restriction});
        assert!(row(inactive, Some("vk12-floor"))
            .validate_content()
            .unwrap_err()
            .contains("not active"));
    }

    /// A row's robustness leg is the one its environment records, and a robustness-off row
    /// records no module compiled under a robust encoding, so the store never accepts a row whose
    /// census would close a requirement of the other leg.
    #[test]
    fn a_row_records_the_leg_its_environment_ran() {
        use crate::coverage::{BufferEncoding, Containment, ImageEncoding, Leg, ModuleCensus};
        let case = case();
        let output_sha256 = sha256_bytes(&[1, 0, 0, 0]);
        let module = |buffers| ModuleCensus {
            stage: Stage::Kernel,
            subgroup: Default::default(),
            strategies: Default::default(),
            precision: Default::default(),
            containment: Some(Containment {
                buffers,
                images: ImageEncoding::Unused,
            }),
        };
        // A row in `slot`, its environment recording `environment` and its census `leg`.
        let in_slot = |slot: &str, environment: serde_json::Value, leg: Option<Leg>, buffers| {
            CandidateObservation {
                timing: Some(crate::observation::test_timing()),
                case_id: case.case_id.clone(),
                air_sha256: case.air_sha256.clone(),
                input_sha256: case.computed_input_sha256().unwrap(),
                golden_output_sha256: output_sha256.clone(),
                spv_sha256: "33".repeat(32),
                translator_fingerprint: "55".repeat(32),
                candidate_output_sha256: output_sha256.clone(),
                output_b64: "AQAAAA==".into(),
                backend: Backend::Vulkan,
                environment_id: slot.into(),
                environment,
                executor_abi: "executor-v1".into(),
                comparison: ComparisonClass::Exact,
                census: Some(ExecutedCensus {
                    out_of_bounds: None,
                    rate_map_pass: None,
                    device: Some(crate::coverage::ExecutionClass::CpuIcd),
                    product: "77".repeat(32),
                    restricted_to: None,
                    leg,
                    modules: vec![module(buffers)],
                    artifact: None,
                }),
                status: CandidateStatus::Match,
            }
        };
        let stated_slot = "vk-env@engine";
        let off_slot = "vk-env@engine+robustness-off";
        // A row in the slot its census leg names.
        let row = |environment: serde_json::Value, leg: Option<Leg>, buffers| {
            let slot = match leg {
                Some(Leg::RobustnessOff) => off_slot,
                _ => stated_slot,
            };
            in_slot(slot, environment, leg, buffers)
        };
        let ran = |leg: &str| serde_json::json!({"robustness": {"leg": leg}});
        let robust = BufferEncoding::Robust;
        let checked = BufferEncoding::Checked;
        assert_eq!(
            row(ran("stated"), Some(Leg::Stated), robust).validate_content(),
            Ok(())
        );
        assert_eq!(
            row(ran("robustness_off"), Some(Leg::RobustnessOff), checked).validate_content(),
            Ok(())
        );
        // A row from before executors ran a leg records none anywhere.
        assert_eq!(
            row(serde_json::json!({}), None, robust).validate_content(),
            Ok(())
        );
        for (environment, leg) in [
            (ran("stated"), Some(Leg::RobustnessOff)),
            (ran("robustness_off"), Some(Leg::Stated)),
            (ran("robustness_off"), None),
            (serde_json::json!({}), Some(Leg::Stated)),
        ] {
            let refused = row(environment, leg, checked)
                .validate_content()
                .unwrap_err();
            assert!(refused.contains("robustness leg"), "{refused}");
        }
        assert!(row(serde_json::json!({"robustness": {}}), None, checked)
            .validate_content()
            .unwrap_err()
            .contains("names no leg"));
        assert!(row(ran("robustness_off"), Some(Leg::RobustnessOff), robust)
            .validate_content()
            .unwrap_err()
            .contains("robust encoding"));
        // The slot names the leg: a robustness-off run is never stored as a stated row (where
        // the coverage report would take it for one), nor any other row as a robustness-off one.
        for (slot, environment, leg) in [
            (stated_slot, ran("robustness_off"), Some(Leg::RobustnessOff)),
            ("vk-env", ran("robustness_off"), Some(Leg::RobustnessOff)),
            (off_slot, ran("stated"), Some(Leg::Stated)),
            (off_slot, serde_json::json!({}), None),
        ] {
            let refused = in_slot(slot, environment, leg, checked)
                .validate_content()
                .unwrap_err();
            assert!(refused.contains("is stored in slot"), "{refused}");
        }
        let mut no_census = row(serde_json::json!({}), None, checked);
        no_census.census = None;
        assert_eq!(no_census.validate_content(), Ok(()));
        no_census.environment_id = off_slot.into();
        assert!(no_census
            .validate_content()
            .unwrap_err()
            .contains("records no census"));
    }

    #[test]
    fn canonical_empty_observation_payload_is_hash_checked() {
        let case = case();
        let empty_sha256 = sha256_bytes(&[]);
        let metal = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: empty_sha256.clone(),
            output_b64: String::new(),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({}),
            oracle_abi: "oracle-v1".into(),
            status: MetalStatus::Qualified,
        };
        assert!(metal.validate_content().is_ok());

        let input_sha256 = case.computed_input_sha256().unwrap();
        let mut candidate = CandidateObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id,
            air_sha256: case.air_sha256,
            input_sha256,
            golden_output_sha256: empty_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: "55".repeat(32),
            candidate_output_sha256: empty_sha256,
            output_b64: String::new(),
            backend: Backend::Vulkan,
            environment_id: "vk-env".into(),
            environment: serde_json::json!({}),
            executor_abi: "executor-v1".into(),
            comparison: ComparisonClass::Exact,
            census: None,
            status: CandidateStatus::Match,
        };
        assert!(candidate.validate_content().is_ok());
        candidate.candidate_output_sha256 = "00".repeat(32);
        assert!(candidate
            .validate_content()
            .unwrap_err()
            .contains("hash mismatch"));
    }

    /// A status decided under a tolerating class holds only where the Metal bytes and the case's
    /// declared lanes say so, and only under the class the recorded census derives; under
    /// `Exact` a byte difference is never a match.
    #[test]
    fn a_tolerant_status_is_re_decided_from_the_metal_bytes() {
        use crate::case::{RenderTargetResource, TextureFormat};
        use metal2vulkan::pipeline::PrecisionNote;
        let mut case = case();
        case.render_targets = vec![RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba16Float,
            dimensions: [1, 1],
            initial_bytes_b64: "AAAAAAAAAAA=".into(),
        }];
        case.output = OutputSelection::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [1, 1],
        };
        case.case_id = case.computed_case_id().unwrap();
        let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
        // Four half lanes: a subnormal, then three ones; a candidate one half ulp away in the
        // second lane, and one two ulps away.
        let golden = [0x01, 0x00, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
        let near = [0x01, 0x00, 0x01, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
        let far = [0x01, 0x00, 0x02, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
        let metal = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: sha256_bytes(&golden),
            output_b64: b64(&golden),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({}),
            oracle_abi: "oracle-v1".into(),
            status: MetalStatus::Qualified,
        };
        // What the executed module records: a half result rounded twice (a `half` fma).
        let census = crate::coverage::ExecutedCensus {
            out_of_bounds: None,
            rate_map_pass: None,
            device: None,
            product: "p".into(),
            restricted_to: None,
            leg: None,
            modules: vec![crate::coverage::ModuleCensus {
                stage: Stage::Fragment,
                subgroup: Default::default(),
                strategies: ["arith.float".to_string()].into(),
                precision: [PrecisionNote::F16FmaDoubleRounding].into(),
                containment: None,
            }],
            artifact: None,
        };
        let one_ulp = census.comparison_class();
        assert_eq!(
            one_ulp,
            ComparisonClass::derive([PrecisionNote::F16FmaDoubleRounding], [])
        );
        let row = |output: &[u8], comparison: ComparisonClass, status| CandidateObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: metal.input_sha256.clone(),
            golden_output_sha256: metal.metal_output_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: "55".repeat(32),
            candidate_output_sha256: sha256_bytes(output),
            output_b64: b64(output),
            backend: Backend::Vulkan,
            environment_id: "vk-env".into(),
            environment: serde_json::json!({}),
            executor_abi: "executor-v1".into(),
            census: (comparison != ComparisonClass::Exact).then(|| census.clone()),
            comparison,
            status,
        };
        let spv_sha256 = "33".repeat(32);
        let translator_fingerprint = "55".repeat(32);
        let dependencies = CandidateDependencies {
            case: &case,
            metal: &metal,
            spv_sha256: &spv_sha256,
            translator_fingerprint: &translator_fingerprint,
            backend: Backend::Vulkan,
            environment_id: "vk-env",
            executor_abi: "executor-v1",
        };

        let accepted = row(&near, one_ulp.clone(), CandidateStatus::Match);
        assert!(accepted.validate_content().is_ok());
        assert!(accepted.class_holds());
        assert!(accepted.dependency_matches(&dependencies));
        // The class does not reach a two-ulp difference: recorded as a match, it does not hold.
        let claimed = row(&far, one_ulp.clone(), CandidateStatus::Match);
        assert!(claimed.validate_content().is_ok());
        assert!(!claimed.dependency_matches(&dependencies));
        assert!(
            row(&far, one_ulp.clone(), CandidateStatus::Mismatch).dependency_matches(&dependencies)
        );
        // A recorded class must be the one the recorded census derives: not without a census,
        // not another class, and not `Exact` over a census that notes a double rounding.
        let mut uncensused = accepted.clone();
        uncensused.census = None;
        assert!(!uncensused.class_holds());
        assert!(!uncensused.dependency_matches(&dependencies));
        let mut other = accepted.clone();
        other.comparison =
            ComparisonClass::derive([PrecisionNote::ReductionOrder], ["simd.reduce.fadd.half"]);
        assert!(!other.dependency_matches(&dependencies));
        let mut tighter = row(&golden, one_ulp, CandidateStatus::Match);
        assert!(tighter.dependency_matches(&dependencies));
        tighter.comparison = ComparisonClass::Exact;
        assert!(tighter.validate_content().is_ok());
        assert!(!tighter.dependency_matches(&dependencies));
        // Under `Exact` differing bytes contradict a match on their own.
        let exact = row(&near, ComparisonClass::Exact, CandidateStatus::Match);
        assert!(exact
            .validate_content()
            .unwrap_err()
            .contains("contradicts"));
        // A mismatch never has equal bytes.
        let equal = row(&golden, ComparisonClass::Exact, CandidateStatus::Mismatch);
        assert!(equal.validate_content().is_err());
    }

    /// A recorded timing reaches no digest, no fingerprint and no verdict.
    ///
    /// This is the guard that protects the corpus. The corpus is never deleted or rewritten and
    /// re-qualifying it on Metal is expensive, so a figure that changes on every run must not be
    /// able to make a stored row stale, nor to decide a verdict. Nothing in this crate derives a
    /// fingerprint *from* a row — a translator fingerprint and an executor ABI are handed to a row
    /// as strings — so what a timing could reach is a row's own content digests and the comparisons
    /// that read a row, and both are pinned here: the three rows differ only in their figures (and
    /// one has none at all), and every answer about them is the same answer.
    #[test]
    fn a_timing_reaches_no_digest_no_fingerprint_and_no_verdict() {
        let case = case();
        let input_sha256 = case.computed_input_sha256().unwrap();
        let output = [1u8, 0, 0, 0];
        let output_sha256 = sha256_bytes(&output);
        let metal = |timing: Option<CaseTiming>| MetalObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: input_sha256.clone(),
            metal_output_sha256: output_sha256.clone(),
            output_b64: "AQAAAA==".into(),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({"device": "test"}),
            oracle_abi: "oracle-v1".into(),
            timing,
            status: MetalStatus::Qualified,
        };
        let fast = CaseTiming {
            runs: 3,
            total_micros: 300,
            least_micros: 90,
        };
        let slow = CaseTiming {
            runs: 3,
            total_micros: 9_000_000,
            least_micros: 2_900_000,
        };
        // The figures do differ where they are stored, so nothing below holds vacuously.
        let member =
            |row: &MetalObservation| serde_json::to_value(row).unwrap().get("timing").cloned();
        assert_ne!(member(&metal(Some(fast))), member(&metal(Some(slow))));
        assert_eq!(member(&metal(None)), None);

        let rows = [metal(None), metal(Some(fast)), metal(Some(slow))];
        for row in &rows {
            // The content digest is over the output bytes and nothing else: a row whose digest
            // took its timing in as well is refused, whatever its figure.
            assert_eq!(row.validate_content(), Ok(()));
            assert_eq!(row.metal_output_sha256, output_sha256);
            assert!(row.dependency_matches(&case, "oracle-v1"));
            assert!(!row.dependency_matches(&case, "oracle-v2"));
            assert_eq!(row.observed_output(), Some(&row.output_b64));
            assert_eq!(row.reference_gap_for(&case, "oracle-v1"), None);
            assert_eq!(
                MetalObservation::current([row.clone()], &case, "metal-env", "oracle-v1"),
                Ok(row.clone())
            );
        }
        let mut folded = metal(Some(fast));
        folded.metal_output_sha256 = sha256_bytes(&[output.as_slice(), b"300"].concat());
        assert!(folded
            .validate_content()
            .unwrap_err()
            .contains("output hash mismatch"));

        // Two rows that differ only in their figures are one row: no comparison can separate
        // them, so a re-measurement is never new evidence.
        assert_eq!(metal(Some(fast)), metal(Some(slow)));
        assert_eq!(fast, slow);

        // No load-bearing field moved: a timed row is the untimed row's bytes plus one member.
        let without_timing = |row: &MetalObservation| {
            let mut value = serde_json::to_value(row).unwrap();
            value.as_object_mut().unwrap().remove("timing");
            value
        };
        let untimed = serde_json::to_value(metal(None)).unwrap();
        assert_eq!(without_timing(&metal(Some(fast))), untimed);
        assert_eq!(without_timing(&metal(Some(slow))), untimed);
        // And an untimed row serializes to exactly the bytes it had: a row recorded before the
        // figure existed is not rewritten by reading and writing it again.
        assert_eq!(
            serde_json::to_string(&metal(None)).unwrap(),
            serde_json::to_string(
                &serde_json::from_str::<MetalObservation>(
                    &serde_json::to_string(&metal(None)).unwrap()
                )
                .unwrap()
            )
            .unwrap()
        );
        assert!(!serde_json::to_string(&metal(None))
            .unwrap()
            .contains("timing"));

        let candidate = |timing: Option<CaseTiming>| CandidateObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: input_sha256.clone(),
            golden_output_sha256: output_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: "55".repeat(32),
            candidate_output_sha256: output_sha256.clone(),
            output_b64: "AQAAAA==".into(),
            backend: Backend::Vulkan,
            environment_id: "vk-env".into(),
            environment: serde_json::json!({"driver": "test"}),
            executor_abi: "executor-v1".into(),
            comparison: ComparisonClass::Exact,
            census: None,
            timing,
            status: CandidateStatus::Match,
        };
        let golden = metal(Some(fast));
        let dependencies = CandidateDependencies {
            case: &case,
            metal: &golden,
            spv_sha256: &"33".repeat(32),
            translator_fingerprint: &"55".repeat(32),
            backend: Backend::Vulkan,
            environment_id: "vk-env",
            executor_abi: "executor-v1",
        };
        for row in [
            candidate(None),
            candidate(Some(fast)),
            candidate(Some(slow)),
        ] {
            assert_eq!(row.validate_content(), Ok(()));
            assert_eq!(row.candidate_output_sha256, output_sha256);
            assert_eq!(row.translator_fingerprint, "55".repeat(32));
            // The verdict a row is held to is the one its bytes decide.
            assert!(row.class_holds());
            assert!(row.dependency_matches(&dependencies));
            let mut mismatched = row.clone();
            mismatched.status = CandidateStatus::Mismatch;
            assert!(mismatched.validate_content().is_err());
        }
        assert_eq!(candidate(Some(fast)), candidate(Some(slow)));
        let candidate_without_timing = |row: &CandidateObservation| {
            let mut value = serde_json::to_value(row).unwrap();
            value.as_object_mut().unwrap().remove("timing");
            value
        };
        let untimed = serde_json::to_value(candidate(None)).unwrap();
        assert_eq!(candidate_without_timing(&candidate(Some(fast))), untimed);
        assert_eq!(candidate_without_timing(&candidate(Some(slow))), untimed);
    }

    /// Every execution's figure is in the timing a row records, and the least one names the
    /// cheapest of them: the first execution's compiles do not decide it.
    #[test]
    fn a_timing_covers_every_execution_it_was_read_from() {
        use std::time::Duration;
        assert_eq!(CaseTiming::of([]), None);
        let one = CaseTiming::of([Duration::from_micros(700)]).unwrap();
        // One execution, the form a candidate run and a refused attempt record. Compared field
        // by field: `assert_eq!` on two timings holds whatever they are, by this type's design.
        let alone = CaseTiming::of_one(Duration::from_micros(700));
        assert_eq!(
            (alone.runs, alone.total_micros, alone.least_micros),
            (one.runs, one.total_micros, one.least_micros)
        );
        assert_eq!(
            (one.runs, one.total_micros, one.least_micros),
            (1, 700, 700)
        );
        let three = CaseTiming::of([
            Duration::from_micros(5_000),
            Duration::from_micros(300),
            Duration::from_micros(412),
        ])
        .unwrap();
        assert_eq!(
            (three.runs, three.total_micros, three.least_micros),
            (3, 5_712, 300)
        );
    }
}
