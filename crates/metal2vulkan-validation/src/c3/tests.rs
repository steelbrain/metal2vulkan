//! The C3 report over synthetic cases and observations (the tests own their inputs), and the
//! committed list's schema.

use super::bound::msl::{Format, Math};
use super::bound::{Operand, Part, Words};
use super::*;
use crate::authored_compile::CurrentArtifacts;
use crate::c1::reviewed::ReviewedList as C1List;
use crate::candidate::engine::{engine_fingerprint_of, ENGINE_EXECUTOR_ABI};
use crate::case::{
    BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection, ResourceRole, Stage,
};
use crate::cutover::SignOff;
use crate::hash::{hex_encode, sha256_bytes};
use crate::observation::MetalStatus;
use crate::observation_contract::ComparisonClass;
use metal2vulkan::pipeline::PrecisionNote;
use reference::ReferenceEnvironment;

const MOLTENVK: &str = "mvk-host";
const LAVAPIPE: &str = "lavapipe-gate";
const RADV: &str = "radv-host";
/// The profile every synthetic engine row records.
const PROFILE: [u8; 32] = [7; 32];

/// The device and driver of the synthetic venue `environment`: MoltenVK's driver for the MoltenVK
/// venue, RADV's otherwise.
fn identity(environment: &str) -> DeviceIdentity {
    DeviceIdentity {
        device: format!("{environment} device"),
        vendor_id: 0x1002,
        device_id: 0x15bf,
        driver_id: if environment == MOLTENVK { 14 } else { 3 },
        driver_version: 109060100,
        driver_info: "Mesa 26.2.4".into(),
    }
}

fn venue(
    kind: VenueKind,
    environment: &str,
    population: Population,
    legs: &[Leg],
    profile: &str,
) -> Venue {
    Venue {
        kind,
        environment_id: environment.into(),
        profile: profile.into(),
        population,
        legs: legs.to_vec(),
        devices: vec![VenueDevice {
            profile: profile.into(),
            identity: identity(environment),
            metal_environment_id: None,
        }],
    }
}

/// MoltenVK (the old engine's baseline) and lavapipe (the public cases), each in its stated leg.
fn venues() -> Venues {
    Venues {
        venues: vec![
            venue(
                VenueKind::Moltenvk,
                MOLTENVK,
                Population::OldEngineMatched,
                &[Leg::Stated],
                "moltenvk-m3",
            ),
            venue(
                VenueKind::CpuIcd,
                LAVAPIPE,
                Population::Public,
                &[Leg::Stated],
                "gate-lavapipe",
            ),
        ],
    }
}

/// [`venues`] and a real GPU that must cover every Metal-qualified case in both legs.
fn three_venues() -> Venues {
    let mut venues = venues();
    venues.venues.push(venue(
        VenueKind::NativeGpu,
        RADV,
        Population::MetalQualified,
        &[Leg::Stated, Leg::RobustnessOff],
        "amd-rdna3-radv",
    ));
    venues
}

/// [`venues`]'s lavapipe venue alone: a test of what the list makes of one venue's rows, which no
/// other venue's unrecorded cases keep from holding.
fn lavapipe_alone() -> Venues {
    let mut venues = venues();
    venues
        .venues
        .retain(|venue| venue.environment_id == LAVAPIPE);
    venues
}

/// `environment`'s identity as a row records it, with the profile.
fn recorded_environment(environment: &str) -> serde_json::Value {
    let identity = identity(environment);
    serde_json::json!({
        "profile_fingerprint": hex_encode(PROFILE),
        "device": identity.device,
        "vendor_id": identity.vendor_id,
        "device_id": identity.device_id,
        "driver_id": identity.driver_id,
        "driver_version": identity.driver_version,
        "driver_info": identity.driver_info,
    })
}

/// A one-word kernel case over the synthetic AIR `tag`.
fn case(tag: u8, name: &str) -> AuthoredCase {
    let mut case = AuthoredCase {
        air_sha256: format!("{tag:02x}").repeat(32),
        case_id: String::new(),
        name: name.into(),
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

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// `case`'s current Metal row on the synthetic primary, reading `output`.
fn metal(case: &AuthoredCase, output: &[u8]) -> MetalObservation {
    metal_in(&reference::testing::primary(), case, output)
}

/// `case`'s current Metal row in `environment`, recording its identity, reading `output`.
fn metal_in(
    environment: &ReferenceEnvironment,
    case: &AuthoredCase,
    output: &[u8],
) -> MetalObservation {
    MetalObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: case.computed_input_sha256().unwrap(),
        metal_output_sha256: sha256_bytes(output),
        output_b64: b64(output),
        environment_id: environment.environment_id.clone(),
        environment: reference::testing::recorded(environment),
        oracle_abi: ORACLE_ABI.into(),
        status: MetalStatus::Qualified,
    }
}

/// The report with the synthetic primary as the only Metal reference.
fn report_of(
    input: &C3Input,
    list: &ReviewedList,
    c1: &C1List,
    venues: &Venues,
    keys: &VenueKeys,
) -> C3Report {
    report(
        input,
        list,
        c1,
        venues,
        keys,
        &reference::testing::references(),
        &reference::testing::NoFacts,
    )
}

/// A row of `case` in `environment_id` reading `output` against Metal's `golden`.
fn row(
    case: &AuthoredCase,
    backend: Backend,
    environment_id: String,
    output: &[u8],
    golden: &[u8],
) -> CandidateObservation {
    CandidateObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: case.computed_input_sha256().unwrap(),
        golden_output_sha256: sha256_bytes(golden),
        spv_sha256: "33".repeat(32),
        translator_fingerprint: String::new(),
        candidate_output_sha256: sha256_bytes(output),
        output_b64: b64(output),
        backend,
        environment_id,
        environment: serde_json::json!({}),
        executor_abi: String::new(),
        comparison: ComparisonClass::Exact,
        census: None,
        status: if output == golden {
            CandidateStatus::Match
        } else {
            CandidateStatus::Mismatch
        },
    }
}

/// A current engine row in the stated slot of `environment`, as the executor records it.
fn new_row(
    case: &AuthoredCase,
    backend: Backend,
    environment: &str,
    output: &[u8],
    golden: &[u8],
) -> CandidateObservation {
    let mut row = row(
        case,
        backend,
        engine_slot(environment, None, Leg::Stated),
        output,
        golden,
    );
    row.executor_abi = ENGINE_EXECUTOR_ABI.into();
    row.translator_fingerprint = engine_fingerprint_of(&PROFILE);
    row.environment = recorded_environment(environment);
    row
}

/// `row` as its executor records it in `leg`: the leg in its environment and its census, and the
/// leg's slot.
fn in_leg(mut row: CandidateObservation, leg: Leg) -> CandidateObservation {
    let environment = row.environment_id.trim_end_matches("@engine").to_string();
    row.environment_id = engine_slot(&environment, None, leg);
    row.environment["robustness"] = serde_json::json!({ "leg": leg });
    row.census = Some(crate::coverage::ExecutedCensus {
        device: None,
        product: crate::observation::TRANSLATOR_FINGERPRINT.into(),
        restricted_to: None,
        leg: Some(leg),
        modules: vec![],
        out_of_bounds: None,
        rate_map_pass: None,
        artifact: None,
    });
    row
}

/// An old-engine MoltenVK row in `environment`, recorded under an old executor ABI.
fn old_row(
    case: &AuthoredCase,
    environment: &str,
    output: &[u8],
    golden: &[u8],
) -> CandidateObservation {
    let mut row = row(case, Backend::Moltenvk, environment.into(), output, golden);
    row.executor_abi = "vulkan-retired-executor-v1".into();
    row.translator_fingerprint = "44".repeat(32);
    row.environment = serde_json::json!({"vulkan_target": {"version": [1, 2]}});
    row
}

/// `row` with one executed kernel module recording the precision notes `notes`.
fn noting(mut row: CandidateObservation, notes: &[PrecisionNote]) -> CandidateObservation {
    row.census = Some(crate::coverage::ExecutedCensus {
        device: None,
        product: crate::observation::TRANSLATOR_FINGERPRINT.into(),
        restricted_to: None,
        leg: None,
        modules: vec![crate::coverage::ModuleCensus {
            stage: Stage::Kernel,
            subgroup: crate::coverage::SubgroupShape::default(),
            strategies: BTreeSet::new(),
            precision: notes.iter().copied().collect(),
            containment: None,
        }],
        out_of_bounds: None,
        rate_map_pass: None,
        artifact: None,
    });
    row
}

/// The reviewed observation of `row`.
fn reviewed(row: &CandidateObservation, golden: &[u8]) -> ReviewedObservation {
    let output = decode(&row.output_b64).unwrap();
    ReviewedObservation {
        environment_id: row.environment_id.clone(),
        spv_sha256: row.spv_sha256.clone(),
        candidate_output_sha256: row.candidate_output_sha256.clone(),
        golden_output_sha256: row.golden_output_sha256.clone(),
        words: word_count(&output),
        differing_words: differing_words(golden, &output).unwrap(),
        values: "word 0: the device's reading against Metal's".into(),
    }
}

/// A lane over word 0 computing `computes` of the bit patterns `operands` (each one `f32`).
fn lane(computes: &str, math: Math, operands: &[u32]) -> BoundedLane {
    BoundedLane {
        words: Words {
            first: 0,
            count: 1,
            stride: 1,
        },
        result: Part::F32,
        math,
        computes: computes.into(),
        operands: operands
            .iter()
            .map(|bits| Operand::Bits {
                format: Format::F32,
                first: *bits,
                stride: 1,
            })
            .collect(),
    }
}

/// A signed-off Apple reading of word 0 as `exp2(0)` (MSL Table 8.1: within 4 ulp of 1.0), which
/// admits [`METAL`] and [`DEVICE`].
fn reading(case: &AuthoredCase, observations: Vec<ReviewedObservation>) -> Reading {
    Reading {
        case_id: case.case_id.clone(),
        name: case.name.clone(),
        cause: Cause::AppleReading,
        bound: "at most 4 ulp".into(),
        observations,
        lanes: vec![lane("exp2(a)", Math::Precise, &[0])],
        review: signed("a bounded reading"),
    }
}

/// A review the user signed off.
fn signed(note: &str) -> Review {
    Review {
        evidence: vec!["a probe".into()],
        note: note.into(),
        sign_off: SignOff::SignedOff {
            decision: "C3 and Apple readings".into(),
            date: "2026-10-02".into(),
        },
    }
}

fn old_cause(case: &AuthoredCase, cause: Cause) -> OldMismatch {
    OldMismatch {
        case_id: case.case_id.clone(),
        name: case.name.clone(),
        cause,
        review: signed("why the old row differs"),
    }
}

fn slot<'a>(report: &'a C3Report, venue: &str, leg: Leg) -> &'a SlotReport {
    report
        .slots
        .iter()
        .find(|slot| slot.venue == venue && slot.leg == leg)
        .unwrap()
}

fn verdict_of(slot: &SlotReport, case: &AuthoredCase) -> Verdict {
    slot.verdict_of(&case.case_id)
        .unwrap_or_else(|| panic!("{} was not judged in {}", case.case_id, slot.slot))
}

const METAL: [u8; 4] = [0, 0, 0x80, 0x3f];
const DEVICE: [u8; 4] = [1, 0, 0x80, 0x3f];

#[test]
fn a_listed_mismatch_at_its_candidate_is_reviewed_and_a_changed_candidate_is_not() {
    let bounded = case(1, "bounded");
    let mut input = C3Input {
        cases: vec![bounded.clone()],
        metals: vec![metal(&bounded, &METAL)],
        public_air: [bounded.air_sha256.clone()].into(),
        ..C3Input::default()
    };
    let lavapipe = new_row(&bounded, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    input.vulkan.push(lavapipe.clone());

    // Unlisted, the current mismatch is a mismatch, and C3 does not hold.
    let report = report_of(
        &input,
        &ReviewedList::default(),
        &C1List::default(),
        &lavapipe_alone(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(stated.population, 1);
    assert_eq!(verdict_of(stated, &bounded), Verdict::Mismatch);
    assert!(!stated.holds());

    // Listed at its candidate, output, and Metal output, it is reviewed; the reading holds.
    let list = ReviewedList {
        readings: vec![reading(&bounded, vec![reviewed(&lavapipe, &METAL)])],
        old_mismatches: vec![],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    let report = report_of(
        &input,
        &list,
        &C1List::default(),
        &lavapipe_alone(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(stated, &bounded), Verdict::Reviewed);
    assert!(stated.holds());
    assert_eq!(report.readings.len(), 1);
    assert_eq!(report.readings[0].state, ReadingState::Holds);
    assert!(report.readings[0].current);
    assert_eq!((report.signed_off(), report.awaiting_sign_off()), (1, 0));
    assert!(report.holds());
    // Until the user signs the entry off, the reading is reviewed but C3 does not hold.
    let mut awaiting = list.clone();
    awaiting.readings[0].review.sign_off = SignOff::AwaitingUser {
        decision: "C3 and Apple readings".into(),
    };
    let report = report_of(
        &input,
        &awaiting,
        &C1List::default(),
        &lavapipe_alone(),
        &VenueKeys::default(),
    );
    assert_eq!(
        verdict_of(slot(&report, LAVAPIPE, Leg::Stated), &bounded),
        Verdict::Reviewed
    );
    assert_eq!(report.awaiting_sign_off(), 1);
    assert_eq!(report.signed_off(), 0);
    assert_eq!(
        report.entries,
        [EntryRow {
            part: ListPart::Reading,
            case_id: bounded.case_id.clone(),
            name: bounded.name.clone(),
            cause: Cause::AppleReading,
            sign_off: awaiting.readings[0].review.sign_off.clone(),
        }]
    );
    assert!(!report.holds());

    // A changed candidate leaves the list: the row is a mismatch again and the entry is stale,
    // whether the SPIR-V or the reading changed.
    let changes: [fn(&mut CandidateObservation); 2] = [
        |row| row.spv_sha256 = "55".repeat(32),
        |row| {
            let output = [2, 0, 0x80, 0x3f];
            row.output_b64 = b64(&output);
            row.candidate_output_sha256 = sha256_bytes(&output);
        },
    ];
    for change in changes {
        let mut changed = input.clone();
        change(&mut changed.vulkan[0]);
        let report = report_of(
            &changed,
            &list,
            &C1List::default(),
            &lavapipe_alone(),
            &VenueKeys::default(),
        );
        let stated = slot(&report, LAVAPIPE, Leg::Stated);
        assert_eq!(verdict_of(stated, &bounded), Verdict::Mismatch);
        assert_eq!(report.readings[0].state, ReadingState::Changed);
        assert!(!report.holds());
    }

    // A changed Metal reading leaves the list too: the case was re-qualified, and its row now
    // compares the same candidate and output with another Metal output.
    let requalified_metal = [3, 0, 0x80, 0x3f];
    let mut requalified = input.clone();
    requalified.metals[0] = metal(&bounded, &requalified_metal);
    requalified.vulkan[0].golden_output_sha256 = sha256_bytes(&requalified_metal);
    let report = report_of(
        &requalified,
        &list,
        &C1List::default(),
        &lavapipe_alone(),
        &VenueKeys::default(),
    );
    assert_eq!(
        verdict_of(slot(&report, LAVAPIPE, Leg::Stated), &bounded),
        Verdict::Mismatch
    );
    assert_eq!(report.readings[0].state, ReadingState::Changed);

    // The list reviews a slot, not a case: the same candidate and reading in another venue's
    // slot is unreviewed until the list names that slot too.
    let mut elsewhere = input.clone();
    let mut moltenvk = lavapipe.clone();
    moltenvk.backend = Backend::Moltenvk;
    moltenvk.environment_id = engine_slot(MOLTENVK, None, Leg::Stated);
    moltenvk.environment = recorded_environment(MOLTENVK);
    elsewhere.moltenvk.push(moltenvk);
    let report = report_of(
        &elsewhere,
        &list,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    assert_eq!(
        verdict_of(slot(&report, MOLTENVK, Leg::Stated), &bounded),
        Verdict::Mismatch
    );
    assert_eq!(
        verdict_of(slot(&report, LAVAPIPE, Leg::Stated), &bounded),
        Verdict::Reviewed
    );
}

#[test]
fn stale_rows_are_not_evidence_and_their_reviewed_reading_still_names_them() {
    let bounded = case(1, "bounded");
    let mut stale = new_row(&bounded, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    // Recorded by another product build.
    stale.translator_fingerprint = engine_fingerprint_of(&[9; 32]);
    let input = C3Input {
        cases: vec![bounded.clone()],
        metals: vec![metal(&bounded, &METAL)],
        vulkan: vec![stale.clone()],
        public_air: [bounded.air_sha256.clone()].into(),
        ..C3Input::default()
    };
    let list = ReviewedList {
        readings: vec![reading(&bounded, vec![reviewed(&stale, &METAL)])],
        old_mismatches: vec![],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    let report = report_of(
        &input,
        &list,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    assert_eq!(
        verdict_of(slot(&report, LAVAPIPE, Leg::Stated), &bounded),
        Verdict::Stale
    );
    assert_eq!(report.readings[0].state, ReadingState::Holds);
    assert!(!report.readings[0].current);
    assert!(!report.holds());
}

/// A row whose case has no current Metal row is stale and counted as such (`stale_without_metal`):
/// a row depends on its Metal row's input and output hashes, and the oracle only decides which
/// Metal row is current. After an oracle change Metal requalifies the case; a Metal row with the
/// same output makes the row current with no device run, and one with another output leaves it
/// stale for that instead. A row stale for another reason, or a stale refusal, beside a case
/// without a current Metal row is not counted.
#[test]
fn a_row_waiting_on_metal_is_stale_without_metal_until_metal_requalifies() {
    use crate::check::admission::NotSupported;
    let awaiting = case(1, "awaiting");
    let rebuilt = case(2, "rebuilt");
    let refused = case(3, "refused");
    // Metal's rows under an older oracle: not current under this one.
    let older = |case: &AuthoredCase| MetalObservation {
        oracle_abi: "metal-literal-resources-v1".into(),
        ..metal(case, &METAL)
    };
    let mut other_build = new_row(&rebuilt, Backend::Vulkan, LAVAPIPE, &METAL, &METAL);
    other_build.translator_fingerprint = engine_fingerprint_of(&[9; 32]);
    let mut stale_refusal = refusal_of(
        &refused,
        LAVAPIPE,
        Leg::Stated,
        &NotSupported::NotEnabled(vec!["rayQuery".into()]),
    );
    stale_refusal.translator_fingerprint = engine_fingerprint_of(&[9; 32]);
    let input = C3Input {
        cases: vec![awaiting.clone(), rebuilt.clone(), refused.clone()],
        metals: vec![older(&awaiting), metal(&rebuilt, &METAL), older(&refused)],
        vulkan: vec![
            new_row(&awaiting, Backend::Vulkan, LAVAPIPE, &METAL, &METAL),
            other_build,
        ],
        refusals: vec![stale_refusal],
        public_air: [&awaiting, &rebuilt, &refused]
            .iter()
            .map(|case| case.air_sha256.clone())
            .collect(),
        ..C3Input::default()
    };
    let judge = |input: &C3Input| {
        report_of(
            input,
            &ReviewedList::default(),
            &C1List::default(),
            &venues(),
            &VenueKeys::default(),
        )
    };
    let report = judge(&input);
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(stated.population, 3);
    for case in [&awaiting, &rebuilt, &refused] {
        assert_eq!(verdict_of(stated, case), Verdict::Stale);
    }
    assert_eq!(stated.stale_without_metal, 1);
    assert_eq!(stated.stale_device, 0);
    assert_eq!(stated.missing_without_metal, 0);
    assert!(!stated.holds());

    // Metal requalifies the case under this oracle with the same output: the row is current again.
    let mut requalified = input.clone();
    requalified.metals[0] = metal(&awaiting, &METAL);
    let report = judge(&requalified);
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(stated, &awaiting), Verdict::Match);
    assert_eq!(stated.stale_without_metal, 0);

    // Requalified with another output, the row is stale for that, not for want of Metal.
    let mut moved = input.clone();
    moved.metals[0] = metal(&awaiting, &DEVICE);
    let report = judge(&moved);
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(stated, &awaiting), Verdict::Stale);
    assert_eq!(stated.stale_without_metal, 0);

    // A row whose case has no Metal row at all is counted alike.
    let mut unqualified = input;
    unqualified.metals.remove(0);
    let report = judge(&unqualified);
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(stated, &awaiting), Verdict::Stale);
    assert_eq!(stated.stale_without_metal, 1);
}

#[test]
fn moltenvk_must_cover_every_case_the_old_engine_matched_without_a_new_mismatch() {
    let covered = case(1, "covered");
    let regressed = case(2, "regressed");
    let refused = case(3, "refused");
    let old_mismatch = case(4, "old-mismatch");
    let new_only = case(5, "new-only");
    let unqualified = case(6, "unqualified");
    let mut input = C3Input {
        cases: vec![
            covered.clone(),
            regressed.clone(),
            refused.clone(),
            old_mismatch.clone(),
            new_only.clone(),
        ],
        ..C3Input::default()
    };
    for case in &input.cases.clone() {
        input.metals.push(metal(case, &METAL));
    }
    // A case the baseline matched that has no current Metal row: missing, and Metal must qualify
    // it before an executor can compare it.
    input.cases.push(unqualified.clone());
    input.moltenvk = vec![
        old_row(&unqualified, MOLTENVK, &METAL, &METAL),
        old_row(&covered, MOLTENVK, &METAL, &METAL),
        old_row(&regressed, MOLTENVK, &METAL, &METAL),
        old_row(&refused, MOLTENVK, &METAL, &METAL),
        old_row(&old_mismatch, MOLTENVK, &DEVICE, &METAL),
        // A probe build's row in another environment is not the baseline.
        old_row(&new_only, "probe-build", &METAL, &METAL),
        new_row(&covered, Backend::Moltenvk, MOLTENVK, &METAL, &METAL),
        new_row(&regressed, Backend::Moltenvk, MOLTENVK, &DEVICE, &METAL),
        new_row(&old_mismatch, Backend::Moltenvk, MOLTENVK, &DEVICE, &METAL),
        new_row(&new_only, Backend::Moltenvk, MOLTENVK, &DEVICE, &METAL),
    ];
    let list = ReviewedList {
        readings: vec![],
        old_mismatches: vec![old_cause(&old_mismatch, Cause::FastMathRegrouping)],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    let report = report_of(
        &input,
        &list,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, MOLTENVK, Leg::Stated);
    // The population is the four cases the baseline matched.
    assert_eq!(stated.population, 4);
    assert_eq!(verdict_of(stated, &unqualified), Verdict::Missing);
    assert_eq!(stated.count(Verdict::Missing), 2);
    assert_eq!(stated.missing_without_metal, 1);
    assert_eq!(verdict_of(stated, &covered), Verdict::Match);
    assert_eq!(verdict_of(stated, &regressed), Verdict::Mismatch);
    assert_eq!(verdict_of(stated, &refused), Verdict::Missing);
    // Judged though outside the population: an old mismatch still mismatching is no new
    // mismatch, and a case the baseline never ran is.
    assert_eq!(verdict_of(stated, &old_mismatch), Verdict::Mismatch);
    assert_eq!(verdict_of(stated, &new_only), Verdict::Mismatch);
    assert_eq!(stated.new_mismatches, 2);
    assert!(!stated.holds());
    // Lavapipe has no public case here, so nothing in its population, and no row: each of the
    // five cases a Metal row judges reads missing there until lavapipe records it; the unqualified
    // case is not judged.
    let lavapipe = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(lavapipe.population, 0);
    assert_eq!(lavapipe.count(Verdict::Missing), 5);
    assert_eq!(lavapipe.missing_without_metal, 0);
    assert_eq!(lavapipe.verdict_of(&unqualified.case_id), None);
    assert!(!lavapipe.holds());
}

/// A case the reference device cannot run reads `reference_gap` where it would read missing or
/// stale: never missing without Metal, and the slot holds, but C3 holds only once the list holds
/// the gap with the user's sign-off.
#[test]
fn a_reference_gap_is_counted_apart_and_held_to_the_users_sign_off() {
    let gap_case = case(1, "needs-family-10");
    let stale_gap = case(2, "stale-row-needs-family-10");
    let unqualified = case(3, "unqualified");
    let message = "simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later";
    let gap_row = |case: &AuthoredCase| MetalObservation {
        metal_output_sha256: sha256_bytes(&[]),
        output_b64: String::new(),
        status: MetalStatus::ReferenceGap(crate::observation::ReferenceGap {
            apple_family: 10,
            message: message.into(),
        }),
        ..metal(case, &METAL)
    };
    let input = C3Input {
        cases: vec![gap_case.clone(), stale_gap.clone(), unqualified.clone()],
        metals: vec![gap_row(&gap_case), gap_row(&stale_gap)],
        moltenvk: vec![
            old_row(&gap_case, MOLTENVK, &METAL, &METAL),
            old_row(&stale_gap, MOLTENVK, &METAL, &METAL),
            old_row(&unqualified, MOLTENVK, &METAL, &METAL),
            // Recorded against a Metal output the oracle no longer holds: stale.
            new_row(&stale_gap, Backend::Moltenvk, MOLTENVK, &METAL, &METAL),
        ],
        ..C3Input::default()
    };
    let unlisted = report_of(
        &input,
        &ReviewedList::default(),
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&unlisted, MOLTENVK, Leg::Stated);
    assert_eq!(stated.population, 3);
    assert_eq!(verdict_of(stated, &gap_case), Verdict::ReferenceGap);
    assert_eq!(verdict_of(stated, &stale_gap), Verdict::ReferenceGap);
    assert_eq!(verdict_of(stated, &unqualified), Verdict::Missing);
    assert_eq!(stated.count(Verdict::ReferenceGap), 2);
    assert_eq!(stated.missing_without_metal, 1);
    assert_eq!(stated.stale_without_metal, 0);
    let rows = |report: &C3Report| {
        report
            .reference_gaps
            .iter()
            .map(|gap| (gap.case_id.clone(), gap.recorded, gap.slots, gap.label()))
            .collect::<Vec<_>>()
    };
    let mut expected = vec![
        (
            gap_case.case_id.clone(),
            Some(10),
            1,
            "unlisted".to_string(),
        ),
        (
            stale_gap.case_id.clone(),
            Some(10),
            1,
            "unlisted".to_string(),
        ),
    ];
    expected.sort();
    assert_eq!(rows(&unlisted), expected);
    assert_eq!(unlisted.awaiting_sign_off(), 2);
    assert!(!unlisted.holds());

    // The unqualified case qualified and recorded on both venues (lavapipe judges it once a Metal
    // row does): every slot holds, and only the gaps' sign-off is left.
    let mut qualified = input.clone();
    qualified.metals.push(metal(&unqualified, &METAL));
    qualified.moltenvk.push(new_row(
        &unqualified,
        Backend::Moltenvk,
        MOLTENVK,
        &METAL,
        &METAL,
    ));
    qualified.vulkan.push(new_row(
        &unqualified,
        Backend::Vulkan,
        LAVAPIPE,
        &METAL,
        &METAL,
    ));
    let entry = |case: &AuthoredCase, apple_family, sign_off| ReferenceGapEntry {
        case_id: case.case_id.clone(),
        name: case.name.clone(),
        apple_family,
        review: Review {
            sign_off,
            ..signed("Apple M3 Max lacks GPU family 10")
        },
    };
    let awaiting = SignOff::AwaitingUser {
        decision: "U-reference-gap".into(),
    };
    let signed_off = signed("").sign_off;
    let list = |first, second| ReviewedList {
        readings: vec![],
        old_mismatches: vec![],
        reference_gaps: vec![entry(&gap_case, 10, first), entry(&stale_gap, 10, second)],
        answer_gaps: vec![],
    };
    let half = report_of(
        &qualified,
        &list(signed_off.clone(), awaiting),
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    assert!(half.slots.iter().all(SlotReport::holds));
    assert_eq!((half.signed_off(), half.awaiting_sign_off()), (1, 1));
    assert!(!half.holds());
    let held = list(signed_off.clone(), signed_off.clone());
    held.check().unwrap();
    let full = report_of(
        &qualified,
        &held,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    assert_eq!((full.signed_off(), full.awaiting_sign_off()), (2, 0));
    assert!(full.holds());

    // A listed gap the store no longer records, or records with another family, is stale.
    let mut other_family = held.clone();
    other_family.reference_gaps[0].apple_family = 11;
    let mut requalified = qualified.clone();
    requalified
        .metals
        .retain(|metal| metal.case_id != stale_gap.case_id);
    requalified.metals.push(metal(&stale_gap, &METAL));
    for (input, list) in [(&qualified, &other_family), (&requalified, &held)] {
        let report = report_of(
            input,
            list,
            &C1List::default(),
            &venues(),
            &VenueKeys::default(),
        );
        assert_eq!(report.stale_reference_gaps(), 1);
        assert_eq!(report.awaiting_sign_off(), 0);
        assert!(!report.holds());
    }

    let mut twice = held.clone();
    twice.reference_gaps.push(twice.reference_gaps[0].clone());
    assert!(twice.check().unwrap_err().contains("listed twice"));
    let mut no_family = held;
    no_family.reference_gaps[0].apple_family = 0;
    assert!(no_family
        .check()
        .unwrap_err()
        .contains("names no Apple GPU family"));
}

#[test]
fn the_ledger_lists_every_old_mismatch_row_with_its_cause_and_the_new_verdict() {
    let fixed = case(1, "fixed");
    let still = case(2, "still");
    let probe_only = case(3, "probe-only");
    let unexplained = case(4, "unexplained");
    let unrelated = case(5, "unrelated");
    let older_metal = [0u8, 0, 0x80, 0xbf];
    let mut input = C3Input {
        cases: vec![
            fixed.clone(),
            still.clone(),
            probe_only.clone(),
            unexplained.clone(),
            unrelated.clone(),
        ],
        ..C3Input::default()
    };
    for case in &input.cases.clone() {
        input.metals.push(metal(case, &METAL));
    }
    // An older Metal reading `fixed`'s old row was compared with, retired since.
    let mut retired = metal(&fixed, &older_metal);
    retired.environment_id = "older-metal".into();
    retired.oracle_abi = "metal-retired-v1".into();
    input.metals.push(retired);
    input.moltenvk = vec![
        // The old engine answered today's Metal reading, but was compared with an older one.
        old_row(&fixed, MOLTENVK, &METAL, &older_metal),
        old_row(&still, MOLTENVK, &DEVICE, &METAL),
        old_row(&probe_only, MOLTENVK, &METAL, &METAL),
        old_row(&probe_only, "probe-build", &DEVICE, &METAL),
        old_row(&unexplained, MOLTENVK, &DEVICE, &METAL),
        new_row(&fixed, Backend::Moltenvk, MOLTENVK, &METAL, &METAL),
        new_row(&still, Backend::Moltenvk, MOLTENVK, &DEVICE, &METAL),
    ];
    let list = ReviewedList {
        readings: vec![reading(&still, vec![reviewed(&input.moltenvk[6], &METAL)])],
        old_mismatches: vec![
            old_cause(&fixed, Cause::FastMathRegrouping),
            old_cause(&still, Cause::FastMathRegrouping),
            old_cause(&probe_only, Cause::FastMathRegrouping),
            old_cause(&unrelated, Cause::AppleReading),
        ],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    let report = report_of(
        &input,
        &list,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    let rows = report
        .ledger
        .iter()
        .map(|row| {
            (
                row.name.as_str(),
                row.environment_id.as_str(),
                row.engine,
                row.cause,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        [
            (
                "fixed",
                MOLTENVK,
                Verdict::Match,
                Some(Cause::FastMathRegrouping)
            ),
            (
                "probe-only",
                "probe-build",
                Verdict::Missing,
                Some(Cause::FastMathRegrouping)
            ),
            (
                "still",
                MOLTENVK,
                Verdict::Reviewed,
                Some(Cause::FastMathRegrouping)
            ),
            ("unexplained", MOLTENVK, Verdict::Missing, None),
        ]
    );
    let fixed_row = &report.ledger[0];
    assert!(!fixed_row.against_current_metal);
    assert!(fixed_row.output_is_current_metal);
    assert_eq!(fixed_row.differing_words, Some(1));
    assert_eq!(fixed_row.words, 1);
    assert_eq!(fixed_row.engine_recorded, Some(CandidateStatus::Match));
    let still_row = &report.ledger[2];
    assert_eq!(still_row.engine_recorded, Some(CandidateStatus::Mismatch));
    assert_eq!(report.ledger[3].engine_recorded, None);
    assert!(still_row.against_current_metal);
    assert!(!still_row.output_is_current_metal);
    assert_eq!(
        report.unexplained(),
        BTreeSet::from([unexplained.case_id.as_str()])
    );
    // A cause for a case no old row mismatched is stale.
    assert_eq!(
        report.stale_causes,
        std::slice::from_ref(&unrelated.case_id)
    );
    assert!(!report.holds());
}

#[test]
fn a_reviewed_reading_that_disagrees_with_the_store_is_reported() {
    let bounded = case(1, "bounded");
    let other = case(2, "other");
    let lavapipe = new_row(&bounded, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    let input = C3Input {
        cases: vec![bounded.clone()],
        metals: vec![metal(&bounded, &METAL)],
        vulkan: vec![lavapipe.clone()],
        public_air: [bounded.air_sha256.clone()].into(),
        ..C3Input::default()
    };
    let mut miscounted = reviewed(&lavapipe, &METAL);
    miscounted.words = 2;
    let mut renamed = reading(&bounded, vec![reviewed(&lavapipe, &METAL)]);
    renamed.name = "renamed".into();
    let mut unobserved = reviewed(&lavapipe, &METAL);
    unobserved.environment_id = engine_slot(LAVAPIPE, None, Leg::RobustnessOff);
    for (entry, expected) in [
        (
            reading(&bounded, vec![miscounted]),
            ReadingState::Inconsistent("the row differs in 1 of 1 words".into()),
        ),
        (
            renamed,
            ReadingState::Inconsistent("the case is named \"bounded\"".into()),
        ),
        (
            reading(&bounded, vec![unobserved]),
            ReadingState::Unobserved,
        ),
        (
            reading(&other, vec![reviewed(&lavapipe, &METAL)]),
            ReadingState::UnknownCase,
        ),
    ] {
        let list = ReviewedList {
            readings: vec![entry],
            old_mismatches: vec![],
            reference_gaps: vec![],
            answer_gaps: vec![],
        };
        let report = report_of(
            &input,
            &list,
            &C1List::default(),
            &venues(),
            &VenueKeys::default(),
        );
        assert_eq!(report.readings[0].state, expected);
        assert_eq!(report.stale_readings(), 1);
        assert!(!report.holds());
    }
}

#[test]
fn reviewed_lists_are_schema_checked() {
    let hash = |tag: &str| tag.repeat(32);
    let observation = |slot: &str, output: &str, words: usize, differing: usize| {
        serde_json::json!({
            "environment_id": slot,
            "spv_sha256": hash("11"),
            "candidate_output_sha256": hash(output),
            "golden_output_sha256": hash("22"),
            "words": words,
            "differing_words": differing,
            "values": "word 0",
        })
    };
    let entry = |case: &str, observations: serde_json::Value| {
        serde_json::json!({
            "case_id": hash(case),
            "name": "case",
            "cause": "apple_reading",
            "bound": "4 ulp",
            "observations": observations,
            "review": {
                "evidence": ["a probe"],
                "note": "reviewed",
                "sign_off": {"state": "awaiting_user", "decision": "C3"},
            },
        })
    };
    let good = serde_json::json!({
        "readings": [entry("aa", serde_json::json!([observation("env@engine", "33", 2, 1)]))],
        "old_mismatches": [{
            "case_id": hash("bb"), "name": "old", "cause": "fast_math_regrouping",
            "review": {
                "evidence": ["a regrouping probe"], "note": "regrouped",
                "sign_off": {"state": "signed_off", "decision": "C3", "date": "2026-10-02"},
            },
        }],
    });
    let list = ReviewedList::parse(&good.to_string()).unwrap();
    // The list round-trips through its committed form, in the order given.
    assert_eq!(ReviewedList::parse(&list.to_json().unwrap()).unwrap(), list);
    let mut reversed = list.clone();
    reversed.readings.insert(
        0,
        Reading {
            case_id: hash("cc"),
            ..list.readings[0].clone()
        },
    );
    assert_eq!(
        ReviewedList::parse(&reversed.to_json().unwrap()).unwrap(),
        reversed
    );
    assert!(list.to_json().unwrap().ends_with("}\n"));
    assert_eq!(
        list.readings[0].review.sign_off,
        SignOff::AwaitingUser {
            decision: "C3".into()
        }
    );
    assert!(list.old_mismatches[0].review.sign_off.is_signed_off());
    let edit = |change: &dyn Fn(&mut serde_json::Value)| {
        let mut list = good.clone();
        change(&mut list);
        list.to_string()
    };
    for (bad, why) in [
        (
            edit(&|list| list["readings"][0]["row"] = "x".into()),
            "unknown field",
        ),
        (
            edit(&|list| list["readings"][0]["cause"] = "guess".into()),
            "unknown cause",
        ),
        (
            edit(&|list| list["readings"][0]["case_id"] = "AA".repeat(32).into()),
            "uppercase hash",
        ),
        (
            edit(&|list| {
                let first = list["readings"][0].clone();
                list["readings"].as_array_mut().unwrap().push(first);
            }),
            "a case twice",
        ),
        (
            edit(&|list| list["readings"][0]["bound"] = " ".into()),
            "no bound",
        ),
        (
            edit(&|list| list["readings"][0]["name"] = " ".into()),
            "no name",
        ),
        (
            edit(&|list| list["old_mismatches"][0]["name"] = " ".into()),
            "a cause without a name",
        ),
        (
            edit(&|list| list["readings"][0]["review"]["evidence"] = serde_json::json!([])),
            "no evidence",
        ),
        (
            edit(&|list| list["readings"][0]["observations"] = serde_json::json!([])),
            "no observation",
        ),
        (
            edit(&|list| {
                list["readings"][0]["observations"] =
                    serde_json::json!([observation("old-engine-env", "33", 2, 1)])
            }),
            "an old-engine slot",
        ),
        (
            edit(&|list| {
                list["readings"][0]["observations"] = serde_json::json!([
                    observation("env@engine", "33", 2, 1),
                    observation("env@engine", "44", 2, 1),
                ])
            }),
            "a slot twice",
        ),
        (
            edit(&|list| {
                list["readings"][0]["observations"] =
                    serde_json::json!([observation("env@engine", "22", 2, 1)])
            }),
            "equal outputs that differ",
        ),
        (
            edit(&|list| {
                list["readings"][0]["observations"] =
                    serde_json::json!([observation("env@engine", "33", 2, 0)])
            }),
            "different outputs that do not",
        ),
        (
            edit(&|list| {
                list["readings"][0]["observations"] =
                    serde_json::json!([observation("env@engine", "33", 1, 2)])
            }),
            "more differing words than words",
        ),
        (
            edit(&|list| list["old_mismatches"][0]["review"]["note"] = "".into()),
            "a cause without a note",
        ),
        (
            edit(&|list| {
                list["readings"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("review");
            }),
            "no review",
        ),
        (
            edit(&|list| {
                list["readings"][0]["review"]
                    .as_object_mut()
                    .unwrap()
                    .remove("sign_off");
            }),
            "no sign-off",
        ),
        (
            edit(&|list| list["old_mismatches"][0]["review"]["by"] = "x".into()),
            "a review with an unknown field",
        ),
        (
            edit(&|list| {
                list["old_mismatches"][0]["review"]["sign_off"] = serde_json::json!(
                    {"state": "signed_off", "decision": "C3", "date": "2026/10/02"}
                )
            }),
            "a sign-off date that is not YYYY-MM-DD",
        ),
        (
            edit(&|list| {
                list["readings"][0]["review"]["sign_off"] =
                    serde_json::json!({"state": "awaiting_user", "decision": " "})
            }),
            "a sign-off that names no decision",
        ),
        (
            edit(&|list| {
                list["readings"][0]["review"]["sign_off"] =
                    serde_json::json!({"state": "awaiting_user", "decision": "C3", "by": "x"})
            }),
            "a sign-off with an unknown field",
        ),
        (
            edit(&|list| {
                list["readings"][0]["review"]["sign_off"] =
                    serde_json::json!({"awaiting_user": {"decision": "C3"}})
            }),
            "a sign-off without its state tag",
        ),
    ] {
        assert!(ReviewedList::parse(&bad).is_err(), "{why}: {bad}");
    }
}

/// Previously an entry carried its evidence, note, and sign-off beside its other fields; the
/// committed list was migrated in-tree, so that shape is rejected rather than read.
#[test]
fn an_entry_in_the_shape_before_the_shared_review_is_rejected() {
    let hash = |tag: &str| tag.repeat(32);
    let flat = serde_json::json!({
        "evidence": ["a regrouping probe"],
        "note": "regrouped",
        "sign_off": {"state": "awaiting_user", "decision": "C3"},
    });
    let observations = serde_json::json!([{
        "environment_id": "env@engine",
        "spv_sha256": hash("11"),
        "candidate_output_sha256": hash("33"),
        "golden_output_sha256": hash("22"),
        "words": 2,
        "differing_words": 1,
        "values": "word 0",
    }]);
    let mut reading = serde_json::json!({
        "case_id": hash("aa"), "name": "case", "cause": "apple_reading", "bound": "4 ulp",
        "observations": observations,
    });
    let mut cause = serde_json::json!({
        "case_id": hash("bb"), "name": "old", "cause": "fast_math_regrouping",
    });
    for (key, value) in flat.as_object().unwrap() {
        reading[key] = value.clone();
        cause[key] = value.clone();
    }
    let old_reading = serde_json::json!({"readings": [reading.clone()], "old_mismatches": []});
    let old_cause = serde_json::json!({"readings": [], "old_mismatches": [cause.clone()]});
    for old in [&old_reading, &old_cause] {
        let error = ReviewedList::parse(&old.to_string()).unwrap_err();
        assert!(error.contains("unknown field"), "{error}");
    }
    // The same entries with their review nested are read.
    for entry in [&mut reading, &mut cause] {
        let object = entry.as_object_mut().unwrap();
        for key in ["evidence", "note", "sign_off"] {
            object.remove(key);
        }
        object.insert("review".into(), flat.clone());
    }
    let migrated = serde_json::json!({"readings": [reading], "old_mismatches": [cause]});
    assert!(ReviewedList::parse(&migrated.to_string()).is_ok());
}

/// A filtered sparse access ran the device's footprint and filter (`SparseResidency`): its mismatch
/// stays open as a deferred near-equivalent with the note's bound, the list may not review it (a
/// reading at the row's candidate reads inconsistent and the row stays a mismatch), and a row under
/// the note that matches Metal is a match.
#[test]
fn a_sparse_residency_row_is_deferred_and_never_a_reviewed_reading() {
    deferred_and_never_reviewed(PrecisionNote::SparseResidency, "SparseResidency:no bound");
}

/// An explicit cube gradient's footprint (`CubeGradientFootprint`) is deferred the same way, with
/// its measured bound.
#[test]
fn a_cube_gradient_footprint_row_is_deferred_and_never_a_reviewed_reading() {
    deferred_and_never_reviewed(
        PrecisionNote::CubeGradientFootprint,
        "CubeGradientFootprint:a cube gradient's footprint",
    );
}

/// A bicubic sample's unpinned arithmetic (`BicubicFilterArithmetic`) is deferred the same way,
/// with its measured bound: no model of Apple's 2D rounding fitted.
#[test]
fn a_bicubic_filter_arithmetic_row_is_deferred_and_never_a_reviewed_reading() {
    deferred_and_never_reviewed(
        PrecisionNote::BicubicFilterArithmetic,
        "BicubicFilterArithmetic:a 2D bicubic sum",
    );
}

/// A varying the device interpolates (`DeviceInterpolation`), which nearly every fragment function
/// records off MoltenVK, defers nothing: a mismatching row recording it stays a mismatch with no
/// reason, never held open as the interpolation's, and one that matches Metal is a match.
#[test]
fn a_device_interpolation_row_stays_a_mismatch() {
    use crate::coverage::{ExecutedCensus, ModuleCensus};
    let note = PrecisionNote::DeviceInterpolation;
    assert!(!DEFERRED_NEAR_EQUIVALENTS.contains(&note));
    let drawn = case(1, "drawn");
    let noted = |output: &[u8]| {
        let mut row = new_row(&drawn, Backend::Vulkan, LAVAPIPE, output, &METAL);
        row.census = Some(ExecutedCensus {
            out_of_bounds: None,
            rate_map_pass: None,
            device: None,
            product: crate::observation::TRANSLATOR_FINGERPRINT.into(),
            restricted_to: None,
            leg: None,
            modules: vec![ModuleCensus {
                stage: crate::case::Stage::Fragment,
                subgroup: crate::coverage::SubgroupShape {
                    full_subgroups: false,
                    required_size: None,
                },
                strategies: Default::default(),
                precision: [note].into(),
                containment: None,
            }],
            artifact: None,
        });
        row.comparison = ComparisonClass::Unbounded([note].into());
        assert!(row.class_holds());
        row
    };
    for (output, want) in [(&DEVICE, Verdict::Mismatch), (&METAL, Verdict::Match)] {
        let input = C3Input {
            cases: vec![drawn.clone()],
            metals: vec![metal(&drawn, &METAL)],
            public_air: [drawn.air_sha256.clone()].into(),
            vulkan: vec![noted(output)],
            ..C3Input::default()
        };
        let report = report_of(
            &input,
            &ReviewedList::default(),
            &C1List::default(),
            &venues(),
            &VenueKeys::default(),
        );
        let stated = slot(&report, LAVAPIPE, Leg::Stated);
        assert_eq!(verdict_of(stated, &drawn), want);
        let open = stated
            .rows
            .iter()
            .find(|row| row.case_id == drawn.case_id)
            .map(|row| (row.verdict, row.reason.clone()));
        let want_open = (want == Verdict::Mismatch).then_some((Verdict::Mismatch, None));
        assert_eq!(open, want_open);
    }
}

/// A vertex function's simdgroup operation on a device that groups vertices otherwise than
/// Apple's GPUs (`VertexSimdgroupGrouping`) is deferred the same way, with its measured bound.
#[test]
fn a_vertex_simdgroup_grouping_row_is_deferred_and_never_a_reviewed_reading() {
    deferred_and_never_reviewed(
        PrecisionNote::VertexSimdgroupGrouping,
        "VertexSimdgroupGrouping:a vertex function's simdgroup",
    );
}

/// A fragment function's simdgroup operation on a device whose fragment subgroups are not Metal's
/// simdgroups (`FragmentSimdgroupGrouping`) is deferred the same way.
#[test]
fn a_fragment_simdgroup_grouping_row_is_deferred_and_never_a_reviewed_reading() {
    deferred_and_never_reviewed(
        PrecisionNote::FragmentSimdgroupGrouping,
        "FragmentSimdgroupGrouping:a fragment function's simdgroup",
    );
}

/// A blended half-float color channel under `textureWriteRoundingMode` 2, which the device's
/// blend rounds (`BlendedWriteRounding`), is deferred the same way.
#[test]
fn a_blended_write_rounding_row_is_deferred_and_never_a_reviewed_reading() {
    deferred_and_never_reviewed(
        PrecisionNote::BlendedWriteRounding,
        "BlendedWriteRounding:a half-float color channel",
    );
}

/// A pipeline covering by alpha under `textureWriteRoundingMode` 2 whose coverage reads the
/// rounded alpha (`CoverageReadsRoundedAlpha`) is deferred the same way.
#[test]
fn a_coverage_reads_rounded_alpha_row_is_deferred_and_never_a_reviewed_reading() {
    deferred_and_never_reviewed(
        PrecisionNote::CoverageReadsRoundedAlpha,
        "CoverageReadsRoundedAlpha:a pipeline covering by alpha",
    );
}

/// An `ARGB8Unorm` attachment blended in none of the measured forms (`RotatedBlendRounding`) is
/// deferred the same way.
#[test]
fn a_rotated_blend_rounding_row_is_deferred_and_never_a_reviewed_reading() {
    deferred_and_never_reviewed(
        PrecisionNote::RotatedBlendRounding,
        "RotatedBlendRounding:an ARGB8Unorm attachment",
    );
}

/// A row that ran `note` and differs from Metal stays a mismatch deferred under the note, whose
/// label starts with `label`; a reading at its candidate reads inconsistent; a row under the note
/// that matches Metal is a match.
fn deferred_and_never_reviewed(note: PrecisionNote, label: &str) {
    use crate::coverage::{ExecutedCensus, ModuleCensus};
    let sampled = case(1, "sampled");
    let noted = |output: &[u8]| {
        let mut row = new_row(&sampled, Backend::Vulkan, LAVAPIPE, output, &METAL);
        row.census = Some(ExecutedCensus {
            out_of_bounds: None,
            rate_map_pass: None,
            device: None,
            product: crate::observation::TRANSLATOR_FINGERPRINT.into(),
            restricted_to: None,
            leg: None,
            modules: vec![ModuleCensus {
                stage: crate::case::Stage::Kernel,
                subgroup: crate::coverage::SubgroupShape {
                    full_subgroups: false,
                    required_size: None,
                },
                strategies: Default::default(),
                precision: [note].into(),
                containment: None,
            }],
            artifact: None,
        });
        row.comparison = ComparisonClass::Unbounded([note].into());
        assert!(row.class_holds());
        row
    };
    let differs = noted(&DEVICE);
    let input = C3Input {
        cases: vec![sampled.clone()],
        metals: vec![metal(&sampled, &METAL)],
        public_air: [sampled.air_sha256.clone()].into(),
        vulkan: vec![differs.clone()],
        ..C3Input::default()
    };
    let list = ReviewedList {
        readings: vec![reading(&sampled, vec![reviewed(&differs, &METAL)])],
        old_mismatches: vec![],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    let report = report_of(
        &input,
        &list,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    let deferred = OpenReason::NearEquivalent(note);
    let open = stated
        .rows
        .iter()
        .find(|row| row.case_id == sampled.case_id)
        .map(|row| (row.verdict, row.reason.clone()));
    assert_eq!(open, Some((Verdict::Mismatch, Some(deferred.clone()))));
    assert!(
        deferred
            .label()
            .starts_with(&format!("deferred{{near_equivalent:{label}")),
        "{}",
        deferred.label()
    );
    assert_eq!(
        report.readings[0].state,
        ReadingState::Inconsistent(format!(
            "the row ran the deferred near-equivalent {note:?}, which stays open and is never a \
             reviewed reading"
        ))
    );
    assert!(!report.holds());
    // Under the note, a row that matches Metal is a match, with no reason.
    let mut matched = input;
    matched.vulkan = vec![noted(&METAL)];
    let empty = ReviewedList::default();
    let report = report_of(
        &matched,
        &empty,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(stated, &sampled), Verdict::Match);
    assert!(stated.rows.iter().all(|row| row.case_id != sampled.case_id));
    // A match is named, not inferred from its absence among the open rows; a case the slot did not
    // judge has no verdict there.
    assert_eq!(stated.matched, [sampled.case_id.clone()].into());
    assert_eq!(stated.verdict_of(&"ff".repeat(32)), None);
}

/// The user's decisions of 2026-10-03: a fast-math regrouping and an Apple reading are Metal's own
/// latitude, which the list may sign off (an Apple reading only with its bounded lanes, and only
/// while every differing word is within them). The cause of the device's unfused `fma` is gone:
/// `fma` is exact, its case matches, and the list no longer parses it. A recorded near-equivalent
/// stays open through the deferred list for a whole row
/// (`a_sparse_residency_row_is_deferred_and_never_a_reviewed_reading`), and word by word through
/// the bound check.
#[test]
fn metals_latitude_may_be_signed_off_and_the_fma_cause_is_gone() {
    let fma = case(3, "fma");
    let row = new_row(&fma, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    assert!(
        serde_json::from_str::<Cause>("\"pinned_by_metal_not_vulkan\"").is_err(),
        "a superseded cause parses"
    );
    for cause in [Cause::FastMathRegrouping, Cause::AppleReading] {
        let mut entry = reading(&fma, vec![reviewed(&row, &METAL)]);
        entry.cause = cause;
        if cause != Cause::AppleReading {
            entry.lanes.clear();
        }
        let list = ReviewedList {
            readings: vec![entry],
            old_mismatches: vec![],
            reference_gaps: vec![],
            answer_gaps: vec![],
        };
        assert_eq!(list.check(), Ok(()), "{}", cause.as_str());
    }
}

/// An Apple reading is signed off only with bounded lanes, only an Apple reading states them, and
/// they must be well formed.
#[test]
fn an_apple_reading_is_signed_off_only_with_well_formed_lanes() {
    let bounded = case(1, "bounded");
    let row = new_row(&bounded, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    let list = |change: &dyn Fn(&mut Reading)| {
        let mut entry = reading(&bounded, vec![reviewed(&row, &METAL)]);
        change(&mut entry);
        ReviewedList {
            readings: vec![entry],
            old_mismatches: vec![],
            reference_gaps: vec![],
            answer_gaps: vec![],
        }
        .check()
    };
    assert_eq!(list(&|_| {}), Ok(()));
    let error = list(&|entry| entry.lanes.clear()).unwrap_err();
    assert!(
        error.contains("signed off only with the bounded lanes"),
        "{error}"
    );
    // Awaiting the user, a reading may state no lanes yet.
    assert_eq!(
        list(&|entry| {
            entry.lanes.clear();
            entry.review.sign_off = SignOff::AwaitingUser {
                decision: "C3".into(),
            };
        }),
        Ok(())
    );
    let error = list(&|entry| entry.cause = Cause::FastMathRegrouping).unwrap_err();
    assert!(error.contains("only an apple_reading does"), "{error}");
    for (change, why) in [
        (
            Box::new(|entry: &mut Reading| entry.lanes[0].computes = "cbrt(a)".into())
                as Box<dyn Fn(&mut Reading)>,
            "no MSL operation",
        ),
        (
            Box::new(|entry: &mut Reading| entry.lanes[0].computes = "atan2(a)".into()),
            "takes 2 arguments",
        ),
        (
            Box::new(|entry: &mut Reading| entry.lanes[0].computes = "exp2(b)".into()),
            "reads 2 operands",
        ),
        (
            Box::new(|entry: &mut Reading| entry.lanes[0].computes = "a".into()),
            "calls no MSL operation",
        ),
        (
            Box::new(|entry: &mut Reading| {
                entry.lanes[0].result = Part::F16Low;
                entry.lanes[0].computes = "pow(a, b)".into();
                let first = entry.lanes[0].operands[0].clone();
                entry.lanes[0].operands.push(first);
            }),
            "state no pow for f16",
        ),
        (
            Box::new(|entry: &mut Reading| entry.lanes[0].words.count = 0),
            "covers no word",
        ),
        (
            Box::new(|entry: &mut Reading| {
                let mut half = entry.lanes[0].clone();
                half.result = Part::F16High;
                entry.lanes.push(half);
            }),
            "covered by two lanes",
        ),
        (
            Box::new(|entry: &mut Reading| {
                entry.lanes[0].words.count = 2;
                entry.lanes[0].operands = vec![Operand::Bits {
                    format: Format::F16,
                    first: 0xffff,
                    stride: 1,
                }];
            }),
            "past f16",
        ),
    ] {
        let error = list(&change).unwrap_err();
        assert!(error.contains(why), "{why}: {error}");
    }
}

/// The report recomputes the word the device differs from Metal in against the lane's bound. Within
/// it the sign-off stands; one representable value past it (and a word no lane covers) the report
/// refuses the sign-off and names the word, and C3 does not hold.
#[test]
fn a_signed_off_apple_reading_holds_only_while_every_differing_word_is_within_apples_bound() {
    let bounded = case(1, "bounded");
    let at = |device: f32, entry: &dyn Fn(&mut Reading)| {
        let device = device.to_le_bytes();
        let row = new_row(&bounded, Backend::Vulkan, LAVAPIPE, &device, &METAL);
        let mut reading = reading(&bounded, vec![reviewed(&row, &METAL)]);
        entry(&mut reading);
        let input = C3Input {
            cases: vec![bounded.clone()],
            metals: vec![metal(&bounded, &METAL)],
            vulkan: vec![row],
            public_air: [bounded.air_sha256.clone()].into(),
            ..C3Input::default()
        };
        let list = ReviewedList {
            readings: vec![reading],
            old_mismatches: vec![],
            reference_gaps: vec![],
            answer_gaps: vec![],
        };
        list.check().unwrap();
        report_of(
            &input,
            &list,
            &C1List::default(),
            &lavapipe_alone(),
            &VenueKeys::default(),
        )
    };
    // exp2(0) without fast math: Table 8.1 admits 4 ulp of 1.0, whose ulp is 2^-24 (MSL section
    // 8.4: at a power of two, the distance to the nearest other value, below it). 1 - 2^-22 is
    // 4 such ulps below.
    let report = at(1.0 - 2f32.powi(-22), &|_| {});
    assert_eq!(report.readings[0].state, ReadingState::Holds);
    let check = report.readings[0].bound.as_ref().unwrap();
    assert_eq!(check.count("within"), 1);
    assert!(report.refused.is_empty());
    assert!(report.holds());

    // One value below that is outside: the sign-off is refused and the word named.
    let report = at(1.0 - 2f32.powi(-22) - 2f32.powi(-24), &|_| {});
    let check = report.readings[0].bound.as_ref().unwrap();
    assert_eq!(check.count("outside"), 1);
    assert_eq!(report.readings[0].state, ReadingState::Holds);
    assert_eq!(report.refused.len(), 1);
    assert!(
        report.refused[0].words[0].contains("word 0 (f32): outside exp2(a)"),
        "{:?}",
        report.refused
    );
    assert!(
        report.refused[0].words[0].contains("5.0 ulp"),
        "{:?}",
        report.refused
    );
    assert!(!report.holds());

    // Awaiting the user, the same word is reported, and no sign-off is refused.
    let report = at(1.0 - 2f32.powi(-22) - 2f32.powi(-24), &|entry| {
        entry.review.sign_off = SignOff::AwaitingUser {
            decision: "C3".into(),
        }
    });
    assert_eq!(
        report.readings[0].bound.as_ref().unwrap().count("outside"),
        1
    );
    assert!(report.refused.is_empty());

    // A word no lane covers is never within.
    let report = at(1.0 - 2f32.powi(-22), &|entry| {
        entry.lanes[0].words.first = 1
    });
    assert_eq!(
        report.readings[0]
            .bound
            .as_ref()
            .unwrap()
            .count("uncovered"),
        1
    );
    assert_eq!(report.refused.len(), 1);

    // A lane that does not describe the word (Metal's own 1.0 is far from exp2(3) = 8) cannot
    // vouch for it.
    let report = at(1.0 - 2f32.powi(-22), &|entry| {
        entry.lanes[0].operands = vec![Operand::Bits {
            format: Format::F32,
            first: 3.0f32.to_bits(),
            stride: 1,
        }]
    });
    assert_eq!(
        report.readings[0]
            .bound
            .as_ref()
            .unwrap()
            .count("not_checkable"),
        1
    );
    assert_eq!(report.refused.len(), 1);
}

/// Where Apple states no accuracy, the device's answer is within Apple's promise, but not the
/// product's approximation. A row whose modules record the product's lowering as a near-equivalent
/// of Metal's (`DeviceArithmetic`) keeps the word open, so the report refuses the sign-off,
/// whatever the unstated-words flag says: fast `fmod` ("Undefined"). Fast `float` `tanh` at
/// `tanh(44)`, whose formula's divisor is past the `2^126` where Table 8.2 bounds the fast
/// division, is unstated too, but the product's fast `tanh` is the device's `Tanh` (its lowering
/// of `air.fast_tanh.f32` records no `DeviceArithmetic`): the device's own answer, within the
/// promise, so the sign-off stands even where the row records the note for another operation.
#[test]
fn an_unstated_word_the_product_approximates_keeps_its_reading_open() {
    /// A one-word fast-math reading: the case's AIR tag, what the word computes of which bit
    /// patterns, and Metal's and the device's answers.
    struct Word {
        tag: u8,
        computes: &'static str,
        operands: Vec<u32>,
        metal: [u8; 4],
        device: [u8; 4],
    }
    let report_of = |word: &Word, notes: &[PrecisionNote]| {
        let case = case(word.tag, word.computes);
        let row = noting(
            new_row(&case, Backend::Vulkan, LAVAPIPE, &word.device, &word.metal),
            notes,
        );
        let mut reading = reading(&case, vec![reviewed(&row, &word.metal)]);
        reading.lanes = vec![lane(word.computes, Math::Fast, &word.operands)];
        let input = C3Input {
            cases: vec![case.clone()],
            metals: vec![metal(&case, &word.metal)],
            vulkan: vec![row],
            public_air: [case.air_sha256.clone()].into(),
            ..C3Input::default()
        };
        let list = ReviewedList {
            readings: vec![reading],
            old_mismatches: vec![],
            reference_gaps: vec![],
            answer_gaps: vec![],
        };
        list.check().unwrap();
        report_of(
            &input,
            &list,
            &C1List::default(),
            &venues(),
            &VenueKeys::default(),
        )
    };
    // fast fmod(1e10, 3): Metal answers -512, the device 0.
    let fmod = Word {
        tag: 3,
        computes: "fmod(a, b)",
        operands: vec![1e10f32.to_bits(), 3.0f32.to_bits()],
        metal: (-512.0f32).to_le_bytes(),
        device: 0.0f32.to_le_bytes(),
    };
    // Fast tanh(44): Metal's fast division zeroes the reciprocal of t + 1 past 2^126 and answers
    // 0; the device answers 1.
    let tanh = Word {
        tag: 4,
        computes: "tanh(a)",
        operands: vec![44.0f32.to_bits()],
        metal: 0.0f32.to_le_bytes(),
        device: 1.0f32.to_le_bytes(),
    };
    for word in [&fmod, &tanh] {
        // The device's own answer: within the promise, so the sign-off stands.
        let report = report_of(word, &[PrecisionNote::DeviceTranscendentals]);
        let check = report.readings[0].bound.as_ref().unwrap();
        assert_eq!(check.count("unstated"), 1, "{}", word.computes);
        assert!(report.refused.is_empty());
    }
    let recorded = [
        PrecisionNote::DeviceArithmetic,
        PrecisionNote::DeviceTranscendentals,
    ];
    // The product's remainder, recorded as a near-equivalent: open, and the sign-off refused.
    let report = report_of(&fmod, &recorded);
    let check = report.readings[0].bound.as_ref().unwrap();
    assert_eq!(check.count("near_equivalent"), 1);
    assert_eq!(report.refused.len(), 1);
    assert!(
        report.refused[0].words[0].contains("near_equivalent")
            && report.refused[0].words[0].contains("DeviceArithmetic"),
        "{:?}",
        report.refused
    );
    assert!(!report.holds());
    let report = report_of(&fmod, &[PrecisionNote::DeviceArithmetic]);
    assert!(
        report.refused[0].words[0].contains("its lowering of air.fast_fmod.f32 (Metal's fmod)"),
        "{:?}",
        report.refused
    );
    // The device's `Tanh`: no near-equivalent of the product's, so the word stays unstated and
    // the sign-off stands, whatever arithmetic notes the row records.
    let report = report_of(&tanh, &recorded);
    let check = report.readings[0].bound.as_ref().unwrap();
    assert_eq!(check.count("unstated"), 1);
    assert_eq!(check.count("near_equivalent"), 0);
    assert!(report.refused.is_empty(), "{:?}", report.refused);
}

/// A word within Apple's bound that shows a recorded near-equivalent's difference keeps its reading
/// open, so the report refuses the sign-off; a word that shows none keeps its verdict, and the
/// sign-off stands, whatever arithmetic notes the row records. A fast `sinpi` of a subnormal: +0
/// where the operand is read as zero, as Metal reads it, the subnormal pi times it
/// where the device keeps it (fast `sinpi`'s lowering records `DeviceDenormals`), and Table 8.2's
/// absolute 2^-13 admits both. A product whose result is a subnormal is open too: lavapipe's
/// multiply is the device's, not measured to flush as Metal does, and records the note. A product
/// one rounding away (toward zero, which "correctly rounded" admits) touches no subnormal. (Precise
/// `fmod` answers Metal at denormals itself.)
#[test]
fn a_word_within_apples_bound_that_shows_a_recorded_near_equivalent_keeps_its_reading_open() {
    let report_of = |computes: &str,
                     math: Math,
                     operands: &[f32],
                     golden: f32,
                     output: f32,
                     notes: &[PrecisionNote]| {
        let case = case(5, "product");
        let (golden, output) = (golden.to_le_bytes(), output.to_le_bytes());
        let row = noting(
            new_row(&case, Backend::Vulkan, LAVAPIPE, &output, &golden),
            notes,
        );
        let mut reading = reading(&case, vec![reviewed(&row, &golden)]);
        let operands = operands.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        reading.lanes = vec![lane(computes, math, &operands)];
        let input = C3Input {
            cases: vec![case.clone()],
            metals: vec![metal(&case, &golden)],
            vulkan: vec![row],
            public_air: [case.air_sha256.clone()].into(),
            ..C3Input::default()
        };
        let list = ReviewedList {
            readings: vec![reading],
            old_mismatches: vec![],
            reference_gaps: vec![],
            answer_gaps: vec![],
        };
        list.check().unwrap();
        report_of(
            &input,
            &list,
            &C1List::default(),
            &venues(),
            &VenueKeys::default(),
        )
    };
    let every = [
        PrecisionNote::DeviceArithmetic,
        PrecisionNote::DeviceTranscendentals,
        PrecisionNote::DeviceDenormals,
        PrecisionNote::DeviceNanEncoding,
    ];
    let tiny = f32::from_bits(1 << 19); // 2^-130
    let kept = (std::f64::consts::PI * f64::from(tiny)) as f32;

    // Without the denormal note the word is Apple's: within, and the sign-off stands.
    let report = report_of("sinpi(a)", Math::Fast, &[tiny], 0.0, kept, &every[..2]);
    let check = report.readings[0].bound.as_ref().unwrap();
    assert_eq!(check.count("within"), 1);
    assert!(report.refused.is_empty());

    // With it, open with the exact path, and the sign-off refused.
    let report = report_of("sinpi(a)", Math::Fast, &[tiny], 0.0, kept, &every);
    let check = report.readings[0].bound.as_ref().unwrap();
    assert_eq!(check.count("near_equivalent"), 1);
    assert_eq!(report.refused.len(), 1);
    let words = &report.refused[0].words[0];
    for part in ["DeviceDenormals", "flushing select", "air.fast_sinpi.f32"] {
        assert!(words.contains(part), "{part}: {words}");
    }
    assert!(!report.holds());

    // 2^-100 * 2^-30, Metal's +0 against a kept 2^-130: the multiply records the denormal note,
    // so open, and the sign-off refused.
    let (a, b) = (f32::from_bits(27 << 23), f32::from_bits((127 - 30) << 23)); // 2^-100, 2^-30
    let report = report_of("mul(a, b)", Math::Precise, &[a, b], 0.0, tiny, &every);
    let check = report.readings[0].bound.as_ref().unwrap();
    assert_eq!(check.count("near_equivalent"), 1);
    assert_eq!(report.refused.len(), 1);
    assert!(report.refused[0].words[0].contains("(Metal's mul)"));

    // (1 / 3)^2 rounded toward zero against Metal's nearest: no subnormal, no NaN, so within,
    // and the sign-off stands with every note recorded.
    let third = 1.0f32 / 3.0;
    let square = third * third;
    let down = f32::from_bits(square.to_bits() - 1);
    let exact = f64::from(third) * f64::from(third);
    assert!(f64::from(down) < exact && exact < f64::from(square));
    let report = report_of(
        "mul(a, b)",
        Math::Precise,
        &[third, third],
        square,
        down,
        &every,
    );
    let check = report.readings[0].bound.as_ref().unwrap();
    assert_eq!(check.count("within"), 1);
    assert!(report.refused.is_empty());
}

#[test]
fn the_committed_list_parses_in_its_committed_form() {
    let path = ReviewedList::committed_path();
    assert!(path.ends_with("cutover/c3-reviewed.json"));
    let list = ReviewedList::load(&path).unwrap();
    assert_eq!(
        list.to_json().unwrap(),
        std::fs::read_to_string(&path).unwrap()
    );
}

/// Three venues, each with its own population in its own legs: MoltenVK covers the cases its
/// old-engine rows matched in its stated leg, lavapipe the public cases in its stated leg, and the
/// real GPU every Metal-qualified case in both legs, public or private.
#[test]
fn each_venue_covers_its_own_population_in_its_own_legs() {
    let public = case(1, "public");
    let private = case(2, "private");
    let unqualified = case(3, "unqualified");
    let old_matched = case(4, "old-matched");
    let mut input = C3Input {
        cases: vec![
            public.clone(),
            private.clone(),
            unqualified.clone(),
            old_matched.clone(),
        ],
        metals: [&public, &private, &old_matched]
            .map(|case| metal(case, &METAL))
            .to_vec(),
        public_air: [public.air_sha256.clone()].into(),
        moltenvk: vec![old_row(&old_matched, MOLTENVK, &METAL, &METAL)],
        ..C3Input::default()
    };
    // The real GPU ran the public case in both legs and the private case in the stated leg only.
    let radv = |case: &AuthoredCase, leg: Leg| {
        in_leg(new_row(case, Backend::Vulkan, RADV, &METAL, &METAL), leg)
    };
    input.vulkan = vec![
        radv(&public, Leg::Stated),
        radv(&public, Leg::RobustnessOff),
        radv(&private, Leg::Stated),
        new_row(&public, Backend::Vulkan, LAVAPIPE, &METAL, &METAL),
    ];
    let report = report_of(
        &input,
        &ReviewedList::default(),
        &C1List::default(),
        &three_venues(),
        &VenueKeys::default(),
    );
    assert_eq!(report.slots.len(), 6);
    let population = |venue: &str, leg: Leg| slot(&report, venue, leg).population;
    assert_eq!(population(MOLTENVK, Leg::Stated), 1);
    assert_eq!(population(MOLTENVK, Leg::RobustnessOff), 0);
    assert_eq!(population(LAVAPIPE, Leg::Stated), 1);
    assert_eq!(population(LAVAPIPE, Leg::RobustnessOff), 0);
    // Every case with a current Metal row, in each leg: the unqualified case is outside.
    assert_eq!(population(RADV, Leg::Stated), 3);
    assert_eq!(population(RADV, Leg::RobustnessOff), 3);

    let radv_stated = slot(&report, RADV, Leg::Stated);
    assert_eq!(radv_stated.kind, VenueKind::NativeGpu);
    assert_eq!(verdict_of(radv_stated, &public), Verdict::Match);
    assert_eq!(verdict_of(radv_stated, &private), Verdict::Match);
    assert_eq!(verdict_of(radv_stated, &old_matched), Verdict::Missing);
    assert_eq!(radv_stated.count(Verdict::Missing), 1);
    assert_eq!(radv_stated.missing_without_metal, 0);
    let radv_off = slot(&report, RADV, Leg::RobustnessOff);
    assert_eq!(verdict_of(radv_off, &public), Verdict::Match);
    assert_eq!(verdict_of(radv_off, &private), Verdict::Missing);
    assert_eq!(radv_off.count(Verdict::Missing), 2);
    assert!(!radv_stated.holds() && !radv_off.holds());
    // The other venues judge their own populations, and read missing every other case a Metal
    // row judges that they never recorded, in the legs they list.
    let lavapipe = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(lavapipe, &public), Verdict::Match);
    assert_eq!(verdict_of(lavapipe, &private), Verdict::Missing);
    assert_eq!(verdict_of(lavapipe, &old_matched), Verdict::Missing);
    assert!(!lavapipe.holds());
    assert_eq!(
        verdict_of(slot(&report, MOLTENVK, Leg::Stated), &old_matched),
        Verdict::Missing
    );
    // Only the venue holding the baseline counts new mismatches against it.
    let mut mismatching = input.clone();
    mismatching.vulkan[2] = new_row(&private, Backend::Vulkan, RADV, &DEVICE, &METAL);
    let report = report_of(
        &mismatching,
        &ReviewedList::default(),
        &C1List::default(),
        &three_venues(),
        &VenueKeys::default(),
    );
    let radv_stated = slot(&report, RADV, Leg::Stated);
    assert_eq!(verdict_of(radv_stated, &private), Verdict::Mismatch);
    assert_eq!(radv_stated.new_mismatches, 0);
}

/// A case outside a venue's population that the venue never recorded reads missing in each leg
/// the venue lists once a Metal row judges it: the venue's lane runs every case C3 reads stale or
/// missing in its slot, so the case is selected and recorded, never left unrun because the venue's
/// population (the old engine's matches, the public cases) is narrower than the corpus. It does not
/// join the population, a case with no Metal row is not judged, and a leg the venue does not list
/// judges nothing.
#[test]
fn a_case_a_venue_never_recorded_is_missing_and_selected_by_its_lane() {
    let private = case(1, "private");
    let unqualified = case(2, "unqualified");
    let input = C3Input {
        cases: vec![private.clone(), unqualified.clone()],
        metals: vec![metal(&private, &METAL)],
        // Only the real GPU recorded the private case; nothing recorded the unqualified one.
        vulkan: [Leg::Stated, Leg::RobustnessOff]
            .map(|leg| {
                in_leg(
                    new_row(&private, Backend::Vulkan, RADV, &METAL, &METAL),
                    leg,
                )
            })
            .to_vec(),
        ..C3Input::default()
    };
    let judge = |input: &C3Input| {
        report_of(
            input,
            &ReviewedList::default(),
            &C1List::default(),
            &three_venues(),
            &VenueKeys::default(),
        )
    };
    let report = judge(&input);
    for venue in [MOLTENVK, LAVAPIPE] {
        let stated = slot(&report, venue, Leg::Stated);
        // Outside the population (no old-engine match, no public fixture): still judged.
        assert_eq!(stated.population, 0, "{venue}");
        assert_eq!(verdict_of(stated, &private), Verdict::Missing, "{venue}");
        // Selected: the lane runs the slot's `stale` and `missing` rows, which corpus-status
        // prints as c3-row lines.
        let selected = stated
            .rows
            .iter()
            .filter(|row| matches!(row.verdict, Verdict::Stale | Verdict::Missing))
            .map(|row| row.case_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(selected, [private.case_id.as_str()], "{venue}");
        // It has a Metal row, so it is no case waiting on Metal's qualification.
        assert_eq!(stated.missing_without_metal, 0, "{venue}");
        // A case with no Metal row is not judged: there is nothing to compare its record with.
        assert_eq!(stated.verdict_of(&unqualified.case_id), None, "{venue}");
        assert!(!stated.holds(), "{venue}");
        // Neither venue lists the robustness-off leg: nothing is judged there.
        let off = slot(&report, venue, Leg::RobustnessOff);
        assert_eq!(off.verdict_of(&private.case_id), None, "{venue}");
        assert!(off.holds(), "{venue}");
    }
    assert_eq!(
        verdict_of(slot(&report, RADV, Leg::Stated), &private),
        Verdict::Match
    );

    // Once each venue records it, the case is judged by its record and every slot holds.
    let mut recorded = input.clone();
    recorded.moltenvk.push(new_row(
        &private,
        Backend::Moltenvk,
        MOLTENVK,
        &METAL,
        &METAL,
    ));
    recorded
        .vulkan
        .push(new_row(&private, Backend::Vulkan, LAVAPIPE, &METAL, &METAL));
    let report = judge(&recorded);
    for venue in [MOLTENVK, LAVAPIPE] {
        let stated = slot(&report, venue, Leg::Stated);
        assert_eq!(verdict_of(stated, &private), Verdict::Match, "{venue}");
        assert_eq!(stated.population, 0, "{venue}");
    }
    assert!(report.slots.iter().all(SlotReport::holds));
}

/// A row recorded under another driver than its venue declares is stale, never current: before
/// the venue's declared identity moves to the new driver, the new driver's rows are stale; after,
/// the older driver's are.
#[test]
fn a_row_recorded_under_another_driver_is_stale_never_current() {
    let bounded = case(1, "bounded");
    let input = |rows: Vec<CandidateObservation>| C3Input {
        cases: vec![bounded.clone()],
        metals: vec![metal(&bounded, &METAL)],
        vulkan: rows,
        ..C3Input::default()
    };
    let declared = new_row(&bounded, Backend::Vulkan, RADV, &METAL, &METAL);
    let mut updated = declared.clone();
    updated.environment["driver_info"] = "Mesa 26.2.5".into();
    updated.environment["driver_version"] = 109060101.into();

    // Under the declared driver the row matches.
    let report = report_of(
        &input(vec![declared.clone()]),
        &ReviewedList::default(),
        &C1List::default(),
        &three_venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, RADV, Leg::Stated);
    assert_eq!(verdict_of(stated, &bounded), Verdict::Match);
    assert_eq!(stated.stale_device, 0);
    assert!(stated.holds());

    // Recorded after the host's driver moved: stale, though every other dependency is current.
    let report = report_of(
        &input(vec![updated.clone()]),
        &ReviewedList::default(),
        &C1List::default(),
        &three_venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, RADV, Leg::Stated);
    assert_eq!(verdict_of(stated, &bounded), Verdict::Stale);
    assert_eq!(stated.stale_device, 1);
    assert!(!stated.holds());

    // Once the venue declares the new driver, the older driver's row is the stale one.
    let mut moved = three_venues();
    moved.venues[2].devices[0].identity.driver_info = "Mesa 26.2.5".into();
    moved.venues[2].devices[0].identity.driver_version = 109060101;
    let report = report_of(
        &input(vec![declared.clone()]),
        &ReviewedList::default(),
        &C1List::default(),
        &moved,
        &VenueKeys::default(),
    );
    assert_eq!(
        verdict_of(slot(&report, RADV, Leg::Stated), &bounded),
        Verdict::Stale
    );
    let report = report_of(
        &input(vec![updated]),
        &ReviewedList::default(),
        &C1List::default(),
        &moved,
        &VenueKeys::default(),
    );
    assert_eq!(
        verdict_of(slot(&report, RADV, Leg::Stated), &bounded),
        Verdict::Match
    );

    // Another device, and a row that records no identity, are stale too; so is a mismatching
    // row's reviewed reading, which is no longer current evidence.
    let mut other_device = new_row(&bounded, Backend::Vulkan, RADV, &DEVICE, &METAL);
    other_device.environment["device_id"] = 0x1681.into();
    let mut unrecorded = new_row(&bounded, Backend::Vulkan, RADV, &METAL, &METAL);
    unrecorded
        .environment
        .as_object_mut()
        .unwrap()
        .remove("driver_info");
    for row in [other_device.clone(), unrecorded] {
        let report = report_of(
            &input(vec![row]),
            &ReviewedList::default(),
            &C1List::default(),
            &three_venues(),
            &VenueKeys::default(),
        );
        let stated = slot(&report, RADV, Leg::Stated);
        assert_eq!(verdict_of(stated, &bounded), Verdict::Stale);
        assert_eq!(stated.stale_device, 1);
    }
    let list = ReviewedList {
        readings: vec![reading(&bounded, vec![reviewed(&other_device, &METAL)])],
        old_mismatches: vec![],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    let report = report_of(
        &input(vec![other_device]),
        &list,
        &C1List::default(),
        &three_venues(),
        &VenueKeys::default(),
    );
    assert!(!report.readings[0].current);
}

/// A reviewed reading names a slot: lavapipe's review of a case does not cover the same bytes on
/// the real GPU until the reading names the GPU's slot too.
#[test]
fn a_reading_reviewed_in_one_venue_does_not_cover_another() {
    let bounded = case(1, "bounded");
    let lavapipe = new_row(&bounded, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    let radv = new_row(&bounded, Backend::Vulkan, RADV, &DEVICE, &METAL);
    assert_eq!(
        (&lavapipe.spv_sha256, &lavapipe.candidate_output_sha256),
        (&radv.spv_sha256, &radv.candidate_output_sha256)
    );
    let input = C3Input {
        cases: vec![bounded.clone()],
        metals: vec![metal(&bounded, &METAL)],
        vulkan: vec![lavapipe.clone(), radv.clone()],
        public_air: [bounded.air_sha256.clone()].into(),
        ..C3Input::default()
    };
    let mut list = ReviewedList {
        readings: vec![reading(&bounded, vec![reviewed(&lavapipe, &METAL)])],
        old_mismatches: vec![],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    let report = report_of(
        &input,
        &list,
        &C1List::default(),
        &three_venues(),
        &VenueKeys::default(),
    );
    assert_eq!(
        verdict_of(slot(&report, LAVAPIPE, Leg::Stated), &bounded),
        Verdict::Reviewed
    );
    assert_eq!(
        verdict_of(slot(&report, RADV, Leg::Stated), &bounded),
        Verdict::Mismatch
    );
    assert!(!report.holds());
    list.readings[0].observations.push(reviewed(&radv, &METAL));
    let report = report_of(
        &input,
        &list,
        &C1List::default(),
        &three_venues(),
        &VenueKeys::default(),
    );
    assert_eq!(
        verdict_of(slot(&report, RADV, Leg::Stated), &bounded),
        Verdict::Reviewed
    );
}

/// `row` recording, in its one executed module, the precision notes `notes` and the unbounded
/// class they derive, in the leg its census recorded.
fn unbounded_by(row: CandidateObservation, notes: &[PrecisionNote]) -> CandidateObservation {
    let leg = row.census.as_ref().and_then(|census| census.leg);
    let mut row = noting(row, notes);
    if let Some(census) = row.census.as_mut() {
        census.leg = leg;
    }
    row.comparison = ComparisonClass::Unbounded(notes.iter().copied().collect());
    assert!(row.class_holds());
    row
}

/// A row left unbounded by `HalfOutputFastMathBlend` (a fast-math function's `half` color output
/// into a blend Metal's compiler fuses and regroups per pipeline) shows the fast-math regrouping
/// cause, as one left unbounded by `FastMathRewrites` does; one left unbounded by
/// `HalfOutputPrecision` alone (a module without fast math) does not.
#[test]
fn a_fast_math_blend_of_a_half_output_shows_the_regrouping_cause() {
    let blend = case(1, "half_blend");
    let row = |notes: &[PrecisionNote]| {
        unbounded_by(
            new_row(&blend, Backend::Moltenvk, MOLTENVK, &DEVICE, &METAL),
            notes,
        )
    };
    let regrouping = Cause::FastMathRegrouping;
    assert!(regrouping.shown_by(&row(&[PrecisionNote::HalfOutputFastMathBlend])));
    assert!(regrouping.shown_by(&row(&[PrecisionNote::FastMathRewrites])));
    assert!(!regrouping.shown_by(&row(&[PrecisionNote::HalfOutputPrecision])));
    assert!(!Cause::AppleReading.shown_by(&row(&[PrecisionNote::HalfOutputFastMathBlend])));
}

/// A `fast_math_regrouping` reading is keyed by its cause. It reviews its case's current mismatch
/// in every venue's slots, both legs, while the row records `FastMathRewrites`, at whatever
/// candidate and words; a row that stops recording it is an open mismatch, and the reviewed
/// observation in that row's slot reads changed, as it does once the row matches Metal. A deferred
/// near-equivalent stays open beside it, and an `apple_reading` stays keyed by candidate.
#[test]
fn a_regrouping_reading_is_keyed_by_its_cause_in_every_slot() {
    const OTHER: [u8; 4] = [2, 0, 0x80, 0x3f];
    let regroup = [PrecisionNote::FastMathRewrites];
    let fft = case(1, "fft");
    let mut rows = [
        new_row(&fft, Backend::Moltenvk, MOLTENVK, &DEVICE, &METAL),
        new_row(&fft, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL),
        new_row(&fft, Backend::Vulkan, RADV, &OTHER, &METAL),
        in_leg(
            new_row(&fft, Backend::Vulkan, RADV, &DEVICE, &METAL),
            Leg::RobustnessOff,
        ),
    ]
    .map(|row| unbounded_by(row, &regroup));
    for (at, row) in rows.iter_mut().enumerate() {
        row.spv_sha256 = format!("{:02x}", 0x60 + at).repeat(32);
    }
    let input_of = |rows: &[CandidateObservation]| C3Input {
        cases: vec![fft.clone()],
        metals: vec![metal(&fft, &METAL)],
        moltenvk: rows[..1].to_vec(),
        vulkan: rows[1..].to_vec(),
        public_air: [fft.air_sha256.clone()].into(),
        ..C3Input::default()
    };
    let slots = [
        (MOLTENVK, Leg::Stated),
        (LAVAPIPE, Leg::Stated),
        (RADV, Leg::Stated),
        (RADV, Leg::RobustnessOff),
    ];
    let verdicts =
        |report: &C3Report| slots.map(|(venue, leg)| verdict_of(slot(report, venue, leg), &fft));
    let judge = |input: &C3Input, list: &ReviewedList| {
        report_of(
            input,
            list,
            &C1List::default(),
            &three_venues(),
            &VenueKeys::default(),
        )
    };
    let input = input_of(&rows);

    // Unlisted, every slot's row is an open mismatch.
    let report = judge(&input, &ReviewedList::default());
    assert_eq!(verdicts(&report), [Verdict::Mismatch; 4]);

    // The reading's one observation is the lavapipe slot's row at an earlier candidate and output:
    // keyed by its cause, it reviews every slot's row that records the cause, and it holds.
    let mut earlier = reviewed(&rows[1], &METAL);
    earlier.spv_sha256 = "99".repeat(32);
    earlier.candidate_output_sha256 = sha256_bytes(&OTHER);
    let mut regrouping = reading(&fft, vec![earlier]);
    regrouping.cause = Cause::FastMathRegrouping;
    regrouping.lanes.clear();
    let list = ReviewedList {
        readings: vec![regrouping],
        old_mismatches: vec![],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    assert_eq!(list.check(), Ok(()));
    let report = judge(&input, &list);
    assert_eq!(verdicts(&report), [Verdict::Reviewed; 4]);
    assert_eq!(report.readings.len(), 1);
    assert_eq!(report.readings[0].state, ReadingState::Holds);
    assert_eq!((report.signed_off(), report.awaiting_sign_off()), (1, 0));
    assert!(report.holds());

    // A row that no longer records the cause is an open mismatch in its own slot alone.
    let mut pinned = rows.clone();
    pinned[2] = new_row(&fft, Backend::Vulkan, RADV, &OTHER, &METAL);
    let report = judge(&input_of(&pinned), &list);
    assert_eq!(
        verdicts(&report),
        [
            Verdict::Reviewed,
            Verdict::Reviewed,
            Verdict::Mismatch,
            Verdict::Reviewed
        ]
    );
    assert_eq!(report.readings[0].state, ReadingState::Holds);
    assert!(!report.holds());

    // In the reviewed observation's own slot, the reading reads changed too: review it again.
    let mut pinned = rows.clone();
    pinned[1] = new_row(&fft, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    let report = judge(&input_of(&pinned), &list);
    assert_eq!(
        verdicts(&report),
        [
            Verdict::Reviewed,
            Verdict::Mismatch,
            Verdict::Reviewed,
            Verdict::Reviewed
        ]
    );
    assert_eq!(report.readings[0].state, ReadingState::Changed);
    assert!(!report.holds());

    // A row that now matches Metal is a match, and the reading no longer describes its slot.
    let mut matched = rows.clone();
    matched[1] = unbounded_by(
        new_row(&fft, Backend::Vulkan, LAVAPIPE, &METAL, &METAL),
        &regroup,
    );
    let report = judge(&input_of(&matched), &list);
    assert_eq!(
        verdicts(&report),
        [
            Verdict::Reviewed,
            Verdict::Match,
            Verdict::Reviewed,
            Verdict::Reviewed
        ]
    );
    assert_eq!(report.readings[0].state, ReadingState::Changed);

    // A deferred near-equivalent beside the cause keeps its row open with its reason.
    let mut deferred = rows.clone();
    deferred[2] = unbounded_by(
        new_row(&fft, Backend::Vulkan, RADV, &OTHER, &METAL),
        &[
            PrecisionNote::FastMathRewrites,
            PrecisionNote::SparseResidency,
        ],
    );
    let report = judge(&input_of(&deferred), &list);
    let stated = slot(&report, RADV, Leg::Stated);
    assert_eq!(verdict_of(stated, &fft), Verdict::Mismatch);
    assert_eq!(
        stated.rows[0].reason,
        Some(OpenReason::NearEquivalent(PrecisionNote::SparseResidency))
    );

    // An apple_reading of the same rows stays keyed by candidate: its observation's slot alone.
    let bounded = ReviewedList {
        readings: vec![reading(&fft, vec![reviewed(&rows[1], &METAL)])],
        ..list.clone()
    };
    let report = judge(&input, &bounded);
    assert_eq!(
        verdicts(&report),
        [
            Verdict::Mismatch,
            Verdict::Reviewed,
            Verdict::Mismatch,
            Verdict::Mismatch
        ]
    );
}

#[test]
fn venue_lists_are_checked() {
    assert_eq!(three_venues().check(), Ok(()));
    type Edit = fn(&mut Venues);
    let edits: [(Edit, &str); 15] = [
        (|venues| venues.venues.clear(), "no venue"),
        (
            |venues| venues.venues[2].environment_id = MOLTENVK.into(),
            "listed twice",
        ),
        (
            |venues| venues.venues[2].environment_id = "radv@engine".into(),
            "has no whitespace",
        ),
        (
            |venues| venues.venues[2].profile = "radv-guess".into(),
            "is not a committed profile",
        ),
        (|venues| venues.venues[2].legs.clear(), "lists no leg"),
        (
            |venues| venues.venues[2].legs = vec![Leg::Stated, Leg::Stated],
            "lists a leg twice",
        ),
        (
            |venues| venues.venues[2].devices[0].identity.driver_info = " ".into(),
            "needs a device name",
        ),
        (
            |venues| venues.venues[2].devices[0].identity.driver_id = 14,
            "a MoltenVK venue runs MoltenVK",
        ),
        (
            |venues| venues.venues[0].devices[0].identity.driver_id = 3,
            "a MoltenVK venue runs MoltenVK",
        ),
        // The allowed-device list.
        (
            |venues| venues.venues[2].devices.clear(),
            "allows no device",
        ),
        (
            |venues| venues.venues[2].profile = "gate-lavapipe".into(),
            "is none of its devices' captures",
        ),
        (
            |venues| {
                let again = venues.venues[2].devices[0].clone();
                venues.venues[2].devices.push(again);
            },
            "twice",
        ),
        (
            |venues| {
                let mut other = venues.venues[2].devices[0].clone();
                other.identity.device_id += 1;
                venues.venues[2].devices.push(other);
            },
            "twice",
        ),
        (
            |venues| {
                let mut other = venues.venues[2].devices[0].clone();
                other.identity.device_id += 1;
                other.profile = "radv-guess".into();
                venues.venues[2].devices.push(other);
            },
            "is not a committed profile",
        ),
        (
            // Two baselines on one backend: lavapipe's and the GPU's.
            |venues| {
                venues.venues[1].population = Population::OldEngineMatched;
                venues.venues[2].population = Population::OldEngineMatched;
            },
            "a second venue holds the old engine's baseline",
        ),
    ];
    for (edit, why) in edits {
        let mut venues = three_venues();
        edit(&mut venues);
        let error = venues.check().unwrap_err();
        assert!(error.contains(why), "{why}: {error}");
    }
    // A venue entry with an unknown field is not read.
    let mut json = serde_json::to_value(three_venues()).unwrap();
    json["venues"][2]["host"] = "homelab".into();
    assert!(Venues::parse(&json.to_string()).is_err());
}

#[test]
fn the_committed_venue_list_parses_and_names_a_real_gpu_of_each_vendor() {
    let path = Venues::committed_path();
    assert!(path.ends_with("cutover/c3-venues.json"));
    let venues = Venues::load(&path).unwrap();
    let gpus = venues
        .venues
        .iter()
        .filter(|venue| venue.kind == VenueKind::NativeGpu)
        .collect::<Vec<_>>();
    assert_eq!(gpus.len(), 3);
    for gpu in gpus {
        assert_eq!(gpu.population, Population::MetalQualified);
        assert_eq!(gpu.legs, [Leg::Stated, Leg::RobustnessOff]);
    }
    let json = serde_json::to_string_pretty(&venues).unwrap() + "\n";
    assert_eq!(json, std::fs::read_to_string(&path).unwrap());
}

/// A venue's profile is a `vulkaninfo` capture of the venue's own device, and every device a venue
/// allows comes with a capture of it that derives the venue's translation key: every committed
/// venue's devices' captures state the identities the venue declares and derive its key, and the
/// list refuses a census profile of another machine, a capture of another device, a device
/// translation tells apart from the venue's, and a driver update declared without its recapture.
#[test]
fn a_venue_profile_is_a_capture_of_the_venues_own_device() {
    let committed = Venues::load(&Venues::committed_path()).unwrap();
    committed.check_profiles().unwrap();
    let refusal = |venue: &str, edit: &dyn Fn(&mut Venue)| {
        let mut venues = committed.clone();
        let found = venues
            .venues
            .iter_mut()
            .find(|candidate| candidate.environment_id == venue)
            .unwrap();
        edit(found);
        let error = venues.check_profiles().unwrap_err();
        assert!(error.contains(venue), "{error}");
        error
    };
    // A gpuinfo report of another card.
    let error = refusal("amd-780m-radv", &|venue| {
        venue.profile = "amd-rdna3-radv".into()
    });
    assert!(error.contains("this one's source is gpuinfo"), "{error}");
    // The arm64 gate's capture under the x86 venue: the same driver, another device.
    let error = refusal("ubuntu-26.04-lavapipe-x86_64", &|venue| {
        venue.devices[0].profile = "gate-lavapipe".into();
    });
    assert!(
        error.contains(
            "device: llvmpipe (LLVM 21.1.8, 256 bits) (llvmpipe (LLVM 21.1.8, 128 bits))"
        ),
        "{error}"
    );
    // The Intel venue allows the RPL-P and the RPL-S, each with its own capture; either capture
    // under the other device is another device's.
    let intel = committed
        .venues
        .iter()
        .find(|venue| venue.environment_id == "intel-irisxe-rplp-anv")
        .unwrap();
    assert_eq!(
        intel
            .devices
            .iter()
            .map(|device| (device.profile.as_str(), device.identity.device_id))
            .collect::<Vec<_>>(),
        [
            ("intel-irisxe-rplp-anv", 0xa7a0),
            ("intel-uhd770-rpls-anv", 0xa780)
        ]
    );
    let error = refusal("intel-irisxe-rplp-anv", &|venue| {
        venue.devices[1].profile = "intel-irisxe-rplp-anv".into();
    });
    assert!(
        error.contains("device: Intel(R) Graphics (RPL-S) (Intel(R) Iris(R) Xe Graphics (RPL-P))")
            && error.contains("device_id: 42880 (42912)"),
        "{error}"
    );
    // A device translation tells apart: its own capture, another key.
    let error = refusal("amd-780m-radv", &|venue| {
        let nvidia = VenueProfile::committed("nvidia-rtx5080-laptop").unwrap();
        venue.devices.push(VenueDevice {
            profile: nvidia.stem,
            identity: nvidia.identity,
            metal_environment_id: None,
        });
    });
    assert!(
        error.contains("derives translation key") && error.contains("needs a venue of its own"),
        "{error}"
    );
    // A driver update declared without recapturing the profile.
    let error = refusal("intel-irisxe-rplp-anv", &|venue| {
        venue.devices[0].identity.driver_info = "Mesa 26.2.5-arch1.1".into();
    });
    assert!(
        error.contains("driver_info: Mesa 26.2.5-arch1.1 (Mesa 26.2.4-arch3.1)"),
        "{error}"
    );
}

/// Each committed venue keys its native engine slot in both legs, for its own backend only,
/// with its committed profile's translation key, which a record holds only by recording it; a venue
/// is found by its operator environment and backend.
#[test]
fn each_venue_keys_its_slots_with_its_profiles_key() {
    let committed = Venues::load(&Venues::committed_path()).unwrap();
    let keys = committed.keys(CurrentArtifacts::default()).unwrap();
    for venue in &committed.venues {
        let profile = VenueProfile::committed(&venue.profile).unwrap();
        assert_eq!(
            VenueProfile::key_of(&profile.description).unwrap(),
            profile.fingerprint
        );
        let backend = venue.backend();
        let other = match backend {
            Backend::Moltenvk => Backend::Vulkan,
            Backend::Vulkan => Backend::Moltenvk,
        };
        for leg in [Leg::Stated, Leg::RobustnessOff] {
            let slot = engine_slot(&venue.environment_id, None, leg);
            assert_eq!(
                keys.expected(backend, &slot),
                Some(profile.fingerprint.as_str())
            );
            assert_eq!(keys.expected(other, &slot), None);
            let mut environment = serde_json::json!({});
            assert!(!keys.holds(backend, &slot, &environment));
            assert!(keys.holds(other, &slot, &environment));
            environment[VENUE_PROFILE] = profile.record();
            assert!(keys.holds(backend, &slot, &environment));
            assert_eq!(
                VenueProfile::recorded_stem(&environment),
                Some(venue.profile.as_str())
            );
        }
        assert_eq!(keys.expected(backend, &venue.environment_id), None);
        assert_eq!(
            committed.venue_of(&venue.environment_id, backend),
            Some(venue)
        );
        assert_eq!(committed.venue_of(&venue.environment_id, other), None);
    }
}

/// Each answer gap is decided from each venue's committed profile. Of the committed venues, arm64
/// lavapipe alone runs fragment subgroups of fewer than two quads (its 4-lane subgroups); x86
/// lavapipe's 8, the GPUs' and MoltenVK's (which pins no fragment size and runs 32) hold two or
/// more. Every venue but MoltenVK, which draws through Metal, has the clipped-primitive split. The
/// venue keys record each for both legs' slots.
#[test]
fn each_answer_gap_is_decided_from_each_venues_profile() {
    let committed = Venues::load(&Venues::committed_path()).unwrap();
    let keys = committed.keys(CurrentArtifacts::default()).unwrap();
    let all: BTreeSet<&str> = committed
        .venues
        .iter()
        .map(|venue| venue.profile.as_str())
        .collect();
    for (gap, expected) in [
        (
            AnswerGap::FragmentSubgroupUnderTwoQuads,
            BTreeSet::from(["gate-lavapipe"]),
        ),
        (
            AnswerGap::ClippedPrimitiveSplit,
            all.iter()
                .copied()
                .filter(|stem| !stem.starts_with("moltenvk-"))
                .collect(),
        ),
        (
            AnswerGap::QuadsSplitByClipping,
            BTreeSet::from(["gate-lavapipe", "gate-lavapipe-x86_64"]),
        ),
    ] {
        let mut with = BTreeSet::new();
        for venue in &committed.venues {
            let profile = VenueProfile::committed(&venue.profile).unwrap();
            let derived = crate::acceptance::derived_profile(profile.description.clone()).unwrap();
            let has = gap.has(&profile.description, &derived);
            if has {
                with.insert(venue.profile.as_str());
            }
            for leg in [Leg::Stated, Leg::RobustnessOff] {
                let slot = engine_slot(&venue.environment_id, None, leg);
                assert_eq!(keys.has_gap(&slot, gap), has, "{slot} {}", gap.as_str());
            }
            assert!(!keys.has_gap(&venue.environment_id, gap));
        }
        assert_eq!(with, expected, "{}", gap.as_str());
    }
}

/// The reason a judged case's row in `slot` names, if any.
fn reason_of(slot: &SlotReport, case: &AuthoredCase) -> Option<OpenReason> {
    slot.rows
        .iter()
        .find(|row| row.case_id == case.case_id)
        .and_then(|row| row.reason.clone())
}

/// A case the list holds under the subgroup gap stays a mismatch where its row mismatches in a
/// slot whose device has the gap, its reason naming the gap: answer gaps stay open until an exact
/// emulation exists (the user's decision of 2026-10-06), so none counts toward sign-offs and the
/// list refuses one signed off. The same mismatch where the device lacks the gap, or of a case the
/// list does not hold, names no reason.
#[test]
fn a_subgroup_gap_names_a_mismatch_only_where_the_device_has_it() {
    let counted = case(1, "counted");
    let other = case(2, "other");
    let mut input = C3Input {
        cases: vec![counted.clone(), other.clone()],
        metals: vec![metal(&counted, &METAL), metal(&other, &METAL)],
        public_air: [counted.air_sha256.clone(), other.air_sha256.clone()].into(),
        ..C3Input::default()
    };
    input.vulkan.push(new_row(
        &counted,
        Backend::Vulkan,
        LAVAPIPE,
        &DEVICE,
        &METAL,
    ));
    input
        .vulkan
        .push(new_row(&other, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL));
    let gap = AnswerGap::FragmentSubgroupUnderTwoQuads;
    let list = ReviewedList {
        answer_gaps: vec![AnswerGapEntry {
            case_id: counted.case_id.clone(),
            name: counted.name.clone(),
            gap,
            review: Review {
                sign_off: SignOff::AwaitingUser {
                    decision: "open until an exact emulation exists".into(),
                },
                ..signed("a fragment subgroup of one quad")
            },
        }],
        ..ReviewedList::default()
    };
    assert_eq!(list.check(), Ok(()));
    let mut keys = VenueKeys::default();
    let judged = |input: &C3Input, keys: &VenueKeys, list: &ReviewedList| {
        report_of(input, list, &C1List::default(), &venues(), keys)
    };

    // The device lacks the gap: both mismatch, naming no reason.
    let report = judged(&input, &keys, &list);
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(stated, &counted), Verdict::Mismatch);
    assert_eq!(reason_of(stated, &counted), None);
    assert_eq!(verdict_of(stated, &other), Verdict::Mismatch);
    assert_eq!(report.answer_gaps[0].slots, 0);

    // The device has it: the listed case is still an open mismatch, its reason the gap; nothing
    // awaits or holds a sign-off, and neither the slot nor C3 holds.
    keys.insert_gaps(LAVAPIPE, &BTreeSet::from([gap]));
    let report = judged(&input, &keys, &list);
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(stated, &counted), Verdict::Mismatch);
    assert_eq!(
        reason_of(stated, &counted),
        Some(OpenReason::AnswerGap(gap))
    );
    assert_eq!(
        OpenReason::AnswerGap(gap).label(),
        "deferred{answer_gap:fragment_subgroup_under_two_quads}"
    );
    assert_eq!(verdict_of(stated, &other), Verdict::Mismatch);
    assert_eq!(reason_of(stated, &other), None);
    assert_eq!(report.answer_gaps.len(), 1);
    assert_eq!(report.answer_gaps[0].slots, 1);
    assert_eq!((report.signed_off(), report.awaiting_sign_off()), (0, 0));
    assert!(!stated.holds());
    assert!(!report.holds());

    // The other case matching, the listed case alone keeps the slot open.
    input.vulkan.retain(|row| row.case_id != other.case_id);
    input
        .vulkan
        .push(new_row(&other, Backend::Vulkan, LAVAPIPE, &METAL, &METAL));
    let report = judged(&input, &keys, &list);
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert!(!stated.holds());
    assert_eq!(stated.count(Verdict::Mismatch), 1);

    // A signed-off answer gap is refused, and so is a case listed twice.
    let mut accepted = list.clone();
    accepted.answer_gaps[0].review = signed("a fragment subgroup of one quad");
    assert!(accepted.check().unwrap_err().contains("stays open"));
    let mut twice = list.clone();
    twice.answer_gaps.push(twice.answer_gaps[0].clone());
    assert!(twice.check().unwrap_err().contains("listed twice"));
}

/// Structural facts for the tests: each case's answer gaps by ID, unknown for any other case.
struct StructuralGaps(BTreeMap<String, BTreeSet<AnswerGap>>);

impl reference::CaseFacts for StructuralGaps {
    fn answer_gaps(&self, case: &AuthoredCase) -> Option<BTreeSet<AnswerGap>> {
        self.0.get(&case.case_id).cloned()
    }
}

/// A case the list holds under the clipped-primitive split names the gap as the reason of its open
/// mismatch in a slot whose device has the gap only when its structure meets the gap: a case whose
/// structure is unknown, or does not meet it, names no reason.
#[test]
fn the_clipped_primitive_split_names_a_mismatch_only_where_the_case_meets_it() {
    let split = case(1, "split");
    let input = C3Input {
        cases: vec![split.clone()],
        metals: vec![metal(&split, &METAL)],
        public_air: [split.air_sha256.clone()].into(),
        vulkan: vec![new_row(&split, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL)],
        ..C3Input::default()
    };
    let gap = AnswerGap::ClippedPrimitiveSplit;
    assert!(gap.decided_by_case());
    let list = ReviewedList {
        answer_gaps: vec![AnswerGapEntry {
            case_id: split.case_id.clone(),
            name: split.name.clone(),
            gap,
            review: Review {
                sign_off: SignOff::AwaitingUser {
                    decision: "open until an exact emulation exists".into(),
                },
                ..signed("a clipped triangle's pieces")
            },
        }],
        ..ReviewedList::default()
    };
    assert_eq!(list.check(), Ok(()));
    let mut keys = VenueKeys::default();
    keys.insert_gaps(LAVAPIPE, &BTreeSet::from([gap]));
    let judged = |facts: &StructuralGaps| {
        let report = report(
            &input,
            &list,
            &C1List::default(),
            &venues(),
            &keys,
            &reference::testing::references(),
            facts,
        );
        let stated = slot(&report, LAVAPIPE, Leg::Stated);
        assert_eq!(verdict_of(stated, &split), Verdict::Mismatch);
        reason_of(stated, &split)
    };
    let met = |gaps: &[AnswerGap]| {
        StructuralGaps(BTreeMap::from([(
            split.case_id.clone(),
            gaps.iter().copied().collect(),
        )]))
    };
    assert_eq!(judged(&StructuralGaps(BTreeMap::new())), None);
    assert_eq!(judged(&met(&[])), None);
    assert_eq!(
        judged(&met(&[AnswerGap::FragmentSubgroupUnderTwoQuads])),
        None
    );
    assert_eq!(judged(&met(&[gap])), Some(OpenReason::AnswerGap(gap)));
    // A device without the gap judges the case as any other.
    let without = VenueKeys::default();
    let report = report(
        &input,
        &list,
        &C1List::default(),
        &venues(),
        &without,
        &reference::testing::references(),
        &met(&[gap]),
    );
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    assert_eq!(verdict_of(stated, &split), Verdict::Mismatch);
    assert_eq!(reason_of(stated, &split), None);
}

/// Every committed case whose structure meets the clipped-primitive split is listed under it, and
/// every case listed under it meets it: a multisampled fragment case drawn by the generated
/// full-screen triangle whose function reads `[[sample_mask]]`. The committed cases are all over
/// public fixtures, so the check needs no private store.
#[test]
fn the_cases_that_meet_the_clipped_primitive_split_are_listed() {
    let root = crate::source::corpus_root();
    let cases = crate::store::CorpusStore::new(&root)
        .read_all_cases()
        .unwrap();
    let gap = AnswerGap::ClippedPrimitiveSplit;
    let mut meeting = BTreeSet::new();
    for case in &cases {
        // `met_by`'s first terms, read before the case is checked.
        if case.stage != Stage::Fragment || case.raster_sample_count.unwrap_or(1) < 2 {
            continue;
        }
        let checked = crate::check::check_case(&root, case.clone())
            .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.name));
        if gap.met_by(&checked) {
            meeting.insert(case.case_id.as_str());
        }
    }
    let list = ReviewedList::load(&ReviewedList::committed_path()).unwrap();
    let listed: BTreeSet<&str> = list
        .answer_gaps
        .iter()
        .filter(|entry| entry.gap == gap)
        .map(|entry| entry.case_id.as_str())
        .collect();
    assert_eq!(meeting, listed);
    assert!(meeting.len() >= 5, "{meeting:?}");
}

/// The identity a profile document states, from whichever capability block holds each structure,
/// and only from a `vulkaninfo` capture.
#[test]
fn a_venue_profile_states_its_identity() {
    let document = |source: &str, driver: &str| {
        serde_json::json!({
            "provenance": {"source": source},
            "capabilities": {
                "device": {"properties": {
                    "VkPhysicalDeviceProperties": {
                        "deviceName": "Toy GPU", "vendorID": 4098, "deviceID": 5567,
                        "driverVersion": 109060100,
                    },
                }},
                "driver": {"properties": {
                    driver: {"driverID": "VK_DRIVER_ID_MESA_RADV", "driverInfo": "Mesa 26.2.4"},
                }},
            },
            "profiles": {"VP_M2V_toy": {
                "api-version": "1.3.250", "capabilities": ["device", "driver"],
            }},
        })
        .to_string()
    };
    let expected = DeviceIdentity {
        device: "Toy GPU".into(),
        vendor_id: 4098,
        device_id: 5567,
        driver_id: 3,
        driver_version: 109060100,
        driver_info: "Mesa 26.2.4".into(),
    };
    for driver in [
        "VkPhysicalDeviceVulkan12Properties",
        "VkPhysicalDeviceDriverProperties",
    ] {
        let profile =
            VenueProfile::parse("toy", document("vulkaninfo", driver).as_bytes()).unwrap();
        assert_eq!(profile.stem, "toy");
        assert_eq!(profile.identity, expected);
        assert!(profile.identity.differences(&expected).is_empty());
    }
    let error = VenueProfile::parse(
        "toy",
        document("gpuinfo", "VkPhysicalDeviceDriverProperties").as_bytes(),
    )
    .unwrap_err();
    assert!(error.contains("this one's source is gpuinfo"), "{error}");
    let mut other = expected.clone();
    other.driver_version = 1;
    assert_eq!(
        expected.differences(&other),
        ["driver_version: 109060100 (1)"]
    );
}

/// A current refusal of `case` in `environment`'s slot of `leg`, as the executor records it after
/// the device opened.
fn refusal_of(
    case: &AuthoredCase,
    environment: &str,
    leg: Leg,
    refused: &crate::check::admission::NotSupported,
) -> RefusalObservation {
    let mut recorded = recorded_environment(environment);
    recorded["robustness"] = serde_json::json!({ "leg": leg });
    RefusalObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: case.computed_input_sha256().unwrap(),
        translator_fingerprint: engine_fingerprint_of(&PROFILE),
        backend: Backend::Vulkan,
        environment_id: engine_slot(environment, None, leg),
        environment: recorded,
        executor_abi: ENGINE_EXECUTOR_ABI.into(),
        refusal: Refusal::of(refused),
        artifact: None,
    }
}

/// The product's typed refusal of a capability the device lacks.
fn lacks_ordered_access() -> crate::check::admission::NotSupported {
    crate::check::admission::NotSupported::Product {
        outcome: crate::engine::Outcome::DeviceCapabilityMissing {
            feature: "FramebufferFetch".into(),
            capability: "Feature(RasterizationOrderColorAttachmentAccess)".into(),
        },
        detail: "no ordered attachment access".into(),
        at_bind: false,
    }
}

/// C1's entry for `profile`'s bucket of [`lacks_ordered_access`], in `class`, with `sign_off`.
fn c1_entry(
    profile: &str,
    class: crate::c1::reviewed::ListClass,
    sign_off: SignOff,
) -> crate::c1::reviewed::Entry {
    crate::c1::reviewed::Entry {
        profile: profile.into(),
        outcome: "device_capability_missing".into(),
        key: "FramebufferFetch needs Feature(RasterizationOrderColorAttachmentAccess)".into(),
        class,
        rows: 1,
        rows_digest: "11".repeat(32),
        new_census: "22".repeat(32),
        old_l3: "33".repeat(32),
        review: Review {
            evidence: vec!["a probe".into()],
            note: "the device has no ordered attachment access".into(),
            sign_off,
        },
    }
}

/// A case the venue's device refused reads `refused`, never `missing`, and keeps its slot open:
/// the refusal is judged current by the rule a row is (the case's inputs, the executor ABI, the
/// translator fingerprint of the recorded profile) and on the venue's declared device, so a
/// refusal of another build or another driver is stale. A slot's row wins over a refusal.
#[test]
fn a_refused_case_reads_refused_never_missing() {
    use crate::check::admission::NotSupported;
    let refused = case(1, "refused");
    let bound = case(2, "bound");
    let mut input = C3Input {
        cases: vec![refused.clone(), bound.clone()],
        metals: vec![metal(&refused, &METAL), metal(&bound, &METAL)],
        ..C3Input::default()
    };
    input.refusals = vec![
        refusal_of(
            &refused,
            RADV,
            Leg::Stated,
            &NotSupported::NotEnabled(vec!["rayQuery".into()]),
        ),
        refusal_of(
            &bound,
            RADV,
            Leg::Stated,
            &NotSupported::Bind("a range past maxStorageBufferRange".into()),
        ),
    ];
    let judge = |input: &C3Input| {
        report_of(
            input,
            &ReviewedList::default(),
            &C1List::default(),
            &three_venues(),
            &VenueKeys::default(),
        )
    };
    let report = judge(&input);
    let stated = slot(&report, RADV, Leg::Stated);
    assert_eq!(verdict_of(stated, &refused), Verdict::Refused);
    assert_eq!(verdict_of(stated, &bound), Verdict::Refused);
    assert_eq!(stated.count(Verdict::Refused), 2);
    assert_eq!(stated.count(Verdict::Missing), 0);
    assert!(!stated.holds());
    let row = stated
        .rows
        .iter()
        .find(|row| row.case_id == refused.case_id)
        .unwrap();
    assert_eq!(row.refusal.as_ref().unwrap().class, RefusalClass::Device);
    // Neither is a product device-capability refusal: no C1 entry is proposed.
    assert!(report.gap_proposals.is_empty() && report.device_gaps.is_empty());
    // The robustness-off leg recorded nothing: missing there.
    assert_eq!(
        verdict_of(slot(&report, RADV, Leg::RobustnessOff), &refused),
        Verdict::Missing
    );

    // Another build, other inputs, or another driver: stale.
    type Edit = fn(&mut RefusalObservation);
    let edits: [(Edit, bool); 4] = [
        (
            |refusal| refusal.executor_abi = "vulkan-engine-v1".into(),
            false,
        ),
        (
            |refusal| refusal.translator_fingerprint = engine_fingerprint_of(&[8; 32]),
            false,
        ),
        (|refusal| refusal.input_sha256 = "55".repeat(32), false),
        (
            |refusal| refusal.environment["driver_version"] = 109060101.into(),
            true,
        ),
    ];
    for (edit, other_device) in edits {
        let mut stale = input.clone();
        edit(&mut stale.refusals[0]);
        let report = judge(&stale);
        let stated = slot(&report, RADV, Leg::Stated);
        assert_eq!(verdict_of(stated, &refused), Verdict::Stale);
        assert_eq!(stated.stale_device, usize::from(other_device));
    }

    // A row in the slot is judged, not a refusal beside it.
    let mut ran = input.clone();
    ran.vulkan = vec![new_row(&refused, Backend::Vulkan, RADV, &METAL, &METAL)];
    assert_eq!(
        verdict_of(slot(&judge(&ran), RADV, Leg::Stated), &refused),
        Verdict::Match
    );
}

/// The product's refusal of a capability the venue's device lacks is a device gap exactly when
/// C1's reviewed list holds its bucket for the venue's profile as `device_lacks` or
/// `near_equivalent`: the slot holds, and the C1 entry's sign-off is one C3 awaits or holds.
/// Without such an entry the case is refused and the entry is proposed; an entry of another
/// profile, or one that calls the old output wrong, is not the venue's device gap.
#[test]
fn a_device_gap_reads_c1s_entry_and_its_sign_off() {
    use crate::c1::reviewed::ListClass;
    let fetch = case(1, "fetch");
    let input = C3Input {
        cases: vec![fetch.clone()],
        metals: vec![metal(&fetch, &METAL)],
        refusals: vec![
            refusal_of(&fetch, RADV, Leg::Stated, &lacks_ordered_access()),
            refusal_of(&fetch, RADV, Leg::RobustnessOff, &lacks_ordered_access()),
        ],
        ..C3Input::default()
    };
    let awaiting = SignOff::AwaitingUser {
        decision: "U11".into(),
    };
    let judge = |c1: Vec<crate::c1::reviewed::Entry>| {
        report_of(
            &input,
            &ReviewedList::default(),
            &C1List { entries: c1 },
            &three_venues(),
            &VenueKeys::default(),
        )
    };

    for class in [ListClass::DeviceLacks, ListClass::NearEquivalent] {
        let report = judge(vec![c1_entry("amd-rdna3-radv", class, awaiting.clone())]);
        for leg in [Leg::Stated, Leg::RobustnessOff] {
            let slot = slot(&report, RADV, leg);
            assert_eq!(verdict_of(slot, &fetch), Verdict::DeviceGap);
            assert!(slot.holds());
        }
        assert_eq!(
            report.device_gaps,
            [DeviceGapRow {
                profile: "amd-rdna3-radv".into(),
                outcome: "device_capability_missing".into(),
                key: "FramebufferFetch needs Feature(RasterizationOrderColorAttachmentAccess)"
                    .into(),
                class,
                sign_off: awaiting.clone(),
                cases: 2,
            }]
        );
        assert!(report.gap_proposals.is_empty());
        assert_eq!(report.awaiting_sign_off(), 1);
        assert!(!report.holds());
    }

    let signed = SignOff::SignedOff {
        decision: "U11".into(),
        date: "2026-10-03".into(),
    };
    let report = judge(vec![c1_entry(
        "amd-rdna3-radv",
        ListClass::DeviceLacks,
        signed,
    )]);
    assert_eq!((report.signed_off(), report.awaiting_sign_off()), (1, 0));

    for (entries, proposed) in [
        (vec![], true),
        (
            vec![c1_entry(
                "moltenvk-m3",
                ListClass::DeviceLacks,
                awaiting.clone(),
            )],
            true,
        ),
        (
            vec![c1_entry(
                "amd-rdna3-radv",
                ListClass::OldOutputWrong,
                awaiting.clone(),
            )],
            false,
        ),
    ] {
        let report = judge(entries);
        let stated = slot(&report, RADV, Leg::Stated);
        assert_eq!(verdict_of(stated, &fetch), Verdict::Refused);
        assert!(!stated.holds());
        assert!(report.device_gaps.is_empty());
        let expected = proposed.then(|| GapProposal {
            profile: "amd-rdna3-radv".into(),
            outcome: "device_capability_missing".into(),
            key: "FramebufferFetch needs Feature(RasterizationOrderColorAttachmentAccess)".into(),
            cases: 2,
        });
        assert_eq!(report.gap_proposals, Vec::from_iter(expected));
    }
}

/// A current product refusal of ray tracing, by its device capabilities or by its feature, is out
/// of the launch's scope: counted apart, never refused work, a device gap or a proposed C1 entry,
/// and it does not keep the slot from holding. A refusal of another build is still stale.
#[test]
fn a_ray_tracing_refusal_is_out_of_scope_never_refused_or_a_gap() {
    use crate::check::admission::NotSupported;
    use crate::cutover::scope::OutOfScope;
    use crate::engine::{Outcome, UnsupportedReason};
    let query = case(1, "query");
    let intersect = case(2, "intersect");
    let lacks_ray_query = NotSupported::Product {
        outcome: Outcome::DeviceCapabilityMissing {
            feature: "Lowering(Access)".into(),
            capability: "Feature(RayQuery)+Feature(AccelerationStructure)".into(),
        },
        detail: "no ray queries".into(),
        at_bind: false,
    };
    let not_representable = NotSupported::Product {
        outcome: Outcome::Unsupported {
            reason: UnsupportedReason::NotRepresentable,
            feature: "RayTracing".into(),
        },
        detail: "intersection functions".into(),
        at_bind: false,
    };
    let mut input = C3Input {
        cases: vec![query.clone(), intersect.clone()],
        metals: vec![metal(&query, &METAL), metal(&intersect, &METAL)],
        ..C3Input::default()
    };
    for leg in [Leg::Stated, Leg::RobustnessOff] {
        input
            .refusals
            .push(refusal_of(&query, RADV, leg, &lacks_ray_query));
        input
            .refusals
            .push(refusal_of(&intersect, RADV, leg, &not_representable));
    }
    let judge = |input: &C3Input| {
        report_of(
            input,
            &ReviewedList::default(),
            &C1List::default(),
            &three_venues(),
            &VenueKeys::default(),
        )
    };
    let report = judge(&input);
    let out_of_scope = Verdict::OutOfScope(OutOfScope::RayTracing);
    assert_eq!(out_of_scope.as_str(), "out_of_scope_ray_tracing");
    for leg in [Leg::Stated, Leg::RobustnessOff] {
        let slot = slot(&report, RADV, leg);
        assert_eq!(verdict_of(slot, &query), out_of_scope);
        assert_eq!(verdict_of(slot, &intersect), out_of_scope);
        assert_eq!(slot.count(out_of_scope), 2);
        assert_eq!(slot.count(Verdict::Refused), 0);
        assert!(slot.holds());
        // The slot lists each with the refusal it holds.
        assert!(slot
            .rows
            .iter()
            .all(|row| row.verdict == out_of_scope && row.refusal.is_some()));
    }
    assert!(report.gap_proposals.is_empty() && report.device_gaps.is_empty());

    // The same refusal of another build is stale, and keeps the slot open.
    let mut stale = input.clone();
    stale.refusals[0].translator_fingerprint = engine_fingerprint_of(&[8; 32]);
    let report = judge(&stale);
    let stated = slot(&report, RADV, Leg::Stated);
    assert_eq!(verdict_of(stated, &query), Verdict::Stale);
    assert!(!stated.holds());
}

/// The user's decision of 2026-10-03: a row that decoded a rasterization-rate map (the decoder's
/// recorded near-equivalent) stays open, with its measured bound as the reason, even when the list
/// reviews it; a case whose rate map the product refuses is missing with the refusal as its reason.
/// Neither holds C3, and an ordinary mismatch carries no reason.
#[test]
fn a_deferred_near_equivalent_stays_open_with_its_reason() {
    use crate::case::{RateMapLayerLiteral, RateMapLiteral, RateSample};
    use crate::coverage::{ExecutedCensus, ModuleCensus};
    let decoded = case(1, "decoded");
    let mut row = new_row(&decoded, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    row.census = Some(ExecutedCensus {
        out_of_bounds: None,
        rate_map_pass: None,
        device: None,
        product: crate::observation::TRANSLATOR_FINGERPRINT.into(),
        restricted_to: None,
        leg: None,
        modules: vec![ModuleCensus {
            stage: crate::case::Stage::Kernel,
            subgroup: crate::coverage::SubgroupShape {
                full_subgroups: false,
                required_size: None,
            },
            strategies: Default::default(),
            precision: [PrecisionNote::RasterizationRateMap].into(),
            containment: None,
        }],
        artifact: None,
    });
    row.comparison = ComparisonClass::Unbounded([PrecisionNote::RasterizationRateMap].into());
    assert!(row.class_holds());
    let mut refused = case(2, "refused");
    refused.buffers[0].rasterization_rate_map = Some(RateMapLiteral {
        screen_size: [70_000, 1],
        layers: vec![RateMapLayerLiteral {
            horizontal: vec![RateSample(1.0)],
            vertical: vec![RateSample(1.0)],
        }],
    });
    refused.case_id = refused.computed_case_id().unwrap();
    let input = C3Input {
        cases: vec![decoded.clone(), refused.clone()],
        metals: vec![metal(&decoded, &METAL), metal(&refused, &METAL)],
        public_air: [decoded.air_sha256.clone(), refused.air_sha256.clone()].into(),
        vulkan: vec![row.clone()],
        ..C3Input::default()
    };
    // Even a signed-off reading at the row's candidate does not review it.
    let list = ReviewedList {
        readings: vec![reading(&decoded, vec![reviewed(&row, &METAL)])],
        old_mismatches: vec![],
        reference_gaps: vec![],
        answer_gaps: vec![],
    };
    let report = report_of(
        &input,
        &list,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    let open = |case: &AuthoredCase| {
        stated
            .rows
            .iter()
            .find(|row| row.case_id == case.case_id)
            .map(|row| (row.verdict, row.reason.clone()))
    };
    let deferred = OpenReason::NearEquivalent(PrecisionNote::RasterizationRateMap);
    assert_eq!(
        open(&decoded),
        Some((Verdict::Mismatch, Some(deferred.clone())))
    );
    assert!(
        deferred.label().starts_with(
            "deferred{near_equivalent:RasterizationRateMap:tile starts at integers where \
                 Apple M3's are fractional"
        ),
        "{}",
        deferred.label()
    );
    let (verdict, reason) = open(&refused).unwrap();
    assert_eq!(verdict, Verdict::Missing);
    assert!(
        matches!(&reason, Some(OpenReason::RateMapRefused(why)) if why.contains("70000")),
        "{reason:?}"
    );
    assert!(!stated.holds());
    assert!(!report.holds());
    // The same row without the decoder's note is an ordinary mismatch, which the list reviews.
    let mut plain = input.clone();
    plain.vulkan[0].census = None;
    plain.vulkan[0].comparison = ComparisonClass::Exact;
    let mut list = list;
    list.readings[0].observations = vec![reviewed(&plain.vulkan[0], &METAL)];
    let report = report_of(
        &plain,
        &list,
        &C1List::default(),
        &venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, LAVAPIPE, Leg::Stated);
    let row = stated
        .rows
        .iter()
        .find(|r| r.case_id == decoded.case_id)
        .unwrap();
    assert_eq!((row.verdict, row.reason.clone()), (Verdict::Reviewed, None));
}

/// A deferred near-equivalent the run announced as a degradation, which no executed module's notes
/// record (the runtime makes a rate-mapped pass's map and announces it), keeps the row open with
/// its reason, as a note its comparison class records does, even where the list holds a reading at
/// it; a cost, a note C3 judges word by word, or a code the vocabulary does not know leaves an
/// ordinary mismatch, which the list reviews.
#[test]
fn an_announced_deferred_near_equivalent_stays_open_with_its_reason() {
    use crate::observation::{EXECUTOR_DEGRADES, RUNTIME_DEGRADES};
    use metal2vulkan_runtime::Degradation;
    let drawn = case(1, "drawn");
    let row = new_row(&drawn, Backend::Vulkan, LAVAPIPE, &DEVICE, &METAL);
    assert_eq!(
        (row.comparison.clone(), row.census.is_none()),
        (ComparisonClass::Exact, true)
    );
    let announcing = |member: &str, codes: &[&str]| {
        let mut row = row.clone();
        row.environment[member] = serde_json::json!(codes);
        row
    };
    let judged = |row: CandidateObservation| {
        let input = C3Input {
            cases: vec![drawn.clone()],
            metals: vec![metal(&drawn, &METAL)],
            public_air: [drawn.air_sha256.clone()].into(),
            vulkan: vec![row.clone()],
            ..C3Input::default()
        };
        let list = ReviewedList {
            readings: vec![reading(&drawn, vec![reviewed(&row, &METAL)])],
            old_mismatches: vec![],
            reference_gaps: vec![],
            answer_gaps: vec![],
        };
        let report = report_of(
            &input,
            &list,
            &C1List::default(),
            &venues(),
            &VenueKeys::default(),
        );
        slot(&report, LAVAPIPE, Leg::Stated)
            .rows
            .iter()
            .find(|judged| judged.case_id == drawn.case_id)
            .map(|judged| (judged.verdict, judged.reason.clone()))
    };
    let rate_map = Degradation::RasterizationRateMap.code();
    assert_eq!(rate_map, "metal2vulkan-degrade-rasterization-rate-map");
    let deferred = Some((
        Verdict::Mismatch,
        Some(OpenReason::NearEquivalent(
            PrecisionNote::RasterizationRateMap,
        )),
    ));
    let cost = Degradation::ShadowedTexture.code();
    assert_eq!(
        judged(announcing(RUNTIME_DEGRADES, &[cost, rate_map])),
        deferred
    );
    // The platform executor's own notices are the same vocabulary, read alike.
    assert_eq!(judged(announcing(EXECUTOR_DEGRADES, &[rate_map])), deferred);
    let ordinary = Some((Verdict::Reviewed, None));
    assert_eq!(judged(row.clone()), ordinary);
    assert_eq!(
        judged(announcing(
            RUNTIME_DEGRADES,
            &[
                cost,
                Degradation::DeviceDenormals.code(),
                "metal2vulkan-degrade-no-such-kind",
            ],
        )),
        ordinary
    );
}

/// A run restricted to a profile the device cannot take is a current device-class refusal of the
/// restricted slot (`not_supported{profile_exceeds_device}`, naming each member), judged by the
/// rule a row is; C3 judges the venue's own slots, so it neither reads that slot's refusal as
/// missing or failed work there nor as refused work of the venue.
#[test]
fn a_refused_restriction_is_a_device_refusal_of_its_restricted_slot_only() {
    use crate::check::admission::NotSupported;
    let refused = case(1, "restricted");
    let refusal = RefusalObservation {
        environment_id: engine_slot(RADV, Some("lunarg-desktop-baseline-2023"), Leg::Stated),
        ..refusal_of(
            &refused,
            RADV,
            Leg::Stated,
            &NotSupported::Restriction {
                profile: "lunarg-desktop-baseline-2023".into(),
                gaps: vec![
                    "limits.max_compute_work_group_invocations 1536 beyond the device's 1024"
                        .into(),
                ],
            },
        )
    };
    refusal.validate_content().unwrap();
    assert_eq!(refusal.refusal.class, RefusalClass::Device);
    assert!(crate::index::refusal_is_current(
        &refused,
        Backend::Vulkan,
        &refusal,
        &VenueKeys::default()
    ));
    let input = C3Input {
        cases: vec![refused.clone()],
        metals: vec![metal(&refused, &METAL)],
        vulkan: vec![new_row(&refused, Backend::Vulkan, RADV, &METAL, &METAL)],
        refusals: vec![refusal],
        ..C3Input::default()
    };
    let report = report_of(
        &input,
        &ReviewedList::default(),
        &C1List::default(),
        &three_venues(),
        &VenueKeys::default(),
    );
    let stated = slot(&report, RADV, Leg::Stated);
    assert_eq!(verdict_of(stated, &refused), Verdict::Match);
    for verdict in [
        Verdict::Refused,
        Verdict::Missing,
        Verdict::Mismatch,
        Verdict::Stale,
    ] {
        assert_eq!(stated.count(verdict), 0, "{verdict:?}");
    }
    assert!(report
        .slots
        .iter()
        .all(|slot| !slot.slot.contains("lunarg-desktop-baseline-2023")));
    assert!(report.gap_proposals.is_empty() && report.device_gaps.is_empty());
}

/// RADV in its stated leg, which must cover every case with a Metal reference row.
fn metal_qualified() -> Venues {
    Venues {
        venues: vec![venue(
            VenueKind::NativeGpu,
            RADV,
            Population::MetalQualified,
            &[Leg::Stated],
            "amd-rdna3-radv",
        )],
    }
}

/// The report over `input` against [`metal_qualified`], with the synthetic reference.
fn judged_by(input: &C3Input) -> C3Report {
    report(
        input,
        &ReviewedList::default(),
        &C1List::default(),
        &metal_qualified(),
        &VenueKeys::default(),
        &reference::testing::references(),
        &reference::testing::NoFacts,
    )
}

/// A reference gap is read from the reference alone, from a row that recorded its declared
/// identity: an environment the list does not name, or a row of another identity, is no reference.
#[test]
fn a_reference_gap_is_read_from_the_reference_alone() {
    let primary = reference::testing::primary();
    let unlisted = ReferenceEnvironment {
        environment_id: "other-gpu-macos3.0".into(),
        identity: reference::testing::identity("Other GPU", "3.0"),
    };
    let impostor = ReferenceEnvironment {
        environment_id: primary.environment_id.clone(),
        identity: reference::testing::identity("Other GPU", "1.0"),
    };
    let gap = |environment: &ReferenceEnvironment, case: &AuthoredCase| MetalObservation {
        metal_output_sha256: sha256_bytes(&[]),
        output_b64: String::new(),
        status: MetalStatus::ReferenceGap(crate::observation::ReferenceGap {
            apple_family: 10,
            message: "supported by GPUFamily10 and later".into(),
        }),
        ..metal_in(environment, case, &METAL)
    };
    let on_primary = case(1, "reference gap");
    let on_unlisted = case(3, "unlisted gap");
    let of_impostor = case(4, "gap of another identity");
    let input = C3Input {
        cases: vec![on_primary.clone(), on_unlisted.clone(), of_impostor.clone()],
        metals: vec![
            gap(&primary, &on_primary),
            gap(&unlisted, &on_unlisted),
            gap(&impostor, &of_impostor),
        ],
        ..C3Input::default()
    };
    let report = judged_by(&input);
    let gaps = report
        .reference_gaps
        .iter()
        .map(|gap| (gap.case_id.as_str(), gap.recorded))
        .collect::<Vec<_>>();
    assert_eq!(gaps, [(on_primary.case_id.as_str(), Some(10))]);
}

/// A rekey list moves venue records only to this build's keys: its targets are the committed
/// profiles as this build reads them (`corpus-status --venue-keys`), each from-key another key,
/// none listed twice; and it matches a record by profile, file and key exactly.
#[test]
fn a_rekey_list_moves_records_only_to_this_builds_keys() {
    let venues = Venues::load(&Venues::committed_path()).unwrap();
    let keys = venues.committed_keys().unwrap();
    assert_eq!(keys.len(), venues.venues.len());
    let (stem, now) = keys
        .iter()
        .find(|(stem, _)| stem == "gate-lavapipe")
        .cloned()
        .unwrap();
    let old = RecordedKey {
        sha256: now.sha256.clone(),
        fingerprint: "00".repeat(32),
    };
    let rekey = VenueRekey {
        stem: stem.clone(),
        from: old.clone(),
        to: now.clone(),
    };
    VenueRekey::check(std::slice::from_ref(&rekey)).unwrap();
    for (rekeys, why) in [
        (
            vec![VenueRekey {
                to: old.clone(),
                ..rekey.clone()
            }],
            "not this build's committed profile",
        ),
        (
            vec![VenueRekey {
                from: now.clone(),
                ..rekey.clone()
            }],
            "the same key",
        ),
        (vec![rekey.clone(), rekey.clone()], "listed twice"),
        (
            vec![VenueRekey {
                stem: "no-such-profile".into(),
                ..rekey.clone()
            }],
            "no-such-profile",
        ),
    ] {
        let error = VenueRekey::check(&rekeys).unwrap_err();
        assert!(error.contains(why), "{why}: {error}");
    }
    let recorded = |stem: &str, key: &RecordedKey| {
        serde_json::json!({
            VENUE_PROFILE: {"profile": stem, "sha256": key.sha256, "fingerprint": key.fingerprint}
        })
    };
    let mut environment = recorded(&stem, &old);
    assert!(rekey.matches(&environment));
    assert!(!rekey.matches(&recorded("moltenvk-m3", &old)));
    assert!(!rekey.matches(&recorded(
        &stem,
        &RecordedKey {
            sha256: "ff".repeat(32),
            ..old.clone()
        }
    )));
    assert!(!rekey.matches(&serde_json::json!({})));
    rekey.apply(&mut environment);
    assert_eq!(environment, recorded(&stem, &now));
    assert!(!rekey.matches(&environment));
}

/// The Apple M3 Max's MoltenVK runs (`apple-m3-max-moltenvk`) compare with its own Metal rows: a
/// run compared with another Metal environment allows no device, and a device stating no Metal
/// environment is allowed under any. The list refuses a second GPU's device (the M3 Max's capture
/// under another GPU's name, in a venue of its own) naming the M3 Max's Metal environment, and a
/// device whose Metal environment is no C3 reference environment.
#[test]
fn each_moltenvk_device_compares_with_its_own_gpus_metal() {
    let committed = Venues::load(&Venues::committed_path()).unwrap();
    let moltenvk = committed
        .venues
        .iter()
        .filter(|venue| venue.kind == VenueKind::Moltenvk)
        .collect::<Vec<_>>();
    let devices = |venue: &Venue| {
        venue
            .devices
            .iter()
            .map(|device| {
                (
                    device.profile.clone(),
                    device.identity.device_id,
                    device.metal_environment_id.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        moltenvk
            .iter()
            .map(|venue| (venue.environment_id.as_str(), devices(venue)))
            .collect::<Vec<_>>(),
        [(
            "apple-m3-max-moltenvk",
            vec![(
                "moltenvk-m3".to_string(),
                452985353,
                Some("apple-m3-max-macos27.0.1".to_string())
            )]
        )]
    );
    let m3 = moltenvk[0];
    assert_eq!(m3.population, Population::OldEngineMatched);
    let names = |venue: &Venue, metal: &str| {
        venue
            .identities_compared_with(metal)
            .into_iter()
            .map(|identity| identity.device)
            .collect::<Vec<_>>()
    };
    assert_eq!(names(m3, "apple-m3-max-macos27.0.1"), ["Apple M3 Max"]);
    assert_eq!(names(m3, "apple-m3-max-macos26.7"), Vec::<String>::new());
    assert_eq!(names(m3, "apple-m3-max-macos26.5.2"), Vec::<String>::new());
    let mut stating_none = m3.clone();
    stating_none.devices[0].metal_environment_id = None;
    assert_eq!(
        names(&stating_none, "apple-m3-max-macos26.5.2"),
        ["Apple M3 Max"]
    );
    let mut m4_device = m3.devices[0].clone();
    m4_device.identity.device = "Apple M4".into();
    // A venue of its own whose device names the M3 Max's Metal environment.
    let mut own = committed.clone();
    let mut venue = m3.clone();
    venue.environment_id = "apple-m4-moltenvk".into();
    venue.devices = vec![m4_device];
    own.venues.push(venue);
    let error = own.check().unwrap_err();
    assert!(
        error.contains("which is another GPU's (Apple M3 Max)"),
        "{error}"
    );
    let mut unknown = committed.clone();
    unknown
        .venues
        .iter_mut()
        .find(|venue| venue.environment_id == "apple-m3-max-moltenvk")
        .unwrap()
        .devices[0]
        .metal_environment_id = Some("apple-m3-max-macos26.0".into());
    let error = unknown.check().unwrap_err();
    assert!(
        error.contains("which is no C3 reference environment"),
        "{error}"
    );
}

/// The split quads' structure: an entry meets it through a quad or simdgroup operation over its
/// active lanes, by the product's decode of the call's `air.*` name, and not through a shuffle or a
/// broadcast from a named lane, which reads a value a helper computes alike, nor through a
/// declaration alone.
#[test]
fn an_operation_over_the_active_lanes_meets_the_split_quads() {
    let call = |name: &str| format!("  %7 = tail call i16 @{name}() #3\n");
    for name in [
        "air.quad_active_threads_mask",
        "air.quad_all",
        "air.quad_ballot",
        "air.quad_is_first",
        "air.quad_max.u.i32",
        "air.simd_sum.s.i32",
    ] {
        assert!(super::reads_active_lanes(&call(name)), "{name}");
    }
    for name in [
        "air.quad_shuffle_xor.u.i32",
        "air.quad_broadcast.u.i32",
        "air.quad_shuffle_rotate_down.u.i32",
        "air.convert.f.f32.s.i32",
    ] {
        assert!(!super::reads_active_lanes(&call(name)), "{name}");
    }
    assert!(!super::reads_active_lanes(
        "declare i16 @air.quad_active_threads_mask() local_unnamed_addr #1\n"
    ));
}
