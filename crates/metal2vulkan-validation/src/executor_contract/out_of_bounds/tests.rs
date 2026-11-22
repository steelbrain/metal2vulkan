use super::*;
use crate::case::out_of_bounds::OutOfBounds;

/// A kernel case reading `src` (binding 0, 16 bytes) out of bounds into `out` (binding 1, 32
/// bytes): bytes 16..32 of the output are the out-of-bounds reads of target 0.
fn case(form: ContainmentForm) -> AuthoredCase {
    let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
    let src: Vec<u8> = (1..=4u32)
        .flat_map(|w| (0xA000_0000 | w).to_le_bytes())
        .collect();
    serde_json::from_value(serde_json::json!({
        "air_sha256": "00".repeat(32),
        "case_id": "11".repeat(32),
        "name": "out-of-bounds",
        "entry": "main",
        "stage": "kernel",
        "buffers": [
            {"binding": 0, "role": "input", "bytes_b64": b64(&src)},
            {"binding": 1, "role": "output", "initial_bytes_b64": b64(&[0xab; 32])},
        ],
        "dispatch": {"grid": [8, 1, 1], "threads_per_threadgroup": [8, 1, 1]},
        "output": {"kind": "buffer", "binding": 1, "offset": 0, "length": 32},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
        "out_of_bounds": {
            "targets": [{
                "resource": {"kind": "buffer", "binding": 0},
                "form": form,
                "reach_after": 256,
            }],
            "reads": [{"offset": 16, "length": 16, "target": 0}],
        },
    }))
    .unwrap()
}

fn run(form: ContainmentForm, medium: Medium, check: Check, written: bool) -> OutOfBoundsRun {
    OutOfBoundsRun {
        targets: vec![TargetRun {
            resource: OutOfBoundsResource::Buffer { binding: 0 },
            form,
            mechanism: Mechanism { medium, check },
            written,
            escaped: Some(0),
            past_end: Some(0),
        }],
    }
}

/// The Metal observation of [`case`]: in-bounds words, then the zeros the oracle records in the
/// read bytes.
fn golden() -> Vec<u8> {
    let mut golden = (1..=4u32)
        .flat_map(|w| (0xA000_0000 | w).to_le_bytes())
        .collect::<Vec<_>>();
    golden.extend([0; 16]);
    golden
}

fn with_reads(words: [u32; 4]) -> Vec<u8> {
    let mut candidate = golden();
    candidate[16..].copy_from_slice(
        &words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    candidate
}

#[test]
fn the_read_rule_follows_the_mechanism_not_the_case() {
    use ContainmentForm as F;
    let rule = |form, medium, check, written| read_rule(form, Mechanism { medium, check }, written);
    // An in-shader check yields zero, for a load and for a skipped atomic, in every medium.
    for medium in [Medium::DescriptorView, Medium::DeviceAddress, Medium::Arena] {
        for form in [F::StorageBuffer, F::BufferAtomic] {
            assert_eq!(rule(form, medium, Check::Shader, true), ReadRule::Zero);
        }
    }
    // A device address and the arena are checked in the shader whatever the leg.
    assert_eq!(
        rule(F::BufferAtomic, Medium::DeviceAddress, Check::Robust2, true),
        ReadRule::Zero
    );
    // Robust buffer access 2: a load is zero, an atomic's return undefined.
    assert_eq!(
        rule(
            F::StorageBuffer,
            Medium::DescriptorView,
            Check::Robust2,
            true
        ),
        ReadRule::Zero
    );
    assert_eq!(
        rule(
            F::BufferAtomic,
            Medium::DescriptorView,
            Check::Robust2,
            true
        ),
        ReadRule::Any
    );
    // Robust buffer access alone: the buffer's own words of a buffer nothing writes.
    assert_eq!(
        rule(
            F::UniformBuffer,
            Medium::DescriptorView,
            Check::Robust,
            false
        ),
        ReadRule::ZeroOrOwnWords
    );
    assert_eq!(
        rule(
            F::StorageBuffer,
            Medium::DescriptorView,
            Check::Robust,
            true
        ),
        ReadRule::Any
    );
}

#[test]
fn a_contained_read_matches_and_an_escaped_one_does_not() {
    let class = ComparisonClass::Exact;
    let form = ContainmentForm::StorageBuffer;
    let checked = run(form, Medium::DescriptorView, Check::Shader, false);
    let case = case(form);
    assert!(matches(&case, &class, &golden(), &golden(), Some(&checked)));
    // A read past the buffer that returned the next buffer's word is not contained.
    let leaked = with_reads([0, 0, 0xC000_0040, 0]);
    assert!(!matches(&case, &class, &golden(), &leaked, Some(&checked)));
    // An in-bounds byte compares with Metal as always.
    let mut wrong = golden();
    wrong[0] ^= 1;
    assert!(!matches(&case, &class, &golden(), &wrong, Some(&checked)));
    // A run without its out-of-bounds facts, or with an escaped guard byte, is no match.
    assert!(!matches(&case, &class, &golden(), &golden(), None));
    let mut escaped = checked.clone();
    escaped.targets[0].escaped = Some(4);
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &golden(),
        Some(&escaped)
    ));
    // A buffer target must have been guarded.
    let mut unguarded = checked.clone();
    unguarded.targets[0].escaped = None;
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &golden(),
        Some(&unguarded)
    ));
    // The run's facts must be of the case's targets.
    let mut other = checked;
    other.targets[0].resource = OutOfBoundsResource::Buffer { binding: 1 };
    assert!(!matches(&case, &class, &golden(), &golden(), Some(&other)));
}

#[test]
fn robust_buffer_access_alone_allows_the_buffers_own_words() {
    let class = ComparisonClass::Exact;
    let form = ContainmentForm::UniformBuffer;
    let robust = run(form, Medium::DescriptorView, Check::Robust, false);
    let case = case(form);
    // Zero, (0, 0, 0, 1.0), and words of the buffer itself are contained.
    let own = with_reads([0xA000_0003, 0, 0, 0x3f80_0000]);
    assert!(matches(&case, &class, &golden(), &own, Some(&robust)));
    // A word of no buffer of the case is not.
    let leaked = with_reads([0xA000_0005, 0, 0, 0]);
    assert!(!matches(&case, &class, &golden(), &leaked, Some(&robust)));
    // Under the second level the read is zero only.
    let robust2 = run(form, Medium::DescriptorView, Check::Robust2, false);
    assert!(!matches(&case, &class, &golden(), &own, Some(&robust2)));
}

#[test]
fn an_atomics_return_under_robustness_is_undefined_but_its_guard_is_not() {
    let class = ComparisonClass::Exact;
    let form = ContainmentForm::BufferAtomic;
    let case = case(form);
    let any = with_reads([0x1234_5678, 0, 0, 0]);
    let robust2 = run(form, Medium::DescriptorView, Check::Robust2, true);
    assert!(matches(&case, &class, &golden(), &any, Some(&robust2)));
    let checked = run(form, Medium::DescriptorView, Check::Shader, true);
    assert!(!matches(&case, &class, &golden(), &any, Some(&checked)));
    let mut escaped = robust2;
    escaped.targets[0].escaped = Some(1);
    assert!(!matches(&case, &class, &golden(), &any, Some(&escaped)));
}

#[test]
fn an_output_written_past_its_end_under_robustness_alone_is_not_compared() {
    let class = ComparisonClass::Exact;
    let mut case = case(ContainmentForm::StorageBuffer);
    // The output is the target: a write past it may change any of its bytes under
    // `robustBufferAccess` alone, so only the guard decides.
    let declaration = case.out_of_bounds.as_mut().unwrap();
    declaration.targets[0].resource = OutOfBoundsResource::Buffer { binding: 1 };
    declaration.reads.clear();
    let mut robust = run(
        ContainmentForm::StorageBuffer,
        Medium::DescriptorView,
        Check::Robust,
        true,
    );
    robust.targets[0].resource = OutOfBoundsResource::Buffer { binding: 1 };
    let mut clobbered = golden();
    clobbered[0] = 0x55;
    assert!(matches(&case, &class, &golden(), &clobbered, Some(&robust)));
    let mut robust2 = robust.clone();
    robust2.targets[0].mechanism.check = Check::Robust2;
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &clobbered,
        Some(&robust2)
    ));
    robust.targets[0].escaped = Some(16);
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &clobbered,
        Some(&robust)
    ));
}

#[test]
fn a_case_without_targets_compares_as_every_case_does() {
    let mut case = case(ContainmentForm::StorageBuffer);
    case.out_of_bounds = None;
    let class = ComparisonClass::Exact;
    assert!(matches(&case, &class, &golden(), &golden(), None));
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &with_reads([1, 0, 0, 0]),
        None
    ));
}

#[test]
fn the_metal_oracle_records_zeros_in_the_read_bytes() {
    let case = case(ContainmentForm::StorageBuffer);
    let recorded = metal_recorded(&case, vec![0xE7; 32]);
    assert_eq!(&recorded[..16], &[0xE7; 16]);
    assert_eq!(&recorded[16..], &[0; 16]);
    let OutOfBounds { targets, .. } = case.out_of_bounds.clone().unwrap();
    assert_eq!(buffer_reach(&case, 0), Some((0, targets[0].reach_after)));
    assert_eq!(buffer_reach(&case, 1), None);
    assert_eq!(threadgroup_reach(&case, 0), 0);
}

/// [`case`] whose target is the texture at binding 0 (`texture_type` `ty`, four `r32_uint`
/// texels), of `form`, with `reach_after`, and no read bytes: Metal defines its out-of-bounds
/// results, so the whole output compares with Metal.
fn texture_case(ty: &str, form: ContainmentForm, reach_after: u32) -> AuthoredCase {
    let mut case = case(form);
    let mut value = serde_json::to_value(&case).unwrap();
    value["textures"] = serde_json::json!([{
        "binding": 0,
        "role": "input",
        "texture_type": ty,
        "format": "r32_uint",
        "dimensions": [4, 1, 1],
        "bytes_b64": base64::engine::general_purpose::STANDARD.encode([0u8; 16]),
    }]);
    case = serde_json::from_value(value).unwrap();
    let declaration = case.out_of_bounds.as_mut().unwrap();
    declaration.targets[0].resource = OutOfBoundsResource::Texture { binding: 0 };
    declaration.targets[0].reach_after = reach_after;
    declaration.reads.clear();
    case
}

/// A texture buffer is host-visible buffer memory, which the candidate guards: a run without a
/// guard count around one, or whose guard changed, is no match although every compared byte
/// equals Metal's (a write past the buffer that the readback never sees). An image's memory is
/// opaque: it has no guard.
#[test]
fn a_texel_buffer_target_is_judged_by_its_guard() {
    let class = ComparisonClass::Exact;
    let case = texture_case("buffer", ContainmentForm::TexelBuffer, 16384);
    let ran = |check, escaped, past_end| OutOfBoundsRun {
        targets: vec![TargetRun {
            resource: OutOfBoundsResource::Texture { binding: 0 },
            form: ContainmentForm::TexelBuffer,
            mechanism: Mechanism {
                medium: Medium::TexelBuffer,
                check,
            },
            written: true,
            escaped,
            past_end,
        }],
    };
    let output = golden();
    let judge = |run: OutOfBoundsRun| matches(&case, &class, &output, &output, Some(&run));
    assert!(judge(ran(Check::Shader, Some(0), Some(0))));
    // No guard around the texture buffer: nothing shows that its writes stayed inside it.
    assert!(!judge(ran(Check::Shader, None, None)));
    assert!(!judge(ran(Check::Robust2, Some(4), Some(0))));
    // The memory bound to it past its end: only robustness 1 may write there.
    assert!(!judge(ran(Check::Shader, Some(0), Some(4))));
    assert!(!judge(ran(Check::Robust2, Some(0), Some(4))));
    assert!(judge(ran(Check::Robust, Some(0), Some(4))));
    assert!(!judge(ran(Check::Robust, Some(0), None)));

    let image = texture_case("d1", ContainmentForm::Texture, 0);
    let image_run = |escaped, past_end| OutOfBoundsRun {
        targets: vec![TargetRun {
            resource: OutOfBoundsResource::Texture { binding: 0 },
            form: ContainmentForm::Texture,
            mechanism: Mechanism {
                medium: Medium::Image,
                check: Check::Shader,
            },
            written: true,
            escaped,
            past_end,
        }],
    };
    let judge = |run: OutOfBoundsRun| matches(&image, &class, &output, &output, Some(&run));
    assert!(judge(image_run(None, None)));
    assert!(!judge(image_run(Some(0), Some(0))));
}

/// A buffer target's memory past its end (its memory requirement's padding) is its own under
/// robustness 1, which may write it and read it back; an in-shader check and robustness 2 keep
/// every access inside the buffer.
#[test]
fn only_robustness_1_may_reach_a_buffers_padding() {
    let class = ComparisonClass::Exact;
    let form = ContainmentForm::StorageBuffer;
    let case = case(form);
    for (check, allowed) in [
        (Check::Shader, false),
        (Check::Robust2, false),
        (Check::Robust, true),
    ] {
        let mut padded = run(form, Medium::DescriptorView, check, false);
        padded.targets[0].past_end = Some(3);
        assert_eq!(
            matches(&case, &class, &golden(), &golden(), Some(&padded)),
            allowed,
            "{check:?}"
        );
    }
    // A robustness-1 read may return the padding's bytes, never the guard's.
    let robust = run(form, Medium::DescriptorView, Check::Robust, false);
    let padding = u32::from_ne_bytes([PAD_FILL; 4]);
    let guard = u32::from_ne_bytes([GUARD_FILL; 4]);
    let read = |word| with_reads([0, word, 0, 0]);
    assert!(matches(
        &case,
        &class,
        &golden(),
        &read(padding),
        Some(&robust)
    ));
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &read(guard),
        Some(&robust)
    ));
}

/// [`case`] whose one target is `resource` of `form`, reading bytes 16..32 of the output; an
/// argument-buffer texture holds `texel` (the texels a handle past a handle array may read).
fn embedded_case(resource: OutOfBoundsResource, form: ContainmentForm, texel: u32) -> AuthoredCase {
    let mut case = case(form);
    let mut value = serde_json::to_value(&case).unwrap();
    value["argument_buffer_textures"] = serde_json::json!([{
        "buffer_binding": 0,
        "field_offset": 0,
        "role": "input",
        "texture_type": "d2",
        "format": "r32_uint",
        "dimensions": [1, 1, 1],
        "bytes_b64": base64::engine::general_purpose::STANDARD.encode(texel.to_le_bytes()),
    }]);
    case = serde_json::from_value(value).unwrap();
    case.out_of_bounds.as_mut().unwrap().targets[0].resource = resource;
    case
}

fn embedded_run(
    resource: OutOfBoundsResource,
    form: ContainmentForm,
    medium: Medium,
    guard: Option<u64>,
) -> OutOfBoundsRun {
    OutOfBoundsRun {
        targets: vec![TargetRun {
            resource,
            form,
            mechanism: Mechanism {
                medium,
                check: Check::Shader,
            },
            written: true,
            escaped: guard,
            past_end: guard,
        }],
    }
}

/// A buffer an argument buffer points to is reached through the loaded pointer by its device
/// address, checked in the shader in every leg: a read past it is zero, and the candidate guards it
/// as a buffer, so a run without guard counts, or whose guard changed, is no match.
#[test]
fn a_loaded_pointers_buffer_reads_zero_and_is_guarded() {
    let class = ComparisonClass::Exact;
    let resource = OutOfBoundsResource::ArgumentBufferBuffer {
        buffer_binding: 0,
        field_offset: 0,
    };
    let form = ContainmentForm::LoadedPointer;
    let case = embedded_case(resource, form, 0);
    let ran = |guard| embedded_run(resource, form, Medium::DeviceAddress, guard);
    assert!(matches(
        &case,
        &class,
        &golden(),
        &golden(),
        Some(&ran(Some(0)))
    ));
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &golden(),
        Some(&ran(None))
    ));
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &golden(),
        Some(&ran(Some(4)))
    ));
    let leaked = with_reads([0, 0xE7E7_E7E7, 0, 0]);
    assert!(!matches(
        &case,
        &class,
        &golden(),
        &leaked,
        Some(&ran(Some(0)))
    ));
    assert_eq!(
        argument_buffer_reach(&case, (0, 0)),
        Some((
            0,
            case.out_of_bounds.as_ref().unwrap().targets[0].reach_after
        ))
    );
    assert_eq!(argument_buffer_reach(&case, (0, 8)), None);
    assert_eq!(buffer_reach(&case, 0), None);
}

/// A thread-local array is a typed object and a handle array lies in its argument buffer: no
/// host memory holds either beside the case's own, so neither has a guard. A read past the array is
/// zero; a read past the handle array, through the handle table or a masked resource id, is a texel
/// of a texture the case binds, or zero (the null handle or descriptor), never another value.
#[test]
fn an_in_layout_target_has_no_guard_and_its_own_read_rule() {
    let class = ComparisonClass::Exact;
    let texel = 0xE200_0000;
    for (resource, form, medium) in [
        (
            OutOfBoundsResource::ThreadMemory,
            ContainmentForm::SizedArray,
            Medium::TypedObject,
        ),
        (
            OutOfBoundsResource::ArgumentBufferTextureArray {
                buffer_binding: 0,
                field_offset: 0,
            },
            ContainmentForm::HandleArrayIndex,
            Medium::HandleTable,
        ),
        (
            OutOfBoundsResource::ArgumentBufferTextureArray {
                buffer_binding: 0,
                field_offset: 0,
            },
            ContainmentForm::HandleArrayIndex,
            Medium::ResourceId,
        ),
    ] {
        let case = embedded_case(resource, form, texel);
        let ran = |guard| embedded_run(resource, form, medium, guard);
        assert!(matches(
            &case,
            &class,
            &golden(),
            &golden(),
            Some(&ran(None))
        ));
        assert!(!matches(
            &case,
            &class,
            &golden(),
            &golden(),
            Some(&ran(Some(0)))
        ));
        let resident = with_reads([texel, 0, texel, 0]);
        assert_eq!(
            matches(&case, &class, &golden(), &resident, Some(&ran(None))),
            medium != Medium::TypedObject,
            "{form}"
        );
        let foreign = with_reads([0xA000_0005, 0, 0, 0]);
        assert!(!matches(
            &case,
            &class,
            &golden(),
            &foreign,
            Some(&ran(None))
        ));
    }
    let rule = |medium, check| {
        read_rule(
            ContainmentForm::SizedArray,
            Mechanism { medium, check },
            true,
        )
    };
    for check in [Check::Shader, Check::Robust, Check::Robust2] {
        assert_eq!(rule(Medium::TypedObject, check), ReadRule::Zero);
        assert_eq!(rule(Medium::HandleTable, check), ReadRule::ResidentTexels);
    }
}

const SPV_SHA256: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const TRANSLATOR: &str = "5555555555555555555555555555555555555555555555555555555555555555";

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// `case`'s Metal row, holding [`golden`].
fn metal_row(case: &AuthoredCase) -> crate::observation::MetalObservation {
    let golden = golden();
    crate::observation::MetalObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: "22".repeat(32),
        metal_output_sha256: crate::hash::sha256_bytes(&golden),
        output_b64: b64(&golden),
        environment_id: "metal-env".into(),
        environment: serde_json::json!({}),
        oracle_abi: "oracle-v1".into(),
        status: crate::observation::MetalStatus::Qualified,
    }
}

/// A candidate row against `metal` of a run with out-of-bounds facts `run` and `output`, under
/// `Exact`.
fn candidate_row(
    metal: &crate::observation::MetalObservation,
    run: OutOfBoundsRun,
    output: &[u8],
    status: crate::observation::CandidateStatus,
) -> crate::observation::CandidateObservation {
    crate::observation::CandidateObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: metal.case_id.clone(),
        air_sha256: metal.air_sha256.clone(),
        input_sha256: metal.input_sha256.clone(),
        golden_output_sha256: metal.metal_output_sha256.clone(),
        spv_sha256: Some(SPV_SHA256.into()),
        translator_fingerprint: TRANSLATOR.into(),
        candidate_output_sha256: crate::hash::sha256_bytes(output),
        output_b64: b64(output),
        environment_id: "vk-env".into(),
        environment: serde_json::json!({}),
        executor_abi: "executor-v1".into(),
        comparison: ComparisonClass::Exact,
        census: Some(crate::coverage::ExecutedCensus {
            device: None,
            product: "p".into(),
            restricted_to: None,
            modules: vec![],
            out_of_bounds: Some(run),
            rate_map_pass: None,
            artifact: None,
        }),
        status,
        held_open_by: None,
    }
}

fn dependencies<'a>(
    case: &'a AuthoredCase,
    metal: &'a crate::observation::MetalObservation,
) -> crate::observation::CandidateDependencies<'a> {
    crate::observation::CandidateDependencies {
        case,
        metal,
        spv_sha256: Some(SPV_SHA256),
        translator_fingerprint: TRANSLATOR,
        environment_id: "vk-env",
        executor_abi: "executor-v1",
    }
}

/// Metal's row records zero in a read's bytes, and a read past a handle array may return a
/// resident texel: a candidate row that read one records a match whose bytes differ from Metal's
/// under `Exact`. Its content is consistent, and its status is re-decided from the Metal bytes; a
/// run whose every read is zero still contradicts such a match on its own.
#[test]
fn a_row_differs_from_metal_only_where_a_target_may_read_a_value() {
    use crate::observation::CandidateStatus;
    let texel = 0xE200_0000;
    let handles = OutOfBoundsResource::ArgumentBufferTextureArray {
        buffer_binding: 0,
        field_offset: 0,
    };
    let case = embedded_case(handles, ContainmentForm::HandleArrayIndex, texel);
    let metal = metal_row(&case);
    let row = |run, output: &[u8], status| candidate_row(&metal, run, output, status);
    let dependencies = dependencies(&case, &metal);
    let table = embedded_run(
        handles,
        ContainmentForm::HandleArrayIndex,
        Medium::HandleTable,
        None,
    );
    assert!(table.tolerates_differences());
    let resident = with_reads([texel, 0, texel, 0]);
    let read = row(table.clone(), &resident, CandidateStatus::Match);
    assert_eq!(read.validate_content(), Ok(()));
    assert!(read.dependency_matches(&dependencies));
    // The status is the containment verdict over the case: a resident texel is no mismatch,
    // and any other value is no match.
    let mismatch = row(table.clone(), &resident, CandidateStatus::Mismatch);
    assert_eq!(mismatch.validate_content(), Ok(()));
    assert!(!mismatch.dependency_matches(&dependencies));
    let foreign = with_reads([0xA000_0005, 0, 0, 0]);
    let claimed = row(table.clone(), &foreign, CandidateStatus::Match);
    assert_eq!(claimed.validate_content(), Ok(()));
    assert!(!claimed.dependency_matches(&dependencies));
    assert!(row(table, &foreign, CandidateStatus::Mismatch).dependency_matches(&dependencies));
    // A typed array's read is zero: differing bytes contradict a match under `Exact`.
    let array = embedded_run(
        OutOfBoundsResource::ThreadMemory,
        ContainmentForm::SizedArray,
        Medium::TypedObject,
        None,
    );
    assert!(!array.tolerates_differences());
    assert!(row(array, &resident, CandidateStatus::Match)
        .validate_content()
        .unwrap_err()
        .contains("contradicts"));
}

/// A write that escapes a loaded pointer's buffer into its guard while the output stays equal to
/// Metal's (`85c504e0`'s output is the buffer itself) is a mismatch row: its content is
/// consistent, and the verdict over the case holds it. A run that changed no guard byte, or only
/// padding robustness 1 may write, still contradicts a mismatch with Metal's bytes.
#[test]
fn an_escape_that_leaves_the_output_unchanged_is_a_mismatch_row() {
    use crate::observation::CandidateStatus;
    let resource = OutOfBoundsResource::ArgumentBufferBuffer {
        buffer_binding: 0,
        field_offset: 0,
    };
    let form = ContainmentForm::LoadedPointer;
    let case = embedded_case(resource, form, 0);
    let metal = metal_row(&case);
    let dependencies = dependencies(&case, &metal);
    let ran = |escaped, past_end, check| {
        let mut run = embedded_run(resource, form, Medium::DeviceAddress, Some(0));
        run.targets[0].escaped = Some(escaped);
        run.targets[0].past_end = Some(past_end);
        run.targets[0].mechanism.check = check;
        run
    };
    for (escaped, past_end) in [(4, 0), (0, 4)] {
        let run = ran(escaped, past_end, Check::Shader);
        assert!(run.escapes());
        let row = candidate_row(&metal, run.clone(), &golden(), CandidateStatus::Mismatch);
        assert_eq!(row.validate_content(), Ok(()));
        assert!(row.dependency_matches(&dependencies));
        let claimed = candidate_row(&metal, run, &golden(), CandidateStatus::Match);
        assert!(!claimed.dependency_matches(&dependencies));
    }
    for run in [ran(0, 0, Check::Shader), ran(0, 4, Check::Robust)] {
        assert!(!run.escapes());
        assert!(
            candidate_row(&metal, run, &golden(), CandidateStatus::Mismatch)
                .validate_content()
                .unwrap_err()
                .contains("contradicts")
        );
    }
}

/// Thread memory is contained when the module keeps it as a typed variable and checks a dynamic
/// index against its array's length; the private arena checks only the arena's bounds, and a
/// module that indexes no thread-local array dynamically exercises nothing. A target in an argument
/// buffer the module does not bind is not accessed.
#[test]
fn thread_memory_is_contained_by_the_typed_arrays_check_only() {
    use metal2vulkan::pipeline::{StrategyCensus, StrategyUse};
    let profiles = crate::coverage::tier_table::committed_profiles().unwrap();
    let profile = &profiles
        .iter()
        .find(|entry| entry.tier.stem == "lavapipe")
        .unwrap()
        .profile;
    let slots = BTreeMap::new();
    let census = |names: &[&str]| {
        let mut census = StrategyCensus::default();
        for name in names {
            census
                .entries
                .insert((*name).to_string(), StrategyUse::default());
        }
        census
    };
    let contain = |resource, names: &[&str]| {
        let strategies = census(names);
        let module = ModuleFacts {
            stage: SymStage::Kernel,
            slots: &slots,
            attributes: &BTreeMap::new(),
            resident_roots: true,
            handle_table: true,
            policy: DevicePolicy::RESIDENT_TABLES,
            strategies: &strategies,
        };
        let target = OutOfBoundsTarget {
            resource,
            form: ContainmentForm::SizedArray,
            reach_before: 0,
            reach_after: 0,
        };
        containment(&target, &module, profile)
    };
    let thread = OutOfBoundsResource::ThreadMemory;
    let typed = contain(
        thread,
        &[
            "access.typed.alloca",
            "access.typed.load.clamped",
            "access.typed.load",
        ],
    )
    .unwrap();
    assert_eq!(
        (typed.form, typed.mechanism.label(), typed.written),
        (
            ContainmentForm::SizedArray,
            "typed-object:shader".into(),
            false
        )
    );
    assert!(
        contain(thread, &["access.typed.store.clamped"])
            .unwrap()
            .written
    );
    assert!(contain(
        thread,
        &["access.typed.load.clamped", "access.private.load.checked"]
    )
    .unwrap_err()
    .contains("private arena"));
    assert!(contain(thread, &["access.typed.load"])
        .unwrap_err()
        .contains("no thread-local array at a dynamic index"));
    for resource in [
        OutOfBoundsResource::ArgumentBufferBuffer {
            buffer_binding: 0,
            field_offset: 0,
        },
        OutOfBoundsResource::ArgumentBufferTextureArray {
            buffer_binding: 0,
            field_offset: 0,
        },
    ] {
        let names = [
            "access.dynamic.load_pointer",
            "access.dynamic.load",
            "access.handle.lookup",
        ];
        assert!(contain(resource, &names)
            .unwrap_err()
            .contains("does not access"));
    }
}

/// Under the identity domains a pointer loaded from an argument buffer is the device address it
/// holds, which nothing checks, so the run is refused before anything is made; an argument
/// buffer's texture array is the descriptor arrays the loaded id's masked index reads, whose read
/// past the array is a texel of a texture the case binds or zero. Under the resident table and the
/// handle table, the same module facts without the table's reads access neither target.
#[test]
fn the_identity_domains_leave_a_loaded_pointer_unchecked_and_mask_a_resource_id() {
    use crate::candidate::engine::out_of_bounds_run;
    use crate::check::admission::{ExecutorError, NotSupported};
    use metal2vulkan::pipeline::{StrategyCensus, StrategyUse};
    let profiles = crate::coverage::tier_table::committed_profiles().unwrap();
    let profile = &profiles
        .iter()
        .find(|entry| entry.tier.stem == "lavapipe")
        .unwrap()
        .profile;
    let mut identity = DevicePolicy::default();
    identity.pointer_domain = PointerDomain::Identity;
    identity.handle_domain = HandleDomain::Identity;
    let slots = BTreeMap::from([(
        SlotRef {
            stage: SymStage::Kernel,
            class: SlotClass::Buffer,
            index: 1,
        },
        // The argument buffer's slot, read through its device address (the interface has no
        // constructor outside the product).
        serde_json::from_value::<SlotInterface>(serde_json::json!({"Buffer": {
            "access": {"read": true, "write": false},
            "window": "Whole",
            "form": "Psb",
            "kind": "Buffer",
        }}))
        .unwrap(),
    )]);
    let mut strategies = StrategyCensus::default();
    for name in [
        "access.dynamic.load_pointer",
        "access.dynamic.load",
        "access.handle.lookup",
    ] {
        strategies
            .entries
            .insert(name.into(), StrategyUse::default());
    }
    let attributes = BTreeMap::new();
    let module = |policy| ModuleFacts {
        stage: SymStage::Kernel,
        slots: &slots,
        attributes: &attributes,
        resident_roots: false,
        handle_table: false,
        policy,
        strategies: &strategies,
    };
    let target = |resource, form| OutOfBoundsTarget {
        resource,
        form,
        reach_before: 0,
        reach_after: 16,
    };
    let pointer = target(
        OutOfBoundsResource::ArgumentBufferBuffer {
            buffer_binding: 1,
            field_offset: 0,
        },
        ContainmentForm::LoadedPointer,
    );
    let handles = target(
        OutOfBoundsResource::ArgumentBufferTextureArray {
            buffer_binding: 1,
            field_offset: 0,
        },
        ContainmentForm::HandleArrayIndex,
    );
    let loaded = containment(&pointer, &module(identity), profile).unwrap();
    assert_eq!(
        (loaded.form, loaded.mechanism.label(), loaded.written),
        (
            ContainmentForm::LoadedPointer,
            "device-address:unchecked".into(),
            false
        )
    );
    let masked = containment(&handles, &module(identity), profile).unwrap();
    assert_eq!(
        (masked.form, masked.mechanism.label()),
        (
            ContainmentForm::HandleArrayIndex,
            "resource-id:shader".into()
        )
    );
    assert_eq!(
        read_rule(ContainmentForm::HandleArrayIndex, masked.mechanism, false),
        ReadRule::ResidentTexels
    );
    assert!(
        containment(&pointer, &module(DevicePolicy::RESIDENT_TABLES), profile)
            .unwrap_err()
            .contains("no memory through a pointer loaded")
    );
    assert!(
        containment(&handles, &module(DevicePolicy::RESIDENT_TABLES), profile)
            .unwrap_err()
            .contains("resolves no handle")
    );
    // The unchecked target refuses the run, naming its form; the masked one runs.
    match out_of_bounds_run(profile, &module(identity), &[pointer, handles.clone()]) {
        Err(ExecutorError::NotSupported(refused @ NotSupported::Uncontained { .. })) => {
            assert_eq!(
                refused.outcome(),
                "not_supported{uncontained:unchecked loaded_pointer}"
            );
            assert!(
                refused.to_string().contains("checks the accesses"),
                "{refused}"
            );
        }
        other => panic!("{other:?}"),
    }
    let run = out_of_bounds_run(profile, &module(identity), &[handles])
        .unwrap()
        .unwrap();
    assert_eq!(run.targets[0].mechanism, masked.mechanism);
}

/// A draw's targets: a fixed-function attribute is unchecked on every device (no buffer
/// robustness bounds it); a direct indexed draw's index buffer is contained by the padded copy of
/// its indices on every device; a top-level texture array by the clamped handle-array load. A pulled attribute is
/// contained as the slot its fetch names, which a module that does not bind it does not access.
#[test]
fn a_draws_targets_are_contained_by_their_fetch() {
    use metal2vulkan::pipeline::{StrategyCensus, StrategyUse};
    let profiles = crate::coverage::tier_table::committed_profiles().unwrap();
    let slots = BTreeMap::new();
    let mut strategies = StrategyCensus::default();
    let contain = |profile: &DeviceProfile,
                   resource,
                   form,
                   stage,
                   attributes: &BTreeMap<u32, Fetch>,
                   strategies: &StrategyCensus| {
        let module = ModuleFacts {
            stage,
            slots: &slots,
            attributes,
            resident_roots: false,
            handle_table: false,
            policy: DevicePolicy::RESIDENT_TABLES,
            strategies,
        };
        let target = OutOfBoundsTarget {
            resource,
            form,
            reach_before: 0,
            reach_after: 16,
        };
        containment(&target, &module, profile)
    };
    let none = BTreeMap::new();
    let fixed = BTreeMap::from([(0, Fetch::FixedFunction)]);
    let pulled = BTreeMap::from([(0, Fetch::Pulled { buffer_index: 30 })]);
    let attribute = OutOfBoundsResource::AttributeBuffer { location: 0 };
    for committed in &profiles {
        let profile = &committed.profile;
        let label = &committed.tier.stem;
        let vertex = |resource, form, attributes, strategies| {
            contain(
                profile,
                resource,
                form,
                SymStage::Vertex,
                attributes,
                strategies,
            )
        };
        let contained = vertex(attribute, ContainmentForm::VertexInput, &fixed, &strategies)
            .unwrap_or_else(|why| panic!("{label}: {why}"));
        assert_eq!(contained.mechanism.medium, Medium::VertexInput, "{label}");
        assert_eq!(contained.mechanism.check, Check::Unchecked, "{label}");
        assert!(!contained.written, "{label}");
        assert!(
            vertex(
                attribute,
                ContainmentForm::VertexInput,
                &pulled,
                &strategies
            )
            .unwrap_err()
            .contains("does not access"),
            "{label}"
        );
        assert!(
            vertex(attribute, ContainmentForm::VertexInput, &none, &strategies)
                .unwrap_err()
                .contains("fetches no attribute 0")
        );
        let index = vertex(
            OutOfBoundsResource::IndexBuffer,
            ContainmentForm::IndexBuffer,
            &none,
            &strategies,
        )
        .unwrap();
        assert_eq!(
            index.mechanism,
            Mechanism {
                medium: Medium::IndexBuffer,
                check: Check::Padded,
            },
            "{label}"
        );
        // A read past it is zero.
        assert_eq!(
            read_rule(ContainmentForm::IndexBuffer, index.mechanism, false),
            ReadRule::Zero
        );
        assert!(contain(
            profile,
            OutOfBoundsResource::IndexBuffer,
            ContainmentForm::IndexBuffer,
            SymStage::Kernel,
            &BTreeMap::new(),
            &strategies,
        )
        .is_err());
    }
    // A top-level texture array's target needs its slot and the clamped load.
    let lavapipe = &profiles
        .iter()
        .find(|entry| entry.tier.stem == "lavapipe")
        .unwrap()
        .profile;
    let array = OutOfBoundsResource::TextureArray { binding: 0 };
    assert!(contain(
        lavapipe,
        array,
        ContainmentForm::HandleArrayIndex,
        SymStage::Kernel,
        &BTreeMap::new(),
        &strategies,
    )
    .unwrap_err()
    .contains("does not access"));
    strategies.entries.insert(
        "access.handle.array_load.clamped".into(),
        StrategyUse::default(),
    );
    let handle_array = Mechanism {
        medium: Medium::HandleArray,
        check: Check::Shader,
    };
    assert_eq!(handle_array.label(), "handle-array:shader");
    assert_eq!(
        read_rule(ContainmentForm::HandleArrayIndex, handle_array, false),
        ReadRule::ResidentTexels
    );
}
