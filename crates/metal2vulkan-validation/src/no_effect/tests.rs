use super::*;
use crate::case::{BufferResource, ResourceRole};
use crate::metal::ORACLE_ABI;
use crate::store::CorpusStore;
use std::collections::HashMap;

/// A case whose output is bytes 4..8 of an in/out buffer authored as `0..16`.
fn case(no_effect: Option<&str>) -> AuthoredCase {
    let mut case = crate::case::attachmentless_fragment_test_case("air".into(), "f".into());
    case.buffers = vec![BufferResource {
        binding: 0,
        role: ResourceRole::InOut,
        bytes_b64: None,
        initial_bytes_b64: Some(
            base64::engine::general_purpose::STANDARD.encode((0..16).collect::<Vec<u8>>()),
        ),
        rasterization_rate_map: None,
    }];
    case.output = OutputSelection::Buffer {
        binding: 0,
        offset: 4,
        length: 4,
    };
    case.no_effect = no_effect.map(str::to_string);
    case
}

fn row(case: &AuthoredCase, output: &[u8], status: MetalStatus) -> MetalObservation {
    MetalObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: String::new(),
        metal_output_sha256: String::new(),
        output_b64: base64::engine::general_purpose::STANDARD.encode(output),
        environment_id: "metal".into(),
        environment: serde_json::Value::Null,
        oracle_abi: ORACLE_ABI.into(),
        status,
    }
}

/// The four ways an answer meets a declaration, judged from the selected bytes alone: an answer
/// equal to them stands only declared, and a declaration stands only while the answer equals
/// them.
#[test]
fn an_answer_equal_to_the_authored_output_stands_only_when_declared() {
    let unchanged = [4, 5, 6, 7];
    let written = [4, 5, 6, 0];
    let undeclared = case(None);
    let declared = case(Some("the parameters select an empty extent"));
    let qualified = |case: &AuthoredCase, output: &[u8]| {
        judge_row(case, &row(case, output, MetalStatus::Qualified))
    };
    let refusal = qualified(&undeclared, &unchanged).unwrap_err();
    assert!(
        refusal.contains("exactly as authored (4 bytes)"),
        "{refusal}"
    );
    assert_eq!(qualified(&declared, &unchanged), Ok(()));
    assert_eq!(qualified(&undeclared, &written), Ok(()));
    let refusal = qualified(&declared, &written).unwrap_err();
    assert!(
        refusal.contains("Metal changed the selected output"),
        "{refusal}"
    );
    // A row with no output (a reference gap) has nothing to judge.
    let gap = MetalStatus::ReferenceGap(crate::observation::ReferenceGap {
        apple_family: 10,
        message: "GPUFamily10".into(),
    });
    assert_eq!(judge_row(&undeclared, &row(&undeclared, &[], gap)), Ok(()));
    // An empty selection observes nothing, so equality says nothing.
    assert_eq!(judge(&undeclared, &[], &[]), Ok(()));
}

#[test]
fn a_declaration_is_a_bounded_reason_over_a_selected_output() {
    assert!(declaration_errors(&case(None)).is_empty());
    assert!(declaration_errors(&case(Some("an early exit"))).is_empty());
    for reason in ["", " padded", &"x".repeat(MAX_REASON_BYTES + 1)] {
        assert_eq!(
            declaration_errors(&case(Some(reason))).len(),
            1,
            "{reason:?}"
        );
    }
    let mut none = case(Some("an early exit"));
    none.output = OutputSelection::None;
    assert_eq!(
        declaration_errors(&none),
        vec!["no_effect declares an output left as authored, but the case selects none"]
    );
}

/// Every current qualified Metal row in the committed store agrees with its case's declaration,
/// and every declaration is well formed.
#[test]
fn every_committed_metal_answer_agrees_with_its_cases_declaration() {
    let store = CorpusStore::new(metal2vulkan_devtools::validation_dir().join("corpus"));
    let cases = store.read_all_cases().unwrap();
    let by_id = cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<HashMap<_, _>>();
    let mut failures = cases
        .iter()
        .flat_map(|case| {
            declaration_errors(case)
                .into_iter()
                .map(move |error| format!("{} {}: {error}", case.case_id, case.name))
        })
        .collect::<Vec<_>>();
    let mut judged = 0;
    for row in store.read_metal().unwrap() {
        if row.oracle_abi != ORACLE_ABI {
            continue;
        }
        let case = by_id[row.case_id.as_str()];
        judged += 1;
        if let Err(error) = judge_row(case, &row) {
            failures.push(format!("{} {}: {error}", case.case_id, case.name));
        }
    }
    assert!(judged > 0, "the committed store holds current Metal rows");
    assert!(
        failures.is_empty(),
        "{} committed case(s) disagree with their no_effect declaration:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
