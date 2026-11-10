use super::*;

/// A dispatch Metal drops for being wider than the pipeline completes without an error, so the
/// executor has to refuse it itself. The tile pipeline for an explicit-imageblock function is
/// the case that found this: Metal capped it at 4 threads while the case dispatched 16x16.
#[cfg(target_os = "macos")]
#[test]
fn a_threadgroup_wider_than_the_pipeline_permits_is_refused() {
    use crate::metal::platform::threads_per_threadgroup_within_limit as within;
    assert_eq!(within([16, 16, 1], 256, "Metal tile"), Ok(()));
    assert_eq!(within([16, 16, 1], 257, "Metal tile"), Ok(()));
    let refused = within([16, 16, 1], 4, "Metal tile").expect_err("256 threads exceed 4");
    assert_eq!(
        refused,
        "Metal tile dispatch asks for 256 threads a threadgroup ([16, 16, 1]), but the pipeline \
         permits 4"
    );
    assert_eq!(within([1, 1, 1], 1, "Metal compute"), Ok(()));
    let overflow = within([u32::MAX, u32::MAX, u32::MAX], usize::MAX, "Metal compute")
        .expect_err("the product overflows a thread count");
    assert!(overflow.contains("overflows a thread count"), "{overflow}");
}

/// Every run reporting the same reference gap qualifies the gap, with no output; a gap in some
/// runs and output in others, or different gaps, qualify nothing.
#[test]
fn a_reference_gap_qualifies_only_when_every_run_reports_it() {
    let initial = [0xab, 0xab, 0xab, 0xab];
    let gap = ReferenceGap {
        apple_family: 10,
        message: "simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later"
            .into(),
    };
    assert_eq!(
        qualify_answers(&initial, vec![Err(gap.clone()); QUALIFICATION_RUNS]).unwrap(),
        (Vec::new(), MetalStatus::ReferenceGap(gap.clone()))
    );
    let output = vec![42, 0, 0, 0];
    assert_eq!(
        qualify_answers(&initial, vec![Ok(output.clone()); QUALIFICATION_RUNS]).unwrap(),
        (output.clone(), MetalStatus::Qualified)
    );
    let mixed = vec![Ok(output.clone()), Err(gap.clone()), Ok(output)];
    assert!(qualify_answers(&initial, mixed)
        .unwrap_err()
        .contains("nondeterministic"));
    let other = ReferenceGap {
        apple_family: 11,
        message: "needs GPUFamily11".into(),
    };
    let differing = vec![Err(gap.clone()), Err(other), Err(gap.clone())];
    assert!(qualify_answers(&initial, differing)
        .unwrap_err()
        .contains("different reference gaps"));
    assert!(qualify_answers(&initial, vec![Err(gap.clone()), Err(gap)])
        .unwrap_err()
        .contains("exactly 3 runs"));
    let unnamed = ReferenceGap {
        apple_family: 10,
        message: "no family".into(),
    };
    assert!(qualify_answers(&initial, vec![Err(unnamed); QUALIFICATION_RUNS]).is_err());
}

#[test]
fn qualification_requires_three_identical_outputs_and_accepts_identity() {
    let initial = [0xab, 0xab, 0xab, 0xab];
    let output = vec![42, 0, 0, 0];
    assert_eq!(
        qualify_outputs(
            &initial,
            vec![output.clone(), output.clone(), output.clone()]
        )
        .unwrap(),
        output
    );
    assert!(qualify_outputs(
        &initial,
        vec![vec![42, 0, 0, 0], vec![43, 0, 0, 0], vec![42, 0, 0, 0]]
    )
    .unwrap_err()
    .contains("nondeterministic"));
    assert_eq!(
        qualify_outputs(
            &initial,
            vec![initial.to_vec(), initial.to_vec(), initial.to_vec()]
        )
        .unwrap(),
        initial
    );
}
