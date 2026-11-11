//! The class derivation table and the lane comparator.

use super::*;
use PrecisionNote as N;

const HALF_ONE_ULP: LaneTolerance = LaneTolerance { ulps: 1 };

fn half_table(half: LaneTolerance) -> UlpTable {
    UlpTable {
        half,
        float: LaneTolerance::EXACT,
    }
}

fn class(notes: &[PrecisionNote], strategies: &[&str]) -> ComparisonClass {
    ComparisonClass::derive(notes.iter().copied(), strategies.iter().copied())
}

fn halves(lanes: &[u16]) -> Vec<u8> {
    lanes.iter().flat_map(|lane| lane.to_le_bytes()).collect()
}

fn floats(lanes: &[u32]) -> Vec<u8> {
    lanes.iter().flat_map(|lane| lane.to_le_bytes()).collect()
}

/// Every note the product records today, with its row of the derivation table. A note added to
/// the product without a row here derives `Unbounded` (the wildcard arm), never a tolerance.
#[test]
fn every_precision_note_has_its_table_row() {
    let rows = [
        (
            N::F16FmaDoubleRounding,
            Allowance::Lanes(half_table(HALF_ONE_ULP)),
        ),
        (
            N::F16IntConvDoubleRounding,
            Allowance::Lanes(half_table(HALF_ONE_ULP)),
        ),
        (N::ReductionOrder, Allowance::Reduction),
        (N::GradFromBias, Allowance::Unbounded),
        (N::DeviceTranscendentals, Allowance::Unbounded),
        (N::DeviceDenormals, Allowance::Recorded),
        (N::PixelSamplerScaling, Allowance::Unbounded),
        (N::ArgumentPixelSamplerScaling, Allowance::Unbounded),
        (N::DeviceArithmetic, Allowance::Unbounded),
        (N::DeviceSignedZeroInfNan, Allowance::Unbounded),
        (N::DeviceRayIntersection, Allowance::Unbounded),
        (N::FastMathRewrites, Allowance::Unbounded),
        (N::TessellationWinding, Allowance::Unbounded),
        (N::TessellationFactorEdgeCases, Allowance::Unbounded),
        (N::TessellationFactorScale, Allowance::Unbounded),
        (N::TessellationFactorInstance, Allowance::Unbounded),
        (N::Fp8Value, Allowance::Unbounded),
        (N::MultisamplePosition, Allowance::Unbounded),
        (N::RasterizationRateMap, Allowance::Unbounded),
        (N::BicubicFilterArithmetic, Allowance::Unbounded),
        (N::SparseResidency, Allowance::Unbounded),
        (N::DeviceNanEncoding, Allowance::Recorded),
        (N::RestrictedDepthRange, Allowance::Unbounded),
        (N::CubeGradientFootprint, Allowance::Unbounded),
        (N::HalfTexelConversion, Allowance::Unbounded),
        (N::HalfSampleFilter, Allowance::Unbounded),
        (N::SrgbTexelDecode, Allowance::Unbounded),
        (N::FloatSampleFilter, Allowance::Unbounded),
        (N::FunctionStorageAssumed, Allowance::Recorded),
        (N::ViewBaseTexelOffset, Allowance::Recorded),
        (N::DeviceColorConversion, Allowance::Recorded),
        (N::DeviceInterpolation, Allowance::Unbounded),
        (N::ExtrapolatedInterpolation, Allowance::Unbounded),
        (N::NativeConversionRounding, Allowance::Unbounded),
        (N::QuadsSplitByClipping, Allowance::Unbounded),
        (N::VertexSimdgroupGrouping, Allowance::Unbounded),
        (N::FragmentSimdgroupGrouping, Allowance::Unbounded),
        (N::BlendedWriteRounding, Allowance::Unbounded),
        (N::CoverageReadsRoundedAlpha, Allowance::Unbounded),
        (N::DeviceAlphaCoverage, Allowance::Unbounded),
        (N::RotatedBlendRounding, Allowance::Unbounded),
        (N::ComputedSampleDepth, Allowance::Unbounded),
    ];
    for (note, expected) in rows {
        assert_eq!(allowance(note), expected, "{note:?}");
    }
}

/// Two notes' differences can be told value by value; no other note's is judged so.
#[test]
fn only_the_environment_notes_are_told_value_by_value() {
    let per_value = [
        (N::DeviceNanEncoding, ValueDifference::NanEncoding),
        (N::DeviceDenormals, ValueDifference::Denormal),
    ];
    for (note, kind) in per_value {
        assert_eq!(value_difference(note), Some(kind), "{note:?}");
    }
    for note in [
        N::F16FmaDoubleRounding,
        N::F16IntConvDoubleRounding,
        N::ReductionOrder,
        N::GradFromBias,
        N::DeviceTranscendentals,
        N::PixelSamplerScaling,
        N::DeviceArithmetic,
        N::DeviceSignedZeroInfNan,
        N::DeviceRayIntersection,
        N::FastMathRewrites,
        N::TessellationWinding,
        N::TessellationFactorEdgeCases,
        N::TessellationFactorScale,
        N::TessellationFactorInstance,
        N::Fp8Value,
        N::MultisamplePosition,
        N::RasterizationRateMap,
        N::BicubicFilterArithmetic,
        N::SparseResidency,
        N::ViewBaseTexelOffset,
        N::DeviceColorConversion,
    ] {
        assert_eq!(value_difference(note), None, "{note:?}");
    }
}

#[test]
fn the_class_is_derived_from_the_notes_and_the_census() {
    // No note: byte for byte, whatever the census holds.
    assert_eq!(
        class(&[], &["simd.reduce", "arith.float"]),
        ComparisonClass::Exact
    );
    // Lane allowances widen one table.
    assert_eq!(
        class(&[N::F16FmaDoubleRounding, N::F16IntConvDoubleRounding], &[]),
        ComparisonClass::UlpBounded(half_table(HALF_ONE_ULP))
    );
    // A float sum bounds the lanes of its own format by the widest sum of that format the
    // census records; the other format stays exact.
    let reordered = |half, float| {
        let lane = |ulps| LaneTolerance { ulps };
        ComparisonClass::ReductionOrderTolerant(UlpTable {
            half: lane(half),
            float: lane(float),
        })
    };
    assert_eq!(
        class(&[N::ReductionOrder], &["simd.reduce.fadd.float"]),
        reordered(0, 62)
    );
    assert_eq!(
        class(
            &[N::ReductionOrder],
            &["simd.reduce.fmul.float", "simd.scan.fadd.float"]
        ),
        reordered(0, 62)
    );
    assert_eq!(
        class(&[N::ReductionOrder], &["simd.reduce.fadd.half"]),
        reordered(63, 0)
    );
    assert_eq!(
        class(
            &[N::ReductionOrder],
            &["simd.reduce.fadd.float", "simd.scan.fmul.half"]
        ),
        reordered(63, 62)
    );
    // A float sum's table widens with the lane allowances of the other notes.
    assert_eq!(
        class(
            &[N::ReductionOrder, N::F16FmaDoubleRounding],
            &["simd.reduce.fadd.float"]
        ),
        reordered(1, 62)
    );
    // Integer and bitwise group reductions sum no float lanes: they bound nothing.
    assert_eq!(
        class(
            &[N::ReductionOrder],
            &["simd.reduce", "simd.scan", "quad.reduce"]
        ),
        ComparisonClass::Unbounded([N::ReductionOrder].into())
    );
    // Nor does a float sum of a format no observation holds as lanes, beside any other.
    assert_eq!(
        class(
            &[N::ReductionOrder],
            &["simd.reduce.fadd.float", "simd.reduce.fadd.double"]
        ),
        ComparisonClass::Unbounded([N::ReductionOrder].into())
    );
    assert_eq!(
        class(&[N::ReductionOrder], &["simd.scan.fadd.bfloat"]),
        ComparisonClass::Unbounded([N::ReductionOrder].into())
    );
    // A reduction note no known reduction accounts for bounds nothing.
    assert_eq!(
        class(&[N::ReductionOrder], &["simd.matrix_multiply_accumulate"]),
        ComparisonClass::Unbounded([N::ReductionOrder].into())
    );
    // Any unbounded note makes the class unbounded, and the class names every such note.
    assert_eq!(
        class(
            &[
                N::F16FmaDoubleRounding,
                N::DeviceArithmetic,
                N::ReductionOrder,
                N::DeviceTranscendentals,
            ],
            &["simd.matrix_multiply_accumulate"],
        ),
        ComparisonClass::Unbounded(
            [
                N::DeviceArithmetic,
                N::ReductionOrder,
                N::DeviceTranscendentals
            ]
            .into()
        )
    );
    // Repeated notes from several stages are one set.
    assert_eq!(
        class(&[N::F16FmaDoubleRounding, N::F16FmaDoubleRounding], &[]),
        class(&[N::F16FmaDoubleRounding], &[])
    );
}

#[test]
fn reassociation_bounds_follow_the_unit_roundoff() {
    assert_eq!(reorder_ulps(1, FloatLane::Float), 0);
    assert_eq!(reorder_ulps(32, FloatLane::Float), 62);
    assert_eq!(reorder_ulps(32, FloatLane::Half), 63);
    assert_eq!(reorder_ulps(4, FloatLane::Float), 6);
    assert_eq!(reorder_ulps(4, FloatLane::Half), 6);
    assert_eq!(reduction_table(["arith.float", "access.load"]), None);
}

#[test]
fn a_tolerance_applies_only_to_typed_float_lanes() {
    let one_ulp = class(&[N::F16FmaDoubleRounding], &[]);
    let (golden, candidate) = (halves(&[0x3c00, 0x4000]), halves(&[0x3c01, 0x4000]));
    assert!(one_ulp.matches(
        ObservationLanes::Float(FloatLane::Half),
        &golden,
        &candidate
    ));
    for lanes in [ObservationLanes::Untyped, ObservationLanes::Integer] {
        assert!(!one_ulp.matches(lanes, &golden, &candidate), "{lanes:?}");
        assert!(one_ulp.matches(lanes, &golden, &golden));
    }
    // The half table does not reach float lanes.
    assert!(!one_ulp.matches(
        ObservationLanes::Float(FloatLane::Float),
        &floats(&[0x3f80_0000]),
        &floats(&[0x3f80_0001])
    ));
    // An unbounded class never accepts a difference.
    let unbounded = class(&[N::DeviceTranscendentals], &[]);
    assert!(!unbounded.tolerates_differences());
    assert!(!unbounded.matches(
        ObservationLanes::Float(FloatLane::Half),
        &golden,
        &candidate
    ));
    assert!(unbounded.matches(ObservationLanes::Untyped, &golden, &golden));
    // Lengths that do not split into lanes, or differ, never match.
    assert!(!one_ulp.matches(
        ObservationLanes::Float(FloatLane::Half),
        &[0x00, 0x3c, 0x00],
        &[0x01, 0x3c, 0x00]
    ));
    assert!(!one_ulp.matches(
        ObservationLanes::Float(FloatLane::Half),
        &golden,
        &candidate[..2]
    ));
}

#[test]
fn lanes_compare_by_distance_within_one_sign() {
    let two = LaneTolerance { ulps: 2 };
    let f = FloatLane::Float;
    // Within the distance, across a binade.
    assert!(lane_within(f, two, 0x3f80_0000, 0x3f7f_ffff));
    assert!(lane_within(f, two, 0x3f80_0000, 0x3f7f_fffe));
    assert!(!lane_within(f, two, 0x3f80_0000, 0x3f7f_fffd));
    assert!(lane_within(f, two, 0xbf80_0000, 0xbf80_0002));
    // Never across signs, not even between zeros.
    assert!(!lane_within(f, two, 0x0000_0000, 0x8000_0000));
    assert!(!lane_within(f, two, 0x0000_0001, 0x8000_0001));
    // Infinities and NaNs only bit for bit: the largest finite value is not near infinity.
    assert!(!lane_within(f, two, 0x7f80_0000, 0x7f7f_ffff));
    assert!(lane_within(f, two, 0x7fc0_0000, 0x7fc0_0000));
    assert!(!lane_within(f, two, 0x7fc0_0000, 0x7fc0_0001));
}

#[test]
fn observation_lanes_follow_the_selected_output_format() {
    use TextureFormat as F;
    assert_eq!(
        ObservationLanes::of_format(F::Rgba16Float),
        ObservationLanes::Float(FloatLane::Half)
    );
    assert_eq!(
        ObservationLanes::of_format(F::Depth32Float),
        ObservationLanes::Float(FloatLane::Float)
    );
    assert_eq!(
        ObservationLanes::of_format(F::Rgba8Unorm),
        ObservationLanes::Integer
    );

    let mut case = crate::case::attachmentless_fragment_test_case("11".repeat(32), "main".into());
    assert_eq!(ObservationLanes::of(&case), ObservationLanes::Untyped);
    case.output = OutputSelection::Buffer {
        binding: 0,
        offset: 0,
        length: 4,
    };
    assert_eq!(ObservationLanes::of(&case), ObservationLanes::Untyped);
    case.render_targets = vec![crate::case::RenderTargetResource {
        index: 1,
        format: F::Rgba32Float,
        dimensions: [1, 1],
        initial_bytes_b64: String::new(),
    }];
    case.output = OutputSelection::RenderTarget {
        index: 1,
        origin: [0, 0],
        dimensions: [1, 1],
    };
    assert_eq!(
        ObservationLanes::of(&case),
        ObservationLanes::Float(FloatLane::Float)
    );
    // An output naming no declared resource has no declared type.
    case.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [1, 1],
    };
    assert_eq!(ObservationLanes::of(&case), ObservationLanes::Untyped);
    case.output = OutputSelection::Stencil {
        origin: [0, 0],
        dimensions: [1, 1],
    };
    assert_eq!(ObservationLanes::of(&case), ObservationLanes::Integer);
}

/// Stored observations spell `Exact` as `"exact"`; every class round-trips.
#[test]
fn classes_serialize_compatibly_with_stored_observations() {
    assert_eq!(
        serde_json::to_value(ComparisonClass::Exact).unwrap(),
        serde_json::json!("exact")
    );
    for class in [
        ComparisonClass::Exact,
        class(&[N::F16FmaDoubleRounding], &[]),
        class(&[N::ReductionOrder], &["simd.reduce.fadd.float"]),
        class(&[N::DeviceArithmetic, N::GradFromBias], &[]),
    ] {
        let json = serde_json::to_string(&class).unwrap();
        assert_eq!(
            serde_json::from_str::<ComparisonClass>(&json).unwrap(),
            class,
            "{json}"
        );
    }
}

/// `DeviceDenormals` and `DeviceNanEncoding` are recorded, not allowances: a module that records
/// them derives the class it derives without them, so they neither explain a difference nor take
/// away the tolerance another note derives.
#[test]
fn recorded_notes_leave_the_class_as_it_is() {
    let recorded = [N::DeviceDenormals, N::DeviceNanEncoding];
    for others in [
        vec![],
        vec![N::F16FmaDoubleRounding],
        vec![N::ReductionOrder],
        vec![N::DeviceArithmetic],
    ] {
        let strategies = ["simd.reduce.fadd.float"];
        let without = ComparisonClass::derive(others.iter().copied(), strategies);
        let with = ComparisonClass::derive(others.iter().copied().chain(recorded), strategies);
        assert_eq!(with, without, "{others:?}");
    }
    assert_eq!(
        ComparisonClass::derive(recorded, std::iter::empty()),
        ComparisonClass::Exact
    );
}
