//! The backend's report lines by kind, without a device.

use super::{Skips, Substitutes};
use crate::replay::stream::Call;
use crate::replay::value::{Name, Value};
use metal2vulkan::error::{Error, Feature, Limit, LowerFamily, Unsupported};
use metal2vulkan_runtime as rt;
use std::collections::BTreeMap;

/// A pipeline state's making, object `result`.
fn made(result: u64) -> Call {
    Call {
        selector: "newRenderPipelineStateWithDescriptor:error:".into(),
        receiver: 1,
        result,
        args: Vec::new(),
        answers: Vec::new(),
    }
}

/// Each translator refusal is counted by its kind, the refused feature or the exceeded limit, not
/// under one `Translation` key; a runtime gap by the gap.
#[test]
fn translator_refusals_are_counted_by_kind() {
    let tessellation = || {
        let feature = Feature::Lowering(LowerFamily::Tessellation);
        rt::Error::Translation(Unsupported::not_implemented(feature).into())
    };
    let answers = [
        (7, tessellation()),
        (8, tessellation()),
        (
            9,
            rt::Error::Translation(Error::LimitExceeded(Limit::SpirvWords)),
        ),
        (
            10,
            rt::Error::Unsupported(rt::Gap::TextureView {
                kind: rt::TextureViewKind::TextureType,
            }),
        ),
    ];
    let mut states = BTreeMap::new();
    for (result, error) in &answers {
        assert!(Skips::skipped_state(&mut states, &made(*result), error));
    }
    let counted: Vec<(&str, u64, u64)> = (states.iter())
        .map(|(kind, (count, first, _))| (kind.as_str(), *count, *first))
        .collect();
    assert_eq!(
        counted,
        [
            ("TextureView { kind: TextureType }", 1, 10),
            ("Translation(LimitExceeded(SpirvWords))", 1, 9),
            ("Translation(Unsupported(Lowering(Tessellation)))", 2, 7),
        ]
    );
}

/// Each substitute is counted by the runtime's gap it stands in for, with the first one's object,
/// in one line per gap.
#[test]
fn substitutes_are_counted_by_kind() {
    let view = rt::Gap::TextureView {
        kind: rt::TextureViewKind::TextureType,
    };
    let format = rt::Gap::TextureView {
        kind: rt::TextureViewKind::PixelFormat,
    };
    let mut substitutes = Substitutes::default();
    substitutes.made(format, 5);
    substitutes.made(view, 9);
    substitutes.made(view, 11);
    assert_eq!(substitutes.gaps, 3);
    assert_eq!(
        substitutes.lines().collect::<Vec<_>>(),
        [
            "substituted textures TextureView { kind: PixelFormat } 1, the first #5",
            "substituted textures TextureView { kind: TextureType } 2, the first #9",
        ]
    );
}

/// A render pipeline record with the `tessellation` key holds the program's tessellation state;
/// one without it (a recorder that did not carry the key) does not, and the report says how many
/// patch draws the player drew with Metal's defaults in its place.
#[test]
fn a_record_without_the_tessellation_key_is_unrecorded() {
    let map = |keys: &[&str]| {
        vec![Value::Map(
            (keys.iter())
                .map(|&key| (Name::from(key), Value::Nil))
                .collect(),
        )]
    };
    assert!(super::render::records_tessellation(&map(&[
        "vertex",
        "tessellation"
    ])));
    assert!(!super::render::records_tessellation(&map(&[
        "vertex", "fragment"
    ])));
    assert!(!super::render::records_tessellation(&[]));
    let mut unrecorded = super::render::UnrecordedTessellation::default();
    unrecorded.patch_draws = 3;
    assert_eq!(
        unrecorded.line(),
        "patch draws through a render pipeline state whose record lacks tessellation 3: the \
         program's tessellation state is unknown, and the player drew them with Metal's defaults"
    );
}
