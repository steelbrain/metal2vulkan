//! The backend's report lines by kind, without a device.

use super::{Skips, Substitutes, Via};
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
        rt::Error::Translation {
            error: Unsupported::not_implemented(feature).into(),
            functions: Vec::new(),
        }
    };
    let answers = [
        (7, tessellation()),
        (8, tessellation()),
        (
            9,
            rt::Error::Translation {
                error: Error::LimitExceeded(Limit::SpirvWords),
                functions: Vec::new(),
            },
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

/// A creation the play does not attempt is counted by its selector and why, at its root: from a
/// library from source the play skipped, the function made of it (its receiver is skipped), and
/// two render pipeline states of that function (an argument names it). The states' line adds up
/// the made, the refused and the not attempted.
#[test]
fn creations_not_attempted_are_counted_by_kind_and_root() {
    let mut skips = Skips::default();
    skips.rooted(2, &Name::from("newLibraryWithSource:options:error:"));
    let function = Call {
        selector: "newFunctionWithName:".into(),
        receiver: 2,
        result: 3,
        args: vec![Value::Str("vertex_main".into())],
        answers: Vec::new(),
    };
    skips.not_attempted(&function, Via::Receiver, 2);
    skips.not_attempted(&made(4), Via::Argument, 3);
    skips.not_attempted(&made(5), Via::Argument, 3);
    let refused = rt::Error::Translation {
        error: Error::LimitExceeded(Limit::SpirvWords),
        functions: Vec::new(),
    };
    assert!(Skips::skipped_state(
        &mut skips.render_states,
        &made(6),
        &refused
    ));
    skips.states_made = [7, 1];
    assert_eq!(skips.not_attempted_count(), 3);
    assert_eq!(
        skips.not_attempted_lines().collect::<Vec<_>>(),
        [
            "not attempted newFunctionWithName: 1: its receiver is skipped, of \
             newLibraryWithSource:options:error:, the first #3 (its root #2)",
            "not attempted newRenderPipelineStateWithDescriptor:error: 2: it names a skipped \
             object, of newLibraryWithSource:options:error:, the first #4 (its root #2)",
            "render states made 7, refused 1, not attempted 2; compute states made 1, refused 0, \
             not attempted 0",
        ]
    );
}

/// Each decoder route by its name: the handle its video surfaces are exported with, and their
/// layout. The NT handle, a Windows host's opaque handle, takes the opaque fd's layout.
#[test]
fn each_video_handle_names_its_export_and_layout() {
    use super::VideoHandle;
    let routes = [
        ("dma-buf", VideoHandle::DmaBuf, rt::ExportHandle::DmaBuf),
        (
            "opaque-fd",
            VideoHandle::OpaqueFd,
            rt::ExportHandle::OpaqueFd,
        ),
        (
            "opaque-win32",
            VideoHandle::OpaqueWin32,
            rt::ExportHandle::OpaqueWin32,
        ),
    ];
    for (name, handle, export) in routes {
        assert_eq!(VideoHandle::parse(name), Some(handle), "{name}");
        assert_eq!(handle.name(), name);
        assert_eq!(handle.export(), export, "{name}");
    }
    assert_eq!(VideoHandle::parse("win32"), None);
    // A 1080p picture: a dma-buf's planes hold the coded rows below them, an opaque handle's none.
    assert_eq!(VideoHandle::DmaBuf.layout(1080, 540), (256, 256, [8, 4]));
    assert_eq!(VideoHandle::OpaqueFd.layout(1080, 540), (4, 1, [0, 0]));
    assert_eq!(VideoHandle::OpaqueWin32.layout(1080, 540), (4, 1, [0, 0]));
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
