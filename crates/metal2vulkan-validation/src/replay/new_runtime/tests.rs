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
            functions: Box::default(),
        }
    };
    let answers = [
        (7, tessellation()),
        (8, tessellation()),
        (
            9,
            rt::Error::Translation {
                error: Error::LimitExceeded(Limit::SpirvWords),
                functions: Box::default(),
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
        functions: Box::default(),
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

/// A tessellation factor scale of 1, Metal's default, plays as nothing, as a float or a double;
/// any other scale is a skipped render command of its own kind; a scale that is no float is
/// refused.
#[test]
fn a_tessellation_factor_scale_other_than_1_is_a_skipped_render_command() {
    let call = |scale: Value| Call {
        selector: "setTessellationFactorScale:".into(),
        receiver: 4,
        result: 0,
        args: vec![scale],
        answers: Vec::new(),
    };
    let read = |scale| super::render::factor_scale(&call(scale));
    assert_eq!(read(Value::F32(1.0)).unwrap(), None);
    assert_eq!(read(Value::F64(1.0)).unwrap(), None);
    assert_eq!(
        read(Value::F32(2.0)).unwrap().as_deref(),
        Some("TessellationFactorScale")
    );
    assert_eq!(
        read(Value::F32(0.5)).unwrap().as_deref(),
        Some("TessellationFactorScale")
    );
    assert!(read(Value::Uint(1)).is_err());
}

/// A fault's access is placed among the stream's buffers and heaps: in a live buffer and the live
/// heap it lies in, by address, with its offset in each; in a released buffer, with the frame and
/// call of its release, newest first; outside all of them where none meets its range, which is
/// the address rounded down to its precision. The released ones kept are the last `FREED_KEPT`.
#[test]
fn a_fault_s_access_is_placed_among_the_stream_s_buffers_and_heaps() {
    use super::faulted::{place, Freed, Span, FREED_KEPT};
    let at = |address, precision| rt::FaultAddress {
        kind: ash::vk::DeviceFaultAddressTypeEXT::READ_INVALID,
        address,
        precision,
    };
    let span = |object, heap, start, len| Span {
        object,
        heap,
        start,
        len,
    };
    let live = [
        span(7, false, 0x1_2000, 0x1000),
        span(3, true, 0x1_0000, 0x1_0000),
        span(9, false, 0x3_0080, 0x100),
    ];
    let mut freed = Freed::default();
    freed.released(span(11, false, 0x5_0000, 0x2000), (4, 120));
    freed.released(span(12, false, 0x5_1000, 0x40), (6, 33));
    assert_eq!(
        place(&at(0x1_2345, 0x10), &live, &freed),
        " in #3 heap +0x2345 live, #7 buffer +0x345 live"
    );
    // A page's precision meets a buffer that starts inside the page, past the address.
    assert_eq!(
        place(&at(0x3_0010, 0x1000), &live, &freed),
        " in #9 buffer -0x70 live"
    );
    assert_eq!(
        place(&at(0x5_1008, 0x4), &live, &freed),
        " in #12 buffer +0x8 released at frame 6 call 33, #11 buffer +0x1008 released at frame 4 call 120"
    );
    assert_eq!(
        place(&at(0x9_0000, 0x1000), &live, &freed),
        " outside every buffer and heap of the stream"
    );
    // Filled past what it keeps, the oldest released buffer goes.
    for object in 0..FREED_KEPT as u64 {
        freed.released(span(100 + object, false, 0x100_0000, 0x10), (7, object));
    }
    let placed = place(&at(0x5_0000, 0x4), &[], &freed);
    assert_eq!(placed, " outside every buffer and heap of the stream");
}

/// A pass's sample positions are Metal's defaults where none are set, or where they are the
/// defaults of the pass's own sample count, each coordinate equal; any other set differs: one
/// position off, the defaults in another order, another count's defaults (fewer positions than
/// the pass's samples), and any set at a count whose defaults the tree does not hold (Vulkan's
/// standard locations of 8 samples among them). The report's line reads none counted as such.
#[test]
fn sample_positions_differ_from_metal_s_defaults_of_the_pass_s_count() {
    use super::render::{at_metal_defaults, OtherPositions};
    let one = [[0.5, 0.5]];
    let two = [[0.75, 0.75], [0.25, 0.25]];
    let four = [
        [0.375, 0.125],
        [0.875, 0.375],
        [0.125, 0.625],
        [0.625, 0.875],
    ];
    let eight = [
        [0.5625, 0.3125],
        [0.4375, 0.6875],
        [0.8125, 0.5625],
        [0.3125, 0.1875],
        [0.1875, 0.8125],
        [0.0625, 0.4375],
        [0.6875, 0.9375],
        [0.9375, 0.0625],
    ];
    for samples in [0, 1, 2, 4, 8, 16] {
        assert!(at_metal_defaults(samples, &[]), "{samples}");
    }
    assert!(at_metal_defaults(1, &one));
    assert!(at_metal_defaults(2, &two));
    assert!(at_metal_defaults(4, &four));
    let mut off = four;
    off[3][1] = 0.8125;
    let reversed = [two[1], two[0]];
    assert!(!at_metal_defaults(4, &off));
    assert!(!at_metal_defaults(2, &reversed));
    assert!(!at_metal_defaults(4, &two));
    assert!(!at_metal_defaults(4, &[[0.25, 0.5]; 4]));
    assert!(!at_metal_defaults(1, &[[0.25, 0.25]]));
    assert!(!at_metal_defaults(8, &eight));
    assert!(!at_metal_defaults(0, &one));
    assert_eq!(
        OtherPositions::default().line(),
        "passes at sample positions other than Metal's defaults 0: rasterized at the defaults, \
         Vulkan's standard locations; the first: -"
    );
}

/// The `over copies by frame` line: each frame's over-copy bytes by the buffer's memory class and
/// by cause, as the calls are located, their mean and the largest frame's, with the textures in
/// an image of their own that frame's calls named, a view as its texture and a released one no
/// more; the open frame counts once a call of it is located.
#[test]
fn over_copies_by_frame_give_the_mean_and_the_largest_frame() {
    let mut frames = super::over_frames::OverFrames::default();
    let mut m = rt::Metrics::default();
    assert_eq!(frames.line(&m), "over copies by frame: no frame");
    frames.made(5);
    frames.made(6);
    frames.viewed(5, 7);
    // 9 is in no image of its own, nor its view 10.
    frames.viewed(9, 10);
    // Frame 0: a pitch copy of a Shared buffer's, of texture 5, named itself and through its view.
    frames.located();
    frames.named(&[5, 7, 9, 10]);
    m.over_copy_bytes = 100;
    m.over_copy_bytes_shared = 100;
    m.over_copy_bytes_pitch = 100;
    frames.ended(&m);
    // Frame 1: a Private buffer's whose linear image the device does not make, and a dma-buf's,
    // of textures 5 and 6; 6 is released.
    frames.located();
    frames.named(&[6, 5]);
    m.over_copy_bytes = 400;
    m.over_copy_bytes_private = 200;
    m.over_copy_bytes_shared = 200;
    m.over_copy_bytes_linear = 200;
    m.over_copy_bytes_dma_buf = 100;
    frames.released(6);
    frames.ended(&m);
    // Frame 2, open: host memory's, and 6 named once released.
    frames.located();
    frames.named(&[6]);
    m.over_copy_bytes = 420;
    m.over_copy_bytes_shared = 220;
    m.over_copy_bytes_host_memory = 20;
    assert_eq!(
        frames.line(&m),
        "over copies by frame, 3 frames as the calls played: mean bytes=140 private=66 shared=73 \
         pitch=33 dma_buf=33 host_memory=6 own_image=66; largest, frame 1 bytes=300 private=200 \
         shared=100 pitch=0 dma_buf=100 host_memory=0 own_image=200, 2 textures in an image of \
         their own its calls named"
    );
}
