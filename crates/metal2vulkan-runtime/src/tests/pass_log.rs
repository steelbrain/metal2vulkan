//! The pass log's filter and its volume (`pass_log.rs`).

use crate::pass_log::{fresh_in, parse, Watched, LINES};
use metal2vulkan::metal::PixelFormat;
use std::collections::BTreeSet;
use std::sync::Mutex;

/// The textures a value watches: each entry's format and size, a name as the notices print it or
/// as Metal names it, `*` for any; spaces around an entry are left out.
#[test]
fn the_filter_reads_formats_and_sizes() {
    let watched = |format, size| Watched { format, size };
    let rgba16f = Some(PixelFormat::RGBA16Float);
    assert_eq!(
        parse("RGBA16Float:128x128"),
        Ok(vec![watched(rgba16f, Some([128, 128]))])
    );
    assert_eq!(
        parse("RGBA16Float:128x128, BGRA8Unorm:224x128"),
        Ok(vec![
            watched(rgba16f, Some([128, 128])),
            watched(Some(PixelFormat::BGRA8Unorm), Some([224, 128])),
        ])
    );
    for name in ["MTLPixelFormatRGBA16Float", "RGBA16Float"] {
        assert_eq!(
            parse(&format!("{name}:1920x1080")),
            Ok(vec![watched(rgba16f, Some([1920, 1080]))]),
            "{name}"
        );
    }
    let srgb = Some(PixelFormat::BGRA8UnormSRGB);
    for name in [
        "BGRA8UnormSRGB",
        "BGRA8Unorm_sRGB",
        "MTLPixelFormatBGRA8Unorm_sRGB",
    ] {
        assert_eq!(
            parse(&format!("{name}:*")),
            Ok(vec![watched(srgb, None)]),
            "{name}"
        );
    }
    assert_eq!(
        parse("*:224x128"),
        Ok(vec![watched(None, Some([224, 128]))])
    );
    assert_eq!(parse("*:*"), Ok(vec![watched(None, None)]));
}

/// A value the log cannot read names what it lacks: no `:`, a format of no name, a size that is
/// not two nonzero sides, an empty entry.
#[test]
fn the_filter_refuses_what_it_cannot_read() {
    for (value, why) in [
        ("RGBA16Float", "`RGBA16Float` is not FORMAT:WIDTHxHEIGHT"),
        (
            "RGBA16:128x128",
            "`RGBA16:128x128`: no pixel format `RGBA16`",
        ),
        (
            "RGBA16Float:128",
            "`RGBA16Float:128`: `128` is not WIDTHxHEIGHT",
        ),
        (
            "RGBA16Float:0x128",
            "`RGBA16Float:0x128`: `0x128` is not WIDTHxHEIGHT",
        ),
        (
            "RGBA16Float:128x",
            "`RGBA16Float:128x`: `128x` is not WIDTHxHEIGHT",
        ),
        ("RGBA16Float:128x128,", "`` is not FORMAT:WIDTHxHEIGHT"),
    ] {
        assert_eq!(parse(value), Err(why.to_owned()), "{value}");
    }
}

/// An entry watches the textures of its format and size, either part any where it is `*`.
#[test]
fn an_entry_watches_its_format_and_size() {
    let [entry] = parse("RGBA16Float:128x128").unwrap()[..] else {
        panic!("one entry");
    };
    assert!(entry.watches(PixelFormat::RGBA16Float, [128, 128]));
    assert!(!entry.watches(PixelFormat::RGBA16Float, [128, 64]));
    assert!(!entry.watches(PixelFormat::RGBA16Float, [64, 128]));
    assert!(!entry.watches(PixelFormat::BGRA8Unorm, [128, 128]));
    let [any_format] = parse("*:128x128").unwrap()[..] else {
        panic!("one entry");
    };
    assert!(any_format.watches(PixelFormat::BGRA8Unorm, [128, 128]));
    assert!(!any_format.watches(PixelFormat::BGRA8Unorm, [128, 129]));
    let [any_size] = parse("BGRA8Unorm:*").unwrap()[..] else {
        panic!("one entry");
    };
    assert!(any_size.watches(PixelFormat::BGRA8Unorm, [1, 4096]));
    assert!(!any_size.watches(PixelFormat::RGBA8Unorm, [1, 4096]));
}

/// A line the same as one printed before is not printed again, and no more than [`LINES`] are,
/// the last saying so.
#[test]
fn the_log_prints_each_line_once_and_at_most_its_bound() {
    let printed = Mutex::new(BTreeSet::new());
    let mut first = "a pass".to_owned();
    assert!(fresh_in(&printed, &mut first));
    assert_eq!(first, "a pass");
    assert!(!fresh_in(&printed, &mut "a pass".to_owned()));
    for n in 1..LINES - 1 {
        assert!(fresh_in(&printed, &mut format!("pass {n}")), "line {n}");
    }
    let mut last = "the last pass".to_owned();
    assert!(fresh_in(&printed, &mut last));
    assert_eq!(
        last,
        format!("the last pass; the pass log's last line (it prints at most {LINES})")
    );
    assert!(!fresh_in(&printed, &mut "one more".to_owned()));
}
