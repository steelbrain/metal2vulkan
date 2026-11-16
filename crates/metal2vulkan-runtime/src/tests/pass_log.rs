//! The pass log's filter and its volume (`pass_log.rs`).

use crate::encoder_trail::label_hash;
use crate::pass_log::{fresh_in, parse, Watched, LINES};
use metal2vulkan::metal::PixelFormat;
use std::collections::BTreeSet;
use std::sync::Mutex;

/// The textures a value watches: each entry's format and size, a name as the notices print it or
/// as Metal names it, `*` for any; spaces around an entry are left out.
#[test]
fn the_filter_reads_formats_and_sizes() {
    let watched = |format, size| Watched::Texture { format, size };
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
        (
            "RGBA16Float",
            "`RGBA16Float` is not FORMAT:WIDTHxHEIGHT or state:HASH",
        ),
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
        (
            "RGBA16Float:128x128,",
            "`` is not FORMAT:WIDTHxHEIGHT or state:HASH",
        ),
        (
            "state:3c1ab914c5068a5",
            "`state:3c1ab914c5068a5`: `3c1ab914c5068a5` is not a label hash (16 hex digits)",
        ),
        (
            "state:3c1ab914c5068a5g",
            "`state:3c1ab914c5068a5g`: `3c1ab914c5068a5g` is not a label hash (16 hex digits)",
        ),
        (
            "state:+c1ab914c5068a5f",
            "`state:+c1ab914c5068a5f`: `+c1ab914c5068a5f` is not a label hash (16 hex digits)",
        ),
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

/// A `state:HASH` entry watches the render pipeline state of that label hash, as the encoder times
/// name it, beside texture entries: none of the textures, and a texture entry no state.
#[test]
fn a_state_entry_watches_the_state_of_its_label_hash() {
    let hash = label_hash(Some("a state"), &["vertex_main", "fragment_main"]);
    let other = label_hash(Some("another state"), &["vertex_main", "fragment_main"]);
    let unlabeled = label_hash(None, &["vertex_main", "fragment_main"]);
    let entries = parse(&format!("state:{hash}, BGRA8Unorm:*, state:{unlabeled}")).unwrap();
    assert_eq!(entries.len(), 3);
    let watching = |hash: &str| entries.iter().any(|entry| entry.watches_state(hash));
    assert!(watching(&hash));
    assert!(watching(&unlabeled));
    assert!(!watching(&other));
    assert_eq!(
        entries[0],
        Watched::State(u64::from_str_radix(&hash, 16).unwrap())
    );
    // Upper-case digits read the same hash.
    let upper = parse(&format!("state:{}", hash.to_uppercase())).unwrap();
    assert!(upper[0].watches_state(&hash));
    // A state entry watches no texture, a texture entry no state.
    assert!(!entries[0].watches(PixelFormat::BGRA8Unorm, [128, 128]));
    assert!(entries[1].watches(PixelFormat::BGRA8Unorm, [128, 128]));
    assert!(!entries[1].watches_state(&hash));
}

/// A pass the same as one printed before prints no line again, whatever its counts, and no more
/// than [`LINES`] lines are printed, the last saying so.
#[test]
fn the_log_prints_each_pass_once_and_at_most_its_bound() {
    let printed = Mutex::new(BTreeSet::new());
    let mut first = "a pass; 3 draw call(s)".to_owned();
    assert!(fresh_in(&printed, "a pass", &mut first));
    assert_eq!(first, "a pass; 3 draw call(s)");
    let mut again = "a pass; 4 draw call(s)".to_owned();
    assert!(!fresh_in(&printed, "a pass", &mut again));
    for n in 1..LINES - 1 {
        let pass = format!("pass {n}");
        assert!(fresh_in(&printed, &pass, &mut pass.clone()), "line {n}");
    }
    let mut last = "the last pass".to_owned();
    assert!(fresh_in(&printed, "the last pass", &mut last));
    assert_eq!(
        last,
        format!("the last pass; the pass log's last line (it prints at most {LINES})")
    );
    assert!(!fresh_in(&printed, "one more", &mut "one more".to_owned()));
}
