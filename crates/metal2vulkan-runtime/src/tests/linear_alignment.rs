//! `texture/linear_alignment.rs`: the record of a device's linear texture alignments, read back
//! for its driver alone and only whole.

use super::capabilities::{committed_descriptions, profile_value};
use crate::in_place::DriverIdentity;
use crate::texture::linear_alignment::{
    backed_formats, decode, encode, placed_answer, record_name, USAGES,
};
use metal2vulkan::metal::PixelFormat;
use std::collections::HashMap;

/// A format's answer is the largest of its usages' in-place alignments, at least the device's
/// line of memory (`nonCoherentAtomSize`), and Metal's 16 where no usage is ever in place. Over
/// the committed profiles' lines, with the probe answers the venues' drivers gave for reads
/// (ANV's 16, lavapipe's 64, NVIDIA's 128, RADV's 256, MoltenVK's 16): ANV's 16 rises to its
/// 64-byte line, which its probe alone answered before, and every other venue's stays.
#[test]
fn the_answer_is_the_usages_largest_and_at_least_the_line() {
    assert_eq!(USAGES.len(), 3);
    let line = |stem: &str| {
        profile_value(stem, "nonCoherentAtomSize")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("{stem}: no nonCoherentAtomSize"))
    };
    // (profile, the probe's read answer, the answer before the line, after it)
    let venues = [
        ("intel-irisxe-rplp-anv", 16, 16, 64),
        ("intel-uhd770-rpls-anv", 16, 16, 64),
        ("gate-lavapipe", 64, 64, 64),
        ("nvidia-rtx5080-laptop", 128, 128, 128),
        ("amd-780m-radv", 256, 256, 256),
        ("moltenvk-m3", 16, 16, 16),
    ];
    for (stem, read, before, after) in venues {
        let line = line(stem);
        assert_eq!(placed_answer(&[Some(read)], 1), before, "{stem}");
        assert_eq!(
            placed_answer(&[Some(read), None, None], line),
            after,
            "{stem}"
        );
        // A usage that needs a wider pitch raises the answer; one never in place leaves it.
        assert_eq!(
            placed_answer(&[Some(read), Some(512), None], line),
            512,
            "{stem}"
        );
        assert_eq!(placed_answer(&[None, None, None], line), 16, "{stem}");
    }
    for (stem, _) in committed_descriptions() {
        let line = line(&stem);
        assert!(line.is_power_of_two() && line <= 256, "{stem}: {line}");
        assert_eq!(placed_answer(&[Some(16)], line), line.max(16), "{stem}");
    }
}

fn identity() -> DriverIdentity {
    DriverIdentity {
        vendor_id: 0x1002,
        device_id: 0x15bf,
        driver_version: 0x0680_2004,
        driver_id: 3,
        driver_name: "radv".to_owned(),
        driver_info: "Mesa 26.2.4".to_owned(),
        device_uuid: [5; 16],
    }
}

/// Every format a buffer backs at 256 bytes, `BGRA8Unorm` at `bgra`.
fn answers(bgra: u64) -> HashMap<PixelFormat, u64> {
    let mut answers: HashMap<PixelFormat, u64> = backed_formats().map(|pf| (pf, 256)).collect();
    answers.insert(PixelFormat::BGRA8Unorm, bgra);
    answers
}

/// A record reads back its answers for its identity alone, whole and unchanged; one that misses
/// a format a buffer backs, answers one no buffer backs, or holds an answer the probe could not
/// give (a power of two from 16 to 4096) reads back nothing.
#[test]
fn the_record_reads_back_every_format_s_answer_for_its_driver_alone() {
    let identity = identity();
    for bgra in [16, 64, 4096] {
        let answers = answers(bgra);
        let bytes = encode(&identity, &answers);
        assert_eq!(decode(&identity, &bytes), Some(answers.clone()));
        let mut changed = bytes.clone();
        changed[60] ^= 1;
        assert_eq!(decode(&identity, &changed), None, "a byte changed");
        assert_eq!(decode(&identity, &bytes[1..]), None, "cut short");
        let mut other = identity.clone();
        other.driver_info.push('x');
        assert_eq!(decode(&other, &bytes), None, "another identity");
        assert_ne!(record_name(&identity), record_name(&other));
    }
    let mut missing = answers(16);
    missing.remove(&PixelFormat::RGBA8Unorm);
    assert_eq!(
        decode(&identity, &encode(&identity, &missing)),
        None,
        "a format missing"
    );
    let mut extra = answers(16);
    extra.insert(PixelFormat::BC1RGBA, 16);
    assert_eq!(
        decode(&identity, &encode(&identity, &extra)),
        None,
        "a format no buffer backs"
    );
    for unanswerable in [0, 8, 48, 8192] {
        let bytes = encode(&identity, &answers(unanswerable));
        assert_eq!(decode(&identity, &bytes), None, "{unanswerable}");
    }
    // Its name is not the in-place check's of the same identity.
    assert_ne!(
        record_name(&identity),
        crate::in_place::record_name(&identity)
    );
}
