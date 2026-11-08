//! `texture/linear_alignment.rs`: the record of a device's linear texture alignments, read back
//! for its driver alone and only whole.

use crate::in_place::DriverIdentity;
use crate::texture::linear_alignment::{backed_formats, decode, encode, record_name};
use metal2vulkan::metal::PixelFormat;
use std::collections::HashMap;

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
