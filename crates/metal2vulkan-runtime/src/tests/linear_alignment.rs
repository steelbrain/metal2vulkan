//! `texture/linear_alignment.rs`: the record of a pixel format's linear texture alignment, read
//! back for its driver and format alone.

use crate::in_place::DriverIdentity;
use crate::texture::linear_alignment::{decode, encode, record_name};
use metal2vulkan::metal::PixelFormat;

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

/// A record reads back its answer for its identity and format alone, whole and unchanged, and
/// only an answer the probe could give (a power of two from 16 to 4096).
#[test]
fn the_record_reads_back_its_answer_for_its_driver_and_format_alone() {
    let identity = identity();
    let pf = PixelFormat::BGRA8Unorm;
    for answer in [16, 256, 4096] {
        let bytes = encode(&identity, pf, answer);
        assert_eq!(decode(&identity, pf, &bytes), Some(answer));
        let mut changed = bytes.clone();
        changed[50] ^= 1;
        assert_eq!(decode(&identity, pf, &changed), None, "a byte changed");
        assert_eq!(decode(&identity, pf, &bytes[1..]), None, "cut short");
        let mut other = identity.clone();
        other.driver_info.push('x');
        assert_eq!(decode(&other, pf, &bytes), None, "another identity");
        assert_ne!(record_name(&identity, pf), record_name(&other, pf));
        let rgba = PixelFormat::RGBA8Unorm;
        assert_eq!(decode(&identity, rgba, &bytes), None, "another format");
        assert_ne!(record_name(&identity, pf), record_name(&identity, rgba));
    }
    for unanswerable in [0, 8, 48, 8192] {
        let bytes = encode(&identity, pf, unanswerable);
        assert_eq!(decode(&identity, pf, &bytes), None, "{unanswerable}");
    }
    // Its name is not the in-place check's of the same identity.
    assert_ne!(
        record_name(&identity, pf),
        crate::in_place::record_name(&identity)
    );
}
