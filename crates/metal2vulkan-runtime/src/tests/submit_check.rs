//! The device's submission checks' verdicts and records (`submit_check.rs`), from what they
//! measure: no device.

use crate::in_place::DriverIdentity;
use crate::submit_check::{
    decode, encode, merge_verdict, record_name, tax_verdict, Measured, Verdict, MERGE_RATIO,
    TAX_FLOOR,
};

/// The probe's medians (2026-11-07, Mesa 26.2.6 under Linux 7.3.1, MoltenVK 1.4.2 under macOS
/// 27.0.1), in nanoseconds: one call of four infos and four calls apart; a submission with an
/// import of 16 MiB held and without.
const RADEON_780M: ([i64; 2], [i64; 2]) = ([13_200, 37_300], [20_700, 10_400]);
const RX_7900_XTX: ([i64; 2], [i64; 2]) = ([13_400, 35_000], [15_200, 8_400]);
const RX_550: ([i64; 2], [i64; 2]) = ([11_600, 35_100], [54_700, 8_900]);
const IRIS_XE: ([i64; 2], [i64; 2]) = ([5_100, 13_100], [4_700, 4_500]);
const M3_MAX: ([i64; 2], [i64; 2]) = ([4_500, 4_600], [2_700, 3_000]);

/// Mesa's drivers merge the calls (RADV on three generations, ANV) and MoltenVK does not; RADV
/// taxes every submission for a held import and ANV and MoltenVK do not: the verdicts each
/// device's numbers give, whatever its identity.
#[test]
fn the_probe_s_numbers_give_each_device_its_verdicts() {
    let cases = [
        ("Radeon 780M", RADEON_780M, Verdict::Holds, Verdict::Holds),
        ("RX 7900 XTX", RX_7900_XTX, Verdict::Holds, Verdict::Holds),
        ("RX 550", RX_550, Verdict::Holds, Verdict::Holds),
        ("Iris Xe", IRIS_XE, Verdict::Holds, Verdict::Fails),
        ("M3 Max", M3_MAX, Verdict::Fails, Verdict::Fails),
    ];
    for (name, ([merged, apart], [held, bare]), merges, taxed) in cases {
        assert_eq!(merge_verdict(merged, apart), merges, "{name}");
        assert_eq!(tax_verdict(held, bare), taxed, "{name}");
    }
}

/// The bounds themselves: a call at the ratio merges and one past it does not; a held import
/// that adds the floor, or an eighth of a slow submission, is no tax, and one past it is; no
/// measurement (a check that timed nothing) merges.
#[test]
fn the_verdicts_meet_their_bounds() {
    let apart = 10_000;
    let at = (MERGE_RATIO * apart as f64) as i64;
    assert_eq!(merge_verdict(at, apart), Verdict::Holds);
    assert_eq!(merge_verdict(at + 1, apart), Verdict::Fails);
    assert_eq!(merge_verdict(0, 0), Verdict::Fails);
    assert_eq!(tax_verdict(4_000 + TAX_FLOOR, 4_000), Verdict::Fails);
    assert_eq!(tax_verdict(4_001 + TAX_FLOOR, 4_000), Verdict::Holds);
    let slow = 80_000;
    assert_eq!(tax_verdict(slow + slow / 8, slow), Verdict::Fails);
    assert_eq!(tax_verdict(slow + slow / 8 + 1, slow), Verdict::Holds);
}

fn identity(device_id: u32) -> DriverIdentity {
    DriverIdentity {
        vendor_id: 0x1002,
        device_id,
        driver_version: 0x6802006,
        driver_id: 3,
        driver_name: "radv".into(),
        driver_info: "Mesa 26.2.6".into(),
        device_uuid: [7; 16],
    }
}

/// A record decodes for its own identity alone, every verdict and measurement as written, and
/// names itself apart from another identity's; a record with a pending verdict, a truncated one
/// and one whose bytes changed decode to nothing.
#[test]
fn a_record_holds_its_identity_s_verdicts() {
    let (one, other) = (identity(0x15bf), identity(0x1114));
    let verdicts = [Verdict::Holds, Verdict::Holds, Verdict::Fails];
    let measured = Measured {
        merged: 13_200,
        apart: 37_300,
        held: 20_700,
        bare: 10_400,
    };
    let bytes = encode(&one, verdicts, measured);
    assert_eq!(decode(&one, &bytes), Some((verdicts, measured)));
    assert_eq!(decode(&other, &bytes), None);
    assert_ne!(record_name(&one), record_name(&other));
    assert_eq!(decode(&one, &bytes[..bytes.len() - 1]), None);
    let mut changed = bytes.clone();
    changed[60] ^= 1;
    assert_eq!(decode(&one, &changed), None);
    let pending = encode(
        &one,
        [Verdict::Holds, Verdict::Pending, Verdict::Fails],
        measured,
    );
    assert_eq!(decode(&one, &pending), None);
}
