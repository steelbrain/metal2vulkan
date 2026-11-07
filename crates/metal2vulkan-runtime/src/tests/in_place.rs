//! Textures over host pages in place (`in_place.rs`): which texture is tried as an image in place
//! and why one is a shadow, and the record that keeps the device's check's verdicts.
//!
//! - **The decision.** Under `HostPageTexturePolicy::Shadow` every texture is a shadow, the check
//!   never asked; under `InPlace` a texture whose usage writes it through a shader
//!   (`ShaderWrite`, `ShaderAtomic`, `Unknown`, whatever else it holds) is one counted as a shader
//!   write, the check never asked; and a texture that samples, reads or renders is tried in place
//!   exactly where its class's check passed for its access (one that renders, `RenderTarget`, its
//!   render check; else its read check), else a shadow naming the check's answer (pending,
//!   failed, unchecked) and the access.
//! - **The record.** Its verdicts read back for the identity it was written for, and nothing else
//!   reads: a record cut short or grown, of another check version, of another identity, with a
//!   byte changed (its checksum), or holding a pending or unknown verdict.
//! - **Its name** changes with every field of the driver's identity, and with nothing else: not
//!   the runtime's version, so a pin move keeps the driver's verdicts.
//! - **A waiter** for one verdict wakes as that verdict is given, or as the check settles.

use crate::in_place::{
    decode, encode, in_place, record_name, verdict_index, Access, DriverIdentity, InPlaceChecks,
    PagesClass, ShadowCause, Verdict, CHECK_VERSION, VERDICTS,
};
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::TextureUsage;
use sha2::{Digest, Sha256};
use std::time::Duration;

const ANSWERS: [Verdict; 4] = [
    Verdict::Pending,
    Verdict::Passed,
    Verdict::Failed,
    Verdict::Unchecked,
];

/// Every usage bit Metal has, each alone.
fn each_usage() -> Vec<TextureUsage> {
    vec![
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_WRITE,
        TextureUsage::RENDER_TARGET,
        TextureUsage::PIXEL_FORMAT_VIEW,
        TextureUsage::SHADER_ATOMIC,
    ]
}

/// The shadow `usage` takes whatever the check answers: a shader write's where it writes the
/// texture through a shader, whatever else it holds; `None` where it samples, reads or renders.
fn written(usage: TextureUsage) -> Option<ShadowCause> {
    match usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC) {
        true => Some(ShadowCause::ShaderWrite),
        false => None,
    }
}

/// The access whose check `usage` asks: an image that renders where it holds `RenderTarget`.
fn access_of(usage: TextureUsage) -> Access {
    match usage.contains(TextureUsage::RENDER_TARGET) {
        true => Access::Renders,
        false => Access::Reads,
    }
}

#[test]
fn the_decision_takes_a_texture_that_does_not_write_whose_class_passed_its_access() {
    let mut usages = each_usage();
    usages.push(TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW);
    usages.push(TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET);
    usages.push(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE);
    usages.push(TextureUsage::RENDER_TARGET | TextureUsage::SHADER_WRITE);
    usages.push(TextureUsage::RENDER_TARGET | TextureUsage::SHADER_ATOMIC);
    usages.push(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC);
    // An `Unknown` usage, as Metal reports it: every usage.
    usages.push(
        TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::RENDER_TARGET
            | TextureUsage::PIXEL_FORMAT_VIEW
            | TextureUsage::SHADER_ATOMIC,
    );
    usages.push(TextureUsage::empty());
    for class in PagesClass::ALL {
        for &usage in &usages {
            for verdict in ANSWERS {
                // Shadow: always the exact form, the check never asked.
                let asked = std::cell::Cell::new(false);
                let decided = in_place(HostPageTexturePolicy::Shadow, usage, class, |_, _| {
                    asked.set(true);
                    verdict
                });
                assert_eq!(decided, Err(ShadowCause::Exact));
                assert!(!asked.get(), "the exact form asks no check");
                // In place: written textures never asked; the rest follow the verdict of their
                // access.
                let asked = std::cell::Cell::new(false);
                let access = access_of(usage);
                let decided = in_place(HostPageTexturePolicy::InPlace, usage, class, |of, by| {
                    assert_eq!((of, by), (class, access));
                    asked.set(true);
                    verdict
                });
                let expected = if let Some(cause) = written(usage) {
                    Err(cause)
                } else {
                    match verdict {
                        Verdict::Passed => Ok(()),
                        Verdict::Pending => Err(ShadowCause::CheckPending(class, access)),
                        Verdict::Failed => Err(ShadowCause::CheckFailed(class, access)),
                        Verdict::Unchecked => Err(ShadowCause::Unchecked(class, access)),
                    }
                };
                assert_eq!(decided, expected, "{class:?} {usage:?} {verdict:?}");
                assert_eq!(asked.get(), written(usage).is_none(), "{class:?} {usage:?}");
            }
        }
    }
}

#[test]
fn every_shadow_cause_names_itself() {
    let causes = [
        ShadowCause::Exact,
        ShadowCause::ShaderWrite,
        ShadowCause::CheckPending(PagesClass::Pointer, Access::Reads),
        ShadowCause::CheckPending(PagesClass::Pointer, Access::Renders),
        ShadowCause::CheckFailed(PagesClass::DmaBuf, Access::Reads),
        ShadowCause::CheckFailed(PagesClass::DmaBuf, Access::Renders),
        ShadowCause::Unchecked(PagesClass::Fresh, Access::Reads),
        ShadowCause::Unchecked(PagesClass::Fresh, Access::Renders),
    ];
    let words: Vec<String> = causes.iter().map(ShadowCause::words).collect();
    for (at, word) in words.iter().enumerate() {
        assert!(!word.is_empty());
        assert!(
            words.iter().skip(at + 1).all(|other| other != word),
            "{word}"
        );
    }
    assert!(words[0].contains("HostPageTexturePolicy::Shadow"));
    assert!(words[1].contains("ShaderWrite"));
    assert!(words[2].contains("not answered") && words[3].contains("not answered"));
    assert!(words[3].contains("render pass"));
    assert!(words[4].contains("dma-bufs"));
}

/// A driver identity, every field set.
fn identity() -> DriverIdentity {
    DriverIdentity {
        vendor_id: 0x1002,
        device_id: 0x15bf,
        driver_version: 0x0068_2004,
        driver_id: 3,
        driver_name: "radv".to_owned(),
        driver_info: "Mesa 26.2.4".to_owned(),
        device_uuid: [7; 16],
    }
}

/// `bytes` with their checksum (the last 32 bytes) made again over the rest.
fn rechecked(mut bytes: Vec<u8>) -> Vec<u8> {
    let body = bytes.len() - 32;
    let checksum: [u8; 32] = Sha256::digest(&bytes[..body]).into();
    bytes[body..].copy_from_slice(&checksum);
    bytes
}

#[test]
fn a_record_reads_back_for_its_identity_alone() {
    let identity = identity();
    let verdicts = [
        Verdict::Passed,
        Verdict::Failed,
        Verdict::Unchecked,
        Verdict::Passed,
        Verdict::Failed,
        Verdict::Unchecked,
    ];
    let bytes = encode(&identity, verdicts);
    assert_eq!(decode(&identity, &bytes), Some(verdicts));

    // Cut short at every length, or grown.
    for length in 0..bytes.len() {
        assert_eq!(decode(&identity, &bytes[..length]), None, "{length} bytes");
    }
    let mut grown = bytes.clone();
    grown.push(0);
    assert_eq!(decode(&identity, &grown), None);

    // Any byte changed: the checksum fails.
    for at in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[at] ^= 0x01;
        assert_eq!(decode(&identity, &changed), None, "byte {at}");
    }

    // Another check version, with a checksum that holds.
    let mut version = bytes.clone();
    version[8..12].copy_from_slice(&(CHECK_VERSION + 1).to_le_bytes());
    assert_eq!(decode(&identity, &rechecked(version)), None);

    // A pending or unknown verdict, at any of their places, with a checksum that holds.
    for at in 44..44 + VERDICTS {
        for byte in [0u8, 4, 0xff] {
            let mut verdict = bytes.clone();
            verdict[at] = byte;
            assert_eq!(
                decode(&identity, &rechecked(verdict)),
                None,
                "verdict {byte} at {at}"
            );
        }
    }

    // Another identity.
    let mut other = identity.clone();
    other.driver_version += 1;
    assert_eq!(decode(&other, &bytes), None);
}

#[test]
fn a_record_s_name_follows_every_field_of_the_identity() {
    let base = identity();
    let name = record_name(&base);
    assert_eq!(name.len(), 64);
    // The name hashes the check's tag, kind and version and the identity alone: no runtime
    // version is in it.
    let mut hash = Sha256::new();
    hash.update(b"metal2vulkan in-place check record\0");
    hash.update(b"first move\0");
    hash.update(CHECK_VERSION.to_le_bytes());
    hash.update(base.vendor_id.to_le_bytes());
    hash.update(base.device_id.to_le_bytes());
    hash.update(base.driver_version.to_le_bytes());
    hash.update(base.driver_id.to_le_bytes());
    for text in [&base.driver_name, &base.driver_info] {
        hash.update((text.len() as u64).to_le_bytes());
        hash.update(text.as_bytes());
    }
    hash.update(base.device_uuid);
    let alone: [u8; 32] = hash.finalize().into();
    let hex: String = alone.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(name, hex);
    let changes: [fn(&mut DriverIdentity); 7] = [
        |id| id.vendor_id += 1,
        |id| id.device_id += 1,
        |id| id.driver_version += 1,
        |id| id.driver_id += 1,
        |id| id.driver_name.push('x'),
        |id| id.driver_info.push('x'),
        |id| id.device_uuid[15] ^= 1,
    ];
    let mut names = vec![name];
    for change in changes {
        let mut changed = base.clone();
        change(&mut changed);
        let name = record_name(&changed);
        assert!(!names.contains(&name), "{changed:?}");
        names.push(name);
    }
    // Text fields are framed by their lengths: moving a byte between them changes the name.
    let mut moved = base.clone();
    moved.driver_name = "radvM".to_owned();
    moved.driver_info = "esa 26.2.4".to_owned();
    assert_ne!(record_name(&moved), record_name(&base));
}

#[test]
fn the_checks_state_settles_once_and_wakes_its_waiters() {
    let checks = InPlaceChecks::new(false);
    assert!(!checks.forced_failure());
    assert_eq!(checks.verdicts(), [Verdict::Pending; VERDICTS]);
    assert!(!checks.wait(Duration::from_millis(1)));
    assert!(checks.start());
    assert!(!checks.start(), "the check starts once");
    checks.set(PagesClass::Fresh, Access::Renders, Verdict::Passed);
    assert_eq!(
        checks.verdict(PagesClass::Fresh, Access::Renders),
        Verdict::Passed
    );
    assert_eq!(
        checks.verdict(PagesClass::Fresh, Access::Reads),
        Verdict::Pending
    );
    assert_eq!(
        checks.verdict(PagesClass::Pointer, Access::Renders),
        Verdict::Pending
    );
    // Each class and access has a place of its own.
    let mut places: Vec<usize> = PagesClass::ALL
        .into_iter()
        .flat_map(|class| Access::ALL.map(|access| verdict_index(class, access)))
        .collect();
    places.sort_unstable();
    assert_eq!(places, (0..VERDICTS).collect::<Vec<_>>());
    std::thread::scope(|scope| {
        let waiter = scope.spawn(|| checks.wait(Duration::from_secs(30)));
        checks.settle(Some(Duration::from_millis(5)));
        assert!(waiter.join().unwrap_or(false));
    });
    assert_eq!(checks.took(), Some(Duration::from_millis(5)));
    assert!(InPlaceChecks::new(true).forced_failure());
}

#[test]
fn a_waiter_for_a_verdict_wakes_as_it_is_given() {
    let checks = InPlaceChecks::new(false);
    let (class, access) = (PagesClass::Pointer, Access::Renders);
    // Nothing given: the wait runs out, pending.
    assert_eq!(
        checks.wait_for(class, access, Duration::from_millis(1)),
        Verdict::Pending
    );
    std::thread::scope(|scope| {
        let waiter = scope.spawn(|| checks.wait_for(class, access, Duration::from_secs(60)));
        // Another verdict wakes it to wait on; its own ends the wait.
        checks.set(class, Access::Reads, Verdict::Failed);
        checks.set(class, access, Verdict::Passed);
        assert_eq!(waiter.join().ok(), Some(Verdict::Passed));
    });
    // A settled check ends a wait for a verdict it never gave.
    checks.settle(None);
    assert_eq!(
        checks.wait_for(PagesClass::DmaBuf, access, Duration::from_secs(60)),
        Verdict::Pending
    );
}
