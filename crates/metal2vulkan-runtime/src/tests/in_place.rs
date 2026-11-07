//! Textures over host pages in place (`in_place.rs`): which texture is tried as an image in place
//! and why one is a shadow, and the record that keeps the device's check's verdicts.
//!
//! - **The decision.** Under `HostPageTexturePolicy::Shadow` every texture is a shadow, the check
//!   never asked; under `InPlace` a texture a shader writes (`ShaderWrite`, `ShaderAtomic`,
//!   `Unknown`, whatever else it holds) of texels the check's kernel stores none of (1 or 2
//!   bytes) is one counted by its width, the check never asked; and every other texture is tried
//!   in place exactly where its class's check passed for each access it takes (one that renders,
//!   `RenderTarget`, its render check; one a shader writes, its storage check; one that does
//!   neither, its read check), asked in that order until one did not pass, else a shadow naming
//!   that check's answer (pending, failed, unchecked) and its access.
//! - **The record.** Its verdicts read back for the identity it was written for, and nothing else
//!   reads: a record cut short or grown, of another check version, of another identity, with a
//!   byte changed (its checksum), or holding a pending or unknown verdict.
//! - **Its name** changes with every field of the driver's identity, and with nothing else: not
//!   the runtime's version, so a pin move keeps the driver's verdicts.
//! - **A waiter** for one verdict wakes as that verdict is given, or as the check settles.

use crate::in_place::{
    decode, encode, in_place, record_name, verdict_index, Access, DriverIdentity, InPlaceChecks,
    PagesClass, ShadowCause, Verdict, CHECK_VERSION, STORED_TEXEL_BYTES, VERDICTS,
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

/// The shadow a texture of `usage` and texels of `texel` bytes takes whatever the check answers:
/// its width's where a shader writes it and the check's kernel stores no texel of that width;
/// `None` elsewhere.
fn unchecked_width(usage: TextureUsage, texel: u64) -> Option<ShadowCause> {
    let writes = usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC);
    match writes && !STORED_TEXEL_BYTES.contains(&texel) {
        true => Some(ShadowCause::StoreWidth(texel)),
        false => None,
    }
}

/// The accesses whose checks `usage` asks, in their order: renders where it holds
/// `RenderTarget`, writes where it holds `ShaderWrite` or `ShaderAtomic`, reads where it holds
/// neither.
fn accesses_of(usage: TextureUsage) -> Vec<Access> {
    let renders = usage.contains(TextureUsage::RENDER_TARGET);
    let writes = usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC);
    let mut accesses = Vec::new();
    if !renders && !writes {
        accesses.push(Access::Reads);
    }
    if renders {
        accesses.push(Access::Renders);
    }
    if writes {
        accesses.push(Access::Writes);
    }
    accesses
}

#[test]
fn the_decision_takes_a_texture_whose_class_passed_each_of_its_accesses() {
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
    // Each access's verdict, every one of them with every other.
    let mut answers = Vec::new();
    for reads in ANSWERS {
        for renders in ANSWERS {
            for writes in ANSWERS {
                answers.push([reads, renders, writes]);
            }
        }
    }
    for class in PagesClass::ALL {
        for &usage in &usages {
            for texel in [1u64, 2, 4, 8, 16] {
                for answer in &answers {
                    let of = |access: Access| answer[access.index()];
                    // Shadow: always the exact form, the check never asked.
                    let asked = std::cell::RefCell::new(Vec::new());
                    let pages = (class, texel);
                    let policy = HostPageTexturePolicy::Shadow;
                    let decided = in_place(policy, usage, pages, |_, access| {
                        asked.borrow_mut().push(access);
                        of(access)
                    });
                    assert_eq!(decided, Err(ShadowCause::Exact));
                    assert!(asked.borrow().is_empty(), "the exact form asks no check");
                    // In place: a written texture of a width no check stores never asked; the
                    // rest follow the verdicts of their accesses, asked in order until one did
                    // not pass.
                    let asked = std::cell::RefCell::new(Vec::new());
                    let policy = HostPageTexturePolicy::InPlace;
                    let decided = in_place(policy, usage, pages, |of_class, access| {
                        assert_eq!(of_class, class);
                        asked.borrow_mut().push(access);
                        of(access)
                    });
                    let (expected, due) = match unchecked_width(usage, texel) {
                        Some(cause) => (Err(cause), Vec::new()),
                        None => {
                            let accesses = accesses_of(usage);
                            let first = accesses.iter().position(|&a| of(a) != Verdict::Passed);
                            match first {
                                None => (Ok(()), accesses),
                                Some(at) => {
                                    let access = accesses[at];
                                    let cause = match of(access) {
                                        Verdict::Pending => ShadowCause::CheckPending,
                                        Verdict::Failed => ShadowCause::CheckFailed,
                                        _ => ShadowCause::Unchecked,
                                    };
                                    (Err(cause(class, access)), accesses[..=at].to_vec())
                                }
                            }
                        }
                    };
                    let case = format!("{class:?} {usage:?} {texel} {answer:?}");
                    assert_eq!(decided, expected, "{case}");
                    assert_eq!(*asked.borrow(), due, "{case}");
                }
            }
        }
    }
}

#[test]
fn every_shadow_cause_names_itself() {
    let causes = [
        ShadowCause::Exact,
        ShadowCause::StoreWidth(2),
        ShadowCause::CheckPending(PagesClass::Pointer, Access::Reads),
        ShadowCause::CheckPending(PagesClass::Pointer, Access::Renders),
        ShadowCause::CheckFailed(PagesClass::DmaBuf, Access::Reads),
        ShadowCause::CheckFailed(PagesClass::DmaBuf, Access::Renders),
        ShadowCause::Unchecked(PagesClass::Fresh, Access::Reads),
        ShadowCause::Unchecked(PagesClass::Fresh, Access::Renders),
        ShadowCause::CheckPending(PagesClass::Pointer, Access::Writes),
        ShadowCause::CheckFailed(PagesClass::DmaBuf, Access::Writes),
        ShadowCause::Unchecked(PagesClass::Fresh, Access::Writes),
        ShadowCause::StoreWidth(1),
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
    assert!(words[1].contains("ShaderWrite") && words[1].contains("not of 2"));
    assert!(words[2].contains("not answered") && words[3].contains("not answered"));
    assert!(words[3].contains("render pass"));
    assert!(words[4].contains("dma-bufs"));
    assert!(words[8].contains("not answered") && words[8].contains("storage writes"));
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
        Verdict::Unchecked,
        Verdict::Passed,
        Verdict::Failed,
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
