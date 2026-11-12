//! Textures over host pages in place (`in_place.rs`): which texture is tried as an image in place
//! and why one is a shadow, and the record that keeps the device's check's verdicts.
//!
//! - **The decision.** Under `HostPageTexturePolicy::Shadow` every texture is a shadow, the check
//!   never asked; under `InPlace` a texture a shader writes (`ShaderWrite`, `ShaderAtomic`,
//!   `Unknown`, whatever else it holds) of texels of a width the check's kernel stores none of
//!   (none of 1, 2, 4, 8 and 16 bytes) is one counted by its width, the check never asked; and
//!   every other texture is tried in place exactly where its class's check passed, for texels
//!   that lie in lines as its own do, for each access it takes (one that renders,
//!   `RenderTarget`, its render check; one a shader writes, the storage check of its texels'
//!   width, those of 1 or 2 bytes apart from the rest; one that does neither, its read check),
//!   asked in that order until one did not pass, else a shadow naming that check's answer
//!   (pending, failed, unchecked), its access and its row ends. A texture a shader writes whose
//!   rows' padding is free asks the check of writes that lets that padding change in the stead of
//!   the one that keeps it. A texture that renders takes the storage usage too exactly where no
//!   shader writes it by its usage and the storage check of its texels' width that keeps every
//!   byte passed.
//! - **The padding rule.** Rows' padding is free exactly over a buffer's own rows whose every row,
//!   to its pitch, shares no line with bytes past them; a round of storage writes that changed
//!   bytes of that padding alone (any row's, the last's included) fails the check of writes that
//!   keeps it and counts for the one that lets it change, while a change past the last row's
//!   pitch, of a texel, or of padding that is not free fails both.
//! - **Row ends.** Texels share no line with bytes beside them exactly where their first byte,
//!   their end and, with bytes between rows (or slices), each row's (or slice's) start and end lie
//!   on a line; and the check's verdicts by row ends fail both where texels that share none
//!   changed a byte, and those that share one alone where they did.
//! - **The record.** Its verdicts read back for the identity it was written for, and nothing else
//!   reads: a record cut short or grown, of another check version, of another identity, with a
//!   byte changed (its checksum), or holding a pending or unknown verdict.
//! - **Its name** changes with every field of the driver's identity, and with nothing else: not
//!   the runtime's version, so a pin move keeps the driver's verdicts.
//! - **A waiter** for one verdict wakes as that verdict is given, or as the check settles.

use crate::in_place::{
    decode, encode, in_place, record_name, storable, verdict_index, Access, DriverIdentity,
    InPlaceChecks, Padding, PagesClass, RowEnds, ShadowCause, Verdict, CHECK_VERSION,
    NARROW_TEXEL_BYTES, STORED_TEXEL_BYTES, VERDICTS,
};
use crate::in_place_check::{
    carried_store_failures, mark, near_bytes, verdicts, Bytes, Due, Found, Pattern, Round, Shape,
    Tally, Walk, Written, NARROW_STORE_SHAPES, NEAR_STEPS, SHAPES, STORE_SHAPES,
};
use crate::linear_plan::BufferLayout;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::TextureUsage;
use sha2::{Digest, Sha256};
use std::ptr::NonNull;
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

/// The check of a shader's writes of texels of `texel` bytes, a width the check's kernel stores,
/// of a texture whose rows' padding is as `padding` says: those of 1 or 2 bytes apart from the
/// rest, each in the form that lets the padding change where it is free (declared unobserved or
/// not).
fn writes_of(texel: u64, padding: Padding) -> Access {
    match (NARROW_TEXEL_BYTES.contains(&texel), padding) {
        (false, Padding::Kept) => Access::Writes,
        (false, Padding::Free | Padding::Unobserved) => Access::BufferWrites,
        (true, Padding::Kept) => Access::NarrowWrites,
        (true, Padding::Free | Padding::Unobserved) => Access::NarrowBufferWrites,
    }
}

/// The accesses whose checks `usage` asks of a texture of texels of `texel` bytes whose rows'
/// padding is as `padding` says, in their order: renders where it holds `RenderTarget` (in the
/// form that lets the padding change where its buffer declares it unobserved and it is free),
/// writes where it holds `ShaderWrite` or `ShaderAtomic` (of its texels' width, [`writes_of`]),
/// reads where it holds neither.
fn accesses_of(usage: TextureUsage, padding: Padding, texel: u64) -> Vec<Access> {
    let renders = usage.contains(TextureUsage::RENDER_TARGET);
    let writes = usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC);
    let mut accesses = Vec::new();
    if !renders && !writes {
        accesses.push(Access::Reads);
    }
    if renders {
        accesses.push(match padding {
            Padding::Unobserved => Access::UnobservedRenders,
            Padding::Kept | Padding::Free => Access::Renders,
        });
    }
    if writes {
        accesses.push(writes_of(texel, padding));
    }
    accesses
}

/// The answer after `verdict` in [`ANSWERS`]' order (the last's, the first): the narrow checks'
/// answers in the decision's test, so that each differs from its wider twin's.
fn next(verdict: Verdict) -> Verdict {
    let at = ANSWERS.iter().position(|&answer| answer == verdict);
    ANSWERS[(at.unwrap_or(0) + 1) % ANSWERS.len()]
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
    // Each access's verdict, every one of them with every other; the narrow checks' each the
    // answer after its wider twin's, and the renders that let the padding change the answer
    // after the renders that keep it, so that each takes every answer and differs from the twin's.
    let mut answers = Vec::new();
    for reads in ANSWERS {
        for renders in ANSWERS {
            for writes in ANSWERS {
                for buffer_writes in ANSWERS {
                    let narrow = [next(writes), next(buffer_writes)];
                    let unobserved = next(renders);
                    answers.push([
                        reads,
                        renders,
                        unobserved,
                        writes,
                        buffer_writes,
                        narrow[0],
                        narrow[1],
                    ]);
                }
            }
        }
    }
    let each = PagesClass::ALL.into_iter().flat_map(|class| {
        RowEnds::ALL.into_iter().flat_map(move |rows| {
            [Padding::Kept, Padding::Free, Padding::Unobserved]
                .map(|padding| (class, rows, padding))
        })
    });
    for (class, rows, padding) in each {
        for &usage in &usages {
            // A 3-byte texel, of no format a buffer backs, stands for a width the kernel stores
            // none of.
            for texel in [1u64, 2, 3, 4, 8, 16] {
                for answer in &answers {
                    let of = |access: Access| answer[access.index()];
                    // Shadow: always the exact form, the check never asked.
                    let asked = std::cell::RefCell::new(Vec::new());
                    let pages = (class, rows, texel, padding);
                    let policy = HostPageTexturePolicy::Shadow;
                    let decided = in_place(policy, usage, pages, |_, access, _| {
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
                    let decided = in_place(policy, usage, pages, |of_class, access, of_rows| {
                        assert_eq!((of_class, of_rows), (class, rows));
                        asked.borrow_mut().push(access);
                        of(access)
                    });
                    let (expected, due) = match unchecked_width(usage, texel) {
                        Some(cause) => (Err(cause), Vec::new()),
                        None => {
                            let accesses = accesses_of(usage, padding, texel);
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
                                    let cause = cause(class, access, rows);
                                    (Err(cause), accesses[..=at].to_vec())
                                }
                            }
                        }
                    };
                    let case =
                        format!("{class:?} {rows:?} {padding:?} {usage:?} {texel} {answer:?}");
                    assert_eq!(decided, expected, "{case}");
                    assert_eq!(*asked.borrow(), due, "{case}");
                    // The storage usage of a texture that renders: the check of its texels'
                    // width that keeps every byte, where no shader writes it by its usage.
                    let renders = usage.contains(TextureUsage::RENDER_TARGET);
                    let writes =
                        usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC);
                    let stores = STORED_TEXEL_BYTES.contains(&texel);
                    let due = renders
                        && !writes
                        && stores
                        && of(writes_of(texel, Padding::Kept)) == Verdict::Passed;
                    let asked = std::cell::RefCell::new(Vec::new());
                    let takes =
                        storable(usage, (class, rows, texel), |of_class, access, of_rows| {
                            assert_eq!((of_class, of_rows), (class, rows));
                            asked.borrow_mut().push(access);
                            of(access)
                        });
                    assert_eq!(takes, due, "storable: {case}");
                    let wanted = match renders && !writes && stores {
                        true => vec![writes_of(texel, Padding::Kept)],
                        false => Vec::new(),
                    };
                    assert_eq!(*asked.borrow(), wanted, "storable: {case}");
                }
            }
        }
    }
}

#[test]
fn every_shadow_cause_names_itself() {
    let (inside, on) = (RowEnds::InLines, RowEnds::OnLines);
    let causes = [
        ShadowCause::Exact,
        ShadowCause::StoreWidth(2),
        ShadowCause::CheckPending(PagesClass::Pointer, Access::Reads, inside),
        ShadowCause::CheckPending(PagesClass::Pointer, Access::Renders, inside),
        ShadowCause::CheckFailed(PagesClass::DmaBuf, Access::Reads, inside),
        ShadowCause::CheckFailed(PagesClass::DmaBuf, Access::Renders, inside),
        ShadowCause::Unchecked(PagesClass::Fresh, Access::Reads, inside),
        ShadowCause::Unchecked(PagesClass::Fresh, Access::Renders, inside),
        ShadowCause::CheckPending(PagesClass::Pointer, Access::Writes, inside),
        ShadowCause::CheckFailed(PagesClass::DmaBuf, Access::Writes, inside),
        ShadowCause::Unchecked(PagesClass::Fresh, Access::Writes, inside),
        ShadowCause::StoreWidth(1),
        ShadowCause::CheckFailed(PagesClass::Pointer, Access::Renders, inside),
        ShadowCause::CheckFailed(PagesClass::Pointer, Access::Renders, on),
        ShadowCause::CheckFailed(PagesClass::Pointer, Access::BufferWrites, inside),
        ShadowCause::CheckPending(PagesClass::Fresh, Access::BufferWrites, on),
        ShadowCause::CheckFailed(PagesClass::Pointer, Access::NarrowWrites, inside),
        ShadowCause::Unchecked(PagesClass::DmaBuf, Access::NarrowBufferWrites, on),
        ShadowCause::StoreWidth(3),
        ShadowCause::Unchecked(PagesClass::Pooled, Access::Renders, inside),
        ShadowCause::CheckFailed(PagesClass::Pooled, Access::UnobservedRenders, inside),
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
    assert!(words[16].contains("1- or 2-byte texels") && !words[16].contains("over a buffer"));
    assert!(words[17].contains("1- or 2-byte texels") && words[17].contains("over a buffer"));
    assert!(words[18].contains("1, 2, 4, 8 and 16 bytes, not of 3"));
    assert!(words[2].contains("not answered") && words[3].contains("not answered"));
    assert!(words[3].contains("render pass"));
    assert!(words[4].contains("dma-bufs"));
    assert!(words[8].contains("not answered") && words[8].contains("storage writes"));
    assert!(words[12].contains("share a line") && words[13].contains("share no line"));
    assert!(words[14].contains("over a buffer's rows, beside their padding"));
    assert!(words[15].contains("not answered") && words[15].contains("over a buffer's rows"));
    assert!(words[19].contains("pooled memory"));
    assert!(words[20].contains("render pass") && words[20].contains("declares unobserved"));
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
    // Each verdict a record holds: measured, unchecked for a cause that lasts, and unmeasured (an
    // unchecked one whose cause may pass, written pending for the next device's check).
    let held = [
        Verdict::Passed,
        Verdict::Failed,
        Verdict::Unchecked,
        Verdict::Pending,
    ];
    let verdicts: [Verdict; VERDICTS] = std::array::from_fn(|at| held[at % held.len()]);
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

    // An unknown verdict, at any of their places, with a checksum that holds: never written.
    for at in 44..44 + VERDICTS {
        for byte in [4u8, 0x80, 0xff] {
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
    let checks = InPlaceChecks::new(false, 64);
    assert_eq!(checks.line(), 64);
    assert_eq!(
        InPlaceChecks::new(false, 0).line(),
        1,
        "a line is a byte at least"
    );
    assert!(!checks.forced_failure());
    assert_eq!(checks.verdicts(), [Verdict::Pending; VERDICTS]);
    assert!(!checks.wait(Duration::from_millis(1)));
    assert!(checks.start());
    assert!(!checks.start(), "the check starts once");
    let (inside, on) = (RowEnds::InLines, RowEnds::OnLines);
    checks.set(PagesClass::Fresh, Access::Renders, on, Verdict::Passed);
    assert_eq!(
        checks.verdict(PagesClass::Fresh, Access::Renders, on),
        Verdict::Passed
    );
    assert_eq!(
        checks.verdict(PagesClass::Fresh, Access::Renders, inside),
        Verdict::Pending
    );
    assert_eq!(
        checks.verdict(PagesClass::Fresh, Access::Reads, on),
        Verdict::Pending
    );
    assert_eq!(
        checks.verdict(PagesClass::Pointer, Access::Renders, on),
        Verdict::Pending
    );
    // Each class, access and row ends has a place of its own, those of texels that share a line
    // first, as the record holds them.
    let mut places: Vec<usize> = Vec::new();
    for rows in RowEnds::ALL {
        for class in PagesClass::ALL {
            places.extend(Access::ALL.map(|access| verdict_index(class, access, rows)));
        }
    }
    assert_eq!(places, (0..VERDICTS).collect::<Vec<_>>());
    std::thread::scope(|scope| {
        let waiter = scope.spawn(|| checks.wait(Duration::from_secs(30)));
        checks.settle(Some(Duration::from_millis(5)));
        assert!(waiter.join().unwrap_or(false));
    });
    assert_eq!(checks.took(), Some(Duration::from_millis(5)));
    assert!(InPlaceChecks::new(true, 64).forced_failure());
}

/// A texture's layout in a buffer: `rows` rows of `row_bytes` bytes, `row_pitch` apart, from
/// `offset`, in one slice.
fn rows_of(offset: u64, row_bytes: u64, row_pitch: u64, rows: u64) -> BufferLayout {
    BufferLayout {
        offset,
        row_pitch,
        slice_pitch: row_pitch * rows,
        row_bytes,
        rows,
        slices: 1,
    }
}

/// The shapes at a driver's own pitch run only where no shape of their texel size whose texels
/// share a line was made before them in the class (`Shape::runs`): none on a device that makes
/// every shape at its pitch, and those at its own pitch on a device that lays rows at the next
/// multiple of 16 bytes (MoltenVK, its line 16 bytes) or 64 (llvmpipe, its line 64) and makes no
/// other image with bytes between rows; the 64-byte pair none on the first.
#[test]
fn shapes_at_a_driver_s_own_pitch_run_only_where_none_of_their_texel_size_was_made() {
    // A class's walk as `check_class` takes it: the shapes whose texels share no line first, each
    // where `Shape::runs` takes it, made where `makes` says; the shapes it tried.
    let walk = |shapes: &[Shape], line: u64, makes: &dyn Fn(&Shape) -> bool| {
        let mut shapes = shapes.to_vec();
        shapes.sort_by_key(|shape| shape.rows(line) != RowEnds::OnLines);
        let mut made = Vec::new();
        let mut tried = Vec::new();
        for shape in shapes {
            if !shape.runs(line, &made) {
                continue;
            }
            tried.push(shape);
            let shares = shape.rows(line) == RowEnds::InLines;
            if makes(&shape) && shares && !made.contains(&shape.texel) {
                made.push(shape.texel);
            }
        }
        tried
    };
    let pitch_of =
        |shape: &Shape, multiple: u64| (shape.width * shape.texel).next_multiple_of(multiple);
    let tried_at = |tried: &[Shape], multiple: u64| {
        let at = |shape: &&Shape| shape.tight && shape.bytes_per_row == pitch_of(shape, multiple);
        tried.iter().filter(at).count()
    };
    let store: Vec<Shape> = STORE_SHAPES.iter().map(|(shape, _)| *shape).collect();
    let narrow: Vec<Shape> = NARROW_STORE_SHAPES
        .iter()
        .map(|(shape, _)| *shape)
        .collect();
    for shapes in [SHAPES.to_vec(), store, narrow] {
        let tight = |multiple| {
            shapes
                .iter()
                .filter(|shape| shape.tight && shape.bytes_per_row == pitch_of(shape, multiple))
                .count()
        };
        assert_eq!((tight(16), tight(64)), (2, 2));
        // Every shape made at its pitch (RADV, ANV, NVIDIA): no shape at a driver's own pitch.
        let tried = walk(&shapes, 64, &|_| true);
        assert!(tried.iter().all(|shape| !shape.tight));
        // Rows at the next multiple of 16 or 64 bytes alone, or one after another.
        for (multiple, line) in [(16, 16), (64, 64)] {
            let makes = |shape: &Shape| {
                let row = shape.width * shape.texel;
                shape.bytes_per_row == row || shape.bytes_per_row == row.next_multiple_of(multiple)
            };
            let tried = walk(&shapes, line, &makes);
            assert_eq!(tried_at(&tried, multiple), 2, "{multiple}-byte rows");
        }
        let tried = walk(&shapes, 16, &|shape: &Shape| {
            let row = shape.width * shape.texel;
            shape.bytes_per_row == row || shape.bytes_per_row == row.next_multiple_of(16)
        });
        assert_eq!(
            tried_at(&tried, 64),
            0,
            "the 64-byte pair after the 16-byte one"
        );
    }
}

#[test]
fn texels_share_no_line_exactly_where_each_gap_starts_and_ends_on_one() {
    let (inside, on) = (RowEnds::InLines, RowEnds::OnLines);
    // A 3840x2160 BGRA8 surface: 15360-byte rows, 240 lines each.
    assert_eq!(RowEnds::of(0, &rows_of(0, 15360, 15360, 2160), 64), on);
    assert_eq!(RowEnds::of(0, &rows_of(0, 15360, 16384, 2160), 64), on);
    // A 3662-wide one at a 14656-byte pitch: 8 bytes of each row's last line lie beside it.
    assert_eq!(RowEnds::of(0, &rows_of(0, 14648, 14656, 2039), 64), inside);
    // Rows one after another share lines with each other alone, but the image's end must lie
    // on one.
    assert_eq!(RowEnds::of(0, &rows_of(0, 61, 61, 64), 64), on);
    assert_eq!(RowEnds::of(0, &rows_of(0, 61, 61, 37), 64), inside);
    // The first byte off a line, by the buffer's address or by the texture's offset.
    assert_eq!(RowEnds::of(16, &rows_of(0, 256, 256, 16), 64), inside);
    assert_eq!(RowEnds::of(0, &rows_of(32, 256, 256, 16), 64), inside);
    assert_eq!(RowEnds::of(32, &rows_of(32, 256, 256, 16), 64), on);
    // The pitch off a line, with bytes between rows.
    assert_eq!(RowEnds::of(0, &rows_of(0, 256, 288, 16), 64), inside);
    // One row: no bytes between rows, its end on a line.
    assert_eq!(RowEnds::of(0, &rows_of(0, 320, 512, 1), 64), on);
    // Slices with bytes between them, each starting on a line or not.
    let slices = |slice_pitch| BufferLayout {
        offset: 0,
        row_pitch: 256,
        slice_pitch,
        row_bytes: 256,
        rows: 4,
        slices: 3,
    };
    assert_eq!(RowEnds::of(0, &slices(1024), 64), on);
    assert_eq!(RowEnds::of(0, &slices(1088), 64), on);
    assert_eq!(RowEnds::of(0, &slices(1056), 64), inside);
    // A line of a byte: every byte its own line.
    assert_eq!(RowEnds::of(3, &rows_of(5, 61, 67, 37), 1), on);
    // The check's shapes at 64-byte lines: seven share none, nine share one.
    let rows = |line| SHAPES.map(|shape| shape.rows(line));
    let shared = rows(64).iter().filter(|&&rows| rows == inside).count();
    assert_eq!((SHAPES.len() - shared, shared), (7, 9));
    // A driver that makes no linear image with bytes between rows but at the next multiple of its
    // own (MoltenVK's 16 bytes, its line 16 bytes; llvmpipe's 64, its line 64; NVIDIA's 128 and
    // RADV's 256, their lines 64) has shapes at that pitch whose texels share a line, for each
    // access's shapes.
    for (multiple, line) in [(16, 16), (64, 64), (128, 64), (256, 64)] {
        let at = |shape: &Shape| {
            let pitch = (shape.width * shape.texel).next_multiple_of(multiple);
            shape.bytes_per_row == pitch && shape.rows(line) == inside
        };
        assert!(SHAPES.iter().any(at), "{multiple}-byte rows");
        for stored in [&STORE_SHAPES[..], &NARROW_STORE_SHAPES[..]] {
            let shapes = stored.iter().map(|(shape, _)| shape);
            assert!(shapes.clone().any(at), "{multiple}-byte rows");
        }
    }
    // At the 256-byte lines of a device whose atom is that long, two still share none.
    assert!(rows(256).iter().filter(|&&rows| rows == on).count() >= 2);
    // And each row ends has shapes of each check of storage writes.
    for rows in RowEnds::ALL {
        assert!(STORE_SHAPES.iter().any(|(shape, _)| shape.rows(64) == rows));
        assert!(NARROW_STORE_SHAPES
            .iter()
            .any(|(shape, _)| shape.rows(64) == rows));
    }
}

#[test]
fn a_change_beside_texels_that_share_no_line_fails_both_row_ends() {
    use Verdict::{Failed, Passed, Unchecked};
    let (inside, on) = (RowEnds::InLines, RowEnds::OnLines);
    assert_eq!(verdicts([9, 21], None, false), [Passed, Passed]);
    assert_eq!(verdicts([9, 21], None, true), [Failed, Failed]);
    // Texels that share a line changed a byte: those that share none keep their rounds' answer.
    assert_eq!(verdicts([0, 21], Some(inside), false), [Failed, Passed]);
    assert_eq!(verdicts([0, 21], Some(inside), true), [Failed, Failed]);
    assert_eq!(verdicts([0, 0], Some(inside), false), [Failed, Unchecked]);
    // Texels that share none changed one: both fail.
    assert_eq!(verdicts([0, 4], Some(on), false), [Failed, Failed]);
    // Row ends no round kept every byte for are unchecked.
    assert_eq!(verdicts([0, 21], None, false), [Unchecked, Passed]);
    assert_eq!(verdicts([9, 0], None, false), [Passed, Unchecked]);
}

/// The Iris Xe's verdicts of one fresh check in three (ms01, job 7688): the dma-buf class's
/// storage-write rounds of rows that end inside lines saw no change where the pointer and
/// fresh-pages classes' did. A store failure of a class whose reads of those row ends passed
/// fails the others' passes of that access and row ends; a reads failure, an unchecked verdict,
/// a store failure of a class whose reads failed, and one at a texel carry nowhere.
#[test]
fn a_store_failure_fails_the_other_classes_passes_and_nothing_else_carries() {
    use Access::{BufferWrites, NarrowBufferWrites, NarrowWrites, Reads, Renders, Writes};
    use PagesClass::{DmaBuf, Fresh, Pointer};
    use RowEnds::{InLines, OnLines};
    use Verdict::{Failed, Passed, Unchecked};
    let mut found: Vec<Found> = Vec::new();
    for class in [Pointer, Fresh, DmaBuf] {
        for rows in [InLines, OnLines] {
            for access in Access::ALL {
                found.push(Found::new(class, access, rows, Passed));
            }
        }
    }
    fn set(
        found: &mut [Found],
        class: PagesClass,
        access: Access,
        rows: RowEnds,
        verdict: Verdict,
    ) {
        let at = (found.iter()).position(|f| (f.class, f.access, f.rows) == (class, access, rows));
        found[at.unwrap()].verdict = verdict;
    }
    // The Iris Xe: renders and storage writes of rows that end inside lines fail in the pointer
    // and fresh-pages classes; the dma-buf class's storage writes passed in that check.
    for class in [Pointer, Fresh] {
        set(&mut found, class, Renders, InLines, Failed);
        set(&mut found, class, Writes, InLines, Failed);
    }
    set(&mut found, DmaBuf, Renders, InLines, Failed);
    // The dma-buf class's buffer writes of rows on lines: unchecked (no dispatch made).
    set(&mut found, DmaBuf, BufferWrites, OnLines, Unchecked);
    // The fresh-pages class's reads of rows on lines failed, and so did its buffer writes there:
    // the first move's, which says nothing of the other classes.
    set(&mut found, Fresh, Reads, OnLines, Failed);
    set(&mut found, Fresh, BufferWrites, OnLines, Failed);
    assert_eq!(
        carried_store_failures(&found),
        [(DmaBuf, Writes, InLines, Pointer)],
        "the dma-buf class's pass of storage writes inside lines alone fails, as the pointer \
         class's failure"
    );
    // A failure the dma-buf class's own rounds found carries to the others' passes too.
    set(&mut found, Pointer, Writes, InLines, Passed);
    set(&mut found, Fresh, Writes, InLines, Passed);
    set(&mut found, DmaBuf, Writes, InLines, Failed);
    assert_eq!(
        carried_store_failures(&found),
        [
            (Pointer, Writes, InLines, DmaBuf),
            (Fresh, Writes, InLines, DmaBuf)
        ]
    );
    // No store failure: nothing carries, whatever reads did.
    set(&mut found, DmaBuf, Writes, InLines, Passed);
    for class in [Pointer, Fresh, DmaBuf] {
        set(&mut found, class, Renders, InLines, Passed);
        set(&mut found, class, Reads, InLines, Failed);
    }
    assert!(carried_store_failures(&found).is_empty());
    // A store failure at a texel (the CPU did not see the GPU's stores through the class's
    // mapping) carries nowhere; one beside the texels does.
    for class in [Pointer, Fresh, DmaBuf] {
        set(&mut found, class, Reads, InLines, Passed);
    }
    set(&mut found, Pointer, Writes, InLines, Failed);
    for f in found
        .iter_mut()
        .filter(|f| f.class == Pointer && f.access == Writes)
    {
        f.beside = false;
    }
    assert!(carried_store_failures(&found).is_empty());
    for f in found
        .iter_mut()
        .filter(|f| f.class == Pointer && f.access == Writes)
    {
        f.beside = true;
    }
    assert_eq!(
        carried_store_failures(&found),
        [
            (Fresh, Writes, InLines, Pointer),
            (DmaBuf, Writes, InLines, Pointer)
        ]
    );
    // The narrow rounds' stores carry as the others' do: storage writes of 1- and 2-byte texels,
    // and their buffer writes.
    set(&mut found, Pointer, Writes, InLines, Passed);
    set(&mut found, Fresh, NarrowWrites, InLines, Failed);
    set(&mut found, DmaBuf, NarrowBufferWrites, OnLines, Failed);
    assert_eq!(
        carried_store_failures(&found),
        [
            (Pointer, NarrowWrites, InLines, Fresh),
            (Pointer, NarrowBufferWrites, OnLines, DmaBuf),
            (Fresh, NarrowBufferWrites, OnLines, DmaBuf),
            (DmaBuf, NarrowWrites, InLines, Fresh)
        ]
    );
    // Renders over padding declared unobserved store as renders do: a byte past their padding
    // one class's rounds changed fails the others' passes.
    set(&mut found, Fresh, NarrowWrites, InLines, Passed);
    set(&mut found, DmaBuf, NarrowBufferWrites, OnLines, Passed);
    set(
        &mut found,
        Pointer,
        Access::UnobservedRenders,
        InLines,
        Failed,
    );
    assert_eq!(
        carried_store_failures(&found),
        [
            (Fresh, Access::UnobservedRenders, InLines, Pointer),
            (DmaBuf, Access::UnobservedRenders, InLines, Pointer)
        ]
    );
}

#[test]
fn rows_padding_is_free_over_a_buffer_s_own_rows_that_end_on_lines() {
    let (kept, free) = (Padding::Kept, Padding::Free);
    // The guest's surface: 3662 BGRA8 texels a row at a 14656-byte pitch, 2039 rows. Its texels
    // share a line with its padding; its rows, each to its pitch, share none with bytes past them.
    let surface = rows_of(0, 14648, 14656, 2039);
    assert_eq!(RowEnds::of(0, &surface, 64), RowEnds::InLines);
    assert_eq!(Padding::of(0, &surface, 64, true), free);
    // Not the buffer's own rows (an IOSurface plane's, or a buffer that ends before the last
    // row's pitch): kept.
    assert_eq!(Padding::of(0, &surface, 64, false), kept);
    // The last row's pitch ends inside a line: kept. 416 x 51 bytes end 32 bytes into a 64-byte
    // line, and on a 16-byte one.
    let tight = rows_of(0, 404, 416, 51);
    assert_eq!(Padding::of(0, &tight, 64, true), kept);
    assert_eq!(Padding::of(0, &tight, 16, true), free);
    // The first texel off a line, by the buffer's address or by the texture's offset: kept; both
    // together back on one: free.
    assert_eq!(Padding::of(16, &surface, 64, true), kept);
    assert_eq!(
        Padding::of(0, &rows_of(16, 14648, 14656, 2039), 64, true),
        kept
    );
    assert_eq!(
        Padding::of(16, &rows_of(48, 14648, 14656, 2039), 64, true),
        free
    );
    // A pitch off a line: a line between two rows holds one's padding and the next one's texels
    // alone, so the padding is free where the rows start and end on lines.
    assert_eq!(Padding::of(0, &rows_of(0, 256, 288, 16), 64, true), free);
    // Bytes between slices are no row's padding: slices apart must each start and end on lines.
    let slices = |slice_pitch| BufferLayout {
        offset: 0,
        row_pitch: 288,
        slice_pitch,
        row_bytes: 256,
        rows: 4,
        slices: 3,
    };
    assert_eq!(Padding::of(0, &slices(1152), 64, true), free);
    assert_eq!(Padding::of(0, &slices(1216), 64, true), free);
    assert_eq!(Padding::of(0, &slices(1184), 64, true), kept);
    // A buffer that declares its padding unobserved lets a render pass change it only where it
    // is free: the rows, each to its pitch, share no line with bytes past them.
    assert_eq!(free.declared(true), Padding::Unobserved);
    assert_eq!(free.declared(false), free);
    assert_eq!(kept.declared(true), kept);
    let cache = rows_of(0, 688, 704, 67);
    assert_eq!(
        Padding::of(0, &cache, 64, true).declared(true),
        Padding::Unobserved
    );
    assert_eq!(Padding::of(32, &cache, 64, true).declared(true), kept);
    assert_eq!(Padding::of(0, &cache, 64, false).declared(true), kept);
    // The check's storage shapes: free at 16-byte lines, and at 64-byte ones but for the pair at
    // the 16-byte pitch of a driver that lays rows tight, whose rows end inside such a line.
    for (shape, _) in STORE_SHAPES.into_iter().chain(NARROW_STORE_SHAPES) {
        assert_eq!(shape.padding(16), free, "{shape:?}");
        let tight16 = shape.tight && shape.bytes_per_row % 64 != 0;
        let due = match tight16 {
            true => kept,
            false => free,
        };
        assert_eq!(shape.padding(64), due, "{shape:?}");
    }
}

#[test]
fn a_round_of_storage_writes_tells_a_change_of_free_padding_apart() {
    // A 101x51 shape of 4-byte texels at a 512-byte pitch, of the check's storage shapes: 404
    // bytes of texels a row, then 108 of padding, and the window's bytes after the image.
    let (shape, _) = STORE_SHAPES[3];
    assert_eq!(
        (shape.width, shape.height, shape.texel, shape.bytes_per_row),
        (101, 51, 4, 512)
    );
    assert_eq!(shape.padding(64), Padding::Free);
    let (pitch, row) = (shape.bytes_per_row, shape.width * shape.texel);
    let image = pitch * shape.height;
    let length = image + 4096;
    let pattern = Pattern::new(7);
    // A round's compare of the window as it leaves it: the pattern, every texel zero (as after a
    // pass's clear), the byte at `change` flipped where one is named, and `written` the writer's.
    let compare = |change: Option<u64>, free: bool, written: Written| {
        let mut window: Vec<u8> = (0..length).map(|offset| pattern.at(offset)).collect();
        for y in 0..shape.height {
            window[(y * pitch) as usize..(y * pitch + row) as usize].fill(0);
        }
        if let Some(at) = change {
            window[at as usize] ^= 0x01;
        }
        let bytes = Bytes {
            start: NonNull::new(window.as_mut_ptr()).unwrap(),
            length,
        };
        bytes.compare(&pattern, (Some((shape, Due::Zero)), free), written)
    };
    // The writer wrote nothing (its first sweep reached no byte) and found nothing.
    let idle = || Written {
        sweep: 1,
        reached: 0,
        near: Vec::new(),
        near_sweep: 1,
        near_reached: 0,
        changed: None,
        padding: None,
    };
    let flipped = |at: u64, base: u8| (at, base, base ^ 0x01);
    assert_eq!(compare(None, true, idle()), Round::Kept);
    // A byte of a row's padding, the first row's, the last row's and the last row's last before
    // its pitch: padding alone where it is free, a change where it is not.
    let last = (shape.height - 1) * pitch;
    for at in [row, pitch + row + 50, last + row, image - 1] {
        let changed = flipped(at, pattern.at(at));
        assert_eq!(
            compare(Some(at), true, idle()),
            Round::Padded(changed),
            "{at}"
        );
        assert_eq!(
            compare(Some(at), false, idle()),
            Round::Changed(changed),
            "{at}"
        );
    }
    // The first byte past the last row's pitch, and a texel: a change either way.
    let past = flipped(image, pattern.at(image));
    assert_eq!(compare(Some(image), true, idle()), Round::Changed(past));
    assert_eq!(
        compare(Some(last + 4), true, idle()),
        Round::Changed(flipped(last + 4, 0))
    );
    // What the writer found: a byte of the padding, kept to the end; another byte, a change.
    let found = (row + 3, 0x11, 0x22);
    let padded = || Written {
        padding: Some(found),
        ..idle()
    };
    assert_eq!(compare(None, true, padded()), Round::Padded(found));
    assert_eq!(compare(Some(image), true, padded()), Round::Changed(past));
    let changed = Written {
        changed: Some((image + 9, 0x11, 0x22)),
        ..padded()
    };
    assert_eq!(
        compare(None, true, changed),
        Round::Changed((image + 9, 0x11, 0x22))
    );
}

#[test]
fn a_round_of_renders_keeps_its_unobserved_form_only_for_bytes_inside_its_rows_padding() {
    // The 172x67 shape of 4-byte texels at a 704-byte pitch: 688 bytes of texels a row, then 16
    // of padding inside the 64-byte line that holds the row's last texels; the last row's pitch
    // ends at 47168 bytes, 64 past a multiple of 128 and of 256, and the window's bytes after it.
    let shape = (SHAPES.iter())
        .find(|shape| (shape.width, shape.height) == (172, 67))
        .copied()
        .unwrap();
    assert_eq!((shape.texel, shape.bytes_per_row), (4, 704));
    assert_eq!(
        (shape.rows(64), shape.padding(64)),
        (RowEnds::InLines, Padding::Free)
    );
    let (pitch, row) = (shape.bytes_per_row, shape.width * shape.texel);
    let image = pitch * shape.height;
    assert_eq!((image % 128, image % 256), (64, 64));
    let length = image + 4096;
    let pattern = Pattern::new(3);
    // A round's compare of the window as a pass left it: the pattern, every texel zero (its
    // clear), and each byte of `changed` flipped.
    let compare = |changed: &[u64], free: bool| {
        let mut window: Vec<u8> = (0..length).map(|offset| pattern.at(offset)).collect();
        for y in 0..shape.height {
            window[(y * pitch) as usize..(y * pitch + row) as usize].fill(0);
        }
        for &at in changed {
            window[at as usize] ^= 0x01;
        }
        let bytes = Bytes {
            start: NonNull::new(window.as_mut_ptr()).unwrap(),
            length,
        };
        let idle = Written {
            sweep: 1,
            reached: 0,
            near: Vec::new(),
            near_sweep: 1,
            near_reached: 0,
            changed: None,
            padding: None,
        };
        bytes.compare(&pattern, (Some((shape, Due::Zero)), free), idle)
    };
    let flipped = |at: u64| (at, pattern.at(at), pattern.at(at) ^ 0x01);
    let last = (shape.height - 1) * pitch;
    // The Iris Xe's write-back of the 64-byte line that holds each row's last texels: the rows'
    // padding alone, the last row's to its pitch: the unobserved form keeps it, the other not.
    let lines: Vec<u64> = (0..shape.height)
        .flat_map(|y| y * pitch + row..(y + 1) * pitch)
        .collect();
    assert_eq!(compare(&lines, true), Round::Padded(flipped(row)));
    assert_eq!(compare(&lines, false), Round::Changed(flipped(row)));
    // A write-back of 128 bytes around the last row's last texels: its padding, then 64 bytes
    // past its pitch, which are no texture's padding: a change, whatever the padding.
    let wide: Vec<u64> = (last + row..image + 64).collect();
    assert_eq!(compare(&wide, true), Round::Changed(flipped(image)));
    // The first byte past the last row's pitch alone, and a texel: a change either way.
    assert_eq!(compare(&[image], true), Round::Changed(flipped(image)));
    assert_eq!(
        compare(&[last + 4], true),
        Round::Changed((last + 4, 0, 0x01))
    );
}

#[test]
fn a_round_that_changed_padding_alone_fails_only_the_check_of_writes_that_keeps_it() {
    use Verdict::{Failed, Passed, Unchecked};
    let (inside, on) = (RowEnds::InLines, RowEnds::OnLines);
    // The Iris Xe's walk: shapes whose texels share no line keep every byte; those that share
    // one change their rows' padding alone (the line of a row's last texels, written back whole).
    let mut tally = Tally::default();
    for _ in 0..12 {
        assert!(!tally.kept(on, false));
    }
    assert!(tally.kept(inside, true), "the first such round is noted");
    assert!(!tally.kept(inside, true));
    assert_eq!(
        tally.verdicts(Access::Writes, None, false),
        vec![[Failed, Passed], [Passed, Passed]]
    );
    // The narrow check's rounds answer its own two alike, for 1- and 2-byte texels.
    assert_eq!(
        tally.verdicts(Access::NarrowWrites, None, false),
        vec![[Failed, Passed], [Passed, Passed]]
    );
    // The renders' rounds answer theirs and those of renders over padding declared unobserved
    // alike; the reads' rounds their own alone.
    assert_eq!(
        tally.verdicts(Access::Renders, None, false),
        vec![[Failed, Passed], [Passed, Passed]]
    );
    assert_eq!(
        tally.verdicts(Access::Reads, None, false),
        vec![[Failed, Passed]]
    );
    // A byte past the padding changed too: both fail for its row ends.
    assert_eq!(
        tally.verdicts(Access::Writes, Some(inside), false),
        vec![[Failed, Passed], [Failed, Passed]]
    );
    // Padding changed beside texels that share no line: the check that keeps it fails for both
    // row ends, the other passes.
    let mut tally = Tally::default();
    assert!(tally.kept(on, true));
    assert!(!tally.kept(inside, false));
    assert_eq!(
        tally.verdicts(Access::Writes, None, false),
        vec![[Failed, Failed], [Passed, Passed]]
    );
    assert_eq!(
        tally.verdicts(Access::Writes, Some(on), false),
        vec![[Failed, Failed], [Failed, Failed]]
    );
    // Row ends no round kept are unchecked, and a forced failure fails the rest.
    let mut tally = Tally::default();
    tally.kept(on, false);
    assert_eq!(
        tally.verdicts(Access::Writes, None, false),
        vec![[Unchecked, Passed], [Unchecked, Passed]]
    );
    assert_eq!(
        tally.verdicts(Access::Writes, None, true),
        vec![[Unchecked, Failed], [Unchecked, Failed]]
    );
}

/// Each check of storage writes stores its texels' width through a view of that width: the narrow
/// shapes 1- and 2-byte texels (`R8Uint` through `R8_UINT`, `R16Uint` through `R16_UINT`), each a
/// word of the kernel's buffers, with rows that end inside a word and odd heights among each
/// width's; the others 4, 8 and 16 bytes, each its view's words. Every narrow shape's window fits
/// a class's buffer.
#[test]
fn each_check_of_storage_writes_stores_its_texels_width() {
    use metal2vulkan::bind::ShadowTexelView;
    use metal2vulkan::metal::PixelFormat;
    for (shape, view) in NARROW_STORE_SHAPES {
        let due = match shape.texel {
            1 => (PixelFormat::R8Uint, ShadowTexelView::R8Uint),
            2 => (PixelFormat::R16Uint, ShadowTexelView::R16Uint),
            other => panic!("a narrow shape of {other}-byte texels"),
        };
        assert_eq!((shape.format, view), due, "{shape:?}");
        assert_eq!(view.words(), 1, "{shape:?}");
        assert!(shape.width * shape.height * 4 <= 40 << 20, "{shape:?}");
    }
    for (shape, view) in STORE_SHAPES {
        assert!(!NARROW_TEXEL_BYTES.contains(&shape.texel), "{shape:?}");
        assert_eq!(4 * u64::from(view.words()), shape.texel, "{shape:?}");
    }
    for texel in NARROW_TEXEL_BYTES {
        let ends_inside = |(shape, _): &&(Shape, ShadowTexelView)| {
            shape.texel == texel
                && !(shape.width * shape.texel).is_multiple_of(4)
                && shape.height % 2 == 1
        };
        assert!(
            NARROW_STORE_SHAPES.iter().any(|shape| ends_inside(&shape)),
            "{texel}-byte texels"
        );
    }
}

/// The check's shapes whose texels share a line of 64 bytes (the Iris Xe's) with bytes beside
/// them: of renders, of storage writes and of storage writes of 1- and 2-byte texels.
fn sharing_shapes() -> Vec<Shape> {
    let stored = STORE_SHAPES.into_iter().chain(NARROW_STORE_SHAPES);
    (SHAPES.into_iter())
        .chain(stored.map(|(shape, _)| shape))
        .filter(|shape| shape.rows(64) == RowEnds::InLines)
        .collect()
}

/// Whether the byte at `offset` is one of `shape`'s texels, its texture starting byte 0.
fn texel(shape: &Shape, offset: u64) -> bool {
    offset < shape.bytes_per_row * shape.height
        && offset % shape.bytes_per_row < shape.width * shape.texel
}

#[test]
fn a_round_s_writer_comes_back_to_the_bytes_beside_the_texels_in_their_lines_many_times_a_sweep() {
    let line = 64;
    let shapes = sharing_shapes();
    assert!(shapes.len() >= 10, "{}", shapes.len());
    for shape in shapes {
        let length = shape.bytes_per_row * shape.height + 8192;
        let mut window = vec![0u8; length as usize];
        let bytes = Bytes {
            start: NonNull::new(window.as_mut_ptr()).unwrap(),
            length,
        };
        // The bytes beside the texels in their lines are exactly the bytes, but texels, of the
        // lines that hold a texel.
        let near = near_bytes(shape, line, length);
        let holds_texel = |offset: u64| {
            (offset - offset % line..)
                .take(line as usize)
                .any(|at| texel(&shape, at))
        };
        let due: Vec<u64> = (0..length)
            .filter(|&offset| !texel(&shape, offset) && holds_texel(offset))
            .collect();
        assert_eq!(near, due, "{shape:?}");
        assert!(!near.is_empty(), "{shape:?}");
        // Each of them comes back within a pass round them and the stride's bytes between its
        // steps there, wherever the walk starts: a walk of the window's stride alone came back
        // to a byte once a sweep of the window, past which a device's short read and write-back
        // of its line met no write of the writer's.
        let count = near.len() as u64;
        let bound = count + count.div_ceil(u64::from(NEAR_STEPS)) + 1;
        let mut walk = Walk::new(&bytes, Some(shape), line);
        let mut seen = vec![None; length as usize];
        for step in 0..4 * bound {
            let (offset, _) = walk.step();
            assert!(!texel(&shape, offset), "{shape:?}: a texel at {offset}");
            if near.binary_search(&offset).is_ok() {
                if let Some(before) = seen[offset as usize] {
                    let gap = step - before;
                    assert!(gap <= bound, "{shape:?}: byte {offset} after {gap} steps");
                }
                seen[offset as usize] = Some(step);
            }
        }
        assert!(
            near.iter().all(|&offset| seen[offset as usize].is_some()),
            "{shape:?}"
        );
        // And the stride's walk still meets every other byte but the texels once a sweep.
        let mut walk = Walk::new(&bytes, Some(shape), line);
        let others = (0..length)
            .filter(|&offset| !texel(&shape, offset) && near.binary_search(&offset).is_err())
            .count() as u64;
        let mut met = vec![false; length as usize];
        for _ in 0..others * (u64::from(NEAR_STEPS) + 1) {
            met[walk.step().0 as usize] = true;
        }
        let missed = (0..length)
            .filter(|&offset| !texel(&shape, offset) && !met[offset as usize])
            .count();
        assert_eq!(missed, 0, "{shape:?}");
    }
}

#[test]
fn a_round_finds_a_byte_beside_the_texels_in_their_line_put_back_as_the_walk_found_it_before() {
    // The 61x37 shape of 1-byte texels at a 256-byte pitch, of the check's narrow storage shapes:
    // 3 bytes after each row's 61 texels in their 64-byte line, of which a round of the Iris Xe's
    // storage writes caught the line's write-back in 2 of 100 rounds with a walk of the stride
    // alone.
    let (shape, _) = NARROW_STORE_SHAPES[2];
    assert_eq!(
        (shape.width, shape.height, shape.texel, shape.bytes_per_row),
        (61, 37, 1, 256)
    );
    let line = 64;
    let length = shape.bytes_per_row * shape.height + 8192;
    let pattern = Pattern::new(5);
    let mut window: Vec<u8> = (0..length).map(|offset| pattern.at(offset)).collect();
    let bytes = Bytes {
        start: NonNull::new(window.as_mut_ptr()).unwrap(),
        length,
    };
    // The writer's walk for some steps, as it writes; then its texels cleared, as a pass does.
    let mut walk = Walk::new(&bytes, Some(shape), line);
    let near = near_bytes(shape, line, length);
    assert_eq!(near.len(), 3 * 37);
    let steps = 3 * near.len() as u64 * u64::from(NEAR_STEPS + 1) + 17;
    let mut last = std::collections::HashMap::new();
    for _ in 0..steps {
        let (offset, sweep) = walk.step();
        let previous = last.insert(offset, sweep).unwrap_or(0);
        assert!(sweep > previous);
        // SAFETY: the window's byte, which nothing else uses.
        unsafe {
            bytes
                .start
                .as_ptr()
                .add(offset as usize)
                .write(pattern.at(offset) ^ mark(sweep))
        };
    }
    let texels = |window: &mut Vec<u8>| {
        for y in 0..shape.height {
            let start = (y * shape.bytes_per_row) as usize;
            window[start..start + shape.width as usize].fill(0);
        }
    };
    let round = |window: &mut Vec<u8>, walk: Walk, free: bool| {
        let bytes = Bytes {
            start: NonNull::new(window.as_mut_ptr()).unwrap(),
            length,
        };
        bytes.compare(
            &pattern,
            (Some((shape, Due::Zero)), free),
            walk.written(None, None),
        )
    };
    let mut kept = window.clone();
    texels(&mut kept);
    assert_eq!(round(&mut kept, walk.clone(), true), Round::Kept);
    // A write-back of the 12th row's line as it read it a pass round the line's bytes before:
    // the byte after its last texel holds the walk's sweep before over it.
    let at = 11 * shape.bytes_per_row + 61;
    let sweep = last[&at];
    assert!(sweep >= 2);
    let mut put_back = window.clone();
    texels(&mut put_back);
    put_back[at as usize] = pattern.at(at) ^ mark(sweep - 1);
    let found = (
        at,
        pattern.at(at) ^ mark(sweep),
        pattern.at(at) ^ mark(sweep - 1),
    );
    assert_eq!(
        round(&mut put_back.clone(), walk.clone(), true),
        Round::Padded(found)
    );
    assert_eq!(round(&mut put_back, walk, false), Round::Changed(found));
}
