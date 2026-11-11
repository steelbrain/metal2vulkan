//! The device's check of the first move in place (`in_place_check.rs`) takes no range of the Metal
//! address space: its buffers, one per class of pages, lie outside it (`address.rs`, "Outside the
//! space"), so no pin or pick of the program's depends on whether or when the check runs, and the
//! space's state changes no verdict. Nor does it take a texture ID: its textures are the
//! runtime's own (`resource_ids.rs`), and its queue touches no ID rule of Metal's device
//! (`queue.rs`), so the program's IDs and reservations are Metal's whether or when it runs.
//! And it checks fresh pages wherever the device takes them, also where its held imports cost its
//! submissions and no long buffer of the product's lies in them.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Each device's shader cache is a fresh
//! directory, so no record holds its verdicts and its check runs; the two devices the placement
//! test compares find there only a record of the same submission-check answers ([`answering`]).

use super::blit_holds::{device, options, Scratch};
use crate::address::{PAGE, SPACE_BASE, SPACE_END};
use crate::fresh_pages::FRESH_PAGES_PAST;
use crate::in_place::{verdict_index, Access, PagesClass, RowEnds, Verdict as InPlaceVerdict};
use crate::submit_check::{encode, record_name, Check, Measured, Verdict};
use crate::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, DebugMessage, Device,
    DeviceOptions, Error, GpuAddress, MessageSink, ResourceId, ResourceIdKind, ResourceOptions,
    SamplerState, SamplerStateOptions, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest a check may take.
const WAIT: Duration = Duration::from_secs(120);

/// The low part of the space the pins cover: its first GiB.
const LOW: u64 = 1 << 30;

/// A pin every this many bytes of [`LOW`], of [`PINNED`] bytes: every second one held through the
/// check leaves no free run of the space a check's buffer of 40 MiB fits in.
const STRIDE: u64 = 8 << 20;
const PINNED: u64 = 16 << 10;

/// The submission checks' answers each device the placement test compares recalls
/// ([`answering`]), in `Check::ALL`'s order: a held import adds nothing to a submission, so the
/// device's long buffers take fresh pages where it makes them; the merge and the follows check
/// answer as a driver without either property does.
const ANSWERS: [Verdict; 3] = [Verdict::Fails, Verdict::Fails, Verdict::Fails];

/// [`ANSWERS`] but that a held import adds to every submission's cost, as on RADV: the device's
/// long buffers take no fresh pages (`fresh_pages.rs`).
const TAXED: [Verdict; 3] = [Verdict::Fails, Verdict::Holds, Verdict::Fails];

/// A record of submission-check answers for the driver of the device the cfg selects, and its
/// path under a shader cache's directory.
struct AnswersRecord {
    answers: [Verdict; 3],
    bytes: Vec<u8>,
    under: PathBuf,
}

impl AnswersRecord {
    /// The record of `answers`, its identity and path read from a device made over a cache of its
    /// own and dropped before any device that recalls the record is made.
    fn new(answers: [Verdict; 3]) -> AnswersRecord {
        let scratch = Scratch::new("in-place-check-identity");
        let probe = device(&scratch.0);
        let identity = probe.driver_identity();
        let path = (probe.shared().shader_cache.content_store().directory())
            .record_path(&record_name(&identity));
        let under = path.strip_prefix(&scratch.0).unwrap().to_owned();
        AnswersRecord {
            answers,
            bytes: encode(&identity, answers, Measured::default()),
            under,
        }
    }
}

/// A device over `scratch`'s shader cache that recalls `record`'s answers, written there before
/// it is made, and runs no submission check (`submit_check.rs`): what places the product's long
/// buffers in fresh pages or not (`fresh_pages.rs`). Two devices made so place them alike, where
/// the submission checks each timed under the load beside it can answer apart: a held import
/// measured to cost 1.2 us a submission on one llvmpipe device and 0.3 us on the next, either side
/// of the check's 1 us (`TAX_FLOOR`).
fn answering(scratch: &Scratch, record: &AnswersRecord) -> Device {
    let path = scratch.0.join(&record.under);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &record.bytes).unwrap();
    let device = device(&scratch.0);
    let counts = device.work_counts();
    assert_eq!(
        (counts.submit_checks_recalled, counts.submit_checks_run),
        (1, 0),
        "the device recalled the written answers and ran no submission check"
    );
    let recalled = Check::ALL.map(|check| device.shared().submit_checks.verdict(check));
    assert_eq!(
        recalled, record.answers,
        "the recalled submission-check verdicts"
    );
    device
}

/// A buffer of [`PINNED`] bytes pinned at `at`.
fn pin(device: &Device, at: u64) -> Result<Buffer, Error> {
    let descriptor = BufferDescriptor::new(PINNED, ResourceOptions::shared(), BufferContents::Zero)
        .with_placement(AddressPlacement::At(GpuAddress(at)));
    device.new_buffer_with_descriptor(descriptor)
}

/// The pin addresses of [`LOW`] at even multiples of [`STRIDE`], or at odd ones.
fn pins(odd: bool) -> impl Iterator<Item = u64> {
    (0..LOW / STRIDE)
        .filter(move |k| k % 2 == u64::from(odd))
        .map(|k| SPACE_BASE + k * STRIDE)
}

/// A device whose check runs with the Metal space held but for runs shorter than a check's
/// buffer (the space past its first GiB by one pool, a pin every 16 MiB of that GiB) answers
/// every class as a device whose check ran with the space empty does (both recalling the same
/// submission-check answers: [`answering`]), and each pin between the held ones, made and
/// released in turn until the check settles, is made. A check whose buffer
/// took a range of the space would find none: that class would be unchecked, or a pin refused
/// (`PlacementConflict::Held`).
#[test]
fn the_in_place_check_takes_no_range_of_the_metal_space() {
    let verdicts = |device: &Device| -> Vec<String> {
        let lines = crate::in_place::VERDICTS;
        device
            .in_place_check_report()
            .into_iter()
            .take(lines)
            .collect()
    };
    let record = AnswersRecord::new(ANSWERS);
    let reference = {
        let scratch = Scratch::new("in-place-check-reference");
        let device = answering(&scratch, &record);
        assert!(
            device.settle_in_place_checks(WAIT),
            "the reference check answered"
        );
        verdicts(&device)
    };
    let scratch = Scratch::new("in-place-check-placement");
    let device = answering(&scratch, &record);
    let rest = SPACE_BASE + LOW;
    // The pool's span is its pages and one guard page, to the space's end.
    let pool = device
        .reserve_address_pool(
            AddressPlacement::At(GpuAddress(rest)),
            SPACE_END - rest - PAGE,
        )
        .unwrap();
    let held: Vec<Buffer> = pins(false)
        .map(|at| pin(&device, at).unwrap_or_else(|error| panic!("the pin at {at:#x}: {error}")))
        .collect();
    let started = Instant::now();
    let mut rounds = 0;
    while !device.settle_in_place_checks(Duration::ZERO) {
        assert!(started.elapsed() < WAIT, "the check answered");
        let made: Vec<Buffer> = pins(true)
            .map(|at| {
                pin(&device, at).unwrap_or_else(|error| {
                    panic!("the pin at {at:#x} while the check ran: {error}")
                })
            })
            .collect();
        drop(made);
        rounds += 1;
    }
    let answered = verdicts(&device);
    eprintln!(
        "IN-PLACE-CHECK-PLACEMENT on {}: {answered:?} with the space held, {rounds} rounds of \
         pins while it ran; {reference:?} with it empty",
        device.name()
    );
    assert_eq!(answered, reference, "the verdicts with the space held");
    drop((held, pool));
}

/// A device whose held imports cost its submissions (recalled: [`TAXED`]), so that no long buffer
/// of the product's lies in fresh pages, checks the `Fresh` class all the same wherever it takes
/// such pages, as a rebind's copy under a texture in place lies in them where that class passed
/// (`buffer_descriptor.rs`'s `copied_pages`): each of its `Fresh` verdicts is passed or failed
/// where a device recalling [`ANSWERS`] lays a long buffer of its own in fresh pages, and
/// unchecked where it does not.
#[test]
fn the_in_place_check_checks_fresh_pages_where_held_imports_cost_submissions() {
    let takes = {
        let scratch = Scratch::new("in-place-check-untaxed");
        let device = answering(&scratch, &AnswersRecord::new(ANSWERS));
        let buffer = device.new_buffer(2 * FRESH_PAGES_PAST, ResourceOptions::shared());
        buffer.unwrap().in_fresh_pages()
    };
    let scratch = Scratch::new("in-place-check-taxed");
    let device = answering(&scratch, &AnswersRecord::new(TAXED));
    assert!(
        !device.fresh_pages_pay(),
        "the recalled answers keep long buffers out of fresh pages"
    );
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    let fresh: Vec<InPlaceVerdict> = RowEnds::ALL
        .into_iter()
        .flat_map(|rows| {
            Access::ALL.map(|access| device.in_place_verdict(PagesClass::Fresh, access, rows))
        })
        .collect();
    eprintln!(
        "IN-PLACE-CHECK-TAXED on {}: a long buffer in fresh pages where held imports cost \
         nothing {takes}; the Fresh verdicts where they cost {fresh:?}; fresh-page buffers made {}",
        device.name(),
        device.work_counts().fresh_page_buffers_made
    );
    match takes {
        true => assert!(
            !fresh.contains(&InPlaceVerdict::Unchecked),
            "every Fresh verdict checked: {fresh:?}"
        ),
        false => assert_eq!(
            fresh,
            vec![InPlaceVerdict::Unchecked; RowEnds::ALL.len() * Access::ALL.len()],
            "no fresh pages to check"
        ),
    }
}

/// A class whose pages the device's check could not make for a cause that may pass answers
/// unchecked on that device, its reason noted as such, and is no measurement: its record keeps it
/// unmeasured (`in_place::encode`), so the next device of the same driver and check version
/// recalls every other verdict its check gave, finds that class's pending, and checks it again;
/// the device after that recalls its answer too. The pointer class's first buffer, of the
/// check's reads, is refused here as an import of the pages that fails
/// (`Device::fail_next_host_imports`), which the device answers with a copy of them
/// (`copied.rs`), no pages of the class. A verdict unchecked for a cause that lasts for the
/// driver is recalled as unchecked: on a driver that renders an explicit plane's rows past its
/// height (`RuntimeDriverDefect::ExplicitPlaneRowsPastHeight`, lavapipe), the dma-buf class's
/// reads of rows that end inside lines, of which the device makes no image (no shape of the
/// check whose rows end inside lines has a height that is a multiple of 4). Printed
/// (`IN-PLACE-UNMADE`): the note, the verdicts each device gave the pointer class's reads, and
/// how many each device after the first recalled, unchecked among them.
#[test]
fn an_unchecked_verdict_is_checked_again_where_its_cause_may_pass_and_kept_where_it_lasts() {
    let scratch = Scratch::new("in-place-check-unmade-pages");
    let record = AnswersRecord::new(ANSWERS);
    let pointer_reads = |device: &Device| {
        let checks = &device.shared().in_place;
        RowEnds::ALL.map(|rows| checks.verdict(PagesClass::Pointer, Access::Reads, rows))
    };
    let (first, note, first_notes) = {
        let device = answering(&scratch, &record);
        device.fail_next_host_imports(1);
        assert!(
            device.settle_in_place_checks(WAIT),
            "the first check answered"
        );
        let mut report = device.in_place_check_report();
        let note = (report.iter())
            .find(|line| line.starts_with("Pointer reads: no pages to check: "))
            .cloned();
        assert!(
            note.as_ref()
                .is_some_and(|note| note.ends_with("(a cause that may pass, checked again)")),
            "the refused pages' reason noted, a cause that may pass: {report:?}"
        );
        let notes = report.split_off(crate::in_place::VERDICTS);
        (device.shared().in_place.verdicts(), note, notes)
    };
    let first_reads =
        RowEnds::ALL.map(|rows| first[verdict_index(PagesClass::Pointer, Access::Reads, rows)]);
    assert_eq!(
        first_reads,
        [InPlaceVerdict::Unchecked; 2],
        "the pointer class's reads, whose pages were refused"
    );
    let past_height = |device: &Device| {
        (device.properties().driver_defects)
            .contains(&metal2vulkan::error::RuntimeDriverDefect::ExplicitPlaneRowsPastHeight)
    };
    let dma_buf_reads = verdict_index(PagesClass::DmaBuf, Access::Reads, RowEnds::InLines);
    let (second_reads, second_recalled) = {
        let device = answering(&scratch, &record);
        let recalled = device.shared().in_place.verdicts();
        let held = recalled
            .iter()
            .filter(|&&v| v != InPlaceVerdict::Pending)
            .count() as u64;
        assert_eq!(
            device.work_counts().in_place_checks_recalled,
            held,
            "each verdict the record held counted as recalled"
        );
        for (at, (&was, &now)) in first.iter().zip(&recalled).enumerate() {
            match was {
                InPlaceVerdict::Passed | InPlaceVerdict::Failed => {
                    assert_eq!(now, was, "verdict {at}, measured, recalled")
                }
                InPlaceVerdict::Unchecked => assert!(
                    matches!(now, InPlaceVerdict::Unchecked | InPlaceVerdict::Pending),
                    "verdict {at}, unchecked, recalled so or checked again: {now:?}"
                ),
                InPlaceVerdict::Pending => panic!("verdict {at} pending after the first check"),
            }
        }
        assert_eq!(
            pointer_reads(&device),
            [InPlaceVerdict::Pending; 2],
            "the reads unchecked for a cause that may pass not recalled"
        );
        if past_height(&device) {
            assert_eq!(
                recalled[dma_buf_reads],
                InPlaceVerdict::Unchecked,
                "the dma-buf class's reads of rows inside lines, of which the driver makes no \
                 image, recalled unchecked; the first check's notes {first_notes:#?}"
            );
        }
        assert!(
            device.settle_in_place_checks(WAIT),
            "the second check answered"
        );
        assert!(
            device.work_counts().in_place_checks_run > 0,
            "the second device checked"
        );
        let reads = pointer_reads(&device);
        assert!(
            !reads.contains(&InPlaceVerdict::Unchecked),
            "the pointer class's reads checked again: {reads:?}, {:?}",
            device.in_place_check_report()
        );
        let unchecked = recalled
            .iter()
            .filter(|&&v| v == InPlaceVerdict::Unchecked)
            .count();
        (reads, (held, unchecked))
    };
    let device = answering(&scratch, &record);
    assert_eq!(
        pointer_reads(&device),
        second_reads,
        "the third device recalls the reads the second measured"
    );
    eprintln!(
        "IN-PLACE-UNMADE on {}: {note:?}; Pointer reads {first_reads:?}, then {second_reads:?}; \
         recalled by the second device {second_recalled:?} (verdicts, unchecked among them), by \
         the third {}; the dma-buf class's reads of rows inside lines recalled {:?}",
        device.name(),
        device.work_counts().in_place_checks_recalled,
        device.shared().in_place.verdicts()[dma_buf_reads]
    );
}

/// The check's note of a shape it made no image of in place says why in words: the placement's
/// answer, then the making's refusal (`in_place_check.rs`'s `unmade`, `Error::NotInPlace`), with
/// no internal error and no value's debug form in it; and whether its cause lasts, which decides
/// whether its verdict is kept, is as before. On a driver that renders an explicit plane's rows
/// past its height (`RuntimeDriverDefect::ExplicitPlaneRowsPastHeight`, lavapipe), whose linear
/// image lays rows tight, the device makes no image at the pitch of the check's 20x9
/// `RGBA32Float` shape at 512-byte rows (a height no multiple of 4), over host pages imported by
/// their pointer (`Device::host_shadowed` refuses it) or in its pooled memory (refused over the
/// device's memory): each noted for the pitch, a cause that lasts, as before; and the dma-buf
/// class's reads of rows that end inside lines (no shape of the check whose rows end inside
/// lines has a height that is a multiple of 4) stay unchecked. Printed (`IN-PLACE-NOTE`): each
/// such note.
#[test]
fn the_in_place_checks_notes_say_in_words_why_it_made_no_image() {
    let scratch = Scratch::new("in-place-check-notes");
    let device = device(&scratch.0);
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    let mut report = device.in_place_check_report();
    let notes = report.split_off(crate::in_place::VERDICTS);
    let unmade: Vec<&String> = (notes.iter())
        .filter(|note| note.contains(" image of ") && note.contains(" bytes apart: "))
        .collect();
    for note in &unmade {
        eprintln!("IN-PLACE-NOTE on {}: {note}", device.name());
    }
    eprintln!(
        "IN-PLACE-NOTES on {}: {} of shapes made no image of",
        device.name(),
        unmade.len()
    );
    for note in &unmade {
        for debug in ["Internal", "compile failed", "Shadow(", "Refused(", " { "] {
            assert!(
                !note.contains(debug),
                "a note in words, no {debug:?}: {note}"
            );
        }
    }
    let past_height = (device.properties().driver_defects)
        .contains(&metal2vulkan::error::RuntimeDriverDefect::ExplicitPlaneRowsPastHeight);
    if past_height {
        let verdict = device.in_place_verdict(PagesClass::DmaBuf, Access::Reads, RowEnds::InLines);
        assert_eq!(
            verdict,
            InPlaceVerdict::Unchecked,
            "the dma-buf class's reads of rows inside lines: {notes:#?}"
        );
        let pitch = crate::InPlaceRefusal::Pitch.words();
        for class in ["Pointer", "Pooled"] {
            let head = format!(
                "{class} reads, rows on lines: no RGBA32Float image of 20x9 with rows 512 bytes \
                 apart: {pitch}: "
            );
            let named = (unmade.iter()).any(|note| {
                note.starts_with(&head) && note.ends_with("(a cause that lasts, kept)")
            });
            assert!(named, "noted {head:?}, a cause that lasts: {notes:#?}");
        }
    }
}

/// A private 4x4 `RGBA8Unorm` texture of the program's, at a plain ID.
fn texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// Whether the device's check made a texture: a class it checked, passed or failed.
fn checked_a_texture(device: &Device) -> bool {
    let counts = device.work_counts();
    counts.in_place_checks_passed + counts.in_place_checks_failed > 0
}

/// The device's check takes no texture ID: its textures are the runtime's own, which answer 0
/// and count in no plain allocation (`resource_ids.rs`). A check run to its end before the
/// program's first texture leaves the program's reservation of texture ID 5 taken, so its
/// textures take 1 to 4, then 6, as Metal's do. A check texture that took a plain ID would have
/// started the pool, and the reservation would be ignored (Metal's pool logs "reserve() must be
/// called before any allocation takes place"): 5 would follow 4.
#[test]
fn the_in_place_check_takes_no_texture_id() {
    let scratch = Scratch::new("in-place-check-ids");
    let device = device(&scratch.0);
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    if !checked_a_texture(&device) {
        eprintln!(
            "IN-PLACE-CHECK-IDS on {}: the check made no texture ({:?})",
            device.name(),
            device.in_place_check_report()
        );
        return;
    }
    device
        .reserve_resource_ids(ResourceIdKind::Texture, &[ResourceId(5)])
        .unwrap();
    let made: Vec<Texture> = (0..5).map(|_| texture(&device)).collect();
    let ids: Vec<u64> = made.iter().map(|t| t.gpu_resource_id().0).collect();
    assert_eq!(
        ids,
        [1, 2, 3, 4, 6],
        "after the check, on {}",
        device.name()
    );
}

/// The device the cfg selects over `cache`, as [`device`]'s, with a sink that keeps the code of
/// every message it gives.
fn hearing(cache: &Path) -> (Device, Arc<Mutex<Vec<String>>>) {
    let heard: Arc<Mutex<Vec<String>>> = Arc::default();
    let kept = Arc::clone(&heard);
    let options = DeviceOptions {
        messages: Some(MessageSink::new(move |message: &DebugMessage| {
            kept.lock().unwrap().push(message.id_name.clone());
        })),
        ..options(cache)
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

/// A plain sampler state of the program's, its LOD clamped from `lod`, so that each `lod` gives
/// a state of its own; its ID readable (`support_argument_buffers`).
fn sampler_state(device: &Device, lod: u32) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.lod_min_clamp = lod as f32;
    let options = SamplerStateOptions {
        support_argument_buffers: true,
        ..SamplerStateOptions::default()
    };
    (device.new_sampler_state_with_options(&descriptor, options)).unwrap()
}

/// Whether the device's check made the queue its render passes take (`Access::Renders`): a
/// class it checked for rendering, passed or failed, made it first.
fn check_rendered(device: &Device) -> bool {
    let verdicts = device.shared().in_place.verdicts();
    (PagesClass::ALL.into_iter()).any(|class| {
        RowEnds::ALL.into_iter().any(|rows| {
            let verdict = verdicts[verdict_index(class, Access::Renders, rows)];
            matches!(verdict, InPlaceVerdict::Passed | InPlaceVerdict::Failed)
        })
    })
}

/// The program's reservations hold whatever the device's check did before them: its render
/// passes take a queue of the runtime's own (`queue.rs`), which touches no ID rule of Metal's
/// device. Once the check has run to its end, a reservation of texture IDs 1 to 64 and of sampler
/// IDs 1 to 16 holds: plain textures made after it take IDs past 64, plain sampler states IDs
/// past 16, and no late-reservation line is given. A check queue that counted as the program's
/// first queue would have made the texture pool, so the reservation kept the pool's first range
/// beside its own and the plain textures took 1 to 15 again; and it would have taken the device's
/// sampler ID, so the sampler reservation was ignored, with a late-reservation line, and the
/// plain states took the reserved IDs. A reservation made as soon as the device is made, as a
/// consumer's first call reserves its guest's IDs, holds as well, whatever the check does on its
/// thread meanwhile.
#[test]
fn reservations_hold_whatever_the_in_place_check_did() {
    const TEXTURES: u64 = 64;
    const SAMPLERS: u64 = 16;
    for settled in [true, false] {
        let scratch = Scratch::new("in-place-check-reservation");
        let (device, heard) = hearing(&scratch.0);
        if settled {
            assert!(device.settle_in_place_checks(WAIT), "the check answered");
        }
        let rendered = check_rendered(&device);
        let reserve = |kind, last: u64| {
            let ids: Vec<ResourceId> = (1..=last).map(ResourceId).collect();
            device.reserve_resource_ids(kind, &ids).unwrap();
        };
        reserve(ResourceIdKind::Texture, TEXTURES);
        reserve(ResourceIdKind::SamplerState, SAMPLERS);
        let textures: Vec<Texture> = (0..32).map(|_| texture(&device)).collect();
        let states: Vec<SamplerState> = (0..8).map(|lod| sampler_state(&device, lod)).collect();
        let texture_ids: Vec<u64> = textures.iter().map(|t| t.gpu_resource_id().0).collect();
        let sampler_ids: Vec<u64> = states.iter().map(|s| s.gpu_resource_id().0).collect();
        let late = (heard.lock().unwrap().iter())
            .filter(|code| *code == "metal2vulkan-late-reservation")
            .count();
        eprintln!(
            "IN-PLACE-CHECK-RESERVATION on {}, the check settled first {settled}, its render \
             queue made before the reservation {rendered}: plain textures took {texture_ids:?}, \
             plain sampler states {sampler_ids:?}; late-reservation lines {late}",
            device.name()
        );
        assert!(
            texture_ids.iter().all(|&id| id > TEXTURES),
            "plain textures took reserved IDs: {texture_ids:?}"
        );
        assert!(
            sampler_ids.iter().all(|&id| id > SAMPLERS),
            "plain sampler states took reserved IDs: {sampler_ids:?}"
        );
        assert_eq!(late, 0, "a reservation was ignored as late");
    }
}

/// The program's textures keep Metal's IDs while the device's check runs on its thread: a
/// texture made and released over and over takes ID 1 each time, as on a device that makes no
/// other texture. A check texture holding a plain ID while the program made one would move the
/// program's to 2, which is how a consumer's sequence of IDs came out one past Metal's under
/// load (the check started at its texture over a surface's pages, and its first texture took the
/// next ID).
#[test]
fn texture_ids_keep_metals_order_while_the_in_place_check_runs() {
    let scratch = Scratch::new("in-place-check-ids-during");
    let device = device(&scratch.0);
    let started = Instant::now();
    let mut made = 0;
    let mut ids = std::collections::BTreeMap::<u64, u64>::new();
    while !device.settle_in_place_checks(Duration::ZERO) {
        assert!(started.elapsed() < WAIT, "the check answered");
        let t = texture(&device);
        *ids.entry(t.gpu_resource_id().0).or_default() += 1;
        drop(t);
        made += 1;
    }
    if !checked_a_texture(&device) {
        eprintln!(
            "IN-PLACE-CHECK-IDS-DURING on {}: the check made no texture ({:?})",
            device.name(),
            device.in_place_check_report()
        );
        return;
    }
    eprintln!(
        "IN-PLACE-CHECK-IDS-DURING on {}: {made} textures made while the check ran, by ID: \
         {ids:?}",
        device.name()
    );
    assert!(made > 0, "no texture was made while the check ran");
    assert_eq!(
        ids.keys().copied().collect::<Vec<_>>(),
        [1],
        "the IDs of the textures made while the check ran"
    );
}

/// A no-copy buffer of `length` bytes over an allocation of `device`'s own exported as a dma-buf
/// and mapped (`dma_buf::ExportedPages`, the check's dma-buf pages), with a source of that
/// dma-buf, imported only as it where `only` (`HostMemory::dma_buf_only`), else as any no-copy
/// buffer's pages are, the pointer first. `None` where the device exports no such pages; pages it
/// exports but refuses to import fail the test, as the check's class would stay unchecked.
#[cfg(target_os = "linux")]
fn exported_buffer(device: &Device, length: u64, only: bool) -> Option<Buffer> {
    let exported = match crate::dma_buf::ExportedPages::make(device, length) {
        Ok(exported) => exported,
        Err(why) => {
            eprintln!(
                "IN-PLACE-DMA-BUF on {}: no pages exported: {why}",
                device.name()
            );
            return None;
        }
    };
    let start = exported.start();
    let source = exported.source();
    // SAFETY: the pages are the device's mapped allocation, readable and writable, which stays
    // mapped until the keep-alive lets go of it; the source answers every request over them with
    // a dma-buf of exactly those pages.
    let memory = unsafe { crate::HostMemory::with_dma_buf(start, source) };
    let memory = match only {
        true => memory.dma_buf_only(),
        false => memory,
    };
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: Some(Box::new(exported)),
    };
    let descriptor = BufferDescriptor::new(length, ResourceOptions::shared(), contents)
        .with_placement(AddressPlacement::Outside);
    match device.new_buffer_with_descriptor(descriptor) {
        Ok(buffer) => Some(buffer),
        Err(error) => panic!(
            "{}: the device exports dma-buf pages it refuses to import (dma-buf only {only}): \
             {error:?}",
            device.name()
        ),
    }
}

/// The device's check imports its dma-buf class's pages only as their dma-buf, never by their
/// pointer, which a driver may also take (a mapping of its own exported allocation): such pages
/// lie over the dma-buf where a plain no-copy import of the same kind of pages takes the pointer,
/// and the check answers for the class: its reads of rows that end on lines never unchecked, nor of
/// rows that end inside lines but on a driver that renders an explicit plane's rows past its height
/// (`RuntimeDriverDefect::ExplicitPlaneRowsPastHeight`, lavapipe), where no image over dma-buf
/// pages of a height that is no multiple of 4 is made, as no shape of the check whose rows end
/// inside lines has such a height. A texture of the guest's shape over such pages (234x18
/// `BGRA8Unorm`, rows 1024 bytes apart, `ShaderRead`) is then in place exactly
/// where the class's verdict for reads of its row ends passed. Printed (`IN-PLACE-DMA-BUF`): where
/// each import lay, the class's verdicts and the texture's path. Where the device exports no such
/// pages, it says so and passes.
#[cfg(target_os = "linux")]
#[test]
fn the_in_place_check_imports_its_dma_buf_pages_as_dma_bufs() {
    let scratch = Scratch::new("in-place-check-dma-buf");
    let device = device(&scratch.0);
    let length = 1 << 20;
    let (Some(pointer_first), Some(only)) = (
        exported_buffer(&device, length, false),
        exported_buffer(&device, length, true),
    ) else {
        eprintln!(
            "IN-PLACE-DMA-BUF on {}: the device exports no dma-buf pages to check",
            device.name()
        );
        return;
    };
    assert!(
        only.over_dma_buf(),
        "the dma-buf-only import lies over the dma-buf"
    );
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    let verdicts: Vec<InPlaceVerdict> = RowEnds::ALL
        .into_iter()
        .flat_map(|rows| {
            Access::ALL.map(|access| device.in_place_verdict(PagesClass::DmaBuf, access, rows))
        })
        .collect();
    let (w, h, bpr) = (234u64, 18u64, 1024u64);
    let layout = crate::linear_plan::BufferLayout {
        offset: 0,
        row_pitch: bpr,
        slice_pitch: bpr * h,
        row_bytes: w * 4,
        rows: h,
        slices: 1,
    };
    let start = only.contents().map_or(0, |at| at.as_ptr().addr() as u64);
    let rows = RowEnds::of(start, &layout, device.shared().in_place.line());
    let reads = device.in_place_verdict(PagesClass::DmaBuf, Access::Reads, rows);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = only.resource_options();
    let texture = only.new_texture(&descriptor, 0, bpr).unwrap();
    // SAFETY: the handle is only read, while the texture lives.
    let in_place = unsafe { texture.vk_image() }.is_some();
    eprintln!(
        "IN-PLACE-DMA-BUF on {}: pointer-first import over the dma-buf {}, dma-buf-only {}; the \
         DmaBuf verdicts (rows in lines, then on lines; reads, renders, writes, buffer writes, \
         narrow writes, narrow buffer writes) \
         {verdicts:?}; a 234x18 BGRA8Unorm texture at 1024-byte rows ({rows:?}, reads {reads:?}) \
         in place {in_place}",
        device.name(),
        pointer_first.over_dma_buf(),
        only.over_dma_buf()
    );
    // Reads take only the first move of an image over the dma-buf, which a device that makes
    // such images makes for every format the check reads, but of a height that is no multiple of 4
    // where its driver renders an explicit plane's rows past the height (every shape whose rows
    // end inside lines has such a height); the other accesses may need a feature of the linear
    // modifier the device lacks, which leaves them unchecked.
    let past_height = (device.properties().driver_defects)
        .contains(&metal2vulkan::error::RuntimeDriverDefect::ExplicitPlaneRowsPastHeight);
    for rows in RowEnds::ALL {
        if rows == RowEnds::InLines && past_height {
            continue;
        }
        let verdict = device.in_place_verdict(PagesClass::DmaBuf, Access::Reads, rows);
        assert_ne!(verdict, InPlaceVerdict::Unchecked, "DmaBuf reads, {rows:?}");
    }
    assert_eq!(in_place, reads == InPlaceVerdict::Passed, "{reads:?}");
    drop(texture);
    drop((pointer_first, only));
}
