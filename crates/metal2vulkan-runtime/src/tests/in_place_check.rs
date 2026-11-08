//! The device's check of the first move in place (`in_place_check.rs`) takes no range of the Metal
//! address space: its buffers, one per class of pages, lie outside it (`address.rs`, "Outside the
//! space"), so no pin or pick of the program's depends on whether or when the check runs, and the
//! space's state changes no verdict. Nor does it take a texture ID: its textures are the
//! runtime's own (`resource_ids.rs`), so the program's IDs are Metal's whether or when it runs.
//! And it checks fresh pages wherever the device takes them, also where its held imports cost its
//! submissions and no long buffer of the product's lies in them.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Each device's shader cache is a fresh
//! directory, so no record holds its verdicts and its check runs; the two devices the placement
//! test compares find there only a record of the same submission-check answers ([`answering`]).

use super::blit_holds::{device, Scratch};
use crate::address::{PAGE, SPACE_BASE, SPACE_END};
use crate::fresh_pages::FRESH_PAGES_PAST;
use crate::in_place::{Access, PagesClass, Verdict as InPlaceVerdict};
use crate::submit_check::{encode, record_name, Check, Measured, Verdict};
use crate::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, Device, Error, GpuAddress,
    ResourceId, ResourceIdKind, ResourceOptions, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::PathBuf;
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
    let fresh = Access::ALL.map(|access| device.in_place_verdict(PagesClass::Fresh, access));
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
            [InPlaceVerdict::Unchecked; 3],
            "no fresh pages to check"
        ),
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
