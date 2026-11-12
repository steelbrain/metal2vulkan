//! Device tests of no-copy buffers' keep-alives (`KeepAlive`, a runtime extension) over one
//! import of host pages, on the device the cfg selects, under the validation layer: the
//! consumer's presents, each a new no-copy buffer over the presented surface's whole memory made
//! before the last one's release, so that some buffer over the surface always lives.
//!
//! Every no-copy buffer made over the same bytes as a live one lies in its import
//! (`host_aliases.rs`), and the import keeps the pages imported while anything holds it. A
//! keep-alive goes at its buffer's release, after the deallocator, where the import's pages stay
//! mapped without it: another buffer lies in the import, or the import keeps another's keep-alive
//! already; else the import keeps it until it is freed. An import freed with only a keep-alive
//! waiting for the free (the deallocator ran at the release) is freed on one of the device's
//! handler threads, and the keep-alive goes there after the free, a moment after the release
//! returned (`HostImport`'s drop): the tests wait for it there. So an import keeps one keep-alive
//! at most, however many buffers come and go over its bytes (`WorkCounts::host_keep_alives_kept`
//! held now, `WorkCounts::host_keep_alives_covered` dropped at the release). A buffer made over
//! part of a live buffer's bytes, or over the same bytes as memory of another class, lies inside
//! that buffer's import where the device binds it there (`host_aliases.rs`, "One page, one
//! import": no page is imported twice), with no import of its own
//! (`WorkCounts::object_churn.buffers_bound_in_import`), and covers nothing of it.
//!
//! What holds an import past its buffers' release is a texture's image bound to its pages, while
//! a running command buffer reaches the image through the texture's ID
//! (`resource_ids::Reaching`): a texture over a buffer keeps the buffer, as Metal's retains it, so
//! the buffer goes at the texture's release, its deallocator run then, as Metal's
//! (`resource_ids.rs`), while the image stays until that command buffer completed. A buffer that
//! lies inside the import holds it too, until that buffer's release: the import maps the pages the
//! released buffer's keep-alive keeps, so the keep-alive goes once both are gone.
//!
//! The memory is private anonymous pages, which every measured driver imports by their pointer;
//! a driver that refuses them has each test say so and pass without running.

use super::degrade_notices::{made_once, on_device as on_device_noticed};
use super::host_memory::{not_importable, on_device, Mapping, GUEST_PAGE};
use super::resource_ids::{Sampling, DELAYED_DEALLOCATORS};
use super::WAIT;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, BufferContents, BufferDescriptor, CpuCacheMode, Deallocator, Degradation, Device,
    Error, HazardTrackingMode, HostMemory, KeepAlive, ResourceOptions, StorageMode, Texture,
    TextureDescriptor, WorkCounts,
};
use std::collections::BTreeSet;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Buffers made and released over one range while another over it lives.
const ROUNDS: usize = 64;

/// The bytes between rows of the textures over the buffers: Metal's linear alignment, and an
/// IOSurface's.
const ROW_BYTES: u64 = 256;

/// How long a keep-alive an import keeps is watched for a drop that must not come: far longer
/// than a handler thread takes to free an import nothing holds, and to drop its keep-alive.
const STAY: Duration = Duration::from_millis(250);

/// The deallocators run and the keep-alives dropped, of the buffers made with one tally.
#[derive(Default)]
struct Tally {
    deallocated: AtomicUsize,
    dropped: AtomicUsize,
}

impl Tally {
    fn deallocated(&self) -> usize {
        self.deallocated.load(Ordering::SeqCst)
    }

    fn dropped(&self) -> usize {
        self.dropped.load(Ordering::SeqCst)
    }
}

/// A keep-alive that counts its drop.
struct Counted(Arc<Tally>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

/// A no-copy buffer over `len` bytes at `at`, of `storage`, with a deallocator and a keep-alive
/// that count into `tally`.
///
/// # Safety
///
/// [`HostMemory::new`]'s contract for the bytes.
unsafe fn counted(
    device: &Device,
    at: NonNull<u8>,
    len: usize,
    storage: StorageMode,
    tally: &Arc<Tally>,
) -> Result<Buffer, Error> {
    let keep_alive: KeepAlive = Box::new(Counted(Arc::clone(tally)));
    // SAFETY: the caller's contract.
    unsafe { counted_with(device, at, len, tally, Some((storage, keep_alive))) }
}

/// A no-copy buffer over `len` bytes at `at`, with a deallocator that counts into `tally`: of the
/// storage mode and with the keep-alive `kept` gives, else `Shared` with none.
///
/// # Safety
///
/// [`HostMemory::new`]'s contract for the bytes.
unsafe fn counted_with(
    device: &Device,
    at: NonNull<u8>,
    len: usize,
    tally: &Arc<Tally>,
    kept: Option<(StorageMode, KeepAlive)>,
) -> Result<Buffer, Error> {
    let seen = Arc::clone(tally);
    let deallocator: Deallocator = Box::new(move |_, _| {
        seen.deallocated.fetch_add(1, Ordering::SeqCst);
    });
    let (storage, keep_alive) = match kept {
        Some((storage, keep_alive)) => (storage, Some(keep_alive)),
        None => (StorageMode::Shared, None),
    };
    // SAFETY: the caller's contract.
    let memory = unsafe { HostMemory::new(at) };
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: Some(deallocator),
        keep_alive,
    };
    let options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    device.new_buffer_with_descriptor(BufferDescriptor::new(len as u64, options, contents))
}

/// A texture of two 256-byte rows over `buffer`'s first bytes, and whether its image lies in the
/// buffer's pages (it holds their import then), not a shadow of them.
fn texture_over(buffer: &Buffer) -> (Texture, bool) {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Uint, ROW_BYTES / 4, 2, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, ROW_BYTES).unwrap();
    // SAFETY: the handle is only compared, while the texture lives.
    let bound = unsafe { texture.vk_image() }.is_some();
    (texture, bound)
}

/// Wait until `done`, at most [`WAIT`]: an image and its import, and an import's deallocator,
/// may go on a runtime thread.
pub(super) fn eventually(done: impl Fn() -> bool) -> bool {
    eventually_within(WAIT, done)
}

/// Wait until `done`, at most `most`.
fn eventually_within(most: Duration, done: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + most;
    while !done() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    done()
}

/// The keep-alives the device holds now, and those it dropped at a release, since `base`.
fn since(device: &Device, base: &WorkCounts) -> (u64, u64) {
    let counts = device.work_counts();
    (
        counts.host_keep_alives_kept - base.host_keep_alives_kept,
        counts.host_keep_alives_covered - base.host_keep_alives_covered,
    )
}

/// Whether a no-copy buffer was bound inside a live import since `base` (`host_aliases.rs`, "One
/// page, one import"): over pages the import holds all of, with no import of its own, holding
/// that import until it goes.
fn bound_inside(device: &Device, base: &WorkCounts) -> bool {
    let churn = |counts: &WorkCounts| counts.object_churn.buffers_bound_in_import;
    churn(&device.work_counts()) > churn(base)
}

/// The no-copy buffers' releases since `base` that a buffer bound inside their import held past
/// the release (`WorkCounts::host_releases_held_inside`): the keep-alive kept, or, given none, the
/// deallocator waiting, until that buffer went too.
fn held_inside(device: &Device, base: &WorkCounts) -> u64 {
    device.work_counts().host_releases_held_inside - base.host_releases_held_inside
}

/// [`ROUNDS`] buffers made over one range, each before the last one's release, as a consumer's
/// presents make them over a surface: each released one's deallocator runs and its keep-alive is
/// dropped at its release, covered by the next, and the device keeps none. The last goes at the
/// release of a texture over it, which keeps it till then; where the texture's image lies in its
/// pages and a running command buffer reaches the image, the import keeps its keep-alive until
/// that command buffer completed; a texture whose image is a shadow holds no import, and the
/// keep-alive goes at the release.
#[test]
fn keep_alives_over_one_import_stay_at_most_one_as_buffers_come_and_go() {
    let len = 2 * GUEST_PAGE;
    let mapping = Mapping::anonymous(len);
    on_device(|device| {
        let sampling = Sampling::new(device);
        let tally = Arc::new(Tally::default());
        let base = device.work_counts();
        let make = || {
            // SAFETY: the mapping outlives the device, and so every buffer over it.
            unsafe { counted(device, mapping.pointer(), len, StorageMode::Shared, &tally) }
        };
        let mut live = match make() {
            Ok(buffer) => buffer,
            Err(error) if not_importable(&error) => {
                println!("KEEP-ALIVES {}: refused {error:?}", device.name());
                return;
            }
            Err(error) => panic!("a no-copy buffer on {}: {error:?}", device.name()),
        };
        for round in 1..=ROUNDS {
            let next = make().unwrap();
            drop(live);
            live = next;
            assert_eq!(
                tally.deallocated(),
                round,
                "round {round}: deallocators run"
            );
            assert_eq!(tally.dropped(), round, "round {round}: keep-alives dropped");
            assert_eq!(
                since(device, &base),
                (0, round as u64),
                "round {round}: (held, covered)"
            );
        }
        let (texture, bound) = texture_over(&live);
        drop(live);
        println!(
            "KEEP-ALIVES {}: {ROUNDS} covered, an image bound to the bytes {bound}",
            device.name()
        );
        assert_eq!(tally.deallocated(), ROUNDS, "the texture keeps the last");
        assert_eq!(tally.dropped(), ROUNDS, "the texture keeps the last");
        let reaching = sampling.reaching(device, &texture);
        drop(texture);
        assert_eq!(tally.deallocated(), ROUNDS + 1, "the last at the release");
        let held = u64::from(bound);
        // Where nothing holds the import, its free and the keep-alive go on a handler thread.
        assert!(
            eventually(|| tally.dropped() == ROUNDS + 1 - held as usize),
            "the last keep-alive goes with the import unless the image holds it"
        );
        assert_eq!(
            since(device, &base),
            (held, ROUNDS as u64),
            "(held, covered)"
        );
        reaching.complete(&sampling);
        assert!(
            eventually(|| tally.dropped() == ROUNDS + 1),
            "the kept keep-alive goes as the import is freed"
        );
        assert_eq!(since(device, &base), (0, ROUNDS as u64), "(held, covered)");
    });
}

/// The keep-alive an import keeps covers a later buffer's over the same bytes. A buffer of
/// another memory class over them, made while the first lives, keeps the bytes noted, so a
/// buffer made over them while the import of a released one is held lies in that import
/// (`host_aliases.rs`). Where the device binds it there, that other class's buffer lies inside
/// the first's import itself ("One page, one import": no page imported twice) and holds it until
/// its release; else it is a copy of the pages, holding none. Released one after another,
/// [`ROUNDS`] such buffers leave the device one keep-alive, the first's, until nothing holds the
/// import: the command buffer reaching the texture's image, where the image lies in the pages,
/// completed, and the other class's buffer, where it lies inside the import, went. Each later
/// one's is dropped at its release. The first's deallocator runs at its release, as Metal's. Where
/// nothing holds the import (the texture's image a shadow, the other class's buffer a copy) or the
/// device refuses the memory or the class, nothing is kept and the test says so.
#[test]
fn a_kept_keep_alive_covers_later_buffers_over_its_bytes() {
    let len = 2 * GUEST_PAGE;
    let mapping = Mapping::anonymous(len);
    on_device(|device| {
        let sampling = Sampling::new(device);
        let tally = Arc::new(Tally::default());
        let other = Arc::new(Tally::default());
        let base = device.work_counts();
        let make = |storage, tally: &Arc<Tally>| {
            // SAFETY: the mapping outlives the device, and so every buffer over it.
            unsafe { counted(device, mapping.pointer(), len, storage, tally) }
        };
        let (first, noting) = match (
            make(StorageMode::Shared, &tally),
            make(StorageMode::Managed, &other),
        ) {
            (Ok(first), Ok(noting)) => (first, noting),
            (Err(error), _) | (_, Err(error)) if not_importable(&error) => {
                println!("KEEP-ALIVES-KEPT {}: refused {error:?}", device.name());
                return;
            }
            (Err(error), _) | (_, Err(error)) => {
                panic!("a no-copy buffer on {}: {error:?}", device.name())
            }
        };
        let inside = bound_inside(device, &base);
        let (texture, bound) = texture_over(&first);
        let reaching = sampling.reaching(device, &texture);
        drop(first);
        drop(texture);
        println!(
            "KEEP-ALIVES-KEPT {}: an image bound to the bytes {bound}, the other class's buffer \
             inside the first's import {inside}",
            device.name()
        );
        assert_eq!(tally.deallocated(), 1, "the first's at the release");
        if !bound && !inside {
            assert!(
                eventually(|| tally.dropped() == 1),
                "no import held: the keep-alive goes with it"
            );
            assert_eq!(since(device, &base), (0, 0), "(held, covered)");
            assert_eq!(held_inside(device, &base), 0, "no release held inside");
            reaching.complete(&sampling);
            return;
        }
        assert_eq!(since(device, &base), (1, 0), "the import keeps the first's");
        assert_eq!(
            held_inside(device, &base),
            u64::from(inside),
            "the first's release held by the buffer inside its import"
        );
        for round in 1..=ROUNDS {
            let later = make(StorageMode::Shared, &tally).unwrap();
            drop(later);
            assert_eq!(
                tally.deallocated(),
                1 + round,
                "round {round}: deallocators run"
            );
            assert_eq!(tally.dropped(), round, "round {round}: keep-alives dropped");
            assert_eq!(
                since(device, &base),
                (1, round as u64),
                "round {round}: (held, covered)"
            );
        }
        reaching.complete(&sampling);
        if inside {
            // The other class's buffer still lies inside the import, which keeps the first's
            // keep-alive until that buffer goes.
            let stays = !eventually_within(STAY, || tally.dropped() > ROUNDS);
            println!(
                "KEEP-ALIVES-KEPT {}: after the command buffer completed, (held, covered) {:?}, \
                 the first's keep-alive dropped {}",
                device.name(),
                since(device, &base),
                !stays
            );
            assert!(stays, "the other class's buffer holds the first's import");
            assert_eq!(since(device, &base), (1, ROUNDS as u64), "(held, covered)");
        }
        drop(noting);
        assert_eq!(other.deallocated(), 1, "the other class's deallocator");
        assert!(
            eventually(|| tally.dropped() == ROUNDS + 1),
            "the first's keep-alive goes as its import is freed"
        );
        assert!(
            eventually(|| other.dropped() == 1),
            "the other class's keep-alive goes with its buffer"
        );
        println!(
            "KEEP-ALIVES-KEPT {}: after the other class's release, (held, covered) {:?}, keep-alives \
             dropped {} of {}",
            device.name(),
            since(device, &base),
            tally.dropped() + other.dropped(),
            ROUNDS + 2
        );
        assert_eq!(since(device, &base), (0, ROUNDS as u64), "(held, covered)");
    });
}

/// A narrower buffer over the first half of a wider live one's bytes covers nothing of the wider
/// one's. Where the device binds it there, it lies inside the wider one's import ("One page, one
/// import": no page imported twice) and holds it until its release: the import keeps the wider
/// one's keep-alive past the wider one's release, as it maps both halves, until the narrower one
/// went, and while an image over the wider one's pages is reached by a running command buffer,
/// until that command buffer completed. Else the narrower one is a copy of its half, holding no
/// import. The wider one's deallocator runs at its release, as Metal's. The narrower one's own
/// keep-alive goes as its hold on the pages goes, on a handler thread. No keep-alive is counted
/// covered.
#[test]
fn a_narrower_live_buffer_covers_no_wider_released_one() {
    let len = 2 * GUEST_PAGE;
    let mapping = Mapping::anonymous(len);
    on_device(|device| {
        let sampling = Sampling::new(device);
        let (wide_tally, narrow_tally) = (Arc::new(Tally::default()), Arc::new(Tally::default()));
        let base = device.work_counts();
        let make = |len, tally: &Arc<Tally>| {
            // SAFETY: the mapping outlives the device, and so every buffer over it.
            unsafe { counted(device, mapping.pointer(), len, StorageMode::Shared, tally) }
        };
        let (wide, narrow) = match (make(len, &wide_tally), make(GUEST_PAGE, &narrow_tally)) {
            (Ok(wide), Ok(narrow)) => (wide, narrow),
            (Err(error), _) | (_, Err(error)) if not_importable(&error) => {
                println!("KEEP-ALIVES-NARROW {}: refused {error:?}", device.name());
                return;
            }
            (Err(error), _) | (_, Err(error)) => {
                panic!("a no-copy buffer on {}: {error:?}", device.name())
            }
        };
        let inside = bound_inside(device, &base);
        let (texture, bound) = texture_over(&wide);
        let reaching = sampling.reaching(device, &texture);
        drop(wide);
        drop(texture);
        println!(
            "KEEP-ALIVES-NARROW {}: an image bound to the bytes {bound}, the narrower one inside \
             the wider one's import {inside}",
            device.name()
        );
        assert_eq!(
            wide_tally.deallocated(),
            1,
            "the wider one's deallocator at the release"
        );
        // The image or the narrower one holds the wider one's import, which keeps its keep-alive.
        // Where neither does, its free and the keep-alive go on a handler thread.
        let held = u64::from(bound || inside);
        let stays = !eventually_within(STAY, || wide_tally.dropped() > 0);
        assert!(
            eventually(|| wide_tally.dropped() == 1 - held as usize),
            "the wider one's keep-alive stays while its import is held"
        );
        assert_eq!(stays, held == 1, "the wider one's keep-alive stays");
        assert_eq!(since(device, &base), (held, 0), "(held, covered)");
        assert_eq!(
            held_inside(device, &base),
            u64::from(inside),
            "the wider one's release held by the narrower one inside its import"
        );
        println!(
            "KEEP-ALIVES-NARROW {}: before the narrower one's release, (held, covered) {:?}, the \
             wider one's keep-alive dropped {}",
            device.name(),
            since(device, &base),
            wide_tally.dropped()
        );
        drop(narrow);
        assert_eq!(
            narrow_tally.deallocated(),
            1,
            "the narrower one's deallocator"
        );
        assert!(
            eventually(|| narrow_tally.dropped() == 1),
            "the narrower one's keep-alive with its hold on the pages"
        );
        // The narrower one's release lets go of the wider one's import, freed then unless the
        // image holds it.
        let held = u64::from(bound);
        assert!(
            eventually(|| wide_tally.dropped() == 1 - held as usize),
            "the wider one's keep-alive goes with its import unless the image holds it"
        );
        assert_eq!(since(device, &base), (held, 0), "(held, covered)");
        println!(
            "KEEP-ALIVES-NARROW {}: after the narrower one's release, (held, covered) {:?}, \
             keep-alives dropped: the wider one's {}, the narrower one's {}",
            device.name(),
            since(device, &base),
            wide_tally.dropped(),
            narrow_tally.dropped()
        );
        reaching.complete(&sampling);
        assert!(
            eventually(|| wide_tally.dropped() == 1),
            "the wider one's keep-alive goes as its import is freed"
        );
        assert_eq!(since(device, &base), (0, 0), "(held, covered)");
    });
}

/// Given no keep-alive, a no-copy buffer's deallocator waits for its pages' import to be freed
/// where something holds the import past the release. A narrower buffer made over the first half
/// of a wider live one's bytes lies inside the wider one's import where the device binds it there
/// ("One page, one import": no page imported twice), holding it: the wider one's deallocator runs
/// once the narrower one went, where Metal's runs at the wider one's release. Counted
/// (`WorkCounts::host_deallocators_delayed`, `host_releases_held_inside`) and announced once
/// (`host-deallocator-delayed`, naming the buffer inside). Where the device does not bind the
/// narrower one there, it is a copy of its half (`copied-host-pages`), and the deallocator runs at
/// the release, as Metal's. Nothing else is announced: the buffers lie in the caller's pages,
/// which no `shared-in-system-memory` notice names on a device without unified memory (the runtime
/// announces it of memory it allocates for a `Shared` resource, which the caller's pages are
/// not).
#[test]
fn a_buffer_inside_an_import_holds_a_deallocator_given_no_keep_alive() {
    let len = 2 * GUEST_PAGE;
    let mapping = Mapping::anonymous(len);
    on_device_noticed(|device, messages| {
        let (wide_tally, narrow_tally) = (Arc::new(Tally::default()), Arc::new(Tally::default()));
        let base = device.work_counts();
        let make = |len, tally: &Arc<Tally>| {
            // SAFETY: the mapping outlives the device, and so every buffer over it.
            unsafe { counted_with(device, mapping.pointer(), len, tally, None) }
        };
        let (wide, narrow) = match (make(len, &wide_tally), make(GUEST_PAGE, &narrow_tally)) {
            (Ok(wide), Ok(narrow)) => (wide, narrow),
            (Err(error), _) | (_, Err(error)) if not_importable(&error) => {
                println!("KEEP-ALIVES-NONE {}: refused {error:?}", device.name());
                return;
            }
            (Err(error), _) | (_, Err(error)) => {
                panic!("a no-copy buffer on {}: {error:?}", device.name())
            }
        };
        let inside = bound_inside(device, &base);
        drop(wide);
        let delayed =
            device.work_counts().host_deallocators_delayed - base.host_deallocators_delayed;
        println!(
            "KEEP-ALIVES-NONE {}: the narrower one inside the wider one's import {inside}; at the \
             wider one's release its deallocator ran {}, delayed {delayed}, releases held inside {}",
            device.name(),
            wide_tally.deallocated(),
            held_inside(device, &base)
        );
        assert_eq!(
            wide_tally.deallocated(),
            usize::from(!inside),
            "the wider one's deallocator at its release, unless the narrower one holds its import"
        );
        assert_eq!(delayed, u64::from(inside), "delayed deallocators");
        assert_eq!(
            held_inside(device, &base),
            u64::from(inside),
            "releases held inside"
        );
        drop(narrow);
        assert!(
            eventually(|| narrow_tally.deallocated() == 1),
            "the narrower one's deallocator"
        );
        assert!(
            eventually(|| wide_tally.deallocated() == 1),
            "the wider one's deallocator once its import is freed"
        );
        println!(
            "KEEP-ALIVES-NONE {}: after the narrower one's release, deallocators run: the wider \
             one's {}, the narrower one's {}",
            device.name(),
            wide_tally.deallocated(),
            narrow_tally.deallocated()
        );
        let kinds: &[Degradation] = match inside {
            true => &DELAYED_DEALLOCATORS,
            false => &[Degradation::CopiedHostPages],
        };
        let notices = made_once(messages, kinds, &BTreeSet::new());
        let reason = (notices.iter())
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::HostDeallocatorDelayed)
            .map(|notice| notice.reason.clone());
        if inside {
            let reason = reason.expect("the delayed deallocator announced");
            assert!(
                reason.starts_with("a buffer bound inside its pages' import"),
                "{reason}"
            );
        }
    });
}
