//! The residency of no-copy buffers' pointer imports (`host_residency.rs`), on the device the cfg
//! selects.
//!
//! The memory is guest RAM as QEMU's `memory-backend-memfd,share=on` makes it, a sealed `memfd`
//! mapped shared, imported by its pointer (`HostMemory::new`), as ANV, NVIDIA and lavapipe import
//! it; on a device that refuses that pointer (RADV, which imports the `memfd`'s dma-bufs,
//! `tests/host_residency.rs`, and makes a no-copy buffer over it with no source a copy of its
//! pages), private anonymous pages, which RADV imports by their pointer. Where the device's check
//! found a held pointer import adds to every submission's cost (`submit_check.rs`, the import tax:
//! `Device::releases_idle_pointer_imports`), the device releases such an import while idle and
//! imports it again at the same pointer at its next blit; elsewhere it keeps it
//! ([`an_idle_pointer_import_is_released_where_held_imports_cost_submissions`]). The other tests
//! hold the device to the release whatever its check found (`SubmitChecks::force`): the pages stay
//! mapped at the pointer while the buffer lives (`HostMemory::new`), so an import again reads what
//! the kept one read on every driver, and the path is the same on each.
//! The tests release imports at once through the crate's hook (`Device::release_idle_host_imports`)
//! and check:
//! - the counts that show the release and the import again ran (`WorkCounts::host_imports_released`,
//!   `host_imports_reimported`, `host_imports_resident`, `host_imports_live`), and none of a
//!   dma-buf's descriptors kept;
//! - the bytes a blit reads and writes after a release, the CPU's writes made while released among
//!   them;
//! - that a released import, made again at its pointer, reads the pages a remap put there while
//!   no import of them lived;
//! - that a buffer whose Metal address was handed out stays imported.
//!
//! Where the device refuses the pointers of both kinds of pages each test says so and passes
//! without running.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on Linux, on the kind of device the cfg selects. No test maps pages over a live
//! import of them but [`a_file_s_pages_mapped_under_a_kept_pointer_import`], which breaks
//! `VK_EXT_external_memory_host`'s rule on purpose to measure what a driver then does, a GPU
//! hazard: on a GPU it compiles only with `m2v_gpu_hazards` too (docs/VALIDATION.md).

use super::host_residency::{
    counting, device, gpu_fill, gpu_read, host_page, read, release, residency, Ram, Scratch,
};
use crate::submit_check::{Check, Verdict};
use crate::{Buffer, Device, Error, HostBacking, HostMemory, ResourceOptions};
use metal2vulkan::error::Feature;
use std::ptr::NonNull;

/// Hold `device` to releasing the idle pointer imports of the no-copy buffers made from now on,
/// whatever its check found (the module docs): once its checks answered, their import tax holds.
pub(super) fn release_pointer_imports(device: &Device) {
    assert!(
        device.settle_submit_checks(crate::tests::SETTLE),
        "the submission checks never answered"
    );
    (device.shared().submit_checks).force(Check::ImportTax, Verdict::Holds);
    assert!(device.releases_idle_pointer_imports());
}

/// Run `test` on the device the cfg selects.
fn on_device(test: impl FnOnce(&Device)) {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    test(&device);
}

/// The pages a test's no-copy buffer lies over (the module docs).
enum Pages {
    /// Guest RAM's view, its file twice the view's length: a remap maps the file's second half
    /// over the view.
    Shared(Ram),
    /// Private anonymous pages: a remap maps fresh ones over them.
    Private { at: NonNull<u8>, len: usize },
}

impl Pages {
    /// Guest RAM's view of `len` bytes.
    fn shared(len: usize) -> Pages {
        Pages::Shared(Ram::new(2 * len, 0, len))
    }

    /// `len` bytes of private anonymous pages.
    fn private(len: usize) -> Pages {
        // SAFETY: a new private mapping, the object's until dropped.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED, "mmap of private pages");
        Pages::Private {
            at: NonNull::new(at.cast()).unwrap(),
            len,
        }
    }

    /// The pages' address `offset` bytes in.
    fn pointer(&self, offset: usize) -> NonNull<u8> {
        match self {
            Pages::Shared(ram) => ram.pointer(offset),
            Pages::Private { at, len } => {
                assert!(offset < *len);
                // SAFETY: inside the mapping.
                unsafe { NonNull::new_unchecked(at.as_ptr().add(offset)) }
            }
        }
    }

    /// Write `bytes` at `offset`: guest RAM's into its file, not through the view; private pages'
    /// through their mapping.
    fn write(&self, offset: usize, bytes: &[u8]) {
        match self {
            Pages::Shared(ram) => ram.write_file(offset, bytes),
            Pages::Private { len, .. } => {
                assert!(offset + bytes.len() <= *len);
                // SAFETY: inside the mapping; each test orders the CPU's and the GPU's accesses.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        self.pointer(offset).as_ptr(),
                        bytes.len(),
                    )
                };
            }
        }
    }

    /// The bytes at the pages' addresses.
    fn view(&self) -> Vec<u8> {
        match self {
            Pages::Shared(ram) => ram.view(),
            // SAFETY: the mapping's bytes; each test orders the CPU's and the GPU's accesses.
            Pages::Private { at, len } => unsafe {
                std::slice::from_raw_parts(at.as_ptr(), *len).to_vec()
            },
        }
    }

    /// Other pages holding `bytes` mapped over the pages' addresses (`mmap` with `MAP_FIXED`), as a
    /// guest remaps its RAM there: guest RAM's file's second half, or fresh private pages.
    fn remap(&self, bytes: &[u8]) {
        match self {
            Pages::Shared(ram) => {
                ram.write_file(bytes.len(), bytes);
                ram.remap(bytes.len());
            }
            Pages::Private { at, len } => {
                assert_eq!(bytes.len(), *len);
                // SAFETY: the mapping's own range, mapped again.
                let mapped = unsafe {
                    libc::mmap(
                        at.as_ptr().cast(),
                        *len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                        -1,
                        0,
                    )
                };
                assert_eq!(mapped, at.as_ptr().cast(), "mmap over the private pages");
                self.write(0, bytes);
            }
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Pages::Shared(_) => "a shared memfd's pages",
            Pages::Private { .. } => "private anonymous pages",
        }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        if let Pages::Private { at, len } = self {
            // SAFETY: the mapping is this object's, and no buffer uses it any more (each test
            // drops its buffer first).
            unsafe { libc::munmap(at.as_ptr().cast(), *len) };
        }
    }
}

/// A no-copy buffer over `bytes.len()` bytes of new pages holding `bytes`, imported by its
/// pointer: guest RAM's view, or, where the device refuses its pointer (the call refused, or the
/// buffer a copy of the pages, `HostBacking::Copied`), private anonymous pages (the module docs).
/// `None`, printed, where the device refuses both. The buffer drops before the pages.
fn no_copy(test: &str, device: &Device, bytes: &[u8]) -> Option<(Pages, Buffer)> {
    let mut refused = Vec::new();
    for pages in [Pages::shared(bytes.len()), Pages::private(bytes.len())] {
        pages.write(0, bytes);
        let before = device.work_counts().host_imports_live;
        // SAFETY: each test's pages outlive its buffer.
        let made = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(pages.pointer(0)),
                bytes.len() as u64,
                ResourceOptions::default(),
                None,
            )
        };
        match made {
            Ok(buffer) if buffer.host_backing() == Some(HostBacking::Copied) => {
                refused.push(format!("{}: copied", pages.kind()));
            }
            Ok(buffer) => {
                let imported = device.work_counts().host_imports_live - before;
                assert_eq!(imported, 1, "{test}: one import of the pointer");
                eprintln!("{test}: on {}: over {}", device.name(), pages.kind());
                return Some((pages, buffer));
            }
            Err(Error::Unsupported(unsupported)) if unsupported.feature == Feature::HostMemory => {
                refused.push(format!("{}: {unsupported:?}", pages.kind()));
            }
            Err(error) => panic!("{test}: new_buffer_with_bytes_no_copy: {error:?}"),
        }
    }
    eprintln!(
        "{test}: on {}: the device refuses the pointers: {refused:?}",
        device.name()
    );
    None
}

/// A no-copy buffer over 4 pages, imported by its pointer, on a device held to releasing it (the
/// module docs):
/// - read by a blit, its import released: the device's live host imports fall by one, no
///   descriptor is kept, and the hook releases it no further while released;
/// - the CPU's writes made while it is released, through `contents` and through guest RAM's file
///   (or the private pages' mapping), read by the next blit, which imports it again;
/// - released again, a GPU fill lands in the pages;
/// - released again, its Metal address handed out imports it again at once and for good, and the
///   hook releases it no more.
#[test]
fn an_idle_pointer_import_is_released_and_imported_again_at_its_next_blit() {
    const TEST: &str = "pointer residency";
    on_device(|device| {
        release_pointer_imports(device);
        let page = host_page();
        let len = 4 * page;
        let queue = device.new_command_queue().unwrap();
        let first = counting(len, 7);
        let Some((pages, buffer)) = no_copy(TEST, device, &first) else {
            return;
        };
        assert!(read(&queue, &buffer, len) == first, "{TEST}");
        let imported = device.work_counts();
        assert_eq!(residency(&imported), (0, 0, 1, 0), "{TEST}");
        let live = imported.host_imports_live;

        release(TEST, device, 1);
        let released = device.work_counts();
        assert_eq!(residency(&released), (1, 0, 0, 0), "{TEST}: released");
        assert_eq!(released.host_imports_live, live - 1, "{TEST}");
        assert_eq!(
            device.release_idle_host_imports(),
            0,
            "{TEST}: released twice"
        );

        // The CPU writes while it is released: half through `contents`, half through the file
        // (or the private pages' mapping).
        let second = counting(len, 59);
        let contents = buffer.contents().unwrap();
        // SAFETY: the buffer's own bytes, which no command reads now.
        unsafe { std::ptr::copy_nonoverlapping(second.as_ptr(), contents.as_ptr(), len / 2) };
        pages.write(len / 2, &second[len / 2..]);
        let (bytes, importing) = gpu_read(&queue, &buffer, len);
        assert!(bytes == second, "{TEST}: the bytes written while released");
        let (bytes, imported) = gpu_read(&queue, &buffer, len);
        assert!(bytes == second, "{TEST}");
        let again = device.work_counts();
        assert_eq!(residency(&again), (1, 1, 1, 0), "{TEST}: imported again");
        assert_eq!(again.host_imports_live, live, "{TEST}");

        // A GPU write after a release lands in the pages.
        release(TEST, device, 1);
        gpu_fill(&queue, &buffer, page as u64..3 * page as u64, 0x6d);
        let mut filled = second.clone();
        filled[page..3 * page].fill(0x6d);
        assert!(pages.view() == filled, "{TEST}: the GPU's fill");
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (2, 2, 1, 0), "{TEST}: filled");

        // Its Metal address handed out keeps it imported for good, from the moment it is.
        release(TEST, device, 1);
        let address = buffer.gpu_address();
        assert_ne!(address.0, 0, "{TEST}");
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (3, 3, 1, 0), "{TEST}: exposed");
        assert_eq!(counts.host_imports_live, live, "{TEST}");
        assert!(read(&queue, &buffer, len) == filled, "{TEST}");
        assert_eq!(device.release_idle_host_imports(), 0, "{TEST}: exposed");
        drop(buffer);
        eprintln!(
            "{TEST}: on {}: released 3 times, imported again 3 times; a copy's call took \
             {importing:?} importing it again, {imported:?} imported",
            device.name()
        );
    });
}

/// A no-copy buffer over 2 pages, imported by its pointer and read by a blit, on the device as its
/// check found it: the hook releases its import where a held pointer import adds to every
/// submission's cost (`Device::releases_idle_pointer_imports`, the import tax), and the next blit
/// imports it again; elsewhere the import stays, counted as kept for its pointer
/// (`WorkCounts::host_imports_kept_pointer`), and the hook releases nothing. Printed with the
/// check's report (`POINTER-RESIDENCY-GATE`): the evidence of the path the device took. A device
/// that releases pointer imports took the check's own, so it refuses no pointer of both kinds.
#[test]
fn an_idle_pointer_import_is_released_where_held_imports_cost_submissions() {
    const TEST: &str = "pointer residency gate";
    on_device(|device| {
        assert!(
            device.settle_submit_checks(crate::tests::SETTLE),
            "{TEST}: the submission checks never answered"
        );
        let releases = device.releases_idle_pointer_imports();
        let len = 2 * host_page();
        let queue = device.new_command_queue().unwrap();
        let bytes = counting(len, 23);
        let before = device.work_counts();
        let Some((_pages, buffer)) = no_copy(TEST, device, &bytes) else {
            assert!(
                !releases,
                "{TEST}: a device that releases pointer imports refused both pointers"
            );
            return;
        };
        assert!(read(&queue, &buffer, len) == bytes, "{TEST}");
        let made = device.work_counts();
        let kept = made.host_imports_kept_pointer - before.host_imports_kept_pointer;
        let resident = made.host_imports_resident - before.host_imports_resident;
        assert_eq!(
            (kept, resident),
            (u64::from(!releases), u64::from(releases)),
            "{TEST}: kept for its pointer, or releasable"
        );
        match releases {
            true => release(TEST, device, 1),
            false => assert_eq!(device.release_idle_host_imports(), 0, "{TEST}: released"),
        }
        assert!(
            read(&queue, &buffer, len) == bytes,
            "{TEST}: after the hook"
        );
        let after = device.work_counts();
        let moved = (
            after.host_imports_released - before.host_imports_released,
            after.host_imports_reimported - before.host_imports_reimported,
        );
        assert_eq!(
            moved,
            (u64::from(releases), u64::from(releases)),
            "{TEST}: released and imported again"
        );
        drop(buffer);
        eprintln!(
            "POINTER-RESIDENCY-GATE {TEST}: on {}: idle pointer imports released {releases}; {}",
            device.name(),
            device.submit_check_report()
        );
    });
}

/// A no-copy buffer over 2 pages, imported by its pointer on a device held to releasing it (the
/// module docs), read by a blit and its import released; then, with no import of them live, other
/// pages mapped over its addresses and the device told: a remap under a live buffer, which
/// `HostMemory::new`'s contract forbids, but under no live import. The next blit imports it again
/// at its pointer and reads the new pages, the runtime's own answer where such a remap leaves the
/// buffer's bytes unspecified (`docs/FAST-PATHS.md`, known gaps).
#[test]
fn a_released_pointer_import_is_made_again_over_the_pages_mapped_at_its_pointer() {
    const TEST: &str = "pointer residency remap";
    on_device(|device| {
        release_pointer_imports(device);
        let page = host_page();
        let len = 2 * page;
        let queue = device.new_command_queue().unwrap();
        let old = counting(len, 13);
        let new = counting(len, 131);
        let Some((pages, buffer)) = no_copy(TEST, device, &old) else {
            return;
        };
        assert!(read(&queue, &buffer, len) == old, "{TEST}");
        release(TEST, device, 1);

        pages.remap(&new);
        device.host_range_remapped(pages.pointer(0), len as u64);
        assert!(
            read(&queue, &buffer, len) == new,
            "{TEST}: the released import"
        );
        let counts = device.work_counts();
        assert_eq!(residency(&counts), (1, 1, 1, 0), "{TEST}: imported again");
        drop(buffer);
        eprintln!(
            "{TEST}: on {}: the import made again read the new pages",
            device.name()
        );
    });
}

/// What a command buffer did, read once it completed or [`FILE_REMAP_WAIT`] passed: `None` where
/// it completed with no error.
#[cfg(any(m2v_cpu_icd, all(m2v_gpu, m2v_gpu_hazards)))]
fn outcome(command_buffer: &crate::CommandBuffer) -> Option<String> {
    let waited = command_buffer.wait_until_completed_for(FILE_REMAP_WAIT);
    match (waited, command_buffer.error()) {
        (Ok(()), None) => None,
        (Ok(()), Some(error)) => Some(format!("failed: {error:?}")),
        (Err(timed_out), _) => Some(format!("not completed: {timed_out:?}")),
    }
}

/// How long [`a_file_s_pages_mapped_under_a_kept_pointer_import`] waits for each command buffer.
#[cfg(any(m2v_cpu_icd, all(m2v_gpu, m2v_gpu_hazards)))]
const FILE_REMAP_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// A file's pages (a `memfd`'s, mapped shared) mapped over a no-copy buffer's private anonymous
/// pages while its pointer import is kept, the device told (`Device::host_range_remapped`): a
/// remap `HostMemory::new`'s contract forbids under a live buffer (`docs/FAST-PATHS.md`, known
/// gaps). Then a command buffer that fills another buffer alone, and one that copies the no-copy
/// buffer. On RADV both fail and the `VkDevice` is lost: amdgpu's userptr takes only pages with no
/// file behind them and gets each live import's pages again at every submission, whose list holds
/// every memory the device imported (Mesa 26.2.4). The device then recovers, a new `VkDevice` behind
/// it (`Degradation::DeviceRecovered`, its command buffers in flight failed as the hang and its
/// victims), or answers as lost where that new one could not be made: either passes. On lavapipe
/// both complete and the copy reads the file's pages. On any other driver the test prints what each
/// did. A GPU hazard (`m2v_gpu_hazards`): it loses a GPU's device on purpose, and on the RX 7900 XTX
/// (RADV) the loss put the GPU through a recovery.
#[cfg(any(m2v_cpu_icd, all(m2v_gpu, m2v_gpu_hazards)))]
#[test]
fn a_file_s_pages_mapped_under_a_kept_pointer_import() {
    const TEST: &str = "pointer import file remap";
    on_device(|device| {
        let page = host_page();
        let len = 2 * page;
        let queue = device.new_command_queue().unwrap();
        let old = counting(len, 13);
        let new = counting(len, 131);
        let pages = Pages::private(len);
        pages.write(0, &old);
        // SAFETY: the pages outlive the buffer; the remap below is the measurement.
        let made = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(pages.pointer(0)),
                len as u64,
                ResourceOptions::default(),
                None,
            )
        };
        let buffer = match made {
            Ok(buffer) => buffer,
            Err(error) => {
                eprintln!(
                    "{TEST}: on {}: private pages refused: {error:?}",
                    device.name()
                );
                return;
            }
        };
        assert!(read(&queue, &buffer, len) == old, "{TEST}");

        // SAFETY: a new anonymous file, owned at once, sized and mapped over the pages' range.
        unsafe {
            let fd = libc::memfd_create(c"m2v-file-remap".as_ptr(), libc::MFD_CLOEXEC);
            assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
            assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0);
            let at = pages.pointer(0).as_ptr().cast();
            let flags = libc::MAP_SHARED | libc::MAP_FIXED;
            let mapped = libc::mmap(at, len, libc::PROT_READ | libc::PROT_WRITE, flags, fd, 0);
            assert_eq!(mapped, at, "mmap of the file over the pages");
            libc::close(fd);
        }
        pages.write(0, &new);
        device.host_range_remapped(pages.pointer(0), len as u64);

        let other = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let fill = queue.command_buffer().unwrap();
        let mut blit = fill.blit_command_encoder().unwrap();
        blit.fill_buffer(&other, 0..64, 0x3c).unwrap();
        blit.end_encoding().unwrap();
        let filled = fill
            .commit()
            .map_err(|e| format!("{e:?}"))
            .err()
            .or_else(|| outcome(&fill));

        let read_back = (|| {
            let refused = |error: Error| format!("refused: {error:?}");
            let copy =
                (device.new_buffer(len as u64, ResourceOptions::shared())).map_err(refused)?;
            let copied = queue.command_buffer().map_err(refused)?;
            let mut blit = copied.blit_command_encoder().map_err(refused)?;
            (blit.copy_from_buffer_to_buffer(&buffer, 0, &copy, 0, len as u64)).map_err(refused)?;
            blit.end_encoding().map_err(refused)?;
            copied.commit().map_err(refused)?;
            if let Some(failed) = outcome(&copied) {
                return Err(failed);
            }
            let at = copy.contents().ok_or("no contents")?;
            // SAFETY: the copy's own bytes, which its completed command buffer wrote.
            Ok(unsafe { std::slice::from_raw_parts(at.as_ptr(), len).to_vec() })
        })();
        let read_of = match &read_back {
            Ok(bytes) if *bytes == new => "read the file's pages".to_owned(),
            Ok(bytes) if *bytes == old => "read the pages it was made over".to_owned(),
            Ok(_) => "read neither".to_owned(),
            Err(failed) => failed.clone(),
        };
        let lost = device.is_lost();
        let recoveries = device.work_counts().device_recoveries;
        eprintln!(
            "FILE-REMAP {TEST}: on {} (idle pointer imports released: {}): another buffer's fill \
             {}; the copy {read_of}; the device lost: {lost}; its recoveries: {recoveries}",
            device.name(),
            device.releases_idle_pointer_imports(),
            filled.as_deref().unwrap_or("completed"),
        );
        match device.properties().description.driver_id {
            Some(metal2vulkan::vk::DriverId::MESA_RADV) => {
                assert!(filled.is_some(), "{TEST}: another buffer's fill completed");
                assert!(read_back.is_err(), "{TEST}: the copy {read_of}");
                assert!(
                    lost || recoveries > 0,
                    "{TEST}: the device is neither lost nor recovered"
                );
            }
            Some(metal2vulkan::vk::DriverId::MESA_LLVMPIPE) => {
                assert_eq!(filled, None, "{TEST}: another buffer's fill");
                assert!(read_back == Ok(new), "{TEST}: the copy {read_of}");
            }
            _ => {}
        }
        drop(buffer);
    });
}
