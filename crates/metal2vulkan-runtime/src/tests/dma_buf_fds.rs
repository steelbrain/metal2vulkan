//! What a driver does with the dma-buf descriptor an import hands it (`memory::import_dma_buf`):
//! for udmabufs of host pages of a few sizes, each memory type the dma-buf imports as, and each
//! form (dedicated to nothing, and dedicated to a buffer made for dma-bufs), one
//! `vkAllocateMemory` of a clone of the descriptor, then whether the clone's number is still open
//! once the call returned, and, where it is not, whether the driver closes that number again later
//! (in the same call's failure path, a later import or the free of the memory): the number is
//! taken at once by a descriptor of `/dev/null` of the test's own (`F_DUPFD_CLOEXEC` from the
//! number, kept only where it returned the number itself: `dup2` would close whatever another
//! thread opened under it meanwhile), and after the later calls it must still be that file. Vulkan takes the descriptor only on a successful import
//! (`VkImportMemoryFdInfoKHR`); a driver that closes it on a failure, or closes a number twice,
//! closes whatever the process opened under that number meanwhile.
//!
//! Prints one line per case (`DMA-BUF-FDS`) and checks that no number the driver let go of was
//! closed again, nor any number the test kept: the runtime keeps a handed number from other
//! descriptors only while the import runs (`handed_descriptor.rs`), so a driver's later close of
//! it would reach whatever took the number since. A second close within the one call, which
//! NVIDIA's 620.70.06 made in some failed imports under eight threads, is not seen here, single
//! threaded (`tests::handed_descriptor` covers what the runtime makes of one).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, or the device lists no dma-buf route, it says so and
//! passes without running.

use super::host_residency::{host_page, udmabuf, udmabuf_openable, Scratch};
use super::no_pointer_import::{device, lists_dma_buf_route};
use crate::release;
use ash::vk;
use std::os::fd::{AsFd, AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};

const TEST: &str = "DMA-BUF-FDS";

/// Whether descriptor `fd` is open.
fn open(fd: RawFd) -> bool {
    // SAFETY: `F_GETFD` reads the descriptor's flags and changes nothing.
    unsafe { libc::fcntl(fd, libc::F_GETFD) >= 0 }
}

/// The device and inode of the file descriptor `fd` refers to, where it is open.
fn identity(fd: RawFd) -> Option<(libc::dev_t, libc::ino_t)> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fstat` writes the structure where it succeeds.
    if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the call succeeded.
    let stat = unsafe { stat.assume_init() };
    Some((stat.st_dev, stat.st_ino))
}

/// A number the driver let go of, now holding the test's own descriptor of `/dev/null`.
struct Placed {
    fd: RawFd,
    what: String,
    null: (libc::dev_t, libc::ino_t),
}

/// A sealed anonymous `memfd` of `bytes`.
fn memfd(bytes: u64) -> OwnedFd {
    // SAFETY: a new anonymous file, owned at once, then sized and sealed.
    unsafe {
        let fd = libc::memfd_create(
            c"m2v-dma-buf-fds".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        );
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        let fd = OwnedFd::from_raw_fd(fd);
        assert_eq!(libc::ftruncate(fd.as_raw_fd(), bytes as libc::off_t), 0);
        let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
        fd
    }
}

#[test]
fn a_driver_closes_no_descriptor_number_it_let_go_of() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0, None);
    let name = device.name().to_owned();
    if !lists_dma_buf_route(&device) {
        println!("{TEST} on {name}: no dma-buf route: not run");
        return;
    }
    let page = host_page() as u64;
    let file = memfd(4 << 20);
    let null = std::fs::File::open("/dev/null").unwrap();
    let null_identity = identity(null.as_raw_fd()).unwrap();
    let mut placed: Vec<Placed> = Vec::new();
    // Numbers the driver let go of that another descriptor took before the test could.
    let mut taken_meanwhile = 0;
    let mut freed = Vec::new();
    let mut lines = Vec::new();
    let shared = device.shared();
    let raw = shared.vk();
    let importer = shared.host_import.as_ref().unwrap();
    for bytes in [page, 16 * page, 1 << 20] {
        for dedicated in [false, true] {
            let chunk = udmabuf(&file, 0, bytes).unwrap();
            // SAFETY: the importer is this device's, and the descriptor a dma-buf.
            let types = unsafe { importer.dma_buf_types(raw, chunk.as_fd()) }.unwrap_or(0);
            assert!(
                open(chunk.as_raw_fd()),
                "{TEST} on {name}: vkGetMemoryFdPropertiesKHR closed the descriptor it read"
            );
            let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let info = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .push_next(&mut external);
            // SAFETY: the device is live and `info` valid.
            let buffer = unsafe { crate::creation::buffer(raw, &info) }.unwrap();
            // SAFETY: `buffer` is a live buffer of the device.
            let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
            let allowed = types & requirements.memory_type_bits;
            for type_index in (0..32).filter(|index| allowed & (1 << index) != 0) {
                let handed = chunk.try_clone().unwrap().into_raw_fd();
                let mut import = vk::ImportMemoryFdInfoKHR::default()
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                    .fd(handed);
                let mut to_buffer = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
                let mut allocate = vk::MemoryAllocateInfo::default()
                    .allocation_size(bytes.max(requirements.size))
                    .memory_type_index(type_index)
                    .push_next(&mut import);
                if dedicated {
                    allocate = allocate.push_next(&mut to_buffer);
                }
                // SAFETY: `allocate` and its chain live for the call; the type is the device's
                // and allows the dma-buf and the buffer.
                let made = unsafe { crate::creation::allocate_memory(raw, &allocate) };
                let still = open(handed);
                let what = format!(
                    "{bytes} bytes, type {type_index}, {}: {}",
                    match dedicated {
                        true => "dedicated to its buffer",
                        false => "dedicated to nothing",
                    },
                    match &made {
                        Ok(_) => "imported".to_owned(),
                        Err(error) => format!("{error}"),
                    }
                );
                let after = match (&made, still) {
                    (Ok(_), true) => "the descriptor open after the import",
                    (Ok(_), false) => "the descriptor closed by the import",
                    (Err(_), true) => "the descriptor left open by the failure",
                    (Err(_), false) => "the descriptor closed by the failure",
                };
                lines.push(format!("{what}, {after}"));
                if still {
                    if made.is_err() {
                        // SAFETY: the failure left the number open, and it is the test's again.
                        drop(unsafe { OwnedFd::from_raw_fd(handed) });
                    }
                } else {
                    // The number the driver let go of, taken at once by the test's own file:
                    // a clone of `/dev/null` at the lowest free number from it, which is the
                    // number itself unless another thread (the driver's own among them) opened a
                    // descriptor under it meanwhile; that number is not the test's to watch.
                    // SAFETY: `F_DUPFD_CLOEXEC` makes a new descriptor and closes none.
                    let taken =
                        unsafe { libc::fcntl(null.as_raw_fd(), libc::F_DUPFD_CLOEXEC, handed) };
                    assert!(taken >= 0, "F_DUPFD_CLOEXEC");
                    match taken == handed {
                        true => placed.push(Placed {
                            fd: handed,
                            what: what.clone(),
                            null: null_identity,
                        }),
                        false => {
                            taken_meanwhile += 1;
                            // SAFETY: the clone is the test's own and nothing else holds it.
                            drop(unsafe { OwnedFd::from_raw_fd(taken) });
                        }
                    }
                }
                if let Ok(memory) = made {
                    freed.push((memory, what));
                }
            }
            // SAFETY: nothing is bound to the buffer, which no command used.
            unsafe { release::object(raw, buffer) };
        }
    }
    // The frees, after which the driver holds none of the descriptors.
    for (memory, _) in &freed {
        // SAFETY: nothing is bound to the memory any more, and no command used it.
        unsafe { release::object(raw, *memory) };
    }
    let closed_again = |placed: &[Placed]| -> Vec<String> {
        (placed.iter())
            .filter(|placed| identity(placed.fd) != Some(placed.null))
            .map(|placed| placed.what.clone())
            .collect()
    };
    let by_frees = closed_again(&placed);
    drop(device);
    let by_teardown = closed_again(&placed);
    println!(
        "{TEST} on {name}: {}; numbers let go of and taken by the test: {} (taken by another \
         descriptor first: {taken_meanwhile}), closed again by the driver by the frees: {}, by \
         the device's teardown: {}",
        lines.join(" | "),
        placed.len(),
        by_frees.len(),
        by_teardown.len()
    );
    for placed in placed {
        // SAFETY: the test's own descriptor of `/dev/null` where it is still open.
        if identity(placed.fd) == Some(placed.null) {
            drop(unsafe { OwnedFd::from_raw_fd(placed.fd) });
        }
    }
    assert!(
        by_teardown.is_empty(),
        "{TEST} on {name}: the driver closed again a number it let go of: {by_teardown:?}"
    );
    assert!(open(null.as_raw_fd()) && open(file.as_raw_fd()));
}
