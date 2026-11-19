//! Imports: the two type rules, and under a device cfg, on every device of the cfg's kind, that
//! host memory is imported into the first host-visible, host-coherent type its query offers, that
//! a GPU write reaches it and its release runs only once its memory is freed, that an ordinary
//! Shared allocation succeeds after an import of a shared mapping of a file (which RADV refuses
//! at the import), that an import counts against the device's allocations, and that a dma-buf is
//! imported where its first offered type is host-visible and host-coherent and refused at the
//! query, with nothing imported, everywhere else.

use crate::import::host_type;

/// The dma-buf rule takes the first type the fd's query offers, only where it is host-visible,
/// host-coherent and listed by the buffer's requirements; never a later one.
#[cfg(unix)]
#[test]
fn a_dma_buf_takes_only_the_first_type_offered() {
    use crate::import::dma_buf_type;
    // Coherent types 1, 2, 3, 4 and 5; the buffer lists 0 to 5.
    let coherent = 0b11_1110;
    let listed = 0b11_1111;
    let cases = [
        // The first offered (0) is not coherent: refused, though 4 is offered and coherent.
        (0x11, None),
        (0x7f, None),
        // The first offered (5) is coherent and listed.
        (0x420, Some(5)),
        (0x20, Some(5)),
        // The first offered (1) is coherent but not listed by a buffer that lists 0 and 2.
        (0x06, None),
        (0, None),
    ];
    for (offered, expected) in cases {
        let listed = if offered == 0x06 { 0b101 } else { listed };
        assert_eq!(
            dma_buf_type(offered, listed, coherent),
            expected,
            "offered {offered:#x}"
        );
    }
}

/// The host-memory rule takes the first type offered that is host-visible, host-coherent and
/// listed by the buffer's requirements.
#[test]
fn host_memory_takes_the_first_coherent_listed_type_offered() {
    // Offered 3 and 4, both coherent: 3.
    assert_eq!(host_type(0x18, 0x3b, 0b1_1110), Some(3));
    // Offered 0 to 6, 0 not coherent: 1.
    assert_eq!(host_type(0x7f, 0x07, 0b110), Some(1));
    // Offered 5 alone, coherent and listed.
    assert_eq!(host_type(0x20, 0x7ad, 0b10_0000), Some(5));
    // Nothing offered that is coherent and listed.
    assert_eq!(host_type(0x01, 0xff, 0b1110), None);
    assert_eq!(host_type(0x08, 0x07, 0b1000), None);
    assert_eq!(host_type(0, 0xff, 0xff), None);
}

/// On unix, where the tests map host memory themselves.
#[cfg(all(unix, any(m2v_gpu, m2v_cpu_icd)))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{Device, DeviceOptions, Error, ImportHandle, Limit, MemoryClass};
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// The runtime's own allocation besides the pool's: the zero source of Private fills.
    const ZERO_SOURCE: u32 = 1;

    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// `len` bytes of fresh anonymous memory at a multiple of `alignment`, each byte its offset
    /// times 7; and the mapping to unmap, its address and length.
    fn anonymous(len: u64, alignment: u64) -> (NonNull<u8>, (usize, usize)) {
        let whole = (len + alignment) as usize;
        // SAFETY: a new private mapping, which nothing else uses.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                whole,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED, "an anonymous mapping");
        let start = (base as u64).next_multiple_of(alignment) as usize;
        let pointer = NonNull::new(start as *mut u8).expect("a mapping's address");
        for at in 0..len as usize {
            // SAFETY: `at` is inside the mapping.
            unsafe { pointer.add(at).write((at * 7) as u8) };
        }
        (pointer, (base as usize, whole))
    }

    fn unmap((address, len): (usize, usize)) {
        // SAFETY: a mapping the test made, which nothing uses any more.
        assert_eq!(
            unsafe { libc::munmap(address as *mut libc::c_void, len) },
            0
        );
    }

    /// A release that sets a flag, and the flag.
    fn release() -> (Option<Box<dyn FnOnce() + Send>>, Arc<AtomicBool>) {
        let ran = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&ran);
        let release: Box<dyn FnOnce() + Send> = Box::new(move || set.store(true, Ordering::SeqCst));
        (Some(release), ran)
    }

    /// Host memory is imported into a Shared block whose contents are the memory: a fill of its
    /// second half by the GPU reaches the host, the first half keeps the host's bytes, and the
    /// release runs once the last handle and buffer are gone and the memory is freed, with the
    /// gauges back. A device that does not import host memory refuses it typed.
    #[test]
    fn every_device_imports_host_memory_and_releases_it_after_the_free() {
        each_device("import_host_memory", |name, _, device| {
            let counters = device.counters();
            let Some((_, alignment)) = device.core().imports.host.as_ref() else {
                let (pointer, mapping) = anonymous(64 << 10, 4096);
                let (release, _) = release();
                // SAFETY: the memory is the test's, mapped until it unmaps it below.
                match unsafe { device.import_host_memory(pointer, 64 << 10, release) } {
                    Err(Error::Limit(Limit::NoImport {
                        handle: ImportHandle::HostMemory,
                    })) => {}
                    other => panic!("{name}: {other:?}"),
                }
                unmap(mapping);
                println!("import: {name}: imports no host memory, refused typed");
                return;
            };
            let alignment = *alignment;
            let len = (64u64 << 10).next_multiple_of(alignment);
            let (pointer, mapping) = anonymous(len, alignment);
            let before = counters.gauges();
            let (release, released) = release();
            // SAFETY: the memory is the test's, mapped until it unmaps it after the release.
            let (block, memory_type) = unsafe { device.import_host_memory(pointer, len, release) }
                .unwrap_or_else(|error| panic!("{name}: an import: {error}"));
            assert_eq!(
                (block.class, block.contents, block.len, block.offset),
                (MemoryClass::Shared, Some(pointer), len, 0),
                "{name}"
            );
            let live = counters.gauges();
            assert_eq!(
                (live.imports, live.blocks),
                (1, before.blocks + 1),
                "{name}"
            );
            let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
            assert_eq!(buffer.contents(), Some(pointer), "{name}");
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .fill_buffer(&buffer, len / 2..len, 0x5a)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            for at in 0..len as usize {
                // SAFETY: `at` is inside the imported memory, which the test still maps.
                let byte = unsafe { pointer.add(at).read_volatile() };
                let expected = if at < len as usize / 2 {
                    (at * 7) as u8
                } else {
                    0x5a
                };
                assert_eq!(byte, expected, "{name}: byte {at}");
            }
            drop((command_buffer, queue, buffer));
            assert!(
                !released.load(Ordering::SeqCst),
                "{name}: released while held"
            );
            let device_local = block.device_local;
            let held = counters.gauges().blocks;
            drop(block);
            wait_until("the release", || released.load(Ordering::SeqCst));
            let after = counters.gauges();
            assert_eq!((after.imports, after.blocks), (0, held - 1), "{name}");
            unmap(mapping);
            println!(
                "import: {name}: host memory imported into type {memory_type} (device-local \
                 {device_local}), written by the GPU, released after its free"
            );
        });
    }

    /// After an import of 64 KiB of a shared mapping of a file (the shape of a guest's RAM, which
    /// RADV refuses at the import), whatever its answer, an ordinary Shared allocation succeeds:
    /// a take of 32 MiB, past the pool's 16 MiB allocations, which makes its own. A refused
    /// import leaves the gauges as they were.
    #[test]
    fn every_device_allocates_shared_memory_after_an_import_of_a_shared_file_mapping() {
        each_device("import_shared_file", |name, _, device| {
            let counters = device.counters();
            let alignment = match device.core().imports.host.as_ref() {
                Some((_, alignment)) => *alignment,
                None => {
                    println!("import: {name}: imports no host memory");
                    return;
                }
            };
            let len = (64u64 << 10).next_multiple_of(alignment);
            let file = tempfile(len);
            // SAFETY: a new shared mapping of the test's file, which nothing else uses.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    std::os::fd::AsRawFd::as_raw_fd(&file),
                    0,
                )
            };
            assert_ne!(base, libc::MAP_FAILED, "{name}: a shared mapping");
            let pointer = NonNull::new(base.cast::<u8>()).expect("a mapping's address");
            let before = counters.gauges();
            let (release, released) = release();
            // SAFETY: the memory is the test's, mapped until it unmaps it below.
            let answer = unsafe { device.import_host_memory(pointer, len, release) };
            let answered = match answer {
                Ok((block, _)) => {
                    drop(block);
                    wait_until("the release", || released.load(Ordering::SeqCst));
                    "imported".to_string()
                }
                Err(Error::Vulkan { call, result }) => {
                    assert_eq!(counters.gauges(), before, "{name}: a refused import");
                    format!("refused at {call} ({result:?})")
                }
                Err(Error::Limit(Limit::ImportType { offered, .. })) => {
                    assert_eq!(counters.gauges(), before, "{name}: a refused import");
                    format!("refused at the query (offered {offered:#x})")
                }
                Err(other) => panic!("{name}: {other}"),
            };
            let blocks = counters.gauges().blocks;
            let shared = (device.take_block(MemoryClass::Shared, 32 << 20))
                .unwrap_or_else(|error| panic!("{name}: a Shared allocation after it: {error}"));
            assert_eq!(
                counters.gauges().blocks,
                blocks + 1,
                "{name}: its own allocation"
            );
            let contents = shared.contents.expect("a Shared block's contents");
            // SAFETY: the block's mapping spans it.
            unsafe { contents.as_ptr().write_volatile(1) };
            drop(shared);
            // SAFETY: the test's mapping, which nothing uses any more.
            assert_eq!(unsafe { libc::munmap(base, len as usize) }, 0);
            println!(
                "import: {name}: a shared file mapping {answered}; a Shared allocation after it"
            );
        });
    }

    /// A file of `len` bytes in the OS temp dir, removed from the directory at once.
    fn tempfile(len: u64) -> std::fs::File {
        let path = std::env::temp_dir().join(format!("m2v-import-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("a scratch file");
        std::fs::remove_file(&path).expect("the scratch file's removal");
        file.set_len(len).expect("the scratch file's length");
        file
    }

    /// An import counts against the device's `maxMemoryAllocationCount`: with the count answered
    /// as one past the device's allocations at its making, one import is made, and the next is
    /// `Limit::MemoryAllocations` before the driver is called, its release dropped unrun.
    #[test]
    fn every_device_counts_imports_against_its_allocations() {
        const MAX: u32 = ZERO_SOURCE + 3;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::open(physical, DeviceOptions::default(), |limits| {
                limits.max_memory_allocation_count = MAX;
            })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let Some((_, alignment)) = device.core().imports.host.as_ref() else {
                println!("import: {name}: imports no host memory");
                continue;
            };
            let alignment = *alignment;
            wait_until("the first allocation of each class", || {
                counters.gauges().blocks == u64::from(MAX - 1)
            });
            let len = (64u64 << 10).next_multiple_of(alignment);
            let (first, first_mapping) = anonymous(len, alignment);
            let (second, second_mapping) = anonymous(len, alignment);
            let (release, first_released) = release();
            // SAFETY: the memory is the test's, mapped until it unmaps it after the release.
            let (block, _) = unsafe { device.import_host_memory(first, len, release) }
                .unwrap_or_else(|error| panic!("{name}: the first import: {error}"));
            assert_eq!(counters.gauges().blocks, u64::from(MAX), "{name}");
            let (release, second_released) = release();
            // SAFETY: as above.
            match unsafe { device.import_host_memory(second, len, release) } {
                Err(Error::Limit(Limit::MemoryAllocations { max: MAX })) => {}
                other => panic!("{name}: an import past the count: {other:?}"),
            }
            assert_eq!(counters.gauges().blocks, u64::from(MAX), "{name}");
            drop(block);
            wait_until("the release", || first_released.load(Ordering::SeqCst));
            assert!(!second_released.load(Ordering::SeqCst), "{name}");
            drop(device);
            unmap(first_mapping);
            unmap(second_mapping);
            println!("import: {name}: an import past the count refused");
        }
    }

    /// A dma-buf of 16 pages of a memfd, through `/dev/udmabuf`, and the memfd; `None` where the
    /// kernel offers no `/dev/udmabuf` to this process.
    #[cfg(target_os = "linux")]
    fn dma_buf(len: u64) -> Option<(std::os::fd::OwnedFd, std::os::fd::OwnedFd)> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        #[repr(C)]
        struct Create {
            memfd: u32,
            flags: u32,
            offset: u64,
            size: u64,
        }
        // `_IOW('u', 0x42, struct udmabuf_create)`.
        const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;
        const UDMABUF_FLAGS_CLOEXEC: u32 = 1;
        let device = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .ok()?;
        // SAFETY: a new memfd, owned at once.
        let memfd = unsafe {
            let fd = libc::memfd_create(
                c"m2v-import".as_ptr(),
                libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC,
            );
            assert!(fd >= 0, "a memfd");
            OwnedFd::from_raw_fd(fd)
        };
        // SAFETY: the memfd is ours.
        unsafe {
            assert_eq!(libc::ftruncate(memfd.as_raw_fd(), len as libc::off_t), 0);
            assert_eq!(
                libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK),
                0
            );
        }
        let create = Create {
            memfd: memfd.as_raw_fd() as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset: 0,
            size: len,
        };
        // SAFETY: the ioctl reads `create`, which lives through it.
        let fd = unsafe {
            libc::ioctl(
                device.as_raw_fd(),
                UDMABUF_CREATE,
                std::ptr::from_ref(&create),
            )
        };
        assert!(fd >= 0, "a dma-buf: {}", std::io::Error::last_os_error());
        // SAFETY: the ioctl returned a new fd.
        Some((unsafe { OwnedFd::from_raw_fd(fd) }, memfd))
    }

    /// A dma-buf is imported where the first type its fd's query offers is host-visible and
    /// host-coherent, and a GPU fill reaches its pages; elsewhere it is refused at the query with
    /// nothing imported, and an ordinary Shared allocation and a host-memory import still
    /// succeed after it. A device that does not import a dma-buf refuses it typed.
    #[cfg(target_os = "linux")]
    #[test]
    fn every_device_imports_a_dma_buf_only_into_its_first_offered_type() {
        use crate::import::dma_buf_type;
        each_device("import_dma_buf", |name, _, device| {
            const LEN: u64 = 64 << 10;
            let Some((fd, memfd)) = dma_buf(LEN) else {
                println!("import: {name}: no /dev/udmabuf here; the dma-buf import not run");
                return;
            };
            let counters = device.counters();
            let before = counters.gauges();
            if device.core().imports.dma_buf.is_none() {
                match device.import_dma_buf(fd, LEN, None) {
                    Err(Error::Limit(Limit::NoImport {
                        handle: ImportHandle::DmaBuf,
                    })) => {}
                    other => panic!("{name}: {other:?}"),
                }
                println!("import: {name}: imports no dma-buf, refused typed");
                return;
            }
            let (release, released) = release();
            match device.import_dma_buf(fd, LEN, release) {
                Ok((block, memory_type)) => {
                    assert_eq!(
                        (block.class, block.contents, block.len, block.offset),
                        (MemoryClass::Shared, None, LEN, 0),
                        "{name}"
                    );
                    let buffer = (device.new_buffer(block.id, 0, LEN, None)).expect("a buffer");
                    let queue = device.new_command_queue(4).expect("a queue");
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                    encoder
                        .fill_buffer(&buffer, 0..LEN, 0x3c)
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    encoder.end_encoding();
                    command_buffer.commit().expect("a commit");
                    finished(name, &command_buffer);
                    assert_eq!(command_buffer.error(), None, "{name}");
                    let mut pages = vec![0u8; LEN as usize];
                    let read = std::os::unix::fs::FileExt::read_exact_at(
                        &std::fs::File::from(memfd),
                        &mut pages,
                        0,
                    );
                    read.expect("the memfd's bytes");
                    assert!(pages.iter().all(|&byte| byte == 0x3c), "{name}: the pages");
                    drop((command_buffer, queue, buffer, block));
                    wait_until("the release", || released.load(Ordering::SeqCst));
                    assert_eq!(counters.gauges().imports, 0, "{name}");
                    println!(
                        "import: {name}: a dma-buf imported into its first offered type, \
                         {memory_type}"
                    );
                }
                Err(Error::Limit(Limit::ImportType {
                    handle: ImportHandle::DmaBuf,
                    offered,
                    listed,
                })) => {
                    let coherent = device.core().imports.coherent;
                    assert_eq!(dma_buf_type(offered, listed, coherent), None, "{name}");
                    assert_eq!(counters.gauges(), before, "{name}: nothing imported");
                    assert!(!released.load(Ordering::SeqCst), "{name}");
                    let shared = (device.take_block(MemoryClass::Shared, 32 << 20))
                        .unwrap_or_else(|error| panic!("{name}: a Shared allocation: {error}"));
                    drop(shared);
                    if let Some((_, alignment)) = device.core().imports.host.as_ref() {
                        let len = LEN.next_multiple_of(*alignment);
                        let (pointer, mapping) = anonymous(len, *alignment);
                        let (release, host_released) = release();
                        // SAFETY: the memory is the test's, mapped until it unmaps it below.
                        let (block, _) =
                            unsafe { device.import_host_memory(pointer, len, release) }
                                .unwrap_or_else(|error| panic!("{name}: host memory: {error}"));
                        drop(block);
                        wait_until("the release", || host_released.load(Ordering::SeqCst));
                        unmap(mapping);
                    }
                    println!(
                        "import: {name}: a dma-buf refused at the query (offered {offered:#x}, \
                         listed {listed:#x}); a Shared allocation and host memory after it"
                    );
                }
                Err(other) => panic!("{name}: {other}"),
            }
        });
    }
}
