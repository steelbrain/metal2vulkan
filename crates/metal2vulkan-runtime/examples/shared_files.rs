//! Not a consumer's example: what one kernel submission costs beside no-copy buffers over
//! dma-bufs of their pages, each a dma-buf of its own against pieces of files shared among them
//! (`DmaBufChunk`, one import per file), on a device that refuses the pages' pointer: RADV refuses
//! the pointer of a `memfd`'s shared mapping, as of QEMU's guest RAM, and lists every import in
//! every kernel submission. Linux only, with access to `/dev/udmabuf`.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example shared_files [-- BUFFERS [PER_FILE ...]]
//! ```
//!
//! For each count of buffers a file (by default 1, 8 and `BUFFERS`, which is 1000 by default):
//! `BUFFERS` no-copy buffers of 64 KiB over one sealed `memfd` mapped shared, each
//! answered by a source with a piece of the `udmabuf` of its file's run of the `memfd` (1: a
//! `udmabuf` of its own pages, a dma-buf of its own), each buffer's Metal address handed out (so
//! its import stays imported, as a guest's buffers' do); then 300 command buffers of one 4-KiB
//! fill, one at a time, each waited for, after 30 untimed. One `SHARED-FILES` line
//! each: the imports made, the pieces joined to a live import, the dma-buf memories and shared
//! imports live (`WorkCounts::dma_buf_imports`, `dma_buf_pieces_joined`, `dma_buf_memories_live`,
//! `dma_buf_shared_imports_live`), the imports released past the most the device keeps and those
//! imported again (`host_imports_released_over_most`, `host_imports_reimported`: a buffer released
//! while the others were made is imported again as its address is handed out, a shared file's
//! piece joining the file's live import again), a command buffer's time from its kernelStartTime
//! to its kernelEndTime (50th and 90th percentiles), and the mean time of a `vkQueueSubmit` call
//! in the driver (`WorkCounts::batch_submit_nanos` over `queue_submissions`). The device's
//! submission checks answer first, so that every phase runs under their verdicts and none beside
//! their work. A device that imports the pointer (NVIDIA, lavapipe) makes no dma-buf import, which
//! the line shows.

#[cfg(target_os = "linux")]
mod linux {
    use metal2vulkan_runtime::{
        Buffer, CommandBufferStatus, CommandQueue, Device, DmaBufChunk, DmaBufRequest,
        DmaBufSource, HostMemory, ResourceOptions,
    };
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::time::Duration;

    /// The bytes of each buffer: one 64-KiB chunk, RADV 26.2.4's sparse granule.
    const BUFFER_BYTES: usize = 64 << 10;
    /// Command buffers committed untimed first.
    const WARM: usize = 30;
    /// Command buffers timed.
    const COMMITS: usize = 300;
    /// The longest a command buffer may take, and the device's submission checks.
    pub const WAIT: Duration = Duration::from_secs(60);

    /// `struct udmabuf_create` of Linux's `include/uapi/linux/udmabuf.h`.
    #[repr(C)]
    struct Create {
        memfd: u32,
        flags: u32,
        offset: u64,
        size: u64,
    }

    /// `UDMABUF_FLAGS_CLOEXEC`.
    const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;

    /// `UDMABUF_CREATE`, `_IOW('u', 0x42, struct udmabuf_create)` in the generic ioctl encoding.
    const UDMABUF_CREATE: u64 =
        (1 << 30) | ((size_of::<Create>() as u64) << 16) | ((b'u' as u64) << 8) | 0x42;

    /// A dma-buf of `memfd`'s `size` bytes from `offset`.
    fn udmabuf(memfd: &OwnedFd, offset: u64, size: u64) -> std::io::Result<OwnedFd> {
        let device = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")?;
        let create = Create {
            memfd: memfd.as_raw_fd() as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset,
            size,
        };
        // SAFETY: the ioctl reads `create`, a valid `udmabuf_create`.
        let fd = unsafe { libc::ioctl(device.as_raw_fd(), UDMABUF_CREATE as _, &create) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: a new descriptor the ioctl returned, owned at once.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Guest RAM: a `memfd` of `len` bytes sealed against growing and shrinking, mapped shared
    /// whole; unmapped when dropped.
    struct Ram {
        fd: OwnedFd,
        at: NonNull<u8>,
        len: usize,
    }

    impl Ram {
        fn new(len: usize) -> Ram {
            // SAFETY: a new anonymous file, owned at once, then sized and sealed.
            let fd = unsafe {
                let fd = libc::memfd_create(
                    c"m2v-shared-files".as_ptr(),
                    libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
                );
                assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
                OwnedFd::from_raw_fd(fd)
            };
            // SAFETY: the descriptor is this object's.
            unsafe {
                assert_eq!(libc::ftruncate(fd.as_raw_fd(), len as libc::off_t), 0);
                let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
                assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
            }
            // SAFETY: a new shared mapping of the whole file.
            let at = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(at, libc::MAP_FAILED, "mmap of the memfd");
            Ram {
                fd,
                at: NonNull::new(at.cast()).unwrap(),
                len,
            }
        }

        /// The mapping's address `offset` bytes in.
        fn pointer(&self, offset: usize) -> NonNull<u8> {
            assert!(offset < self.len);
            // SAFETY: inside the mapping.
            unsafe { NonNull::new_unchecked(self.at.as_ptr().add(offset)) }
        }

        /// A source answering each request with a piece of the file of `per_file` buffers'
        /// pages its pages lie in (made now, held by the source), or, for 1, a `udmabuf` of its
        /// own pages made as it is asked.
        fn source(&self, per_file: usize) -> DmaBufSource {
            let base = self.at.as_ptr() as u64;
            let file_bytes = (per_file * BUFFER_BYTES) as u64;
            let memfd = Arc::new(self.fd.try_clone().unwrap());
            let files: Vec<Arc<OwnedFd>> = match per_file {
                1 => Vec::new(),
                _ => (0..(self.len as u64).div_ceil(file_bytes))
                    .map(|index| {
                        let at = index * file_bytes;
                        let size = file_bytes.min(self.len as u64 - at);
                        Arc::new(udmabuf(&memfd, at, size).expect("a udmabuf of a file's run"))
                    })
                    .collect(),
            };
            Arc::new(move |request: DmaBufRequest| {
                let at = (request.start.as_ptr() as u64)
                    .checked_sub(base)
                    .ok_or_else(|| std::io::Error::other("pages before the mapping"))?;
                if files.is_empty() {
                    return Ok(vec![DmaBufChunk {
                        file: Arc::new(udmabuf(&memfd, at, request.length)?),
                        offset: 0,
                        length: request.length,
                    }]);
                }
                let (index, offset) = (at / file_bytes, at % file_bytes);
                let file = (files.get(index as usize))
                    .filter(|_| offset + request.length <= file_bytes)
                    .ok_or_else(|| std::io::Error::other("pages across two files"))?;
                Ok(vec![DmaBufChunk {
                    file: Arc::clone(file),
                    offset,
                    length: request.length,
                }])
            })
        }
    }

    impl Drop for Ram {
        fn drop(&mut self) {
            // SAFETY: the mapping is this object's, and its buffers went first.
            unsafe { libc::munmap(self.at.as_ptr().cast(), self.len) };
        }
    }

    /// Commit a fill of `target`'s 4 KiB and wait for it: the microseconds from its
    /// kernelStartTime to its kernelEndTime.
    fn submitted(queue: &CommandQueue, target: &Buffer, value: u8) -> f64 {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(target, 0..4096, value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        (command_buffer.kernel_end_time() - command_buffer.kernel_start_time()) * 1e6
    }

    /// The phase of `buffers` buffers, `per_file` a file: held, measured and printed.
    pub fn phase(
        device: &Device,
        queue: &CommandQueue,
        target: &Buffer,
        buffers: usize,
        per_file: usize,
    ) {
        let ram = Ram::new(buffers * BUFFER_BYTES);
        let source = ram.source(per_file);
        let before = device.work_counts();
        let held: Vec<Buffer> = (0..buffers)
            .map(|index| {
                let at = ram.pointer(index * BUFFER_BYTES);
                // SAFETY: the mapping outlives the buffers (dropped first), and the source makes
                // dma-bufs of exactly the pages asked for.
                unsafe {
                    device.new_buffer_with_bytes_no_copy(
                        HostMemory::with_dma_buf(at, Arc::clone(&source)),
                        BUFFER_BYTES as u64,
                        ResourceOptions::shared(),
                        None,
                    )
                }
                .unwrap()
            })
            .collect();
        drop(source);
        // Handed out, as a guest's buffers' addresses are: each import stays imported for good.
        for buffer in &held {
            buffer.gpu_address();
        }
        for value in 0..WARM {
            submitted(queue, target, value as u8);
        }
        let counts = device.work_counts();
        let mut times: Vec<f64> = (0..COMMITS)
            .map(|value| submitted(queue, target, value as u8))
            .collect();
        let after = device.work_counts();
        times.sort_by(f64::total_cmp);
        let at = |p: usize| times[(times.len() * p / 100).min(times.len() - 1)];
        let submits = after.queue_submissions - counts.queue_submissions;
        let in_driver = (after.batch_submit_nanos - counts.batch_submit_nanos) as f64
            / submits.max(1) as f64
            / 1e3;
        println!(
            "SHARED-FILES {}: {buffers} buffers, {per_file} a file: {} imports made, {} pieces \
             joined, {} dma-buf memories live ({} shared imports), {} released past the most, {} \
             imported again; submit p50 {:.1} us (p90 {:.1}); vkQueueSubmit {in_driver:.1} us \
             each in the driver over {submits} calls",
            device.name(),
            counts.dma_buf_imports - before.dma_buf_imports,
            counts.dma_buf_pieces_joined - before.dma_buf_pieces_joined,
            counts.dma_buf_memories_live,
            counts.dma_buf_shared_imports_live,
            counts.host_imports_released_over_most - before.host_imports_released_over_most,
            counts.host_imports_reimported - before.host_imports_reimported,
            at(50),
            at(90),
        );
        drop(held);
    }
}

#[cfg(target_os = "linux")]
fn main() {
    use metal2vulkan_runtime::{Device, ResourceOptions};
    let mut args = std::env::args()
        .skip(1)
        .map(|arg| arg.parse::<usize>().unwrap());
    let buffers = args.next().unwrap_or(1000);
    let mut per_file: Vec<usize> = args.collect();
    if per_file.is_empty() {
        per_file = vec![1, 8, buffers];
    }
    let device = Device::system_default().unwrap();
    let answered = device.settle_submit_checks(linux::WAIT);
    println!(
        "device: {}; submission checks answered: {answered}; idle imports released: {}",
        device.name(),
        device.releases_idle_pointer_imports(),
    );
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    for each in per_file {
        linux::phase(&device, &queue, &target, buffers, each);
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("shared_files: Linux only (udmabuf)");
}
