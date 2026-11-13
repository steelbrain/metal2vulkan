//! A detach waiting for submitted work settles when the device is lost (`detach.rs`): a no-copy
//! buffer a submitted kernel names, spinning until the host raises a word, is detached on another
//! thread, which blocks; the device is then marked lost (as `device/relay.rs` marks it), the word
//! not yet raised, and the detach answers within its poll: the device recovers from the loss
//! (`device/line.rs`), and the buffer, made before, is refused
//! (`NotImplemented(MadeBeforeRecovery)`).
//!
//! The kernel stores its count into the no-copy buffer's bytes once it sees the word, so they stay
//! mapped until its command buffer completed (`HostMemory::new`'s contract), which the test waits
//! for before they go: dropping the device's handle does not wait for its work in flight. The
//! word is raised late, on a thread of its own, so that a test that let the bytes go first would
//! fault there every time (`Host` unmaps them): a heap allocation freed under the store took the
//! count into its free list's link, and the process faulted in `exit`'s frees.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects, and on a GPU only with `m2v_gpu_hazards` too (docs/VALIDATION.md). It is a GPU
//! hazard: one dispatch spins up to 2^25 loads (about 4.5 s on the M3 Max, 2 s on RADV) where the
//! GPU does not see the host raise the word while the kernel runs, past what Metal ends as a hang.

use crate::error::{Error, RuntimeGap};
use crate::{
    ComputePipelineState, Device, DeviceKind, DeviceOptions, HostMemory, PipelineOptions,
    ResourceOptions, ShaderCacheOptions, Size,
};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// The longest a command buffer may take to start.
const WAIT: Duration = Duration::from_secs(60);

/// The no-copy buffer's bytes: a whole page of any host.
const BYTES: usize = 64 << 10;

/// How long after the detach answered the word is raised: past the test's own steps, while the
/// command buffer's wait holds the bytes mapped.
const RAISE_LATE: Duration = Duration::from_millis(200);

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-detach-lost-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// [`BYTES`] of pages of their own, mapped private and zeroed, unmapped when dropped, so that a
/// write after they went faults at once: declared first, dropped last.
struct Host(NonNull<u8>);

impl Host {
    fn new() -> Host {
        // SAFETY: a new anonymous mapping, unmapped as the host drops.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                BYTES,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(
            mapped,
            libc::MAP_FAILED,
            "{}",
            std::io::Error::last_os_error()
        );
        Host(NonNull::new(mapped.cast()).unwrap())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: mapped with this length in `Host::new`; nothing reads or writes it any more.
        unsafe { libc::munmap(self.0.as_ptr().cast(), BYTES) };
    }
}

/// The flag's word, for the thread that raises it.
struct Word(*mut u32);

// SAFETY: the word is the flag's `Shared` bytes, mapped while the command buffer that loads it is
// in flight, and only written through this.
unsafe impl Send for Word {}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The owned fixture's kernel that loads a word no one raises until its cap
/// (`kernel_spin_on_an_unraised_word`).
fn spin(device: &Device) -> ComputePipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../metal2vulkan-validation/fixtures/public/engine/kernel_spin_on_an_unraised_word.ll",
    );
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let function = library
        .new_function_with_name("spin_on_an_unraised_word")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// A buffer's detach blocked on submitted work that took it answers once the device is lost, the
/// work not yet complete: refused as made before the device's recovery. The bytes go once the
/// work completed, the word raised late (the module docs).
#[test]
fn a_detach_waiting_for_submitted_work_settles_on_a_lost_device() {
    let host = Host::new();
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    // SAFETY: the memory stays mapped until the command buffer that uses the buffer completed (the
    // wait below) and the buffer was dropped (dropped after it, below), and nothing else uses it
    // meanwhile.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            None,
        )
    };
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(error) => {
            eprintln!("detach lost: no buffer over host memory: {error:?}");
            return;
        }
    };
    let queue = device.new_command_queue().unwrap();
    let state = spin(&device);
    let flag = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&flag), 0, 0).unwrap();
    encoder.set_buffer(Some(&buffer), 0, 1).unwrap();
    encoder.set_bytes(&(1u32 << 25).to_le_bytes(), 2).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    let deadline = Instant::now() + WAIT;
    while command_buffer.kernel_start_time() == 0.0 {
        assert!(
            Instant::now() < deadline,
            "the first batch was never submitted"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let word = Word(flag.contents().unwrap().as_ptr().cast::<u32>());
    let (waited, answer) = std::thread::scope(|scope| {
        let detaching = scope.spawn(|| buffer.detach_backing());
        std::thread::sleep(Duration::from_millis(100));
        let waited = !detaching.is_finished();
        device.mark_lost();
        (waited, detaching.join().unwrap())
    });
    // The GPU is never left spinning: the word is raised late, whatever the test finds.
    let raising = std::thread::spawn(move || {
        std::thread::sleep(RAISE_LATE);
        let word = word;
        // SAFETY: the flag's `Shared` bytes stay mapped while the command buffer that loads them
        // is in flight, aligned for a word, and the GPU only loads them.
        unsafe { word.0.write_volatile(1) };
    });
    // The kernel stores into the bytes once it sees the word: they go only after its command
    // buffer completed.
    let completed = command_buffer.wait_until_completed();
    raising.join().unwrap();
    assert!(waited, "the detach did not wait for the submitted work");
    assert_eq!(
        answer,
        Err(Error::NotImplemented(RuntimeGap::MadeBeforeRecovery))
    );
    assert!(completed.is_ok(), "{completed:?}");
}
