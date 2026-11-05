//! A detach waiting for submitted work settles when the device is lost (`detach.rs`): a no-copy
//! buffer a submitted kernel names, spinning until the host raises a word, is detached on another
//! thread, which blocks; the device is then marked lost (as `device/relay.rs` marks it), the word
//! not yet raised, and the detach answers `Err(DeviceLost)` within its poll.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects, and on a GPU only with `m2v_gpu_hazards` too (docs/VALIDATION.md). It is a GPU
//! hazard: one dispatch spins up to 2^25 loads (about 4.5 s on the M3 Max, 2 s on RADV) where the
//! GPU does not see the host raise the word while the kernel runs, past what Metal ends as a hang.

use crate::error::Error;
use crate::{
    ComputePipelineState, Device, DeviceKind, DeviceOptions, HostMemory, PipelineOptions,
    ResourceOptions, ShaderCacheOptions, Size,
};
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// The longest a command buffer may take to start.
const WAIT: Duration = Duration::from_secs(60);

/// The no-copy buffer's bytes: a whole page of any host.
const BYTES: usize = 64 << 10;

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

/// Page-aligned host memory of [`BYTES`], freed when dropped: declared first, after the device
/// that used it.
struct Host(NonNull<u8>);

impl Host {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, BYTES).unwrap()
    }

    fn new() -> Host {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Host::layout()) };
        Host(NonNull::new(pointer).unwrap())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Host::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Host::layout()) };
    }
}

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

/// A buffer's detach blocked on submitted work that took it answers `Err(DeviceLost)` once the
/// device is lost, the work not yet complete.
#[test]
fn a_detach_waiting_for_submitted_work_settles_on_a_lost_device() {
    let host = Host::new();
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    // SAFETY: the memory outlives the device (dropped after it, below), and nothing else uses
    // it meanwhile.
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
    let raise = || {
        let word = flag.contents().unwrap().as_ptr().cast::<u32>();
        // SAFETY: the flag's `Shared` bytes are mapped for its life, aligned for a word, and the
        // GPU only loads them.
        unsafe { word.write_volatile(1) };
    };
    std::thread::scope(|scope| {
        let detaching = scope.spawn(|| buffer.detach_backing());
        std::thread::sleep(Duration::from_millis(100));
        let waited = !detaching.is_finished();
        device.mark_lost();
        let answer = detaching.join().unwrap();
        // The GPU is never left spinning.
        raise();
        assert!(waited, "the detach did not wait for the submitted work");
        assert_eq!(answer, Err(Error::DeviceLost));
    });
}
