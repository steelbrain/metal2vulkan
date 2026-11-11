//! The first dma-buf imports of a kind (`memory::import_dma_buf`): until an import of the kind has
//! succeeded on the device, its imports run one at a time; once one has, they run side by side.
//! Driven with the memory's stand-in for a driver that refuses every import, timed where the
//! driver's import runs, on a stand-in device the import never calls.

use crate::device::lock;
use crate::memory::{import_dma_buf, DmaBufFor, Memory, MemoryClass};
use ash::vk;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The imports each case makes at once, one a thread.
const THREADS: usize = 4;

/// How long an import's stand-in waits for another import to run beside it.
const BESIDE: Duration = Duration::from_millis(200);

/// A device of one host-visible, coherent memory type, whose every dma-buf import the stand-in
/// refuses as out of device memory, having counted the imports running there at once.
fn memory(most: &Arc<AtomicUsize>) -> Mutex<Memory> {
    use vk::MemoryPropertyFlags as P;
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 1,
        memory_heap_count: 1,
        ..Default::default()
    };
    properties.memory_types[0] = vk::MemoryType {
        property_flags: P::HOST_VISIBLE | P::HOST_COHERENT,
        heap_index: 0,
    };
    properties.memory_heaps[0] = vk::MemoryHeap {
        size: 1 << 30,
        flags: vk::MemoryHeapFlags::empty(),
    };
    let mut memory = Memory::new(&properties, 1 << 30, false, true, vec![0], None, None);
    let inside = Arc::new(AtomicUsize::new(0));
    let most = Arc::clone(most);
    // Where the driver's import runs: counted, then held until another import runs beside it
    // or `BESIDE` passed.
    memory.import_hold = Some(Arc::new(move || {
        let now = inside.fetch_add(1, Ordering::SeqCst) + 1;
        most.fetch_max(now, Ordering::SeqCst);
        let start = Instant::now();
        while inside.load(Ordering::SeqCst) < 2 && start.elapsed() < BESIDE {
            std::thread::yield_now();
        }
        most.fetch_max(inside.load(Ordering::SeqCst), Ordering::SeqCst);
        inside.fetch_sub(1, Ordering::SeqCst);
    }));
    memory.import_refusal = Some(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    Mutex::new(memory)
}

/// A device whose functions are none: the stand-ins answer before any would be called.
fn device() -> ash::Device {
    let none = |_: &std::ffi::CStr| std::ptr::null::<std::ffi::c_void>();
    ash::Device::from_parts_1_3(
        vk::Device::null(),
        ash::DeviceFnV1_0::load(none),
        ash::DeviceFnV1_1::load(none),
        ash::DeviceFnV1_2::load(none),
        ash::DeviceFnV1_3::load(none),
    )
}

/// `THREADS` imports of a dma-buf for shared memory at once, each refused; the most that ran at
/// once where the driver's import runs.
fn imports_at_once(memory: &Mutex<Memory>, most: &AtomicUsize) -> usize {
    let device = device();
    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|| {
                let fd = OwnedFd::from(std::fs::File::open("/dev/null").unwrap());
                let imported = import_dma_buf(
                    memory,
                    &device,
                    fd,
                    4096,
                    (1, MemoryClass::Shared),
                    DmaBufFor::Shared,
                );
                assert!(imported.is_err(), "the stand-in refuses every import");
            });
        }
    });
    most.load(Ordering::SeqCst)
}

#[test]
fn a_kind_s_first_imports_run_one_at_a_time_until_one_succeeded() {
    let most = Arc::default();
    let memory = memory(&most);
    assert_eq!(
        imports_at_once(&memory, &most),
        1,
        "imports of a kind none has succeeded as ran side by side"
    );
}

#[test]
fn a_kind_s_imports_run_side_by_side_once_one_succeeded() {
    let most = Arc::default();
    let memory = memory(&most);
    lock(&memory).dma_buf_imported(DmaBufFor::Shared, MemoryClass::Shared, 0);
    assert!(
        imports_at_once(&memory, &most) >= 2,
        "imports of a kind one has succeeded as ran one at a time"
    );
}
