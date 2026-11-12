//! The device's free thread (`free_thread.rs`), held at each free through the crate's hook so a
//! posted free stays pending as long as a case needs: a resource's release returns with its
//! block's free posted (`WorkCounts::memory_frees_posted`), and its pinned address and its
//! forced resource ID take a new resource at once, the old block not yet freed; an allocation at
//! `maxMemoryAllocationCount` waits for the posted frees, which still count, and is then made,
//! where it would otherwise be refused, the frees it waits for unpaced; a block whose making would
//! take its heap past the heap's size waits for the frees of that heap first; a burst of posted
//! frees runs paced, each after the frees before it (`step_aside::pace`,
//! `WorkCounts::memory_frees_paced`); and the device's teardown returns only once every posted
//! free ran and the thread ended (each `memory_free_waits` counting the wait).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::free_thread::FreeThread;
use crate::{
    AddressPlacement, BufferContents, BufferDescriptor, Device, DeviceKind, DeviceOptions,
    GpuAddress, IdPlacement, ResourceOptions, TextureDescriptor,
};
use metal2vulkan::metal::PixelFormat;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a free may stay held, or a waiting allocation take to count its wait.
const WAIT: Duration = Duration::from_secs(60);

/// An address high in the Metal address space, far from the ranges the runtime picks first.
const HIGH: u64 = 0x7100_0000_0000;

/// A buffer's bytes past half a block and past the 64 MiB of buffers the recycler parks
/// (`recycle::BUFFERS`): a dedicated block of its own, freed as the buffer goes on the devices
/// these tests make, which keep no empty block ([`device`]).
fn dedicated() -> u64 {
    crate::memory::BLOCK_BYTES / 2 + (64 << 20)
}

/// A device of the cfg's kind, as the product makes it, but keeping no empty block for reuse past
/// its pools' two (the heaps' budget's test stand-in at 0, `memory.rs`'s `KEPT_EMPTY_BYTES`): a
/// block that empties is freed, posted to the free thread these tests hold.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    crate::device::lock(&device.shared().memory).kept_budget_stand_in = Some(0);
    device
}

/// The free thread held at each free until this drops, a failing case's unwinding too, so the
/// device's teardown, which waits for the frees, never waits for a hold left behind.
struct Hold(Arc<FreeThread>);

impl Hold {
    fn new(frees: &Arc<FreeThread>) -> Hold {
        frees.held().hold();
        Hold(Arc::clone(frees))
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.0.held().release();
    }
}

/// Wait until `done` holds, failing past [`WAIT`].
fn until(what: &str, done: impl Fn() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < WAIT, "{what} never happened");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_released_block_s_address_and_resource_id_take_a_new_resource_while_its_free_waits() {
    let device = device();
    let frees = Arc::clone(&device.shared().frees);
    let pinned = |address: u64| {
        device.new_buffer_with_descriptor(
            BufferDescriptor::new(
                dedicated(),
                ResourceOptions::private(),
                BufferContents::Zero,
            )
            .with_placement(AddressPlacement::At(GpuAddress(address))),
        )
    };
    // A texture of a block of its own, past the bytes the recycler parks: 256 MiB.
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8192, 8192, false);
    descriptor.resource_options = ResourceOptions::private();
    let buffer = pinned(HIGH).unwrap();
    let texture = device.new_texture(&descriptor).unwrap();
    let id = texture.gpu_resource_id();
    let before = device.work_counts();
    let hold = Hold::new(&frees);
    drop(buffer);
    drop(texture);
    frees.held().until_waiting(1, WAIT);
    let released = device.work_counts();
    assert_eq!(
        released.memory_frees_posted - before.memory_frees_posted,
        2,
        "both blocks' frees were posted"
    );
    assert_eq!(frees.pending().0, 2, "neither block is freed yet");
    // The address and the ID are free as the releases return, the blocks not yet freed.
    let again = pinned(HIGH).expect("the pinned address takes a new buffer at once");
    assert_eq!(again.gpu_address(), GpuAddress(HIGH));
    descriptor.placement = IdPlacement::At(id);
    let same =
        (device.new_texture(&descriptor)).expect("the forced ID takes a new texture at once");
    assert_eq!(same.gpu_resource_id(), id);
    assert_eq!(frees.pending().0, 2, "the old blocks wait still");
    drop(hold);
    until("the posted frees", || frees.pending().0 == 0);
    eprintln!(
        "POSTED-FREES: a pinned address and a forced resource ID took new resources with their \
         old blocks' two frees pending on the free thread"
    );
}

#[test]
fn an_allocation_at_the_limit_waits_for_the_posted_frees_and_is_made() {
    let device = device();
    let frees = Arc::clone(&device.shared().frees);
    let allocations = Arc::clone(&device.shared().allocations.allocations);
    let buffers: Vec<_> = (0..3)
        .map(|_| (device.new_buffer(dedicated(), ResourceOptions::private())).unwrap())
        .collect();
    let before = device.work_counts();
    // Each free owes its next pace the longest a pace waits ([`FreeThread::frees_take`]), so
    // that the frees' running unpaced below is the waiting allocation's doing.
    frees.frees_take(Some(Duration::MAX));
    let hold = Hold::new(&frees);
    drop(buffers);
    frees.held().until_waiting(1, WAIT);
    let posted = device.work_counts();
    assert_eq!(posted.memory_frees_posted - before.memory_frees_posted, 3);
    // The posted frees still count: the limit at the live allocations leaves no room.
    let limit = allocations.limit();
    allocations.limit_to(allocations.live());
    let maker = {
        let device = device.clone();
        std::thread::spawn(move || device.new_buffer(dedicated(), ResourceOptions::private()))
    };
    let waited = || device.work_counts().memory_free_waits > posted.memory_free_waits;
    until("the allocation's wait or its end", || {
        waited() || maker.is_finished()
    });
    assert!(waited(), "the allocation waited for the posted frees");
    assert!(
        !maker.is_finished(),
        "the allocation waits while the frees are held"
    );
    drop(hold);
    let made = maker.join().unwrap();
    allocations.limit_to(limit);
    let after = device.work_counts();
    assert!(made.is_ok(), "made once the frees ran: {made:?}");
    assert_eq!(
        after.memory_allocations_refused_at_limit, before.memory_allocations_refused_at_limit,
        "nothing refused"
    );
    assert_eq!(
        after.memory_frees_paced, before.memory_frees_paced,
        "no free paced while the allocation waited for them"
    );
    eprintln!(
        "POSTED-FREES: an allocation at the limit waited for {} posted frees and was made",
        posted.memory_frees_posted - before.memory_frees_posted
    );
}

/// Posted in a burst, the frees run one after another, each but the first after a wait as long
/// as the free before it took, less the time since it ended, at most `step_aside::MOST_ASIDE`, so
/// that the device's other threads' submissions go between them (`step_aside::pace`). A device's
/// free may end sooner than the thread reaches its next pace, which then owes nothing, so each
/// free here counts as taking longer than any pace waits ([`FreeThread::frees_take`]): each of
/// the two behind another waits the longest a pace waits.
#[test]
fn a_burst_of_posted_frees_runs_paced() {
    let device = device();
    let frees = Arc::clone(&device.shared().frees);
    let buffers: Vec<_> = (0..3)
        .map(|_| (device.new_buffer(dedicated(), ResourceOptions::private())).unwrap())
        .collect();
    frees.frees_take(Some(Duration::MAX));
    let hold = Hold::new(&frees);
    drop(buffers);
    frees.held().until_waiting(1, WAIT);
    assert_eq!(frees.pending().0, 3, "the three frees wait");
    // Counted from here: the first free's pace, owed what the thread freed before the burst,
    // went before its hold.
    let before = device.work_counts();
    drop(hold);
    until("the posted frees", || frees.pending().0 == 0);
    let after = device.work_counts();
    let paced = after.memory_frees_paced - before.memory_frees_paced;
    let nanos = after.memory_frees_paced_nanos - before.memory_frees_paced_nanos;
    assert_eq!(
        paced, 2,
        "{paced} of the two frees behind another waited first"
    );
    let most = u64::try_from(crate::step_aside::MOST_ASIDE.as_nanos()).unwrap();
    assert_eq!(nanos, 2 * most, "each waited the longest a pace waits");
    eprintln!(
        "POSTED-FREES: {paced} of a burst's two frees behind another waited first, {:.3} ms in all",
        nanos as f64 / 1e6
    );
}

#[test]
fn a_block_past_its_heap_s_size_waits_for_the_posted_frees_of_the_heap() {
    let device = device();
    let frees = Arc::clone(&device.shared().frees);
    let buffer = (device.new_buffer(dedicated(), ResourceOptions::private())).unwrap();
    let before = device.work_counts();
    let hold = Hold::new(&frees);
    drop(buffer);
    frees.held().until_waiting(1, WAIT);
    // Each heap is full with its live blocks and the frees posted: a new block passes its size.
    let sizes = crate::device::lock(&device.shared().memory).hold_heaps_to_live();
    let maker = {
        let device = device.clone();
        std::thread::spawn(move || device.new_buffer(dedicated(), ResourceOptions::private()))
    };
    let waited = || device.work_counts().memory_free_waits > before.memory_free_waits;
    until("the block's wait or its end", || {
        waited() || maker.is_finished()
    });
    assert!(waited(), "the block waited for the posted frees");
    assert!(
        !maker.is_finished(),
        "the block waits while the frees are held"
    );
    drop(hold);
    let made = maker.join().unwrap();
    crate::device::lock(&device.shared().memory).put_heaps_back(&sizes);
    assert!(made.is_ok(), "made once the frees ran: {made:?}");
    assert_eq!(
        device.work_counts().memory_allocations_refused_at_limit,
        before.memory_allocations_refused_at_limit
    );
}

#[test]
fn the_teardown_runs_every_posted_free_and_ends_the_thread_before_it_returns() {
    let device = device();
    let frees = Arc::clone(&device.shared().frees);
    let buffers: Vec<_> = (0..2)
        .map(|_| (device.new_buffer(dedicated(), ResourceOptions::private())).unwrap())
        .collect();
    frees.held().hold();
    drop(buffers);
    frees.held().until_waiting(1, WAIT);
    assert_eq!(frees.pending(), (2, true));
    // Let the frees go a while after the teardown began, from another thread.
    let release = {
        let frees = Arc::clone(&frees);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            frees.held().release();
        })
    };
    let start = Instant::now();
    drop(device);
    let took = start.elapsed();
    assert_eq!(
        frees.pending(),
        (0, false),
        "the teardown returned with every posted free run and the thread joined"
    );
    release.join().unwrap();
    eprintln!("POSTED-FREES: the teardown waited {took:?} for the two held frees");
}
