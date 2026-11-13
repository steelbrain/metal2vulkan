//! A completed command buffer's release of host imports it holds last (`free_thread.rs`, **What
//! is posted**): where a deallocator waits for the free, the free goes to the device's free
//! thread, and the deallocator runs there after it; where nothing waits, the import is freed
//! before the command buffer's waiters return; and where the program's last drop of the device ran
//! on a thread of the runtime's while such a free waited, the device's teardown runs on the free
//! thread as the import goes, and ends.

use super::blit_holds::{busy, device, until, Host, Scratch, BYTES, FILLED};
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Deallocator, Device, Error,
    HostBacking, ResourceOptions,
};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The name of the thread a deallocator ran on, once it ran.
type RanOn = Arc<Mutex<Option<Option<String>>>>;

/// A no-copy buffer over `host`, with a deallocator that notes its thread in `ran_on` where
/// given; `None`, printed, where the device imports no host memory.
fn no_copy(device: &Device, host: &Host, ran_on: Option<&RanOn>) -> Option<Buffer> {
    let deallocator = ran_on.map(|ran_on| {
        let ran_on = Arc::clone(ran_on);
        Box::new(move |_: NonNull<u8>, _: u64| {
            let name = std::thread::current().name().map(str::to_owned);
            *ran_on.lock().unwrap() = Some(name);
        }) as Deallocator
    });
    // SAFETY: the memory outlives the buffer (dropped after the device), and only the buffer's
    // commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            host.memory(),
            BYTES as u64,
            ResourceOptions::default(),
            deallocator,
        )
    };
    match made {
        Ok(buffer) if buffer.host_backing() == Some(HostBacking::ImportedByPointer) => Some(buffer),
        Ok(buffer) => {
            eprintln!(
                "COMPLETION-IMPORT-FREES: the pages are not imported by their pointer ({:?})",
                buffer.host_backing()
            );
            None
        }
        Err(Error::Unsupported(gap)) => {
            eprintln!("COMPLETION-IMPORT-FREES: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A command buffer of `queue`'s that copies a word of `source` into `target`, committed behind
/// fills that keep the queue busy, and scheduled: the busy one, then it, and whether it was still
/// pending as it was scheduled.
fn copy_behind_fills(
    queue: &CommandQueue,
    source: &Buffer,
    target: &Buffer,
) -> (CommandBuffer, CommandBuffer, bool) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(source, 0, target, 0, 4)
        .unwrap();
    blit.end_encoding().unwrap();
    let busy = busy(queue, target);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_scheduled().unwrap();
    let pending = command_buffer.status() != CommandBufferStatus::Completed;
    (busy, command_buffer, pending)
}

/// A no-copy buffer released while the command buffer that copies from it is in flight, with a
/// deallocator: that command buffer's completion lets go of the import last, and its free goes to
/// the free thread, where the deallocator runs after it.
#[test]
fn a_completed_release_frees_an_import_a_deallocator_waits_for_on_the_free_thread() {
    let scratch = Scratch::new("completion-import-frees-posted");
    let host = Host::new();
    let device = device(&scratch.0);
    // The submission checks' own imports and frees, out of the way first.
    let _settled = device.settle_submit_checks(WAIT);
    let queue = device.new_command_queue().unwrap();
    let ran_on = RanOn::default();
    let Some(buffer) = no_copy(&device, &host, Some(&ran_on)) else {
        return;
    };
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let before = device.work_counts();
    let (busy, command_buffer, pending) = copy_behind_fills(&queue, &buffer, &target);
    drop(buffer);
    let still = command_buffer.status() != CommandBufferStatus::Completed;
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    busy.wait_until_completed_for(WAIT).unwrap();
    until("the deallocator never ran", || {
        ran_on.lock().unwrap().is_some()
    });
    let posted = device.work_counts().host_import_frees_posted - before.host_import_frees_posted;
    let thread = ran_on.lock().unwrap().clone().flatten();
    eprintln!(
        "COMPLETION-IMPORT-FREES {}: pending at the release {pending}, still after it {still}; \
         posted {posted}, the deallocator ran on {thread:?}",
        device.name()
    );
    if !(pending && still) {
        eprintln!("COMPLETION-IMPORT-FREES: the copy completed before the release: nothing held");
        return;
    }
    assert_eq!(posted, 1, "the completion's free was not posted");
    assert_eq!(thread.as_deref(), Some("metal2vulkan-frees"));
}

/// The same release with no deallocator: nothing waits for the free, and the caller may unmap the
/// pages once the command buffer's waiters returned, so the completion frees the import before
/// they return, posting nothing.
#[test]
fn a_completed_release_frees_an_import_nothing_waits_for_before_its_waiters_return() {
    let scratch = Scratch::new("completion-import-frees-in-place");
    let host = Host::new();
    let device = device(&scratch.0);
    // The submission checks' own imports and frees, out of the way first.
    let _settled = device.settle_submit_checks(WAIT);
    let queue = device.new_command_queue().unwrap();
    let live = device.work_counts().host_imports_live;
    let Some(buffer) = no_copy(&device, &host, None) else {
        return;
    };
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let before = device.work_counts();
    let (busy, command_buffer, pending) = copy_behind_fills(&queue, &buffer, &target);
    drop(buffer);
    let still = command_buffer.status() != CommandBufferStatus::Completed;
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    busy.wait_until_completed_for(WAIT).unwrap();
    eprintln!(
        "COMPLETION-IMPORT-FREES {} in place: pending at the release {pending}, still after it \
         {still}; imports live {live} before, {} as the wait returned",
        device.name(),
        after.host_imports_live
    );
    assert_eq!(
        after.host_import_frees_posted - before.host_import_frees_posted,
        0,
        "a free nothing waits for was posted"
    );
    assert_eq!(
        after.host_imports_live, live,
        "the import outlived the command buffer's wait"
    );
}

/// The program lets go of the buffer, the command buffers, the queue and the device while the
/// copy is in flight: the completion posts the import's free, and the program's last handle goes
/// on the runtime's thread after it, which waits for nothing. With the free thread held until
/// then, the import's objects hold the device's last share, so the teardown runs on the free
/// thread as they go: it frees what it queued in place of waiting for itself, and ends the thread
/// unjoined.
#[test]
fn a_teardown_the_free_thread_meets_as_a_posted_import_goes_ends_it() {
    let scratch = Scratch::new("completion-import-frees-teardown");
    let host = Host::new();
    let device = device(&scratch.0);
    // The submission checks' own imports and frees, out of the way first.
    let _settled = device.settle_submit_checks(WAIT);
    let queue = device.new_command_queue().unwrap();
    let ran_on = RanOn::default();
    let Some(buffer) = no_copy(&device, &host, Some(&ran_on)) else {
        return;
    };
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let frees = Arc::clone(&device.shared().frees);
    let holds = Arc::clone(&device.shared().holds);
    frees.held().hold();
    let (busy, command_buffer, pending) = copy_behind_fills(&queue, &buffer, &target);
    drop(buffer);
    let still = command_buffer.status() != CommandBufferStatus::Completed;
    drop((busy, command_buffer, queue, target, device));
    if !(pending && still) {
        frees.held().release();
        eprintln!("COMPLETION-IMPORT-FREES: the copy completed before the release: nothing held");
        return;
    }
    // The completion posted the free, which waits at the hold; the program's last handle went
    // after it.
    frees.held().until_waiting(1, WAIT);
    until("the program's last handle never went", || holds.stopping());
    frees.held().release();
    until("the deallocator never ran", || {
        ran_on.lock().unwrap().is_some()
    });
    until("the free thread never ended", || {
        frees.pending() == (0, false)
    });
    eprintln!(
        "COMPLETION-IMPORT-FREES teardown: the deallocator ran on {:?}, the teardown on the free \
         thread {}",
        ran_on.lock().unwrap().clone().flatten(),
        frees.stopped_here()
    );
    assert_eq!(
        ran_on.lock().unwrap().clone().flatten().as_deref(),
        Some("metal2vulkan-frees")
    );
    assert!(
        frees.stopped_here(),
        "the device's last share went elsewhere than the free thread"
    );
}
