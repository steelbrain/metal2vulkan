//! A rebind of a no-copy buffer a committed command buffer's dispatch was encoded over, whose
//! `VkPipeline` is still being made: the rebind encodes that encoder again in its own call
//! (`command_buffer/reencode.rs`), and what the encoding again does with a making in flight is
//! what this watches.
//!
//! Metal makes a compute state's pipeline at `newComputePipelineState`, so no Metal call of the
//! program's blocks on a shader compile afterwards: a dispatch waits for none
//! (`dispatch_pending.rs`), and neither does a replace of a buffer's bytes. The runtime makes a
//! region's pipeline on demand instead, in the device's pool, and an encoder's own dispatch
//! encoded while one is being made leaves a pending segment and returns at once
//! (`pending_dispatch.rs`). A rebind's encoding again of that encoder must answer as the first
//! encoding did and return promptly: the making is the submission's to wait for, which is where
//! the first encoding left it, not the program's thread's inside `rebind_bytes_no_copy`.
//!
//! The wait is read causally, not by a clock on a loaded host. The device's pool's test hold
//! (`pool::held`) keeps every making waiting, and a thread drops the hold only after setting a
//! flag; a rebind that returns with the flag unset waited for no making, and one that returns
//! with it set could only have returned because the making it met was let go of. No leg of this
//! rests on how long anything took.
//!
//! The two legs differ only in whether the region pipeline is made before the dispatch is
//! encoded ([`Warm`]). The warm leg is the control: its encoding again finds the pipeline made,
//! so nothing of the pool's hold can hold it, and a flag set there would mean the flag reads
//! something other than the wait.
//!
//! The work must still land where the buffer lies now in both: the dispatch writes the pages of
//! the rebind and leaves those it was encoded over untouched, as `rebind_behind_a_wait.rs` has it
//! over its four keep-alive configurations.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects, whose recordings wait for a making until
//! it is made (`PipelineWaitPolicy::Exact`, `blit_holds::exact_waits_device`): under the default
//! bound, a pass whose making is still held 4.5 s after its commit runs nothing and its command
//! buffer completes, which a loaded host reaches while a leg holds the making. The kernel is
//! `scale_and_offset` of the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`output[i] = input[i] * scale + i`),
//! as `rebind_behind_a_wait.rs` runs it.

use super::blit_holds::{exact_waits_device, Scratch, WAIT};
use super::dispatch_pending::state;
use super::rebind_behind_a_wait::{dispatch, Page, BYTES, COUNT, SCALE};
use crate::pipeline::pool::held;
use crate::{CommandBufferStatus, Device, Error, EventRef, ResourceOptions};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// How long the rebind is given to return while the pool's hold keeps a making waiting.
const WATCHED: Duration = Duration::from_millis(500);

/// Whether the region pipeline is made before the dispatch that the rebind encodes again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Warm {
    /// Made: the encoding again finds it, and waits for nothing.
    Made,
    /// Unmade, its making waiting in the held pool: the encoding again meets it in flight.
    Making,
}

/// The leg of `warm`: a dispatch over a no-copy buffer committed behind a shared event only the
/// host signals, so its first submission comes after the rebind
/// (`rebind_behind_a_wait.rs`); the buffer then detached and rebound. Asserts that
/// `rebind_bytes_no_copy` returned without the pool letting any making go, and that the dispatch
/// wrote the pages the buffer lies in and not those it was encoded over.
fn leg(warm: Warm) {
    // Declared before the device, so dropped after it.
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new(match warm {
        Warm::Made => "rebind-making-wait-made",
        Warm::Making => "rebind-making-wait-making",
    });
    let device: Device = exact_waits_device(&scratch.0);
    let state = state(&device);
    let queue = device.new_command_queue().unwrap();
    let bytes: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let scale = device
        .new_buffer_with_bytes(&SCALE.to_le_bytes(), ResourceOptions::shared())
        .unwrap();

    // The control's region pipeline is made here, over a buffer of the runtime's own, as
    // `rebind_behind_a_wait.rs` warms its state; the other leg leaves it unmade.
    if warm == Warm::Made {
        let warming = device
            .new_buffer(COUNT * 4, ResourceOptions::shared())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        dispatch(&command_buffer, &state, &input, &warming, &scale);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    // From now on the pool starts no piece: a making the dispatch below begins stays in flight.
    let hold = held::hold();

    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let output = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("REBIND-MAKING-WAIT {warm:?}: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("{warm:?}: new_buffer_with_bytes_no_copy: {error:?}"),
    };

    let pooled = device.work_counts().dispatch_pipelines_pooled;
    let gate = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    dispatch(&command_buffer, &state, &input, &output, &scale);
    command_buffer.commit().unwrap();
    let begun = device.work_counts().dispatch_pipelines_pooled - pooled;
    match warm {
        Warm::Made => assert_eq!(begun, 0, "the dispatch found its pipeline made"),
        Warm::Making => assert_eq!(
            begun, 1,
            "the dispatch's making waits in the held pool, so it left a pending segment"
        ),
    }

    // The hold is let go of only after the flag is set, on a thread of its own: a rebind that
    // returns with the flag unset waited for no making. The flag is read before the hold's
    // thread is joined, so nothing of the release is in it.
    let released = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&released);
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(WATCHED);
        flag.store(true, Ordering::SeqCst);
        drop(hold);
    });

    assert_eq!(
        output.detach_backing(),
        Ok(true),
        "{warm:?}: the detach of a buffer whose only use waits for an event"
    );
    // SAFETY: as above.
    unsafe { output.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    let waited = released.load(Ordering::SeqCst);
    let counts = device.work_counts();
    releaser.join().unwrap();

    // The command buffer is let complete before anything is asserted: one left committed behind
    // an unsignalled event would hold the device's teardown.
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let (status, error) = (command_buffer.status(), command_buffer.error());
    let (old, new) = (pages[0].written(), pages[1].written());
    println!(
        "rebind-making-wait {warm:?}: {status:?} {error:?} old {old} new {new} waited {waited} \
         reencoded {} refused {}",
        counts.encoders_reencoded, counts.reencodes_refused,
    );

    assert!(
        !waited,
        "{warm:?}: `rebind_bytes_no_copy` returned only once the pool let the making its \
         encoding again met go: the call waited for a shader compile on the program's thread \
         ({status:?}, {error:?})"
    );
    assert_eq!(
        counts.encoders_reencoded, 1,
        "{warm:?}: the rebind encoded the dispatch's encoder again"
    );
    assert!(
        !old,
        "{warm:?}: the dispatch wrote the pages it was encoded over, which the buffer left \
         before the submission ({status:?}, {error:?})"
    );
    assert_eq!(
        error, None,
        "{warm:?}: the command buffer ran over the new pages"
    );
    assert_eq!(status, CommandBufferStatus::Completed, "{warm:?}");
    assert!(
        new,
        "{warm:?}: the command buffer completed with no error, so the dispatch ran over the \
         pages the buffer lies in"
    );
}

/// A rebind whose encoding again meets the making the dispatch began returns without waiting for
/// it, as the first encoding did, and the dispatch runs over the pages the buffer lies in.
#[test]
fn a_rebind_does_not_wait_for_the_making_its_encoding_again_meets() {
    leg(Warm::Making);
}

/// The control: a rebind whose encoding again finds the pipeline made meets no making, so the
/// pool's hold cannot hold it, and the flag the other leg reads stays unset.
#[test]
fn a_rebind_waits_for_nothing_where_its_encoding_again_finds_the_pipeline_made() {
    leg(Warm::Made);
}

/// How long the command buffer is watched not to complete while its segment's making is held.
const WATCHED_COMPLETION: Duration = Duration::from_millis(500);

/// The safety net: a second hold dropped after this, so a regression that makes the rebind block
/// again ends this test instead of wedging it while it holds the Mac's GPU lock.
const NET: Duration = Duration::from_secs(30);

/// The segment the encoding again leaves is a pending dispatch's, awaited where the first
/// encoding's would have been: its command buffer cannot complete while the making is held,
/// though the rebind returned and the event was signalled, and once the making lands the segment
/// records and the dispatch's writes are in the pages the buffer lies in.
///
/// This is the end-to-end witness that the segment survives `State::splice_replay` -- the splice
/// refuses a place whose parts or segments do not match, and a refusal would show here as
/// `EncodedUseRebound` instead. It is not the regression guard for the wait itself
/// ([`a_rebind_does_not_wait_for_the_making_its_encoding_again_meets`] is): the net below would
/// let a blocking rebind through.
#[test]
fn a_dispatch_the_encoding_again_left_as_a_segment_records_once_its_making_lands() {
    // Declared before the device, so dropped after it.
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new("rebind-making-segment");
    let device: Device = exact_waits_device(&scratch.0);
    let state = state(&device);
    let queue = device.new_command_queue().unwrap();
    let bytes: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let scale = device
        .new_buffer_with_bytes(&SCALE.to_le_bytes(), ResourceOptions::shared())
        .unwrap();

    // Two holds: the net's, dropped on its own after `NET`, and this test's, dropped where it
    // means the making to land. A piece waits while either lives.
    let net = held::hold();
    let hold = held::hold();
    let (past, wedged) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        // Ends as soon as the test says it is past the rebind (the sender dropped), else after
        // `NET`, which is the only case the net is for.
        let _ = wedged.recv_timeout(NET);
        drop(net);
    });

    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let output = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("REBIND-MAKING-SEGMENT: the device imports no host memory: {gap:?}");
            drop(past);
            drop(hold);
            watchdog.join().unwrap();
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };

    let pooled = device.work_counts().dispatch_pipelines_pooled;
    let gate = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    dispatch(&command_buffer, &state, &input, &output, &scale);
    command_buffer.commit().unwrap();
    assert_eq!(
        device.work_counts().dispatch_pipelines_pooled,
        pooled + 1,
        "the dispatch's making waits in the held pool, so it left a pending segment"
    );

    assert_eq!(
        output.detach_backing(),
        Ok(true),
        "the detach of a buffer whose only use waits for an event"
    );
    // SAFETY: as above.
    unsafe { output.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    let counts = device.work_counts();
    assert_eq!(
        (counts.encoders_reencoded, counts.reencodes_refused),
        (1, 0),
        "the rebind encoded the dispatch's encoder again, and the splice took it"
    );

    // Past the rebind: the net is not needed, and the watchdog ends.
    drop(past);

    // Nothing of the GPU's holds it now: its batch waits for the making its segment records.
    gate.set_signaled_value(1);
    assert!(
        command_buffer
            .wait_until_completed_for(WATCHED_COMPLETION)
            .is_err(),
        "the command buffer completed though the making its segment records is held"
    );
    assert_ne!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "the segment the encoding again left awaits its making, as the first encoding's would"
    );
    assert!(
        !pages[0].written() && !pages[1].written(),
        "nothing ran while the making was held"
    );

    drop(hold);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let (status, error) = (command_buffer.status(), command_buffer.error());
    let (old, new) = (pages[0].written(), pages[1].written());
    println!("rebind-making-segment: {status:?} {error:?} old {old} new {new}");
    watchdog.join().unwrap();

    assert!(
        !old,
        "the dispatch wrote the pages it was encoded over ({status:?}, {error:?})"
    );
    assert_eq!(
        error, None,
        "the segment recorded and ran over the new pages"
    );
    assert_eq!(status, CommandBufferStatus::Completed);
    assert!(
        new,
        "the segment recorded once its making landed, over the pages the buffer lies in"
    );
}
