//! A rebind of a no-copy buffer a committed command buffer's *tile* dispatch was encoded over,
//! whose region `VkPipeline` its variant's memo no longer holds: the rebind encodes that render
//! encoder again in its own call (`command_buffer/reencode.rs`), and what the encoding again does
//! for a pipeline it must have is what this watches. The twin of `rebind_making_wait.rs`, which
//! watches a compute encoder's dispatch.
//!
//! Metal makes a tile state's pipeline at `newRenderPipelineStateWithTileDescriptor`, so no Metal
//! call of the program's blocks on a shader compile afterwards: a tile dispatch waits for none
//! (`tile_pending.rs`), and neither does a replace of a buffer's bytes. The runtime makes a
//! region's `VkPipeline` on demand instead, in the device's pool, and keeps at most 32 of a
//! variant's (`pipeline::PIPELINES`): a rebind's encoding again of a dispatch whose region key
//! the memo let go of must answer as the first encoding did -- hand the making to the pool and
//! leave the pass to record later -- and never make the pipeline on the program's thread, nor
//! wait for a making another encoder began.
//!
//! Each leg differs only in what the encoding again finds for its region key ([`Memo`]), which
//! the legs drive by making 32 further keys of the variant, each a tile dispatch of threads per
//! tile of its own, so the key the rebind needs is the one the memo let go of. Nothing here rests
//! on how long anything took:
//!
//! - **Kept** is the control: the key is in the memo, so the encoding again finds it made. Nothing
//!   is made and nothing is pooled in the rebind's call, and a leg that read something there would
//!   mean the counters read something other than the making.
//! - **Let go** has the memo without the key and no making of it anywhere. The device's pool's
//!   test hold (`pool::held`) starts no piece, so a making handed to it stays in flight: the
//!   rebind that pools one makes nothing (`variant_pipelines_made`), and one that makes the
//!   pipeline itself on the program's thread shows as a make and no pooling. The pass the encoding
//!   again leaves is then awaited end to end: its command buffer cannot complete while the hold
//!   lives, and completes with the dispatch's records in the pages the buffer lies in once it
//!   goes.
//! - **In flight** has the memo without the key and its making in the held pool, begun by another
//!   command buffer's dispatch. A rebind that asks the memo for the pipeline waits for that making
//!   with no deadline; so the hold is let go of only after a thread sets a flag, and a rebind that
//!   returns with the flag unset waited for no making.
//!
//! The work must land where the buffer lies now in every leg: the records go into the pages of the
//! rebind, byte for byte what the same dispatch writes into a buffer of the runtime's own, and the
//! pages the dispatch was encoded over stay untouched.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects, whose recordings wait for a making until
//! it is made (`PipelineWaitPolicy::Exact`, `blit_holds::exact_waits_device`): under the default
//! bound, a pass whose making is still held 4.5 s after its commit runs nothing and its command
//! buffer completes, which a loaded host reaches while a leg holds the making. The tile function is
//! the owned fixture `validation/fixtures/public/engine/k10ts_tile_grid.ll`, as `tile_pending.rs`
//! runs it, over one 16x16 tile so that its records fit in one page ([`Page`]).

use super::blit_holds::{exact_waits_device, over_host, Host, Scratch, WAIT};
use super::layer_claim::Watched;
use super::pending_rebind::device as watched_device;
use super::rebind_behind_a_wait::{Page, BYTES};
use super::tile_pending::{tile_state, FORMATS};
use crate::pipeline::pool::held;
use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue, Device, Error,
    EventRef, LoadAction, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SharedEvent, Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::TextureUsage;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// How long the rebind is given to return while the pool's hold keeps a making waiting.
const WATCHED: Duration = Duration::from_millis(500);

/// How long the command buffer is watched not to complete while its pass's making is held.
const WATCHED_COMPLETION: Duration = Duration::from_millis(500);

/// The safety net: the pool's hold goes after this whatever the leg is doing, so a regression that
/// makes the rebind block again ends the leg on its assertions instead of wedging it while it holds
/// the Mac's GPU lock. The leg's own release comes long before.
const NET: Duration = Duration::from_secs(20);

/// The pass's tile and render area: one whole 16x16 tile, so the fixture's slot
/// `(tg.y * 4 + tg.x) * 64 + tidx` is `tidx` alone and its records end well inside a page.
const SIDE: u64 = 16;

/// The threads per tile the leg's own dispatch runs: 8x8, each thread owning the 2x2 cells at
/// twice its position.
const THREADS: u64 = 8;

/// The further region keys of the variant a leg makes to push its own out of the memo: one more
/// than the 32 `VkPipeline`s a variant keeps (`pipeline::PIPELINES`), each a dispatch of threads
/// per tile of its own, none of them 8x8.
const CROWD: u64 = 33;

/// What the encoding again finds in the variant's memo for the region key its tile dispatch needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Memo {
    /// The key, made: the encoding again finds it and asks for nothing.
    Kept,
    /// Not the key, and no making of it: the encoding again must begin one.
    LetGo,
    /// Not the key, its making in the held pool, begun by another command buffer's dispatch.
    InFlight,
}

/// The threads per tile a crowding dispatch may take, in order: every shape up to the tile,
/// without the leg's own. A shape the translator refuses for this tile function is skipped, so the
/// leg takes the first `CROWD` it accepts.
fn crowd_threads() -> impl Iterator<Item = Size> {
    (1..=SIDE)
        .flat_map(|y| (1..=SIDE).map(move |x| (x, y)))
        .filter(|&(x, y)| (x, y) != (THREADS, THREADS))
        .map(|(x, y)| Size::new(x, y, 1))
}

/// The pass's three attachments, zeroed: the fixture's implicit imageblock. What they hold is not
/// what a leg reads -- the records in the no-copy buffer are -- so they are left at zero.
fn attachments(device: &Device) -> [Texture; 3] {
    FORMATS.map(|format| {
        let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = device.new_texture(&descriptor).unwrap();
        let bytes = vec![0u8; (SIDE * SIDE * 4) as usize];
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        texture
            .replace_region(region, 0, 0, &bytes, SIDE * 4, 0)
            .unwrap();
        texture
    })
}

/// A command buffer of `queue` whose one pass over `colors`, of one 16x16 tile, dispatches `state`
/// over it with `out` as its tile buffer and `threads` threads per tile: encoded, uncommitted.
/// `gate` holds its first submission until the host signals it, so a rebind of `out` comes before
/// that submission (`rebind_behind_a_wait.rs`).
fn pass(
    queue: &CommandQueue,
    state: &RenderPipelineState,
    colors: &[Texture; 3],
    (out, threads): (&Buffer, Size),
    gate: Option<&SharedEvent>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    assert!(
        encode(&command_buffer, state, colors, (out, threads), gate),
        "the tile function takes {threads:?} threads per tile"
    );
    command_buffer
}

/// [`pass`] into `command_buffer`, `false` where the translator refuses the dispatch's shape,
/// which leaves the encoder ended and the command buffer uncommitted.
fn encode(
    command_buffer: &CommandBuffer,
    state: &RenderPipelineState,
    colors: &[Texture; 3],
    (out, threads): (&Buffer, Size),
    gate: Option<&SharedEvent>,
) -> bool {
    let mut descriptor = RenderPassDescriptor::new();
    for (attachment, target) in descriptor.color_attachments.iter_mut().zip(colors) {
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    descriptor.tile_width = SIDE;
    descriptor.tile_height = SIDE;
    if let Some(gate) = gate {
        command_buffer
            .encode_wait_for_event(EventRef::Shared(gate), 1)
            .unwrap();
    }
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_tile_buffer(Some(out), 0, 0).unwrap();
    let dispatched = encoder.dispatch_threads_per_tile(threads).is_ok();
    encoder.end_encoding().unwrap();
    dispatched
}

/// Run the pass of `threads` to completion and check it completed.
fn completed(
    queue: &CommandQueue,
    state: &RenderPipelineState,
    colors: &[Texture; 3],
    (out, threads): (&Buffer, Size),
) {
    let command_buffer = pass(queue, state, colors, (out, threads), None);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{threads:?}: {:?} {:?}",
        command_buffer.error(),
        command_buffer.failed_pipeline(),
    );
}

/// The no-copy buffer of `device` over `page`, `None` where the device imports no host memory.
///
/// # Safety
///
/// `page` outlives every buffer over it, and only that buffer's commands use it while it lives.
unsafe fn out_buffer(device: &Device, page: &Page, leg: &str) -> Option<Buffer> {
    // SAFETY: the caller's contract.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            page.memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("{leg}: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("{leg}: new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// The leg of `memo`: a tile dispatch over a no-copy buffer committed behind a shared event only
/// the host signals, so its first submission comes after the rebind; the variant's memo then
/// brought to `memo`, and the buffer detached and rebound. Asserts that the rebind made no
/// `VkPipeline` on the program's thread and waited for no making, that the pass the encoding again
/// left is the command buffer's to await, and that the dispatch's records landed in the pages the
/// buffer lies in.
fn leg(memo: Memo) {
    // Declared before the device, so dropped after it.
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new(match memo {
        Memo::Kept => "tile-rebind-memo-kept",
        Memo::LetGo => "tile-rebind-memo-let-go",
        Memo::InFlight => "tile-rebind-memo-in-flight",
    });
    let device: Device = exact_waits_device(&scratch.0);
    let state = tile_state(&device);
    let colors = attachments(&device);
    let crowd_colors = attachments(&device);
    let queue = device.new_command_queue().unwrap();
    // The crowding passes run on a queue of their own: the leg's own command buffer waits for an
    // event only the host signals, and a queue runs its command buffers in order.
    let crowd_queue = device.new_command_queue().unwrap();
    let threads = Size::new(THREADS, THREADS, 1);
    let own = device
        .new_buffer(BYTES as u64, ResourceOptions::shared())
        .unwrap();

    // What the same dispatch writes into a buffer of the runtime's own: what the leg's rebound
    // pages must hold, byte for byte. It also makes the leg's region key, so the dispatch below
    // finds it and the pass records at the encoder's end, leaving no segment -- which is what lets
    // the rebind encode the encoder again at all.
    completed(&queue, &state, &colors, (&own, threads));
    let reference: Vec<u32> = {
        let contents = own.contents().unwrap();
        // SAFETY: `BYTES` bytes of shared storage, which the runtime maps for the program and no
        // command writes any more (the command buffer above completed).
        let words =
            unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), BYTES / 4) };
        words.to_vec()
    };
    assert!(
        reference.iter().any(|&word| word != 0),
        "the reference run recorded something"
    );

    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let output = unsafe { out_buffer(&device, &pages[0], "TILE-REBIND-MEMO") };
    let Some(output) = output else { return };

    let before = device.work_counts();
    let gate = device.new_shared_event().unwrap();
    let command_buffer = pass(&queue, &state, &colors, (&output, threads), Some(&gate));
    command_buffer.commit().unwrap();
    let counts = device.work_counts();
    assert_eq!(
        counts.tile_pipelines_pooled, before.tile_pipelines_pooled,
        "the dispatch found its region pipeline made, so its pass records at the encoder's end"
    );

    // The memo brought to what the encoding again must meet: 33 further region keys of the
    // variant, one more than the 32 a variant keeps, so the leg's own key is the one let go of.
    if memo != Memo::Kept {
        let made = device.work_counts().variant_pipelines_made;
        // Committed together and awaited once: each pass's making goes to the device's pool, and
        // the last one completing has them all recorded, so every key is in the memo by then.
        let mut crowding: Vec<CommandBuffer> = Vec::with_capacity(CROWD as usize);
        for threads in crowd_threads() {
            if crowding.len() as u64 == CROWD {
                break;
            }
            let command_buffer = crowd_queue.command_buffer().unwrap();
            if !encode(
                &command_buffer,
                &state,
                &crowd_colors,
                (&own, threads),
                None,
            ) {
                continue;
            }
            command_buffer.commit().unwrap();
            crowding.push(command_buffer);
        }
        assert_eq!(
            crowding.len() as u64,
            CROWD,
            "the tile function takes {CROWD} further shapes of its own"
        );
        for (n, command_buffer) in crowding.iter().enumerate() {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "the crowding pass {n}: {:?} {:?}",
                command_buffer.error(),
                command_buffer.failed_pipeline(),
            );
        }
        let crowded = device.work_counts().variant_pipelines_made - made;
        assert_eq!(
            crowded, CROWD,
            "each crowding dispatch is a region key of its own, so the memo let the leg's go"
        );
    }

    // From now on the pool starts no piece: a making handed to it stays in flight. The one hold is
    // a thread's, which drops it when the leg says so or after `NET`, whichever comes first: a leg
    // that blocks in the rebind cannot keep it and wedge the Mac's GPU lock.
    let hold = held::hold();
    let (release, released_at) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        let _ = released_at.recv_timeout(NET);
        drop(hold);
    });

    // Another command buffer's dispatch of the same region, whose making the held pool keeps in
    // flight: what a rebind that asks the memo for the pipeline would wait for.
    let in_flight = (memo == Memo::InFlight).then(|| {
        let pooled = device.work_counts().tile_pipelines_pooled;
        let other = pass(&queue, &state, &crowd_colors, (&own, threads), Some(&gate));
        other.commit().unwrap();
        assert_eq!(
            device.work_counts().tile_pipelines_pooled,
            pooled + 1,
            "the other dispatch handed the leg's region key to the held pool"
        );
        other
    });

    // The hold is let go of only after the flag is set, on a thread of its own: a rebind that
    // returns with the flag unset waited for no making. The flag is read before the thread is
    // joined, so nothing of the release is in it.
    let released = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&released);
    let releaser = (memo == Memo::InFlight).then(|| {
        std::thread::spawn(move || {
            std::thread::sleep(WATCHED);
            flag.store(true, Ordering::SeqCst);
        })
    });
    assert!(WATCHED < NET, "the flag is set while the hold still lives");

    let before = device.work_counts();
    assert_eq!(
        output.detach_backing(),
        Ok(true),
        "{memo:?}: the detach of a buffer whose only use waits for an event"
    );
    // SAFETY: as above.
    unsafe { output.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    let waited = released.load(Ordering::SeqCst);
    let counts = device.work_counts();
    if let Some(releaser) = releaser {
        releaser.join().unwrap();
    }
    let (made, pooled) = (
        counts.variant_pipelines_made - before.variant_pipelines_made,
        counts.tile_pipelines_pooled - before.tile_pipelines_pooled,
    );
    println!(
        "tile-rebind-memo {memo:?}: made {made} pooled {pooled} waited {waited} \
         reencoded {} refused {}",
        counts.encoders_reencoded, counts.reencodes_refused,
    );

    assert_eq!(
        made, 0,
        "{memo:?}: the rebind made a `VkPipeline` on the program's thread: a Metal shader \
         compile inside `rebind_bytes_no_copy`"
    );
    assert!(
        !waited,
        "{memo:?}: `rebind_bytes_no_copy` returned only once the flag was set, so it waited for \
         the making in the held pool"
    );
    assert_eq!(
        (counts.encoders_reencoded, counts.reencodes_refused),
        (1, 0),
        "{memo:?}: the rebind encoded the tile dispatch's encoder again, and the splice took it"
    );
    match memo {
        Memo::Kept => assert_eq!(pooled, 0, "Kept: the encoding again found the key made"),
        Memo::LetGo => assert_eq!(
            pooled, 1,
            "LetGo: the encoding again pooled the key's making"
        ),
        // The other command buffer's making is the one in flight; the encoding again took it as
        // it stood and handed nothing over.
        Memo::InFlight => assert_eq!(pooled, 0, "InFlight: it took the making in flight"),
    }

    // Nothing of the GPU's holds the command buffer now. A leg whose pass the encoding again left
    // for a making cannot complete while the pool holds that making: the end-to-end witness that
    // the pass survives `State::splice_replay`, which would otherwise show as `EncodedUseRebound`.
    gate.set_signaled_value(1);
    if memo != Memo::Kept {
        assert!(
            command_buffer
                .wait_until_completed_for(WATCHED_COMPLETION)
                .is_err(),
            "{memo:?}: the command buffer completed though the making its pass records is held"
        );
        assert_ne!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{memo:?}: the pass the encoding again left awaits its making"
        );
        assert!(
            !pages[0].written() && !pages[1].written(),
            "{memo:?}: nothing ran while the making was held"
        );
    }

    drop(release);
    watchdog.join().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    if let Some(other) = in_flight {
        other.wait_until_completed_for(WAIT).unwrap();
    }
    let (status, error) = (command_buffer.status(), command_buffer.error());
    let (old, new) = (pages[0].written(), pages[1].written());
    println!("tile-rebind-memo {memo:?}: {status:?} {error:?} old {old} new {new}");

    assert!(
        !old,
        "{memo:?}: the dispatch recorded into the pages it was encoded over, which the buffer \
         left before the submission ({status:?}, {error:?})"
    );
    assert_eq!(
        error,
        None,
        "{memo:?}: the pass recorded and ran over the new pages ({:?})",
        command_buffer.failed_pipeline()
    );
    assert_eq!(status, CommandBufferStatus::Completed, "{memo:?}");
    assert!(
        new,
        "{memo:?}: the command buffer completed with no error, so the dispatch ran over the pages \
         the buffer lies in"
    );
    assert_eq!(
        pages[1].words(),
        &reference[..],
        "{memo:?}: the pages the buffer lies in hold what the same dispatch writes into a buffer \
         of the runtime's own"
    );
}

/// The control: a rebind whose encoding again finds the region key in the variant's memo makes
/// nothing and pools nothing, so a make or a wait read in the other legs reads the making and not
/// the rebind itself.
#[test]
fn a_rebind_whose_encoding_again_finds_its_tile_region_made_asks_for_nothing() {
    leg(Memo::Kept);
}

/// A rebind whose encoding again needs a region `VkPipeline` the memo let go of hands the making
/// to the device's pool and leaves the pass to record later, as the first encoding would, instead
/// of running the Metal compile on the program's thread.
#[test]
fn a_rebind_does_not_make_a_tile_region_pipeline_on_the_programs_thread() {
    leg(Memo::LetGo);
}

/// A rebind whose encoding again needs a region `VkPipeline` whose making another encoder began
/// takes that making as it stands and returns, instead of waiting for it with no deadline.
#[test]
fn a_rebind_does_not_wait_for_a_tile_making_another_encoder_began() {
    leg(Memo::InFlight);
}

/// The guard on the other side of the rebind rule, with the validation layer watching.
///
/// A pass left for the pipelines its tile dispatch records holds no buffer's pages past its
/// command buffer's commit (`CommandBuffer::commit`'s `steps_hold`), so a rebind of one is not
/// refused -- Metal's replace never fails for that -- and encodes its encoder again instead. Where
/// that encoding again cannot happen, the pass must record nothing: its instances' secondaries and
/// its tile dispatch's recording resolved the buffer where it lay as they were encoded, and the
/// pages a rebind freed stay mapped, so a recording that bound that `VkBuffer` would have the GPU
/// write into pages the program handed back, with no fault and nothing said. That is the class of
/// bug this guard exists for, so it is watched by the layer and not by a page's contents alone.
///
/// The encoding again is made impossible as `pending_rebind.rs` makes it: the render encoder is
/// opened while the device has no buffer over host pages, so it logs no call (`Unlogged::NoCalls`)
/// and no rebind can replay it. The device's pool is held, so only a host step records the pass,
/// and only once the hold is gone -- after the rebind, in every run.
///
/// Without `DeferredPass::record_forms`' `encoder_rebound` guard the host step records the pass and
/// the layer reports the freed `VkBuffer` in a bound descriptor
/// (`VUID-vkCmdDrawIndexed-None-08114` and its kin), which fails this test.
#[test]
fn a_tile_pass_records_nothing_where_a_rebind_freed_what_its_descriptors_name() {
    // Declared before the device, so dropped after it.
    let hosts = [Host::new(), Host::new()];
    let watched = Watched::default();
    let scratch = Scratch::new("tile-rebind-unlogged");
    let device = watched_device(&watched, &scratch.0);
    let state = tile_state(&device);
    let colors = attachments(&device);
    let queue = device.new_command_queue().unwrap();

    // Every making the pool starts waits from here: only a host step records the pass, and only
    // once this hold is gone.
    let hold = held::hold();
    let mut descriptor = RenderPassDescriptor::new();
    for (attachment, target) in descriptor.color_attachments.iter_mut().zip(&colors) {
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    descriptor.tile_width = SIDE;
    descriptor.tile_height = SIDE;
    let command_buffer = queue.command_buffer().unwrap();
    // Opened while the device has no buffer over host pages, so the encoder logs no call and no
    // rebind encodes it again (`Unlogged::NoCalls`).
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(output) = over_host(&device, &hosts[0], &runs) else {
        return;
    };
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder.set_tile_buffer(Some(&output), 0, 0).unwrap();
    let pooled = device.work_counts().tile_pipelines_pooled;
    encoder
        .dispatch_threads_per_tile(Size::new(THREADS, THREADS, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    assert_eq!(
        device.work_counts().tile_pipelines_pooled,
        pooled + 1,
        "the dispatch handed a making to the pool, so a host step records its pass"
    );
    command_buffer.commit().unwrap();

    // The buffer lies elsewhere from here, and the pages it lay in go: nothing submitted holds
    // them, and the committed command buffer holds them weakly. The rebind is allowed at all only
    // because the pass holds nothing past the commit.
    //
    // SAFETY: the pages outlive every buffer over them, and only its commands use them while it
    // lives.
    unsafe { output.rebind_bytes_no_copy(hosts[1].memory(), None, None) }.unwrap();
    assert_eq!(
        device.work_counts().encoders_reencoded,
        0,
        "the encoder kept no log, so nothing was encoded again"
    );

    // The makings may finish now, so the host step records the pass after the rebind.
    drop(hold);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::EncodedUseRebound),
        "the submission holding the pass ran nothing"
    );
    assert!(hosts[0].holds(0), "the pages the dispatch took");
    assert!(hosts[1].holds(0), "the pages the buffer lies in now");
    watched.layer_reported_nothing(
        "a tile pass recorded after a rebind of a buffer it named bound no freed VkBuffer",
    );
}
