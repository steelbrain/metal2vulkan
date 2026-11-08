//! Where the device's last handle goes on one of the runtime's own threads (the consumer's review
//! of its first commit, item c). A program that learns of its work's end otherwise than by waiting
//! for it (`command_buffer/gate.rs`) may drop its last handles while the runtime still holds the
//! device; the teardown then runs on the thread that lets go last. Each case arranges that with
//! gates the test opens, never a timing: the program drops every handle while the runtime holds the
//! device through one path, the device is checked to be still alive, the gate opens, and the
//! teardown must end, on the thread the path lets go on, with no message (the validation layer,
//! which the device asks for, reports a destroyed object used, an object destroyed twice, or one
//! the teardown left; what the claim of its silence holds is what the instance enabled of it,
//! `tests/layer_claim.rs`).
//!
//! - **Not a pool thread**: a draw's making held waiting in the pool (`pool::held`) holds nothing
//!   of the device, whose last handle goes on the program's thread; one held while it runs
//!   (`pool::held::mid_making_hold`) leaves its state's release at once, and the device to its
//!   reaper's thread. The command buffer is dropped uncommitted.
//! - **Not a pool thread after a wait**: a dispatch's piece held after it settled its making and
//!   before it gave back its variant (`pool::held::settled_hold`) lets the command buffer complete
//!   and its completed handler run, but not its wait return, until the piece let go (its trail,
//!   `trails.rs`): the program's drops after the wait tear the device down on its own thread,
//!   before the last drop returns. Nor a render state's creation return, whose predicted
//!   pipeline's piece is held so: the program's drops of the state and the device right after it
//!   tear the device down on its own thread too.
//! - **The scheduler**: a committed command buffer held at its submission (its submission check
//!   blocked on the scheduler's thread). The teardown ends on the scheduler, whose submitted
//!   batch's hold goes after its lock, or on the queue's handler thread, which releases what the
//!   command buffer retained; never on a queue's waiter, whose completion turn drops its own
//!   handle before it posts a finished command buffer and hands its hold to the posted transition,
//!   and, where it runs a command buffer's transition itself, lets the transition's hold go short
//!   of the command buffer's last reference (`Hold::let_go_short_of_last`): the last then goes on
//!   a handler thread.
//! - **A handler thread, beside a host step's own module**: a module made as a host step makes its
//!   variant's (`exit_guard::host_step`: one of its own, which the device's map keeps only for a
//!   sweep or its teardown), held by a completed handler blocked on its gate; once a command
//!   buffer of another queue committed after it finished, no thread but the handler's holds the
//!   device. The handler lets the module go, then the command buffer's release drops the last
//!   device handle on that thread, whose teardown destroys the module.
//! - **Not a handler thread after the program's use of a queue**: on a device that parks freed
//!   memory (`Device::park_freed_memory`), a use of a queue through `with_vk_queue` leaves its
//!   mark pending and posts the watch of the marks, held as it begins (`freed_memory::held`). The
//!   watch holds nothing of the device, so the program's drops tear the device down on its own
//!   thread before the last drop returns; the teardown waits for the watch, let go once the
//!   teardown began.
//! - **Nor after a wait, beside a block that waits**: likewise, with a released host import's
//!   block waiting for that mark and the release of freed memory that a later command buffer's
//!   completion posted, held too, as the program's wait for that command buffer returns. The
//!   teardown frees the block and runs its deallocator, before the last drop returns.
//! - **Not the queue-switch check's thread**: a device whose first command queue starts its check
//!   of the GPU's idling between `VkQueue`s (`queue_switch_check.rs`), held between two of its
//!   rounds, holds nothing of the device: the program's drops tear the device down on its own
//!   thread before the last drop returns, the teardown stopping the check and waiting for it at
//!   most one step and its release of its objects, within the 16 ms the owner allows a wait. A
//!   device that runs no check (one family of one `VkQueue`, or no calibrated timestamps) skips
//!   it, saying so.
//! - **Nor the submission checks' thread**: a device's checks of its submissions
//!   (`submit_check.rs`), which start as it is made, held between two of their steps, hold
//!   nothing of it either: the program's drop tears the device down on its own thread before it
//!   returns, the teardown stopping the checks at their next step. A device whose checks ended
//!   before the hold skips it, saying so.
//! - **Not the runtime's own holders' threads**: the check of textures over host pages
//!   (`in_place_check.rs`), held before a round while it holds the device, and the warm pass
//!   (`warm/pass.rs`), held before a record while it holds it. The program's last drop stops the
//!   holds and waits for them (`device/holds.rs`), so the device is torn down on the program's
//!   thread before that drop returns. A device whose check makes no round (its policy takes no
//!   texture in place, or it makes no class's pages) skips the check's case, saying so. A program
//!   handle's drop short of the last returns while another thread holds the registry's lock.
//!
//! The teardown's end is seen through the device's message sink, whose last share the instance
//! drops after `vkDestroyInstance`, the teardown's last step: the sink owns a value that, dropped,
//! sends the name of its thread.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The render pipeline is the owned
//! fixture `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::dispatch_pending as pending;
use super::freed_memory::{unused, Host, Runs};
use super::layer_claim::Watched;
use crate::device::holds::Holder;
use crate::device::lock;
use crate::device::prime::SHADER;
use crate::exit_guard;
use crate::pipeline::objects::{shader_module, words_digest};
use crate::pipeline::pool::held;
use crate::trails::blocked;
use crate::{
    Buffer, ClearColor, CommandBufferStatus, Device, DeviceKind, DeviceOptions, LoadAction,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ShaderCacheOptions, Size, StoreAction, Submission, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest a command buffer, or a teardown, may take.
const WAIT: Duration = Duration::from_secs(60);

/// The runtime's threads, by name.
const SCHEDULER: &str = "metal2vulkan-scheduler";
const HANDLERS: &str = "metal2vulkan-handlers";
const REAPER: &str = "metal2vulkan-reaper";

/// The longest a state's release may take (the owner's: a 16 ms wait is already too long).
const RELEASE: Duration = Duration::from_millis(16);

/// The render target's side.
const SIDE: u64 = 8;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(case: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "m2v-teardown-threads-{case}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Sends the name of the thread that drops it: the device's message sink owns it.
struct Ended(mpsc::Sender<String>);

impl Drop for Ended {
    fn drop(&mut self) {
        let thread = std::thread::current();
        let _ = self
            .0
            .send(thread.name().unwrap_or("an unnamed thread").to_owned());
    }
}

/// A watched device's teardown: the thread it ends on, and what its instance had the validation
/// layer report ([`Watched`], which says what a claim of its silence holds).
struct Ending {
    thread: mpsc::Receiver<String>,
    watched: Watched,
}

impl Ending {
    /// The device is not torn down: `holder` holds it still.
    fn not_yet(&self, holder: &str) {
        assert_eq!(
            self.thread.try_recv(),
            Err(TryRecvError::Empty),
            "the device was torn down while {holder} held it"
        );
    }

    /// The teardown already ended, on this thread: the last drop returned with the device gone;
    /// and nothing failed meanwhile.
    fn ended_here(&self) {
        assert_eq!(
            self.thread.try_recv(),
            Ok(program_thread()),
            "the device was not torn down on this thread before its last drop returned"
        );
        self.watched
            .layer_reported_nothing("the teardown that ended here left no object and used none");
    }

    /// The teardown ends, on one of `threads`, and nothing failed meanwhile.
    fn on(&self, threads: &[&str]) {
        let thread = self
            .thread
            .recv_timeout(WAIT)
            .expect("the device's teardown ends");
        assert!(
            threads.contains(&thread.as_str()),
            "the device's teardown ended on {thread}, not on one of {threads:?}"
        );
        self.watched.layer_reported_nothing(&format!(
            "the teardown that ended on {thread} left no object and used none"
        ));
    }
}

/// The device the cfg selects, its shader cache in `cache`, asking for the validation layer, and
/// reporting to a sink that sees its teardown end. What a claim of the layer's silence holds is
/// the instance's own answer ([`Watched::layer_reported_nothing`]), never the loader's listing.
fn watched(cache: &Path) -> (Device, Ending) {
    let (sender, thread) = mpsc::channel();
    let ended = Ended(sender);
    let watched = Watched::default();
    let sink = watched.sink(move |_| {
        // The sink owns the sentinel, which goes with its last share, the instance's.
        let _ = &ended;
    });
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
        layers: watched.layers(),
        messages: Some(sink),
        ..DeviceOptions::default()
    };
    // The options' share of the sink goes as this returns: the instance's is the last.
    let device = Device::system_default_with(&options).unwrap();
    (device, Ending { thread, watched })
}

/// The fixture's render pipeline state: `colored_vertex` and `colored_fragment`, one
/// `RGBA8Unorm` target. Its creation makes the pipeline of a triangle list's draw
/// (`pipeline/predicted.rs`), which a triangle strip shares where the topology is dynamic, and not
/// a line strip's, of another class.
fn state(device: &Device) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name("colored_fragment").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A `Shared` buffer of `values`' bytes.
fn floats(device: &Device, values: impl IntoIterator<Item = f32>) -> Buffer {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// A command buffer of `queue` that fills `buffer`, not committed.
fn filling(queue: &crate::CommandQueue, buffer: &Buffer) -> crate::CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..256, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// A draw's making waiting in the pool holds nothing of the device: its state keeps what it
/// needs (`pipeline/piece_holds.rs`). Once the program let go of the command buffer (dropped
/// uncommitted), the state and the device, the making stops and the last device handle goes on
/// the program's thread, as it drops it.
#[test]
fn a_making_waiting_in_the_pool_stops_and_the_device_goes_on_the_program_s_thread() {
    let scratch = Scratch::new("pool");
    let (device, ending) = watched(&scratch.0);
    let state = state(&device);
    let hold = held::hold();
    let (command_buffer, drawn) = line_strip_drawn(&device, &state);
    drop((command_buffer, drawn));
    drop((state, device));
    ending.on(&[&program_thread()]);
    drop(hold);
}

/// A draw's making running in the pool, which its state's release cannot stop: the release
/// returns at once, within the 16 ms the owner allows a wait (a making takes 20 to 50 s on
/// NVIDIA), and the device's teardown waits for the making's end, then runs on the device's
/// reaper, which lets go of what the making gave back, never on the pool's thread.
#[test]
fn a_making_running_in_the_pool_leaves_its_release_at_once_and_the_device_to_the_reaper() {
    let scratch = Scratch::new("reaper");
    let (device, ending) = watched(&scratch.0);
    let state = state(&device);
    let mid = held::mid_making_hold();
    let (command_buffer, drawn) = line_strip_drawn(&device, &state);
    assert!(
        mid.reached(1, WAIT),
        "the line strip's making runs in the pool"
    );
    drop((command_buffer, drawn));
    let releasing = Instant::now();
    drop(state);
    let release = releasing.elapsed();
    assert!(
        release < RELEASE,
        "the state's release took {release:?} while its making ran"
    );
    drop(device);
    ending.not_yet("the making running in the pool");
    drop(mid);
    ending.on(&[REAPER]);
}

/// A dispatch's making that the pool settled, its piece held before it gives back the variant it
/// took (`pool::held::settled_hold`): the command buffer that waited for the making completes and
/// its completed handler runs meanwhile, but its wait returns only once a wait met the piece's
/// trail and the piece let go (`trails.rs`). So the program's drops after the wait tear the device
/// down on its thread, before the last drop returns. Before trails, the wait returned while the
/// piece held the variant, and the device went on the reaper's thread once the piece gave it back
/// to the released state: `runtime_lane::events`'s drop after a wait outlived the drop so on the
/// Radeon 780M under load.
#[test]
fn a_device_dropped_after_a_wait_goes_on_the_program_s_thread_while_its_piece_trails() {
    let scratch = Scratch::new("trail");
    let (device, ending) = watched(&scratch.0);
    let state = pending::state(&device);
    let buffers = pending::Buffers::new(&device);
    let queue = device.new_command_queue().unwrap();
    let settled = held::settled_hold();
    let pooled = device.work_counts().dispatch_pipelines_pooled;
    let command_buffer = pending::dispatch(
        &queue,
        &state,
        &buffers,
        pending::Grid::One(Size::new(64, 1, 1)),
    );
    assert_eq!(
        device.work_counts().dispatch_pipelines_pooled,
        pooled + 1,
        "the dispatch's making went to the pool"
    );
    let (handled, completed) = mpsc::channel();
    command_buffer
        .add_completed_handler(Box::new(move |_| {
            let _ = handled.send(());
        }))
        .unwrap();
    command_buffer.commit().unwrap();
    assert!(
        settled.reached(1, WAIT),
        "the dispatch's piece settled its making"
    );
    completed
        .recv_timeout(WAIT)
        .expect("the command buffer completes while the piece holds its variant");
    // Released once a wait met the piece's open trail.
    let releasing = std::thread::spawn(move || {
        let met = blocked::reached(1, WAIT);
        drop(settled);
        met
    });
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    drop((command_buffer, queue, buffers, state));
    drop(device);
    ending.ended_here();
    assert!(
        releasing.join().unwrap(),
        "the wait met the piece's open trail"
    );
}

/// A render state's creation whose making the pool settled, its piece held before it gives back
/// the variant it took (`pool::held::settled_hold`): the creation returns only once its wait met
/// the piece's trail and the piece let go (`trails.rs`), so the program's drops of the state and
/// the device right after it tear the device down on its thread, before the last drop returns.
/// Before, the creation returned at the settle while the piece held the variant, and the device
/// went on the pool's or the reaper's thread after the program's last drop returned:
/// `render_stage_digests`' second device over the same options revived the first, its memo warm,
/// or met a directory its teardown had not finished writing, under a loaded host.
#[test]
fn a_device_dropped_after_a_creation_goes_on_the_program_s_thread_while_its_piece_trails() {
    let scratch = Scratch::new("creation-trail");
    let (device, ending) = watched(&scratch.0);
    let settled = held::settled_hold();
    let made = device.work_counts().creation_pipelines_made;
    // Released once a wait met the piece's open trail.
    let releasing = std::thread::spawn(move || {
        let met = blocked::reached(1, WAIT);
        drop(settled);
        met
    });
    let state = state(&device);
    assert!(
        device.work_counts().creation_pipelines_made > made,
        "the creation made its predicted pipeline"
    );
    drop(state);
    drop(device);
    ending.ended_here();
    assert!(
        releasing.join().unwrap(),
        "the creation's wait met the piece's open trail"
    );
}

/// This thread's name, as [`Ended`] sends it.
fn program_thread() -> String {
    let thread = std::thread::current();
    thread.name().unwrap_or("an unnamed thread").to_owned()
}

/// An uncommitted command buffer of `device` whose draw of a line strip with `state` handed its
/// making to the pool, and what else the draw used.
fn line_strip_drawn(
    device: &Device,
    state: &RenderPipelineState,
) -> (crate::CommandBuffer, impl Sized) {
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let positions = floats(
        device,
        [[-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
            .into_iter()
            .flatten(),
    );
    let colors = floats(device, (0..4).flat_map(|_| [1.0, 0.2, 0.0, 1.0]));
    let queue = device.new_command_queue().unwrap();
    let pooled = device.work_counts().pipelines_pooled;
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    encoder
        .draw_primitives(PrimitiveType::LineStrip, 0, 4, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    assert_eq!(
        device.work_counts().pipelines_pooled,
        pooled + 1,
        "the line strip's making went to the pool"
    );
    (command_buffer, (pass, queue, colors, positions, target))
}

/// A committed command buffer held at its submission (its check blocked on the scheduler's
/// thread) holds the device once the program let go of every handle: the last device handle goes
/// on the scheduler or on the queue's handler thread, never on the program's.
#[test]
fn a_command_buffer_held_at_its_submission_tears_the_device_down_on_the_runtime_s_threads() {
    let scratch = Scratch::new("scheduler");
    let (device, ending) = watched(&scratch.0);
    let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = filling(&queue, &buffer);
    let (entered, checking) = mpsc::channel();
    let (open, gate) = mpsc::channel::<()>();
    let gate = Mutex::new(gate);
    command_buffer
        .set_submission_check(Arc::new(move |_: &Submission<'_>| {
            let _ = entered.send(());
            // Until the test drops its end: then the submission passes.
            let _ = lock(&gate).recv();
            None
        }))
        .unwrap();
    command_buffer.commit().unwrap();
    checking
        .recv_timeout(WAIT)
        .expect("the scheduler checks the submission");
    drop((command_buffer, queue, buffer, device));
    ending.not_yet("the scheduler, checking the submission");
    drop(open);
    ending.on(&[SCHEDULER, HANDLERS]);
}

/// A host step's own module (one the device's map keeps only for a sweep or its teardown), held
/// by a completed handler blocked on its gate, goes on the handler's thread just before the last
/// device handle, which the command buffer's release drops there; the teardown destroys the
/// module once.
#[test]
fn a_host_step_s_own_module_and_the_last_device_go_on_a_handler_thread() {
    let scratch = Scratch::new("handler");
    let (device, ending) = watched(&scratch.0);
    let words: Arc<[u32]> = Arc::from(&SHADER[..]);
    let digest = words_digest(&words);
    let module = exit_guard::host_step(|| shader_module(&device, &words, &digest)).unwrap();
    assert!(
        lock(&device.shared().shader_modules).get(&digest).is_none(),
        "a host step's module is its own: no lookup finds it"
    );
    let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let command_buffer = filling(&first, &buffer);
    let (entered, running) = mpsc::channel();
    let (open, gate) = mpsc::channel::<()>();
    command_buffer
        .add_completed_handler(Box::new(move |_| {
            let _ = entered.send(());
            let _ = gate.recv();
            // The module goes here, then the command buffer's release, on this thread.
            drop(module);
        }))
        .unwrap();
    command_buffer.commit().unwrap();
    running
        .recv_timeout(WAIT)
        .expect("the completed handler runs");
    // A command buffer of another queue, committed now, finished: the scheduler let go of what it
    // submitted before it, and no completion turn holds a device handle between batches.
    // Only the handler's thread holds the first command buffer now.
    let after = filling(&second, &buffer);
    after.commit().unwrap();
    after.wait_until_completed_for(WAIT).unwrap();
    drop((after, command_buffer, first, second, buffer, words, device));
    ending.not_yet("the completed handler");
    drop(open);
    ending.on(&[HANDLERS]);
}

/// A thread that lets the hold of `device`'s watch and releases go once its teardown began, or
/// once the test let it go: the teardown waits for what the hold holds.
fn let_go_once_torn_down(device: &Device) -> std::thread::JoinHandle<()> {
    let parking = Arc::clone(device.shared().parking.get().expect("the device parks"));
    std::thread::spawn(move || parking.release_once_closed())
}

/// The program's use of a queue leaves its mark pending; the watch of the marks, held as it
/// begins, holds nothing of the device: the program's drops tear the device down on its own
/// thread before the last drop returns, the teardown waiting for the watch.
#[test]
fn the_watch_of_the_program_s_marks_leaves_the_device_to_the_program_s_thread() {
    let scratch = Scratch::new("marks");
    let (device, ending) = watched(&scratch.0);
    device.park_freed_memory();
    let parking = Arc::clone(device.shared().parking.get().expect("the device parks"));
    let queue = device.new_command_queue().unwrap();
    parking.held.hold();
    // SAFETY: the closure uses the queue for nothing.
    unsafe { queue.with_vk_queue(|_, _, _| ()) };
    parking.held.until_waiting(1, WAIT);
    let releasing = let_go_once_torn_down(&device);
    drop(queue);
    drop(device);
    let torn_down = ending.thread.try_recv();
    parking.held.release();
    releasing.join().unwrap();
    assert_eq!(
        torn_down,
        Ok(program_thread()),
        "the device was not torn down on this thread before its last drop returned"
    );
    ending
        .watched
        .layer_reported_nothing("the teardown that ended here left no object and used none");
}

/// A released host import's block waits for the program's mark, which only the held watch sees;
/// a later command buffer's completion posts the release of freed memory, held too, before the
/// program's wait for it returns. Neither holds the device: the program's drops tear it down on
/// its own thread before the last drop returns, freeing the block and running its deallocator.
#[test]
fn a_release_of_freed_memory_leaves_the_device_to_the_program_s_thread() {
    let scratch = Scratch::new("unpark");
    let host = Host::new();
    let (device, ending) = watched(&scratch.0);
    device.park_freed_memory();
    let parking = Arc::clone(device.shared().parking.get().expect("the device parks"));
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(Runs::default());
    let Some(import) = unused(&device, &host, &runs) else {
        return;
    };
    parking.held.hold();
    // SAFETY: the closure uses the queue for nothing.
    unsafe { queue.with_vk_queue(|_, _, _| ()) };
    parking.held.until_waiting(1, WAIT);
    drop(import);
    assert_eq!(
        device.work_counts().freed_memory_waiting,
        1,
        "the import's block waits for the mark"
    );
    let buffer = floats(&device, [0.0; 64]);
    let command_buffer = filling(&queue, &buffer);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    parking.held.until_waiting(2, WAIT);
    let releasing = let_go_once_torn_down(&device);
    drop((command_buffer, buffer, queue));
    drop(device);
    let torn_down = ending.thread.try_recv();
    parking.held.release();
    releasing.join().unwrap();
    assert_eq!(
        torn_down,
        Ok(program_thread()),
        "the device was not torn down on this thread before its last drop returned"
    );
    assert_eq!(
        runs.count.load(Ordering::SeqCst),
        1,
        "the teardown freed the block and ran its deallocator"
    );
    ending
        .watched
        .layer_reported_nothing("the teardown that freed the block left no object and used none");
    drop(host);
}

/// A device whose check of its GPU's idling between `VkQueue`s runs, held between two of its rounds
/// (`queue_switch_check.rs`): the check holds no device handle, so the program's last drop tears
/// the device down on its own thread before it returns, and the teardown's wait for the check (it
/// stops at its next step and lets its objects go) is within [`RELEASE`]. Without the hold the
/// check runs some milliseconds after the first command queue, the window in which a check that
/// held its device took the last drop's teardown to its own thread.
#[test]
fn a_running_queue_switch_check_leaves_the_device_to_the_program_s_thread() {
    let scratch = Scratch::new("switch");
    let (device, ending) = watched(&scratch.0);
    let switch = Arc::clone(&device.shared().queue_switch);
    if crate::queue_switch_check::pair(&device).is_none()
        || !crate::queue_switch_check::timed(&device)
    {
        println!(
            "SKIPPED {}: the device runs no queue-switch check",
            device.name()
        );
        return;
    }
    switch.held.hold();
    let queue = device.new_command_queue().unwrap();
    switch.held.until_waiting(1, WAIT);
    let releasing = std::thread::spawn({
        let switch = Arc::clone(&switch);
        move || switch.release_once_stopping()
    });
    let dropping = Instant::now();
    drop(queue);
    drop(device);
    let dropped = dropping.elapsed();
    let torn_down = ending.thread.try_recv();
    switch.held.release();
    releasing.join().unwrap();
    assert_eq!(
        torn_down,
        Ok(program_thread()),
        "the device was not torn down on this thread before its last drop returned"
    );
    let waited = switch
        .stop_waited()
        .expect("the teardown waited for the running check");
    let [release, shares] = switch.last_steps();
    println!(
        "the last drop took {dropped:?}, of which the wait for the check {waited:?} (its objects' \
         release {release:?}, its shares' drop {shares:?})"
    );
    assert!(
        waited < RELEASE,
        "the teardown waited {waited:?} for the check"
    );
    ending
        .watched
        .layer_reported_nothing("the teardown that stopped the check left no object and used none");
}

/// A device whose submission checks run, held between two of their steps (`submit_check.rs`):
/// the checks hold no device handle, so the program's last drop tears the device down on its own
/// thread before it returns. Unheld, they run some milliseconds after the device is made, the
/// window in which checks that held their device took a last drop's teardown on one of the
/// runtime's threads to their own.
#[test]
fn running_submit_checks_leave_the_device_to_the_program_s_thread() {
    let scratch = Scratch::new("submit");
    let (device, ending) = watched(&scratch.0);
    let checks = Arc::clone(&device.shared().submit_checks);
    checks.held.hold();
    // The checks started as the device was made: they wait at their next step, or ended.
    let deadline = Instant::now() + WAIT;
    while checks.held.waiting() == 0 && !device.settle_submit_checks(Duration::ZERO) {
        assert!(
            Instant::now() < deadline,
            "the checks neither held nor ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    if checks.held.waiting() == 0 {
        checks.held.release();
        println!(
            "SKIPPED {}: the device's submission checks ended before the hold",
            device.name()
        );
        return;
    }
    let releasing = std::thread::spawn({
        let checks = Arc::clone(&checks);
        move || checks.release_once_stopping()
    });
    drop(device);
    let torn_down = ending.thread.try_recv();
    checks.held.release();
    releasing.join().unwrap();
    assert_eq!(
        torn_down,
        Ok(program_thread()),
        "the device was not torn down on this thread before its last drop returned"
    );
    ending.watched.layer_reported_nothing(
        "the teardown that stopped the checks left no object and used none",
    );
}

/// The device whose holds `holder` takes, `holder` held at its next step while it holds the device
/// (`device/holds.rs`); `start` sets it going. `None` where it ended without reaching the hold,
/// which `ended` answers.
fn held_at_a_step(
    device: &Device,
    holder: Holder,
    start: impl FnOnce(),
    ended: impl Fn() -> bool,
) -> Option<Arc<crate::device::holds::Holds>> {
    let holds = Arc::clone(&device.shared().holds);
    let held = holds.held(holder);
    held.hold();
    start();
    let started = Instant::now();
    while held.waiting() == 0 {
        if ended() {
            held.release();
            return None;
        }
        assert!(
            started.elapsed() < WAIT,
            "{holder:?} never reached its hold"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    Some(holds)
}

/// Drop `device`, the program's last handle, while `holder` holds it at its hold, letting the
/// hold go once the drop stopped the holds: the device is torn down on this thread before the
/// drop returns, having waited for the holder, and nothing failed meanwhile.
fn dropped_while_held(
    device: Device,
    ending: &Ending,
    holds: &Arc<crate::device::holds::Holds>,
    holder: Holder,
) {
    let releasing = std::thread::spawn({
        let holds = Arc::clone(holds);
        move || holds.release_once_stopping(holder)
    });
    let dropping = Instant::now();
    drop(device);
    let dropped = dropping.elapsed();
    let torn_down = ending.thread.try_recv();
    holds.held(holder).release();
    releasing.join().unwrap();
    assert_eq!(
        torn_down,
        Ok(program_thread()),
        "the device was not torn down on this thread before its last drop returned"
    );
    let waited = holds.waited().expect("the last drop waited for the holder");
    println!("the last drop took {dropped:?}, of which the wait for {holder:?} {waited:?}");
    ending.watched.layer_reported_nothing(&format!(
        "the teardown that stopped {holder:?} left no object and used none"
    ));
}

/// A device whose check of textures over host pages runs (`in_place_check.rs`), held before one
/// of its rounds, while it holds the device and its class's buffer: the program's last drop stops
/// the check there and waits for it to let them go, so the device is torn down on the program's
/// thread before the drop returns. Without the wait the check's thread lets the device go last,
/// after the drop returned.
#[test]
fn a_running_in_place_check_leaves_the_device_to_the_program_s_thread() {
    let scratch = Scratch::new("in-place");
    let (device, ending) = watched(&scratch.0);
    let policy = device.compile_device().policy().host_page_textures;
    if policy != metal2vulkan::device::HostPageTexturePolicy::InPlace {
        println!(
            "SKIPPED {}: the device takes no texture in place ({policy:?})",
            device.name()
        );
        return;
    }
    let checks = |device: &Device| device.shared().in_place.wait(Duration::ZERO);
    let start = || {
        device.settle_in_place_checks(Duration::ZERO);
    };
    let Some(holds) = held_at_a_step(&device, Holder::InPlaceCheck, start, || checks(&device))
    else {
        println!(
            "SKIPPED {}: the device's check of textures in place made no round ({:?})",
            device.name(),
            device.in_place_check_report()
        );
        return;
    };
    dropped_while_held(device, &ending, &holds, Holder::InPlaceCheck);
}

/// A device's warm pass over records an earlier device made over the same directory, held before
/// its first record while it holds the device (`warm/pass.rs`): the program's last drop stops the
/// pass there and waits for it to let the device go, so the device is torn down on the program's
/// thread before the drop returns.
#[test]
fn a_running_warm_pass_leaves_the_device_to_the_program_s_thread() {
    let scratch = Scratch::new("warm");
    let records = super::warm::recorded(&scratch.0);
    let (device, ending) = watched(&scratch.0);
    let start = || device.warm(records, u64::MAX);
    let holds = held_at_a_step(&device, Holder::WarmPass, start, || false)
        .expect("the warm pass reaches its hold");
    dropped_while_held(device, &ending, &holds, Holder::WarmPass);
}

/// A program handle's drop that leaves the program other handles counts it gone without the holds
/// registry's lock (`device/holds.rs`): the drops of the handles the program's objects, command
/// buffers and encoders make, a draw's among them, share no lock. One dropped on another thread
/// while this thread holds the lock returns; the program's last drop then tears the device down
/// here. A drop that waited for the lock would not return until the lock went.
#[test]
fn a_program_handle_s_drop_short_of_the_last_takes_no_lock() {
    let scratch = Scratch::new("program-drop");
    let (device, ending) = watched(&scratch.0);
    let other = device.clone();
    let holds = Arc::clone(&device.shared().holds);
    let (returned, dropping) = holds.locked(|| {
        let (sender, dropped) = mpsc::channel();
        let dropping = std::thread::spawn(move || {
            drop(other);
            let _ = sender.send(());
        });
        (dropped.recv_timeout(WAIT).is_ok(), dropping)
    });
    dropping.join().unwrap();
    assert!(
        returned,
        "a program handle's drop short of the last waited for the registry's lock"
    );
    drop(holds);
    drop(device);
    ending.ended_here();
}
