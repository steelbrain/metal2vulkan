//! Another queue's work while one queue's host step records, on the device the cfg selects: one
//! device serves every queue, as a consumer's serves every guest process, and a host step that
//! records for seconds (an indirect command buffer of thousands of draws made resident for
//! writing) holds back only its own queue's command buffers and what Metal orders after it.
//!
//! The draws are `resident_draws.metal`'s entry shape (`runtime_lane::resident_draws`): draw `i`
//! binds entry `i` of an argument table (its position's address, a texture's resource ID and a
//! sampler's), as a page's render bundle binds a bind group a draw, each position in a buffer of
//! its own, and the pass makes the positions, the table and the texture resident, and the
//! commands resident for writing, so a host step records their execution.
//!
//! Queue A's command buffer executes such draws; its executed-commands handler, which the runtime
//! calls on the thread that reads the records, inside the step, holds the step until the test lets
//! it go. Meanwhile B, a queue a kind, commits, each sharing no resource with the held step, and
//! waits for:
//!
//! - **plain** work: a blit's fill of a buffer of B's own, read back;
//! - **stepped** work: an execution of draws of B's own, a host step of its own, read back;
//! - **ordered** work: a copy out of a texture A rendered into before the held command buffer,
//!   after waiting for the event A signalled there, read back.
//!
//! [`another_queue_goes_on_while_a_host_step_records`] checks each completes, with what it wrote,
//! while A's step is held, and again once the program let go of every object of A's while the
//! step is held, as a guest process's end lets go of its context's. It checks too that a step
//! Metal orders after A's (after an event A's command buffer signals, or a pass into A's target)
//! begins only once A's ended. [`steps_apart`] measures,
//! without the layer and with no hold, each kind's commit-to-completion while A's step of
//! [`DRAWS_VAR`] draws records (A's objects kept, and let go of once committed), and on an idle
//! device: printed as `STEPS-APART` lines, the median and largest. It is ignored by default: run
//! it by name with `--run-ignored all`.

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Device,
    Event, EventRef, ExecutedCommands, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    LoadAction, Origin, PipelineOptions, RecordsRead, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ResourceRef, SamplerState, SamplerStateOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::sync::{mpsc, Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// How long a command buffer of B's may take, at most, while A's step is held: B's work waits
/// for nothing of A's, so milliseconds in the usual case; well past any of them.
const BOUND: Duration = Duration::from_secs(10);

/// The side of the texture A renders into and B copies out of.
const SIDE: u64 = 16;

/// The colour A clears that texture to, and the bytes B reads of each texel.
const CLEAR: [f64; 4] = [0.25, 0.5, 0.75, 1.0];
const TEXEL: [u8; 4] = [64, 128, 191, 255];

/// The bytes of B's own buffer.
const FILLED: u64 = 4096;

/// The side of a draws' target.
const TARGET: u64 = 128;

/// The draws of each command buffer the test's step records.
const DRAWS: u64 = 64;

/// The variable that sets the measurement's draw count in A's step (5000 unless it says).
const DRAWS_VAR: &str = "STEPS_APART_DRAWS";

/// The variable that sets the measurement's idle command buffers of each kind (20 unless it says).
const IDLE_VAR: &str = "STEPS_APART_IDLE";

/// The draws' state, and the texture and sampler their entries name.
struct Scene {
    device: Device,
    entry: RenderPipelineState,
    sampled: Texture,
    sampler: SamplerState,
}

impl Scene {
    fn new(device: &Device) -> Scene {
        let bitcode = llvm_assemble(&engine_dir().join("resident_draws.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("rd_entry_vertex"));
        descriptor.fragment = Some(function("rd_entry_fragment"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.support_indirect_command_buffers = true;
        let entry = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut sampled = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        sampled.usage = TextureUsage::SHADER_READ;
        sampled.resource_options = ResourceOptions::shared();
        let sampled = device.new_texture(&sampled).unwrap();
        let texels: Vec<u8> = [40u8, 160, 220, 255].repeat(16);
        (sampled.replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &texels, 16, 0)).unwrap();
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let sampler = device
            .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
            .unwrap();
        Scene {
            device: device.clone(),
            entry,
            sampled,
            sampler,
        }
    }
}

/// `n` draws of the entry shape (the module docs), their positions, table, commands and target.
struct Draws<'s> {
    scene: &'s Scene,
    positions: Vec<Buffer>,
    table: Buffer,
    icb: IndirectCommandBuffer,
    target: Texture,
}

impl<'s> Draws<'s> {
    fn new(scene: &'s Scene, n: u64) -> Draws<'s> {
        let device = &scene.device;
        let positions: Vec<Buffer> = (0..n)
            .map(|i| {
                let position = [
                    (i % 71) as f32 / 36.0 - 1.0,
                    (i / 71) as f32 / 36.0 - 1.0,
                    0.0,
                    1.0,
                ];
                let bytes: Vec<u8> = position.into_iter().flat_map(f32::to_le_bytes).collect();
                buffer_with(device, &bytes).unwrap()
            })
            .collect();
        let texture = scene.sampled.gpu_resource_id().0;
        let sampler = scene.sampler.gpu_resource_id().0;
        let entries: Vec<u8> = (positions.iter())
            .flat_map(|position| [position.gpu_address().0, texture, sampler])
            .flat_map(u64::to_le_bytes)
            .collect();
        let table = buffer_with(device, &entries).unwrap();
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.max_vertex_buffer_bind_count = 1;
        descriptor.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&descriptor, n, ResourceOptions::shared())
            .unwrap();
        for i in 0..n {
            let command = icb.indirect_render_command_at(i).unwrap();
            command.set_render_pipeline_state(&scene.entry).unwrap();
            command.set_vertex_buffer(Some(&table), 24 * i, 0).unwrap();
            command
                .set_fragment_buffer(Some(&table), 24 * i, 0)
                .unwrap();
            command
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        let mut target =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, TARGET, TARGET, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        Draws {
            scene,
            positions,
            table,
            icb,
            target: device.new_texture(&target).unwrap(),
        }
    }

    /// Encode, into a new command buffer of `queue`, [`pass`](Self::pass) into its own target.
    /// Not committed, so that a handler can be added.
    fn encode(&self, queue: &CommandQueue) -> CommandBuffer {
        let command_buffer = queue.command_buffer().unwrap();
        self.pass(&command_buffer, &self.target);
        command_buffer
    }

    /// Encode into `command_buffer` a pass that clears `target`, makes the draws' resources
    /// resident, and the commands resident for writing, and executes every command: a host step
    /// records them.
    fn pass(&self, command_buffer: &CommandBuffer, target: &Texture) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let vertex = RenderStages::VERTEX;
        let both = RenderStages::VERTEX | RenderStages::FRAGMENT;
        let positions: Vec<ResourceRef> = self.positions.iter().map(ResourceRef::Buffer).collect();
        (encoder.use_resources(&positions, ResourceUsage::READ, vertex)).unwrap();
        (encoder.use_resource(self.table.as_resource(), ResourceUsage::READ, both)).unwrap();
        let sampled = self.scene.sampled.as_resource();
        (encoder.use_resource(sampled, ResourceUsage::READ, RenderStages::FRAGMENT)).unwrap();
        (encoder.use_resource(self.icb.as_resource(), ResourceUsage::WRITE, vertex)).unwrap();
        let n = self.positions.len() as u64;
        encoder.execute_commands_in_buffer(&self.icb, 0..n).unwrap();
        encoder.end_encoding().unwrap();
    }

    /// Whether the target holds the draws: some texel drawn, of the sampled colour.
    fn drew(&self) -> bool {
        drew_into(&self.target)
    }
}

/// Whether `target`, a draws' target, holds draws: some texel drawn, of the sampled colour.
fn drew_into(target: &Texture) -> bool {
    let mut bytes = vec![0u8; (4 * TARGET * TARGET) as usize];
    let region = Region::new_2d(0, 0, TARGET, TARGET);
    (target.get_bytes(&mut bytes, 4 * TARGET, 0, region, 0, 0)).unwrap();
    let drawn: Vec<&[u8]> = bytes.chunks(4).filter(|texel| texel[3] != 0).collect();
    !drawn.is_empty() && drawn.iter().all(|&texel| texel == [40, 160, 220, 255])
}

/// What B's command buffers do (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Plain,
    Stepped,
    Ordered,
}

const KINDS: [Kind; 3] = [Kind::Plain, Kind::Stepped, Kind::Ordered];

/// Queue B, a queue a kind, and what its command buffers use, apart from anything of A's held
/// step: its own buffer, its own draws, and the buffer it copies A's texture into. Each kind has a
/// queue of its own, so that none waits for another's in its queue's order.
struct Other<'s> {
    queues: [CommandQueue; 3],
    filled: Buffer,
    stepped: Draws<'s>,
    copied: Buffer,
    drawn: Texture,
    rendered: Event,
}

impl<'s> Other<'s> {
    /// Queue B's resources, A's texture `drawn` and the event `rendered` A signalled once it
    /// rendered into it (at value 1).
    fn new(scene: &'s Scene, drawn: &Texture, rendered: &Event) -> Other<'s> {
        let device = &scene.device;
        Other {
            queues: KINDS.map(|_| device.new_command_queue().unwrap()),
            filled: (device.new_buffer(FILLED, ResourceOptions::shared())).unwrap(),
            stepped: Draws::new(scene, DRAWS),
            copied: (device.new_buffer(4 * SIDE * SIDE, ResourceOptions::shared())).unwrap(),
            drawn: drawn.clone(),
            rendered: rendered.clone(),
        }
    }

    /// Commit one command buffer of `kind`, its `round`-th: what it writes is what
    /// [`check`](Self::check) reads.
    fn commit(&self, kind: Kind, round: u8) -> CommandBuffer {
        let queue = &self.queues[kind as usize];
        let command_buffer = match kind {
            Kind::Plain => {
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&self.filled, 0..FILLED, round).unwrap();
                blit.end_encoding().unwrap();
                command_buffer
            }
            Kind::Stepped => self.stepped.encode(queue),
            Kind::Ordered => {
                let command_buffer = queue.command_buffer().unwrap();
                (command_buffer.encode_wait_for_event(EventRef::from(&self.rendered), 1)).unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &self.drawn,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(SIDE, SIDE, 1),
                    &self.copied,
                    0,
                    4 * SIDE,
                    4 * SIDE * SIDE,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command_buffer
            }
        };
        command_buffer.commit().unwrap();
        command_buffer
    }

    /// Whether what the completed command buffer of `kind`, its `round`-th, wrote reads back.
    fn check(&self, kind: Kind, round: u8) -> bool {
        match kind {
            Kind::Plain => {
                let bytes = read_buffer(&self.filled, 0, FILLED).unwrap();
                bytes.iter().all(|&byte| byte == round)
            }
            Kind::Stepped => self.stepped.drew(),
            Kind::Ordered => {
                let bytes = read_buffer(&self.copied, 0, 4 * SIDE * SIDE).unwrap();
                bytes.chunks(4).all(|texel| texel == TEXEL)
            }
        }
    }

    /// Commit one command buffer of `kind`, its `round`-th, and wait for it for `bound` at most:
    /// its commit-to-completion in milliseconds, where it completed with what it wrote.
    fn run(&self, kind: Kind, round: u8, bound: Duration) -> Option<f64> {
        let started = Instant::now();
        let command_buffer = self.commit(kind, round);
        let completed = command_buffer.wait_until_completed_for(bound).is_ok()
            && command_buffer.status() == CommandBufferStatus::Completed;
        let took = started.elapsed().as_secs_f64() * 1e3;
        (completed && self.check(kind, round)).then_some(took)
    }
}

/// A `SIDE`x`SIDE` texture A renders into and B copies out of, cleared to [`CLEAR`] on `queue`,
/// with `rendered` signalled at 1 after it, waited for.
fn rendered_texture(device: &Device, queue: &CommandQueue, rendered: &Event) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let drawn = device.new_texture(&descriptor).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(drawn.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(CLEAR[0], CLEAR[1], CLEAR[2], CLEAR[3]);
    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.end_encoding().unwrap();
    (command_buffer.encode_signal_event(EventRef::from(rendered), 1)).unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    drawn
}

/// What holds a host step in its command buffer's executed-commands handler, on the thread that
/// records it, until it is let go of: by [`release`](Self::release), or as it is dropped.
struct StepHold {
    released: Arc<(Mutex<bool>, Condvar)>,
    entered: mpsc::Receiver<()>,
}

impl StepHold {
    /// Hold `command_buffer`'s host step, before its commit.
    fn on(command_buffer: &CommandBuffer) -> StepHold {
        let released = Arc::new((Mutex::new(false), Condvar::new()));
        let (enter, entered) = mpsc::channel();
        let holding = Arc::clone(&released);
        command_buffer
            .add_executed_commands_handler(Arc::new(move |executed: &ExecutedCommands| {
                if executed.read != RecordsRead::HostStep {
                    return;
                }
                let _ = enter.send(());
                let (released, let_go) = &*holding;
                let mut released = released.lock().unwrap_or_else(PoisonError::into_inner);
                while !*released {
                    released = let_go
                        .wait(released)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }))
            .unwrap();
        StepHold { released, entered }
    }

    /// Watch `command_buffer`'s host step, before its commit: as it begins, whether this hold
    /// was let go of by then.
    fn watch(&self, command_buffer: &CommandBuffer) -> mpsc::Receiver<bool> {
        let (began, watched) = mpsc::channel();
        let released = Arc::clone(&self.released);
        command_buffer
            .add_executed_commands_handler(Arc::new(move |executed: &ExecutedCommands| {
                if executed.read == RecordsRead::HostStep {
                    let (released, _) = &*released;
                    let _ = began.send(*released.lock().unwrap_or_else(PoisonError::into_inner));
                }
            }))
            .unwrap();
        watched
    }

    /// Let the step go on.
    fn release(&self) {
        let (released, let_go) = &*self.released;
        *released.lock().unwrap_or_else(PoisonError::into_inner) = true;
        let_go.notify_all();
    }
}

impl Drop for StepHold {
    fn drop(&mut self) {
        self.release();
    }
}

/// While queue A's host step is held, inside the step, queue B's plain, stepped and ordered
/// command buffers each complete with what they wrote; then A's does, with what it drew. The same
/// once the program let go of every object of A's (its command buffer, its draws' resources and
/// commands, its queue) while the step is held: its completed handler still runs.
#[test]
fn another_queue_goes_on_while_a_host_step_records() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let rendered = device.new_event().unwrap();
        let first = device.new_command_queue().unwrap();
        let drawn = rendered_texture(device, &first, &rendered);
        let other = Other::new(&scene, &drawn, &rendered);
        // Warm: each kind's variants and pipelines are made once.
        for kind in KINDS {
            assert!(other.run(kind, 1, WAIT).is_some(), "{kind:?} warm");
        }
        let mut late = Vec::new();
        for released in [false, true] {
            let a = device.new_command_queue().unwrap();
            let held = Draws::new(&scene, DRAWS);
            // Warm: A's step's variants and pipelines are made once.
            let warm = held.encode(&a);
            warm.commit().unwrap();
            warm.wait_until_completed_for(WAIT).unwrap();
            let command_buffer = held.encode(&a);
            let after = device.new_event().unwrap();
            (command_buffer.encode_signal_event(EventRef::from(&after), 1)).unwrap();
            let hold = StepHold::on(&command_buffer);
            let (done, completed) = mpsc::channel();
            command_buffer
                .add_completed_handler(Box::new(move |command_buffer: &CommandBuffer| {
                    let _ = done.send(command_buffer.status());
                }))
                .unwrap();
            command_buffer.commit().unwrap();
            (hold.entered.recv_timeout(WAIT)).expect("A's host step began");
            let kept = match released {
                false => Some((a, held, command_buffer)),
                true => {
                    drop((a, held, command_buffer));
                    None
                }
            };
            let ran: Vec<(Kind, Option<f64>)> = (KINDS.iter())
                .map(|&kind| (kind, other.run(kind, 2, BOUND)))
                .collect();
            // Steps Metal orders after A's, each on a queue of its own: one after the event A
            // signals after its pass, one a pass into A's target. Neither records before A's
            // step ends.
            let ordered = kept.as_ref().map(|(_, held, _)| {
                let c = [(); 2].map(|_| device.new_command_queue().unwrap());
                let by_event = Draws::new(&scene, DRAWS);
                let waiting = c[0].command_buffer().unwrap();
                (waiting.encode_wait_for_event(EventRef::from(&after), 1)).unwrap();
                by_event.pass(&waiting, &by_event.target);
                let sharing = c[1].command_buffer().unwrap();
                by_event.pass(&sharing, &held.target);
                let watched = [hold.watch(&waiting), hold.watch(&sharing)];
                waiting.commit().unwrap();
                sharing.commit().unwrap();
                std::thread::sleep(Duration::from_millis(500));
                let early: Vec<bool> = (watched.iter())
                    .map(|watched| watched.try_recv().is_ok())
                    .collect();
                (by_event, [waiting, sharing], watched, early, c)
            });
            let still_held = completed.try_recv().is_err();
            hold.release();
            let status = (completed.recv_timeout(WAIT)).expect("A's command buffer completed");
            assert_eq!(
                status,
                CommandBufferStatus::Completed,
                "released {released}"
            );
            if let Some((_, held, _)) = &kept {
                assert!(held.drew(), "A's draws");
            }
            if let Some((by_event, ordered, watched, early, _c)) = ordered {
                assert_eq!(
                    early,
                    [false, false],
                    "a step Metal orders after A's began first"
                );
                for (command_buffer, watched) in ordered.iter().zip(&watched) {
                    let began = watched.recv_timeout(WAIT).expect("an ordered step began");
                    assert!(
                        began,
                        "a step Metal orders after A's began before A's ended"
                    );
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                }
                assert!(by_event.drew(), "the draws after A's event");
                let (_, held, _) = kept.as_ref().unwrap();
                assert!(held.drew(), "the draws into A's target");
            }
            assert!(
                still_held,
                "released {released}: A's command buffer completed while its step was held"
            );
            eprintln!("released {released}: each kind's commit-to-completion, in ms: {ran:?}");
            late.extend(
                (ran.iter())
                    .filter(|(_, took)| took.is_none())
                    .map(|&(kind, _)| (released, kind)),
            );
        }
        assert!(
            late.is_empty(),
            "queue B's command buffers (whether A's objects were released, and the kind) that did \
             not complete within {BOUND:?} while queue A's host step was held: {late:?}"
        );
    });
}

/// The median and largest of `times`, in milliseconds, as printed.
fn spread(times: &mut [f64]) -> String {
    if times.is_empty() {
        return "none".to_owned();
    }
    times.sort_by(f64::total_cmp);
    format!(
        "{} of them, {:.2} ms median, {:.2} largest",
        times.len(),
        times[times.len() / 2],
        times[times.len() - 1]
    )
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn steps_apart() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let n: u64 = std::env::var(DRAWS_VAR).map_or(5000, |n| n.parse().expect(DRAWS_VAR));
    let idle: usize = std::env::var(IDLE_VAR).map_or(20, |n| n.parse().expect(IDLE_VAR));
    let name = device.name().to_owned();
    let scene = Scene::new(&device);
    let a = device.new_command_queue().unwrap();
    let rendered = device.new_event().unwrap();
    let drawn = rendered_texture(&device, &a, &rendered);
    let other = Other::new(&scene, &drawn, &rendered);
    let slow = Draws::new(&scene, n);
    // Warm: A's frame once, each kind once.
    let warm = slow.encode(&a);
    warm.commit().unwrap();
    warm.wait_until_completed_for(WAIT).unwrap();
    for kind in KINDS {
        assert!(other.run(kind, 1, WAIT).is_some(), "{kind:?} warm");
    }
    for kind in KINDS {
        let mut times: Vec<f64> = (0..idle)
            .filter_map(|round| other.run(kind, round as u8, WAIT))
            .collect();
        eprintln!("STEPS-APART {name}: idle, {kind:?}: {}", spread(&mut times));
    }
    for released in [false, true] {
        for kind in KINDS {
            let before = device.work_counts();
            let started = Instant::now();
            // Released: the program lets go of A's command buffer and draws once it committed.
            let fresh = released.then(|| Draws::new(&scene, n));
            let stepping = fresh.as_ref().unwrap_or(&slow).encode(&a);
            let (done, completed) = mpsc::channel();
            stepping
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    let _ = done.send(());
                }))
                .unwrap();
            stepping.commit().unwrap();
            drop((fresh, stepping));
            let mut times = Vec::new();
            let mut round = 0u8;
            while completed.try_recv().is_err() && started.elapsed() < WAIT {
                if let Some(took) = other.run(kind, round, WAIT) {
                    times.push(took);
                }
                round = round.wrapping_add(1);
            }
            let frame = started.elapsed().as_secs_f64() * 1e3;
            let after = device.work_counts();
            let steps = after.host_steps_recorded - before.host_steps_recorded;
            let stepped = (after.host_step_nanos - before.host_step_nanos) as f64 / 1e6;
            eprintln!(
                "STEPS-APART {name}: during A's frame of {n} draws (released {released}; \
                 {frame:.1} ms; {steps} host steps recorded meanwhile, A's and B's, \
                 {stepped:.1} ms), {kind:?}: {}",
                spread(&mut times)
            );
        }
    }
}
