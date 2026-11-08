//! What committing a consumer's frames costs on the device the cfg selects, without the
//! validation layer: report only, in release builds, to weigh the runtime's commit path before and
//! after a change to it.
//!
//! The workload is a guest window server's, as the consumer (macOS guests over the runtime)
//! commits it: many small command buffers a frame, each with a few draws or blits, over a working
//! set of a few hundred live buffers and textures, with buffers and textures made and released
//! and views made between frames. Printed as `COMMIT-COST` lines:
//!
//! - **One command buffer at a time** (`serial`), each shape [`SERIAL`] times after
//!   [`WARM`] untimed: an empty command buffer; a blit encoder of two 16-KiB buffer copies and a
//!   64x64 texture copy; a render pass of three textured rectangles (`tob_rect`, `tob_sample`);
//!   the same pass sampling a `BGRA8Unorm` view of an `RGBA8Unorm` texture made without
//!   `PixelFormatView` usage (a shadowed view, 10.6) and through a view of the texture's own
//!   format; and a pointer walk (`walk-a-device-pointer-chain-through-a-loop-phi`), whose root
//!   table searches the device's live buffers (10.8). For each: the host's time encoding it, its
//!   `commit` call, the wall time from `commit` until `wait_until_completed` returns (the
//!   latency), its GPU time, and the process's CPU time over the shape's runs per command buffer
//!   (every thread: the program's, the runtime's and the driver's).
//! - **The bare driver** (`bare`): the same device and `VkQueue` (`CommandQueue::with_vk_queue`),
//!   one command buffer of the test's own a run, recorded, submitted with a fence and waited for
//!   (bounded, through `gpu/wait.rs`: one `vkWaitForFences` with a finite timeout), one at a
//!   time: a barrier alone, waited for at its fence and (`timeline wait`) at a timeline
//!   semaphore's value it signals, as the runtime's queue waiter waits for a batch; with GPU
//!   timestamps written as the runtime's timed command buffer writes them, their queries reset
//!   in the same command buffer (`timestamps reset`) or ahead in a submission of their own
//!   (`timestamps reset ahead`); a clear of a working-set texture (`Clear`: a render pass of its
//!   own on MoltenVK); and the blit shape's copies of the same bytes and texels; the last two
//!   also with timestamps around them, as the first and last encoder would write them. Its
//!   latency is the floor the runtime's adds to.
//! - **Plain Metal** (`metal_commit_costs`, on macOS): the same Apple GPU's own command buffers,
//!   committed and waited for one at a time (bounded, `gpu/wait.rs`): one with no encoder, one
//!   with an empty blit encoder, and one fill of a 16-KiB buffer, printed as `METAL-COST` lines:
//!   the latency from `commit` until the wait returned, and what Metal answers for
//!   `GPUStartTime` and `GPUEndTime` against `kernelStartTime` and `kernelEndTime`. And
//!   (`metal_encoderless_times`) what Metal answers for an encoder-less command buffer that cannot
//!   finish at once: committed behind a 256-MiB fill on its queue, and waiting for a shared event
//!   the host signals 2 ms after its commit, printed as `METAL-TIMES` lines.
//! - **Frames** (`frames`, `frames+walk`): [`FRAMES`] frames of [`PER_FRAME`] command buffers
//!   committed in a row and waited for at the frame's end, alternating render passes and blits;
//!   before each frame two buffers and a texture are released and made again and two views of
//!   another texture are made, which one pass each samples (the list of live buffers changes, so
//!   the first commit that needs the resident set after it makes one, 10.8). `+walk` adds a pointer
//!   walk to each frame. For each: the host's time encoding and committing per command buffer,
//!   the frame's wall time, and the process's CPU time per command buffer.
//! - **GPU times from the host** (`... (GPU times from the host)`): the serial shapes and the
//!   frames again on a device made with `gpu_times_from_host`, whose command buffers write no GPU
//!   timestamps, over a working set of its own. On a driver whose timestamps wait for every queue's
//!   (`DriverFact::TimestampsWaitForEveryQueue`: MoltenVK 1.4.1 on Apple's GPUs) both devices
//!   answer from the host's clock, and only the bare shapes with timestamps still write them.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system
//! allocator as a consumer's process does.

use super::{
    bound_buffers, encode_case, pipeline_state, public_case, validated_device_with, CaseResources,
    Messages, PublicCase, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::LoaderWord;
use crate::gpu::wait::{self, SubmissionBackend};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, Size, StoreAction, Texture, TextureDescriptor,
};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

/// The program's live buffers of the working set, each [`BUFFER_BYTES`] long.
const LIVE_BUFFERS: usize = 256;
const BUFFER_BYTES: u64 = 16 * 1024;
/// The working set's sampled textures, `RGBA8Unorm`, [`TEXTURE_SIDE`] texels a side.
const LIVE_TEXTURES: usize = 128;
pub(super) const TEXTURE_SIDE: u64 = 64;
/// The render targets the passes draw into, [`TARGET_SIDE`] texels a side.
const TARGETS: usize = 4;
const TARGET_SIDE: u64 = 256;
/// The untimed and the timed command buffers of each serial shape.
const WARM: usize = 20;
const SERIAL: usize = 200;
/// The frames of each frame run, and the command buffers of a frame.
const FRAMES: usize = 60;
const PER_FRAME: usize = 24;
/// The rectangles a pass draws.
const DRAWS: usize = 3;
/// The pointer walker, whose root table searches the device's live buffers.
const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

/// The process's CPU time so far, every thread's, in microseconds.
pub(super) fn process_cpu_us() -> f64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    let read = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut now) };
    assert_eq!(read, 0, "the process's CPU clock");
    now.tv_sec as f64 * 1e6 + now.tv_nsec as f64 / 1e3
}

/// Microseconds since `at`.
pub(super) fn since(at: Instant) -> f64 {
    at.elapsed().as_secs_f64() * 1e6
}

/// The 10th, 50th and 90th percentiles of `values`.
pub(super) fn percentiles(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    [at(10), at(50), at(90)]
}

/// One timed command buffer's times, in microseconds.
struct Sample {
    encode: f64,
    commit: f64,
    latency: f64,
    gpu: Option<f64>,
}

/// The shapes of one command buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    Empty,
    Blit,
    Draw,
    /// A pass sampling the shadowed `BGRA8Unorm` view (10.6).
    DrawShadowedView,
    /// A pass sampling a view of the texture's own format.
    DrawOwnFormatView,
    Walk,
}

/// The device, its queue, and what the shapes use.
pub(super) struct Workload {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    pipeline: RenderPipelineState,
    sampler: SamplerState,
    buffers: Vec<Buffer>,
    pub(super) textures: Vec<Texture>,
    targets: Vec<Texture>,
    /// A `BGRA8Unorm` view of an `RGBA8Unorm` texture without `PixelFormatView` usage (a shadow
    /// of its texels, 10.6), and an `RGBA8Unorm` one (the same image); each holds the texture.
    shadowed: Texture,
    own_format: Texture,
    walk: Walk,
    /// The next working-set member a command takes.
    next: usize,
}

/// The pointer walker's pipeline and resources.
struct Walk {
    case: PublicCase,
    state: ComputePipelineState,
    resources: CaseResources,
    buffers: BTreeMap<u32, Buffer>,
}

/// A rectangle (`tob_rect`'s corners) in each working-set buffer's first 16 bytes.
const RECT: [f32; 4] = [-0.5, -0.5, 0.5, 0.5];

impl Workload {
    pub(super) fn new(device: Device) -> Workload {
        let queue = device.new_command_queue().unwrap();
        let ll =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/tob_renders.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
        descriptor.fragment = Some(function("tob_sample"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut sampler = SamplerDescriptor::new();
        sampler.min_filter = SamplerMinMagFilter::Linear;
        sampler.mag_filter = SamplerMinMagFilter::Linear;
        let sampler = device.new_sampler_state(&sampler).unwrap();
        let viewed = texture(&device, TEXTURE_SIDE, TextureUsage::SHADER_READ);
        let shadowed = viewed.new_texture_view(PixelFormat::BGRA8Unorm).unwrap();
        let own_format = viewed.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
        let case = public_case(POINTER_WALKER).unwrap();
        let state = pipeline_state(&device, &case).unwrap();
        let resources = CaseResources::new(&device, &case).unwrap();
        let walk_buffers = bound_buffers(&device, &case.compute, &resources).unwrap();
        let rt = TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ);
        Workload {
            buffers: (0..LIVE_BUFFERS).map(|_| buffer(&device)).collect(),
            textures: (0..LIVE_TEXTURES)
                .map(|_| texture(&device, TEXTURE_SIDE, rt))
                .collect(),
            targets: (0..TARGETS)
                .map(|_| texture(&device, TARGET_SIDE, rt))
                .collect(),
            walk: Walk {
                case,
                state,
                resources,
                buffers: walk_buffers,
            },
            shadowed,
            own_format,
            pipeline,
            sampler,
            queue,
            device,
            next: 0,
        }
    }

    /// The next index into a working-set list of `len` members.
    pub(super) fn take(&mut self, len: usize) -> usize {
        self.next = self.next.wrapping_add(7);
        self.next % len
    }

    /// Encode `shape` into a new command buffer: the command buffer.
    pub(super) fn encode(&mut self, shape: Shape) -> CommandBuffer {
        let command_buffer = self.queue.command_buffer().unwrap();
        match shape {
            Shape::Empty => {}
            Shape::Blit => {
                let [a, b, c, d] = [0; 4].map(|_| self.take(self.buffers.len()));
                let [s, t] = [0; 2].map(|_| self.take(self.textures.len()));
                let t = if t == s {
                    (t + 1) % self.textures.len()
                } else {
                    t
                };
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                for (from, to) in [(a, b), (c, d)] {
                    let to = if to == from {
                        (to + 1) % self.buffers.len()
                    } else {
                        to
                    };
                    blit.copy_from_buffer_to_buffer(
                        &self.buffers[from],
                        0,
                        &self.buffers[to],
                        0,
                        BUFFER_BYTES,
                    )
                    .unwrap();
                }
                let at = Origin::new(0, 0, 0);
                let size = Size::new(TEXTURE_SIDE, TEXTURE_SIDE, 1);
                blit.copy_from_texture_to_texture(
                    &self.textures[s],
                    0,
                    0,
                    at,
                    size,
                    &self.textures[t],
                    0,
                    0,
                    at,
                )
                .unwrap();
                blit.end_encoding().unwrap();
            }
            Shape::Draw | Shape::DrawShadowedView | Shape::DrawOwnFormatView => {
                let target = self.take(TARGETS);
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(self.targets[target].clone());
                color.load_action = LoadAction::Load;
                color.store_action = StoreAction::Store;
                let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                encoder.set_render_pipeline_state(&self.pipeline).unwrap();
                encoder
                    .set_fragment_sampler_state(Some(&self.sampler), 0)
                    .unwrap();
                for draw in 0..DRAWS {
                    let vertices = self.take(self.buffers.len());
                    let sampled = match (shape, draw) {
                        (Shape::DrawShadowedView, 0) => self.shadowed.clone(),
                        (Shape::DrawOwnFormatView, 0) => self.own_format.clone(),
                        _ => {
                            let at = self.take(self.textures.len());
                            self.textures[at].clone()
                        }
                    };
                    encoder
                        .set_vertex_buffer(Some(&self.buffers[vertices]), 0, 0)
                        .unwrap();
                    encoder.set_fragment_texture(Some(&sampled), 0).unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                }
                encoder.end_encoding().unwrap();
            }
            Shape::Walk => {
                let walk = &self.walk;
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encode_case(
                    &mut encoder,
                    &walk.state,
                    &walk.case.compute,
                    &walk.buffers,
                    &walk.resources,
                )
                .unwrap();
                encoder.end_encoding().unwrap();
            }
        }
        command_buffer
    }

    /// One command buffer of `shape`, encoded, committed and waited for: its times.
    fn serial(&mut self, shape: Shape) -> Sample {
        let start = Instant::now();
        let command_buffer = self.encode(shape);
        let encode = since(start);
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        let commit = since(committed);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let latency = since(committed);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let gpu = match (
            command_buffer.gpu_start_time(),
            command_buffer.gpu_end_time(),
        ) {
            (Ok(start), Ok(end)) if end > start => Some((end - start) * 1e6),
            _ => None,
        };
        Sample {
            encode,
            commit,
            latency,
            gpu,
        }
    }

    /// Release two buffers and a texture of the working set and make them again, and make two
    /// views of another of its textures, which the frame's passes sample through: a
    /// `BGRA8Unorm` one (a shadow, 10.6) and one of the texture's own format.
    pub(super) fn churn(&mut self, frame: usize) {
        for at in [frame % LIVE_BUFFERS, (frame * 31 + 5) % LIVE_BUFFERS] {
            self.buffers[at] = buffer(&self.device);
        }
        let rt = TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ);
        let at = (frame * 13) % LIVE_TEXTURES;
        self.textures[at] = texture(&self.device, TEXTURE_SIDE, rt);
        let viewed = &self.textures[(frame * 17 + 3) % LIVE_TEXTURES];
        self.shadowed = viewed.new_texture_view(PixelFormat::BGRA8Unorm).unwrap();
        self.own_format = viewed.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
    }
}

/// A `Shared` working-set buffer holding [`RECT`] first.
fn buffer(device: &Device) -> Buffer {
    let buffer = device
        .new_buffer(BUFFER_BYTES, ResourceOptions::shared())
        .unwrap();
    let bytes: Vec<u8> = RECT.iter().flat_map(|v| v.to_le_bytes()).collect();
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds `bytes.len()` bytes at `contents`, and no command uses them yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// A `Private` square `RGBA8Unorm` texture of `usage`, `side` texels a side.
pub(super) fn texture(device: &Device, side: u64, usage: TextureUsage) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// The test's own command buffer on the runtime's device and `VkQueue`, for the bare driver's
/// submissions.
struct Bare {
    driver: ash::Device,
    word: LoaderWord,
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// A timeline semaphore each submission signals, at the next of `signaled`'s values.
    timeline: vk::Semaphore,
    signaled: Cell<u64>,
    /// Timestamp queries: the first two reset in each run that resets them, the rest
    /// ([`AHEAD`] runs' two each) reset once ahead ([`Bare::reset_ahead`]), the next of them at
    /// `ahead`.
    queries: vk::QueryPool,
    ahead: Cell<u32>,
}

/// The runs whose timestamps the bare driver resets ahead: [`WARM`] and [`SERIAL`] of each of
/// its three lines that write them so.
const AHEAD: u32 = 3 * (WARM + SERIAL) as u32;

/// What a bare submission records between its barriers.
#[derive(Clone, Copy)]
struct BareWork<'a> {
    /// Buffer copies of [`BUFFER_BYTES`]: a source and its offset, a destination and its.
    copies: &'a [(vk::Buffer, u64, vk::Buffer, u64)],
    /// Image copies of [`TEXTURE_SIDE`] texels a side, in `GENERAL`.
    images: &'a [(vk::Image, vk::Image)],
    /// Images cleared whole (`vkCmdClearColorImage`), in `GENERAL`: a render pass of its own on
    /// MoltenVK.
    clears: &'a [vk::Image],
}

/// The GPU timestamps a bare submission writes, as the runtime's timed command buffer does: at
/// the top of the pipe before its first barrier and at the bottom after its last.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BareStamps {
    /// None.
    None,
    /// Two queries, reset in the same command buffer first (`vkCmdResetQueryPool`), as the
    /// runtime's are.
    Reset,
    /// Two queries reset ahead, in a submission of their own.
    Ahead,
}

/// What a bare submission's latency waits for.
#[derive(Clone, Copy)]
enum BareWait {
    /// Its fence.
    Fence,
    /// Its timeline semaphore's value, as the runtime's queue waiter waits for a batch;
    /// then its fence, after the latency is taken.
    Timeline,
}

impl Bare {
    /// Made on `device` before it submitted anything, so the loader word is the device's own.
    fn new(device: &Device) -> Bare {
        // SAFETY: read while the device lives; the test's objects are destroyed in `drop`,
        // before the device.
        let raw = unsafe { device.raw() };
        let driver = raw.device.clone();
        // SAFETY: the handle is the live device's.
        let word = unsafe { LoaderWord::of_device(driver.handle()) };
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(raw.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the device is live; the pool is the test's own.
        let pool = unsafe { driver.create_command_pool(&info, None) }.unwrap();
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the device is live; the loader serves the call after the word is put back.
        let buffer = unsafe {
            word.restore();
            #[expect(
                clippy::disallowed_methods,
                reason = "the runtime's device, after the loader word is put back"
            )]
            let buffer = driver.allocate_command_buffers(&allocation);
            buffer
        }
        .unwrap()[0];
        // SAFETY: the device is live.
        let fence = unsafe { driver.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: the device is live; the runtime enables timeline semaphores.
        let timeline = unsafe { driver.create_semaphore(&info, None) }.unwrap();
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 + 2 * AHEAD);
        // SAFETY: the device is live.
        let queries = unsafe { driver.create_query_pool(&info, None) }.unwrap();
        Bare {
            driver,
            word,
            pool,
            buffer,
            fence,
            timeline,
            signaled: Cell::new(0),
            queries,
            ahead: Cell::new(0),
        }
    }

    /// Record a barrier, `copies` (buffer pairs of [`BUFFER_BYTES`], then image pairs of
    /// [`TEXTURE_SIDE`] texels a side, in `GENERAL`), and a barrier to the host; submit it to
    /// `queue`'s `VkQueue` with the fence and the timeline's next value, and wait for `wait`: the
    /// microseconds recording and from the submission until the wait returned.
    fn run(
        &self,
        queue: &CommandQueue,
        work: BareWork<'_>,
        wait: BareWait,
        stamps: BareStamps,
    ) -> (f64, f64) {
        let BareWork {
            copies: buffers,
            images,
            clears,
        } = work;
        let driver = &self.driver;
        let start = Instant::now();
        let all = vk::PipelineStageFlags::ALL_COMMANDS;
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ);
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = vk::ImageCopy::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D {
                width: TEXTURE_SIDE as u32,
                height: TEXTURE_SIDE as u32,
                depth: 1,
            });
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is the test's own and not pending (its last submission was
        // waited for); the buffers and images are live, the runtime's work on them completed,
        // and the images rest in `GENERAL`.
        unsafe {
            driver.begin_command_buffer(self.buffer, &begin).unwrap();
            let first = match stamps {
                BareStamps::None => None,
                BareStamps::Reset => {
                    driver.cmd_reset_query_pool(self.buffer, self.queries, 0, 2);
                    Some(0)
                }
                BareStamps::Ahead => {
                    let run = self.ahead.get();
                    assert!(run < AHEAD, "more runs than the queries reset ahead");
                    self.ahead.set(run + 1);
                    Some(2 + 2 * run)
                }
            };
            if let Some(first) = first {
                driver.cmd_write_timestamp(
                    self.buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries,
                    first,
                );
            }
            driver.cmd_pipeline_barrier(
                self.buffer,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            for &(from, from_offset, to, to_offset) in buffers {
                let copy = vk::BufferCopy {
                    src_offset: from_offset,
                    dst_offset: to_offset,
                    size: BUFFER_BYTES,
                };
                driver.cmd_copy_buffer(self.buffer, from, to, &[copy]);
            }
            for &(from, to) in images {
                driver.cmd_copy_image(
                    self.buffer,
                    from,
                    vk::ImageLayout::GENERAL,
                    to,
                    vk::ImageLayout::GENERAL,
                    &[region],
                );
            }
            let color = vk::ClearColorValue {
                float32: [0.25, 0.5, 0.75, 1.0],
            };
            let whole = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            for &image in clears {
                driver.cmd_clear_color_image(
                    self.buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    &color,
                    &[whole],
                );
            }
            driver.cmd_pipeline_barrier(
                self.buffer,
                all,
                vk::PipelineStageFlags::HOST | all,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
            if let Some(first) = first {
                driver.cmd_write_timestamp(
                    self.buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.queries,
                    first + 1,
                );
            }
            driver.end_command_buffer(self.buffer).unwrap();
        }
        let record = since(start);
        let submitted = Instant::now();
        let buffers = [self.buffer];
        let value = self.signaled.get() + 1;
        self.signaled.set(value);
        let (semaphores, values) = ([self.timeline], [value]);
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&buffers)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline);
        // SAFETY: the command buffer is executable, the fence unsignaled; the submission is
        // made under the runtime's lock of its `VkQueue`, and the call waits for nothing.
        unsafe {
            queue.with_vk_queue(|_, raw, _| {
                driver.queue_submit(raw, &[submit], self.fence).unwrap()
            });
        }
        // Bounded, as every wait of the crate's (`gpu/wait.rs`): one `vkWaitForFences` (or
        // `vkWaitSemaphores`) with the process's bound for a timeout, its result classified
        // after it returned, so the latency is the bare driver's.
        let backend = SubmissionBackend::vulkan(TEST_DEVICE);
        let operation = "the commit-cost lane's bare submission";
        let fence = || wait::vulkan::wait_for_fence(driver, self.fence, backend, operation);
        let latency = match wait {
            BareWait::Fence => {
                fence().unwrap();
                since(submitted)
            }
            BareWait::Timeline => {
                wait::vulkan::wait_for_timeline(driver, self.timeline, value, backend, operation)
                    .unwrap();
                let latency = since(submitted);
                fence().unwrap();
                latency
            }
        };
        // SAFETY: the fence is signaled and its submission completed.
        unsafe {
            driver.reset_fences(&[self.fence]).unwrap();
            driver
                .reset_command_buffer(self.buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }
        (record, latency)
    }
}

impl Bare {
    /// Reset the queries [`BareStamps::Ahead`] runs write, in one submission to `queue`'s
    /// `VkQueue`, waited for.
    fn reset_ahead(&self, queue: &CommandQueue) {
        let driver = &self.driver;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is the test's own and not pending; the queries are in no
        // pending submission.
        unsafe {
            driver.begin_command_buffer(self.buffer, &begin).unwrap();
            driver.cmd_reset_query_pool(self.buffer, self.queries, 2, 2 * AHEAD);
            driver.end_command_buffer(self.buffer).unwrap();
        }
        let buffers = [self.buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        // SAFETY: as `run`'s submission.
        unsafe {
            queue.with_vk_queue(|_, raw, _| {
                driver.queue_submit(raw, &[submit], self.fence).unwrap()
            });
        }
        let backend = SubmissionBackend::vulkan(TEST_DEVICE);
        wait::vulkan::wait_for_fence(driver, self.fence, backend, "the bare queries' reset")
            .unwrap();
        // SAFETY: the fence is signaled and its submission completed.
        unsafe {
            driver.reset_fences(&[self.fence]).unwrap();
            driver
                .reset_command_buffer(self.buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }
    }
}

impl Drop for Bare {
    fn drop(&mut self) {
        // SAFETY: nothing of the test's is pending (each submission was waited for).
        unsafe {
            self.word.restore();
            self.driver.destroy_fence(self.fence, None);
            self.driver.destroy_semaphore(self.timeline, None);
            self.driver.destroy_query_pool(self.queries, None);
            self.driver.destroy_command_pool(self.pool, None);
        }
    }
}

/// Print one serial or bare line: `what`'s samples.
fn print_serial(device: &str, what: &str, samples: &[Sample], cpu: f64) {
    let [_, encode, encode_90] = percentiles(samples.iter().map(|s| s.encode).collect());
    let [_, commit, commit_90] = percentiles(samples.iter().map(|s| s.commit).collect());
    let [low, latency, high] = percentiles(samples.iter().map(|s| s.latency).collect());
    let gpus: Vec<f64> = samples.iter().filter_map(|s| s.gpu).collect();
    let gpu = match gpus.is_empty() {
        true => "none".to_owned(),
        false => format!("{:.1}", percentiles(gpus)[1]),
    };
    eprintln!(
        "COMMIT-COST {device}: {what}: encode {encode:.1} (p90 {encode_90:.1}) us, commit \
         {commit:.1} (p90 {commit_90:.1}) us, latency {latency:.1} (p10 {low:.1}, p90 {high:.1}) \
         us, gpu {gpu} us, process cpu {cpu:.1} us per command buffer"
    );
}

/// The commit path's costs, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn commit_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let bare = Bare::new(&device);
    let name = device.name();
    let mut work = Workload::new(device.clone());
    eprintln!(
        "COMMIT-COST {name}: working set {LIVE_BUFFERS} buffers of {BUFFER_BYTES} bytes, \
         {LIVE_TEXTURES} textures {TEXTURE_SIDE}x{TEXTURE_SIDE}, {TARGETS} targets \
         {TARGET_SIDE}x{TARGET_SIDE}; {SERIAL} serial command buffers a shape after {WARM}; \
         {FRAMES} frames of {PER_FRAME}"
    );
    serial_shapes(&mut work, "");
    // The bare driver: a barrier alone, and the blit shape's copies, between buffers and
    // textures of the working set the runtime's work on which completed.
    // SAFETY: each lives until the end of the test, and the runtime's work on it completed.
    let handle = |buffer: &Buffer| unsafe { buffer.vk_buffer() };
    // SAFETY: as above; runtime-made images, which rest in `GENERAL`.
    let image = |texture: &Texture| unsafe { texture.vk_image() }.expect("an image of its own");
    let (a, b, c, d) = (
        handle(&work.buffers[0]),
        handle(&work.buffers[1]),
        handle(&work.buffers[2]),
        handle(&work.buffers[3]),
    );
    let copies = [(a.0, a.1, b.0, b.1), (c.0, c.1, d.0, d.1)];
    let images = [(image(&work.textures[0]), image(&work.textures[1]))];
    bare.reset_ahead(&work.queue);
    let (fence, timeline) = (BareWait::Fence, BareWait::Timeline);
    let nothing = BareWork {
        copies: &[],
        images: &[],
        clears: &[],
    };
    let blit = BareWork {
        copies: &copies,
        images: &images,
        clears: &[],
    };
    let cleared = [image(&work.textures[2])];
    let clear = BareWork {
        clears: &cleared,
        ..nothing
    };
    let (none, reset, ahead) = (BareStamps::None, BareStamps::Reset, BareStamps::Ahead);
    for (what, bare_work, wait, stamps) in [
        ("bare barrier", nothing, fence, none),
        ("bare barrier, timeline wait", nothing, timeline, none),
        ("bare barrier, timestamps reset", nothing, fence, reset),
        (
            "bare barrier, timestamps reset ahead",
            nothing,
            fence,
            ahead,
        ),
        ("bare Clear", clear, fence, none),
        ("bare Clear, timestamps reset ahead", clear, fence, ahead),
        ("bare Blit", blit, fence, none),
        ("bare Blit, timestamps reset ahead", blit, fence, ahead),
    ] {
        for _ in 0..WARM {
            bare.run(&work.queue, bare_work, wait, stamps);
        }
        let cpu = process_cpu_us();
        let samples: Vec<Sample> = (0..SERIAL)
            .map(|_| {
                let (record, latency) = bare.run(&work.queue, bare_work, wait, stamps);
                Sample {
                    encode: record,
                    commit: 0.0,
                    latency,
                    gpu: None,
                }
            })
            .collect();
        let cpu = (process_cpu_us() - cpu) / SERIAL as f64;
        print_serial(name, what, &samples, cpu);
    }
    frames(&mut work, "");
    drop(work);
    drop(bare);
    // The same on a device that answers GPU times from the host: no timestamps.
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
        options.gpu_times_from_host = true;
    })
    .unwrap();
    let mut work = Workload::new(device);
    serial_shapes(&mut work, " (GPU times from the host)");
    frames(&mut work, " (GPU times from the host)");
}

/// Each serial shape on `work`'s device, [`SERIAL`] command buffers after [`WARM`], printed with
/// `label` after its name.
fn serial_shapes(work: &mut Workload, label: &str) {
    let name = work.device.name().to_owned();
    let shapes = [
        Shape::Empty,
        Shape::Blit,
        Shape::Draw,
        Shape::DrawOwnFormatView,
        Shape::DrawShadowedView,
        Shape::Walk,
    ];
    for shape in shapes {
        for _ in 0..WARM {
            work.serial(shape);
        }
        let cpu = process_cpu_us();
        let samples: Vec<Sample> = (0..SERIAL).map(|_| work.serial(shape)).collect();
        let cpu = (process_cpu_us() - cpu) / SERIAL as f64;
        print_serial(&name, &format!("serial {shape:?}{label}"), &samples, cpu);
    }
}

/// [`FRAMES`] frames on `work`'s device, without and with a pointer walk, printed with `label`
/// after their name.
fn frames(work: &mut Workload, label: &str) {
    let name = work.device.name().to_owned();
    // Frames: render passes and blits alternating, one pass through the shadowed view; churn
    // before each frame.
    for walk in [false, true] {
        let what = if walk { "frames+walk" } else { "frames" };
        let what = format!("{what}{label}");
        let mut encode = Vec::new();
        let mut commit = Vec::new();
        let mut walls = Vec::new();
        let cpu = process_cpu_us();
        for frame in 0..FRAMES {
            work.churn(frame);
            let begin = Instant::now();
            let mut last = None;
            for at in 0..PER_FRAME {
                let shape = match at {
                    0 => Shape::DrawShadowedView,
                    1 if walk => Shape::Walk,
                    2 => Shape::DrawOwnFormatView,
                    _ if at % 2 == 0 => Shape::Draw,
                    _ => Shape::Blit,
                };
                let start = Instant::now();
                let command_buffer = work.encode(shape);
                encode.push(since(start));
                let committed = Instant::now();
                command_buffer.commit().unwrap();
                commit.push(since(committed));
                last = Some(command_buffer);
            }
            let last = last.expect("a frame's command buffers");
            last.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(last.status(), CommandBufferStatus::Completed);
            walls.push(since(begin));
        }
        let count = (FRAMES * PER_FRAME) as f64;
        let cpu = (process_cpu_us() - cpu) / count;
        let [_, encode, encode_90] = percentiles(encode);
        let [_, commit, commit_90] = percentiles(commit);
        let [low, wall, high] = percentiles(walls);
        eprintln!(
            "COMMIT-COST {name}: {what}: encode {encode:.1} (p90 {encode_90:.1}) us, commit \
             {commit:.1} (p90 {commit_90:.1}) us per command buffer; frame {wall:.0} (p10 \
             {low:.0}, p90 {high:.0}) us, {:.1} us per command buffer; process cpu {cpu:.1} us \
             per command buffer",
            wall / PER_FRAME as f64
        );
    }
}

/// Plain Metal's commit path on this Mac's GPU, printed (the module docs): what the runtime's
/// latency compares to, and what Metal answers as an encoder-less command buffer's GPU times.
#[cfg(all(m2v_gpu, target_os = "macos"))]
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn metal_commit_costs() {
    use objc2_foundation::NSRange;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice,
        MTLResourceOptions,
    };
    let device = crate::metal::system_device().unwrap();
    let queue = device.newCommandQueue().unwrap();
    let buffer = device
        .newBufferWithLength_options(BUFFER_BYTES as usize, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let name = device.name().to_string();
    for shape in ["no encoder", "an empty blit encoder", "one fill"] {
        // Per command buffer, in microseconds: the latency, GPUEndTime less GPUStartTime,
        // GPUStartTime less kernelStartTime, and GPUEndTime less kernelEndTime.
        let mut rows: Vec<[f64; 4]> = Vec::new();
        let cpu = process_cpu_us();
        for run in 0..WARM + SERIAL {
            let command_buffer = queue.commandBuffer().unwrap();
            if shape != "no encoder" {
                let blit = command_buffer.blitCommandEncoder().unwrap();
                if shape == "one fill" {
                    let range = NSRange::new(0, BUFFER_BYTES as usize);
                    blit.fillBuffer_range_value(&buffer, range, 0x5a);
                }
                blit.endEncoding();
            }
            let committed = Instant::now();
            wait::metal::commit_and_wait(&command_buffer, "the Metal commit-cost probe").unwrap();
            let latency = since(committed);
            let (start, end) = (command_buffer.GPUStartTime(), command_buffer.GPUEndTime());
            let (kernel_start, kernel_end) = (
                command_buffer.kernelStartTime(),
                command_buffer.kernelEndTime(),
            );
            if run >= WARM {
                rows.push([
                    latency,
                    (end - start) * 1e6,
                    (start - kernel_start) * 1e6,
                    (end - kernel_end) * 1e6,
                ]);
            }
        }
        let cpu = (process_cpu_us() - cpu) / (WARM + SERIAL) as f64;
        let column = |at: usize| percentiles(rows.iter().map(|row| row[at]).collect());
        let [low, latency, high] = column(0);
        let [gpu_low, gpu, gpu_high] = column(1);
        let [_, after_kernel_start, _] = column(2);
        let [_, after_kernel_end, _] = column(3);
        let zero = rows.iter().filter(|row| row[1] == 0.0).count();
        eprintln!(
            "METAL-COST {name}: {shape}: latency {latency:.1} (p10 {low:.1}, p90 {high:.1}) us, \
             GPUEndTime-GPUStartTime {gpu:.2} (p10 {gpu_low:.2}, p90 {gpu_high:.2}) us, {zero} of \
             {SERIAL} zero, GPUStartTime-kernelStartTime {after_kernel_start:.1} us, \
             GPUEndTime-kernelEndTime {after_kernel_end:.1} us, process cpu {cpu:.1} us per \
             command buffer"
        );
    }
}

/// What plain Metal answers as an encoder-less command buffer's times where it cannot finish at
/// once (the module docs), printed: the rule the runtime's answer for one keeps.
#[cfg(all(m2v_gpu, target_os = "macos"))]
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn metal_encoderless_times() {
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSRange;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
        MTLCommandQueue, MTLDevice, MTLResourceOptions, MTLSharedEvent,
    };
    const RUNS: usize = 50;
    const BIG: usize = 256 << 20;
    let device = crate::metal::system_device().unwrap();
    let queue = device.newCommandQueue().unwrap();
    let name = device.name().to_string();
    let us = |seconds: f64| seconds * 1e6;
    let print = |shape: &str, columns: &[&str], rows: &[Vec<f64>], extra: String| {
        let line: Vec<String> = columns
            .iter()
            .enumerate()
            .map(|(at, column)| {
                let [low, middle, high] = percentiles(rows.iter().map(|row| row[at]).collect());
                format!("{column} {middle:.1} (p10 {low:.1}, p90 {high:.1}) us")
            })
            .collect();
        eprintln!("METAL-TIMES {name}: {shape}: {}{extra}", line.join(", "));
    };
    // Behind a fill of a private buffer on the same queue.
    let big = device
        .newBufferWithLength_options(BIG, MTLResourceOptions::StorageModePrivate)
        .unwrap();
    let mut rows = Vec::new();
    let mut fill_done = 0;
    for _ in 0..RUNS {
        let fill = queue.commandBuffer().unwrap();
        let blit = fill.blitCommandEncoder().unwrap();
        blit.fillBuffer_range_value(&big, NSRange::new(0, BIG), 0x5a);
        blit.endEncoding();
        let fill_watch = wait::metal::watch(&fill);
        fill.commit();
        let empty = queue.commandBuffer().unwrap();
        wait::metal::commit_and_wait(&empty, "an encoder-less command buffer behind a fill")
            .unwrap();
        fill_done += usize::from(fill.status() == MTLCommandBufferStatus::Completed);
        wait::metal::settle_watched(&fill, fill_watch, "a fill ahead of an empty command buffer")
            .unwrap();
        let fill_end = fill.GPUEndTime();
        rows.push(vec![
            us(empty.kernelStartTime() - fill_end),
            us(empty.kernelEndTime() - fill_end),
            us(empty.GPUStartTime() - fill_end),
            us(empty.GPUEndTime() - fill_end),
            us(empty.GPUStartTime() - empty.kernelEndTime()),
            us(fill_end - fill.GPUStartTime()),
        ]);
    }
    print(
        "no encoder behind a 256-MiB fill",
        &[
            "kernelStartTime-fill's GPUEndTime",
            "kernelEndTime-fill's GPUEndTime",
            "GPUStartTime-fill's GPUEndTime",
            "GPUEndTime-fill's GPUEndTime",
            "GPUStartTime-kernelEndTime",
            "the fill's GPU time",
        ],
        &rows,
        format!(", the fill completed when its wait returned in {fill_done} of {RUNS}"),
    );
    // Waiting for a shared event the host signals 2 ms after the commit.
    let event = device.newSharedEvent().unwrap();
    let mut rows = Vec::new();
    for value in 1..=RUNS as u64 {
        let waiting = queue.commandBuffer().unwrap();
        waiting.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), value);
        let watch = wait::metal::watch(&waiting);
        let committed = metal2vulkan_runtime::host_time();
        waiting.commit();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let signalled = metal2vulkan_runtime::host_time();
        event.setSignaledValue(value);
        wait::metal::settle_watched(&waiting, watch, "an encoder-less command buffer's wait")
            .unwrap();
        rows.push(vec![
            us(waiting.kernelStartTime() - committed),
            us(waiting.kernelStartTime() - signalled),
            us(waiting.kernelEndTime() - signalled),
            us(waiting.GPUStartTime() - signalled),
            us(waiting.GPUEndTime() - waiting.GPUStartTime()),
        ]);
    }
    print(
        "no encoder waiting for an event the host signals 2 ms after the commit",
        &[
            "kernelStartTime-commit",
            "kernelStartTime-signal",
            "kernelEndTime-signal",
            "GPUStartTime-signal",
            "GPUEndTime-GPUStartTime",
        ],
        &rows,
        String::new(),
    );
}
