//! Command buffers of one queue encoded on several threads at once, then committed in a fixed
//! order (the consumer's host encodes a queue's command buffers so): Metal runs a queue's command
//! buffers in commit order, whatever order they were made and encoded in, and orders a tracked
//! resource's accesses, and a fence, in that order. Each test runs one sequence (`program.rs`),
//! every command buffer on one queue, serially (made, encoded and committed one after another)
//! and in parallel under a seeded [`Schedule`], on the device the cfg selects under the
//! validation layer, and requires of every run:
//!
//! - **The bytes** of every buffer, heap window and texture slice: the CPU model's (`model.rs`,
//!   which runs the commands in commit order), byte for byte, the serial run's too.
//! - **No layer message** in either run. With `METAL2VULKAN_LANE_VALIDATION=sync` the layer checks
//!   the batches' synchronization, so a barrier the parallel run's encode order left out fails it.
//! - **The same notices.** The kinds of degrade notice and precision note the parallel runs'
//!   device announced are the serial runs' device's: each runs the sequence the same number of
//!   times, each on a device of its own (a device announces a kind once), the pipelines made and
//!   the pool settled first on both.
//!
//! **The schedules.** Every command buffer is encoded by one of `threads` threads, started
//! together at a barrier; a seeded generator (the schedule's seed, printed with any failure)
//! picks, per run:
//! - **who encodes what and when** ([`Order`]): `Reversed`, each command buffer encoded only once
//!   the one committed after it was encoded whole, on another thread (a baton passed down), so a
//!   later-committed command buffer is always encoded first and must still see every earlier one's
//!   writes; `Lockstep`, every thread making one encoding call in its turn (a turnstile), so the
//!   encoders of different command buffers interleave call by call; and `Spread`, each thread
//!   given command buffers at random, in a random order, with a random pause (none, a yield, a
//!   spin or a sleep of up to 50 µs) before each call;
//! - **where they are made** ([`Made`]): on the committing thread in commit order or in reverse,
//!   or each on the thread that encodes it;
//! - **when they commit** ([`Commits`]): all once every one was encoded, or each as soon as it
//!   and every earlier one were (the committing thread waits for them in turn), while later ones
//!   are still being encoded.
//!
//! **The sequences.** [`chained`] is written for these tests: each command buffer reads what the
//! one committed before it wrote and overwrites what that one read (read-after-write and
//! write-after-read across command buffers through tracked buffers and a tracked texture), its
//! encoders blits, serial and concurrent compute and render passes in turn, with an untracked
//! buffer ordered across command buffers by a fence alone (Metal evaluates a fence at
//! submission, so the wait follows the update in commit order). [`merged`] is several seeds'
//! generated sequences on one queue, their command buffers interleaved (each keeps its own
//! order): heaps whose buffers overlap, untracked resources ordered by events and fences, views,
//! indirect commands; race-free, as `order::problems` checks.

use super::generate::{generate, Rng};
use super::model::{byte_differences, readback_records, run as model_run, State};
use super::order::problems;
use super::program::{
    At, BufferSpec, Command, CommandBufferSpec, Encoder, Format, Kernel, Kind, Load, Place,
    Program, Role, Step, Surface, TextureSpec, ViewForm, BUFFER_BYTES, HEAP_WINDOW, RECORD_BYTES,
    RECT_VERTICES, ROW_BYTES, TARGET, TEXEL_READ, TEXEL_WRITE, TEXTURE_BYTES,
};
use super::run::{pixel_format, swizzle};
use super::FIXTURE;
use crate::air_library::AirLibrary as _;
use crate::gpu::TEST_DEVICE;
use crate::parity::record::Value;
use crate::runtime_lane::{validated_device_with, Messages, ScratchCache};
use metal2vulkan::metal::{
    BarrierScope, DispatchType, PixelFormat, PrimitiveType, RenderStages, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue,
    ComputeCommandEncoder, ComputePipelineState, CpuCacheMode, Degradation, Device, EventRef,
    Fence, HazardTrackingMode, Heap, HeapDescriptor, HeapType, HostMemory, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SharedEvent, Size,
    StorageMode, StoreAction, Texture, TextureDescriptor,
};
use std::alloc::Layout;
use std::collections::BTreeSet;
use std::ptr::NonNull;
use std::sync::{Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The longest a command buffer of these tests may take, and the longest a thread waits for its
/// turn before it goes on without it.
const WAIT: Duration = Duration::from_secs(60);

/// The longest a [`Order::Lockstep`] thread waits for its turn: past it, it goes on (a call that
/// blocks on another thread's work would otherwise hold the turnstile).
const TURN: Duration = Duration::from_millis(200);

/// The size of a heap of buffers, and of a texture's own heap (`run.rs`'s).
const HEAP_BYTES: u64 = 16 << 10;
const TEXTURE_HEAP_BYTES: u64 = 1 << 20;

/// One host page a buffer over host memory wraps (`run.rs`'s).
const HOST_PAGE: usize = 16 << 10;

/// Who encodes which command buffer, and in what order (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    Reversed,
    Lockstep,
    Spread,
}

/// Where the command buffers are made (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Made {
    InOrder,
    Reversed,
    OnWorkers,
}

/// When the command buffers commit (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Commits {
    AfterAll,
    AsReady,
}

/// A parallel run's schedule (the module docs); `threads` of 1 with [`Order::Spread`] and
/// [`Made::InOrder`] is never one: the serial run makes, encodes and commits each in turn.
#[derive(Clone, Copy, Debug)]
struct Schedule {
    threads: usize,
    order: Order,
    made: Made,
    commits: Commits,
    seed: u64,
}

impl Schedule {
    /// Run `run`'s schedule on `threads` threads: its order by `run`, the rest by `seed`.
    fn of(threads: usize, run: u64, seed: u64) -> Schedule {
        let mut rng = Rng::new(seed);
        let order = [Order::Reversed, Order::Lockstep, Order::Spread][(run % 3) as usize];
        let made = [Made::InOrder, Made::Reversed, Made::OnWorkers][rng.below(3) as usize];
        // A reversed order commits once every command buffer was encoded: the first committed
        // is the last encoded.
        let commits = match order {
            Order::Reversed => Commits::AfterAll,
            _ if rng.chance(50) => Commits::AsReady,
            _ => Commits::AfterAll,
        };
        Schedule {
            threads,
            order,
            made,
            commits,
            seed,
        }
    }
}

/// Host pages, one [`HOST_PAGE`] each, zeroed and aligned to it, for a sequence's buffers over
/// host memory; kept until the device that used them is gone.
struct HostPages {
    pages: Vec<NonNull<u8>>,
    layout: Layout,
}

// SAFETY: the pages are plain host memory the runner owns; threads reach them only through the
// buffers that wrap them, as the runtime's contract for host memory allows.
unsafe impl Send for HostPages {}
// SAFETY: as above; the runner writes and reads them only while no command buffer uses them.
unsafe impl Sync for HostPages {}

impl HostPages {
    fn new() -> HostPages {
        HostPages {
            pages: Vec::new(),
            layout: Layout::from_size_align(HOST_PAGE, HOST_PAGE).expect("a page's layout"),
        }
    }

    /// A new page holding `bytes` at its start.
    fn page(&mut self, bytes: &[u8]) -> NonNull<u8> {
        assert!(bytes.len() <= HOST_PAGE);
        // SAFETY: the layout's size is not zero.
        let page = NonNull::new(unsafe { std::alloc::alloc_zeroed(self.layout) }).expect("a page");
        // SAFETY: the page holds `bytes`; nothing uses it yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), page.as_ptr(), bytes.len()) };
        self.pages.push(page);
        page
    }
}

impl Drop for HostPages {
    fn drop(&mut self) {
        for page in self.pages.drain(..) {
            // SAFETY: allocated with this layout in `page`.
            unsafe { std::alloc::dealloc(page.as_ptr(), self.layout) };
        }
    }
}

/// A device for the sequences, with the fixture's pipelines, its queue, and its notices.
struct Lane {
    device: Device,
    queue: CommandQueue,
    copy: ComputePipelineState,
    mix: ComputePipelineState,
    texel_read: ComputePipelineState,
    texel_write: ComputePipelineState,
    draw_rgba: RenderPipelineState,
    draw_bgra: RenderPipelineState,
    messages: Messages,
}

impl Lane {
    /// A lane device keeping its shader cache in `cache`, its pipelines made and its pool settled.
    fn open(cache: &ScratchCache) -> Lane {
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let ll = engine_dir().join(format!("{FIXTURE}.ll"));
        let bitcode = llvm_assemble(&ll).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let kernel = |entry: &str| {
            let function = library.new_function_with_name(entry).unwrap();
            device
                .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
                .unwrap()
        };
        let draw = |format: PixelFormat| {
            let function = |name| library.new_function_with_name(name).unwrap();
            let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
                function("seq_rect").function_descriptor(),
            ));
            descriptor.fragment = Some(StageSource::Function(
                function("seq_color").function_descriptor(),
            ));
            descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let lane = Lane {
            copy: kernel(Kernel::Copy.entry()),
            mix: kernel(Kernel::Mix.entry()),
            texel_read: kernel(TEXEL_READ),
            texel_write: kernel(TEXEL_WRITE),
            draw_rgba: draw(pixel_format(Format::Rgba8Unorm)),
            draw_bgra: draw(pixel_format(Format::Bgra8Unorm)),
            queue: device.new_command_queue().unwrap(),
            device,
            messages,
        };
        let until = Instant::now() + WAIT;
        while lane.device.work_counts().pipeline_pool_busy != 0 {
            assert!(Instant::now() < until, "the pipeline pool still busy");
            std::thread::sleep(Duration::from_millis(1));
        }
        lane
    }
}

/// The resources of one run of a sequence, made on a lane's device, their first bytes in.
struct Objects<'a> {
    lane: &'a Lane,
    program: &'a Program,
    windows: Vec<Buffer>,
    buffers: Vec<Buffer>,
    textures: Vec<Texture>,
    views: Vec<Texture>,
    events: Vec<SharedEvent>,
    fences: Vec<Fence>,
    /// Kept for their resources' lives.
    _heaps: Vec<Heap>,
    /// The bytes of each buffer over host memory, by buffer.
    pages: Vec<Option<NonNull<u8>>>,
}

// SAFETY: every handle is the runtime's (`Send + Sync`); the page pointers are read only after
// every command buffer completed (`HostPages`).
unsafe impl Send for Objects<'_> {}
// SAFETY: as above.
unsafe impl Sync for Objects<'_> {}

fn options(private: bool, tracked: bool) -> ResourceOptions {
    ResourceOptions::new(
        if private {
            StorageMode::Private
        } else {
            StorageMode::Shared
        },
        CpuCacheMode::DefaultCache,
        if tracked {
            HazardTrackingMode::Tracked
        } else {
            HazardTrackingMode::Untracked
        },
    )
}

fn texture_descriptor(spec: &TextureSpec) -> TextureDescriptor {
    let side = u64::from(TARGET);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
    if spec.slices > 1 {
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = u64::from(spec.slices);
    }
    descriptor.usage = TextureUsage::RENDER_TARGET
        | TextureUsage::SHADER_READ
        | TextureUsage::SHADER_WRITE
        | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = options(true, spec.tracked);
    descriptor
}

/// A `Shared` buffer holding `bytes`.
fn shared_with(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    write(&buffer, bytes);
    buffer
}

fn write(buffer: &Buffer, bytes: &[u8]) {
    let contents = buffer.contents().expect("a Shared buffer");
    assert!(bytes.len() as u64 <= buffer.length());
    // SAFETY: the mapping covers the buffer's length; no command buffer uses it now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a Shared buffer");
    let length = usize::try_from(buffer.length()).unwrap();
    // SAFETY: the mapping covers the buffer's length; the command buffers that wrote it completed.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec()
}

/// Wait for `command_buffer`, which must complete without an error.
fn completed(command_buffer: &CommandBuffer, what: &str) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None),
        "{what}"
    );
}

/// The copy's texels: whole rows of a 2D texture.
fn rows_region(rows: (u32, u32)) -> (Origin, Size) {
    (
        Origin::new(0, u64::from(rows.0), 0),
        Size::new(u64::from(TARGET), u64::from(rows.1 - rows.0), 1),
    )
}

impl<'a> Objects<'a> {
    /// `program`'s resources on `lane`'s device, their first bytes copied in by a setup command
    /// buffer, completed.
    fn make(lane: &'a Lane, program: &'a Program, pages: &mut HostPages) -> Objects<'a> {
        let device = &lane.device;
        let mut staged: Vec<(Buffer, Staged)> = Vec::new();
        let mut heaps = Vec::new();
        let mut windows = Vec::new();
        for (index, spec) in program.heaps.iter().enumerate() {
            let heap_options = options(spec.private, spec.tracked);
            let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP_BYTES, heap_options);
            let heap = device.new_heap(&descriptor).unwrap();
            let window = heap
                .new_buffer_at_offset(HEAP_WINDOW, heap_options, 0)
                .unwrap();
            if spec.private {
                staged.push((shared_with(device, &spec.initial), Staged::Heap(index)));
            } else {
                write(&window, &spec.initial);
            }
            heaps.push(heap);
            windows.push(window);
        }
        let mut buffers = Vec::new();
        let mut page_of = Vec::new();
        for (index, spec) in program.buffers.iter().enumerate() {
            let buffer_options = options(spec.private, spec.tracked);
            let (buffer, page) = match spec.place {
                Place::Own => {
                    let buffer = device.new_buffer(BUFFER_BYTES, buffer_options).unwrap();
                    if spec.private {
                        staged.push((shared_with(device, &spec.initial), Staged::Buffer(index)));
                    } else {
                        write(&buffer, &spec.initial);
                    }
                    (buffer, None)
                }
                Place::NoCopy => {
                    let page = pages.page(&spec.initial);
                    // SAFETY: the page stays mapped, readable and writable until the device is
                    // gone (`HostPages` outlives the lane).
                    let buffer = unsafe {
                        device.new_buffer_with_bytes_no_copy(
                            HostMemory::new(page),
                            BUFFER_BYTES,
                            buffer_options,
                            None,
                        )
                    }
                    .unwrap();
                    (buffer, Some(page))
                }
                Place::Heap { heap, offset } => (
                    heaps[heap]
                        .new_buffer_at_offset(BUFFER_BYTES, buffer_options, offset)
                        .unwrap(),
                    None,
                ),
            };
            buffers.push(buffer);
            page_of.push(page);
        }
        let mut textures = Vec::new();
        for (index, spec) in program.textures.iter().enumerate() {
            let descriptor = texture_descriptor(spec);
            let texture = if spec.heap {
                let heap = HeapDescriptor::new(
                    HeapType::Placement,
                    TEXTURE_HEAP_BYTES,
                    descriptor.resource_options,
                );
                let heap = device.new_heap(&heap).unwrap();
                let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
                heaps.push(heap);
                texture
            } else {
                device.new_texture(&descriptor).unwrap()
            };
            for slice in 0..spec.slices {
                let start = (u64::from(slice) * TEXTURE_BYTES) as usize;
                let bytes = &spec.initial[start..start + TEXTURE_BYTES as usize];
                staged.push((shared_with(device, bytes), Staged::Slice(index, slice)));
            }
            textures.push(texture);
        }
        let views = (program.views.iter())
            .map(|view| {
                let texture = &textures[view.texture];
                let format = pixel_format(view.format);
                let slices = u64::from(view.slice)..u64::from(view.slice) + 1;
                match view.form {
                    ViewForm::Whole => texture.new_texture_view(format),
                    ViewForm::Slices => texture.new_texture_view_with_levels_slices(
                        format,
                        TextureType::Type2D,
                        0..1,
                        slices,
                    ),
                    ViewForm::Swizzled(channels) => texture.new_texture_view_with_swizzle(
                        format,
                        TextureType::Type2D,
                        0..1,
                        slices,
                        swizzle(channels),
                    ),
                }
                .unwrap()
            })
            .collect();
        let setup = lane.queue.command_buffer().unwrap();
        let mut blit = setup.blit_command_encoder().unwrap();
        for (staging, what) in &staged {
            match *what {
                Staged::Buffer(index) => blit
                    .copy_from_buffer_to_buffer(staging, 0, &buffers[index], 0, BUFFER_BYTES)
                    .unwrap(),
                Staged::Heap(index) => blit
                    .copy_from_buffer_to_buffer(staging, 0, &windows[index], 0, HEAP_WINDOW)
                    .unwrap(),
                Staged::Slice(index, slice) => blit
                    .copy_from_buffer_to_texture(
                        staging,
                        0,
                        ROW_BYTES,
                        TEXTURE_BYTES,
                        Size::new(u64::from(TARGET), u64::from(TARGET), 1),
                        &textures[index],
                        u64::from(slice),
                        0,
                        Origin::new(0, 0, 0),
                        BlitOption::None,
                    )
                    .unwrap(),
            }
        }
        blit.end_encoding().unwrap();
        setup.commit().unwrap();
        completed(&setup, "the setup");
        Objects {
            lane,
            program,
            windows,
            buffers,
            textures,
            views,
            events: (0..program.events)
                .map(|_| device.new_shared_event().unwrap())
                .collect(),
            fences: (0..program.fences).map(|_| device.new_fence()).collect(),
            _heaps: heaps,
            pages: page_of,
        }
    }

    fn at(&self, at: At) -> (&Buffer, u64) {
        (&self.buffers[at.buffer], at.offset)
    }

    fn surface(&self, surface: Surface) -> &Texture {
        match surface {
            Surface::Texture(index) => &self.textures[index],
            Surface::View(index) => &self.views[index],
        }
    }

    /// Encode command buffer `index` of the sequence into `cb`, calling `pace` before each call.
    fn encode(&self, index: usize, cb: &CommandBuffer, pace: &mut dyn FnMut()) {
        let spec = &self.program.command_buffers[index];
        for step in &spec.steps {
            pace();
            match step {
                Step::Wait(event) => cb
                    .encode_wait_for_event(EventRef::from(&self.events[*event]), 1)
                    .unwrap(),
                Step::Signal(event) => cb
                    .encode_signal_event(EventRef::from(&self.events[*event]), 1)
                    .unwrap(),
                Step::Encoder(encoder) => self.encoder(cb, encoder, pace),
            }
        }
    }

    fn encoder(&self, cb: &CommandBuffer, encoder: &Encoder, pace: &mut dyn FnMut()) {
        match encoder.kind {
            Kind::Blit => {
                let mut blit = cb.blit_command_encoder().unwrap();
                for &fence in &encoder.waits {
                    pace();
                    blit.wait_for_fence(&self.fences[fence]).unwrap();
                }
                for command in &encoder.commands {
                    pace();
                    match *command {
                        Command::Fill {
                            buffer,
                            start,
                            end,
                            value,
                        } => blit
                            .fill_buffer(&self.buffers[buffer], start..end, value)
                            .unwrap(),
                        Command::Copy { from, to, size } => {
                            let ((from, from_offset), (to, to_offset)) =
                                (self.at(from), self.at(to));
                            blit.copy_from_buffer_to_buffer(from, from_offset, to, to_offset, size)
                                .unwrap();
                        }
                        Command::ToTexture {
                            from,
                            to,
                            slice,
                            rows,
                        } => {
                            let (buffer, offset) = self.at(from);
                            let (origin, size) = rows_region(rows);
                            blit.copy_from_buffer_to_texture(
                                buffer,
                                offset,
                                ROW_BYTES,
                                ROW_BYTES * size.height,
                                size,
                                self.surface(to),
                                u64::from(slice),
                                0,
                                origin,
                                BlitOption::None,
                            )
                            .unwrap();
                        }
                        Command::FromTexture {
                            from,
                            slice,
                            rows,
                            to,
                        } => {
                            let (buffer, offset) = self.at(to);
                            let (origin, size) = rows_region(rows);
                            blit.copy_from_texture_to_buffer(
                                self.surface(from),
                                u64::from(slice),
                                0,
                                origin,
                                size,
                                buffer,
                                offset,
                                ROW_BYTES,
                                ROW_BYTES * size.height,
                                BlitOption::None,
                            )
                            .unwrap();
                        }
                        ref other => panic!("{other} in a blit encoder"),
                    }
                }
                for &fence in &encoder.updates {
                    pace();
                    blit.update_fence(&self.fences[fence]).unwrap();
                }
                pace();
                blit.end_encoding().unwrap();
            }
            Kind::Compute { concurrent } => {
                let dispatch_type = match concurrent {
                    true => DispatchType::Concurrent,
                    false => DispatchType::Serial,
                };
                let mut compute = cb
                    .compute_command_encoder_with_dispatch_type(dispatch_type)
                    .unwrap();
                for &fence in &encoder.waits {
                    pace();
                    compute.wait_for_fence(&self.fences[fence]).unwrap();
                }
                for command in &encoder.commands {
                    pace();
                    self.compute_command(&mut compute, command);
                }
                for &fence in &encoder.updates {
                    pace();
                    compute.update_fence(&self.fences[fence]).unwrap();
                }
                pace();
                compute.end_encoding().unwrap();
            }
            Kind::Render {
                target,
                slice,
                load,
            } => {
                let mut descriptor = RenderPassDescriptor::new();
                let attachment = &mut descriptor.color_attachments[0];
                attachment.texture = Some(self.surface(target).clone());
                attachment.slice = u64::from(slice);
                match load {
                    Load::Clear(color) => {
                        let unorm = |byte: u8| f64::from(byte) / 255.0;
                        attachment.load_action = LoadAction::Clear;
                        attachment.clear_color = ClearColor::new(
                            unorm(color[0]),
                            unorm(color[1]),
                            unorm(color[2]),
                            unorm(color[3]),
                        );
                    }
                    Load::Load => attachment.load_action = LoadAction::Load,
                }
                attachment.store_action = StoreAction::Store;
                let mut render = cb.render_command_encoder(&descriptor).unwrap();
                for &fence in &encoder.waits {
                    pace();
                    render
                        .wait_for_fence(&self.fences[fence], RenderStages::VERTEX)
                        .unwrap();
                }
                let state = match self.program.seen(target, 0).format {
                    Format::Rgba8Unorm => &self.lane.draw_rgba,
                    Format::Bgra8Unorm => &self.lane.draw_bgra,
                };
                for command in &encoder.commands {
                    pace();
                    let Command::Draw {
                        rects,
                        colors,
                        instances,
                        indirect,
                    } = *command
                    else {
                        panic!("{command} in a render encoder");
                    };
                    let ((rects, rects_offset), (colors, colors_offset)) =
                        (self.at(rects), self.at(colors));
                    render.set_render_pipeline_state(state).unwrap();
                    render
                        .set_vertex_buffer(Some(rects), rects_offset, 0)
                        .unwrap();
                    render
                        .set_fragment_buffer(Some(colors), colors_offset, 0)
                        .unwrap();
                    match indirect.map(|at| self.at(at)) {
                        Some((arguments, offset)) => render
                            .draw_primitives_indirect(PrimitiveType::Triangle, arguments, offset)
                            .unwrap(),
                        None => render
                            .draw_primitives(
                                PrimitiveType::Triangle,
                                0,
                                RECT_VERTICES,
                                instances,
                                0,
                            )
                            .unwrap(),
                    }
                }
                for &fence in &encoder.updates {
                    pace();
                    render
                        .update_fence(&self.fences[fence], RenderStages::FRAGMENT)
                        .unwrap();
                }
                pace();
                render.end_encoding().unwrap();
            }
        }
    }

    fn compute_command(&self, compute: &mut ComputeCommandEncoder, command: &Command) {
        let one = Size::new(1, 1, 1);
        match *command {
            Command::Dispatch {
                kernel,
                a,
                b,
                indirect,
            } => {
                let pipeline = match kernel {
                    Kernel::Copy => &self.lane.copy,
                    Kernel::Mix => &self.lane.mix,
                };
                let ((a, a_offset), (b, b_offset)) = (self.at(a), self.at(b));
                compute.set_compute_pipeline_state(pipeline).unwrap();
                compute.set_buffer(Some(a), a_offset, 0).unwrap();
                compute.set_buffer(Some(b), b_offset, 1).unwrap();
                match indirect.map(|at| self.at(at)) {
                    Some((arguments, offset)) => compute
                        .dispatch_threadgroups_indirect(arguments, offset, one)
                        .unwrap(),
                    None => compute.dispatch_threads(one, one).unwrap(),
                }
            }
            Command::TexelRead { from, rows, to } => {
                self.texel_dispatch(compute, &self.lane.texel_read, from, to, rows)
            }
            Command::TexelWrite { from, to, rows } => {
                self.texel_dispatch(compute, &self.lane.texel_write, to, from, rows)
            }
            Command::Barrier { textures } => {
                let scope = match textures {
                    true => BarrierScope::BUFFERS | BarrierScope::TEXTURES,
                    false => BarrierScope::BUFFERS,
                };
                compute.memory_barrier_with_scope(scope);
            }
            ref other => panic!("{other} in a compute encoder"),
        }
    }

    fn texel_dispatch(
        &self,
        compute: &mut ComputeCommandEncoder,
        kernel: &ComputePipelineState,
        surface: Surface,
        words: At,
        rows: u32,
    ) {
        let (words, offset) = self.at(words);
        compute.set_compute_pipeline_state(kernel).unwrap();
        compute.set_texture(Some(self.surface(surface)), 0).unwrap();
        compute.set_buffer(Some(words), offset, 0).unwrap();
        compute
            .dispatch_threads(
                Size::new(u64::from(TARGET), u64::from(rows), 1),
                Size::new(u64::from(TARGET), 1, 1),
            )
            .unwrap();
    }

    /// Every memory's bytes once the sequence ran: a readback command buffer copies each
    /// `Private` buffer and heap window and each texture slice into a `Shared` buffer.
    fn readback(&self) -> State {
        let device = &self.lane.device;
        let program = self.program;
        let shared = |len: u64| device.new_buffer(len, ResourceOptions::shared()).unwrap();
        let cb = self.lane.queue.command_buffer().unwrap();
        let mut blit = cb.blit_command_encoder().unwrap();
        let mut copies = Vec::new();
        for (index, spec) in program.buffers.iter().enumerate() {
            if spec.place == Place::Own && spec.private {
                let into = shared(BUFFER_BYTES);
                blit.copy_from_buffer_to_buffer(&self.buffers[index], 0, &into, 0, BUFFER_BYTES)
                    .unwrap();
                copies.push((index, into));
            }
        }
        let mut heap_copies = Vec::new();
        for (index, spec) in program.heaps.iter().enumerate() {
            if spec.private {
                let into = shared(HEAP_WINDOW);
                blit.copy_from_buffer_to_buffer(&self.windows[index], 0, &into, 0, HEAP_WINDOW)
                    .unwrap();
                heap_copies.push((index, into));
            }
        }
        let mut texture_copies = Vec::new();
        for (index, texture) in self.textures.iter().enumerate() {
            for slice in 0..program.textures[index].slices {
                let into = shared(TEXTURE_BYTES);
                let (origin, size) = rows_region((0, TARGET));
                blit.copy_from_texture_to_buffer(
                    texture,
                    u64::from(slice),
                    0,
                    origin,
                    size,
                    &into,
                    0,
                    ROW_BYTES,
                    TEXTURE_BYTES,
                    BlitOption::None,
                )
                .unwrap();
                texture_copies.push(into);
            }
        }
        blit.end_encoding().unwrap();
        cb.commit().unwrap();
        completed(&cb, "the readback");
        let buffers = (program.buffers.iter().enumerate())
            .map(|(index, spec)| match spec.place {
                Place::Own if spec.private => {
                    let (_, copy) = copies.iter().find(|(at, _)| *at == index).unwrap();
                    read(copy)
                }
                Place::Own => read(&self.buffers[index]),
                Place::NoCopy => {
                    let page = self.pages[index].expect("a host page");
                    // SAFETY: the page holds the buffer's bytes; every command buffer completed.
                    unsafe { std::slice::from_raw_parts(page.as_ptr(), BUFFER_BYTES as usize) }
                        .to_vec()
                }
                Place::Heap { .. } => Vec::new(),
            })
            .collect();
        let heaps = (program.heaps.iter().enumerate())
            .map(|(index, spec)| match spec.private {
                true => {
                    let (_, copy) = heap_copies.iter().find(|(at, _)| *at == index).unwrap();
                    read(copy)
                }
                false => read(&self.windows[index]),
            })
            .collect();
        let mut texture_copies = texture_copies.iter();
        let textures = (program.textures.iter())
            .map(|spec| {
                (0..spec.slices)
                    .flat_map(|_| read(texture_copies.next().unwrap()))
                    .collect()
            })
            .collect();
        State {
            buffers,
            heaps,
            textures,
        }
    }
}

/// What a setup copy fills.
enum Staged {
    Buffer(usize),
    Heap(usize),
    Slice(usize, u32),
}

/// Run every command buffer in turn: made, encoded and committed, then all waited for.
fn serial(objects: &Objects) {
    let mut committed = Vec::new();
    for index in 0..objects.program.command_buffers.len() {
        let cb = objects.lane.queue.command_buffer().unwrap();
        objects.encode(index, &cb, &mut || {});
        cb.commit().unwrap();
        committed.push(cb);
    }
    for (index, cb) in committed.iter().enumerate() {
        completed(cb, &format!("cb{index}"));
    }
}

/// The encodes done so far, by command buffer, and the turnstile's state.
struct Progress {
    encoded: Vec<bool>,
    /// The next command buffer a [`Order::Reversed`] run may encode.
    baton: usize,
    /// The thread whose turn it is in a [`Order::Lockstep`] run, and the threads still encoding.
    turn: usize,
    active: Vec<bool>,
    /// The command buffers made on the worker threads ([`Made::OnWorkers`]).
    made: Vec<Option<CommandBuffer>>,
}

/// The threads' shared state for one parallel run.
struct Shared {
    progress: Mutex<Progress>,
    changed: Condvar,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, Progress> {
        self.progress.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn wait<'g>(
        &self,
        guard: std::sync::MutexGuard<'g, Progress>,
        timeout: Duration,
    ) -> std::sync::MutexGuard<'g, Progress> {
        self.changed
            .wait_timeout(guard, timeout)
            .unwrap_or_else(|p| p.into_inner())
            .0
    }

    /// In a lockstep run, wait for thread `me`'s turn (at most [`TURN`]), then pass it on.
    fn take_turn(&self, me: usize) {
        let until = Instant::now() + TURN;
        let mut progress = self.lock();
        while progress.turn != me && Instant::now() < until {
            progress = self.wait(progress, until.saturating_duration_since(Instant::now()));
        }
        if progress.turn == me {
            progress.turn = next_active(&progress.active, me);
        }
        drop(progress);
        self.changed.notify_all();
    }

    /// Thread `me` encodes no more: its turn, if it holds it, passes on.
    fn leave(&self, me: usize) {
        let mut progress = self.lock();
        progress.active[me] = false;
        if progress.turn == me {
            progress.turn = next_active(&progress.active, me);
        }
        drop(progress);
        self.changed.notify_all();
    }
}

/// The next active thread after `me`, wrapping; `me` where none is.
fn next_active(active: &[bool], me: usize) -> usize {
    (1..=active.len())
        .map(|step| (me + step) % active.len())
        .find(|&thread| active[thread])
        .unwrap_or(me)
}

/// A pause before an encoding call, drawn from `rng` (the module docs).
fn pause(rng: &mut Rng) {
    match rng.below(20) {
        0..=10 => {}
        11..=15 => std::thread::yield_now(),
        16..=18 => {
            let until = Instant::now() + Duration::from_micros(rng.below(20));
            while Instant::now() < until {
                std::hint::spin_loop();
            }
        }
        _ => std::thread::sleep(Duration::from_micros(rng.below(50))),
    }
}

/// Run the sequence under `schedule` (the module docs).
fn parallel(objects: &Objects, schedule: Schedule) {
    let count = objects.program.command_buffers.len();
    let threads = schedule.threads.min(count).max(1);
    let mut rng = Rng::new(schedule.seed ^ 0xA11C_E5ED);
    // Who encodes which command buffer, in the order each thread encodes them.
    let mut lists: Vec<Vec<usize>> = vec![Vec::new(); threads];
    match schedule.order {
        Order::Reversed | Order::Lockstep => {
            for index in (0..count).rev() {
                lists[index % threads].push(index);
            }
        }
        Order::Spread => {
            for index in 0..count {
                lists[rng.below(threads as u64) as usize].push(index);
            }
            for list in &mut lists {
                for at in (1..list.len()).rev() {
                    list.swap(at, rng.below(at as u64 + 1) as usize);
                }
            }
        }
    }
    let queue = &objects.lane.queue;
    let made: Vec<Option<CommandBuffer>> = match schedule.made {
        Made::InOrder => (0..count)
            .map(|_| Some(queue.command_buffer().unwrap()))
            .collect(),
        Made::Reversed => {
            let mut made: Vec<_> = (0..count)
                .map(|_| Some(queue.command_buffer().unwrap()))
                .collect();
            made.reverse();
            made
        }
        Made::OnWorkers => (0..count).map(|_| None).collect(),
    };
    let shared = Shared {
        progress: Mutex::new(Progress {
            encoded: vec![false; count],
            baton: count - 1,
            turn: 0,
            active: vec![true; threads],
            made: (0..count).map(|_| None).collect(),
        }),
        changed: Condvar::new(),
    };
    let start = Barrier::new(threads + 1);
    let seeds: Vec<u64> = (0..threads).map(|_| rng.next()).collect();
    std::thread::scope(|scope| {
        for (me, list) in lists.iter().enumerate() {
            let (shared, start, made, seed) = (&shared, &start, &made, seeds[me]);
            scope.spawn(move || {
                let mut rng = Rng::new(seed);
                start.wait();
                for &index in list {
                    if schedule.order == Order::Reversed {
                        let until = Instant::now() + WAIT;
                        let mut progress = shared.lock();
                        while progress.baton != index {
                            assert!(Instant::now() < until, "cb{index} never had the baton");
                            progress = shared.wait(progress, WAIT);
                        }
                    }
                    let cb = match &made[index] {
                        Some(cb) => cb.clone(),
                        None => queue.command_buffer().unwrap(),
                    };
                    let mut pace = || match schedule.order {
                        Order::Lockstep => shared.take_turn(me),
                        Order::Spread => pause(&mut rng),
                        Order::Reversed => {}
                    };
                    objects.encode(index, &cb, &mut pace);
                    let mut progress = shared.lock();
                    progress.encoded[index] = true;
                    progress.made[index] = Some(cb);
                    progress.baton = index.saturating_sub(1);
                    drop(progress);
                    shared.changed.notify_all();
                }
                shared.leave(me);
            });
        }
        start.wait();
        // Commit in order: each as soon as it and every earlier one were encoded, or once all
        // were.
        let mut committed = Vec::new();
        for index in 0..count {
            let until = Instant::now() + WAIT;
            let mut progress = shared.lock();
            let ready = |progress: &Progress| match schedule.commits {
                Commits::AsReady => progress.encoded[index],
                Commits::AfterAll => progress.encoded.iter().all(|&done| done),
            };
            while !ready(&progress) {
                assert!(Instant::now() < until, "cb{index} was never encoded");
                progress = shared.wait(progress, WAIT);
            }
            let cb = progress.made[index]
                .take()
                .expect("an encoded command buffer");
            drop(progress);
            cb.commit().unwrap();
            committed.push(cb);
        }
        for (index, cb) in committed.iter().enumerate() {
            completed(cb, &format!("cb{index} under {schedule:?}"));
        }
    });
}

/// The records of a run's bytes, named as the model's.
fn records(program: &Program, state: &State) -> Vec<(String, Value)> {
    readback_records(program, state)
        .into_iter()
        .map(|(name, bytes)| (name, Value::bytes(&bytes)))
        .collect()
}

/// The kinds `lane`'s device announced since its last take.
fn kinds(lane: &Lane) -> BTreeSet<Degradation> {
    (lane.messages.take_notices().iter())
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| notice.degradation)
        .collect()
}

/// Run `program` `runs` times serially on one device and `runs` times in parallel on `threads`
/// threads on another (the module docs), comparing every run's bytes with the model's and the
/// devices' notices and layer messages. `name` and `seed` name the case in a failure.
fn compare(name: &str, program: &Program, threads: usize, runs: u64, seed: u64) {
    let expected = records(program, &model_run(program));
    let cache = ScratchCache::new().unwrap();
    let mut pages = HostPages::new();
    let mut problems = Vec::new();
    let mut check = |lane: &Lane, what: String, state: &State| {
        let differences = byte_differences(&records(program, state), &expected);
        if !differences.is_empty() {
            problems.push(format!("{what}: {}", differences.join("; ")));
        }
        let layer = lane.messages.take();
        if !layer.is_empty() {
            let reported: Vec<String> = (layer.iter())
                .map(|message| format!("{}: {}", message.id_name, message.text))
                .collect();
            problems.push(format!("{what}: the layer reported {reported:?}"));
        }
    };
    let serial_lane = Lane::open(&cache);
    let mut serial_kinds = kinds(&serial_lane);
    for run in 0..runs {
        let objects = Objects::make(&serial_lane, program, &mut pages);
        serial(&objects);
        let state = objects.readback();
        check(&serial_lane, format!("serial run {run}"), &state);
        serial_kinds.extend(kinds(&serial_lane));
    }
    let parallel_lane = Lane::open(&cache);
    let mut parallel_kinds = kinds(&parallel_lane);
    let mut schedules = Vec::new();
    for run in 0..runs {
        let schedule = Schedule::of(threads, run, seed.wrapping_mul(1_000_003) ^ run);
        let objects = Objects::make(&parallel_lane, program, &mut pages);
        parallel(&objects, schedule);
        let state = objects.readback();
        check(&parallel_lane, format!("{schedule:?}"), &state);
        parallel_kinds.extend(kinds(&parallel_lane));
        schedules.push(schedule);
    }
    let device = parallel_lane.device.name().to_owned();
    drop((serial_lane, parallel_lane));
    drop(pages);
    if serial_kinds != parallel_kinds {
        problems.push(format!(
            "the notices differ: serially {serial_kinds:?}, in parallel {parallel_kinds:?}"
        ));
    }
    assert!(
        problems.is_empty(),
        "{name} (seed {seed}) on {device}, {threads} threads:\n{}\n{program}",
        problems.join("\n")
    );
    println!(
        "PARALLEL-ENCODE {name} seed {seed} on {device}: {} command buffers, {runs} runs each way \
         on {threads} threads, bytes the model's, notices {serial_kinds:?} both ways",
        program.command_buffers.len()
    );
}

/// The seeds of [`merged`]'s sequences one test merges.
const MERGED: usize = 4;

/// Runs each way per test: every [`Order`] four times or more.
const RUNS: u64 = 12;

/// The generated sequences of `seeds` on one queue (the module docs): each one's resources,
/// events and fences renumbered after the earlier ones', its command buffers interleaved with the
/// others' in turn, each keeping its order.
fn merged(seeds: &[u64]) -> Program {
    let mut program = Program {
        heaps: Vec::new(),
        buffers: Vec::new(),
        textures: Vec::new(),
        views: Vec::new(),
        queues: 1,
        events: 0,
        fences: 0,
        command_buffers: Vec::new(),
    };
    let mut lists = Vec::new();
    for &seed in seeds {
        let one = generate(seed);
        let base = Base {
            heap: program.heaps.len(),
            buffer: program.buffers.len(),
            texture: program.textures.len(),
            view: program.views.len(),
            event: program.events,
            fence: program.fences,
        };
        program.heaps.extend(one.heaps.iter().cloned());
        program
            .buffers
            .extend(one.buffers.iter().map(|spec| BufferSpec {
                place: match spec.place {
                    Place::Heap { heap, offset } => Place::Heap {
                        heap: heap + base.heap,
                        offset,
                    },
                    other => other,
                },
                ..spec.clone()
            }));
        program.textures.extend(one.textures.iter().cloned());
        program.views.extend(one.views.iter().map(|view| {
            let mut view = *view;
            view.texture += base.texture;
            view
        }));
        program.events += one.events;
        program.fences += one.fences;
        let specs: Vec<CommandBufferSpec> = (one.command_buffers.iter())
            .map(|spec| CommandBufferSpec {
                queue: 0,
                steps: spec.steps.iter().map(|step| base.step(step)).collect(),
            })
            .collect();
        lists.push(specs.into_iter());
    }
    loop {
        let before = program.command_buffers.len();
        for list in &mut lists {
            program.command_buffers.extend(list.next());
        }
        if program.command_buffers.len() == before {
            break;
        }
    }
    program
}

/// The first index of each kind a merged sequence's part takes.
struct Base {
    heap: usize,
    buffer: usize,
    texture: usize,
    view: usize,
    event: usize,
    fence: usize,
}

impl Base {
    fn at(&self, at: At) -> At {
        At {
            buffer: at.buffer + self.buffer,
            offset: at.offset,
        }
    }

    fn surface(&self, surface: Surface) -> Surface {
        match surface {
            Surface::Texture(index) => Surface::Texture(index + self.texture),
            Surface::View(index) => Surface::View(index + self.view),
        }
    }

    fn step(&self, step: &Step) -> Step {
        match step {
            Step::Wait(event) => Step::Wait(event + self.event),
            Step::Signal(event) => Step::Signal(event + self.event),
            Step::Encoder(encoder) => Step::Encoder(Encoder {
                kind: match encoder.kind {
                    Kind::Render {
                        target,
                        slice,
                        load,
                    } => Kind::Render {
                        target: self.surface(target),
                        slice,
                        load,
                    },
                    other => other,
                },
                waits: encoder.waits.iter().map(|f| f + self.fence).collect(),
                updates: encoder.updates.iter().map(|f| f + self.fence).collect(),
                commands: encoder.commands.iter().map(|c| self.command(c)).collect(),
            }),
        }
    }

    fn command(&self, command: &Command) -> Command {
        match *command {
            Command::Fill {
                buffer,
                start,
                end,
                value,
            } => Command::Fill {
                buffer: buffer + self.buffer,
                start,
                end,
                value,
            },
            Command::Copy { from, to, size } => Command::Copy {
                from: self.at(from),
                to: self.at(to),
                size,
            },
            Command::ToTexture {
                from,
                to,
                slice,
                rows,
            } => Command::ToTexture {
                from: self.at(from),
                to: self.surface(to),
                slice,
                rows,
            },
            Command::FromTexture {
                from,
                slice,
                rows,
                to,
            } => Command::FromTexture {
                from: self.surface(from),
                slice,
                rows,
                to: self.at(to),
            },
            Command::Dispatch {
                kernel,
                a,
                b,
                indirect,
            } => Command::Dispatch {
                kernel,
                a: self.at(a),
                b: self.at(b),
                indirect: indirect.map(|at| self.at(at)),
            },
            Command::TexelRead { from, rows, to } => Command::TexelRead {
                from: self.surface(from),
                rows,
                to: self.at(to),
            },
            Command::TexelWrite { from, to, rows } => Command::TexelWrite {
                from: self.at(from),
                to: self.surface(to),
                rows,
            },
            Command::Barrier { textures } => Command::Barrier { textures },
            Command::Draw {
                rects,
                colors,
                instances,
                indirect,
            } => Command::Draw {
                rects: self.at(rects),
                colors: self.at(colors),
                instances,
                indirect: indirect.map(|at| self.at(at)),
            },
        }
    }
}

/// The chained sequence of `count` command buffers (the module docs). Buffers: 0, a tracked
/// `Private` data buffer (its word 0 the chain's accumulator, which each command buffer mixes a
/// word it read into with `seq_mix`, and the draws' colours); 1, a tracked `Shared` data buffer
/// (the texture's rows as words); 2, the rectangles; 3, an untracked `Private` data buffer one
/// command buffer writes and the next reads, ordered by fence 0 alone. Texture 0 (tracked) is
/// the passes' target. Command buffer `k`, by `k % 4`:
/// - 0: a serial compute encoder mixing word `k` of buffer 1 into the accumulator, then copying
///   it into word `k` of buffer 3, updating fence 0;
/// - 1: a render pass loading texture 0 and drawing rectangle `k % 4` in the accumulator's
///   colour; a blit, waiting for fence 0, copying word `k - 1` of buffer 3 into word 1 of buffer
///   0; a serial compute encoder mixing that word into the accumulator;
/// - 2: a blit copying texture 0's rows into buffer 1, then a concurrent compute encoder writing
///   four rows of buffer 1's words back into texture 0's rows `0..4` and, after a memory barrier
///   of buffers, mixing word 2 of buffer 1 into the accumulator;
/// - 3: a render pass clearing texture 0 to a colour of `k`'s and drawing rectangle `k % 4` in
///   the colour of word 1 of buffer 0, then a serial compute encoder reading texture 0 into
///   buffer 1's words `64..128` and mixing one of them into the accumulator.
fn chained(count: usize) -> Program {
    let mut rng = Rng::new(0xC4A1_4ED5 ^ count as u64);
    let data = |rng: &mut Rng, private: bool, tracked: bool| BufferSpec {
        role: Role::Data,
        private,
        tracked,
        place: Place::Own,
        initial: (0..BUFFER_BYTES).map(|_| rng.next() as u8).collect(),
    };
    // Four rectangles, each inside the target and none the whole of it.
    let rects: [[u32; 4]; 4] = [[0, 0, 5, 3], [2, 1, 8, 6], [1, 4, 7, 8], [3, 2, 6, 7]];
    let mut rect_words: Vec<u8> = rects
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    rect_words.resize(BUFFER_BYTES as usize, 0);
    let buffers = vec![
        data(&mut rng, true, true),
        data(&mut rng, false, true),
        BufferSpec {
            role: Role::Rects,
            private: false,
            tracked: true,
            place: Place::Own,
            initial: rect_words,
        },
        data(&mut rng, true, false),
    ];
    let texture = TextureSpec {
        tracked: true,
        slices: 1,
        heap: false,
        initial: (0..TEXTURE_BYTES).map(|_| rng.next() as u8).collect(),
    };
    let word = |buffer: usize, index: u64| At {
        buffer,
        offset: 4 * index,
    };
    let encoder = |kind: Kind, commands: Vec<Command>| Encoder {
        kind,
        waits: Vec::new(),
        updates: Vec::new(),
        commands,
    };
    let target = Surface::Texture(0);
    let draw = |k: usize, colors: At| Command::Draw {
        rects: At {
            buffer: 2,
            offset: RECORD_BYTES * (k as u64 % 4),
        },
        colors,
        instances: 1,
        indirect: None,
    };
    // Word 0 of buffer 0 mixed with `salt` in place: the chain's every command buffer folds a word
    // it read into it, so its last value depends on the order of them all.
    let fold = |salt: At| Command::Dispatch {
        kernel: Kernel::Mix,
        a: word(0, 0),
        b: salt,
        indirect: None,
    };
    let mut command_buffers = Vec::new();
    for k in 0..count {
        let kk = k as u64;
        let steps = match k % 4 {
            0 => {
                let mut mix = encoder(
                    Kind::Compute { concurrent: false },
                    vec![
                        fold(word(1, kk % 32)),
                        Command::Dispatch {
                            kernel: Kernel::Copy,
                            a: word(0, 0),
                            b: word(3, kk % 32),
                            indirect: None,
                        },
                    ],
                );
                mix.updates.push(0);
                vec![Step::Encoder(mix)]
            }
            1 => {
                let pass = encoder(
                    Kind::Render {
                        target,
                        slice: 0,
                        load: Load::Load,
                    },
                    vec![draw(k, word(0, 0))],
                );
                let mut fenced = encoder(
                    Kind::Blit,
                    vec![Command::Copy {
                        from: word(3, (kk - 1) % 32),
                        to: word(0, 1),
                        size: 4,
                    }],
                );
                fenced.waits.push(0);
                let fold = encoder(Kind::Compute { concurrent: false }, vec![fold(word(0, 1))]);
                vec![
                    Step::Encoder(pass),
                    Step::Encoder(fenced),
                    Step::Encoder(fold),
                ]
            }
            2 => vec![
                Step::Encoder(encoder(
                    Kind::Blit,
                    vec![Command::FromTexture {
                        from: target,
                        slice: 0,
                        rows: (0, TARGET),
                        to: word(1, 0),
                    }],
                )),
                Step::Encoder(encoder(
                    Kind::Compute { concurrent: true },
                    vec![
                        Command::TexelWrite {
                            from: word(1, 8 * (kk / 4 % 4)),
                            to: target,
                            rows: 4,
                        },
                        Command::Barrier { textures: false },
                        fold(word(1, 2)),
                    ],
                )),
            ],
            _ => {
                let color = [kk as u8, 0x40 | kk as u8, 0x80, 0xff];
                vec![
                    Step::Encoder(encoder(
                        Kind::Render {
                            target,
                            slice: 0,
                            load: Load::Clear(color),
                        },
                        vec![draw(k, word(0, 1))],
                    )),
                    Step::Encoder(encoder(
                        Kind::Compute { concurrent: false },
                        vec![
                            Command::TexelRead {
                                from: target,
                                rows: TARGET,
                                to: word(1, 64),
                            },
                            fold(word(1, 64 + kk % 64)),
                        ],
                    )),
                ]
            }
        };
        command_buffers.push(CommandBufferSpec { queue: 0, steps });
    }
    Program {
        heaps: Vec::new(),
        buffers,
        textures: vec![texture],
        views: Vec::new(),
        queues: 1,
        events: 0,
        fences: 1,
        command_buffers,
    }
}

/// What `order::problems` finds in `program` but the races on buffer 3 of [`chained`] and its
/// waits for fence 0, which orders them across command buffers (Metal evaluates a fence at
/// submission; `order.rs` relies on no fence across command buffers).
fn races_but_fenced(program: &Program) -> Vec<String> {
    let fenced = format!("{:?}", super::program::Resource::Buffer(3));
    problems(program)
        .into_iter()
        .filter(|problem| !problem.contains(&fenced) && !problem.starts_with("fence0: "))
        .collect()
}

/// The chained sequence's hazards: every command buffer reads the one before's writes and
/// overwrites its reads, and the model's bytes depend on that order.
fn chain(threads: usize) {
    let program = chained(4 * threads.max(3));
    assert_eq!(races_but_fenced(&program), Vec::<String>::new());
    // The order matters: the same command buffers committed in another order end otherwise.
    let mut swapped = program.clone();
    swapped.command_buffers.swap(1, 2);
    assert_ne!(
        model_run(&program),
        model_run(&swapped),
        "a chain the order matters to"
    );
    compare("chained", &program, threads, RUNS, threads as u64);
}

/// [`merged`]'s sequences of seeds from `first`, [`MERGED`] of them a test.
fn generated(threads: usize, first: u64) {
    let seeds: Vec<u64> = (first..first + MERGED as u64).collect();
    let program = merged(&seeds);
    assert_eq!(problems(&program), Vec::<String>::new(), "{program}");
    compare("merged", &program, threads, RUNS, first);
}

#[test]
fn two_threads_keep_a_chain_s_order() {
    chain(2);
}

#[test]
fn four_threads_keep_a_chain_s_order() {
    chain(4);
}

#[test]
fn eight_threads_keep_a_chain_s_order() {
    chain(8);
}

#[test]
fn two_threads_keep_generated_sequences_answers() {
    generated(2, 0);
}

#[test]
fn four_threads_keep_generated_sequences_answers() {
    generated(4, 8);
}

#[test]
fn eight_threads_keep_generated_sequences_answers() {
    generated(8, 16);
}
