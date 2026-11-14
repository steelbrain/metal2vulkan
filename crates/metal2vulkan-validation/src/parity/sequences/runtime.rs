//! The runtime's side of the sequences: [`SeqApi`] on `metal2vulkan_runtime`, on the device the
//! lane's cfg selects, under the validation layer with the lane's checks
//! (`runtime_lane::validated_device_with`), the fixture's AIR assembled from its `.ll` text. It notes
//! each call that binds buffers with their `VkBuffer`s ([`Encoded`]), and takes the layer's
//! messages after each commit with the command buffers whose batches they may come from, which
//! the lane reads the layer's hazards with (`hazards`).

use super::super::api::{Answer, Completion};
use super::super::runtime::refused;
use super::api::SeqApi;
use super::hazards::{Call, Encoded};
use super::program::{Load, ViewForm, ViewSpec};
use super::run::{pixel_format, swizzle};
use super::FIXTURE;
use crate::air_library::AirLibrary as _;
use crate::gpu::TEST_DEVICE;
use crate::runtime_lane::{at_plan, validated_device_with, LaneChecks, Messages, ScratchCache};
use ash::vk::Handle as _;
use metal2vulkan::metal::{
    BarrierScope, DispatchType, PixelFormat, PrimitiveType, RenderStages, TextureType,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus,
    CommandQueue, ComputeCommandEncoder, ComputePipelineState, DebugMessage, Device, EventRef,
    Fence, Heap, HeapDescriptor, HeapType, HostMemory, Library, LoadAction, Origin,
    PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SharedEvent, Size, StoreAction, Texture, TextureDescriptor,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::ops::Range;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// The longest a command buffer of a sequence may take.
const WAIT: Duration = Duration::from_secs(60);

/// The longest the lane waits after a commit for the command buffer to be scheduled before it
/// takes the layer's messages; past it, the command buffer's batches may yet come, and the
/// messages of later takes may be theirs too.
const SCHEDULING: Duration = Duration::from_millis(100);

/// Messages the lane took, each with the command buffers of the sequence whose batches it may
/// come from.
pub(crate) type Taken = Vec<(BTreeSet<usize>, DebugMessage)>;

/// A runtime device, its validation messages, its shader cache, and the fixture's library, for
/// one sequence, with the calls it encoded and the command buffers committed. The device is
/// dropped before the cache directory is removed.
pub(crate) struct RuntimeSeq {
    /// The command buffers committed and not seen scheduled at a take, with their commit
    /// numbers (the runner's setup is 0, command buffer `k` of the sequence `k + 1`). Dropped
    /// before the device.
    unscheduled: RefCell<Vec<(usize, CommandBuffer)>>,
    library: Library,
    device: Device,
    pub(crate) messages: Messages,
    pub(crate) encoded: RefCell<Vec<Encoded>>,
    committed: Cell<usize>,
    taken: RefCell<Taken>,
    _cache: ScratchCache,
}

impl RuntimeSeq {
    /// A device with the lane's checks (`METAL2VULKAN_LANE_VALIDATION`).
    pub(crate) fn open() -> Result<RuntimeSeq, String> {
        RuntimeSeq::open_checked(None)
    }

    /// A device whose layer checks synchronization, whatever the lane's switch says.
    pub(crate) fn open_synchronized() -> Result<RuntimeSeq, String> {
        RuntimeSeq::open_checked(Some(LaneChecks {
            synchronization: true,
            ..LaneChecks::default()
        }))
    }

    fn open_checked(checks: Option<LaneChecks>) -> Result<RuntimeSeq, String> {
        let cache = ScratchCache::new().map_err(|error| error.to_string())?;
        let messages = Messages::default();
        // A program's command queues are its own, as many as it asks for: the lane's queue limits
        // would change the second one's placement, so the device takes its plan.
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            at_plan(options);
            if let Some(checks) = checks {
                checks.apply(options);
            }
        })?;
        let ll = engine_dir().join(format!("{FIXTURE}.ll"));
        let bitcode = llvm_assemble(&ll).map_err(|error| format!("{}: {error:?}", ll.display()))?;
        let library = device
            .new_library_with_air(&bitcode)
            .map_err(|error| format!("{FIXTURE}: {error}"))?;
        Ok(RuntimeSeq {
            unscheduled: RefCell::new(Vec::new()),
            library,
            device,
            messages,
            encoded: RefCell::new(Vec::new()),
            committed: Cell::new(0),
            taken: RefCell::new(Vec::new()),
            _cache: cache,
        })
    }

    /// Note an encoded `call` that passed `buffers`.
    fn note(&self, call: Call, buffers: &[&Buffer]) {
        let buffers = buffers
            .iter()
            .map(|buffer| {
                // SAFETY: the handle is only compared with the layer's reports, never used.
                let (raw, _) = unsafe { buffer.vk_buffer() };
                raw.as_raw()
            })
            .collect();
        self.encoded.borrow_mut().push(Encoded {
            committed: self.committed.get(),
            call,
            buffers,
        });
    }
}

impl RuntimeSeq {
    /// Take the layer's messages so far, after waiting up to [`SCHEDULING`] for the last command
    /// buffer committed to be scheduled: each may come from the batches of any command buffer not
    /// seen scheduled at the last take, the layer checking a batch as it is submitted. A command
    /// buffer seen scheduled before the take sends no batch after it.
    fn take_messages(&self) {
        let mut unscheduled = self.unscheduled.borrow_mut();
        let from = cbs(&unscheduled);
        if let Some((_, last)) = unscheduled.last() {
            let deadline = Instant::now() + SCHEDULING;
            while !scheduled(last.status()) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        unscheduled.retain(|(_, cb)| !scheduled(cb.status()));
        let messages = self.messages.take();
        self.taken
            .borrow_mut()
            .extend(messages.into_iter().map(|message| (from.clone(), message)));
    }

    /// The messages taken, and the command buffers of the sequence the messages not yet taken
    /// may come from.
    pub(crate) fn taken(&self) -> (Taken, BTreeSet<usize>) {
        (self.taken.take(), cbs(&self.unscheduled.borrow()))
    }
}

/// The command buffers of the sequence among `committed`, by commit number: the setup's and the
/// readback's are none of them.
fn cbs(committed: &[(usize, CommandBuffer)]) -> BTreeSet<usize> {
    committed
        .iter()
        .filter_map(|(number, _)| number.checked_sub(1))
        .collect()
}

/// Whether a command buffer of `status` sent its every batch.
fn scheduled(status: CommandBufferStatus) -> bool {
    matches!(
        status,
        CommandBufferStatus::Scheduled
            | CommandBufferStatus::Completed
            | CommandBufferStatus::Error
    )
}

/// The copy's texels: whole rows of a 2D texture.
fn rows_region(rows: &Range<u32>, width: u32) -> (Origin, Size) {
    (
        Origin::new(0, u64::from(rows.start), 0),
        Size::new(u64::from(width), u64::from(rows.end - rows.start), 1),
    )
}

impl SeqApi for RuntimeSeq {
    type Heap = Heap;
    type Buffer = Buffer;
    type Texture = Texture;
    type Kernel = ComputePipelineState;
    type Draw = RenderPipelineState;
    type Queue = CommandQueue;
    type CommandBuffer = CommandBuffer;
    type Event = SharedEvent;
    type Fence = Fence;
    type Blit = BlitCommandEncoder;
    type Compute = ComputeCommandEncoder;
    type Render = RenderCommandEncoder;

    fn buffer(&self, length: u64, options: ResourceOptions) -> Answer<Buffer> {
        self.device.new_buffer(length, options).map_err(refused)
    }

    unsafe fn no_copy_buffer(
        &self,
        bytes: NonNull<u8>,
        length: u64,
        options: ResourceOptions,
    ) -> Answer<Buffer> {
        // SAFETY: the caller's contract, `HostMemory::new`'s.
        unsafe {
            self.device
                .new_buffer_with_bytes_no_copy(HostMemory::new(bytes), length, options, None)
        }
        .map_err(refused)
    }

    fn heap(&self, size: u64, options: ResourceOptions) -> Answer<Heap> {
        let descriptor = HeapDescriptor::new(HeapType::Placement, size, options);
        self.device.new_heap(&descriptor).map_err(refused)
    }

    fn heap_buffer(
        &self,
        heap: &Heap,
        length: u64,
        options: ResourceOptions,
        offset: u64,
    ) -> Answer<Buffer> {
        heap.new_buffer_at_offset(length, options, offset)
            .map_err(refused)
    }

    fn heap_texture(
        &self,
        heap: &Heap,
        descriptor: &TextureDescriptor,
        offset: u64,
    ) -> Answer<Texture> {
        heap.new_texture_at_offset(descriptor, offset)
            .map_err(refused)
    }

    fn view(&self, texture: &Texture, view: &ViewSpec) -> Answer<Texture> {
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
        .map_err(refused)
    }

    fn write(&self, buffer: &Buffer, bytes: &[u8]) {
        let contents = buffer.contents().expect("a Shared buffer");
        assert!(bytes.len() as u64 <= buffer.length());
        // SAFETY: the mapping covers the buffer's length, which holds `bytes`; no command
        // buffer uses the buffer while the runner writes it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    }

    fn read(&self, buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        let length = usize::try_from(buffer.length()).expect("a buffer the host maps");
        // SAFETY: the mapping covers the buffer's length; the command buffers that wrote it
        // completed.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec()
    }

    fn texture(&self, descriptor: &TextureDescriptor) -> Answer<Texture> {
        self.device.new_texture(descriptor).map_err(refused)
    }

    fn kernel(&self, entry: &str) -> Answer<ComputePipelineState> {
        let function = self
            .library
            .new_function_with_name(entry)
            .map_err(refused)?;
        self.device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .map_err(refused)
    }

    fn draw_state(&self, format: PixelFormat) -> Answer<RenderPipelineState> {
        let function = |name| self.library.new_function_with_name(name).map_err(refused);
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            function("seq_rect")?.function_descriptor(),
        ));
        descriptor.fragment = Some(StageSource::Function(
            function("seq_color")?.function_descriptor(),
        ));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(refused)
    }

    fn queue(&self) -> Answer<CommandQueue> {
        self.device.new_command_queue().map_err(refused)
    }

    fn command_buffer(&self, queue: &CommandQueue) -> Answer<CommandBuffer> {
        queue.command_buffer().map_err(refused)
    }

    fn event(&self) -> Answer<SharedEvent> {
        self.device.new_shared_event().map_err(refused)
    }

    fn fence(&self) -> Answer<Fence> {
        Ok(self.device.new_fence())
    }

    fn signal(&self, cb: &CommandBuffer, event: &SharedEvent, value: u64) -> Answer<()> {
        cb.encode_signal_event(EventRef::from(event), value)
            .map_err(refused)
    }

    fn wait(&self, cb: &CommandBuffer, event: &SharedEvent, value: u64) -> Answer<()> {
        cb.encode_wait_for_event(EventRef::from(event), value)
            .map_err(refused)
    }

    fn blit(&self, cb: &CommandBuffer) -> Answer<BlitCommandEncoder> {
        cb.blit_command_encoder().map_err(refused)
    }

    fn fill(
        &self,
        blit: &mut BlitCommandEncoder,
        buffer: &Buffer,
        range: Range<u64>,
        value: u8,
    ) -> Answer<()> {
        self.note(Call::Fill, &[buffer]);
        blit.fill_buffer(buffer, range, value).map_err(refused)
    }

    fn copy(
        &self,
        blit: &mut BlitCommandEncoder,
        (from, from_offset): (&Buffer, u64),
        (to, to_offset): (&Buffer, u64),
        size: u64,
    ) -> Answer<()> {
        self.note(Call::Copy, &[from, to]);
        blit.copy_from_buffer_to_buffer(from, from_offset, to, to_offset, size)
            .map_err(refused)
    }

    fn buffer_to_texture(
        &self,
        blit: &mut BlitCommandEncoder,
        (from, offset): (&Buffer, u64),
        (texture, slice): (&Texture, u32),
        rows: Range<u32>,
        (width, row_bytes): (u32, u64),
    ) -> Answer<()> {
        self.note(Call::ToTexture, &[from]);
        let (origin, size) = rows_region(&rows, width);
        blit.copy_from_buffer_to_texture(
            from,
            offset,
            row_bytes,
            row_bytes * size.height,
            size,
            texture,
            u64::from(slice),
            0,
            origin,
            BlitOption::None,
        )
        .map_err(refused)
    }

    fn texture_to_buffer(
        &self,
        blit: &mut BlitCommandEncoder,
        (texture, slice): (&Texture, u32),
        rows: Range<u32>,
        (width, row_bytes): (u32, u64),
        (to, offset): (&Buffer, u64),
    ) -> Answer<()> {
        self.note(Call::FromTexture, &[to]);
        let (origin, size) = rows_region(&rows, width);
        blit.copy_from_texture_to_buffer(
            texture,
            u64::from(slice),
            0,
            origin,
            size,
            to,
            offset,
            row_bytes,
            row_bytes * size.height,
            BlitOption::None,
        )
        .map_err(refused)
    }

    fn blit_update_fence(&self, blit: &mut BlitCommandEncoder, fence: &Fence) -> Answer<()> {
        blit.update_fence(fence).map_err(refused)
    }

    fn blit_wait_fence(&self, blit: &mut BlitCommandEncoder, fence: &Fence) -> Answer<()> {
        blit.wait_for_fence(fence).map_err(refused)
    }

    fn end_blit(&self, blit: BlitCommandEncoder) -> Answer<()> {
        blit.end_encoding().map_err(refused)
    }

    fn compute(&self, cb: &CommandBuffer, concurrent: bool) -> Answer<ComputeCommandEncoder> {
        let dispatch_type = if concurrent {
            DispatchType::Concurrent
        } else {
            DispatchType::Serial
        };
        cb.compute_command_encoder_with_dispatch_type(dispatch_type)
            .map_err(refused)
    }

    fn dispatch(
        &self,
        compute: &mut ComputeCommandEncoder,
        kernel: &ComputePipelineState,
        (a, a_offset): (&Buffer, u64),
        (b, b_offset): (&Buffer, u64),
        indirect: Option<(&Buffer, u64)>,
    ) -> Answer<()> {
        let mut bound = vec![a, b];
        bound.extend(indirect.map(|(arguments, _)| arguments));
        self.note(Call::Dispatch, &bound);
        compute
            .set_compute_pipeline_state(kernel)
            .map_err(refused)?;
        compute.set_buffer(Some(a), a_offset, 0).map_err(refused)?;
        compute.set_buffer(Some(b), b_offset, 1).map_err(refused)?;
        let one = Size::new(1, 1, 1);
        match indirect {
            Some((arguments, offset)) => compute
                .dispatch_threadgroups_indirect(arguments, offset, one)
                .map_err(refused),
            None => compute.dispatch_threads(one, one).map_err(refused),
        }
    }

    fn texel_dispatch(
        &self,
        compute: &mut ComputeCommandEncoder,
        kernel: &ComputePipelineState,
        texture: &Texture,
        (words, offset): (&Buffer, u64),
        (width, rows): (u32, u32),
    ) -> Answer<()> {
        self.note(Call::TexelDispatch, &[words]);
        compute
            .set_compute_pipeline_state(kernel)
            .map_err(refused)?;
        compute.set_texture(Some(texture), 0).map_err(refused)?;
        compute
            .set_buffer(Some(words), offset, 0)
            .map_err(refused)?;
        compute
            .dispatch_threads(
                Size::new(u64::from(width), u64::from(rows), 1),
                Size::new(u64::from(width), 1, 1),
            )
            .map_err(refused)
    }

    fn barrier(&self, compute: &mut ComputeCommandEncoder, textures: bool) -> Answer<()> {
        let scope = if textures {
            BarrierScope::BUFFERS | BarrierScope::TEXTURES
        } else {
            BarrierScope::BUFFERS
        };
        compute.memory_barrier_with_scope(scope);
        Ok(())
    }

    fn compute_update_fence(
        &self,
        compute: &mut ComputeCommandEncoder,
        fence: &Fence,
    ) -> Answer<()> {
        compute.update_fence(fence).map_err(refused)
    }

    fn compute_wait_fence(&self, compute: &mut ComputeCommandEncoder, fence: &Fence) -> Answer<()> {
        compute.wait_for_fence(fence).map_err(refused)
    }

    fn end_compute(&self, compute: ComputeCommandEncoder) -> Answer<()> {
        compute.end_encoding().map_err(refused)
    }

    fn render(
        &self,
        cb: &CommandBuffer,
        (target, slice): (&Texture, u32),
        load: Load,
    ) -> Answer<RenderCommandEncoder> {
        let mut descriptor = RenderPassDescriptor::new();
        let attachment = &mut descriptor.color_attachments[0];
        attachment.texture = Some(target.clone());
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
        cb.render_command_encoder(&descriptor).map_err(refused)
    }

    fn draw(
        &self,
        render: &mut RenderCommandEncoder,
        state: &RenderPipelineState,
        (rects, rects_offset): (&Buffer, u64),
        (colors, colors_offset): (&Buffer, u64),
        (vertices, instances): (u32, u32),
        indirect: Option<(&Buffer, u64)>,
    ) -> Answer<()> {
        let mut bound = vec![rects, colors];
        bound.extend(indirect.map(|(arguments, _)| arguments));
        self.note(Call::Draw, &bound);
        render.set_render_pipeline_state(state).map_err(refused)?;
        render
            .set_vertex_buffer(Some(rects), rects_offset, 0)
            .map_err(refused)?;
        render
            .set_fragment_buffer(Some(colors), colors_offset, 0)
            .map_err(refused)?;
        match indirect {
            Some((arguments, offset)) => render
                .draw_primitives_indirect(PrimitiveType::Triangle, arguments, offset)
                .map_err(refused),
            None => render
                .draw_primitives(PrimitiveType::Triangle, 0, vertices, instances, 0)
                .map_err(refused),
        }
    }

    fn render_update_fence(&self, render: &mut RenderCommandEncoder, fence: &Fence) -> Answer<()> {
        render
            .update_fence(fence, RenderStages::FRAGMENT)
            .map_err(refused)
    }

    fn render_wait_fence(&self, render: &mut RenderCommandEncoder, fence: &Fence) -> Answer<()> {
        render
            .wait_for_fence(fence, RenderStages::VERTEX)
            .map_err(refused)
    }

    fn end_render(&self, render: RenderCommandEncoder) -> Answer<()> {
        render.end_encoding().map_err(refused)
    }

    fn commit(&self, cb: &CommandBuffer) -> Answer<()> {
        let number = self.committed.get();
        self.committed.set(number + 1);
        cb.commit().map_err(refused)?;
        self.unscheduled.borrow_mut().push((number, cb.clone()));
        self.take_messages();
        Ok(())
    }

    fn completion(&self, cb: CommandBuffer) -> Result<Completion, String> {
        cb.wait_until_completed_for(WAIT)
            .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
        let status = match cb.status() {
            CommandBufferStatus::Completed => "completed",
            CommandBufferStatus::Error => "error",
            other => return Err(format!("a completed command buffer is {other:?}")),
        };
        Ok(Completion {
            status: status.to_owned(),
            times: None,
        })
    }
}
