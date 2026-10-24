//! Running a [`Work`] program on `metal2vulkan_runtime`.
//!
//! The interpreter makes the program's resources in order, encodes its steps into one blit encoder
//! at a time, and answers the reads. Every call answers the runtime's typed error, so a resource
//! the venue cannot make, or a copy it refuses, comes back as a [`Failure`] carrying what the
//! venue lacks where the refusal names a capability — which is how the suite tells the venue's
//! honest gap from ours without reading a device name.
//!
//! A wait is bounded, so no case here can hang: a command buffer that does not complete inside the
//! bound is a recorded fact, not a block.

use crate::case::{Answer, Failure};
use crate::work::{Clear, Done, Plane, Rect, Resource, Step, TextureKind, TextureSpec, Work};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, Fence,
    Heap, HeapDescriptor, HeapType, LoadAction, Origin, PipelineOptions, Region,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use std::time::Duration;

/// The longest the interpreter waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// A resource the program made, or a slot a [`Step::Release`] emptied.
enum Made {
    Buffer(Buffer),
    Heap(Heap),
    Texture(Texture),
    Fence(Fence),
    /// The resource a [`Step::Release`] dropped. It is an explicit state rather than a hole: a
    /// step that names a released resource is refused by the same path that refuses one that was
    /// never made, and a [`Step::Make`] fills the slot again.
    Released,
}

/// One of the resource kinds a call that takes Metal's `id<MTLResource>` can be handed, held so
/// the encoder can be borrowed after the pool was read ([`Run::resource`]).
enum OneResource {
    Buffer(Buffer),
    Texture(Texture),
}

/// A read-back the device has to run before its bytes can be answered.
struct Pending {
    /// The read's place among the program's reads.
    slot: usize,
    buffer: Buffer,
    length: usize,
    /// The bytes of one sample the read keeps, and the words the kernel wrote, where the read is a
    /// per-sample one whose words must be narrowed.
    samples: Option<(usize, usize)>,
}

/// Run `work` on `device`, answering what it read.
pub fn run(device: &Device, queue: &CommandQueue, work: &Work) -> Answer<Done> {
    let mut made: Vec<Made> = Vec::with_capacity(work.resources.len());
    for (at, resource) in work.resources.iter().enumerate() {
        made.push(make(device, &made, resource).map_err(|failure| failure.at_resource(at))?);
    }
    let mut state = Run {
        device,
        queue,
        made,
        resources: &work.resources,
        kernels: work.kernels,
        command_buffer: None,
        encoder: None,
        pending: Vec::new(),
        reads: vec![None; work.reads()],
        next_read: 0,
        statuses: Vec::new(),
    };
    for (at, step) in work.steps.iter().enumerate() {
        state.step(step).map_err(|failure| failure.at_step(at))?;
    }
    state.close()?;
    let reads = state
        .reads
        .into_iter()
        .enumerate()
        .map(|(slot, bytes)| {
            bytes.ok_or_else(|| Failure::named("UnansweredRead", format!("read {slot}")))
        })
        .collect::<Answer<Vec<Vec<u8>>>>()?;
    Ok(Done {
        reads,
        statuses: state.statuses,
    })
}

/// The descriptor `spec` names.
fn descriptor(spec: &TextureSpec) -> Answer<TextureDescriptor> {
    let format = PixelFormat::try_from(spec.format).map_err(|problem| {
        Failure::named("NoSuchPixelFormat", format!("{}: {problem:?}", spec.format))
    })?;
    let mut usage = TextureUsage::SHADER_READ;
    if spec.render_target {
        usage |= TextureUsage::RENDER_TARGET | TextureUsage::PIXEL_FORMAT_VIEW;
    }
    if let Some(stated) = spec.usage {
        usage = TextureUsage::try_from(stated).map_err(|problem| {
            Failure::named("NoSuchTextureUsage", format!("{stated}: {problem:?}"))
        })?;
    }
    if spec.kind == TextureKind::TextureBuffer {
        return Ok(TextureDescriptor::texture_buffer(
            format,
            spec.width,
            crate::runtime_side::resource_options(spec.storage),
            usage,
        ));
    }
    let mut d = TextureDescriptor::texture_2d(format, spec.width, spec.height, false);
    d.texture_type = match spec.kind {
        TextureKind::Type2D => TextureType::Type2D,
        TextureKind::Type2DArray => TextureType::Type2DArray,
        TextureKind::Cube => TextureType::Cube,
        TextureKind::Type3D => TextureType::Type3D,
        TextureKind::Type2DMultisample => TextureType::Type2DMultisample,
        TextureKind::Type2DMultisampleArray => TextureType::Type2DMultisampleArray,
        TextureKind::TextureBuffer => TextureType::TextureBuffer,
    };
    d.depth = spec.depth.max(1);
    d.array_length = spec.slices.max(1);
    d.mipmap_level_count = spec.levels.max(1);
    d.sample_count = spec.samples.max(1);
    d.usage = usage;
    d.allow_gpu_optimized_contents = spec.gpu_optimized;
    d.resource_options = crate::runtime_side::resource_options(spec.storage);
    Ok(d)
}

/// Make one resource.
fn make(device: &Device, made: &[Made], resource: &Resource) -> Answer<Made> {
    let buffer_at = |at: usize| match made.get(at) {
        Some(Made::Buffer(buffer)) => Ok(buffer),
        _ => Err(Failure::named("NoSuchResource", format!("buffer {at}"))),
    };
    let heap_at = |at: usize| match made.get(at) {
        Some(Made::Heap(heap)) => Ok(heap),
        _ => Err(Failure::named("NoSuchResource", format!("heap {at}"))),
    };
    let texture_at = |at: usize| match made.get(at) {
        Some(Made::Texture(texture)) => Ok(texture),
        _ => Err(Failure::named("NoSuchResource", format!("texture {at}"))),
    };
    match resource {
        Resource::Buffer { bytes } => {
            let buffer = device
                .new_buffer(bytes.len().max(1) as u64, ResourceOptions::shared())
                .map_err(|error| Failure::of(&error))?;
            let contents = buffer
                .contents()
                .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
            // SAFETY: the buffer was just made with at least `bytes.len()` bytes, the host
            // reaches them, and no command uses it yet.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len());
            }
            Ok(Made::Buffer(buffer))
        }
        Resource::UnwrittenBuffer { length, storage } => {
            let buffer = device
                .new_buffer(
                    (*length).max(1),
                    crate::runtime_side::resource_options(*storage),
                )
                .map_err(|error| Failure::of(&error))?;
            Ok(Made::Buffer(buffer))
        }
        Resource::Fence => Ok(Made::Fence(device.new_fence())),
        Resource::Heap { size, storage } => {
            let options = crate::runtime_side::resource_options(*storage);
            let heap = device
                .new_heap(&HeapDescriptor::new(HeapType::Placement, *size, options))
                .map_err(|error| Failure::of(&error))?;
            Ok(Made::Heap(heap))
        }
        Resource::HeapForTexture { spec, storage } => {
            let options = crate::runtime_side::resource_options(*storage);
            let size = device
                .heap_texture_size_and_align(&descriptor(spec)?)
                .map_err(|error| Failure::of(&error))?
                .size;
            let heap = device
                .new_heap(&HeapDescriptor::new(HeapType::Placement, size, options))
                .map_err(|error| Failure::of(&error))?;
            Ok(Made::Heap(heap))
        }
        Resource::HeapBuffer {
            heap,
            length,
            offset,
        } => {
            let held = heap_at(*heap)?;
            let options = held.resource_options();
            let buffer = held
                .new_buffer_at_offset(*length, options, *offset)
                .map_err(|error| Failure::of(&error))?;
            Ok(Made::Buffer(buffer))
        }
        Resource::Texture { spec } => {
            let texture = device
                .new_texture(&descriptor(spec)?)
                .map_err(|error| Failure::of(&error))?;
            Ok(Made::Texture(texture))
        }
        Resource::HeapTexture { heap, spec, offset } => {
            let texture = heap_at(*heap)?
                .new_texture_at_offset(&descriptor(spec)?, *offset)
                .map_err(|error| Failure::of(&error))?;
            Ok(Made::Texture(texture))
        }
        Resource::TextureOverBuffer {
            buffer,
            spec,
            offset,
            bytes_per_row,
        } => {
            let texture = buffer_at(*buffer)?
                .new_texture(&descriptor(spec)?, *offset, *bytes_per_row)
                .map_err(|error| Failure::of(&error))?;
            Ok(Made::Texture(texture))
        }
        Resource::View {
            texture,
            format,
            level,
            level_count,
            slice,
            slice_count,
            flatten,
        } => {
            let held = texture_at(*texture)?;
            let format = PixelFormat::try_from(*format).map_err(|problem| {
                Failure::named("NoSuchPixelFormat", format!("{format}: {problem:?}"))
            })?;
            let kind = if *flatten {
                TextureType::Type2DMultisample
            } else {
                held.texture_type()
            };
            let view = held
                .new_texture_view_with_levels_slices(
                    format,
                    kind,
                    *level..*level + *level_count,
                    *slice..*slice + *slice_count,
                )
                .map_err(|error| Failure::of(&error))?;
            Ok(Made::Texture(view))
        }
    }
}

/// The interpreter's state as it runs a program.
struct Run<'a> {
    device: &'a Device,
    queue: &'a CommandQueue,
    made: Vec<Made>,
    /// The program's declarations, which a [`Step::Make`] reads to make a released resource again.
    resources: &'a [Resource],
    kernels: Option<&'static str>,
    command_buffer: Option<CommandBuffer>,
    encoder: Option<BlitCommandEncoder>,
    pending: Vec<Pending>,
    reads: Vec<Option<Vec<u8>>>,
    next_read: usize,
    statuses: Vec<String>,
}

impl Run<'_> {
    /// The buffer resource `at`.
    fn buffer(&self, at: usize) -> Answer<Buffer> {
        match self.made.get(at) {
            Some(Made::Buffer(buffer)) => Ok(buffer.clone()),
            _ => Err(Failure::named("NoSuchResource", format!("buffer {at}"))),
        }
    }

    /// The texture resource `at`.
    fn texture(&self, at: usize) -> Answer<Texture> {
        match self.made.get(at) {
            Some(Made::Texture(texture)) => Ok(texture.clone()),
            _ => Err(Failure::named("NoSuchResource", format!("texture {at}"))),
        }
    }

    /// The fence resource `at`.
    fn fence(&self, at: usize) -> Answer<Fence> {
        match self.made.get(at) {
            Some(Made::Fence(fence)) => Ok(fence.clone()),
            _ => Err(Failure::named("NoSuchResource", format!("fence {at}"))),
        }
    }

    /// The resource `at` as one of the two kinds a call over `id<MTLResource>` takes
    /// ([`Step::SynchronizeResource`]).
    fn resource(&self, at: usize) -> Answer<OneResource> {
        match self.made.get(at) {
            Some(Made::Buffer(buffer)) => Ok(OneResource::Buffer(buffer.clone())),
            Some(Made::Texture(texture)) => Ok(OneResource::Texture(texture.clone())),
            _ => Err(Failure::named("NoSuchResource", format!("resource {at}"))),
        }
    }

    /// The command buffer the steps encode into, made on first use.
    fn command_buffer(&mut self) -> Answer<CommandBuffer> {
        if self.command_buffer.is_none() {
            self.command_buffer = Some(
                self.queue
                    .command_buffer()
                    .map_err(|error| Failure::of(&error))?,
            );
        }
        Ok(self
            .command_buffer
            .clone()
            .expect("the command buffer was just made"))
    }

    /// The blit encoder the copies and fills go into, opened on first use.
    fn encoder(&mut self) -> Answer<&mut BlitCommandEncoder> {
        if self.encoder.is_none() {
            let command_buffer = self.command_buffer()?;
            self.encoder = Some(
                command_buffer
                    .blit_command_encoder()
                    .map_err(|error| Failure::of(&error))?,
            );
        }
        Ok(self.encoder.as_mut().expect("the encoder was just opened"))
    }

    /// The slot the next read answers.
    fn slot(&mut self) -> usize {
        let slot = self.next_read;
        self.next_read += 1;
        slot
    }

    /// Run one step.
    fn step(&mut self, step: &Step) -> Answer<()> {
        match step {
            Step::FillTexture {
                texture,
                level,
                slice,
                region,
                pattern,
            } => {
                let texture = self.texture(*texture)?;
                self.encoder()?
                    .fill_texture(&texture, *level, *slice, region_of(region), pattern)
                    .map_err(|error| Failure::of(&error))
            }
            Step::FillTextureColor {
                texture,
                level,
                slice,
                region,
                color,
                packed_in,
            } => {
                let texture = self.texture(*texture)?;
                let packed = packed_in
                    .map(|raw| {
                        PixelFormat::try_from(raw).map_err(|problem| {
                            Failure::named("NoSuchPixelFormat", format!("{raw}: {problem:?}"))
                        })
                    })
                    .transpose()?;
                let [r, g, b, a] = *color;
                self.encoder()?
                    .fill_texture_color(
                        &texture,
                        *level,
                        *slice,
                        region_of(region),
                        ClearColor::new(r, g, b, a),
                        packed,
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::CopyBufferToTexture {
                buffer,
                offset,
                bytes_per_row,
                bytes_per_image,
                size,
                texture,
                slice,
                level,
                origin,
                plane,
            } => {
                let buffer = self.buffer(*buffer)?;
                let texture = self.texture(*texture)?;
                self.encoder()?
                    .copy_from_buffer_to_texture(
                        &buffer,
                        *offset,
                        *bytes_per_row,
                        *bytes_per_image,
                        size_of(size),
                        &texture,
                        *slice,
                        *level,
                        origin_of(origin),
                        option_of(*plane),
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::CopyTextureToBuffer {
                texture,
                slice,
                level,
                origin,
                size,
                buffer,
                offset,
                bytes_per_row,
                bytes_per_image,
                plane,
            } => {
                let buffer = self.buffer(*buffer)?;
                let texture = self.texture(*texture)?;
                self.encoder()?
                    .copy_from_texture_to_buffer(
                        &texture,
                        *slice,
                        *level,
                        origin_of(origin),
                        size_of(size),
                        &buffer,
                        *offset,
                        *bytes_per_row,
                        *bytes_per_image,
                        option_of(*plane),
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::CopyTextureToTexture {
                from,
                from_slice,
                from_level,
                from_origin,
                size,
                to,
                to_slice,
                to_level,
                to_origin,
            } => {
                let source = self.texture(*from)?;
                let destination = self.texture(*to)?;
                self.encoder()?
                    .copy_from_texture_to_texture(
                        &source,
                        *from_slice,
                        *from_level,
                        origin_of(from_origin),
                        size_of(size),
                        &destination,
                        *to_slice,
                        *to_level,
                        origin_of(to_origin),
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::CopyBufferToTextureWithoutOptions {
                buffer,
                offset,
                bytes_per_row,
                bytes_per_image,
                size,
                texture,
                slice,
                level,
                origin,
            } => {
                let buffer = self.buffer(*buffer)?;
                let texture = self.texture(*texture)?;
                // Metal binds two selectors and the runtime one method: the un-optioned selector
                // **is** the optioned one with `MTLBlitOptionNone`, which is what Apple's header
                // says of it, so our leg reaches the same call with [`BlitOption::None`]. What a
                // case over this step can still fail on is Metal's own two selectors disagreeing,
                // which is where the item is unexercised.
                self.encoder()?
                    .copy_from_buffer_to_texture(
                        &buffer,
                        *offset,
                        *bytes_per_row,
                        *bytes_per_image,
                        size_of(size),
                        &texture,
                        *slice,
                        *level,
                        origin_of(origin),
                        BlitOption::None,
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::CopyTextureToBufferWithoutOptions {
                texture,
                slice,
                level,
                origin,
                size,
                buffer,
                offset,
                bytes_per_row,
                bytes_per_image,
            } => {
                let buffer = self.buffer(*buffer)?;
                let texture = self.texture(*texture)?;
                // One method for Metal's two selectors, as above.
                self.encoder()?
                    .copy_from_texture_to_buffer(
                        &texture,
                        *slice,
                        *level,
                        origin_of(origin),
                        size_of(size),
                        &buffer,
                        *offset,
                        *bytes_per_row,
                        *bytes_per_image,
                        BlitOption::None,
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::CopyWholeTexture { from, to } => {
                let source = self.texture(*from)?;
                let destination = self.texture(*to)?;
                self.encoder()?
                    .copy_texture(&source, &destination)
                    .map_err(|error| Failure::of(&error))
            }
            Step::CopyTextureSlices {
                from,
                from_slice,
                from_level,
                to,
                to_slice,
                to_level,
                slice_count,
                level_count,
            } => {
                let source = self.texture(*from)?;
                let destination = self.texture(*to)?;
                self.encoder()?
                    .copy_texture_slices(
                        &source,
                        *from_slice,
                        *from_level,
                        &destination,
                        *to_slice,
                        *to_level,
                        *slice_count,
                        *level_count,
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::GenerateMipmaps { texture } => {
                let texture = self.texture(*texture)?;
                self.encoder()?
                    .generate_mipmaps(&texture)
                    .map_err(|error| Failure::of(&error))
            }
            Step::OptimizeForGpuAccess { texture, at } => {
                let texture = self.texture(*texture)?;
                let encoder = self.encoder()?;
                match at {
                    None => encoder.optimize_contents_for_gpu_access(&texture),
                    Some((slice, level)) => encoder
                        .optimize_contents_for_gpu_access_slice_level(&texture, *slice, *level),
                }
                .map_err(|error| Failure::of(&error))
            }
            Step::OptimizeForCpuAccess { texture, at } => {
                let texture = self.texture(*texture)?;
                let encoder = self.encoder()?;
                match at {
                    None => encoder.optimize_contents_for_cpu_access(&texture),
                    Some((slice, level)) => encoder
                        .optimize_contents_for_cpu_access_slice_level(&texture, *slice, *level),
                }
                .map_err(|error| Failure::of(&error))
            }
            Step::SynchronizeResource { resource } => {
                let held = self.resource(*resource)?;
                let encoder = self.encoder()?;
                match &held {
                    OneResource::Buffer(buffer) => encoder.synchronize_resource(buffer.into()),
                    OneResource::Texture(texture) => encoder.synchronize_resource(texture.into()),
                }
                .map_err(|error| Failure::of(&error))
            }
            Step::SynchronizeTexture {
                texture,
                slice,
                level,
            } => {
                let texture = self.texture(*texture)?;
                self.encoder()?
                    .synchronize_texture(&texture, *slice, *level)
                    .map_err(|error| Failure::of(&error))
            }
            Step::CopyBufferToBuffer {
                from,
                from_offset,
                to,
                to_offset,
                length,
            } => {
                let source = self.buffer(*from)?;
                let destination = self.buffer(*to)?;
                self.encoder()?
                    .copy_from_buffer_to_buffer(
                        &source,
                        *from_offset,
                        &destination,
                        *to_offset,
                        *length,
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::FillBuffer {
                buffer,
                offset,
                length,
                value,
            } => {
                let buffer = self.buffer(*buffer)?;
                self.encoder()?
                    .fill_buffer(&buffer, *offset..*offset + *length, *value)
                    .map_err(|error| Failure::of(&error))
            }
            Step::Release { resource } => self.release(*resource),
            Step::Make { resource } => self.make_again(*resource),
            Step::ClearPass {
                texture,
                slice,
                clear,
            } => self.clear_pass(*texture, *slice, *clear),
            Step::ReplaceRegion {
                texture,
                level,
                slice,
                region,
                bytes,
                // The runtime checks the slice it is handed against the region and the pitches
                // itself, and answers a typed error, so the stated rows are the Metal side's
                // guard alone.
                rows: _,
                bytes_per_row,
                bytes_per_image,
            } => {
                let texture = self.texture(*texture)?;
                texture
                    .replace_region(
                        region_of(region),
                        *level,
                        *slice,
                        bytes,
                        *bytes_per_row,
                        *bytes_per_image,
                    )
                    .map_err(|error| Failure::of(&error))
            }
            Step::GetBytes {
                texture,
                level,
                slice,
                region,
                bytes_per_row,
                bytes_per_image,
                length,
                before,
            } => {
                let texture = self.texture(*texture)?;
                let mut bytes = vec![*before; *length];
                texture
                    .get_bytes(
                        &mut bytes,
                        *bytes_per_row,
                        *bytes_per_image,
                        region_of(region),
                        *level,
                        *slice,
                    )
                    .map_err(|error| Failure::of(&error))?;
                let slot = self.slot();
                self.reads[slot] = Some(bytes);
                Ok(())
            }
            Step::ReadBuffer {
                buffer,
                offset,
                length,
            } => {
                let buffer = self.buffer(*buffer)?;
                let bytes = read_buffer(&buffer, *offset, *length)?;
                let slot = self.slot();
                self.reads[slot] = Some(bytes);
                Ok(())
            }
            Step::ReadSamples {
                texture,
                kernel,
                width,
                height,
                samples,
                texel_bytes,
                words,
            } => {
                let slot = self.slot();
                self.read_samples(
                    slot,
                    *texture,
                    kernel,
                    [*width, *height],
                    *samples,
                    *texel_bytes,
                    *words,
                )
            }
            Step::EndEncoder => self.end_encoding(),
            Step::UpdateFence { fence } => {
                let fence = self.fence(*fence)?;
                self.encoder()?
                    .update_fence(&fence)
                    .map_err(|error| Failure::of(&error))
            }
            Step::WaitForFence { fence } => {
                let fence = self.fence(*fence)?;
                self.encoder()?
                    .wait_for_fence(&fence)
                    .map_err(|error| Failure::of(&error))
            }
            Step::Commit => self.close(),
        }
    }

    /// Drop the side's reference to the resource at `at`.
    ///
    /// Refused while a command buffer is open: a resource an uncommitted command buffer refers to
    /// is not freed by this, so a case that released one there would read a fresh resource that
    /// could not have taken the old one's memory and would prove less than it says.
    fn release(&mut self, at: usize) -> Answer<()> {
        if self.command_buffer.is_some() || self.encoder.is_some() {
            return Err(Failure::named(
                "ReleaseWhileEncoding",
                format!("resource {at}, with a command buffer still open"),
            ));
        }
        match self.made.get(at) {
            Some(Made::Released) | None => Err(Failure::named(
                "NoSuchResource",
                format!("resource {at} is not made"),
            )),
            // The assignment drops what the slot held, which is the release.
            Some(_) => {
                self.made[at] = Made::Released;
                Ok(())
            }
        }
    }

    /// Make the resource at `at` again, from its own declaration.
    fn make_again(&mut self, at: usize) -> Answer<()> {
        match self.made.get(at) {
            None => {
                return Err(Failure::named(
                    "NoSuchResource",
                    format!("resource {at} is not declared"),
                ))
            }
            Some(Made::Released) => {}
            Some(_) => {
                return Err(Failure::named(
                    "AlreadyMade",
                    format!("resource {at} is still made; release it first"),
                ))
            }
        }
        let declared = self
            .resources
            .get(at)
            .ok_or_else(|| Failure::named("NoSuchResource", format!("resource {at}")))?;
        let one = make(self.device, &self.made, declared)?;
        self.made[at] = one;
        Ok(())
    }

    /// A render pass over one slice that clears it and draws nothing.
    fn clear_pass(&mut self, texture: usize, slice: u64, clear: Clear) -> Answer<()> {
        // A pass must not open while the blit encoder is: Metal takes one encoder at a time.
        self.end_encoding()?;
        let texture = self.texture(texture)?;
        let command_buffer = self.command_buffer()?;
        let mut pass = RenderPassDescriptor::new();
        match clear {
            Clear::Color([r, g, b, a]) => {
                let attachment = &mut pass.color_attachments[0];
                attachment.texture = Some(texture);
                attachment.slice = slice;
                attachment.load_action = LoadAction::Clear;
                attachment.clear_color = ClearColor::new(r, g, b, a);
                attachment.store_action = StoreAction::Store;
            }
            Clear::Depth(depth) => {
                let attachment = &mut pass.depth_attachment;
                attachment.texture = Some(texture);
                attachment.slice = slice;
                attachment.load_action = LoadAction::Clear;
                attachment.clear_depth = depth;
                attachment.store_action = StoreAction::Store;
            }
            Clear::Stencil(stencil) => {
                let attachment = &mut pass.stencil_attachment;
                attachment.texture = Some(texture);
                attachment.slice = slice;
                attachment.load_action = LoadAction::Clear;
                attachment.clear_stencil = stencil;
                attachment.store_action = StoreAction::Store;
            }
            Clear::DepthStencil(depth, stencil) => {
                let depth_attachment = &mut pass.depth_attachment;
                depth_attachment.texture = Some(texture.clone());
                depth_attachment.slice = slice;
                depth_attachment.load_action = LoadAction::Clear;
                depth_attachment.clear_depth = depth;
                depth_attachment.store_action = StoreAction::Store;
                let stencil_attachment = &mut pass.stencil_attachment;
                stencil_attachment.texture = Some(texture);
                stencil_attachment.slice = slice;
                stencil_attachment.load_action = LoadAction::Clear;
                stencil_attachment.clear_stencil = stencil;
                stencil_attachment.store_action = StoreAction::Store;
            }
        }
        let encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(|error| Failure::of(&error))?;
        encoder.end_encoding().map_err(|error| Failure::of(&error))
    }

    /// Encode a per-sample read-back of `texture` by the committed fixture's `kernel`.
    #[allow(clippy::too_many_arguments)]
    fn read_samples(
        &mut self,
        slot: usize,
        texture: usize,
        kernel: &str,
        [width, height]: [u32; 2],
        samples: u32,
        texel_bytes: usize,
        words: usize,
    ) -> Answer<()> {
        self.end_encoding()?;
        let source = self.texture(texture)?;
        let fixture = self.kernels.ok_or_else(|| {
            Failure::named("NoKernels", "the program states no fixture for its kernels")
        })?;
        let library = self.library(fixture)?;
        let function = library
            .new_function_with_name(kernel)
            .map_err(|error| Failure::of(&error))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .map_err(|error| Failure::of(&error))?;
        let length = width as usize * height as usize * samples as usize * words * 4;
        let out = self
            .device
            .new_buffer(length.max(1) as u64, ResourceOptions::shared())
            .map_err(|error| Failure::of(&error))?;
        // The shape the fixture's kernels read: the extent's width and each pixel's samples.
        let mut shape_bytes = width.to_le_bytes().to_vec();
        shape_bytes.extend_from_slice(&samples.to_le_bytes());
        let shape = self
            .device
            .new_buffer(shape_bytes.len() as u64, ResourceOptions::shared())
            .map_err(|error| Failure::of(&error))?;
        let contents = shape
            .contents()
            .ok_or_else(|| Failure::named("NoContents", "the shape buffer"))?;
        // SAFETY: the buffer was just made with exactly `shape_bytes.len()` bytes, the host
        // reaches them, and no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                shape_bytes.as_ptr(),
                contents.as_ptr(),
                shape_bytes.len(),
            );
        }
        let command_buffer = self.command_buffer()?;
        let mut encoder = command_buffer
            .compute_command_encoder()
            .map_err(|error| Failure::of(&error))?;
        let failed = |error: metal2vulkan_runtime::Error| Failure::of(&error);
        encoder
            .set_compute_pipeline_state(&pipeline)
            .map_err(failed)?;
        encoder.set_texture(Some(&source), 0).map_err(failed)?;
        encoder.set_buffer(Some(&out), 0, 0).map_err(failed)?;
        encoder.set_buffer(Some(&shape), 0, 1).map_err(failed)?;
        // One thread a sample, one thread a threadgroup: no grid edge to guard.
        encoder
            .dispatch_threadgroups(
                Size::new(width.into(), height.into(), samples.into()),
                Size::new(1, 1, 1),
            )
            .map_err(failed)?;
        encoder.end_encoding().map_err(failed)?;
        self.pending.push(Pending {
            slot,
            buffer: out,
            length,
            samples: Some((texel_bytes, words * 4)),
        });
        Ok(())
    }

    /// The fixture's library, assembled from its committed text.
    fn library(&self, fixture: &str) -> Answer<metal2vulkan_runtime::Library> {
        let path = engine_dir().join(format!("{fixture}{}", crate::fixture::RUNTIME_TEXT));
        let bitcode = llvm_assemble(&path).map_err(|error| {
            Failure::named("NoFixture", format!("{}: {error:?}", path.display()))
        })?;
        let library = metal2vulkan::library::Library::parse(&bitcode).map_err(|error| {
            Failure::named("BadFixture", format!("{}: {error}", path.display()))
        })?;
        Ok(self.device.new_library_with_compile_library(library))
    }

    /// Close the blit encoder, if one is open.
    fn end_encoding(&mut self) -> Answer<()> {
        match self.encoder.take() {
            Some(encoder) => encoder.end_encoding().map_err(|error| Failure::of(&error)),
            None => Ok(()),
        }
    }

    /// Close the encoder, commit the command buffer, wait for it, record its status and answer
    /// every read-back asked for so far.
    fn close(&mut self) -> Answer<()> {
        self.end_encoding()?;
        let Some(command_buffer) = self.command_buffer.take() else {
            return Ok(());
        };
        command_buffer
            .commit()
            .map_err(|error| Failure::of(&error))?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            self.statuses.push("NeverReturned".to_owned());
            return Err(Failure::named(
                "WaitRanOut",
                "the command buffer did not complete inside the bound",
            ));
        }
        self.statuses.push(format!("{:?}", command_buffer.status()));
        let pending = std::mem::take(&mut self.pending);
        for read in pending {
            let bytes = read_buffer(&read.buffer, 0, read.length)?;
            let bytes = match read.samples {
                Some((texel_bytes, stride)) => bytes
                    .chunks(stride)
                    .flat_map(|chunk| chunk.iter().take(texel_bytes))
                    .copied()
                    .collect(),
                None => bytes,
            };
            self.reads[read.slot] = Some(bytes);
        }
        Ok(())
    }
}

/// `length` bytes of `buffer` from `offset`, as the host reads a shared buffer.
fn read_buffer(buffer: &Buffer, offset: u64, length: usize) -> Answer<Vec<u8>> {
    let past = offset
        .checked_add(length as u64)
        .is_none_or(|end| end > buffer.length());
    if past {
        return Err(Failure::named(
            "PastTheBuffer",
            format!(
                "{length} bytes at {offset} of a buffer of {}",
                buffer.length()
            ),
        ));
    }
    let contents = buffer
        .contents()
        .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
    // SAFETY: `[offset, offset + length)` is inside the buffer's bytes, the host reaches them,
    // and the caller waited for the command buffer that wrote them.
    let bytes =
        unsafe { std::slice::from_raw_parts(contents.as_ptr().add(offset as usize), length) };
    Ok(bytes.to_vec())
}

/// The region a rectangle names.
fn region_of(rect: &Rect) -> Region {
    Region {
        origin: origin_of(&rect.origin),
        size: size_of(&rect.size),
    }
}

/// The origin an axis triple names.
fn origin_of(origin: &[u64; 3]) -> Origin {
    Origin::new(origin[0], origin[1], origin[2])
}

/// The size an axis triple names.
fn size_of(size: &[u64; 3]) -> Size {
    Size::new(size[0], size[1], size[2])
}

/// The blit option a plane names. The suite's one mapping of [`Plane`] onto the runtime's own
/// option, which a render case's texture read takes too.
pub(crate) fn option_of(plane: Plane) -> BlitOption {
    match plane {
        Plane::Whole => BlitOption::None,
        Plane::Depth => BlitOption::DepthFromDepthStencil,
        Plane::Stencil => BlitOption::StencilFromDepthStencil,
        Plane::RowLinearPvrtc => BlitOption::RowLinearPVRTC,
    }
}
