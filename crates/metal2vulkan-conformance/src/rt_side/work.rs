//! Running a [`Work`] program on `metal2vulkan-rt`, as [`crate::runtime_work`] runs it on the old
//! runtime: its resources made in order, its steps encoded into one blit encoder at a time, its
//! reads answered, each wait bounded.
//!
//! The runtime places memory where its consumer says, and this side is the consumer: a buffer
//! takes a block of its own, a Managed buffer a Shared block for its bytes and, where the device
//! keeps a second copy, a Private block for it, and a placement heap is a block of the heap's
//! storage that its buffers and textures are placed in. A texture a shader reaches takes the next
//! texture index.
//!
//! A step the runtime has no counterpart of answers `Unavailable` with its Metal call: the two
//! fills (Apple SPI, whose Metal answers are carried), the whole-texture and slice copies,
//! the two optimize calls, `synchronizeTexture:slice:level:`, and a
//! copy with a depth, stencil or PVRTC blit option. A heap other than Private or Shared, and a
//! buffer of Memoryless storage, which Metal makes no buffer of, are unavailable too.

use super::{waited_for, RtDevice, WAIT};
use crate::case::{Answer, Failure};
use crate::texture_query::Storage;
use crate::work::{Clear, Done, Plane, Rect, Resource, Step, TextureKind, TextureSpec, Work};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_rt as rt;
use std::collections::BTreeSet;

/// A typed error of the runtime as the failure it records.
fn of_rt(error: rt::Error) -> Failure {
    Failure::of_rt(&error)
}

/// `value` as the runtime's `u32`, which Metal's `NSUInteger` arguments are narrowed to.
pub(super) fn narrow(value: u64, what: &str) -> Answer<u32> {
    u32::try_from(value).map_err(|_| {
        Failure::named(
            "PastU32",
            format!("{what} {value}: the runtime takes a u32"),
        )
    })
}

/// Each of three axes narrowed ([`narrow`]).
fn narrow3(values: &[u64; 3], what: &str) -> Answer<[u32; 3]> {
    Ok([
        narrow(values[0], what)?,
        narrow(values[1], what)?,
        narrow(values[2], what)?,
    ])
}

/// The region a rectangle names.
fn region_of(rect: &Rect) -> Answer<rt::Region> {
    Ok(rt::Region {
        origin: narrow3(&rect.origin, "an origin")?,
        size: narrow3(&rect.size, "a size")?,
    })
}

/// The runtime's storage mode of a case's.
pub(super) fn storage_mode(storage: Storage) -> rt::StorageMode {
    match storage {
        Storage::Shared => rt::StorageMode::Shared,
        Storage::Managed => rt::StorageMode::Managed,
        Storage::Private => rt::StorageMode::Private,
        Storage::Memoryless => rt::StorageMode::Memoryless,
    }
}

/// The descriptor `spec` names. The runtime's descriptor has no `allowGPUOptimizedContents`
/// ([`Side::texture`](crate::case::Side::texture) on this side says why it asks nothing).
fn descriptor(spec: &TextureSpec) -> Answer<rt::TextureDescriptor> {
    let format = PixelFormat::try_from(spec.format).map_err(|problem| {
        Failure::named("NoSuchPixelFormat", format!("{}: {problem:?}", spec.format))
    })?;
    let mut usage = rt::TextureUsage::SHADER_READ;
    if spec.render_target {
        usage |= rt::TextureUsage::RENDER_TARGET | rt::TextureUsage::PIXEL_FORMAT_VIEW;
    }
    if let Some(stated) = spec.usage {
        usage = rt::TextureUsage::try_from(stated).map_err(|problem| {
            Failure::named("NoSuchTextureUsage", format!("{stated}: {problem:?}"))
        })?;
    }
    let texture_type = match spec.kind {
        TextureKind::Type2D => rt::TextureType::Type2D,
        TextureKind::Type2DArray => rt::TextureType::Type2DArray,
        TextureKind::Cube => rt::TextureType::Cube,
        TextureKind::Type3D => rt::TextureType::Type3D,
        TextureKind::Type2DMultisample => rt::TextureType::Type2DMultisample,
        TextureKind::Type2DMultisampleArray => rt::TextureType::Type2DMultisampleArray,
        TextureKind::TextureBuffer => rt::TextureType::TextureBuffer,
    };
    let (width, height) = (
        narrow(spec.width, "a width")?,
        narrow(spec.height, "a height")?,
    );
    let mut d = rt::TextureDescriptor::new(texture_type, format, width, height);
    d.depth = narrow(spec.depth.max(1), "a depth")?;
    d.array_length = narrow(spec.slices.max(1), "an array length")?;
    d.mipmap_level_count = narrow(spec.levels.max(1), "a level count")?;
    d.sample_count = narrow(spec.samples.max(1), "a sample count")?;
    d.usage = usage;
    d.storage_mode = storage_mode(spec.storage);
    Ok(d)
}

/// A resource the program made, or a slot a [`Step::Release`] emptied.
enum Made {
    /// A buffer, with the blocks this side took for it alone: none for a heap's buffer, whose
    /// heap holds its block.
    Buffer {
        buffer: rt::Buffer,
        _blocks: Vec<rt::Block>,
    },
    /// A placement heap: the block of its storage and the range it is.
    Heap {
        _block: rt::Block,
        range: rt::HeapRange,
    },
    Texture(rt::Texture),
    Fence(rt::Fence),
    /// The resource a [`Step::Release`] dropped, an explicit state so that a step naming it is
    /// refused as one naming a resource never made is, and a [`Step::Make`] fills the slot again.
    Released,
}

impl Made {
    /// A texture's static note (`Texture::precision_note`); no other resource carries one.
    fn precision_note(&self) -> Option<rt::PrecisionNote> {
        match self {
            Made::Texture(texture) => texture.precision_note(),
            _ => None,
        }
    }
}

/// A read-back the device runs before its bytes are answered.
struct Pending {
    /// The read's place among the program's reads.
    slot: usize,
    buffer: rt::Buffer,
    /// The buffer's block, which this side holds while the buffer lives.
    _block: rt::Block,
    length: usize,
    /// The bytes of one sample the read keeps and the bytes the kernel wrote for it, for a
    /// per-sample read whose words are narrowed.
    samples: Option<(usize, usize)>,
}

/// A Shared buffer of `length` bytes in a block of its own, zeroed as `newBufferWithLength:` is.
fn shared_buffer(on: &RtDevice, length: u64) -> Answer<(rt::Buffer, rt::Block)> {
    let length = length.max(1);
    let block = (on.device.take_block(rt::MemoryClass::Shared, length)).map_err(of_rt)?;
    let buffer = (on.device.new_buffer(block.id, 0, length, Some(0))).map_err(of_rt)?;
    Ok((buffer, block))
}

/// `bytes` written into a buffer the host reaches, at its start.
pub(super) fn write(buffer: &rt::Buffer, bytes: &[u8]) -> Answer<()> {
    let contents = (buffer.contents())
        .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
    // SAFETY: the buffer holds at least `bytes.len()` bytes, the host reaches them, and no command
    // uses it yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    Ok(())
}

/// `length` bytes of `buffer` from `offset`, as the host reads a buffer it reaches.
pub(super) fn read_buffer(buffer: &rt::Buffer, offset: u64, length: usize) -> Answer<Vec<u8>> {
    let past = (offset.checked_add(length as u64)).is_none_or(|end| end > buffer.length());
    if past {
        return Err(Failure::named(
            "PastTheBuffer",
            format!(
                "{length} bytes at {offset} of a buffer of {}",
                buffer.length()
            ),
        ));
    }
    let contents = (buffer.contents())
        .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
    // SAFETY: `[offset, offset + length)` is inside the buffer's bytes, the host reaches them, and
    // the caller waited for the command buffer that wrote them.
    let bytes =
        unsafe { std::slice::from_raw_parts(contents.as_ptr().add(offset as usize), length) };
    Ok(bytes.to_vec())
}

/// A buffer of `length` bytes in `storage`, zeroed as `newBufferWithLength:options:` is, with
/// the blocks this side took for it.
pub(super) fn buffer_of(
    on: &RtDevice,
    length: u64,
    storage: Storage,
) -> Answer<(rt::Buffer, Vec<rt::Block>)> {
    let length = length.max(1);
    let device = &on.device;
    match storage {
        Storage::Shared => {
            let (buffer, block) = shared_buffer(on, length)?;
            Ok((buffer, vec![block]))
        }
        Storage::Private => {
            let block = (device.take_block(rt::MemoryClass::Private, length)).map_err(of_rt)?;
            let buffer = (device.new_buffer(block.id, 0, length, Some(0))).map_err(of_rt)?;
            Ok((buffer, vec![block]))
        }
        Storage::Managed => {
            let bytes = (device.take_block(rt::MemoryClass::Shared, length)).map_err(of_rt)?;
            let copy = match device.has_unified_memory() {
                true => None,
                false => {
                    Some((device.take_block(rt::MemoryClass::Private, length)).map_err(of_rt)?)
                }
            };
            let at = |block: &rt::Block| (block.id, 0);
            let buffer = (device.new_managed_buffer(at(&bytes), copy.as_ref().map(at), length, 0))
                .map_err(of_rt)?;
            // The device copy is zeroed; the consumer's bytes are this side's to zero.
            write(&buffer, &vec![0; length as usize])?;
            Ok((buffer, [Some(bytes), copy].into_iter().flatten().collect()))
        }
        Storage::Memoryless => Err(Failure::unavailable(
            "newBufferWithLength:options: of MTLStorageModeMemoryless",
        )),
    }
}

/// A placement heap of `size` bytes in `storage`.
fn heap(on: &RtDevice, size: u64, storage: Storage) -> Answer<Made> {
    let (block, range) = heap_of(on, size, storage)?;
    Ok(Made::Heap {
        _block: block,
        range,
    })
}

/// The block a heap of `size` bytes in `storage` is, and the range of it the heap is.
pub(super) fn heap_of(
    on: &RtDevice,
    size: u64,
    storage: Storage,
) -> Answer<(rt::Block, rt::HeapRange)> {
    let class = match storage {
        Storage::Shared => rt::MemoryClass::Shared,
        Storage::Private => rt::MemoryClass::Private,
        Storage::Managed | Storage::Memoryless => {
            return Err(Failure::unavailable(
                "newHeapWithDescriptor: of other than Private or Shared storage",
            ))
        }
    };
    let block = (on.device.take_block(class, size.max(1))).map_err(of_rt)?;
    let range = rt::HeapRange {
        block: block.id,
        offset: 0,
        len: size.max(1),
    };
    Ok((block, range))
}

/// Make one resource, of the ones `made` already holds.
fn make(on: &RtDevice, made: &[Made], resource: &Resource) -> Answer<Made> {
    let buffer_at = |at: usize| match made.get(at) {
        Some(Made::Buffer { buffer, .. }) => Ok(buffer),
        _ => Err(Failure::named("NoSuchResource", format!("buffer {at}"))),
    };
    let heap_at = |at: usize| match made.get(at) {
        Some(Made::Heap { range, .. }) => Ok(*range),
        _ => Err(Failure::named("NoSuchResource", format!("heap {at}"))),
    };
    let texture_at = |at: usize| match made.get(at) {
        Some(Made::Texture(texture)) => Ok(texture),
        _ => Err(Failure::named("NoSuchResource", format!("texture {at}"))),
    };
    let device = &on.device;
    match resource {
        Resource::Buffer { bytes } => {
            let (buffer, block) = shared_buffer(on, bytes.len() as u64)?;
            write(&buffer, bytes)?;
            Ok(Made::Buffer {
                buffer,
                _blocks: vec![block],
            })
        }
        Resource::UnwrittenBuffer { length, storage } => {
            let (buffer, blocks) = buffer_of(on, *length, *storage)?;
            Ok(Made::Buffer {
                buffer,
                _blocks: blocks,
            })
        }
        Resource::Fence => Ok(Made::Fence(device.new_fence())),
        Resource::Heap { size, storage } => heap(on, *size, *storage),
        Resource::HeapForTexture { spec, storage } => {
            let sized = (device.heap_texture_size_and_align(&descriptor(spec)?)).map_err(of_rt)?;
            heap(on, sized.size, *storage)
        }
        Resource::HeapBuffer {
            heap,
            length,
            offset,
        } => {
            let range = heap_at(*heap)?;
            // A heap's memory is not zeroed for a buffer placed in it, as Metal's is not.
            let buffer = (device.new_heap_buffer(range, *offset, *length)).map_err(of_rt)?;
            Ok(Made::Buffer {
                buffer,
                _blocks: Vec::new(),
            })
        }
        Resource::Texture { spec } => {
            let d = descriptor(spec)?;
            let texture = (device.new_texture(&d, on.texture_index(d.pixel_format, d.usage)))
                .map_err(of_rt)?;
            Ok(Made::Texture(texture))
        }
        Resource::HeapTexture { heap, spec, offset } => {
            let d = descriptor(spec)?;
            let range = heap_at(*heap)?;
            let index = on.texture_index(d.pixel_format, d.usage);
            let texture = (device.new_heap_texture(&d, index, range, *offset)).map_err(of_rt)?;
            Ok(Made::Texture(texture))
        }
        Resource::TextureOverBuffer {
            buffer,
            spec,
            offset,
            bytes_per_row,
        } => {
            let d = descriptor(spec)?;
            let index = on.texture_index(d.pixel_format, d.usage);
            let texture = (buffer_at(*buffer)?.new_texture(&d, index, *offset, *bytes_per_row))
                .map_err(of_rt)?;
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
            let parent = texture_at(*texture)?;
            let format = PixelFormat::try_from(*format).map_err(|problem| {
                Failure::named("NoSuchPixelFormat", format!("{format}: {problem:?}"))
            })?;
            let kind = match flatten {
                true => rt::TextureType::Type2DMultisample,
                false => parent.descriptor().texture_type,
            };
            let (level, slice) = (narrow(*level, "a level")?, narrow(*slice, "a slice")?);
            let levels = level..level + narrow(*level_count, "a level count")?;
            let slices = slice..slice + narrow(*slice_count, "a slice count")?;
            let index = on.texture_index(format, parent.descriptor().usage);
            let view =
                (parent.new_texture_view(format, kind, levels, slices, index)).map_err(of_rt)?;
            Ok(Made::Texture(view))
        }
    }
}

/// The interpreter's state as it runs a program.
struct Run<'a> {
    on: &'a RtDevice,
    queue: &'a rt::CommandQueue,
    /// The static notes of the textures the program made.
    notes: &'a mut BTreeSet<rt::PrecisionNote>,
    made: Vec<Made>,
    /// The program's declarations, which a [`Step::Make`] reads to make a released resource again.
    resources: &'a [Resource],
    kernels: Option<&'static str>,
    command_buffer: Option<rt::CommandBuffer>,
    encoder: Option<rt::BlitCommandEncoder>,
    pending: Vec<Pending>,
    reads: Vec<Option<Vec<u8>>>,
    next_read: usize,
    statuses: Vec<String>,
}

/// Run `work` on `on`'s device through `queue`, answering what it read, with the static note of
/// each texture it makes (`Texture::precision_note`) in `notes`.
pub(super) fn run(
    on: &RtDevice,
    queue: &rt::CommandQueue,
    work: &Work,
    notes: &mut BTreeSet<rt::PrecisionNote>,
) -> Answer<Done> {
    let mut made: Vec<Made> = Vec::with_capacity(work.resources.len());
    for (at, resource) in work.resources.iter().enumerate() {
        let one = make(on, &made, resource).map_err(|failure| failure.at_resource(at))?;
        notes.extend(one.precision_note());
        made.push(one);
    }
    let mut state = Run {
        on,
        queue,
        notes,
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
    let reads = (state.reads.into_iter().enumerate())
        .map(|(slot, bytes)| {
            bytes.ok_or_else(|| Failure::named("UnansweredRead", format!("read {slot}")))
        })
        .collect::<Answer<Vec<Vec<u8>>>>()?;
    Ok(Done {
        reads,
        statuses: state.statuses,
    })
}

impl Run<'_> {
    fn buffer(&self, at: usize) -> Answer<rt::Buffer> {
        match self.made.get(at) {
            Some(Made::Buffer { buffer, .. }) => Ok(buffer.clone()),
            _ => Err(Failure::named("NoSuchResource", format!("buffer {at}"))),
        }
    }

    fn texture(&self, at: usize) -> Answer<rt::Texture> {
        match self.made.get(at) {
            Some(Made::Texture(texture)) => Ok(texture.clone()),
            _ => Err(Failure::named("NoSuchResource", format!("texture {at}"))),
        }
    }

    fn fence(&self, at: usize) -> Answer<rt::Fence> {
        match self.made.get(at) {
            Some(Made::Fence(fence)) => Ok(fence.clone()),
            _ => Err(Failure::named("NoSuchResource", format!("fence {at}"))),
        }
    }

    /// The command buffer the steps encode into, made on first use.
    fn command_buffer(&mut self) -> Answer<rt::CommandBuffer> {
        if let Some(command_buffer) = &self.command_buffer {
            return Ok(command_buffer.clone());
        }
        let command_buffer = self.queue.command_buffer().map_err(of_rt)?;
        self.command_buffer = Some(command_buffer.clone());
        Ok(command_buffer)
    }

    /// The blit encoder the copies go into, opened on first use.
    fn encoder(&mut self) -> Answer<&mut rt::BlitCommandEncoder> {
        if self.encoder.is_none() {
            let encoder = self
                .command_buffer()?
                .blit_command_encoder()
                .map_err(of_rt)?;
            self.encoder = Some(encoder);
        }
        Ok(self.encoder.as_mut().expect("the encoder was just opened"))
    }

    /// The slot the next read answers.
    fn slot(&mut self) -> usize {
        self.next_read += 1;
        self.next_read - 1
    }

    /// A place in a texture, narrowed.
    fn place(slice: u64, level: u64, origin: &[u64; 3]) -> Answer<rt::TexturePlace> {
        Ok(rt::TexturePlace {
            slice: narrow(slice, "a slice")?,
            level: narrow(level, "a level")?,
            origin: narrow3(origin, "an origin")?,
        })
    }

    /// Run one step. A copy between a buffer and a texture with a blit option other than
    /// `MTLBlitOptionNone` is the runtime's to build; the selector without options is the one
    /// with `MTLBlitOptionNone`, as Apple's header says of it.
    fn step(&mut self, step: &Step) -> Answer<()> {
        match step {
            Step::FillTexture { .. } => Err(Failure::unavailable(
                "fillTexture:level:slice:region:bytes:length:",
            )),
            Step::FillTextureColor { .. } => Err(Failure::unavailable(
                "fillTexture:level:slice:region:color:",
            )),
            Step::CopyBufferToTexture { plane, .. } | Step::CopyTextureToBuffer { plane, .. }
                if *plane != Plane::Whole =>
            {
                Err(Failure::unavailable(plane.option_name()))
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
                plane: _,
            }
            | Step::CopyBufferToTextureWithoutOptions {
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
                let (source, destination) = (self.buffer(*buffer)?, self.texture(*texture)?);
                let layout = rt::BufferLayout {
                    offset: *offset,
                    bytes_per_row: *bytes_per_row,
                    bytes_per_image: *bytes_per_image,
                };
                let (size, to) = (
                    narrow3(size, "a size")?,
                    Self::place(*slice, *level, origin)?,
                );
                (self
                    .encoder()?
                    .copy_from_buffer(&source, layout, size, &destination, to))
                .map_err(of_rt)
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
                plane: _,
            }
            | Step::CopyTextureToBufferWithoutOptions {
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
                let (source, destination) = (self.texture(*texture)?, self.buffer(*buffer)?);
                let layout = rt::BufferLayout {
                    offset: *offset,
                    bytes_per_row: *bytes_per_row,
                    bytes_per_image: *bytes_per_image,
                };
                let (from, size) = (
                    Self::place(*slice, *level, origin)?,
                    narrow3(size, "a size")?,
                );
                (self.encoder()?)
                    .copy_from_texture_to_buffer(&source, from, size, &destination, layout)
                    .map_err(of_rt)
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
                let (source, destination) = (self.texture(*from)?, self.texture(*to)?);
                let at = Self::place(*from_slice, *from_level, from_origin)?;
                let into = Self::place(*to_slice, *to_level, to_origin)?;
                let size = narrow3(size, "a size")?;
                (self
                    .encoder()?
                    .copy_from_texture(&source, at, size, &destination, into))
                .map_err(of_rt)
            }
            Step::CopyWholeTexture { .. } => {
                Err(Failure::unavailable("copyFromTexture:toTexture:"))
            }
            Step::CopyTextureSlices { .. } => Err(Failure::unavailable(
                "copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:\
                 destinationLevel:sliceCount:levelCount:",
            )),
            Step::GenerateMipmaps { texture } => {
                let texture = self.texture(*texture)?;
                (self.encoder()?.generate_mipmaps(&texture)).map_err(of_rt)
            }
            Step::OptimizeForGpuAccess { .. } => {
                Err(Failure::unavailable("optimizeContentsForGPUAccess:"))
            }
            Step::OptimizeForCpuAccess { .. } => {
                Err(Failure::unavailable("optimizeContentsForCPUAccess:"))
            }
            Step::SynchronizeResource { resource } => {
                let held = match self.made.get(*resource) {
                    Some(Made::Buffer { buffer, .. }) => Held::Buffer(buffer.clone()),
                    Some(Made::Texture(texture)) => Held::Texture(texture.clone()),
                    _ => {
                        return Err(Failure::named(
                            "NoSuchResource",
                            format!("resource {resource}"),
                        ))
                    }
                };
                let encoder = self.encoder()?;
                match &held {
                    Held::Buffer(buffer) => {
                        encoder.synchronize_resource(rt::Resource::Buffer(buffer))
                    }
                    Held::Texture(texture) => {
                        encoder.synchronize_resource(rt::Resource::Texture(texture))
                    }
                }
                .map_err(of_rt)
            }
            Step::SynchronizeTexture { .. } => {
                Err(Failure::unavailable("synchronizeTexture:slice:level:"))
            }
            Step::CopyBufferToBuffer {
                from,
                from_offset,
                to,
                to_offset,
                length,
            } => {
                let (source, destination) = (self.buffer(*from)?, self.buffer(*to)?);
                (self.encoder()?.copy_from_buffer_to_buffer(
                    &source,
                    *from_offset,
                    &destination,
                    *to_offset,
                    *length,
                ))
                .map_err(of_rt)
            }
            Step::FillBuffer {
                buffer,
                offset,
                length,
                value,
            } => {
                let buffer = self.buffer(*buffer)?;
                (self
                    .encoder()?
                    .fill_buffer(&buffer, *offset..*offset + *length, *value))
                .map_err(of_rt)
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
                // The runtime checks the bytes it is handed against the region and the pitches
                // and answers a typed error, so the stated rows are the Metal side's guard alone.
                rows: _,
                bytes_per_row,
                bytes_per_image,
            } => {
                let texture = self.texture(*texture)?;
                let (level, slice) = (narrow(*level, "a level")?, narrow(*slice, "a slice")?);
                (texture.replace_region(
                    region_of(region)?,
                    level,
                    slice,
                    bytes.as_slice().into(),
                    *bytes_per_row,
                    *bytes_per_image,
                ))
                .map_err(of_rt)
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
                let (level, slice) = (narrow(*level, "a level")?, narrow(*slice, "a slice")?);
                let mut bytes = vec![*before; *length];
                (texture.get_bytes(
                    &mut bytes,
                    *bytes_per_row,
                    *bytes_per_image,
                    region_of(region)?,
                    level,
                    slice,
                ))
                .map_err(of_rt)?;
                let slot = self.slot();
                self.reads[slot] = Some(bytes);
                Ok(())
            }
            Step::ReadBuffer {
                buffer,
                offset,
                length,
            } => {
                let bytes = read_buffer(&self.buffer(*buffer)?, *offset, *length)?;
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
                let read = SampleRead {
                    texture: *texture,
                    kernel,
                    extent: [*width, *height],
                    samples: *samples,
                    texel_bytes: *texel_bytes,
                    words: *words,
                };
                self.read_samples(slot, &read)
            }
            Step::EndEncoder => {
                self.end_encoding();
                Ok(())
            }
            Step::UpdateFence { fence } => {
                let fence = self.fence(*fence)?;
                self.encoder()?.update_fence(&fence).map_err(of_rt)
            }
            Step::WaitForFence { fence } => {
                let fence = self.fence(*fence)?;
                self.encoder()?.wait_for_fence(&fence).map_err(of_rt)
            }
            Step::Commit => self.close(),
        }
    }

    /// Drop the side's handles of the resource at `at`, refused while a command buffer is open, as
    /// [`crate::runtime_work`] refuses it: a resource an uncommitted command buffer names is not
    /// freed by the release.
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
        let declared = (self.resources.get(at))
            .ok_or_else(|| Failure::named("NoSuchResource", format!("resource {at}")))?;
        let one = make(self.on, &self.made, declared)?;
        self.notes.extend(one.precision_note());
        self.made[at] = one;
        Ok(())
    }

    /// A render pass over one slice that clears it and draws nothing.
    fn clear_pass(&mut self, texture: usize, slice: u64, clear: Clear) -> Answer<()> {
        // A pass does not open while the blit encoder is: Metal takes one encoder at a time.
        self.end_encoding();
        let texture = self.texture(texture)?;
        let slice = narrow(slice, "a slice")?;
        let mut pass = rt::RenderPassDescriptor::default();
        let (depth, stencil) = match clear {
            Clear::Color([r, g, b, a]) => {
                let attachment = &mut pass.color_attachments[0];
                attachment.texture = Some(texture.clone());
                attachment.slice = slice;
                attachment.load_action = rt::LoadAction::Clear;
                attachment.clear_color = rt::ClearColor::new(r, g, b, a);
                attachment.store_action = rt::StoreAction::Store;
                (None, None)
            }
            Clear::Depth(depth) => (Some(depth), None),
            Clear::Stencil(stencil) => (None, Some(stencil)),
            Clear::DepthStencil(depth, stencil) => (Some(depth), Some(stencil)),
        };
        if let Some(depth) = depth {
            let attachment = &mut pass.depth_attachment;
            attachment.texture = Some(texture.clone());
            attachment.slice = slice;
            attachment.load_action = rt::LoadAction::Clear;
            attachment.clear_depth = depth;
            attachment.store_action = rt::StoreAction::Store;
        }
        if let Some(stencil) = stencil {
            let attachment = &mut pass.stencil_attachment;
            attachment.texture = Some(texture);
            attachment.slice = slice;
            attachment.load_action = rt::LoadAction::Clear;
            attachment.clear_stencil = stencil;
            attachment.store_action = rt::StoreAction::Store;
        }
        let encoder = (self.command_buffer()?.render_command_encoder(&pass)).map_err(of_rt)?;
        encoder.end_encoding();
        Ok(())
    }

    /// Encode a per-sample read-back of a texture by the committed fixture's kernel.
    fn read_samples(&mut self, slot: usize, read: &SampleRead<'_>) -> Answer<()> {
        self.end_encoding();
        let source = self.texture(read.texture)?;
        let fixture = self.kernels.ok_or_else(|| {
            Failure::named("NoKernels", "the program states no fixture for its kernels")
        })?;
        let path = engine_dir().join(format!("{fixture}{}", crate::fixture::RUNTIME_TEXT));
        let bitcode = llvm_assemble(&path).map_err(|error| {
            Failure::named("NoFixture", format!("{}: {error:?}", path.display()))
        })?;
        let on = self.on;
        let library =
            rt::Library::from_air(&bitcode).map_err(|error| on.translation_failure(&error))?;
        let function =
            (library.function(read.kernel)).map_err(|error| on.translation_failure(&error))?;
        let descriptor = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(
            rt::FunctionDescriptor::new(function),
        ));
        let pipeline = (on.device)
            .new_compute_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| on.failure(&error))?;
        let [width, height] = read.extent;
        let length = width as usize * height as usize * read.samples as usize * read.words * 4;
        let (out, out_block) = shared_buffer(on, length as u64)?;
        // The shape the fixture's kernels read: the extent's width and each pixel's samples.
        let shape_bytes: Vec<u8> = [width, read.samples]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (shape, _shape_block) = shared_buffer(on, shape_bytes.len() as u64)?;
        write(&shape, &shape_bytes)?;
        let command_buffer = self.command_buffer()?;
        let mut encoder =
            (command_buffer.compute_command_encoder(rt::DispatchType::Serial)).map_err(of_rt)?;
        encoder
            .set_compute_pipeline_state(&pipeline)
            .map_err(of_rt)?;
        encoder.set_texture(Some(&source), 0).map_err(of_rt)?;
        encoder.set_buffer(Some(&out), 0, 0).map_err(of_rt)?;
        encoder.set_buffer(Some(&shape), 0, 1).map_err(of_rt)?;
        // One thread a sample, one thread a threadgroup: no grid edge to guard.
        (encoder.dispatch_threadgroups([width, height, read.samples], [1, 1, 1])).map_err(of_rt)?;
        encoder.end_encoding();
        self.pending.push(Pending {
            slot,
            buffer: out,
            _block: out_block,
            length,
            samples: Some((read.texel_bytes, read.words * 4)),
        });
        Ok(())
    }

    /// Close the blit encoder, if one is open.
    fn end_encoding(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.end_encoding();
        }
    }

    /// Close the encoder, commit the command buffer, wait for it within [`WAIT`], record its
    /// status and answer every read-back asked for so far.
    fn close(&mut self) -> Answer<()> {
        self.end_encoding();
        let Some(command_buffer) = self.command_buffer.take() else {
            return Ok(());
        };
        command_buffer.commit().map_err(of_rt)?;
        let Some(word) = waited_for(&command_buffer, WAIT) else {
            self.statuses.push("NeverReturned".to_owned());
            return Err(Failure::named(
                "WaitRanOut",
                "the command buffer did not complete inside the bound",
            ));
        };
        self.statuses.push(word);
        for read in self.pending.drain(..) {
            let bytes = read_buffer(&read.buffer, 0, read.length)?;
            let bytes = match read.samples {
                Some((texel_bytes, stride)) => (bytes.chunks(stride))
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

/// A buffer or a texture a [`Step::SynchronizeResource`] names, held so the encoder can be
/// borrowed after the resources were read.
enum Held {
    Buffer(rt::Buffer),
    Texture(rt::Texture),
}

/// What a [`Step::ReadSamples`] reads.
struct SampleRead<'a> {
    texture: usize,
    kernel: &'a str,
    extent: [u32; 2],
    samples: u32,
    texel_bytes: usize,
    words: usize,
}
