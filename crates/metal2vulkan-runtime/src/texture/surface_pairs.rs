//! A 4:2:2 texture over a buffer's bytes (`GBGR422` or `BGRG422` over an IOSurface plane,
//! `Buffer::new_texture_with_iosurface_plane`; `chroma_pairs.rs`): an image of its texels as Metal
//! reads them, which its copy in writes from the pairs where they lie
//! (`ShadowSource::Converted`, [`PairsConversion`]).
//!
//! **Why.** Vulkan views a 4:2:2 format only through a sampler Y′CbCr conversion, which no
//! pipeline's layout bakes (`chroma_pairs.rs`), so a texture of its own keeps its bytes in a
//! carrier and a decoded twin beside it (`texture/decoded_twin.rs`). A texture over a buffer's
//! bytes needs no carrier: the bytes are its truth, which its copies, blits and CPU calls move
//! (`blit_texture.rs`'s **Textures over buffers**), so it is an `R8G8B8A8_UNORM` image of its
//! texels alone, which every sampled view presents. Its reads, samples and gathers read Metal's
//! texels, which the device filters as it filters that format.
//!
//! **Kept in step** as a converted two-plane YCbCr texture's image is (`shadow.rs`'s **Converted
//! texels**): its copy in, before the first command of a command buffer that reads it and again
//! after a command wrote the bytes, is one dispatch of `bind::chroma_convert_module` that reads
//! the pairs where they lie, at any byte of their `VkBuffer`, through one storage-buffer binding,
//! and writes every texel's codes through the image's `R32_UINT` view: Metal's texels, exact
//! (`bind::ChromaDecode`). No byte is copied. Nothing else writes the image: Metal drops a
//! function's writes to the texture, and no render pass takes it (`chroma_pairs.rs`). The
//! binding names the buffer's `VkBuffer`, which a rebind of a no-copy buffer changes
//! (`rebind.rs`): the copy in's planning makes the set again over the new one
//! ([`PairsConversion::make`]), and the command buffers that took the old one hold it.
//!
//! **Through the device's own conversion.** Where the device converts the pairs' format with
//! Metal's reconstruction (`ycbcr_device.rs`, `YCbCrSampler::pairs`), the copy in is that
//! conversion instead: the plane's rows copied into the texture's pairs' image
//! (`PairsSampling`), sampled once per texel through the conversion into a scratch of the
//! texels (`bind::chroma_sample_module`), which is copied into the image. Its texels are the
//! device's, announced `decoded-texel-precision` where a command first reads them (the planning
//! of its copy in).
//!
//! **Cost**, announced once per device (`carried-decoded-twin`, performance): the image's memory,
//! 4 bytes a texel (with the device's conversion also the pairs' image, 2 bytes a texel, and the
//! scratch, 4), and one dispatch at each copy in.
//!
//! **Where it is not made** the texture keeps the carrier of a 4:2:2 texture of its own, its
//! copies moving its bytes and a command that reads it refused (`NotImplemented(Format)`, naming
//! the format; `docs/FAST-PATHS.md`'s known gaps): rows that share bytes, texels past the
//! buffer's length, pairs whose last word passes the buffer's bytes or that lie past what one
//! storage binding reaches (`maxStorageBufferRange`), and, with the device's conversion, a first
//! byte or rows off 4 bytes in their `VkBuffer`.

use super::{IdUse, Storage, Texture};
use crate::binding_view::VkView;
use crate::buffer_texture::Texels;
use crate::degradation::Degradation;
use crate::device::functions::Vk;
use crate::device::lock;
use crate::error::{internal, Error};
use crate::image_plan::{sample_count, ImagePlan};
use crate::owned_kernel::{OwnedKernel, OwnedModule, OwnedSet, OwnedSets};
use crate::resource::ResourceKey;
use crate::shadow::{Conversion, ShadowBuffer, ShadowCopies, ShadowSource};
use crate::shadow_merge::ScratchSpares;
use crate::texture_descriptor::{effective_usage, TextureDescriptor, TextureLayout};
use crate::tracker::{Access, Barrier, ImagePlace, Use, UseRange};
use crate::values::ResourceOptions;
use crate::view_plan::ViewRecord;
use crate::ycbcr_device::{PairsSampling, YCbCrSampler};
use crate::{Buffer, Device};
use ash::vk;
use metal2vulkan::bind::{ChromaConvertModule, ChromaDecode};
use metal2vulkan::metal::format::{image_type, FormatMapping};
use metal2vulkan::metal::{TextureType, TextureUsage};
use metal2vulkan::vk::{
    ComponentMapping, Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags,
    ImageUsageFlags, ImageViewType,
};
use std::any::Any;
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

/// The format of the image of the texels.
const TEXELS: Format = Format::R8G8B8A8_UNORM;

/// The format of the view the in-place kernel writes the texels through, a word each.
const WORDS: Format = Format::R32_UINT;

/// Why the image's texels are not Metal's where the device's conversion writes them
/// (`decoded-texel-precision`), as a decoded twin of that conversion says.
const DEVICE_TEXELS: &str = "its image holds the device's own Y′CbCr conversion of its pairs \
                             (Vulkan's midpoint reconstruction through a linear chroma filter, at \
                             the device's precision), each channel as the nearest 8-bit code with \
                             ties to even, where Metal's 3:1 sum of the codes rounds its ties up";

/// Where a 4:2:2 texture's pairs lie in their buffer's `VkBuffer`, which a rebind keeps
/// (`rebind.rs`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PairsWindow {
    /// The first row's first byte.
    at: u64,
    /// The bytes from one row to the next.
    row: u64,
    /// The bytes from the first row's first to the last row's last.
    span: u64,
}

/// How a 4:2:2 texture over a buffer's bytes is converted ([`Device::pairs_in_place`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PairsPlace {
    decode: ChromaDecode,
    window: PairsWindow,
    /// The device's own conversion of the pairs, where it converts their format with Metal's
    /// reconstruction; `None` where the in-place kernel writes Metal's texels.
    sampler: Option<YCbCrSampler>,
}

/// What a 4:2:2 texture's copy in writes: the image's `R32_UINT` storage view, which the in-place
/// kernel writes, or its `VkImage`, which the device's conversion's texels are copied into
/// through the device's conversion `sampler` (each the image's own for its life,
/// `Image::own_view`).
enum Target {
    Kernel(vk::ImageView),
    Device(YCbCrSampler, vk::Image),
}

/// The conversion of a 4:2:2 texture's pairs where they lie into its image: its copy in (the
/// module docs). What it records ([`PairsMade`]) is made at the planning of its first copy in
/// ([`make`](Self::make)), and again there after a rebind gave the buffer another `VkBuffer`.
pub(crate) struct PairsConversion {
    device: Device,
    decode: ChromaDecode,
    /// The texture, as its notices name it.
    object: String,
    /// The texture's width and height.
    size: [u32; 2],
    window: PairsWindow,
    target: Target,
    made: Mutex<Option<Arc<PairsMade>>>,
}

/// What a 4:2:2 texture's copy in records, the command buffers that record it holding it until
/// they complete.
enum PairsMade {
    /// The in-place kernel's set over `bound`, the buffer's `VkBuffer` it was made at, and its
    /// shape's buffer.
    Kernel {
        bound: vk::Buffer,
        kernel: OwnedKernel,
        sets: OwnedSets,
        groups: [u32; 2],
        _shape: Buffer,
    },
    /// The device's conversion's objects.
    Device(Box<DeviceMade>),
}

/// The objects of a 4:2:2 texture's copy in through the device's own conversion: the pairs'
/// image, the scratch the sample kernel writes the texels into (`scratch_raw`, from `at`, its
/// tracked use `scratch_use`), the kernel's set and shape, the copies of the pairs in and of the
/// texels out, and the workgroups.
struct DeviceMade {
    pairs: PairsSampling,
    /// The pairs' sample kernel ([`PairsSampling::module`]).
    kernel: OwnedKernel,
    _scratch: Buffer,
    scratch_raw: vk::Buffer,
    scratch_use: (ResourceKey, Range<u64>),
    sets: OwnedSets,
    _shape: Buffer,
    pairs_in: vk::BufferImageCopy,
    texels_out: vk::BufferImageCopy,
    groups: [u32; 3],
}

impl PairsMade {
    /// Whether it records the copy in of pairs in `raw`, the buffer's `VkBuffer` now: the kernel's
    /// set binds the one it was made at, the device's conversion copies from `raw` as it records.
    fn serves(&self, raw: vk::Buffer) -> bool {
        match self {
            PairsMade::Kernel { bound, .. } => *bound == raw,
            PairsMade::Device(_) => true,
        }
    }
}

/// The plan of the image of a 4:2:2 texture of `layout` with Metal `usage` over a buffer's bytes
/// (the module docs): `R8G8B8A8_UNORM`, sampled and copied, and stored through its `R32_UINT`
/// view where the in-place kernel writes it (`device`: the device's conversion writes it by
/// copies). `None` for a layout no image takes.
fn pairs_plan(layout: &TextureLayout, usage: TextureUsage, device: bool) -> Option<ImagePlan> {
    let (image_type, flags) = image_type(layout.texture_type)?;
    let copies = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    let (usage_flags, flags, view_formats) = match device {
        true => (ImageUsageFlags::SAMPLED | copies, flags, Vec::new()),
        false => (
            ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE | copies,
            flags | ImageCreateFlags::MUTABLE_FORMAT,
            vec![TEXELS, WORDS],
        ),
    };
    Some(ImagePlan {
        pixel_format: layout.pixel_format,
        flags,
        image_type,
        format: TEXELS,
        extent: layout.extent,
        levels: layout.levels,
        layers: layout.layers,
        samples: sample_count(layout.samples)?,
        usage: usage_flags,
        view_formats,
        mapping: FormatMapping::direct(TEXELS, ImageAspectFlags::COLOR),
        // As a carrier's (`chroma_pairs::plan_pairs`): Metal drops a function's writes to it, and
        // no pass renders into it.
        unavailable: effective_usage(usage).intersection(
            TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC | TextureUsage::RENDER_TARGET,
        ),
    })
}

/// How a 4:2:2 texture of `layout` is named in its notices.
fn object(layout: &TextureLayout) -> String {
    let [width, height, _] = layout.extent;
    format!(
        "a {:?} {:?} texture of {width}x{height} over a buffer's bytes",
        layout.pixel_format, layout.texture_type
    )
}

impl Device {
    /// How a 4:2:2 texture of `layout` over `texels` is converted where its pairs lie so (the
    /// module docs): the in-place kernel's window, or the device's own conversion where it
    /// converts the format with Metal's reconstruction. `None` for a texture of another format,
    /// and for one whose pairs lie as the module docs' **Where it is not made** says.
    pub(crate) fn pairs_in_place(
        &self,
        texels: &Texels,
        layout: &TextureLayout,
    ) -> Option<PairsPlace> {
        let decode = ChromaDecode::of(layout.pixel_format)?;
        let one = layout.texture_type == TextureType::Type2D
            && layout.levels == 1
            && layout.layers == 1
            && layout.samples == 1;
        if !one || texels.runs_past() || texels.plane_rows_overlap() {
            return None;
        }
        let gpu = &texels.layout;
        let [width, height, _] = layout.extent;
        let start = texels.buffer.start();
        let at = start.checked_add(gpu.offset)?;
        let span = gpu.end().checked_sub(gpu.offset)?;
        let end = start.checked_add(texels.buffer.length())?;
        let window = PairsWindow {
            at,
            row: gpu.row_pitch,
            span,
        };
        let limits = &self.properties().description.limits;
        let local = ChromaConvertModule::LOCAL_SIZE;
        let groups = [width.div_ceil(local), height.div_ceil(local)];
        if groups[0] > limits.max_compute_work_group_count[0]
            || groups[1] > limits.max_compute_work_group_count[1]
        {
            return None;
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let has = |format, features| facts.optimal(format).contains(features);
        match YCbCrSampler::pairs(self, decode, [width, height]) {
            // The pairs' copy into an image of their format: its first byte and rows on whole
            // pairs (VUID-vkCmdCopyBufferToImage-dstImage-07975, -bufferRowLength-09106).
            Some(sampler) => {
                let fits = at.is_multiple_of(4)
                    && window.row.is_multiple_of(4)
                    && u32::try_from(window.row / 2).is_ok()
                    && at.checked_add(span)? <= end;
                let formats = has(
                    TEXELS,
                    FormatFeatureFlags2::SAMPLED_IMAGE | FormatFeatureFlags2::TRANSFER_DST,
                );
                (fits && formats).then_some(PairsPlace {
                    decode,
                    window,
                    sampler: Some(sampler),
                })
            }
            // One storage binding from the first byte rounded down to the device's offset
            // alignment through the last byte's word, which lies in the buffer's bytes.
            None => {
                let align = limits.min_storage_buffer_offset_alignment.max(4);
                let read_end = at.checked_add(span)?.next_multiple_of(4);
                let range = read_end - (at - at % align);
                let fits = read_end <= end
                    && range <= u64::from(limits.max_storage_buffer_range)
                    && u32::try_from(window.row).is_ok();
                let formats = has(
                    TEXELS,
                    FormatFeatureFlags2::SAMPLED_IMAGE | FormatFeatureFlags2::STORAGE_IMAGE,
                ) && has(WORDS, FormatFeatureFlags2::STORAGE_IMAGE);
                (fits && formats).then_some(PairsPlace {
                    decode,
                    window,
                    sampler: None,
                })
            }
        }
    }

    /// The 4:2:2 texture of `descriptor` (checked as `layout`) over `texels`, for Metal's `call`,
    /// converted as `place` says ([`pairs_in_place`](Self::pairs_in_place)): its image and the
    /// conversion's shadow (the module docs), announcing its cost (`carried-decoded-twin`).
    /// Refused as `new_texture` refuses the image and the kernel's view.
    pub(crate) fn new_pairs_texture(
        &self,
        call: &'static str,
        texels: Texels,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        id_use: IdUse,
        place: PairsPlace,
    ) -> Result<Texture, Error> {
        let texels = Arc::new(texels);
        let record = ViewRecord::of_descriptor(layout, descriptor);
        let plan = pairs_plan(layout, descriptor.usage, place.sampler.is_some())
            .ok_or_else(|| internal("a 4:2:2 texture over a buffer of no image"))?;
        let hazard = texels.options().hazard_tracking_mode()?;
        let image = self.new_image(plan, layout, descriptor, hazard)?;
        let target = match place.sampler {
            Some(sampler) => Target::Device(sampler, image.raw.get()),
            // The conversion's set holds it as long as the image (`Image::own_view`).
            None => Target::Kernel(image.own_view(&VkView {
                view_type: ImageViewType::TYPE_2D,
                format: WORDS,
                components: ComponentMapping::IDENTITY,
                aspect: ImageAspectFlags::COLOR,
                base_level: 0,
                level_count: 1,
                base_layer: 0,
                layer_count: 1,
                usage: ImageUsageFlags::STORAGE,
            })?),
        };
        let [width, height, _] = layout.extent;
        let span = texels.layout.offset..texels.layout.end();
        let (raw, _) = texels.buffer.at(span.start);
        let conversion = PairsConversion {
            device: self.clone(),
            decode: place.decode,
            object: object(layout),
            size: [width, height],
            window: place.window,
            target,
            made: Mutex::new(None),
        };
        let (bytes, tracked) = texels.buffer.tracked(span);
        let copies = ShadowCopies {
            source: ShadowSource::Converted {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                alias: texels.buffer.alias().map(|alias| alias.key),
                conversion: Conversion::Pairs(Box::new(conversion)),
                pixel_format: layout.pixel_format,
            },
            shadow_key: image.identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
                place: ImagePlace::WHOLE,
            },
            tracked: texels.tracked(),
            // Metal drops a function's writes to the texture.
            drops_writes: true,
            levels: None,
            decode: Default::default(),
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
            stood: AtomicU64::new(0),
            standing: Default::default(),
        };
        self.degrade(Degradation::CarriedDecodedTwin, || {
            let texels = u64::from(width) * u64::from(height) * 4;
            let how = match place.sampler {
                Some(_) => format!(
                    "the device's own Y′CbCr conversion of the pairs writes at each copy in, \
                     through an image of the pairs ({} bytes) and a scratch of the texels ({texels} \
                     bytes)",
                    texels / 2
                ),
                None => "one dispatch decodes from the pairs where they lie at each copy in, \
                         copying no byte"
                    .to_owned(),
            };
            let why = format!(
                "Vulkan views a 4:2:2 format only through a sampler Y′CbCr conversion, which no \
                 pipeline's layout bakes: its samples read an image of its {texels} bytes of \
                 R8G8B8A8_UNORM texels beside the buffer's pairs, which {how}"
            );
            (object(layout), why)
        });
        let why =
            "a 4:2:2 texture's image: Metal samples the pairs decoded, which an image of the \
                   texels holds, written from the pairs where they lie by its copy in";
        crate::shadow_log::made(self, call, (descriptor, layout), &texels, why);
        let storage = Storage::Shadow {
            image: image.shared(),
            texels,
            copies: Arc::new(copies),
        };
        Ok(Texture::made(self, storage, record, None, id_use))
    }
}

impl PairsConversion {
    /// Make what its copies in record over `raw`, the buffer's `VkBuffer` now, where nothing made
    /// serves it ([`PairsMade::serves`]): the in-place kernel's set over it and its shape, or the
    /// device's conversion's pairs' image, scratch, set and shape. What it made is returned, for
    /// the command buffer of the copy's command to hold until it completes, and announces
    /// `decoded-texel-precision` where the device's conversion writes the texels (a command reads
    /// them next). `None` while the bytes are detached (`raw` the null handle): no copy records
    /// then (`ShadowBuffer::raw`). Refused as the kernels, the image, the buffers and the set are.
    pub(crate) fn make(
        &self,
        raw: vk::Buffer,
    ) -> Result<Option<Arc<dyn Any + Send + Sync>>, Error> {
        if raw == vk::Buffer::null() {
            return Ok(None);
        }
        let made = {
            let mut made = lock(&self.made);
            match made.as_ref().filter(|made| made.serves(raw)) {
                Some(made) => Arc::clone(made),
                None => {
                    let fresh = Arc::new(match &self.target {
                        Target::Kernel(view) => self.kernel_made(raw, *view)?,
                        Target::Device(sampler, _) => self.device_made(*sampler)?,
                    });
                    *made = Some(Arc::clone(&fresh));
                    fresh
                }
            }
        };
        if let Target::Device(..) = self.target {
            self.device.degrade(Degradation::DecodedTexelPrecision, || {
                (self.object.clone(), DEVICE_TEXELS.to_owned())
            });
        }
        let held: Arc<dyn Any + Send + Sync> = made;
        Ok(Some(held))
    }

    /// The in-place kernel's objects over `raw`, writing through `view`.
    fn kernel_made(&self, raw: vk::Buffer, view: vk::ImageView) -> Result<PairsMade, Error> {
        let device = &self.device;
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let PairsWindow { at, row, span } = self.window;
        // As `Device::pairs_in_place` checked it.
        let offset = at - at % align;
        let range = (at + span).next_multiple_of(4) - offset;
        let first = u32::try_from(at - offset)
            .map_err(|_| internal("a 4:2:2 texture's pairs off their window"))?;
        let row = u32::try_from(row).map_err(|_| internal("a 4:2:2 texture's rows past a word"))?;
        let kernel = device.owned_kernel(OwnedModule::ChromaConvert(self.decode))?;
        let shape = ChromaConvertModule::shape(self.size, first, row);
        let (shape_buffer, shape_at) = self.shape_buffer(&shape)?;
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: vec![view],
            buffers: vec![
                (shape_at.0, shape_at.1, shape.len() as u64),
                (raw, offset, range),
            ],
        };
        let sets = OwnedSets::write_each(device, &[kernel], &[set])?;
        let local = ChromaConvertModule::LOCAL_SIZE;
        Ok(PairsMade::Kernel {
            bound: raw,
            kernel,
            sets,
            groups: self.size.map(|side| side.div_ceil(local)),
            _shape: shape_buffer,
        })
    }

    /// The device's conversion's objects through `sampler`.
    fn device_made(&self, sampler: YCbCrSampler) -> Result<PairsMade, Error> {
        let device = &self.device;
        let [width, height] = self.size;
        let pairs = PairsSampling::new(device, sampler, self.size)?;
        let kernel = device.owned_kernel(pairs.module())?;
        let texels = u64::from(width) * u64::from(height) * 4;
        let scratch = device.own_buffer(texels, ResourceOptions::private())?;
        let (scratch_raw, scratch_at) = scratch.at(0);
        let (identity, range) = scratch.tracked(0..texels);
        let (shape, groups) = pairs.step(0, height);
        let (shape_buffer, shape_at) = self.shape_buffer(&shape)?;
        let set = OwnedSet {
            images: vec![pairs.view],
            storage_images: Vec::new(),
            buffers: vec![
                (shape_at.0, shape_at.1, shape.len() as u64),
                (scratch_raw, scratch_at, texels),
            ],
        };
        let sets = OwnedSets::write_each(device, &[kernel], &[set])?;
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        // As `Device::pairs_in_place` checked it: its rows on whole pairs, 2 bytes a texel.
        let row_texels = u32::try_from(self.window.row / 2)
            .map_err(|_| internal("a 4:2:2 texture's rows past a word"))?;
        Ok(PairsMade::Device(Box::new(DeviceMade {
            pairs,
            kernel,
            _scratch: scratch,
            scratch_raw,
            scratch_use: (identity.key, range),
            sets,
            _shape: shape_buffer,
            pairs_in: vk::BufferImageCopy {
                buffer_offset: self.window.at,
                buffer_row_length: row_texels,
                buffer_image_height: 0,
                image_subresource: layers,
                image_offset: vk::Offset3D::default(),
                image_extent: extent,
            },
            texels_out: vk::BufferImageCopy {
                buffer_offset: scratch_at,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: layers,
                image_offset: vk::Offset3D::default(),
                image_extent: extent,
            },
            groups,
        })))
    }

    /// A buffer the host wrote `shape` into, and its `VkBuffer` and the shape's offset there, at
    /// the device's storage-buffer offset alignment.
    fn shape_buffer(&self, shape: &[u8]) -> Result<(Buffer, (vk::Buffer, u64)), Error> {
        let device = &self.device;
        let buffer = device.write_combined_buffer(shape.len() as u64)?;
        let (raw, at) = buffer.at(0);
        let align = device
            .properties()
            .description
            .limits
            .min_storage_buffer_offset_alignment
            .max(4);
        if !at.is_multiple_of(align) {
            return Err(internal("a 4:2:2 conversion's shape off its alignment"));
        }
        let contents = (buffer.mapped_contents())
            .ok_or_else(|| internal("a Shared 4:2:2 conversion's shape without contents"))?;
        // SAFETY: the buffer holds the shape's bytes, in host-coherent memory nothing else uses
        // yet.
        unsafe { std::ptr::copy_nonoverlapping(shape.as_ptr(), contents.as_ptr(), shape.len()) };
        Ok((buffer, (raw, at)))
    }

    /// The copy in's uses: `truth`, the read of the pairs' bytes, then, through the device's
    /// conversion, the scratch's write and read, and the write of every texel of the image of
    /// `copies`: by the kernel's dispatch, or the device's conversion's copy.
    pub(crate) fn uses(
        &self,
        truth: Use<ResourceKey>,
        copies: &ShadowCopies,
    ) -> Vec<Use<ResourceKey>> {
        let written = match self.target {
            Target::Kernel(_) => Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
            Target::Device(..) => Access::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
        };
        let image = Use::writes(
            copies.shadow_key,
            copies.tracked,
            copies.shadow_range.clone(),
            written,
        );
        let scratch = match lock(&self.made).as_deref() {
            Some(PairsMade::Device(made)) => {
                let (key, range) = made.scratch_use.clone();
                Some(Use {
                    key,
                    tracked: copies.tracked,
                    range: UseRange::Bytes(range),
                    read: Access::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_READ,
                    ),
                    write: Access::new(
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    ),
                })
            }
            _ => None,
        };
        [truth].into_iter().chain(scratch).chain([image]).collect()
    }

    /// Record the copy in from the pairs in `source` (the buffer's `VkBuffer` now) into
    /// `command_buffer`: the kernel's dispatch, or the device's conversion (the pairs copied into
    /// its image, sampled into the scratch, the texels copied into the image, each step past a
    /// barrier). Nothing where what was made serves no `source` (a copy whose planning made none:
    /// none records, `ShadowCopies::prepare`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; `source` and the image are live until
    /// it completes, and the image is in the `GENERAL` layout; the copy's planning made what it
    /// records ([`make`](Self::make)), which the command buffer holds.
    pub(crate) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        source: vk::Buffer,
    ) {
        let Some(made) = lock(&self.made).clone().filter(|made| made.serves(source)) else {
            return;
        };
        let point = vk::PipelineBindPoint::COMPUTE;
        let general = vk::ImageLayout::GENERAL;
        // SAFETY: the caller's contract; the set binds the pairs' window in `source` and the
        // image's view (`kernel_made`), or the pairs' image, the shape and the scratch
        // (`device_made`), and each copy lies in its buffer and image.
        unsafe {
            match (&*made, &self.target) {
                (
                    PairsMade::Kernel {
                        kernel,
                        sets,
                        groups,
                        ..
                    },
                    _,
                ) => {
                    device.cmd_bind_pipeline(command_buffer, point, kernel.pipeline);
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        point,
                        kernel.layout,
                        0,
                        &sets.sets,
                        &[],
                    );
                    device.cmd_dispatch(command_buffer, groups[0], groups[1], 1);
                }
                (PairsMade::Device(made), Target::Device(_, image)) => {
                    // The pairs' image's earlier pairs dropped, the pairs copied in, sampled
                    // into the scratch, and the texels copied into the image.
                    let whole = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let into = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_READ)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(general)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(made.pairs.image)
                        .subresource_range(whole);
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[into],
                    );
                    let pairs_in = std::slice::from_ref(&made.pairs_in);
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        source,
                        made.pairs.image,
                        general,
                        pairs_in,
                    );
                    Barrier::everything().record(device, command_buffer);
                    let kernel = &made.kernel;
                    device.cmd_bind_pipeline(command_buffer, point, kernel.pipeline);
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        point,
                        kernel.layout,
                        0,
                        &made.sets.sets,
                        &[],
                    );
                    let [x, y, z] = made.groups;
                    device.cmd_dispatch(command_buffer, x, y, z);
                    Barrier::everything().record(device, command_buffer);
                    let texels_out = std::slice::from_ref(&made.texels_out);
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        made.scratch_raw,
                        *image,
                        general,
                        texels_out,
                    );
                }
                // Made for its target alone (`make`).
                (PairsMade::Device(_), Target::Kernel(_)) => {}
            }
        }
    }
}
