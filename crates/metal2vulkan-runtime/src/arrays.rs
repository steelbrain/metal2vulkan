//! Set 1: the descriptor arrays every pipeline takes its textures and samplers from, each element
//! at the index a `gpuResourceID` holds (`docs/RESOURCE-ID.md`), and the index grants the consumer
//! chooses those indices from.
//!
//! One descriptor set of the translator's identity domains' seven arrays, made at device creation
//! and never rewritten for layouts: binding 0 the sampled textures and binding 1 the storage
//! textures (65,536 each, element `i` of both the texture at index `i`), bindings 2, 3 and 4 the
//! samplers of each use, plain, depth comparison and integer texture (8,000 each: the 4,000 indices
//! a resource ID reaches, then 4,000 variant elements), and bindings 5 and 6 the texture buffers'
//! uniform and storage texel buffers (65,536 each, element `i` of both the texture buffer at index
//! `i`). Every binding is partially bound and updated after bind, also while work that uses the
//! set is pending, so writing one element needs no lock: each element has one owner, which writes
//! it alone.
//!
//! A variant element holds a sampler state bound with LOD clamps of its own (`crate::sampler`): the
//! runtime places it at an index from 4,000 to 7,999, which no grant reaches, and pushes that
//! index to the slot that binds it. Shaders declare the arrays unsized, so their length is this
//! layout's, and an argument buffer's sampler index of 4,000 or more reads element 0, so no
//! program-written ID reaches a variant.
//!
//! Every element no live object owns holds a defined value, so a nil, stale or forged index reads
//! zeros or samples plainly on every device, where an element never written would be undefined
//! behaviour. A texture or texel-buffer element is a null descriptor (`nullDescriptor`): it reads
//! zeros, answers 0 to its size and level queries and drops its writes. The Metal Shading Language
//! leaves every member function of a null texture undefined (section 6.13.18), so these are the
//! runtime's answers, not Metal's. A sampler
//! element is its array's default sampler, nearest and clamped to the edge; the comparison
//! array's compares with `NEVER`, so a comparison through it answers 0, as Metal's default
//! comparison (`MTLCompareFunctionNever`) does. Vulkan has no null sampler. The device writes
//! every element once at creation ([`Arrays::new`]).
//!
//! Where the device reads attachments through textures (`floor::Capabilities::attachment_reads`),
//! the sampled-texture array is twice as long: element `i + 65,536` is texture `i`'s second
//! element, which a draw pushes in place of the texture's own while it reads an attachment of its
//! pass through the texture (`crate::render_encoder`). Through the attachment feedback loop it is
//! the texture's sampled view in the feedback layout, written with element `i` for a texture whose
//! image a pass may move there; on a device without the loop it is the texture's view of its
//! image's shadow, written at the first draw that reads through it where the texture still owns
//! the index (`crate::shadow`). It is a null descriptor otherwise. No index of a grant reaches it.
//!
//! The consumer names each range of indices it takes ([`Device::grant_indices`]), so resource IDs
//! are chosen on the guest's side and reproduce in replay. Two classes: textures of every usage
//! take 1 to 65,535, an index valid in both texture arrays; samplers 1 to 3,999. Index 0 is nil
//! and never granted.

use crate::counters::{Counters, Gauge, Metric, Part};
use crate::device::lock;
use crate::error::{Error, Limit, Refusal};
use crate::floor::{
    BelowFloor, Missing, SAMPLERS, SAMPLER_ARRAYS, SAMPLER_ELEMENTS, SAMPLER_VARIANTS, TEXTURES,
};
use crate::Device;
use ash::vk;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;

/// The binding of the sampled-texture array.
pub(crate) const SAMPLED_BINDING: u32 = 0;
/// The binding of the storage-texture array.
pub(crate) const STORAGE_BINDING: u32 = 1;
/// The binding of the first sampler array; the plain, comparison and integer-texture arrays
/// follow in that order.
pub(crate) const FIRST_SAMPLER_BINDING: u32 = 2;
/// The binding of the uniform texel-buffer array: the texture buffers shaders read.
pub(crate) const UNIFORM_TEXELS_BINDING: u32 = 5;
/// The binding of the storage texel-buffer array: the texture buffers shaders write.
pub(crate) const STORAGE_TEXELS_BINDING: u32 = 6;

/// Set 1's bindings, each its binding, descriptor type and count: the arrays a kernel's layout
/// names under the identity domains, which every pipeline's layout takes from this set. Shaders
/// declare the arrays unsized, and the set's layout holds [`layout_counts`] elements: more in the
/// sampled-texture array with the second elements, and in each sampler array its variants'.
pub(crate) const BINDINGS: [(u32, vk::DescriptorType, u32); 7] = [
    (SAMPLED_BINDING, vk::DescriptorType::SAMPLED_IMAGE, TEXTURES),
    (STORAGE_BINDING, vk::DescriptorType::STORAGE_IMAGE, TEXTURES),
    (FIRST_SAMPLER_BINDING, vk::DescriptorType::SAMPLER, SAMPLERS),
    (
        FIRST_SAMPLER_BINDING + 1,
        vk::DescriptorType::SAMPLER,
        SAMPLERS,
    ),
    (
        FIRST_SAMPLER_BINDING + 2,
        vk::DescriptorType::SAMPLER,
        SAMPLERS,
    ),
    (
        UNIFORM_TEXELS_BINDING,
        vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        TEXTURES,
    ),
    (
        STORAGE_TEXELS_BINDING,
        vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        TEXTURES,
    ),
];

/// No Metal counterpart: the class of a texture or sampler index, which sets the range its grants
/// are taken from (`docs/RESOURCE-ID.md`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexClass {
    /// A texture of any usage shaders reach: 1 to 65,535, one index valid in both the
    /// sampled-texture and the storage-texture array, and in both texel-buffer arrays. A texture
    /// shaders write through a storage view takes the element of both texture arrays at its
    /// index; any other takes the sampled-texture array's alone, and its storage element stays
    /// null. A texture buffer takes the texel-buffer arrays' elements instead: the uniform one
    /// where shaders read it, the storage one where they write it.
    Texture,
    /// A sampler: 1 to 3,999 of each sampler array. A Metal sampler state takes one, and so does
    /// each constexpr sampler a pipeline state numbers.
    Sampler,
}

impl IndexClass {
    /// The indices the class's grants are taken from.
    pub fn range(self) -> Range<u32> {
        match self {
            IndexClass::Texture => 1..TEXTURES,
            IndexClass::Sampler => 1..SAMPLERS,
        }
    }

    fn slot(self) -> usize {
        match self {
            IndexClass::Texture => 0,
            IndexClass::Sampler => 1,
        }
    }
}

/// No Metal counterpart: indices granted to the consumer, `first..first + count` of `class`
/// ([`Device::grant_indices`]). Plain data: it holds nothing of the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IndexGrant {
    /// The class the indices are of.
    pub class: IndexClass,
    /// The first index.
    pub first: u32,
    /// How many indices.
    pub count: u32,
}

impl IndexGrant {
    fn indices(self) -> Range<u32> {
        self.first..self.first + self.count
    }
}

/// No Metal counterpart: `MTLResourceID`, a texture's or a sampler state's `gpuResourceID`: its
/// index in the low word and the translator's word in the high (`docs/RESOURCE-ID.md`). The guest
/// computes the word from the Metal descriptor alone, so it needs no round trip for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ResourceId {
    /// The index in its class's arrays; 0 for nil.
    pub index: u32,
    /// The word: a texture's `TextureWord`, a sampler's `SamplerWord`.
    pub word: u32,
}

impl ResourceId {
    /// `MTLResourceID`'s 64-bit value: `index | word << 32`.
    pub fn to_bits(self) -> u64 {
        u64::from(self.index) | u64::from(self.word) << 32
    }
}

/// One bit per index of a class's array.
struct Bits(Vec<u64>);

impl Bits {
    fn new(len: u32) -> Bits {
        Bits(vec![0; len.div_ceil(64) as usize])
    }

    fn get(&self, index: u32) -> bool {
        self.0[(index / 64) as usize] & (1 << (index % 64)) != 0
    }

    fn put(&mut self, index: u32) {
        self.0[(index / 64) as usize] |= 1 << (index % 64);
    }
}

/// The word that names the object an index holds: the one [`Grants::place`] gave it, given to
/// no other object of the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Owner(u64);

/// An object placed at an index ([`Grants::place`]).
pub(crate) struct Placed {
    /// The object's owner word.
    pub(crate) owner: Owner,
    /// Another object held the index, and no longer does.
    pub(crate) displaced: bool,
}

/// The indices granted, one bit per index of each class's array, and of those the indices an
/// object holds, each with its one owner: the object placed there last.
pub(crate) struct Grants {
    granted: [Bits; 2],
    owners: [HashMap<u32, Owner>; 2],
    /// The last owner word given.
    last: u64,
    /// The variant elements no variant holds, the lowest last.
    variants: Vec<u32>,
}

impl Grants {
    pub(crate) fn new() -> Grants {
        Grants {
            granted: [Bits::new(TEXTURES), Bits::new(SAMPLERS)],
            owners: [HashMap::new(), HashMap::new()],
            last: 0,
            variants: (SAMPLERS..SAMPLER_ELEMENTS).rev().collect(),
        }
    }

    fn bit(&self, class: IndexClass, index: u32) -> bool {
        self.granted[class.slot()].get(index)
    }

    fn set(&mut self, class: IndexClass, index: u32) {
        self.granted[class.slot()].put(index);
    }

    /// Places an object at `index` of `class`, where it is granted: the object owns the index
    /// from now, and an object there before no longer does, as Metal's later texture at a
    /// resource ID takes the ID.
    pub(crate) fn place(&mut self, class: IndexClass, index: u32) -> Result<Placed, Error> {
        if !class.range().contains(&index) || !self.bit(class, index) {
            return Err(Error::Refused(Refusal::IndicesNotGranted { class, index }));
        }
        self.last += 1;
        let owner = Owner(self.last);
        let displaced = self.owners[class.slot()].insert(index, owner).is_some();
        Ok(Placed { owner, displaced })
    }

    /// Whether `owner` owns `index` of `class`.
    pub(crate) fn owns(&self, class: IndexClass, index: u32, owner: Owner) -> bool {
        self.owners[class.slot()].get(&index) == Some(&owner)
    }

    /// Gives back `index` of `class` where `owner` still owns it: whether it did. An object
    /// displaced from the index gives back nothing.
    pub(crate) fn vacate(&mut self, class: IndexClass, index: u32, owner: Owner) -> bool {
        let owns = self.owns(class, index, owner);
        if owns {
            self.owners[class.slot()].remove(&index);
        }
        owns
    }

    /// Grants `first..first + count` of `class` where the range is inside the class's and none of
    /// it is granted.
    pub(crate) fn grant(
        &mut self,
        class: IndexClass,
        first: u32,
        count: u32,
    ) -> Result<IndexGrant, Error> {
        let range = class.range();
        let inside = count > 0
            && first >= range.start
            && first.checked_add(count).is_some_and(|end| end <= range.end);
        if !inside {
            return Err(Error::Limit(Limit::IndexRange {
                class,
                first,
                count,
            }));
        }
        let grant = IndexGrant {
            class,
            first,
            count,
        };
        if let Some(index) = grant.indices().find(|&index| self.bit(class, index)) {
            return Err(Error::Refused(Refusal::IndicesGranted { class, index }));
        }
        for index in grant.indices() {
            self.set(class, index);
        }
        Ok(grant)
    }
}

/// Set 1, its layouts, its default samplers, and the grants of its indices.
pub(crate) struct Arrays {
    /// The empty layout of set 0, which no pipeline reads.
    empty: vk::DescriptorSetLayout,
    /// Set 1's layout.
    layout: vk::DescriptorSetLayout,
    /// The pool of the one set, which destroying it frees.
    pool: vk::DescriptorPool,
    /// The one set.
    set: vk::DescriptorSet,
    /// The default sampler of the plain and integer-texture arrays.
    pub(crate) plain: vk::Sampler,
    /// The default sampler of the comparison array.
    pub(crate) compare: vk::Sampler,
    /// Whether the sampled-texture array holds the second elements.
    seconds: bool,
    pub(crate) grants: Mutex<Grants>,
    /// The sampler each sampler array's element at each index holds, plain, comparison and
    /// integer texture, as last written: a default, or a sampler the index's owner holds, so a
    /// handle here is never one destroyed. Written under the grants' lock.
    samplers: Mutex<Box<[[vk::Sampler; SAMPLER_ARRAYS as usize]]>>,
}

impl Arrays {
    /// Makes set 1 and writes every element its default; with `seconds`, the sampled-texture
    /// array holds the second elements too.
    ///
    /// # Safety
    ///
    /// `device` is live and was made with the floor's features, `nullDescriptor` and the
    /// update-after-bind features among them.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        counters: &Counters,
        seconds: bool,
    ) -> Result<Arrays, Error> {
        let mut made = Made::default();
        // SAFETY: the caller's contract; what was made is destroyed where a later step fails.
        let arrays = unsafe { made.make(device, counters, seconds) };
        if arrays.is_err() {
            // SAFETY: nothing else holds what was made.
            unsafe { made.destroy(device, counters) };
        }
        arrays
    }

    /// Set 1, which each command buffer binds once, before its first pipeline.
    pub(crate) fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// The set layouts of every pipeline: the empty set 0, then set 1.
    pub(crate) fn set_layouts(&self) -> [vk::DescriptorSetLayout; 2] {
        [self.empty, self.layout]
    }

    /// Places a texture at `index` ([`Grants::place`]) and writes its elements before another
    /// object can be placed there: the sampled-texture array's with `sampled` and its layout, a
    /// null descriptor where it is `None`, the storage-texture array's with `storage`, and with
    /// `looped` the second element (`index` plus [`TEXTURES`]) with that view, the texture's
    /// sampled view, in the attachment feedback layout: a colour render target shaders do not
    /// read has it there alone. A texture without a storage view, or without a second element
    /// written here, leaves that element null, as every element starts, unless it displaced a
    /// texture, which may have had one: then it writes that element null, so nothing stale
    /// answers there. A null second element is written shader-read-only on every device, as
    /// [`Arrays::vacate_texture`] writes it: it holds no image, so its layout need only be one
    /// every device takes for a sampled image, which the feedback layout is not: a device without
    /// `VK_EXT_attachment_feedback_loop_layout` refuses it
    /// (`VUID-VkWriteDescriptorSet-descriptorType-04149`). Where it displaced, it writes both
    /// texel-buffer elements null too, which a texture buffer there may have held. The
    /// descriptor-slot gauge rises where no object held the index. Its owner word, which
    /// [`Arrays::vacate_texture`] and [`Arrays::place_shadow`] take.
    ///
    /// Err: [`Refusal::IndicesNotGranted`].
    ///
    /// # Safety
    ///
    /// Each view is live and of the device, of an image in that layout whenever work uses it,
    /// `looped` only where the set was made with the second elements and the image has the
    /// feedback usage, and where it displaces, no work in flight uses the index's elements (the
    /// consumer's duty: Vulkan does not define that write).
    pub(crate) unsafe fn place_texture(
        &self,
        device: &ash::Device,
        counters: &Counters,
        index: u32,
        (sampled, storage): (
            Option<(vk::ImageView, vk::ImageLayout)>,
            Option<vk::ImageView>,
        ),
        looped: Option<vk::ImageView>,
    ) -> Result<Owner, Error> {
        let mut grants = counters.time(Part::TexturePlaceLock, 0, || lock(&self.grants));
        let Placed { owner, displaced } = grants.place(IndexClass::Texture, index)?;
        if !displaced {
            counters.rise(Gauge::DescriptorSlots);
        }
        let storage = storage.or(displaced.then(vk::ImageView::null));
        let second = match looped {
            Some(view) => Some((view, vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)),
            None => (displaced && self.seconds).then(|| {
                (
                    vk::ImageView::null(),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
            }),
        };
        // SAFETY: the caller's contract; the texture owns the index, and the lock is held.
        unsafe {
            self.write_texture(device, index, sampled, storage, second);
            if displaced {
                self.write_texels(device, index, [vk::BufferView::null(); 2]);
            }
        }
        Ok(owner)
    }

    /// Places a texture buffer at `index` ([`Grants::place`]) and writes its elements before
    /// another object can be placed there: the uniform texel-buffer array's with `views[0]` and
    /// the storage one's with `views[1]`, a null descriptor where a view is null. Where it
    /// displaced, it writes the index's texture elements null too (the sampled, the storage, and
    /// the second where the set holds them), which a texture there may have held. The
    /// descriptor-slot gauge rises where no object held the index. Its owner word, which
    /// [`Arrays::vacate_texels`] takes.
    ///
    /// Err: [`Refusal::IndicesNotGranted`].
    ///
    /// # Safety
    ///
    /// Each view is live and of the device whenever work uses it, and where it displaces, no work
    /// in flight uses the index's elements (the consumer's duty: Vulkan does not define that
    /// write).
    pub(crate) unsafe fn place_texels(
        &self,
        device: &ash::Device,
        counters: &Counters,
        index: u32,
        views: [vk::BufferView; 2],
    ) -> Result<Owner, Error> {
        let mut grants = lock(&self.grants);
        let Placed { owner, displaced } = grants.place(IndexClass::Texture, index)?;
        if !displaced {
            counters.rise(Gauge::DescriptorSlots);
        }
        // SAFETY: the caller's contract; the texture buffer owns the index, and the lock is held.
        unsafe {
            self.write_texels(device, index, views);
            if displaced {
                let null = vk::ImageView::null();
                let second =
                    (self.seconds).then_some((null, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
                self.write_texture(device, index, None, Some(null), second);
            }
        }
        Ok(owner)
    }

    /// The texture buffer of `owner` goes from `index`: where it still owns the index, the index
    /// is free and its texel-buffer elements are null again, and the descriptor-slot gauge falls.
    /// A displaced texture buffer's going touches nothing.
    ///
    /// # Safety
    ///
    /// No work in flight reads the texture buffer's elements.
    pub(crate) unsafe fn vacate_texels(
        &self,
        device: &ash::Device,
        counters: &Counters,
        (index, owner): (u32, Owner),
    ) {
        let mut grants = lock(&self.grants);
        if grants.vacate(IndexClass::Texture, index, owner) {
            // SAFETY: the caller's contract; the lock is held, so no object is placed there
            // before its elements are null.
            unsafe { self.write_texels(device, index, [vk::BufferView::null(); 2]) };
            counters.fall(Gauge::DescriptorSlots);
        }
    }

    /// Writes a texture buffer's elements at `index`: the uniform texel-buffer array's with
    /// `views[0]` and the storage one's with `views[1]`, a null descriptor where a view is null.
    ///
    /// # Safety
    ///
    /// The grants' lock is held by a placement or a vacating of the index, and each view is live
    /// and of the device whenever work uses it.
    unsafe fn write_texels(&self, device: &ash::Device, index: u32, views: [vk::BufferView; 2]) {
        let (uniform, storage) = ([views[0]], [views[1]]);
        let writes = [
            texel_write(
                self.set,
                UNIFORM_TEXELS_BINDING,
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                &uniform,
            )
            .dst_array_element(index),
            texel_write(
                self.set,
                STORAGE_TEXELS_BINDING,
                vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                &storage,
            )
            .dst_array_element(index),
        ];
        // SAFETY: the caller's contract; the bindings are update-after-bind and
        // update-unused-while-pending, so elements of other owners may be written meanwhile, and
        // a null buffer view is a null descriptor under `nullDescriptor`.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Writes the second element of the texture of `owner` at `index` with `view`, its view of its
    /// image's shadow, in the shadow's home layout, shader-read-only, where the texture still owns
    /// the index: a draw reads an attachment through it on a device without the attachment
    /// feedback loop (`crate::shadow`). A displaced texture writes nothing: the element is its
    /// displacer's, which wrote it null.
    ///
    /// # Safety
    ///
    /// The set was made with the second elements, and `view` is live and of the device, of an
    /// image in that layout whenever work uses it, until the texture's elements are vacated.
    pub(crate) unsafe fn place_shadow(
        &self,
        device: &ash::Device,
        (index, owner): (u32, Owner),
        view: vk::ImageView,
    ) {
        let grants = lock(&self.grants);
        if grants.owns(IndexClass::Texture, index, owner) {
            let second = (view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            // SAFETY: the caller's contract; the texture owns the index, and the lock is held.
            unsafe { self.write_second(device, index, second) };
        }
    }

    /// The texture of `owner` goes from `index`: where it still owns the index, the index is free
    /// and its elements are null again (the second element too, where `second` says it may hold
    /// one), and the descriptor-slot gauge falls. A displaced texture's going touches nothing.
    ///
    /// # Safety
    ///
    /// No work in flight reads the texture's elements, and `second` holds only where the set was
    /// made with the second elements.
    pub(crate) unsafe fn vacate_texture(
        &self,
        device: &ash::Device,
        counters: &Counters,
        (index, owner): (u32, Owner),
        second: bool,
    ) {
        let mut grants = lock(&self.grants);
        if grants.vacate(IndexClass::Texture, index, owner) {
            let null = vk::ImageView::null();
            let second = second.then_some((null, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
            // SAFETY: the caller's contract; the lock is held, so no object is placed there
            // before its elements are null.
            unsafe { self.write_texture(device, index, None, Some(null), second) };
            counters.fall(Gauge::DescriptorSlots);
        }
    }

    /// Writes a texture's elements at `index`: the sampled-texture array's with `sampled` and its
    /// layout, a null descriptor where it is `None`; the storage-texture array's with `storage`
    /// where it is `Some`, a null descriptor where that view is null; and the second element with
    /// `second` where it is `Some` ([`Arrays::write_second`]).
    ///
    /// # Safety
    ///
    /// The grants' lock is held by a placement or a vacating of the index, each view is live and
    /// of the device, of an image in that layout whenever work uses it, and `second` is `Some`
    /// only where the set was made with the second elements.
    unsafe fn write_texture(
        &self,
        device: &ash::Device,
        index: u32,
        sampled: Option<(vk::ImageView, vk::ImageLayout)>,
        storage: Option<vk::ImageView>,
        second: Option<(vk::ImageView, vk::ImageLayout)>,
    ) {
        let (view, layout) = sampled.unwrap_or((
            vk::ImageView::null(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ));
        let sampled = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(layout)];
        let count = 1 + usize::from(storage.is_some());
        let storage = [vk::DescriptorImageInfo::default()
            .image_view(storage.unwrap_or_default())
            .image_layout(vk::ImageLayout::GENERAL)];
        if let Some(second) = second {
            // SAFETY: the caller's contract.
            unsafe { self.write_second(device, index, second) };
        }
        let writes = [
            write(
                self.set,
                SAMPLED_BINDING,
                vk::DescriptorType::SAMPLED_IMAGE,
                &sampled,
            )
            .dst_array_element(index),
            write(
                self.set,
                STORAGE_BINDING,
                vk::DescriptorType::STORAGE_IMAGE,
                &storage,
            )
            .dst_array_element(index),
        ];
        // SAFETY: the caller's contract; the bindings are update-after-bind and
        // update-unused-while-pending, so elements of other owners may be written meanwhile, and
        // a null image view is a null descriptor under `nullDescriptor`.
        unsafe { device.update_descriptor_sets(&writes[..count], &[]) };
    }

    /// Writes texture `index`'s second element, the sampled-texture array's at `index` plus
    /// [`TEXTURES`], with `(view, layout)`: a null descriptor where the view is null.
    ///
    /// # Safety
    ///
    /// As [`Arrays::write_texture`]'s, the grants' lock held for the index's owner.
    unsafe fn write_second(
        &self,
        device: &ash::Device,
        index: u32,
        (view, layout): (vk::ImageView, vk::ImageLayout),
    ) {
        let second = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(layout)];
        let write = write(
            self.set,
            SAMPLED_BINDING,
            vk::DescriptorType::SAMPLED_IMAGE,
            &second,
        )
        .dst_array_element(index + TEXTURES);
        // SAFETY: the caller's contract; the binding is update-after-bind and
        // update-unused-while-pending, the element is the index's owner's alone, and a null image
        // view is a null descriptor under `nullDescriptor`.
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    /// Places a sampler state or a constexpr sampler at `index` ([`Grants::place`]) and writes its
    /// elements before another object can be placed there: each of `elements`, `(array,
    /// sampler)`, the array 0 plain, 1 comparison and 2 integer texture, the array's default where
    /// the sampler is `None`. An array not listed keeps its element, its default, unless the
    /// object displaced another: then that element is written its default too. The
    /// descriptor-slot gauge rises where no object held the index. Its owner word.
    ///
    /// Err: [`Refusal::IndicesNotGranted`].
    ///
    /// # Safety
    ///
    /// Each sampler is live and of the device whenever work uses it, and where it displaces, no
    /// work in flight uses an element it writes (the consumer's duty: Vulkan does not define that
    /// write).
    pub(crate) unsafe fn place_samplers(
        &self,
        device: &ash::Device,
        counters: &Counters,
        index: u32,
        elements: &[(usize, Option<vk::Sampler>)],
    ) -> Result<Owner, Error> {
        let mut grants = lock(&self.grants);
        // SAFETY: the caller's contract, and the lock is held.
        unsafe { self.place_samplers_locked(&mut grants, device, counters, index, elements) }
    }

    /// One more pipeline state numbers the constexpr sampler of `owner` at `index`. Where the
    /// constexpr sampler still owns the index, writes `added`, the elements of the arrays the
    /// state adds, as [`Arrays::place_samplers`] does, and answers `owner`. Where another object
    /// took the index since, the state's numbering is a placement like any other: it places the
    /// constexpr sampler there again with `all`, every element it holds, and answers the new
    /// owner word.
    ///
    /// Err: [`Refusal::IndicesNotGranted`].
    ///
    /// # Safety
    ///
    /// No pending work uses an element of `added`, each sampler is live and of the device
    /// whenever work uses it, and where it displaces, no work in flight uses an element it
    /// writes (the consumer's duty: Vulkan does not define that write).
    pub(crate) unsafe fn hold_samplers(
        &self,
        device: &ash::Device,
        counters: &Counters,
        (index, owner): (u32, Owner),
        added: &[(usize, Option<vk::Sampler>)],
        all: &[(usize, Option<vk::Sampler>)],
    ) -> Result<Owner, Error> {
        let mut grants = lock(&self.grants);
        if grants.owns(IndexClass::Sampler, index, owner) {
            // SAFETY: the caller's contract; the object owns the index, and the lock is held.
            unsafe { self.write_samplers(device, counters, index, added) };
            return Ok(owner);
        }
        // SAFETY: the caller's contract, and the lock is held.
        unsafe { self.place_samplers_locked(&mut grants, device, counters, index, all) }
    }

    /// [`Arrays::place_samplers`] under the grants' lock, `grants`.
    ///
    /// # Safety
    ///
    /// As [`Arrays::place_samplers`]; `grants` is the guard of `self.grants`.
    unsafe fn place_samplers_locked(
        &self,
        grants: &mut Grants,
        device: &ash::Device,
        counters: &Counters,
        index: u32,
        elements: &[(usize, Option<vk::Sampler>)],
    ) -> Result<Owner, Error> {
        let Placed { owner, displaced } = grants.place(IndexClass::Sampler, index)?;
        if !displaced {
            counters.rise(Gauge::DescriptorSlots);
        }
        let mut all = elements.to_vec();
        if displaced {
            let listed = |array: &usize| elements.iter().any(|&(at, _)| at == *array);
            all.extend(
                (0..SAMPLER_ARRAYS as usize)
                    .filter(|a| !listed(a))
                    .map(|a| (a, None)),
            );
        }
        // SAFETY: the caller's contract; the object owns the index, and the lock is held.
        unsafe { self.write_samplers(device, counters, index, &all) };
        Ok(owner)
    }

    /// The sampler state or constexpr sampler of `owner` goes from `index`: where it still owns
    /// the index, the index is free and each array's element its default again, and the
    /// descriptor-slot gauge falls. A displaced object's going touches nothing.
    ///
    /// # Safety
    ///
    /// No work in flight reads the object's elements.
    pub(crate) unsafe fn vacate_samplers(
        &self,
        device: &ash::Device,
        counters: &Counters,
        index: u32,
        owner: Owner,
    ) {
        let mut grants = lock(&self.grants);
        if grants.vacate(IndexClass::Sampler, index, owner) {
            // SAFETY: the caller's contract; the lock is held, so no object is placed there
            // before every element is its default.
            unsafe {
                self.write_samplers(device, counters, index, &[(0, None), (1, None), (2, None)])
            };
            counters.fall(Gauge::DescriptorSlots);
        }
    }

    /// Places a variant of a sampler state at a variant element no variant holds and writes its
    /// elements there, each of `elements` as [`Arrays::place_samplers`] writes them: its index,
    /// which no grant reaches.
    ///
    /// Err: [`Limit::LodClampVariants`] where every variant element is held.
    ///
    /// # Safety
    ///
    /// Each sampler is live and of the device whenever work uses it.
    pub(crate) unsafe fn place_variant(
        &self,
        device: &ash::Device,
        counters: &Counters,
        elements: &[(usize, Option<vk::Sampler>)],
    ) -> Result<u32, Error> {
        let mut grants = lock(&self.grants);
        let index = (grants.variants.pop()).ok_or(Error::Limit(Limit::LodClampVariants {
            max: SAMPLER_VARIANTS,
        }))?;
        // SAFETY: the caller's contract; no variant held the element, so no work uses it, and the
        // lock is held.
        unsafe { self.write_samplers(device, counters, index, elements) };
        Ok(index)
    }

    /// The variant at variant element `index` goes: each array's element is its default again and
    /// the element is free.
    ///
    /// # Safety
    ///
    /// `index` is a variant element a variant held, and no work in flight reads its elements.
    pub(crate) unsafe fn vacate_variant(
        &self,
        device: &ash::Device,
        counters: &Counters,
        index: u32,
    ) {
        let mut grants = lock(&self.grants);
        // SAFETY: the caller's contract; the lock is held, so no variant is placed there before
        // every element is its default.
        unsafe { self.write_samplers(device, counters, index, &[(0, None), (1, None), (2, None)]) };
        grants.variants.push(index);
    }

    /// Writes elements at `index` of the sampler arrays, each `(array, sampler)`, the array 0
    /// plain, 1 comparison and 2 integer texture: the array's default where the sampler is `None`.
    /// An element that holds that sampler already is not written: an equal sampler state, whose
    /// create infos share the device's `VkSampler`s, placed at the index of one that pending work
    /// still reads writes nothing. The elements not listed are left as they are.
    ///
    /// # Safety
    ///
    /// The grants' lock is held by a placement, an addition or a vacating of the index or of the
    /// variant element, no pending work uses an element it changes, and each sampler is live and
    /// of the device whenever work uses it.
    unsafe fn write_samplers(
        &self,
        device: &ash::Device,
        counters: &Counters,
        index: u32,
        elements: &[(usize, Option<vk::Sampler>)],
    ) {
        let defaults = [self.plain, self.compare, self.plain];
        let mut all = lock(&self.samplers);
        let held = &mut all[index as usize];
        let changed: Vec<(usize, vk::Sampler)> = (elements.iter())
            .map(|&(array, sampler)| (array, sampler.unwrap_or(defaults[array])))
            .filter(|&(array, sampler)| held[array] != sampler)
            .collect();
        if changed.is_empty() {
            return;
        }
        for &(array, sampler) in &changed {
            held[array] = sampler;
            counters.count(Metric::SamplerElementWrites);
        }
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = (changed.iter())
            .map(|&(_, sampler)| [vk::DescriptorImageInfo::default().sampler(sampler)])
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = changed
            .iter()
            .zip(&infos)
            .map(|(&(array, _), info)| {
                write(
                    self.set,
                    FIRST_SAMPLER_BINDING + array as u32,
                    vk::DescriptorType::SAMPLER,
                    info,
                )
                .dst_array_element(index)
            })
            .collect();
        // SAFETY: the caller's contract; the bindings are update-after-bind and
        // update-unused-while-pending, so elements of other owners may be written meanwhile.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Destroys set 1.
    ///
    /// # Safety
    ///
    /// The device is idle, and no pipeline layout made with [`Arrays::set_layouts`] is used again.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device, counters: &Counters) {
        let made = Made {
            empty: Some(self.empty),
            layout: Some(self.layout),
            pool: Some(self.pool),
            plain: Some(self.plain),
            compare: Some(self.compare),
        };
        // SAFETY: the caller's contract.
        unsafe { made.destroy(device, counters) };
    }
}

/// What [`Arrays::new`] has made so far.
#[derive(Default)]
struct Made {
    empty: Option<vk::DescriptorSetLayout>,
    layout: Option<vk::DescriptorSetLayout>,
    pool: Option<vk::DescriptorPool>,
    plain: Option<vk::Sampler>,
    compare: Option<vk::Sampler>,
}

impl Made {
    /// # Safety
    ///
    /// As [`Arrays::new`].
    unsafe fn make(
        &mut self,
        device: &ash::Device,
        counters: &Counters,
        seconds: bool,
    ) -> Result<Arrays, Error> {
        let sampled = sampled_images(seconds);
        let empty = vk::DescriptorSetLayoutCreateInfo::default();
        // SAFETY: the device is live.
        unsafe { supported(device, &empty, 0) }?;
        // SAFETY: the device is live.
        let empty = unsafe { device.create_descriptor_set_layout(&empty, None) }
            .map_err(Error::vulkan("vkCreateDescriptorSetLayout"))?;
        self.empty = Some(empty);
        // SAFETY: the device is live, with the update-after-bind features of each binding's type
        // (the floor's).
        let layout = unsafe {
            with_arrays_layout(layout_counts(seconds), |info| {
                supported(device, info, 1)?;
                device
                    .create_descriptor_set_layout(info, None)
                    .map_err(Error::vulkan("vkCreateDescriptorSetLayout"))
            })
        }?;
        self.layout = Some(layout);
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: sampled,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: TEXTURES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: SAMPLER_ARRAYS * SAMPLER_ELEMENTS,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                descriptor_count: TEXTURES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                descriptor_count: TEXTURES,
            },
        ];
        let info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: the device is live.
        let pool = unsafe { device.create_descriptor_pool(&info, None) }
            .map_err(Error::vulkan("vkCreateDescriptorPool"))?;
        self.pool = Some(pool);
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool is the device's, sized for one set of this layout.
        let set = unsafe { device.allocate_descriptor_sets(&info) }
            .map_err(Error::vulkan("vkAllocateDescriptorSets"))?[0];
        let default = |compare: bool| {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .compare_enable(compare)
                .compare_op(vk::CompareOp::NEVER);
            // SAFETY: the device is live.
            let sampler = unsafe { device.create_sampler(&info, None) }
                .map_err(Error::vulkan("vkCreateSampler"))?;
            counters.rise(Gauge::DeviceSamplers);
            Ok::<_, Error>(sampler)
        };
        let plain = default(false)?;
        self.plain = Some(plain);
        let compare = default(true)?;
        self.compare = Some(compare);
        // SAFETY: the set is the device's and no command buffer uses it yet.
        unsafe { fill(device, set, sampled, plain, compare) };
        Ok(Arrays {
            empty,
            layout,
            pool,
            set,
            plain,
            compare,
            seconds,
            grants: Mutex::new(Grants::new()),
            samplers: Mutex::new(
                vec![[plain, compare, plain]; SAMPLER_ELEMENTS as usize].into_boxed_slice(),
            ),
        })
    }

    /// # Safety
    ///
    /// The device is idle, and nothing made from these objects is used again.
    unsafe fn destroy(&self, device: &ash::Device, counters: &Counters) {
        // SAFETY: the caller's contract; destroying the pool frees its set.
        unsafe {
            for sampler in [self.plain, self.compare].into_iter().flatten() {
                device.destroy_sampler(sampler, None);
                counters.fall(Gauge::DeviceSamplers);
            }
            if let Some(pool) = self.pool {
                device.destroy_descriptor_pool(pool, None);
            }
            for layout in [self.layout, self.empty].into_iter().flatten() {
                device.destroy_descriptor_set_layout(layout, None);
            }
        }
    }
}

/// Runs `f` with set 1's layout, `counts` elements in each binding of [`BINDINGS`].
pub(crate) fn with_arrays_layout<R>(
    counts: [u32; 7],
    f: impl FnOnce(&vk::DescriptorSetLayoutCreateInfo) -> R,
) -> R {
    let bindings = std::array::from_fn::<_, 7, _>(|index| {
        let (binding, ty, _) = BINDINGS[index];
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(ty)
            .descriptor_count(counts[index])
            .stage_flags(vk::ShaderStageFlags::ALL)
    });
    let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
        | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING; 7];
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .bindings(&bindings)
        .push_next(&mut binding_flags);
    f(&info)
}

/// The floor's last item, asked of the created device: the device supports the layout `info` of
/// set `set` (`vkGetDescriptorSetLayoutSupport`), which the limits alone do not promise. Else the
/// device is below the floor, [`Missing::SetLayout`].
///
/// # Safety
///
/// `device` is live, with the features `info` needs.
pub(crate) unsafe fn supported(
    device: &ash::Device,
    info: &vk::DescriptorSetLayoutCreateInfo,
    set: u32,
) -> Result<(), Error> {
    let mut support = vk::DescriptorSetLayoutSupport::default();
    // SAFETY: the caller's contract.
    unsafe { device.get_descriptor_set_layout_support(info, &mut support) };
    match support.supported == vk::TRUE {
        true => Ok(()),
        false => Err(Error::BelowFloor(BelowFloor {
            missing: vec![Missing::SetLayout { set }],
        })),
    }
}

/// The elements of each binding of set 1's layout, by [`BINDINGS`]: theirs, the sampled-texture
/// array's [`sampled_images`], and each sampler array's [`SAMPLER_ELEMENTS`], its variant elements
/// past the [`SAMPLERS`] a pipeline's layout names.
pub(crate) fn layout_counts(seconds: bool) -> [u32; 7] {
    let mut counts = BINDINGS.map(|(_, _, count)| count);
    counts[SAMPLED_BINDING as usize] = sampled_images(seconds);
    for array in 0..SAMPLER_ARRAYS {
        counts[(FIRST_SAMPLER_BINDING + array) as usize] = SAMPLER_ELEMENTS;
    }
    counts
}

/// The elements of the sampled-texture array: [`TEXTURES`], twice that with the second elements.
pub(crate) fn sampled_images(seconds: bool) -> u32 {
    match seconds {
        true => 2 * TEXTURES,
        false => TEXTURES,
    }
}

/// Writes every element of `set` its default: a null descriptor in each texture and texel-buffer
/// array, the sampled-texture array's `sampled` elements, `plain` in the plain and
/// integer-texture sampler arrays and `compare` in the comparison array.
///
/// # Safety
///
/// `set` is set 1 of `device`, of `sampled` sampled-texture elements, `nullDescriptor` is enabled,
/// and no other thread writes an element of it meanwhile.
unsafe fn fill(
    device: &ash::Device,
    set: vk::DescriptorSet,
    sampled: u32,
    plain: vk::Sampler,
    compare: vk::Sampler,
) {
    let sampled = vec![
        vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        sampled as usize
    ];
    let storage = vec![
        vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL);
        TEXTURES as usize
    ];
    let samplers = |sampler| {
        vec![vk::DescriptorImageInfo::default().sampler(sampler); SAMPLER_ELEMENTS as usize]
    };
    let (plains, compares) = (samplers(plain), samplers(compare));
    let texels = vec![vk::BufferView::null(); TEXTURES as usize];
    let writes = [
        write(
            set,
            SAMPLED_BINDING,
            vk::DescriptorType::SAMPLED_IMAGE,
            &sampled,
        ),
        write(
            set,
            STORAGE_BINDING,
            vk::DescriptorType::STORAGE_IMAGE,
            &storage,
        ),
        write(
            set,
            FIRST_SAMPLER_BINDING,
            vk::DescriptorType::SAMPLER,
            &plains,
        ),
        write(
            set,
            FIRST_SAMPLER_BINDING + 1,
            vk::DescriptorType::SAMPLER,
            &compares,
        ),
        write(
            set,
            FIRST_SAMPLER_BINDING + 2,
            vk::DescriptorType::SAMPLER,
            &plains,
        ),
        texel_write(
            set,
            UNIFORM_TEXELS_BINDING,
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            &texels,
        ),
        texel_write(
            set,
            STORAGE_TEXELS_BINDING,
            vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            &texels,
        ),
    ];
    // SAFETY: the caller's contract; a null image or buffer view is a null descriptor under
    // `nullDescriptor`, and each sampler is the device's.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// The write of `views` into texel-buffer `binding` of `set`, from its first element.
fn texel_write(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    views: &[vk::BufferView],
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(ty)
        .texel_buffer_view(views)
}

/// The write of `infos` into `binding` of `set`, from its first element.
fn write(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    infos: &[vk::DescriptorImageInfo],
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(ty)
        .image_info(infos)
}

impl Device {
    /// No Metal counterpart: grants the consumer `first..first + count` of `class`, the range it
    /// names, so the resource IDs it chooses from it reproduce in replay. A texture or a sampler
    /// state is made at an index of a grant of its class, and so is each constexpr sampler a
    /// pipeline state numbers. A grant lasts as long as the device.
    ///
    /// Err: [`Limit::IndexRange`] where the range is empty or leaves the class's
    /// ([`IndexClass::range`]); [`Refusal::IndicesGranted`] where one of its indices is granted
    /// already.
    pub fn grant_indices(
        &self,
        class: IndexClass,
        first: u32,
        count: u32,
    ) -> Result<IndexGrant, Error> {
        lock(&self.core().arrays.grants).grant(class, first, count)
    }
}
