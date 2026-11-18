//! Set 1: the descriptor arrays every pipeline takes its textures and samplers from, each element
//! at the index a `gpuResourceID` holds (`docs/RESOURCE-ID.md`), and the index grants the consumer
//! chooses those indices from.
//!
//! One descriptor set of the translator's identity domains' five arrays, made at device creation
//! and never rewritten for layouts: binding 0 the sampled textures (65,536), binding 1 the
//! storage-capable textures (4,096), and bindings 2, 3 and 4 the samplers of each use, plain,
//! depth comparison and integer texture (4,000 each). Every binding is partially bound and
//! updated after bind, also while work that uses the set is pending, so writing one element needs
//! no lock: each element has one owner, which writes it alone.
//!
//! Every element no live object owns holds a defined value, so a nil, stale or forged index reads
//! zeros or samples plainly on every device, where an element never written would be undefined
//! behaviour. A texture element is a null descriptor (`nullDescriptor`), which reads zeros,
//! answers 0 to its size queries and drops its writes, as Metal's nil texture does. A sampler
//! element is its array's default sampler, nearest and clamped to the edge; the comparison
//! array's compares with `NEVER`, so a comparison through it answers 0, as Metal's default
//! comparison (`MTLCompareFunctionNever`) does. Vulkan has no null sampler. The device writes
//! every element once at creation ([`Arrays::new`]).
//!
//! Where the device reads attachments through textures (`floor::Capabilities::attachment_reads`),
//! the sampled-texture array is twice as long: element `i + 65,536` is texture `i`'s in the
//! attachment feedback layout, written with element `i` for a texture whose image a pass may move
//! there (`crate::render_encoder`), and a null descriptor otherwise. No index of a grant reaches
//! it; a draw pushes it in place of the texture's own while its image is in that layout.
//!
//! The consumer names each range of indices it takes ([`Device::grant_indices`]), so resource IDs
//! are chosen on the guest's side and reproduce in replay. Three classes, disjoint: storage-capable
//! textures take 1 to 4,095, one index valid in both texture arrays; other textures 4,096 to
//! 65,535; samplers 1 to 3,999. Index 0 is nil and never granted.

use crate::counters::{Counters, Gauge};
use crate::device::lock;
use crate::error::{Error, Limit, Refusal};
use crate::floor::{BelowFloor, Missing, SAMPLED_IMAGES, SAMPLERS, SAMPLER_ARRAYS, STORAGE_IMAGES};
use crate::Device;
use ash::vk;
use std::ops::Range;
use std::sync::Mutex;

/// The binding of the sampled-texture array.
pub(crate) const SAMPLED_BINDING: u32 = 0;
/// The binding of the storage-capable texture array.
pub(crate) const STORAGE_BINDING: u32 = 1;
/// The binding of the first sampler array; the plain, comparison and integer-texture arrays
/// follow in that order.
pub(crate) const FIRST_SAMPLER_BINDING: u32 = 2;

/// Set 1's bindings, each its binding, descriptor type and count: the arrays a kernel's layout
/// names under the identity domains, which every pipeline's layout takes from this set.
pub(crate) const BINDINGS: [(u32, vk::DescriptorType, u32); 5] = [
    (
        SAMPLED_BINDING,
        vk::DescriptorType::SAMPLED_IMAGE,
        SAMPLED_IMAGES,
    ),
    (
        STORAGE_BINDING,
        vk::DescriptorType::STORAGE_IMAGE,
        STORAGE_IMAGES,
    ),
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
];

/// No Metal counterpart: the class of a texture or sampler index, which sets the range its grants
/// are taken from (`docs/RESOURCE-ID.md`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexClass {
    /// A texture without shader-write usage: 4,096 to 65,535 of the sampled-texture array.
    Texture,
    /// A texture with shader-write or unknown usage: 1 to 4,095, one index valid in both the
    /// sampled-texture and the storage-texture array.
    StorageTexture,
    /// A sampler: 1 to 3,999 of each sampler array. A Metal sampler state takes one, and so does
    /// each constexpr sampler a pipeline state numbers.
    Sampler,
}

impl IndexClass {
    /// The indices the class's grants are taken from.
    pub fn range(self) -> Range<u32> {
        match self {
            IndexClass::Texture => STORAGE_IMAGES..SAMPLED_IMAGES,
            IndexClass::StorageTexture => 1..STORAGE_IMAGES,
            IndexClass::Sampler => 1..SAMPLERS,
        }
    }

    fn slot(self) -> usize {
        match self {
            IndexClass::Texture => 0,
            IndexClass::StorageTexture => 1,
            IndexClass::Sampler => 2,
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

    fn put(&mut self, index: u32, on: bool) {
        let word = &mut self.0[(index / 64) as usize];
        match on {
            true => *word |= 1 << (index % 64),
            false => *word &= !(1 << (index % 64)),
        }
    }
}

/// The indices granted, and of those the indices an object lives at, one bit per index of each
/// class's array.
pub(crate) struct Grants {
    granted: [Bits; 3],
    occupied: [Bits; 3],
}

impl Grants {
    pub(crate) fn new() -> Grants {
        let bits = || {
            [
                Bits::new(SAMPLED_IMAGES),
                Bits::new(STORAGE_IMAGES),
                Bits::new(SAMPLERS),
            ]
        };
        Grants {
            granted: bits(),
            occupied: bits(),
        }
    }

    fn bit(&self, class: IndexClass, index: u32) -> bool {
        self.granted[class.slot()].get(index)
    }

    fn set(&mut self, class: IndexClass, index: u32, on: bool) {
        self.granted[class.slot()].put(index, on);
    }

    /// Takes `index` of `class` for an object, where it is granted and no object lives there.
    pub(crate) fn occupy(&mut self, class: IndexClass, index: u32) -> Result<(), Error> {
        if !class.range().contains(&index) || !self.bit(class, index) {
            return Err(Error::Refused(Refusal::IndicesNotGranted { class, index }));
        }
        let occupied = &mut self.occupied[class.slot()];
        if occupied.get(index) {
            return Err(Error::Refused(Refusal::IndexInUse { class, index }));
        }
        occupied.put(index, true);
        Ok(())
    }

    /// Gives back `index` of `class`, which [`Grants::occupy`] took.
    pub(crate) fn vacate(&mut self, class: IndexClass, index: u32) {
        self.occupied[class.slot()].put(index, false);
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
            self.set(class, index, true);
        }
        Ok(grant)
    }

    /// Returns `grant`, where every index of it is granted and no object lives at one.
    pub(crate) fn release(&mut self, grant: IndexGrant) -> Result<(), Error> {
        let IndexGrant { class, .. } = grant;
        let range = class.range();
        let inside = grant.first >= range.start
            && grant
                .first
                .checked_add(grant.count)
                .is_some_and(|end| end <= range.end);
        let missing = match inside {
            true => grant.indices().find(|&index| !self.bit(class, index)),
            false => Some(grant.first),
        };
        if let Some(index) = missing {
            return Err(Error::Refused(Refusal::IndicesNotGranted { class, index }));
        }
        let occupied = &self.occupied[class.slot()];
        if let Some(index) = grant.indices().find(|&index| occupied.get(index)) {
            return Err(Error::Refused(Refusal::IndicesInUse { class, index }));
        }
        for index in grant.indices() {
            self.set(class, index, false);
        }
        Ok(())
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
    pub(crate) grants: Mutex<Grants>,
}

impl Arrays {
    /// Makes set 1 and writes every element its default; with `feedback`, the sampled-texture
    /// array holds the feedback elements too.
    ///
    /// # Safety
    ///
    /// `device` is live and was made with the floor's features, `nullDescriptor` and the
    /// update-after-bind features among them.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        counters: &Counters,
        feedback: bool,
    ) -> Result<Arrays, Error> {
        let mut made = Made::default();
        // SAFETY: the caller's contract; what was made is destroyed where a later step fails.
        let arrays = unsafe { made.make(device, counters, feedback) };
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

    /// Writes a texture's elements at `index` of `class`: the sampled-texture array's with
    /// `sampled` and its layout, and for a storage-capable texture the storage-texture array's
    /// with `storage`; a null descriptor where a view is `None`. With `feedback`, also the
    /// sampled-texture array's element at `index` plus [`SAMPLED_IMAGES`], with the sampled view
    /// in the attachment feedback layout.
    ///
    /// # Safety
    ///
    /// The caller owns the elements (the index is occupied for it, or is being vacated by it), and
    /// each view is live and of the device, of an image in that layout whenever work uses it;
    /// `feedback` only where the set was made with the feedback elements.
    pub(crate) unsafe fn write_texture(
        &self,
        device: &ash::Device,
        class: IndexClass,
        index: u32,
        sampled: Option<(vk::ImageView, vk::ImageLayout)>,
        storage: Option<vk::ImageView>,
        feedback: bool,
    ) {
        let (view, layout) = sampled.unwrap_or((
            vk::ImageView::null(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ));
        let looped = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)];
        let sampled = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(layout)];
        let storage = [vk::DescriptorImageInfo::default()
            .image_view(storage.unwrap_or_default())
            .image_layout(vk::ImageLayout::GENERAL)];
        if feedback {
            let write = write(
                self.set,
                SAMPLED_BINDING,
                vk::DescriptorType::SAMPLED_IMAGE,
                &looped,
            )
            .dst_array_element(index + SAMPLED_IMAGES);
            // SAFETY: as below; the element is the texture's alone.
            unsafe { device.update_descriptor_sets(&[write], &[]) };
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
        let count = match class {
            IndexClass::StorageTexture => 2,
            _ => 1,
        };
        // SAFETY: the caller's contract; the bindings are update-after-bind and
        // update-unused-while-pending, so elements of other owners may be written meanwhile, and
        // a null image view is a null descriptor under `nullDescriptor`.
        unsafe { device.update_descriptor_sets(&writes[..count], &[]) };
    }

    /// Writes elements at `index` of the sampler arrays, each `(array, sampler)`, the array 0
    /// plain, 1 comparison and 2 integer texture: the array's default where the sampler is `None`.
    /// The elements not listed are left as they are.
    ///
    /// # Safety
    ///
    /// The caller owns the elements (the index is occupied for it, or is being vacated by it), no
    /// pending work uses one it writes, and each sampler is live and of the device whenever work
    /// uses it.
    pub(crate) unsafe fn write_samplers(
        &self,
        device: &ash::Device,
        index: u32,
        elements: &[(usize, Option<vk::Sampler>)],
    ) {
        let defaults = [self.plain, self.compare, self.plain];
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = elements
            .iter()
            .map(|&(array, sampler)| {
                [vk::DescriptorImageInfo::default().sampler(sampler.unwrap_or(defaults[array]))]
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = elements
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
        feedback: bool,
    ) -> Result<Arrays, Error> {
        let sampled = sampled_images(feedback);
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
            let mut counts = BINDINGS.map(|(_, _, count)| count);
            counts[SAMPLED_BINDING as usize] = sampled;
            with_arrays_layout(counts, |info| {
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
                descriptor_count: STORAGE_IMAGES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: SAMPLER_ARRAYS * SAMPLERS,
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
            grants: Mutex::new(Grants::new()),
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
    counts: [u32; 5],
    f: impl FnOnce(&vk::DescriptorSetLayoutCreateInfo) -> R,
) -> R {
    let bindings = std::array::from_fn::<_, 5, _>(|index| {
        let (binding, ty, _) = BINDINGS[index];
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(ty)
            .descriptor_count(counts[index])
            .stage_flags(vk::ShaderStageFlags::ALL)
    });
    let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
        | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING; 5];
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

/// The elements of the sampled-texture array: [`SAMPLED_IMAGES`], twice that with the feedback
/// elements.
pub(crate) fn sampled_images(feedback: bool) -> u32 {
    match feedback {
        true => 2 * SAMPLED_IMAGES,
        false => SAMPLED_IMAGES,
    }
}

/// Writes every element of `set` its default: a null descriptor in each texture array, the
/// sampled-texture array's `sampled` elements, `plain` in the plain and integer-texture sampler
/// arrays and `compare` in the comparison array.
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
        STORAGE_IMAGES as usize
    ];
    let samplers =
        |sampler| vec![vk::DescriptorImageInfo::default().sampler(sampler); SAMPLERS as usize];
    let (plains, compares) = (samplers(plain), samplers(compare));
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
    ];
    // SAFETY: the caller's contract; a null image view is a null descriptor under
    // `nullDescriptor`, and each sampler is the device's.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
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
    /// state is made at an index of a live grant of its class, and so is each constexpr sampler a
    /// pipeline state numbers.
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

    /// No Metal counterpart: returns a grant [`Device::grant_indices`] made, so its indices may be
    /// granted again.
    ///
    /// Err: [`Refusal::IndicesNotGranted`] where one of its indices is not granted;
    /// [`Refusal::IndicesInUse`] where a texture or a sampler state lives at one, until its last
    /// handle is gone.
    pub fn release_indices(&self, grant: IndexGrant) -> Result<(), Error> {
        lock(&self.core().arrays.grants).release(grant)
    }
}
