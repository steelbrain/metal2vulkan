//! The set-1 descriptor heap and the handle table.
//!
//! [`heap_classes`] is the fixed enumeration of the heap arrays on a profile: every image class the
//! core can declare (`neutral::image_classes`) and the four sampler classes. The texture lowering
//! declares the handle table (`Synthetic(HandleTable)`, one per stage that loads a heap texture)
//! and one heap array per class a heap texture reaches, and link numbers them (set 1 for the
//! arrays). The binder's half:
//!
//! - **Members.** The textures a handle can name: every used image texture
//!   (`BindShape.used`, by consumer token) and every image texture bound at a slot of a stage
//!   with a handle table. A used texture reaches the stages its render stages name (compute:
//!   the kernel), a bound one its own stage; a member reaching no stage with a table is left out.
//!   Used textures come first, by token, then bound ones by slot; member `i` is element `i + 1`
//!   of **every** class array, so the table maps a resource id to one element. Element 0
//!   ([`NEUTRAL_ELEMENT`]) is each class's neutral, the value of the null handle and of every
//!   missed id. Texture buffers are not members (no heap class holds one).
//! - **Elements.** Element `e` of the array of a class holds member `e - 1` when it fits the
//!   class (`fit`: the Metal texture type's dimension, arrayedness, and multisampling, a
//!   sample count above 1 exactly when multisampled, the numeric type, a depth format for a
//!   depth-comparison class (as a compared slot takes), `ShaderRead` usage for a sampled class and
//!   `ShaderWrite` or Apple's private `BlockWritesOnly` for a storage one (Metal's validation takes
//!   a write of either), which never holds a depth or stencil format), through its
//!   view: a sampled one as a slot's (`plan::resources::sampled_view`), a storage one the direct
//!   format (an `A8Unorm` texture held in red its `R8_UNORM` image, whose texels a heap access
//!   converts where the handle carries `A8Unorm`'s class), which the class's formatless declaration
//!   serves exactly when the format's storage choice for every access the device lets a heap make
//!   (`heap_storage_use`) is `Identity` (or `AlphaInRed`) without an alias and the format serves
//!   those accesses without a format
//!   (`metal::format::without_format_gaps`: the device's features and the format's own
//!   `VkFormatFeatureFlags2` bits). A fitting member the heap cannot view (another storage choice,
//!   a format the profile cannot sample or store) is `Unsupported`, as for a slot: a handle may
//!   name it. Every other element up to the members' count is the class's neutral; the elements
//!   past it are left unwritten (the arrays are `PARTIALLY_BOUND`, and no table entry names them).
//!   All heap writes of one member share one layout, `GENERAL` when some storage array holds it; a
//!   member bound at a slot shares it with the slot's own writes, `GENERAL` when either is a
//!   storage image (`one_layout_per_slot`). The plan uses such a member once, at its slot (widened
//!   when the slot's own use is narrower).
//! - **Placement** ([`HeapPlacement`]). The elements above are the planned ones, which a consumer
//!   that writes a heap set per command uses as they are. Under `HeapPlacement::ResourceId` each
//!   member and listed argument sampler is at the element its resource ID names instead (a sampler
//!   at its `ArgumentSampler::element` where the consumer gives one), and the tables map each ID
//!   there: the plan is the same, and [`HeapElements`] gives the consumer each heap write's element
//!   (a member without an ID takes none: no handle names it), so commands whose IDs name the same
//!   resources write the same elements alike and can share one heap set. A consumer that keeps one
//!   heap set for every command buffer can hold every live texture there instead: it writes each
//!   texture's elements as the texture is made ([`heap_element`]) and binds one table of every ID
//!   ([`handle_entry`]) at each handle table, so an ID a function loads from memory reaches its
//!   texture though no command named it, as Metal's does.
//! - **Query-only classes.** A class whose numeric type is a query-only group's default
//!   (`RuntimeFact::HeapNumericType`) reports each member that fits it in everything but its
//!   numeric type (`PlanDiagnostic::IncompatibleHeapEntry`), as a slot reports an incompatible
//!   binding: a query through a handle to it answers the neutral's.
//! - **Capacity.** The members plus the null element must fit every heap array's count (link's
//!   heap capacity, at most [`TABLE_CAPACITY`]), and a table without heap arrays (a function that
//!   only tests handles for null) [`TABLE_CAPACITY`], else [`BindError::HeapTooLarge`].
//! - **Tables** (`table_bytes`). A stage's table has one entry more than the members reaching
//!   it (`WriteSource::HandleTable`); per draw or dispatch it holds each such member's
//!   `(resource id, element, flags)` with a nonzero resource id (0 is the null entry's),
//!   deduplicated by id (the lowest element wins: both name the same texture), sorted by id,
//!   after leading null entries that pad it to its planned length, so it stays sorted, its first
//!   entry is the null entry, and every id resolves as [`lookup_host`] says. A texture no member
//!   names misses. A member's flags are [`FLAG_OUT_OF_BOUNDS_ALPHA_ONE`] where its pixel format
//!   has no alpha (`metal::format::out_of_bounds_alpha_one`): a module whose shader-checked heap
//!   read observes the alpha takes Metal's out-of-bounds alpha from it; and its texel class in
//!   [`FLAG_TEXEL_CLASS`]. Every other bit is 0.
//! - **Heap samplers**. A heap image is sampled through a separate sampler binding
//!   (`BindingSource::HeapSampler`) in set 0: a constexpr sampler is the layout's immutable sampler
//!   and takes no write; a runtime `[[sampler(n)]]` is written with the sampler bound at slot `n`
//!   made for the use, or the neutral sampler where none is. One with an opaque-black border (the
//!   immutable sampler's, or the bound sampler's shape) meets every member a sampled array holds,
//!   so each such member's sampled view must read that border exactly
//!   (`metal::sampler::check_border_view`), as a separate slot sampler's textures must. One that
//!   filters linearly (the immutable sampler's filters, or the filtering
//!   `metal::sampler::vk_sampler` makes of the bound shape for the use) meets every member of its
//!   stage that a sampled float array holds, so each such member's view format must have the filter
//!   features it needs (`plan::resources::require_filter_features`), as a combined or separate
//!   pair's texture must, else `Unsupported{Format}`.
//! - **Argument samplers**. A sampler loaded from an argument buffer is a handle into
//!   `BindState::argument_samplers`, searched in the stage's sampler table
//!   (`Synthetic(SamplerTable)`, the handle table's layout over the samplers' ids, whose flags are
//!   each sampler's [`SamplerWord`]), and read from the set-1 sampler array of its use
//!   (`HandleClass::Sampler`) at its entry's element. Every sampler array and sampler table holds
//!   twice the entries of the shape's capacity class (`ArgumentSamplersShape::entries`, at most the
//!   heap's capacity): element `i + 1` of each array is `WriteSource::ArgumentSampler` for listed
//!   sampler `i`, made for the array's use, and element 0 that use's neutral sampler. The second
//!   half is headroom for states made between a command's encoding and its command buffer's commit,
//!   which a consumer lists again then ([`BindingPlan::relisted_sampler_tables`]). A class whose
//!   least count does not fit the capacity is [`BindError::HeapTooLarge`] at plan time, and a list
//!   longer than the planned entries at instantiation. Every listed shape must pass
//!   `metal::sampler::check_argument_sampler` for every use the arrays declare (the plan cannot see
//!   which sampler a buffer pairs with which use, so a pair that never meets can refuse), and an
//!   opaque-black border among them meets every member a sampled array holds, as a heap sampler's
//!   does. The table bytes list each listed sampler's `(resource id, element, flags)` with a
//!   nonzero id, as a texture table does. A listed cubic sampler, whose bicubic filter no stage
//!   emulates through an argument buffer, is listed with `SamplerWord::CUBIC` in its flags: a stage
//!   that tests the bit (`RuntimeFact::CubicArgumentSampler`) fails its command buffer where it
//!   reaches the sampler, and beside a stage that does not (a fragment stage never does, nor a
//!   stage that cannot write the command buffer's status) the plan is refused
//!   `NotImplemented(SamplerBicubicFilter)`.
//! - **Function tables**. A visible-function table loaded from an argument buffer is searched by
//!   its resource id in the stage's function-table directory (`Synthetic(FunctionTableDirectory)`,
//!   the handle table's layout over `BindState::function_tables`, flags 0) and read from the set-1
//!   function-table array (`HandleClass::FunctionTable`, a storage-buffer array). Both hold the
//!   entries of the shape's capacity class (`FunctionTablesShape::entries`, at most the heap's
//!   capacity): element `i + 1` is `WriteSource::FunctionTable` for listed table `i`, whose words
//!   the consumer binds, and element 0 the zero buffer, the null table. A class whose least count
//!   does not fit the capacity is [`BindError::HeapTooLarge`] at plan time, and a list longer than
//!   the planned entries at instantiation. On a device without the array's features the stage reads
//!   a function-table snapshot (`Synthetic(FunctionTableSnapshot)`) in its place, written
//!   `WriteSource::FunctionTableSnapshot`, and the directory gives each listed id its record's word
//!   offset there (`FunctionTableSnapshot` of the listed sizes) instead of an element; the capacity
//!   is then the handle table's reach alone. Where the listed tables' words may pass
//!   `maxStorageBufferRange` (`FunctionTablesShape::may_pass`), the variant that reads them through
//!   their device addresses (`ResourceTemplate::function_table_addresses`) declares neither: the
//!   directory's entry holds the table's address in its element and flags words, and a listed table
//!   without one cannot be represented.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use crate::bind::neutral::image_classes;
use crate::bind::neutral::NeutralResource;
use crate::bind::plan::compute::internal;
use crate::bind::plan::resources::{
    bind_error, desc_filtering, neutral, opaque_black_border, require_filter_features,
    sampled_view, view_type,
};
use crate::bind::plan::{
    AccessClass, BindingPlan, DescriptorWrite, ImageSource, ImageViewDesc, PlanDiagnostic,
    ResourceTemplate, ResourceUse, UseTarget, WriteSource,
};
use crate::bind::state::{
    ArgumentSamplersShape, BindShape, BindState, BoundTexture, FunctionTablesShape, HeapPlacement,
    ResourceClass, SamplerShape, TextureFacts, TextureShape, UsedResource,
};
use crate::bind::table::FunctionTableSnapshot;
use crate::device::requirements::FormatUsage;
use crate::device::DeviceProfile;
use crate::error::{BindError, Feature, Unsupported};
use crate::memory::handles::{
    lookup_host, ENTRY_ELEMENT, ENTRY_FLAGS, ENTRY_ID_HI, ENTRY_ID_LO,
    FLAG_OUT_OF_BOUNDS_ALPHA_ONE, FLAG_TEXEL_CLASS, NEUTRAL_ELEMENT, TABLE_CAPACITY,
    TABLE_ENTRY_BYTES,
};
use crate::metal::format::{
    is_depth_stencil, sampled_type, storage_choice, vk_format, without_format_gaps, StorageUse,
};
use crate::metal::sampler::{
    check_argument_sampler, check_border_view, vk_filtering, SamplerUse, SamplerWord,
};
use crate::metal::{self, TextureType};
use crate::pipeline::artifact::RuntimeDependency;
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::interface::AccessSet;
use crate::pipeline::layout::{BindingSource, DescriptorBinding, PipelineLayout};
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{
    CombinedSampler, HandleClass, ImageAccess, ImageDim, ImageHandleClass, SampledType, SamplerRef,
    SlotClass, SlotRef, Stage, SyntheticBinding,
};
use crate::vk;

/// What a heap entry holds: a used resource (by consumer token) or the resource bound at a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HeapSource {
    Used(u32),
    Bound(SlotRef),
}

/// Every heap array class of `profile`, sorted and distinct: the image classes the core can declare
/// there, then the sampler classes of every use (with comparison or without, of an integer texture
/// or not).
pub fn heap_classes(profile: &DeviceProfile) -> Vec<HandleClass> {
    let mut out: Vec<HandleClass> = image_classes(profile)
        .into_iter()
        .map(HandleClass::Image)
        .collect();
    for compare in [false, true] {
        for integer in [false, true] {
            out.push(HandleClass::Sampler { compare, integer });
        }
    }
    out.sort();
    out.dedup();
    out
}

/// One heap member (the module docs): element `i + 1` for `ResourceTemplate.heap_members[i]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct HeapMember {
    pub source: HeapSource,
    /// Its view facts as planned (`TextureShape::Texture`); the state's must be these.
    pub shape: TextureShape,
    /// The stages whose handle tables enter it.
    pub stages: Vec<Stage>,
}

/// One stage's handle table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct PlannedHandleTable {
    pub set: u32,
    pub binding: u32,
    pub stage: Stage,
    /// Its entries, the null entry included: one more than the members reaching the stage.
    pub entries: u32,
}

/// How a member fits a heap class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fit {
    /// The member is an element of the class's array.
    Holds,
    /// An unswizzled texture of a form a heap reads (`ExtendedRange::HEAP_READ`, not sparse) in a
    /// sampled float class: the class's array holds its neutral, and the unsigned integer class
    /// of its shape holds its raw texels, which a function's read decodes by the form its
    /// handle's class index names. A stage that samples heap textures does not read it so.
    Raw,
    /// It fits in everything but its numeric type.
    NumericType,
    /// It is another class's.
    No,
}

/// The dimension, arrayedness, and multisampling of a Metal texture type.
fn dims(t: TextureType) -> (ImageDim, bool, bool) {
    use TextureType as T;
    match t {
        T::Type1D => (ImageDim::D1, false, false),
        T::Type1DArray => (ImageDim::D1, true, false),
        T::Type2D => (ImageDim::D2, false, false),
        T::Type2DArray => (ImageDim::D2, true, false),
        T::Type2DMultisample => (ImageDim::D2, false, true),
        T::Type2DMultisampleArray => (ImageDim::D2, true, true),
        T::Cube => (ImageDim::Cube, false, false),
        T::CubeArray => (ImageDim::Cube, true, false),
        T::Type3D => (ImageDim::D3, false, false),
        T::TextureBuffer => (ImageDim::Buffer, false, false),
    }
}

/// How a texture of view facts `shape` fits the heap array of `class` (the module docs).
pub(crate) fn fit(shape: &TextureShape, class: &ImageHandleClass) -> Fit {
    let TextureShape::Texture {
        texture_type,
        pixel_format,
        sample_count,
        usage,
        ..
    } = *shape
    else {
        return Fit::No;
    };
    let storage = class.access == ImageAccess::Storage;
    // Metal's validation takes a write to a texture of `BlockWritesOnly` as one of `ShaderWrite`.
    let needs = if storage {
        metal::TextureUsage::SHADER_WRITE | metal::TextureUsage::BLOCK_WRITES_ONLY
    } else {
        metal::TextureUsage::SHADER_READ
    };
    let shaped = dims(texture_type) == (class.dim, class.arrayed, class.multisampled)
        && class.dim != ImageDim::Buffer
        && (sample_count > 1) == class.multisampled
        && (!class.depth_compare || is_depth_stencil(pixel_format))
        && usage.intersects(needs)
        && !(storage && is_depth_stencil(pixel_format));
    // A resident texture of a format the table holds no representation of (ASTC's 4x2 and 8x4
    // footprints) fits no class: its element is the neutral, and nothing is refused, since the heap
    // does not know which of its textures the function reads.
    match sampled_type(pixel_format) {
        _ if !shaped => Fit::No,
        // A form a heap reads: its raw texels in the unsigned integer class, the neutral in the
        // float one.
        _ if raw_held(shape) && class.access == ImageAccess::Sampled => match class.sampled_type {
            SampledType::Uint => Fit::Holds,
            SampledType::Float => Fit::Raw,
            _ => Fit::NumericType,
        },
        Some(t) if t == class.sampled_type => Fit::Holds,
        Some(_) => Fit::NumericType,
        None => Fit::No,
    }
}

/// Whether a member of facts `shape` is a texture a heap reads by its raw texels: an unswizzled,
/// not sparse texture of one sample, of a form of `ExtendedRange::HEAP_READ`.
fn raw_held(shape: &TextureShape) -> bool {
    match *shape {
        TextureShape::Texture {
            pixel_format,
            swizzle,
            sparse,
            sample_count,
            ..
        } => {
            !sparse
                && sample_count <= 1
                && swizzle.is_default()
                && crate::metal::format::ExtendedRange::of(pixel_format)
                    .and_then(crate::metal::format::ExtendedRange::heap_read)
                    .is_some()
        }
        TextureShape::Buffer { .. } => false,
    }
}

/// The form of `ExtendedRange::HEAP_READ` by whose raw texels alone a heap holds a texture of
/// `facts` (an unswizzled, not sparse texture of one sample of such a form), else `None`: its
/// sampled float classes hold the neutral ([`heap_element`]), which a function's read passes
/// by, decoding the raw texels its unsigned integer class holds, and so does a sample or gather the
/// function emulates; one it does not emulate states `RuntimeFact::HeapRawSamples`, and a plan
/// refuses such a texture its stage reaches.
pub fn heap_raw_form(facts: &TextureFacts) -> Option<crate::metal::format::ExtendedRange> {
    match raw_held(&facts.shape()) {
        true => crate::metal::format::ExtendedRange::of(facts.pixel_format),
        false => None,
    }
}

/// The view through which the heap array of `class` holds a member of facts `shape`, in
/// `layout`, on `profile` (the module docs): a sampled class's as a slot's
/// (`plan::resources::sampled_view`), a storage class's the direct format (an `A8Unorm` texture
/// held in red its `R8_UNORM` image), which the class's formatless declaration must serve.
/// `Unsupported` where the array cannot view it: an extended-range format (its raw texels need its
/// function's decode) but a form a heap reads in the unsigned integer class ([`Fit::Raw`]), its raw
/// view, a storage member whose storage choice or format the formatless declaration cannot serve,
/// or an `A8Unorm` one held in red whose swizzle moves its alpha (its handle's class is not
/// `A8Unorm`'s).
fn member_view(
    profile: &DeviceProfile,
    shape: &TextureShape,
    class: &ImageHandleClass,
    layout: vk::ImageLayout,
) -> Result<ImageViewDesc, BindError> {
    let TextureShape::Texture { pixel_format, .. } = *shape else {
        return Err(internal("a texture-buffer heap member"));
    };
    let raw = raw_held(shape)
        && class.access == ImageAccess::Sampled
        && class.sampled_type == SampledType::Uint;
    if crate::metal::format::ExtendedRange::of(pixel_format).is_some() && !raw {
        return Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::ArgumentBufferTextures,
        )));
    }
    if class.access == ImageAccess::Storage {
        // The class does not say which accesses the function makes, so the member serves
        // every access the device lets a heap make.
        let use_ = heap_storage_use(profile);
        let c = storage_choice(pixel_format, use_, profile).map_err(bind_error)?;
        // The class's one formatless declaration cannot convert per element: a member is viewed
        // through its own format.
        let per_element = c.class == StorageImageClass::Identity;
        let aliased = c.alias.is_some();
        // An `A8Unorm` texture held in red (`StorageImageClass::AlphaInRed`): its `R8_UNORM`
        // image, whose texels a heap access converts where the handle carries `A8Unorm`'s class
        // (`metal::texel_format::heap_alpha_in_red_index`), which a swizzle that moves its alpha
        // does not keep.
        let alpha_in_red = c.class == StorageImageClass::AlphaInRed;
        if alpha_in_red
            && crate::metal::texel_format::heap_class_index(pixel_format, shape.swizzle())
                != crate::metal::texel_format::heap_alpha_in_red_index()
        {
            return Err(BindError::Unsupported(Unsupported::not_implemented(
                Feature::ArgumentBufferTextures,
            )));
        }
        if !(per_element || alpha_in_red) || aliased || c.x_channel {
            return Err(BindError::Unsupported(Unsupported::not_implemented(
                Feature::ArgumentBufferTextures,
            )));
        }
        let format = match alpha_in_red {
            true => c.vk_format,
            false => crate::metal::format::direct_format(pixel_format)
                .ok_or_else(|| internal("a storage heap member without a format"))?,
        };
        // The declaration names no format, so the member's format must serve those accesses
        // without one.
        let gaps = without_format_gaps(format, use_, profile);
        if !gaps.is_empty() {
            return Err(BindError::Unsupported(Unsupported::needs_any_of(
                Feature::ArgumentBufferTextures,
                vec![gaps],
            )));
        }
        return Ok(ImageViewDesc {
            view_type: view_type(class)?,
            format,
            aspect: vk::ImageAspectFlags::COLOR,
            layout,
            components: vk::ComponentMapping::IDENTITY,
            level: None,
            shader_swizzle: false,
        });
    }
    let mut view = sampled_view(profile, pixel_format, class)?;
    view.layout = layout;
    Ok(view)
}

/// What one element of a set-1 heap array holds for a texture ([`heap_element`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapElement {
    /// The array's class's neutral ([`NeutralResource::for_image`]): the texture fits another
    /// class, or no class (a texture buffer, a format the table holds no representation of).
    Neutral,
    /// The texture, through this view.
    View(ImageViewDesc),
}

/// What the element at a texture's resource ID of the set-1 heap array `class` holds for that
/// texture, of `facts`, on `profile`, in `layout`, whichever command reads the ID: the
/// texture's view where it fits the class, as a plan's member's (the module docs' **Elements**),
/// else the class's neutral. A consumer that keeps one heap set for every command buffer (under
/// `HeapPlacement::ResourceId`) writes every live texture's elements so, so an ID a function
/// loads from memory reaches its texture though no command named it, as Metal's does.
///
/// Unlike a plan's member, nothing here checks the texture against a pipeline's heap samplers
/// (a linear filter's format features, an opaque-black border's view): a plan still refuses a
/// member it names that fails them. `Unsupported` where the class holds the texture's kind but
/// cannot view it, as for a member; `Internal` for a sampler or function-table class, which holds
/// no texture. A texture of a form a heap reads by its raw texels ([`heap_raw_form`] names it) is
/// the neutral in the sampled float class, as a plan's float array holds it, and its raw view in
/// the unsigned integer class of its shape, where a function's float read, and a sample or gather
/// it emulates, reach it; nothing here knows whether a function samples it otherwise, which a plan
/// that names it refuses.
pub fn heap_element(
    profile: &DeviceProfile,
    class: &HandleClass,
    facts: &TextureFacts,
    layout: vk::ImageLayout,
) -> Result<HeapElement, BindError> {
    let HandleClass::Image(image) = class else {
        return Err(internal(format!("a texture element of {class:?}")));
    };
    let shape = facts.shape();
    match fit(&shape, image) {
        Fit::Holds => member_view(profile, &shape, image, layout).map(HeapElement::View),
        // The float array holds its neutral, and a function's float read takes its raw texels
        // from the unsigned integer array, which views it ([`Fit::Holds`] there).
        Fit::Raw | Fit::NumericType | Fit::No => Ok(HeapElement::Neutral),
    }
}

/// The flags word of the handle-table entry of a texture of `facts`, for [`handle_entry`]: bit 1
/// where its pixel format has no alpha, so a shader-checked read past the texture takes Metal's
/// out-of-bounds alpha of 1; bits 16 to 28 the index of the texture's texel class
/// (`metal::texel_format::heap_class_index` of its pixel format and swizzle), by which the shader
/// emulates Metal's linear filter of the texture's texels, 0 for a swizzled texture; every other
/// bit 0. A caller passes the word through as is.
pub fn handle_entry_flags(facts: &TextureFacts) -> u32 {
    entry_flags(&facts.shape())
}

/// [`handle_entry_flags`] of a texture of view facts `shape`.
fn entry_flags(shape: &TextureShape) -> u32 {
    let pixel_format = shape.pixel_format();
    let swizzle = shape.swizzle();
    let alpha = match crate::metal::format::out_of_bounds_alpha_one(pixel_format) {
        Some(true) => FLAG_OUT_OF_BOUNDS_ALPHA_ONE,
        _ => 0,
    };
    let index = crate::metal::texel_format::heap_class_index(pixel_format, swizzle);
    let class = index << FLAG_TEXEL_CLASS.trailing_zeros();
    alpha | (class & FLAG_TEXEL_CLASS)
}

/// The bytes of one handle-table entry (the module docs' **Tables**): resource ID `id` at heap
/// element `element`, with `flags`. A consumer that keeps one table of every texture ID for every
/// command (under `HeapPlacement::ResourceId`) binds it at each `Synthetic(HandleTable)` in place
/// of the plan's per-command table: sorted by ID, its first entry the null entry
/// `handle_entry(0, 0, 0)`, at most [`HANDLE_TABLE_CAPACITY`] entries of [`HANDLE_ENTRY_BYTES`]
/// bytes each, bound over the entries the heap arrays' count reaches (an ID past them misses, as
/// one no entry names does).
pub fn handle_entry(id: u64, element: u32, flags: u32) -> [u8; HANDLE_ENTRY_BYTES as usize] {
    let mut bytes = [0u8; HANDLE_ENTRY_BYTES as usize];
    let mut put = |word: u32, v: u32| {
        let at = 4 * word as usize;
        bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    };
    put(ENTRY_ID_LO, id as u32);
    put(ENTRY_ID_HI, (id >> 32) as u32);
    put(ENTRY_ELEMENT, element);
    put(ENTRY_FLAGS, flags);
    bytes
}

/// The bytes of one handle-table entry ([`handle_entry`]).
pub const HANDLE_ENTRY_BYTES: u32 = TABLE_ENTRY_BYTES;

/// The most entries a handle table holds: the reach of the shader's search ([`handle_entry`]).
pub const HANDLE_TABLE_CAPACITY: u32 = TABLE_CAPACITY;

/// The element of every set-1 heap array that holds the array's neutral: a null handle's, and
/// every ID a table misses (the module docs).
pub const NEUTRAL_HEAP_ELEMENT: u32 = NEUTRAL_ELEMENT;

/// The stages with a table of `kind` (a handle table or a sampler table) in `layout`, and each
/// table's `(set, binding)`.
fn tables(layout: &PipelineLayout, kind: SyntheticBinding) -> Vec<(Stage, u32, u32)> {
    let mut out = Vec::new();
    for set in &layout.sets {
        for b in &set.bindings {
            if let BindingSource::Synthetic { stage, binding } = b.source {
                if binding == kind {
                    out.push((stage, set.set, b.binding));
                }
            }
        }
    }
    out
}

/// The heap members of `shape` for a pipeline whose handle tables are in `stages` (the module
/// docs); `reach` gives the stages a used resource's render stages reach.
pub(crate) fn members(
    shape: &BindShape,
    stages: &[Stage],
    reach: impl Fn(metal::RenderStages) -> Vec<Stage>,
) -> Vec<HeapMember> {
    let image = |t: &TextureShape| {
        matches!(t, TextureShape::Texture { texture_type, .. }
            if *texture_type != TextureType::TextureBuffer)
    };
    let mut out = Vec::new();
    for (&token, u) in &shape.used {
        let crate::bind::state::UsedClass::Texture(t) = u.class else {
            continue;
        };
        let reached: Vec<Stage> = reach(u.stages)
            .into_iter()
            .filter(|s| stages.contains(s))
            .collect();
        if image(&t) && !reached.is_empty() {
            out.push(HeapMember {
                source: HeapSource::Used(token),
                shape: t,
                stages: reached,
            });
        }
    }
    for (&stage, s) in &shape.stages {
        if !stages.contains(&stage) {
            continue;
        }
        for (&index, t) in &s.textures {
            if image(t) {
                out.push(HeapMember {
                    source: HeapSource::Bound(SlotRef {
                        stage,
                        class: SlotClass::Texture,
                        index,
                    }),
                    shape: *t,
                    stages: vec![stage],
                });
            }
        }
    }
    out
}

/// A heap sampler that filters linearly, and so needs filter features of the view format of
/// each member it samples: its stage, its filtering (`metal::sampler::vk_filtering` of a runtime
/// one's bound shape, or a constexpr one's immutable sampler), and its comparison. One that filters
/// nearest needs nothing of a format, and an integer texture's always does, so only float-texel
/// samplers are listed.
#[derive(Clone, Copy, Debug)]
struct HeapFilter {
    stage: Stage,
    filtering: (bool, vk::SamplerReductionMode),
    compare: bool,
}

/// What a plan's sampler arrays and tables hold of a sampler state of a shape its bind shape did
/// not list (`ArgumentSamplersShape::shapes`), as a consumer lists the states again when a
/// command buffer commits ([`BindingPlan::relisted_sampler_tables`]): the plan-time checks of a
/// listed shape (`HeapPlanner::new`, the module docs), kept so that a state made between a
/// command's encoding and its commit is listed exactly where they pass.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct SamplerAdmission {
    /// The classes of the plan's sampler arrays: each is a use that every listed shape must pass
    /// (`metal::sampler::check_argument_sampler`).
    pub classes: Vec<HandleClass>,
    /// Every stage with a sampler table tests its entries for a cubic filter
    /// (`RuntimeFact::CubicArgumentSampler`).
    pub cubic: bool,
    /// The members a sampled heap array holds were checked against an opaque-black border
    /// (`HeapPlanner::view`).
    pub opaque_black: bool,
}

impl SamplerAdmission {
    /// The uses of the plan's sampler arrays.
    fn uses(&self) -> impl Iterator<Item = SamplerUse> + '_ {
        self.classes.iter().filter_map(|c| c.sampler_use())
    }

    /// Whether the plan may list a sampler of shape `s` beside its stages' cubic tests: one
    /// that is not cubic, or every stage tests, or no array samples through it.
    fn cubic_held(&self, s: &SamplerShape) -> bool {
        !s.cubic || self.cubic || self.uses().next().is_none()
    }

    /// Whether the plan holds a state of shape `s` on `profile` as it holds a listed one: `s`
    /// passes `check_argument_sampler` for every use, [`cubic_held`](Self::cubic_held), and an
    /// opaque-black border only where the members were checked against one. Where no array
    /// samples (a function that only names the states), every shape.
    fn admits(&self, s: &SamplerShape, profile: &DeviceProfile) -> bool {
        let sampled = self.uses().next().is_some();
        self.uses()
            .all(|use_| check_argument_sampler(s, use_, profile).is_ok())
            && self.cubic_held(s)
            && (!sampled || !s.opaque_black_border || self.opaque_black)
    }
}

/// A plan's heap: its members, the pipeline's heap classes and handle tables, its argument
/// samplers, and what their writes need. A plan hands it every set-1 binding
/// ([`HeapPlanner::array`]), every handle table ([`HeapPlanner::table`]), and every sampler table
/// ([`HeapPlanner::sampler_table`]), then [`HeapPlanner::finish`]es its template.
pub(crate) struct HeapPlanner<'a> {
    profile: &'a DeviceProfile,
    members: Vec<HeapMember>,
    /// The image classes of the pipeline's heap arrays.
    classes: Vec<ImageHandleClass>,
    /// The handle tables, `(stage, set, binding)`.
    tables: Vec<(Stage, u32, u32)>,
    /// The classes whose numeric type is a query-only group's default.
    defaulted: BTreeSet<ImageHandleClass>,
    /// Per member, its views' layout.
    layouts: Vec<vk::ImageLayout>,
    /// Some heap sampler has an opaque-black border.
    opaque_black: bool,
    /// The heap samplers that filter linearly.
    filters: Vec<HeapFilter>,
    /// The stages that may sample, gather, or query a level of a texture a heap holds by its raw
    /// texels alone where they read the neutral (`RuntimeFact::HeapRawSamples`: with no sampler
    /// slot, or naming one whose bound sampler states what the emulation does not answer).
    sampling: BTreeSet<Stage>,
    /// The stages whose mode-2 heap writes the function has not built
    /// (`RuntimeFact::HeapWriteRounding`), each with whether it writes `float` texels there.
    write_rounding: BTreeMap<Stage, bool>,
    /// The heap classes each stage reads where a member of another numeric class reads the
    /// neutral (`RuntimeFact::HeapCrossClassNeutral`), each with whether every such member does
    /// (else only one of a format a crossing read does not take).
    cross_neutral: BTreeMap<Stage, BTreeMap<ImageHandleClass, bool>>,
    /// The sampler tables, `(stage, set, binding)`.
    sampler_tables: Vec<(Stage, u32, u32)>,
    /// The argument samplers' shape, the entries of every sampler array and table, and what
    /// they admit of a shape the bind shape did not list, when the pipeline has a sampler array
    /// or table.
    samplers: Option<(ArgumentSamplersShape, u32, SamplerAdmission)>,
    /// The function-table directories, `(stage, set, binding)`.
    directories: Vec<(Stage, u32, u32)>,
    /// The function-table snapshots, `(stage, set, binding)`, of a device without the
    /// function-table array's features.
    snapshots: Vec<(Stage, u32, u32)>,
    /// The pipeline reads its function tables through the addresses its directories hold
    /// (`VariantKey::addressed_function_tables`).
    addressed_tables: bool,
    /// The function tables' shape and the entries of the function-table array and every
    /// directory, when the pipeline has a directory.
    function_tables: Option<(FunctionTablesShape, u32)>,
    /// The zero buffer's bytes: the null table's (`neutral::zero_buffer_bytes`).
    zero_bytes: u32,
    /// Outputs: the tables planned, and the diagnostics.
    planned: Vec<PlannedHandleTable>,
    planned_samplers: Vec<PlannedHandleTable>,
    planned_directories: Vec<PlannedHandleTable>,
    diagnostics: Vec<PlanDiagnostic>,
}

impl<'a> HeapPlanner<'a> {
    /// The heap of a pipeline with `layout` and `dependencies` for `shape`, or `None` when the
    /// pipeline has no handle or sampler table; `reach` as for [`members`].
    pub fn new(
        profile: &'a DeviceProfile,
        layout: &PipelineLayout,
        dependencies: &[RuntimeDependency],
        shape: &BindShape,
        reach: impl Fn(metal::RenderStages) -> Vec<Stage>,
    ) -> Result<Option<HeapPlanner<'a>>, BindError> {
        let sampler_tables = tables(layout, SyntheticBinding::SamplerTable);
        let directories = tables(layout, SyntheticBinding::FunctionTableDirectory);
        let snapshots = tables(layout, SyntheticBinding::FunctionTableSnapshot);
        let tables = tables(layout, SyntheticBinding::HandleTable);
        let mut classes = Vec::new();
        let mut capacity = TABLE_CAPACITY;
        let mut opaque_black = false;
        let mut filters = Vec::new();
        // A stage that reads the neutral, or whose runtime sampler slot is bound to a sampler
        // whose word states what the emulation does not answer (an unbound slot's word is 0, a
        // nearest sampler's).
        let sampling: BTreeSet<Stage> = dependencies
            .iter()
            .filter_map(|d| match d.fact {
                RuntimeFact::HeapRawSamples {
                    stage,
                    sampler: None,
                } => Some(stage),
                RuntimeFact::HeapRawSamples {
                    stage,
                    sampler: Some(n),
                } => {
                    let word = shape
                        .stages
                        .get(&stage)
                        .and_then(|s| s.samplers.get(&n))
                        .map_or(0, |s| s.word.0);
                    let filter = word & metal::sampler::SamplerWord::LINEAR_FILTER;
                    let answered = filter == 0 || filter == metal::sampler::SamplerWord::LINEAR;
                    (!answered).then_some(stage)
                }
                _ => None,
            })
            .collect();
        let mut cross_neutral: BTreeMap<Stage, BTreeMap<ImageHandleClass, bool>> = BTreeMap::new();
        for d in dependencies {
            if let RuntimeFact::HeapCrossClassNeutral {
                stage,
                class,
                every_member,
            } = d.fact
            {
                *cross_neutral
                    .entry(stage)
                    .or_default()
                    .entry(class)
                    .or_default() |= every_member;
            }
        }
        let mut write_rounding = BTreeMap::new();
        for d in dependencies {
            if let RuntimeFact::HeapWriteRounding { stage, halves } = d.fact {
                *write_rounding.entry(stage).or_insert(false) |= halves;
            }
        }
        for set in &layout.sets {
            for b in &set.bindings {
                match b.source {
                    BindingSource::Heap(class) => {
                        classes.push(class);
                        capacity = capacity.min(b.count);
                    }
                    // A constexpr sampler's border and filtering are its immutable sampler's; a
                    // runtime one's the sampler bound at its slot made for the use. An unbound one
                    // is the neutral sampler, which filters nearest.
                    BindingSource::HeapSampler { stage, sampler } => {
                        let (border, filtering) = match sampler.sampler {
                            SamplerRef::Constexpr(_) => match b.immutable_sampler.as_ref() {
                                Some(d) => (opaque_black_border(d), Some(desc_filtering(d))),
                                None => (false, None),
                            },
                            SamplerRef::Slot(index) => match shape
                                .stages
                                .get(&stage)
                                .and_then(|s| s.samplers.get(&index))
                                .map(|s| &s.shape)
                            {
                                Some(s) => (
                                    s.opaque_black_border,
                                    Some(vk_filtering(s, sampler.sampler_use())),
                                ),
                                None => (false, None),
                            },
                        };
                        opaque_black |= border;
                        if let Some(filtering) = filtering.filter(|f| f.0 && !sampler.integer) {
                            filters.push(HeapFilter {
                                stage,
                                filtering,
                                compare: sampler.compare,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        // An argument sampler's uses, a cube's explicit gradients included.
        let sampler_uses: Vec<SamplerUse> =
            classes.iter().filter_map(|c| c.sampler_use()).collect();
        let function_table_array = classes.contains(&HandleClass::FunctionTable);
        let images = classes
            .iter()
            .filter(|c| matches!(c, HandleClass::Image(_)))
            .count();
        if images > 0 && tables.is_empty() {
            return Err(internal("a heap array without a handle table"));
        }
        if !sampler_uses.is_empty() && sampler_tables.is_empty() {
            return Err(internal("a sampler array without a sampler table"));
        }
        // Directories read the tables from the array, or from a snapshot on a device without
        // the array's features, or at the addresses they hold: one of the three, never two.
        let addressed_tables = dependencies
            .iter()
            .any(|d| d.fact == RuntimeFact::FunctionTableRange { addressed: true });
        let homes = [
            function_table_array,
            !snapshots.is_empty(),
            addressed_tables,
        ]
        .into_iter()
        .filter(|&h| h)
        .count();
        if homes != usize::from(!directories.is_empty()) {
            return Err(internal(
                "function-table directories without one home of the tables' words, or with two",
            ));
        }
        if tables.is_empty() && sampler_tables.is_empty() && directories.is_empty() {
            return Ok(None);
        }
        // The function tables: the class's entries.
        let function_tables = if directories.is_empty() {
            None
        } else {
            let t = &shape.function_tables;
            if t.least_count() > u64::from(capacity) {
                return Err(BindError::HeapTooLarge {
                    count: t.least_count(),
                    limit: u64::from(capacity),
                });
            }
            let entries = u32::try_from(t.entries().min(u64::from(capacity)))
                .map_err(|_| internal("a function-table directory beyond u32 entries"))?;
            Some((*t, entries))
        };
        // The argument samplers: the class's entries, each listed shape made for each use, and an
        // opaque-black border meeting the members.
        let samplers = if sampler_tables.is_empty() {
            None
        } else {
            let a = &shape.argument_samplers;
            if a.least_count() > u64::from(capacity) {
                return Err(BindError::HeapTooLarge {
                    count: a.least_count(),
                    limit: u64::from(capacity),
                });
            }
            for s in &a.shapes {
                for &use_ in &sampler_uses {
                    check_argument_sampler(s, use_, profile).map_err(bind_error)?;
                }
            }
            // A cubic sampler's bicubic filter through an argument buffer is not built.
            // A stage that tests its argument samplers' words for it marks the command buffer
            // where it reaches one (`RuntimeFact::CubicArgumentSampler`), so a listed cubic
            // sampler is refused only beside a stage's sampler table that does not test it.
            let tested = |stage: Stage| {
                dependencies
                    .iter()
                    .any(|d| d.fact == RuntimeFact::CubicArgumentSampler { stage })
            };
            // The same checks admit a shape the bind shape did not list as a command buffer
            // commits (`relisted_sampler_tables`); the members' border check is the plan's own
            // (`finish`).
            let admission = SamplerAdmission {
                classes: classes
                    .iter()
                    .copied()
                    .filter(|c| c.sampler_use().is_some())
                    .collect(),
                cubic: sampler_tables.iter().all(|&(stage, ..)| tested(stage)),
                opaque_black: false,
            };
            if !a.shapes.iter().all(|s| admission.cubic_held(s)) {
                return Err(BindError::Unsupported(Unsupported::not_implemented(
                    Feature::SamplerBicubicFilter,
                )));
            }
            if !sampler_uses.is_empty() {
                opaque_black |= a.shapes.iter().any(|s| s.opaque_black_border);
            }
            // Twice the class's entries: headroom for the states made between a command's
            // encoding and its command buffer's commit, which list again then.
            let entries = u32::try_from(a.entries().saturating_mul(2).min(u64::from(capacity)))
                .map_err(|_| internal("a sampler table beyond u32 entries"))?;
            Some((a.clone(), entries, admission))
        };
        let stages: Vec<Stage> = tables.iter().map(|t| t.0).collect();
        let members = members(shape, &stages, reach);
        let count = members.len() as u64 + 1;
        if count > u64::from(capacity) {
            return Err(BindError::HeapTooLarge {
                count,
                limit: u64::from(capacity),
            });
        }
        let classes: Vec<ImageHandleClass> = classes
            .iter()
            .filter_map(|c| match c {
                HandleClass::Image(i) => Some(*i),
                HandleClass::Sampler { .. } | HandleClass::FunctionTable => None,
            })
            .collect();
        let layouts = members
            .iter()
            .map(|m| {
                let storage = |c: &&ImageHandleClass| c.access == ImageAccess::Storage;
                if classes
                    .iter()
                    .filter(storage)
                    .any(|c| fit(&m.shape, c) == Fit::Holds)
                {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                }
            })
            .collect();
        let defaulted = dependencies
            .iter()
            .filter_map(|d| match d.fact {
                RuntimeFact::HeapNumericType { class } => Some(class),
                _ => None,
            })
            .collect();
        Ok(Some(HeapPlanner {
            profile,
            members,
            classes,
            tables,
            defaulted,
            layouts,
            opaque_black,
            filters,
            sampling,
            write_rounding,
            cross_neutral,
            sampler_tables,
            samplers,
            directories,
            snapshots,
            addressed_tables,
            function_tables,
            zero_bytes: crate::bind::neutral::zero_buffer_bytes(profile),
            planned: Vec::new(),
            planned_samplers: Vec::new(),
            planned_directories: Vec::new(),
            diagnostics: Vec::new(),
        }))
    }

    /// The write of `stage`'s handle table at `(set, binding)`.
    pub fn table(
        &mut self,
        stage: Stage,
        set: u32,
        binding: u32,
    ) -> Result<WriteSource, BindError> {
        if !self.tables.contains(&(stage, set, binding)) {
            return Err(internal(format!(
                "a handle table of {stage:?} out of place"
            )));
        }
        let reaching = self.members.iter().filter(|m| m.stages.contains(&stage));
        let entries = u32::try_from(reaching.count() + 1)
            .map_err(|_| internal("a handle table beyond u32 entries"))?;
        let bytes_len = entries
            .checked_mul(TABLE_ENTRY_BYTES)
            .ok_or_else(|| internal("a handle table beyond u32 bytes"))?;
        self.planned.push(PlannedHandleTable {
            set,
            binding,
            stage,
            entries,
        });
        Ok(WriteSource::HandleTable { stage, bytes_len })
    }

    /// The write of `stage`'s sampler table at `(set, binding)`: a handle table of the argument
    /// samplers' class entries.
    pub fn sampler_table(
        &mut self,
        stage: Stage,
        set: u32,
        binding: u32,
    ) -> Result<WriteSource, BindError> {
        if !self.sampler_tables.contains(&(stage, set, binding)) {
            return Err(internal(format!(
                "a sampler table of {stage:?} out of place"
            )));
        }
        let entries = self.sampler_entries()?;
        let bytes_len = entries
            .checked_mul(TABLE_ENTRY_BYTES)
            .ok_or_else(|| internal("a sampler table beyond u32 bytes"))?;
        self.planned_samplers.push(PlannedHandleTable {
            set,
            binding,
            stage,
            entries,
        });
        Ok(WriteSource::HandleTable { stage, bytes_len })
    }

    /// The write of `stage`'s function-table directory at `(set, binding)`: a handle table of the
    /// function tables' class entries.
    pub fn directory(
        &mut self,
        stage: Stage,
        set: u32,
        binding: u32,
    ) -> Result<WriteSource, BindError> {
        if !self.directories.contains(&(stage, set, binding)) {
            return Err(internal(format!(
                "a function-table directory of {stage:?} out of place"
            )));
        }
        let entries = self.function_table_entries()?;
        let bytes_len = entries
            .checked_mul(TABLE_ENTRY_BYTES)
            .ok_or_else(|| internal("a function-table directory beyond u32 bytes"))?;
        self.planned_directories.push(PlannedHandleTable {
            set,
            binding,
            stage,
            entries,
        });
        Ok(WriteSource::HandleTable { stage, bytes_len })
    }

    /// The write of `stage`'s function-table snapshot at `(set, binding)`: the listed tables'
    /// words, which the consumer writes as the command buffer commits.
    pub fn snapshot(
        &mut self,
        stage: Stage,
        set: u32,
        binding: u32,
    ) -> Result<WriteSource, BindError> {
        if !self.snapshots.contains(&(stage, set, binding)) {
            return Err(internal(format!(
                "a function-table snapshot of {stage:?} out of place"
            )));
        }
        Ok(WriteSource::FunctionTableSnapshot)
    }

    /// The entries of the function-table array and every directory.
    fn function_table_entries(&self) -> Result<u32, BindError> {
        self.function_tables
            .as_ref()
            .map(|(_, entries)| *entries)
            .ok_or_else(|| internal("function tables without a directory"))
    }

    /// The writes of the function-table array `b` of set `set`: the zero buffer, the null table, at
    /// element 0, and listed table `e - 1` at each element `e` of the class.
    fn function_table_array(
        &self,
        set: u32,
        b: &DescriptorBinding,
    ) -> Result<Vec<DescriptorWrite>, BindError> {
        if b.ty != vk::DescriptorType::STORAGE_BUFFER {
            return Err(internal(format!("a function-table array as {:?}", b.ty)));
        }
        let entries = self.function_table_entries()?;
        if entries > b.count {
            return Err(internal("a function-table array shorter than its entries"));
        }
        let write = |element: u32, source: WriteSource| DescriptorWrite {
            set,
            binding: b.binding,
            array_element: element,
            ty: b.ty,
            source,
        };
        let mut out = Vec::with_capacity(entries as usize);
        out.push(write(
            NEUTRAL_ELEMENT,
            WriteSource::Neutral(NeutralResource::ZeroBuffer {
                min_bytes: self.zero_bytes,
            }),
        ));
        for element in 1..entries {
            out.push(write(
                element,
                WriteSource::FunctionTable { index: element - 1 },
            ));
        }
        Ok(out)
    }

    /// The entries of every sampler array and table.
    fn sampler_entries(&self) -> Result<u32, BindError> {
        self.samplers
            .as_ref()
            .map(|(_, entries, _)| *entries)
            .ok_or_else(|| internal("argument samplers without a sampler table"))
    }

    /// The runtime sampler slot that feeds `stage`'s heap sampler `sampler`, whose layout binding
    /// carries `immutable`: `None` for a constexpr sampler, which is that immutable sampler and
    /// takes no write, and the `[[sampler(n)]]` slot for a runtime one, whose write is the sampler
    /// bound there made for the use (`ResourcePlanner::heap_sampler`). Every heap member a sampled
    /// array holds meets a runtime sampler's opaque-black border and its filtering, so the border
    /// and filter checks cover it (`HeapPlanner::new`, `HeapPlanner::array`).
    pub fn sampler(
        &self,
        stage: Stage,
        sampler: CombinedSampler,
        immutable: Option<&vk::SamplerDesc>,
    ) -> Result<Option<SlotRef>, BindError> {
        match (sampler.sampler, immutable) {
            (SamplerRef::Constexpr(_), Some(_)) => Ok(None),
            (SamplerRef::Constexpr(_), None) => Err(internal(
                "a constexpr heap sampler without its immutable sampler",
            )),
            (SamplerRef::Slot(index), None) => Ok(Some(SlotRef {
                stage,
                class: SlotClass::Sampler,
                index,
            })),
            (SamplerRef::Slot(_), Some(_)) => {
                Err(internal("a runtime heap sampler with an immutable sampler"))
            }
        }
    }

    /// The writes of the heap array `b` of set `set`: element 0 and every member the array does
    /// not hold its class's neutral, every member it holds that member's view.
    pub fn array(
        &mut self,
        set: u32,
        b: &DescriptorBinding,
    ) -> Result<Vec<DescriptorWrite>, BindError> {
        let (binding, ty) = (b.binding, b.ty);
        let BindingSource::Heap(class) = b.source else {
            return Err(internal(format!("{:?} is no heap array", b.source)));
        };
        if set != 1 {
            return Err(internal(format!("a heap array in set {set}")));
        }
        let image = match class {
            HandleClass::Image(image) => image,
            // The array's `VkSampler`s are made for its use: the comparison and the texel class.
            HandleClass::Sampler { compare, integer } => {
                return self.sampler_array(set, b, (compare, integer))
            }
            HandleClass::FunctionTable => return self.function_table_array(set, b),
        };
        let expected = if image.access == ImageAccess::Storage {
            vk::DescriptorType::STORAGE_IMAGE
        } else {
            vk::DescriptorType::SAMPLED_IMAGE
        };
        if ty != expected {
            return Err(internal(format!("a heap array of {class:?} as {ty:?}")));
        }
        let n = neutral(&image)?;
        let write = |element: u32, source: WriteSource| DescriptorWrite {
            set,
            binding,
            array_element: element,
            ty,
            source,
        };
        let mut out = Vec::with_capacity(self.members.len() + 1);
        out.push(write(NEUTRAL_ELEMENT, WriteSource::Neutral(n)));
        for (i, m) in self.members.iter().enumerate() {
            let element =
                u32::try_from(i + 1).map_err(|_| internal("a heap element beyond u32"))?;
            let source = match fit(&m.shape, &image) {
                // A texture a stage may write by a mode-2 write the function has not built.
                Fit::Holds if image.access == ImageAccess::Storage && self.rounds(m) => {
                    return Err(BindError::Unsupported(Unsupported::not_implemented(
                        Feature::PrivateRenderState(
                            crate::error::PrivateRenderField::TextureWriteRounding,
                        ),
                    )));
                }
                Fit::Holds => {
                    let layout = self
                        .layouts
                        .get(i)
                        .copied()
                        .ok_or_else(|| internal("a heap member without a layout"))?;
                    let view = self.view(&m.shape, &image, layout)?;
                    self.check_filters(&image, &m.stages, view.format)?;
                    WriteSource::HeapEntry {
                        class,
                        element,
                        source: m.source,
                        view,
                    }
                }
                // The unsigned integer class holds its raw texels, which a stage's read decodes,
                // and its sample or gather where the function emulates it; a stage that samples
                // it otherwise would read this neutral (`RuntimeFact::HeapRawSamples`).
                Fit::Raw if m.stages.iter().any(|s| self.sampling.contains(s)) => {
                    return Err(BindError::Unsupported(Unsupported::not_implemented(
                        Feature::ArgumentBufferTextures,
                    )));
                }
                Fit::Raw => WriteSource::Neutral(n),
                fit => {
                    if fit == Fit::NumericType && self.defaulted.contains(&image) {
                        self.diagnostics
                            .push(PlanDiagnostic::IncompatibleHeapEntry {
                                class,
                                element,
                                source: m.source,
                                substituted: n,
                            });
                    }
                    // A member of another numeric class a stage reaches by an access that reads
                    // this neutral.
                    if fit == Fit::NumericType {
                        let crosses = match m.shape {
                            TextureShape::Texture { pixel_format, .. } => {
                                crate::metal::texel_format::crosses_classes(pixel_format)
                            }
                            _ => false,
                        };
                        for &stage in &m.stages {
                            let reached = self
                                .cross_neutral
                                .get(&stage)
                                .and_then(|c| c.get(&image))
                                .is_some_and(|&every| every || !crosses);
                            if reached {
                                self.diagnostics
                                    .push(PlanDiagnostic::CrossClassHeapNeutral {
                                        stage,
                                        class,
                                        element,
                                        source: m.source,
                                    });
                            }
                        }
                    }
                    WriteSource::Neutral(n)
                }
            };
            out.push(write(element, source));
        }
        Ok(out)
    }

    /// Whether a stage member `m` reaches writes it by a mode-2 write the function states it has
    /// not built (`RuntimeFact::HeapWriteRounding`): `m` is an `RG11B10Float` or `RGB9E5Float`
    /// texture, or a half-float one where the stage writes `float` texels
    /// (`metal::texel_format::mode_two_rounds`).
    fn rounds(&self, m: &HeapMember) -> bool {
        let TextureShape::Texture { pixel_format, .. } = m.shape else {
            return false;
        };
        m.stages.iter().any(|s| {
            self.write_rounding.get(s).is_some_and(|&halves| {
                crate::metal::texel_format::mode_two_rounds(pixel_format, halves)
            })
        })
    }

    /// The writes of the sampler array `b` of set `set` for the use (`compare`, `integer`):
    /// element 0 the use's neutral sampler, element `i + 1` argument sampler `i` made for the use,
    /// through the class's entries.
    fn sampler_array(
        &self,
        set: u32,
        b: &DescriptorBinding,
        (compare, integer): (bool, bool),
    ) -> Result<Vec<DescriptorWrite>, BindError> {
        if b.ty != vk::DescriptorType::SAMPLER {
            return Err(internal(format!(
                "a sampler array of {:?} as {:?}",
                b.source, b.ty
            )));
        }
        let entries = self.sampler_entries()?;
        if entries > b.count {
            return Err(internal("a sampler array shorter than its entries"));
        }
        let write = |element: u32, source: WriteSource| DescriptorWrite {
            set,
            binding: b.binding,
            array_element: element,
            ty: b.ty,
            source,
        };
        let mut out = Vec::with_capacity(entries as usize);
        out.push(write(
            NEUTRAL_ELEMENT,
            WriteSource::Neutral(NeutralResource::Sampler { compare }),
        ));
        for element in 1..entries {
            out.push(write(
                element,
                WriteSource::ArgumentSampler {
                    index: element - 1,
                    compare,
                    integer,
                },
            ));
        }
        Ok(out)
    }

    /// The view of a member of facts `shape` that the array of `class` holds, in `layout`
    /// ([`member_view`]), checked against the plan's heap samplers' opaque-black border.
    fn view(
        &self,
        shape: &TextureShape,
        class: &ImageHandleClass,
        layout: vk::ImageLayout,
    ) -> Result<ImageViewDesc, BindError> {
        if let (
            true,
            TextureShape::Texture {
                pixel_format,
                swizzle,
                ..
            },
        ) = (
            self.opaque_black && class.access != ImageAccess::Storage,
            *shape,
        ) {
            // A heap sampler's opaque-black border must read Metal's texel through the member's
            // view, its swizzle composed.
            let usages = [FormatUsage::SampledImage].into();
            let m = vk_format(pixel_format, &usages, self.profile).map_err(bind_error)?;
            check_border_view(true, pixel_format, &m, swizzle, self.profile).map_err(bind_error)?;
        }
        member_view(self.profile, shape, class, layout)
    }

    /// Check that a member reaching `stages` that the array of `class` holds through a view of
    /// `format` has the filter features each heap sampler of those stages that filters linearly
    /// needs of it (`plan::resources::require_filter_features`), as a combined or separate pair's
    /// texture must, else `Unsupported{Format}` naming the first missing one. A float-texel sampler
    /// samples only float classes, and a storage class is never sampled. The check is conservative
    /// as the border's is: the layout does not say which float classes a heap sampler samples, so a
    /// member of a class none of them reaches is checked too.
    fn check_filters(
        &self,
        class: &ImageHandleClass,
        stages: &[Stage],
        format: vk::Format,
    ) -> Result<(), BindError> {
        if class.access != ImageAccess::Sampled || class.sampled_type != SampledType::Float {
            return Ok(());
        }
        for f in self.filters.iter().filter(|f| stages.contains(&f.stage)) {
            require_filter_features(self.profile, f.filtering, f.compare, format)?;
        }
        Ok(())
    }

    /// Merge into `uses` the members bound at a slot that some heap array holds (a used texture
    /// is in the plan's `uses` already): read, and written when a storage array holds it, in its
    /// stage (`stages`). A slot the plan already uses keeps one use, widened to both.
    fn merge_uses(
        &self,
        uses: &mut Vec<ResourceUse>,
        stages: impl Fn(Stage) -> vk::PipelineStageFlags2,
    ) {
        for m in &self.members {
            let HeapSource::Bound(slot) = m.source else {
                continue;
            };
            let held = |a: ImageAccess| {
                self.classes
                    .iter()
                    .any(|c| c.access == a && fit(&m.shape, c) == Fit::Holds)
            };
            let (sampled, storage) = (held(ImageAccess::Sampled), held(ImageAccess::Storage));
            if !(sampled || storage) {
                continue;
            }
            let target = UseTarget::Slot(slot);
            match uses.iter_mut().find(|u| u.resource == target) {
                Some(u) => {
                    u.access.read = true;
                    u.access.write |= storage;
                    if storage {
                        u.class = AccessClass::StorageImage;
                    }
                    u.stages |= stages(slot.stage);
                }
                None => uses.push(ResourceUse {
                    resource: target,
                    access: AccessSet {
                        read: true,
                        write: storage,
                    },
                    class: if storage {
                        AccessClass::StorageImage
                    } else {
                        AccessClass::Sampled
                    },
                    stages: stages(slot.stage),
                    range: None,
                }),
            }
        }
    }

    /// Complete a plan: one layout per bound texture across its `descriptors`
    /// ([`one_layout_per_slot`]), the members' `uses` merged in (in `stages`), the members and
    /// tables into its `template`, and the diagnostics into its `diagnostics`.
    pub fn finish(
        self,
        descriptors: &mut [DescriptorWrite],
        uses: &mut Vec<ResourceUse>,
        stages: impl Fn(Stage) -> vk::PipelineStageFlags2,
        template: &mut ResourceTemplate,
        diagnostics: &mut Vec<PlanDiagnostic>,
    ) {
        one_layout_per_slot(descriptors);
        self.merge_uses(uses, stages);
        template.heap_members = self.members;
        template.handle_tables = self.planned;
        template.sampler_tables = self.planned_samplers;
        // A shape admitted as a command buffer commits meets the members as the plan checked
        // them against an opaque-black border.
        let opaque_black = self.opaque_black;
        let (shape, admission) = match self.samplers {
            Some((shape, _, admission)) => (
                Some(shape),
                SamplerAdmission {
                    opaque_black,
                    ..admission
                },
            ),
            None => (None, SamplerAdmission::default()),
        };
        template.argument_samplers = shape;
        template.sampler_admission = admission;
        template.function_table_directories = self.planned_directories;
        template.function_tables = self.function_tables.map(|(shape, _)| shape);
        template.function_table_snapshot = !self.snapshots.is_empty();
        template.function_table_addresses = self.addressed_tables;
        diagnostics.extend(self.diagnostics);
    }
}

/// The slot and view of a write of the texture bound at a slot: a plain or combined texture
/// write, or a heap element holding a bound member.
fn slot_view(w: &mut WriteSource) -> Option<(SlotRef, &mut ImageViewDesc)> {
    match w {
        WriteSource::Texture { slot, view }
        | WriteSource::CombinedImageSampler {
            image: ImageSource::Texture { slot, view },
            ..
        }
        | WriteSource::HeapEntry {
            source: HeapSource::Bound(slot),
            view,
            ..
        } => Some((*slot, view)),
        _ => None,
    }
}

/// Give every write of a texture bound at a slot one layout: `GENERAL` when some write of it is a
/// storage image (its slot's own storage binding, or a storage heap array holding it), its views'
/// layout otherwise. A static slot's storage write and the heap's sampled element of the same
/// texture, or a sampled slot and a storage heap element, would otherwise name two layouts for one
/// texture.
pub(crate) fn one_layout_per_slot(descriptors: &mut [DescriptorWrite]) {
    let mut general = BTreeSet::new();
    for d in descriptors.iter_mut() {
        if d.ty == vk::DescriptorType::STORAGE_IMAGE {
            if let Some((slot, _)) = slot_view(&mut d.source) {
                general.insert(slot);
            }
        }
    }
    for d in descriptors.iter_mut() {
        if let Some((slot, view)) = slot_view(&mut d.source) {
            if general.contains(&slot) {
                view.layout = vk::ImageLayout::GENERAL;
            }
        }
    }
}

/// The heap planner of a plan that writes a heap binding or a handle table.
pub(crate) fn planner<'p, 'a>(
    heap: &'p mut Option<HeapPlanner<'a>>,
) -> Result<&'p mut HeapPlanner<'a>, BindError> {
    heap.as_mut()
        .ok_or_else(|| internal("a heap binding without a handle table"))
}

/// The used resources of a bind state by token, in the state's order: a list the consumer gives
/// in token order (`BindState::used_in_token_order`), whose entries of a token a heap member finds
/// by search forward from where the last member's ended (a plan's members ascend by token,
/// [`members`]), so a plan's members together cost about the list's length; else gathered by
/// token once per instantiation, so each heap member finds its entries without rescanning the
/// state.
enum UsedByToken<'s> {
    /// The list, and where the last lookup's entries ended in it.
    Listed(&'s [UsedResource], Cell<usize>),
    Gathered(BTreeMap<u32, Vec<&'s UsedResource>>),
}

#[cfg(test)]
thread_local! {
    /// The entries of a listed state that lookups of its used resources compared
    /// ([`forward`]), on this thread.
    static LISTED_LOOKED_AT: Cell<u64> = const { Cell::new(0) };
}

/// The partition point of `items` under `below`, which holds of a prefix of them: found by
/// doubling a bound from their start, then searching under it, at the logarithm of the point
/// rather than of their count.
fn forward<T>(items: &[T], below: impl Fn(&T) -> bool) -> usize {
    let below = |item: &T| {
        #[cfg(test)]
        LISTED_LOOKED_AT.with(|looked| looked.set(looked.get() + 1));
        below(item)
    };
    let mut end = 1;
    while end <= items.len() && below(&items[end - 1]) {
        end *= 2;
    }
    let start = end / 2;
    start + items[start..end.min(items.len())].partition_point(below)
}

impl<'s> UsedByToken<'s> {
    /// `state`'s used resources by token.
    fn of(state: &'s BindState) -> UsedByToken<'s> {
        let used = &state.used_resources;
        if state.used_in_token_order {
            debug_assert!(
                used.windows(2).all(|w| w[0].token <= w[1].token),
                "used resources listed in token order are not"
            );
            return UsedByToken::Listed(used, Cell::new(0));
        }
        let mut gathered: BTreeMap<u32, Vec<&UsedResource>> = BTreeMap::new();
        for u in used {
            gathered.entry(u.token).or_default().push(u);
        }
        UsedByToken::Gathered(gathered)
    }

    /// The entries of `token`, in the state's order; `None` where it lists none. A listed
    /// state's are searched forward from where the last lookup's ended, where they lie past it,
    /// else from the list's start.
    fn get(&self, token: u32) -> Option<impl Iterator<Item = &'s UsedResource> + '_> {
        let (listed, gathered): (&'s [UsedResource], &[&'s UsedResource]) = match self {
            UsedByToken::Listed(used, from) => {
                let before = from.get().checked_sub(1).and_then(|at| used.get(at));
                let base = match before {
                    Some(before) if before.token < token => from.get(),
                    Some(_) | None => 0,
                };
                let start = base + forward(&used[base..], |u| u.token < token);
                let end = start + forward(&used[start..], |u| u.token == token);
                from.set(end);
                (&used[start..end], &[])
            }
            UsedByToken::Gathered(gathered) => (&[], gathered.get(&token)?),
        };
        let none = listed.is_empty() && gathered.is_empty();
        (!none).then(|| listed.iter().chain(gathered.iter().copied()))
    }
}

/// The resource id of heap member `m` in `state` (whose used resources are `used`), checked
/// against the plan's shape.
fn member_id(
    m: &HeapMember,
    used: &UsedByToken<'_>,
    state: &BindState,
) -> Result<Option<u64>, BindError> {
    let mismatch = || {
        internal(format!(
            "the bind state's shape differs from the plan's at heap member {:?}",
            m.source
        ))
    };
    match m.source {
        HeapSource::Used(token) => {
            let mut id = None;
            let entries = used.get(token).ok_or_else(mismatch)?;
            for u in entries {
                let ResourceClass::Texture(facts) = u.class else {
                    return Err(mismatch());
                };
                // A plan is keyed by `BindShape::for_plan`, which states a used texture at one
                // level: the member's level count is not the plan's.
                if facts.shape().at_one_level() != m.shape.at_one_level() {
                    return Err(mismatch());
                }
                match (id, u.resource_id) {
                    (Some(a), Some(b)) if a != b => {
                        return Err(internal(format!(
                            "used resource {token} lists two resource ids"
                        )))
                    }
                    (None, b) => id = b,
                    _ => {}
                }
            }
            Ok(id)
        }
        HeapSource::Bound(slot) => match state
            .stages
            .get(&slot.stage)
            .and_then(|t| t.textures.get(&slot.index))
        {
            Some(&BoundTexture::Texture { facts, resource_id })
                if facts.shape().at_one_level() == m.shape.at_one_level() =>
            {
                Ok(resource_id)
            }
            _ => Err(mismatch()),
        },
    }
}

/// The bytes of each planned handle table and sampler table of `template` for `state` (the module
/// docs): 16-byte entries `{id_lo, id_hi, element, flags}` of `u32` words, little-endian, a
/// member's flags its out-of-bounds-alpha flag, a sampler's its pixel-coordinate flag. Linear in
/// the members, the state's used resources, and its argument samplers (plus sorting each table).
pub(crate) fn table_bytes(
    template: &ResourceTemplate,
    state: &BindState,
) -> Result<Vec<crate::bind::instantiate::DescriptorBytes>, BindError> {
    let mut out = handle_table_bytes(&template.heap_members, &template.handle_tables, state)?;
    if let Some(shape) = &template.argument_samplers {
        out.extend(sampler_table_bytes(shape, &template.sampler_tables, state)?);
    } else if !template.sampler_tables.is_empty() {
        return Err(internal("a sampler table without its argument samplers"));
    }
    if let Some(shape) = &template.function_tables {
        out.extend(directory_bytes(
            shape,
            &template.function_table_directories,
            template.function_table_snapshot,
            template.function_table_addresses,
            state,
        )?);
    } else if !template.function_table_directories.is_empty() {
        return Err(internal("a function-table directory without its tables"));
    }
    Ok(out)
}

/// The bytes of the function-table directories `tables`, planned for the tables' `shape`, for
/// `state`: listed table `i` is element `i + 1` (or its resource ID's under
/// `HeapPlacement::ResourceId`), its flags 0; with `snapshot`, the word offset of its record in the
/// function-table snapshot of the listed sizes, whatever the placement; with `addresses`, the
/// device address of its words, low word as the element and high word as the flags. A state of
/// another shape is the plan's misuse; a list longer than the planned entries is
/// [`BindError::HeapTooLarge`], a snapshot past `u32::MAX` words is not representable, and a listed
/// nonzero id without an address, or an addressed directory that lists no table (whose null entry's
/// loads need one's address), is the consumer's [`BindError::Unsupported`] (`DynamicCall`, not
/// representable).
fn directory_bytes(
    shape: &FunctionTablesShape,
    tables: &[PlannedHandleTable],
    snapshot: bool,
    addresses: bool,
    state: &BindState,
) -> Result<Vec<crate::bind::instantiate::DescriptorBytes>, BindError> {
    if FunctionTablesShape::of(&state.function_tables) != *shape {
        return Err(internal(
            "the bind state's shape differs from the plan's at its function tables",
        ));
    }
    let offsets = match snapshot {
        true => Some(
            FunctionTableSnapshot::of(state.function_tables.iter().map(|f| f.size)).ok_or(
                BindError::Unsupported(Unsupported::not_representable(Feature::DynamicCall)),
            )?,
        ),
        false => None,
    };
    let element = |i: usize, id: u64| match &offsets {
        Some(snapshot) => snapshot
            .offsets()
            .get(i)
            .copied()
            .ok_or_else(|| internal("a listed table without its snapshot record")),
        None => placed(i, id, state.heap_placement),
    };
    let count = state.function_tables.len() as u64 + 1;
    let mut out = Vec::with_capacity(tables.len());
    for t in tables {
        if count > u64::from(t.entries) {
            return Err(BindError::HeapTooLarge {
                count,
                limit: u64::from(t.entries),
            });
        }
        let mut entries = Vec::with_capacity(state.function_tables.len());
        let mut listed = false;
        for (i, f) in state.function_tables.iter().enumerate() {
            if f.resource_id == 0 {
                continue;
            }
            if addresses {
                if f.address == 0 {
                    return Err(BindError::Unsupported(Unsupported::not_representable(
                        Feature::DynamicCall,
                    )));
                }
                listed = true;
                entries.push((f.resource_id, f.address as u32, (f.address >> 32) as u32));
                continue;
            }
            entries.push((f.resource_id, element(i, f.resource_id)?, 0));
        }
        // The null entry's loads take the last entry's address, a listed table's: a directory of
        // null entries alone gives them none.
        if addresses && !listed {
            return Err(BindError::Unsupported(Unsupported::not_representable(
                Feature::DynamicCall,
            )));
        }
        out.push(table_of(t, entries)?);
    }
    Ok(out)
}

/// The bytes of the handle tables `tables` of the heap members `members` for `state`.
fn handle_table_bytes(
    members: &[HeapMember],
    tables: &[PlannedHandleTable],
    state: &BindState,
) -> Result<Vec<crate::bind::instantiate::DescriptorBytes>, BindError> {
    if members.is_empty() && tables.is_empty() {
        return Ok(Vec::new());
    }
    let used = UsedByToken::of(state);
    let mut ids = Vec::with_capacity(members.len());
    for m in members {
        ids.push(member_id(m, &used, state)?);
    }
    // A member's flags: Metal's out-of-bounds read of a pixel format without alpha returns
    // alpha 1, and its texel class.
    let flags: Vec<u32> = members.iter().map(|m| entry_flags(&m.shape)).collect();
    let mut out = Vec::with_capacity(tables.len());
    for t in tables {
        let mut entries: Vec<(u64, u32, u32)> = Vec::new();
        for (i, ((m, id), &f)) in members.iter().zip(&ids).zip(&flags).enumerate() {
            let Some(id) = id.filter(|&id| id != 0) else {
                continue;
            };
            if m.stages.contains(&t.stage) {
                entries.push((id, placed(i, id, state.heap_placement)?, f));
            }
        }
        out.push(table_of(t, entries)?);
    }
    Ok(out)
}

/// The bytes of the sampler tables `tables`, planned for the argument samplers' `shape`, for
/// `state`: listed sampler `i` is element `i + 1`, its flags its [`SamplerWord`]. A state of
/// another shape is the plan's misuse; a list longer than the planned entries (a class past the
/// heap's capacity) is [`BindError::HeapTooLarge`].
fn sampler_table_bytes(
    shape: &ArgumentSamplersShape,
    tables: &[PlannedHandleTable],
    state: &BindState,
) -> Result<Vec<crate::bind::instantiate::DescriptorBytes>, BindError> {
    if ArgumentSamplersShape::of(&state.argument_samplers) != *shape {
        return Err(internal(
            "the bind state's shape differs from the plan's at its argument samplers",
        ));
    }
    listed_sampler_tables(tables, &state.argument_samplers, state.heap_placement)
}

/// The bytes of the sampler tables `tables` listing `samplers` under `placement`
/// ([`sampler_table_bytes`]); a list longer than the planned entries is
/// [`BindError::HeapTooLarge`].
fn listed_sampler_tables(
    tables: &[PlannedHandleTable],
    samplers: &[crate::bind::ArgumentSampler],
    placement: HeapPlacement,
) -> Result<Vec<crate::bind::instantiate::DescriptorBytes>, BindError> {
    let count = samplers.len() as u64 + 1;
    let mut out = Vec::with_capacity(tables.len());
    for t in tables {
        if count > u64::from(t.entries) {
            return Err(BindError::HeapTooLarge {
                count,
                limit: u64::from(t.entries),
            });
        }
        let mut entries = Vec::with_capacity(samplers.len());
        for (i, s) in samplers.iter().enumerate() {
            if s.resource_id == 0 {
                continue;
            }
            let element = placed(i, s.placed_at(), placement)?;
            entries.push((s.resource_id, element, SamplerWord::of(&s.desc).0));
        }
        out.push(table_of(t, entries)?);
    }
    Ok(out)
}

/// Whether the plan of a pipeline of `dependencies` for a bind shape of function tables `shape`
/// reads the tables through their device addresses (`VariantKey::addressed_function_tables`): a
/// pipeline compiled so states it, and one that reads tables by their ids otherwise asks for it
/// where the shape's tables may pass `maxStorageBufferRange` and the fact's mechanism is a variant.
/// Where it is a plan-time check (a device without device addresses), the plan binds the tables'
/// words whole, as before such a variant, and the consumer refuses a table, or a snapshot, that
/// passes the range where it binds it (`BufferRange`, naming device addresses): the shape's class
/// is too coarse to refuse it here without refusing tables that fit.
pub(crate) fn addressed_function_tables(
    profile: &DeviceProfile,
    dependencies: &[RuntimeDependency],
    shape: &FunctionTablesShape,
) -> bool {
    let mut reads = dependencies.iter().filter_map(|d| match d.fact {
        RuntimeFact::FunctionTableRange { addressed } => Some((addressed, d.mechanism)),
        _ => None,
    });
    let Some((addressed, mechanism)) = reads.next() else {
        return false;
    };
    addressed
        || (mechanism == Mechanism::Variant
            && shape.may_pass(profile.fields().limits.max_storage_buffer_range))
}

/// The element of planned member or listed sampler `i`, placed at `id` (nonzero: its resource
/// ID, or a listed sampler's element), under `placement`: `i + 1`, or `id` itself.
fn placed(i: usize, id: u64, placement: HeapPlacement) -> Result<u32, BindError> {
    match placement {
        HeapPlacement::Planned => {
            u32::try_from(i + 1).map_err(|_| internal("a heap element beyond u32"))
        }
        HeapPlacement::ResourceId => u32::try_from(id).map_err(|_| BindError::HeapTooLarge {
            count: id.saturating_add(1),
            limit: u64::from(u32::MAX),
        }),
    }
}

/// The heap set's number.
const HEAP_SET: u32 = 1;

/// Where a command's heap writes go under its bind state's [`HeapPlacement`]: the array element of
/// each of the plan's heap writes ([`BindingPlan::heap_elements`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeapElements {
    placement: HeapPlacement,
    /// Under `ResourceId`, by planned element `e`: member `e - 1`'s resource ID, `None` for a
    /// member without one (no handle names it).
    members: Vec<Option<u32>>,
    /// Under `ResourceId`, by planned element `e`: listed sampler `e - 1`'s element, else its
    /// resource ID.
    samplers: Vec<Option<u32>>,
    /// Under `ResourceId`, by planned element `e`: listed function table `e - 1`'s resource ID.
    function_tables: Vec<Option<u32>>,
}

impl HeapElements {
    /// The array element of `write`, one of the plan's descriptor writes: its own element for a
    /// write outside the heap set, the neutral element, and every write under `Planned`; under
    /// `ResourceId`, the resource ID of the member or listed function table it holds, or the
    /// listed sampler's element (its resource ID where it names none), or `None` where the
    /// command needs no write (a member no handle names, a sampler or
    /// table element past the list).
    pub fn element(&self, write: &DescriptorWrite) -> Option<u32> {
        let e = write.array_element;
        if write.set != HEAP_SET || e == NEUTRAL_ELEMENT || self.placement == HeapPlacement::Planned
        {
            return Some(e);
        }
        let held = match write.ty {
            vk::DescriptorType::SAMPLER => &self.samplers,
            vk::DescriptorType::STORAGE_BUFFER => &self.function_tables,
            _ => &self.members,
        };
        held.get(e as usize - 1).copied().flatten()
    }
}

impl BindingPlan {
    /// The resources this plan's commands instantiate: its compute or its graphics template's.
    fn resource_template(&self) -> Option<&ResourceTemplate> {
        match (&self.compute, &self.graphics) {
            (Some(c), _) => Some(&c.resources),
            (None, Some(g)) => Some(&g.resources),
            (None, None) => None,
        }
    }

    /// The `(set, binding)` of each sampler table this plan writes (the module docs'
    /// **Argument samplers**): where a command's bytes list its argument samplers.
    pub fn sampler_tables(&self) -> Vec<(u32, u32)> {
        self.resource_template().map_or_else(Vec::new, |t| {
            t.sampler_tables
                .iter()
                .map(|table| (table.set, table.binding))
                .collect()
        })
    }

    /// The bytes of this plan's sampler tables listing `samplers` under `placement`, where the
    /// plan, made on `device`, holds that list: no more samplers than each table's planned
    /// entries hold beside the null entry (twice the class's entries, the module docs), each of
    /// a shape the plan was made for or one that passes the plan's checks of a listed shape
    /// (`metal::sampler::check_argument_sampler` for every use of its sampler arrays, a cubic
    /// filter only beside stages that test for it, an opaque-black border only where the members
    /// were checked against one). A consumer whose function reaches a sampler state by its ID as
    /// the command runs, as Metal's does, lists the states live as the command buffer commits
    /// over the command's own, and writes their elements with [`WriteSource::ArgumentSampler`]'s
    /// rule.
    /// `None` where the plan does not hold the list (more states than its tables hold, or a shape
    /// its checks refuse), or has no sampler table: the command cannot reach the states as Metal's
    /// would.
    pub fn relisted_sampler_tables(
        &self,
        device: &crate::device::Device,
        samplers: &[crate::bind::ArgumentSampler],
        placement: HeapPlacement,
    ) -> Result<Option<Vec<crate::bind::instantiate::DescriptorBytes>>, BindError> {
        match self.resource_template() {
            Some(template) => {
                relisted_sampler_tables(template, device.profile(), samplers, placement)
            }
            None => Ok(None),
        }
    }
}

/// [`BindingPlan::relisted_sampler_tables`] of a plan whose resources are `template`, made on
/// `profile`.
fn relisted_sampler_tables(
    template: &ResourceTemplate,
    profile: &DeviceProfile,
    samplers: &[crate::bind::ArgumentSampler],
    placement: HeapPlacement,
) -> Result<Option<Vec<crate::bind::instantiate::DescriptorBytes>>, BindError> {
    let Some(shape) = &template.argument_samplers else {
        return Ok(None);
    };
    if template.sampler_tables.is_empty() {
        return Ok(None);
    }
    let count = samplers.len() as u64 + 1;
    // A shape the plan was not made for is held where the plan's checks of a listed one pass.
    let held = samplers.iter().all(|s| {
        let s = s.desc.shape();
        shape.shapes.contains(&s) || template.sampler_admission.admits(&s, profile)
    }) && template
        .sampler_tables
        .iter()
        .all(|t| count <= u64::from(t.entries));
    if !held {
        return Ok(None);
    }
    listed_sampler_tables(&template.sampler_tables, samplers, placement).map(Some)
}

impl BindingPlan {
    /// This plan's heap tables for `state`, written over `tables` (emptied first): the bytes of
    /// its handle tables, sampler tables and function-table directories, which
    /// [`instantiate_draw_into`](Self::instantiate_draw_into) writes after a draw's stage tables
    /// and [`instantiate_draw_apart_into`](Self::instantiate_draw_apart_into) leaves out. They
    /// are a function of the plan and of the state's heap inputs alone: its used resources (and
    /// whether it lists them in token order), the textures it binds at the plan's
    /// [`heap_slots`](Self::heap_slots), its argument samplers, its function tables and its heap
    /// placement. Their cost grows with the used resources and those slots. A state of another
    /// shape than the plan's is an internal error, a list past a table's planned entries
    /// [`BindError::HeapTooLarge`], and a function-table directory the plan cannot represent the
    /// consumer's [`BindError::Unsupported`] (`DynamicCall`); on an error `tables` holds none.
    pub fn heap_tables_into(
        &self,
        state: &BindState,
        tables: &mut Vec<crate::bind::instantiate::DescriptorBytes>,
    ) -> Result<(), BindError> {
        tables.clear();
        let template = self
            .resource_template()
            .ok_or_else(|| internal("a plan without a template"))?;
        tables.extend(table_bytes(template, state)?);
        Ok(())
    }

    /// The texture slots whose textures this plan's heap tables and heap elements read
    /// ([`heap_tables_into`](Self::heap_tables_into), [`heap_elements`](Self::heap_elements)):
    /// those of its heap members a slot binds, beside the used resources. None for a plan whose
    /// handles name used resources alone, or that reads no handle.
    pub fn heap_slots(&self) -> impl Iterator<Item = SlotRef> + '_ {
        let members = self.resource_template().map(|t| &t.heap_members);
        (members.into_iter().flatten()).filter_map(|m| match m.source {
            HeapSource::Bound(slot) => Some(slot),
            HeapSource::Used(_) => None,
        })
    }

    /// Where this plan's heap writes go for `state` ([`HeapElements`]), each heap array having
    /// `capacity` elements (the layout's set-1 count). Under [`HeapPlacement::ResourceId`] a
    /// member's element is its resource ID and a listed sampler's its
    /// [`element`](crate::bind::ArgumentSampler::element) or else its resource ID, refused
    /// [`BindError::HeapTooLarge`] at or past `capacity`; a state of another shape than the
    /// plan's is an internal error.
    pub fn heap_elements(
        &self,
        state: &BindState,
        capacity: u32,
    ) -> Result<HeapElements, BindError> {
        let template = self
            .resource_template()
            .ok_or_else(|| internal("a plan without a template"))?;
        heap_elements(template, state, capacity)
    }
}

/// [`BindingPlan::heap_elements`] of a plan whose resources are `template`.
fn heap_elements(
    template: &ResourceTemplate,
    state: &BindState,
    capacity: u32,
) -> Result<HeapElements, BindError> {
    let placement = state.heap_placement;
    if placement == HeapPlacement::Planned {
        return Ok(HeapElements::default());
    }
    let element = |id: Option<u64>| -> Result<Option<u32>, BindError> {
        match id.filter(|&id| id != 0) {
            Some(id) if id >= u64::from(capacity) => Err(BindError::HeapTooLarge {
                count: id.saturating_add(1),
                limit: u64::from(capacity),
            }),
            Some(id) => Ok(Some(id as u32)),
            None => Ok(None),
        }
    };
    let used = UsedByToken::of(state);
    let members = template
        .heap_members
        .iter()
        .map(|m| element(member_id(m, &used, state)?))
        .collect::<Result<_, _>>()?;
    let samplers = state
        .argument_samplers
        .iter()
        .map(|s| element(Some(s.placed_at())))
        .collect::<Result<_, _>>()?;
    let function_tables = match template.function_tables {
        Some(_) => state
            .function_tables
            .iter()
            .map(|t| element(Some(t.resource_id)))
            .collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    Ok(HeapElements {
        placement,
        members,
        samplers,
        function_tables,
    })
}

/// The bytes of table `t` holding `entries` (`(resource id, element, flags)`, ids nonzero):
/// deduplicated by id (the lowest element wins: both name the same resource), sorted by id, after
/// leading null entries that pad it to its planned length.
fn table_of(
    t: &PlannedHandleTable,
    mut entries: Vec<(u64, u32, u32)>,
) -> Result<crate::bind::instantiate::DescriptorBytes, BindError> {
    entries.sort_unstable();
    entries.dedup_by_key(|e| e.0);
    let len = t.entries as usize;
    if entries.len() >= len {
        return Err(internal("a handle table over its planned length"));
    }
    let mut table = vec![(0u64, NEUTRAL_ELEMENT, 0u32); len - entries.len()];
    table.extend(entries);
    debug_assert!(table.windows(2).all(|w| w[0].0 <= w[1].0));
    debug_assert!({
        let ids: Vec<(u64, u32)> = table.iter().map(|&(id, e, _)| (id, e)).collect();
        ids.iter()
            .all(|&(id, e)| lookup_host(&ids, id) == e || id == 0)
    });
    let bytes = table
        .iter()
        .flat_map(|&(id, element, f)| handle_entry(id, element, f))
        .collect();
    Ok(crate::bind::instantiate::DescriptorBytes {
        set: t.set,
        binding: t.binding,
        bytes,
    })
}

/// The accesses a storage heap element may take: a read or a write through the class's
/// formatless declaration wherever the device's `shaderStorageImage{Read,Write}WithoutFormat`
/// allows one (the lowering refuses a heap access without its feature, `declare_heap`).
fn heap_storage_use(profile: &DeviceProfile) -> StorageUse {
    let images = &profile.fields().images;
    StorageUse::texture(
        images.storage_read_without_format,
        images.storage_write_without_format,
    )
}

#[cfg(test)]
mod tests;
