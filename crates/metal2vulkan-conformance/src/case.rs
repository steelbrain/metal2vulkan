//! The case API: a case is written once and run by either side.
//!
//! A case names itself, sets its resources up, runs, and emits the sites it observes
//! ([`Case::run`]). It is written against [`Side`], the vocabulary both sides answer in, so porting
//! a case is a matter of saying which calls it makes and which sites it emits, and nothing about
//! which implementation is running it.
//!
//! A side with no public call for something a case reads answers [`Failure::unavailable`] rather
//! than pretending: Metal exposes an indirect command's read-back only privately, so the Metal side
//! answers `Unavailable` there and the case's Metal answer is carried from a measurement this tree
//! already holds ([`MetalSource::Carried`]). The case body stays one body.
//!
//! The vocabulary grows one family at a time as cases are ported. Nothing here reads an
//! environment variable, a device name or a vendor: a case that needs a feature finds out by making
//! the call and reading the typed refusal.

use crate::allowance::Allowance;
use crate::recording::ShaderRef;
use crate::transcript::{Node, NodeName, Observed, Sites, Transcript, Value};
use crate::verdict::device_gap;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    AttributeFormat, BindingAccess, CompareFunction, IndexType, PixelFormat, RenderStages,
    SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
    SamplerReductionMode, StepFunction,
};

/// A shader as both sides can take it.
///
/// One shader, two committed texts, because the two sides read different dialects of it: the
/// runtime parses the bitcode `llvm-as` assembles from the fixture's `.ll`, and Metal's own
/// assembler reads the fixture's `.air.ll` (typed pointers), which `xcrun metal-as` reassembles and
/// `xcrun metallib` packages. A case's input fingerprint covers both, so a change to either is a
/// change to the case's inputs.
#[derive(Clone, Copy, Debug)]
pub struct ShaderSource<'a> {
    /// The AIR bitcode the runtime parses.
    pub bitcode: &'a [u8],
    /// The fixture's typed-pointer AIR text, which Metal's assembler reads.
    pub air_text: &'a str,
    /// The entry point, for a side that names it as it makes the library.
    pub entry: &'a str,
}

/// A resource a case made on a side. The number is the side's own; nothing compares it, and no
/// recording holds one, because a Metal object's identity is not observable across hosts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Handle(pub u32);

/// What a dispatch binds: the buffers at 0 up, the textures at 0 up, and the sampler states at 0
/// up.
///
/// One value rather than one argument per kind, so a case that binds a kind the vocabulary gains
/// later does not change every call site, and so a reader of a case sees the whole argument table
/// in one place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bindings<'a> {
    /// The buffers, bound at Metal buffer index 0 up in this order.
    pub buffers: &'a [Handle],
    /// The textures, bound at Metal texture index 0 up in this order.
    pub textures: &'a [Handle],
    /// The sampler states, bound at Metal sampler index 0 up in this order
    /// (`setSamplerState:atIndex:`).
    pub samplers: &'a [Handle],
}

impl<'a> Bindings<'a> {
    /// `buffers` at 0 up, and no texture.
    pub const fn buffers(buffers: &'a [Handle]) -> Self {
        Self {
            buffers,
            textures: &[],
            samplers: &[],
        }
    }

    /// `textures` at 0 up, and no buffer.
    pub const fn textures(textures: &'a [Handle]) -> Self {
        Self {
            buffers: &[],
            textures,
            samplers: &[],
        }
    }

    /// `textures` at 0 up, `buffers` at 0 up, and no sampler.
    pub const fn new(textures: &'a [Handle], buffers: &'a [Handle]) -> Self {
        Self {
            buffers,
            textures,
            samplers: &[],
        }
    }

    /// The bindings with `samplers` bound at sampler index 0 up.
    #[must_use]
    pub const fn sampled(mut self, samplers: &'a [Handle]) -> Self {
        self.samplers = samplers;
        self
    }
}

/// What a case's texture is for, which is the usage and storage the side makes it with.
///
/// A case states this rather than letting each side choose, because Metal refuses a depth texture
/// of shader-write usage and a sampled read of a write-only one: the role is part of what the case
/// means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureRole {
    /// `MTLTextureUsageShaderRead`: a texture a kernel reads or samples.
    Read,
    /// `MTLTextureUsageShaderWrite`: a texture a kernel writes.
    Write,
    /// Both, as [`Side::texture_2d`] makes one.
    ReadWrite,
    /// `MTLTextureUsageRenderTarget`: a texture a pass renders into, and nothing else. A tile
    /// function's implicit imageblock is its pass's attachments, so a tile case's targets carry
    /// this usage alone, as its probe made them.
    RenderTarget,
}

/// A 2D texture a case makes: `MTLTextureDescriptor`'s facts as the cases need them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSpec {
    /// The pixel format, by its raw `MTLPixelFormat` value, so a case may name one of Apple's
    /// private formats (the two-plane extended-range family) as readily as a public one.
    pub format: PixelFormat,
    pub size: [u32; 2],
    /// Bytes a row of the bytes the case hands in and reads back. Every case states its own, since
    /// the pitch is part of what the case means.
    pub row_pitch: u32,
    pub role: TextureRole,
}

impl TextureSpec {
    /// A texture of `format`, `size` texels, at `row_pitch` bytes a row, in the role `role`.
    pub const fn new(
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        role: TextureRole,
    ) -> Self {
        Self {
            format,
            size,
            row_pitch,
            role,
        }
    }
}

/// One slot of an argument buffer a case builds ([`Side::argument_buffer`]): the object whose GPU
/// resource ID goes in that word.
///
/// The slots are what a kernel's argument-buffer struct declares, in `[[id(n)]]` order, and an
/// eight-byte resource ID each, which is how Metal 3 and the runtime both write one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgumentSlot {
    /// A texture's resource ID. A side that binds the buffer makes this texture resident.
    Texture(Handle),
    /// A sampler state's resource ID. The state must support argument buffers.
    Sampler(Handle),
}

/// A sampler state a case makes: `MTLSamplerDescriptor`'s facts a sampled case states.
///
/// The vocabulary is Metal's own enumerations as the translator's public API names them, so a value
/// a case writes reaches `MTLSamplerMinMagFilter`, `MTLSamplerMipFilter`, `MTLSamplerAddressMode`,
/// `MTLSamplerBorderColor`, `MTLCompareFunction` and `MTLSamplerReductionMode` by their raw
/// ordinals and reaches the runtime's descriptor unchanged. There is no second spelling to keep in
/// step.
///
/// Every field of the descriptor is here, and both sides set every one of them, so a field a case
/// leaves alone is Metal's own default **set by the case** rather than a field one leg never
/// touched. The one Metal property left out is `lodBias`, which the runtime's
/// `SamplerDescriptor` has no counterpart for, so a case stating it could reach only one leg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerSpec {
    pub min_filter: SamplerMinMagFilter,
    pub mag_filter: SamplerMinMagFilter,
    /// `mipFilter`: which level, or which pair of levels, a sample reads.
    pub mip_filter: SamplerMipFilter,
    pub s_address_mode: SamplerAddressMode,
    pub t_address_mode: SamplerAddressMode,
    /// `rAddressMode`: the third axis, which only a 3D or a cube sample consults.
    pub r_address_mode: SamplerAddressMode,
    pub border_color: SamplerBorderColor,
    /// `normalizedCoordinates`: `false` is a pixel-coordinate sampler.
    pub normalized_coordinates: bool,
    /// `lodMinClamp`: the level a sample is raised to, explicit levels included.
    pub lod_min_clamp: f32,
    /// `lodMaxClamp`: the level a sample is then lowered to.
    pub lod_max_clamp: f32,
    /// `lodAverage`: whether a fragment's level is averaged over the quad.
    pub lod_average: bool,
    /// `maxAnisotropy`: Metal's validation refuses anything outside 1 to 16, so a case states one
    /// inside it.
    pub max_anisotropy: u32,
    /// `compareFunction`: what a `sample_compare` compares its reference against the texel with.
    pub compare_function: CompareFunction,
    /// `reductionMode`: how a linear footprint's texels are combined.
    pub reduction_mode: SamplerReductionMode,
    /// `supportArgumentBuffers`: whether the state has a GPU resource ID, which is what an
    /// argument buffer needs of it ([`ArgumentSlot::Sampler`]). Metal gives a state made without
    /// it no resource ID at all.
    pub support_argument_buffers: bool,
}

impl SamplerSpec {
    /// `MTLSamplerDescriptor`'s defaults: nearest filtering without mipmaps, clamp to edge on all
    /// three axes, transparent black borders, normalized coordinates, LOD clamps `[0, FLT_MAX]`,
    /// no quad-averaged level, no anisotropy, no comparison, and weighted-average reduction.
    pub const fn new() -> Self {
        Self {
            min_filter: SamplerMinMagFilter::Nearest,
            mag_filter: SamplerMinMagFilter::Nearest,
            mip_filter: SamplerMipFilter::NotMipmapped,
            s_address_mode: SamplerAddressMode::ClampToEdge,
            t_address_mode: SamplerAddressMode::ClampToEdge,
            r_address_mode: SamplerAddressMode::ClampToEdge,
            border_color: SamplerBorderColor::TransparentBlack,
            normalized_coordinates: true,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            lod_average: false,
            max_anisotropy: 1,
            compare_function: CompareFunction::Never,
            reduction_mode: SamplerReductionMode::WeightedAverage,
            support_argument_buffers: false,
        }
    }

    /// This state with `supportArgumentBuffers` set, so an argument buffer can name it.
    #[must_use]
    pub const fn for_argument_buffer(mut self) -> Self {
        self.support_argument_buffers = true;
        self
    }

    /// The defaults with `filter` as both the minifying and the magnifying filter.
    pub const fn filtered(filter: SamplerMinMagFilter) -> Self {
        let mut spec = Self::new();
        spec.min_filter = filter;
        spec.mag_filter = filter;
        spec
    }

    /// The sampler with `mode` on both axes.
    #[must_use]
    pub const fn addressed(mut self, mode: SamplerAddressMode) -> Self {
        self.s_address_mode = mode;
        self.t_address_mode = mode;
        self
    }

    /// The words the case's input fingerprint covers: every field, so a sampler state that moves
    /// is a change to the case's inputs.
    ///
    /// **A field is a word only where it is not Metal's default.** `supportArgumentBuffers`
    /// established the pattern and the eight fields added beside it keep it, for one reason: a
    /// state at the default of every one of them fingerprints exactly as it did before those
    /// fields existed, so every committed recording stays the answer to the inputs it was taken
    /// from and not one of them decays. The two forms still fingerprint apart — one carries the
    /// field's words and the other does not.
    ///
    /// Each added field carries a **tag** word of its own beside its value, and the tags are
    /// outside the range any ordinal, flag or clamp of the six leading words can take
    /// ([`SAMPLER_WORD_TAG`] and up). Without a tag a `mipFilter` of `Nearest` would push the bare
    /// `1` that `supportArgumentBuffers` pushes, and two different states would fingerprint the
    /// same; `tests::the_tagged_sampler_words_tell_every_field_apart` holds that apart.
    pub fn words(&self) -> Vec<u8> {
        let mut words = vec![
            self.min_filter.as_raw(),
            self.mag_filter.as_raw(),
            self.s_address_mode.as_raw(),
            self.t_address_mode.as_raw(),
            self.border_color.as_raw(),
            u32::from(self.normalized_coordinates),
        ];
        if self.support_argument_buffers {
            words.push(1);
        }
        let default = Self::new();
        let mut tagged = |at: u32, value: u32| {
            words.push(SAMPLER_WORD_TAG + at);
            words.push(value);
        };
        if self.mip_filter != default.mip_filter {
            tagged(0, self.mip_filter.as_raw());
        }
        if self.r_address_mode != default.r_address_mode {
            tagged(1, self.r_address_mode.as_raw());
        }
        // The clamps fingerprint as their bits, so a NaN or a signed zero a case states is a
        // distinct input rather than one a comparison of floats would fold into another.
        if self.lod_min_clamp.to_bits() != default.lod_min_clamp.to_bits() {
            tagged(2, self.lod_min_clamp.to_bits());
        }
        if self.lod_max_clamp.to_bits() != default.lod_max_clamp.to_bits() {
            tagged(3, self.lod_max_clamp.to_bits());
        }
        if self.lod_average != default.lod_average {
            tagged(4, u32::from(self.lod_average));
        }
        if self.max_anisotropy != default.max_anisotropy {
            tagged(5, self.max_anisotropy);
        }
        if self.compare_function != default.compare_function {
            tagged(6, self.compare_function.as_raw());
        }
        if self.reduction_mode != default.reduction_mode {
            tagged(7, self.reduction_mode.as_raw());
        }
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    /// The state with `mip_filter` as its mip filter.
    #[must_use]
    pub const fn mipped(mut self, mip_filter: SamplerMipFilter) -> Self {
        self.mip_filter = mip_filter;
        self
    }

    /// The state with `lod_min_clamp` as its lower LOD clamp.
    #[must_use]
    pub const fn clamped_from(mut self, lod_min_clamp: f32) -> Self {
        self.lod_min_clamp = lod_min_clamp;
        self
    }

    /// The state with `compare_function` as its comparison.
    #[must_use]
    pub const fn comparing(mut self, compare_function: CompareFunction) -> Self {
        self.compare_function = compare_function;
        self
    }

    /// The state with `border_color` as its border colour.
    #[must_use]
    pub const fn bordered(mut self, border_color: SamplerBorderColor) -> Self {
        self.border_color = border_color;
        self
    }
}

/// The first tag [`SamplerSpec::words`] writes beside a non-default field's value.
///
/// Above every ordinal, flag and clamp the six leading words can hold, so a tag can never be
/// mistaken for one of them however the SDK's enumerations grow.
pub const SAMPLER_WORD_TAG: u32 = 0x8000_0000;

impl Default for SamplerSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// A kernel's `[[stage_in]]` descriptor, as both sides take it: Metal's
/// `MTLStageInputOutputDescriptor`.
///
/// The vocabulary is Metal's own enumerations, as the translator's public API names them, so a
/// value a case writes reaches `MTLAttributeFormat`, `MTLStepFunction` and `MTLIndexType` by their
/// raw ordinals and reaches the runtime's descriptor unchanged. There is no second spelling to
/// keep in step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageInput {
    /// The attributes a kernel reads, by their Metal attribute index.
    pub attributes: Vec<StageInAttribute>,
    /// The buffer layouts the attributes read, by their Metal buffer index.
    pub layouts: Vec<StageInLayout>,
    /// The index type an indexed step reads.
    pub index_type: IndexType,
    /// The Metal buffer index an indexed step reads its indices from.
    pub index_buffer_index: u32,
}

impl StageInput {
    /// A descriptor of `attributes` and `layouts`, whose indices are 32-bit words at
    /// `index_buffer_index`.
    pub fn new(
        attributes: Vec<StageInAttribute>,
        layouts: Vec<StageInLayout>,
        index_buffer_index: u32,
    ) -> Self {
        Self {
            attributes,
            layouts,
            index_type: IndexType::UInt32,
            index_buffer_index,
        }
    }
}

/// One `[[stage_in]]` attribute: Metal's `MTLAttributeDescriptor` at an attribute index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInAttribute {
    /// The Metal attribute index, as the shader declares it.
    pub index: u32,
    pub format: AttributeFormat,
    /// The attribute's byte offset inside an element.
    pub offset: u32,
    /// The Metal buffer index of the layout it reads.
    pub buffer_index: u32,
}

/// One `[[stage_in]]` buffer layout: Metal's `MTLBufferLayoutDescriptor` at a buffer index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInLayout {
    /// The Metal buffer index the layout is of.
    pub buffer_index: u32,
    /// Bytes between elements; `None` is `MTLBufferLayoutStrideDynamic`, the stride the encoder
    /// binds the buffer with.
    pub stride: Option<u32>,
    pub step_function: StepFunction,
    /// The step rate; Metal requires 0 of a constant step.
    pub step_rate: u32,
}

/// The stage-in region a dispatch fetches its `[[stage_in]]` attributes from: the two calls Metal
/// sets it with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageInRegion {
    /// `setStageInRegion:`.
    Direct { origin: [u32; 3], size: [u32; 3] },
    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`, whose buffer holds the origin's
    /// three words and then the size's.
    Indirect { buffer: Handle, offset: u64 },
}

/// What a side answered a call with.
pub type Answer<T> = Result<T, Failure>;

/// A call that did not answer: the typed error as the transcript records it, and whether it is the
/// venue's honest gap rather than ours.
///
/// # Why the `Err` is not stored
///
/// The comparison reads a **refusal** out of one shape and one only: Metal's site answered and
/// ours is an `Err` variant ([`crate::compare`]). So a failure whose recorded value is anything
/// else is a refusal the suite cannot see — it reads as two sides observing different kinds of
/// value, which is a divergence, which the known-gaps ledger cannot carry as a refusal and the
/// refused count never sees.
///
/// That had happened twice, in two files, for the same reason: a helper that **named where** a
/// failure came from wrapped a fields value around the whole `Err`. So the `Err` layer is not a
/// field here at all. A failure carries the typed error alone, [`Failure::recorded`] adds the
/// `Err` as the site is written, and the annotation ([`Failure::at_step`]) can only reach the
/// inside of it. Nothing a failure carries, now or later, can displace the layer the comparison
/// reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    /// The typed error **inside** the call's `Err`: a `Value::Variant` of the error's variant
    /// path and its payload. Private, so the `Err` layer cannot be lost: see the type's own
    /// documentation.
    error: Value,
    /// What the venue lacks, where the typed refusal names capabilities a device could enable or
    /// a measured driver defect ([`device_gap`]). `None` where the refusal is ours to answer for.
    pub lacks: Option<String>,
}

impl Failure {
    /// The failure a typed runtime error is.
    pub fn of(error: &metal2vulkan_runtime::Error) -> Self {
        Self {
            error: Value::Variant {
                name: error_variant(error),
                payload: Box::new(Value::Lines(vec![error.to_string()])),
            },
            lacks: device_gap(error),
        }
    }

    /// The failure a side with no public call for `call` answers. It is neither a divergence nor a
    /// refusal: it says the side cannot be asked, and a case whose Metal side answers it carries
    /// its Metal answer instead.
    pub fn unavailable(call: &str) -> Self {
        Self::of_variant(
            "Unavailable",
            Value::fields([("call", Value::Lines(vec![call.to_owned()]))]),
        )
    }

    /// A failure of `variant` with `detail`, for a side whose call failed outside the runtime's
    /// typed errors.
    pub fn named(variant: &str, detail: impl Into<String>) -> Self {
        Self::of_variant(variant, Value::Lines(vec![detail.into()]))
    }

    /// A failure of the typed variant `variant` carrying `payload`.
    fn of_variant(variant: &str, payload: Value) -> Self {
        Self {
            error: Value::Variant {
                name: variant.to_owned(),
                payload: Box::new(payload),
            },
            lacks: None,
        }
    }

    /// What the transcript records at a site the call did not answer: `Err` of the typed error,
    /// with what the venue lacks beside it where the refusal is the venue's own gap.
    ///
    /// The only way out of a failure into a value, so every site a failure reaches holds the shape
    /// the comparison reads a refusal from — **and**, since [`Failure::lacks`] rides inside that
    /// value, the shape it reads a device's own gap from.
    ///
    /// # Why `lacks` is in the value
    ///
    /// It used to reach the comparison only through
    /// [`Sites::device_gap`](crate::transcript::Sites::device_gap), keyed by the site's path from
    /// the root. Two places call that: [`call`] and [`probe`], as they emit the site, and
    /// [`crate::replay::replay_against`], at the path a case's own [`Abort`] names. So a case that
    /// pushed the failure's node itself — every case whose answer is a tree of facts has to, and
    /// 268 build their own `Abort` — and named a **different** path in its abort than the node it
    /// pushed had the fact filed under a site that does not exist. The refusal then read as a gap
    /// of ours at the site that does exist, which is the one thing the
    /// [`LacksFeature`](crate::verdict::Verdict::LacksFeature) verdict is there to prevent.
    ///
    /// Measured: `texture-fills-236` pushes its node at `work` and aborts at `work/236/<case>`, so
    /// `MTLPixelFormatASTC_12x12_HDR` refused for want of `VK_EXT_texture_compression_astc_hdr`
    /// read `refused` while the `astc-*-hdr` cases read `lacks-feature` for the same typed error
    /// in the same run. A path a case states twice cannot be made to agree by a comment; carrying
    /// the fact in the value removes the second path altogether.
    pub fn recorded(&self) -> Value {
        match &self.lacks {
            None => Value::err(self.error.clone()),
            Some(needs) => Value::err(Value::fields([
                (LACKS, Value::Lines(vec![needs.clone()])),
                (REFUSED, self.error.clone()),
            ])),
        }
    }

    /// The same failure, said to have happened at step `index` of a [`crate::work::Work`]
    /// program.
    #[must_use]
    pub fn at_step(self, index: usize) -> Self {
        self.inside("step", index)
    }

    /// The same failure, said to have happened while making resource `index` of one.
    #[must_use]
    pub fn at_resource(self, index: usize) -> Self {
        self.inside("resource", index)
    }

    /// [`Failure::at_step`] and [`Failure::at_resource`], which differ only in the field they
    /// name.
    ///
    /// One implementation for both sides of the harness. Each interpreter used to carry its own
    /// copy, with a comment saying the two must agree — and the shape they agreed on was the one
    /// that hid the refusal, so the agreement proved nothing. A program's part is named **inside**
    /// the typed error, never around it.
    fn inside(self, part: &'static str, index: usize) -> Self {
        // Both fields are named, with no `..`, so the next field anyone adds to a failure is a
        // compile error here rather than a field silently dropped as a failure is annotated.
        Self {
            error: Value::fields([(part, Value::Uint(index as u64)), (REFUSED, self.error)]),
            lacks: self.lacks,
        }
    }
}

/// The field a failure's recorded value holds the typed error under, once anything is named beside
/// it. Reserved: no typed error of the runtime records a field of this name.
const REFUSED: &str = "refused";

/// The field a failure's recorded value holds **what the venue lacks** under
/// ([`Failure::recorded`]). Reserved, as [`REFUSED`] is.
const LACKS: &str = "lacks";

/// What the venue honestly lacks at a site a [`Failure`] recorded, read back out of the recorded
/// value: `Some` where the typed refusal names capabilities a device could enable or a driver
/// measured to run the thing wrong ([`device_gap`]), and `None` for a refusal that is ours.
///
/// The comparison's reading of [`Failure::recorded`], and the reason a case cannot lose the fact
/// by naming its abort's site differently from the node it pushed: see that method.
pub fn lacks_recorded(value: &Value) -> Option<&str> {
    if !value.is_error() {
        return None;
    }
    let Value::Variant { payload, .. } = value else {
        return None;
    };
    let Value::Fields(named) = payload.as_ref() else {
        return None;
    };
    match named.get(LACKS) {
        Some(Value::Lines(words)) => words.first().map(String::as_str),
        _ => None,
    }
}

/// The variant path a typed runtime error records as: the error's outermost variant, which is what
/// a consumer matches on.
fn error_variant(error: &metal2vulkan_runtime::Error) -> String {
    let debug = format!("{error:?}");
    let head = debug
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .find(|word| !word.is_empty())
        .unwrap_or("Error");
    format!("Error::{head}")
}

/// Where a case's run stopped: a call it needed did not answer. An abort is not a verdict; the
/// comparison reads the sites emitted up to it, and the refused call is one of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Abort {
    /// The site the call was emitted at.
    pub at: String,
    pub failure: Failure,
}

/// The venue a side runs on: what a result carries so a report can say where it ran. A case never
/// branches on any of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Venue {
    /// The physical device's name.
    pub device: String,
    /// The driver, as the device reports it.
    pub driver: String,
    /// The API version the device runs.
    pub api: String,
    /// The Metal level the side answers as ([`metal2vulkan_runtime::MetalLevel::macos`]).
    pub metal_level: String,
}

impl Venue {
    /// The short label the report's matrix prints as a column.
    pub fn label(&self) -> String {
        self.device.clone()
    }
}

impl std::fmt::Display for Venue {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{} ({}, API {}, Metal level {})",
            self.device, self.driver, self.api, self.metal_level
        )
    }
}

/// The vocabulary a case is written against.
///
/// Every call answers [`Answer`], so a case records what it got and carries on or stops. A side
/// that has no public call for one answers [`Failure::unavailable`].
pub trait Side {
    /// Where the side runs.
    fn venue(&self) -> Venue;

    /// A library of the shader's AIR, from whichever of its forms the side reads
    /// ([`ShaderSource`]).
    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle>;

    /// `newComputePipelineStateWithFunction:error:` of the library's `entry`.
    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle>;

    /// `staticThreadgroupMemoryLength` of a compute pipeline state: the bytes the kernel's own
    /// `threadgroup` arrays take.
    ///
    /// This getter and not its two neighbours. `maxTotalThreadsPerThreadgroup` and
    /// `threadExecutionWidth` answer the **device's** limits for the kernel — Metal's 1024 and 32
    /// on Apple's GPUs, lower where a device's shared memory or its emulated simdgroups leave room
    /// for fewer — so a venue that answers less is not diverging from Metal, it is a venue with a
    /// smaller device, and the suite has no way to say that of a number (only a refused call
    /// carries what the venue lacks). A case pins the device's limit by **dispatching at Metal's
    /// own threadgroup size** instead, which such a venue refuses typed. The static threadgroup
    /// length is a fact about the shader, so it is the same wherever the shader is compiled, and
    /// comparing it is exact.
    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let _ = pipeline;
        Err(Failure::unavailable("static_threadgroup_memory_length"))
    }

    /// `newBufferWithBytes:length:options:` of shared storage.
    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle>;

    /// `newBufferWithLength:options:` of shared storage, zeroed.
    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle>;

    /// `newBufferWithBytesNoCopy:length:options:deallocator:` over pages **the side maps**, every
    /// byte the spec's pattern ([`crate::host_pages::HostBufferSpec`]).
    ///
    /// This is the one buffer in the suite that is not the device's own memory, and it is here for
    /// one reason: Metal's private `MTLResourceOptions` bit 24 is the only way a Metal program can
    /// say "the GPU's writes to this memory go nowhere", and whether we drop such a write exactly
    /// as Metal drops it is the owner's standing rule. A spec with `read_only` off is the control
    /// that makes the read-only run a measurement rather than a guess about whether the write path
    /// ran at all.
    ///
    /// The pages stay mapped for as long as the buffer, and the side never writes through them
    /// after the buffer is made: a CPU write through a read-only mapping ends the process, which is
    /// Metal's own behaviour and not something this suite asks about.
    /// [`Side::read_buffer`] of the handle is then the **CPU's** view of the pages; a copy the GPU
    /// makes out of the buffer into an ordinary one ([`crate::render::RenderWork::CopyBuffer`]) is
    /// the **GPU's**, and the two differing is the whole answer.
    fn host_buffer(&mut self, spec: &crate::host_pages::HostBufferSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("newBufferWithBytesNoCopy"))
    }

    /// A 2D texture of `format`, `size` texels, one level, shader-read and shader-write, of shared
    /// storage, whose level 0 is then `bytes` at `row_pitch` bytes a row
    /// (`replaceRegion:mipmapLevel:withBytes:bytesPerRow:`).
    ///
    /// Every case states its own row pitch, because that pitch is part of what the case means and
    /// a side that computed its own could disagree with the other silently.
    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle>;

    /// `getBytes:bytesPerRow:fromRegion:mipmapLevel:` of a texture's whole level 0, at `row_pitch`
    /// bytes a row.
    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>>;

    /// A 2D texture the case describes whole ([`TextureSpec`]): one level, of shared storage, in
    /// the usage its role names, whose level 0 is then `bytes` at the spec's row pitch.
    ///
    /// [`Side::texture_2d`] is the read-and-write case of this; a case that samples a texture, or
    /// writes one of a format Metal holds no readable-and-writable texture of, states its role
    /// here instead. An empty `bytes` makes the texture and writes nothing into it, which is what a
    /// case that asks what a **fresh** texture reads needs.
    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        let _ = (spec, bytes);
        Err(Failure::unavailable("texture_2d_in_role"))
    }

    /// [`Side::read_texture`] into a buffer that already holds `before`, so the bytes the copy does
    /// **not** write stay visible in the answer.
    ///
    /// A case reading a texture at a row pitch wider than its rows asks two questions at once:
    /// what the rows hold, and whether the copy left the bytes between them alone. A read into
    /// zeroed memory can only answer the first — zero is indistinguishable from a copy that wrote
    /// zero — so the case states the bytes that were there and the answer carries them back. The
    /// bytes outside the rows then compare under [`Gaps::Compared`](crate::transcript::Gaps).
    ///
    /// `before` is the whole destination, `row_pitch * size[1]` bytes.
    fn read_texture_over(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        before: &[u8],
    ) -> Answer<Vec<u8>> {
        let _ = (texture, size, row_pitch, before);
        Err(Failure::unavailable("read_texture_over"))
    }

    /// `newRenderPipelineStateWithDescriptor:options:reflection:error:` of `library`'s `vertex` and
    /// `fragment` functions, over one color attachment of `attachment`, asking for binding info.
    ///
    /// The reflection is asked for at the creation and kept with the state, because that is the
    /// only time Metal answers it: `MTLRenderPipelineState`'s own `reflection` property is nil for
    /// a state an `MTLDevice` made.
    fn render_pipeline_with_reflection(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
    ) -> Answer<Handle> {
        let _ = (library, vertex, fragment, attachment);
        Err(Failure::unavailable("render_pipeline_with_reflection"))
    }

    /// How the state's function of `stage` accesses the texture at each of `indices`: the access
    /// where a used texture argument covers the slot, and `None` where none does or where the
    /// argument that covers it is never referenced.
    ///
    /// `stage` is one stage, not a union. Metal states this per stage — the reflection holds a
    /// binding list a stage — so one bit has a Metal answer and a union has only a composition
    /// rule of ours, which belongs in a lane rather than in a transcript. A side answers
    /// [`Failure::unavailable`] for anything but a single stage.
    fn texture_accesses(
        &mut self,
        pipeline: Handle,
        stage: RenderStages,
        indices: &[u64],
    ) -> Answer<Vec<Option<BindingAccess>>> {
        let _ = (pipeline, stage, indices);
        Err(Failure::unavailable("texture_accesses"))
    }

    /// `replaceRegion:mipmapLevel:withBytes:bytesPerRow:` of a texture's whole level 0, at
    /// `row_pitch` bytes a row.
    ///
    /// A case that writes a texture twice in a stated order needs this: making it with its bytes
    /// cannot say "the kernel wrote the alpha plane first, then the host wrote the colour words",
    /// and that order is part of what such a case means.
    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        let _ = (texture, size, row_pitch, bytes);
        Err(Failure::unavailable("replace_region"))
    }

    /// `newTextureWithDescriptor:` of the texture the case describes whole
    /// ([`crate::texture::TextureRequest`]), loaded with the request's `replaceRegion:` calls in
    /// order.
    ///
    /// [`Side::texture_2d_in_role`] is the one-level shared 2D case of this. A case that needs a
    /// level count, an array's slices, a cube's faces, private storage,
    /// `allowGPUOptimizedContents`, a swizzle, or a block-compressed format whose rows are blocks,
    /// states the whole descriptor here instead.
    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        let _ = request;
        Err(Failure::unavailable("texture"))
    }

    /// `newSamplerStateWithDescriptor:` of the state the case describes ([`SamplerSpec`]).
    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("sampler"))
    }

    /// A buffer of the GPU resource IDs of `slots`, in order: the argument buffer a kernel reads
    /// its texture and its sampler through, as Metal 3 writes one.
    ///
    /// The case states the slots and nothing about residency: a side that binds this buffer makes
    /// the resources it names resident for the dispatch (`useResource:usage:`), because Metal
    /// requires that of a resource a kernel reaches through an argument buffer rather than through
    /// a slot. Binding it is binding a buffer, so it goes in [`Bindings::buffers`] at the index the
    /// kernel declares.
    ///
    /// A sampler a case puts here must be one [`SamplerSpec::support_argument_buffers`] is set on:
    /// Metal has no resource ID for a sampler state made without it.
    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let _ = slots;
        Err(Failure::unavailable("argument_buffer"))
    }

    /// `newFunctionWithName:` of `entry`, then `newArgumentEncoderWithBufferIndex:` of
    /// `buffer_index`: the encoder of the argument buffer that function takes at that buffer
    /// index, as its own AIR lays it out.
    ///
    /// The encoder has its own handle space: an `MTLArgumentEncoder` is its own Metal object, not
    /// another name for a library, a pipeline or a buffer. No argument buffer is set on it yet, so
    /// a case states that step itself ([`Side::set_argument_buffer`]).
    fn argument_encoder(
        &mut self,
        library: Handle,
        entry: &str,
        buffer_index: u32,
    ) -> Answer<Handle> {
        let _ = (library, entry, buffer_index);
        Err(Failure::unavailable("argument_encoder"))
    }

    /// `newArgumentEncoderWithArguments:` of the descriptors `arguments` states: the encoder of a
    /// layout a program describes itself rather than taking from a function.
    ///
    /// It joins the same handle space as [`Side::argument_encoder`]: one encoder kind, made from
    /// two descriptions.
    fn argument_encoder_with_arguments(
        &mut self,
        arguments: &[crate::argument_encoders::ArgumentDescriptorSpec],
    ) -> Answer<Handle> {
        let _ = arguments;
        Err(Failure::unavailable("argument_encoder_with_arguments"))
    }

    /// `setLabel:` of `label`, then what the encoder answers of itself
    /// ([`EncoderFacts`](crate::argument_encoders::EncoderFacts)): `label`, `encodedLength`,
    /// `alignment` and whether `device` is the case's own.
    ///
    /// The label is set in the same call as the getters are read, because what a case means by it
    /// is a round trip: a label Metal keeps is a label Metal answers.
    fn argument_encoder_facts(
        &mut self,
        encoder: Handle,
        label: &str,
    ) -> Answer<crate::argument_encoders::EncoderFacts> {
        let _ = (encoder, label);
        Err(Failure::unavailable("argument_encoder_facts"))
    }

    /// `setArgumentBuffer:offset:` of `buffer` at `offset`, or
    /// `setArgumentBuffer:startOffset:arrayElement:` where `element` names one, which puts the
    /// arguments at `offset + element * encodedLength`.
    ///
    /// One call for the two selectors, because they are the same step of a case: where in a buffer
    /// this encoder's arguments go.
    fn set_argument_buffer(
        &mut self,
        encoder: Handle,
        buffer: Handle,
        offset: u64,
        element: Option<u64>,
    ) -> Answer<()> {
        let _ = (encoder, buffer, offset, element);
        Err(Failure::unavailable("set_argument_buffer"))
    }

    /// `newArgumentEncoderForBufferAtIndex:` of `index`: the encoder of the argument buffer the
    /// buffer argument at that index points at, in the same handle space as
    /// [`Side::argument_encoder`].
    ///
    /// `Ok(None)` is Metal's **nil**, which is what it answers for every index that names no
    /// nested argument buffer — a plain pointer, a pointer to a struct of constants, an inline
    /// struct of resources, a texture, an array, an index past the layout. Measured on an Apple
    /// M3 Max under macOS 27.0.1 (`probes/argument_descriptor_types.m`): nil for each of those and
    /// an encoder only for a pointer whose pointee holds a resource. So the answer is an
    /// `Option` and not a handle: a side that could only answer a handle or an error would have to
    /// call Metal's nil a refusal, and nil is Metal's answer rather than its refusal.
    fn nested_argument_encoder(&mut self, encoder: Handle, index: u32) -> Answer<Option<Handle>> {
        let _ = (encoder, index);
        Err(Failure::unavailable("nested_argument_encoder"))
    }

    /// One write through an argument encoder
    /// ([`ArgumentWrite`](crate::argument_encoders::ArgumentWrite)).
    ///
    /// A side that binds the encoded buffer for a dispatch makes the textures the write names
    /// resident (`useResource:usage:`), because Metal requires that of a resource a function
    /// reaches through an argument buffer rather than through a slot.
    fn encode_argument(
        &mut self,
        encoder: Handle,
        write: &crate::argument_encoders::ArgumentWrite,
    ) -> Answer<()> {
        let _ = (encoder, write);
        Err(Failure::unavailable("encode_argument"))
    }

    /// `newComputePipelineStateWithFunction:options:reflection:error:` of the library's `entry`,
    /// asking for binding information, and every argument the reflection answers
    /// ([`crate::function_bindings`]).
    ///
    /// One call, because what it observes is a **tree**: each binding, the subprotocol its own
    /// `type` says it conforms to, and for an argument buffer the struct type whose members are the
    /// layout contract. A case cannot state that as independent calls, because which getter to ask
    /// of a binding is decided by what an earlier getter answered.
    fn compute_pipeline_bindings(
        &mut self,
        library: Handle,
        entry: &str,
    ) -> Answer<Vec<crate::function_bindings::BindingFacts>> {
        let _ = (library, entry);
        Err(Failure::unavailable("compute_pipeline_bindings"))
    }

    /// Run an archive-and-dynamic-library program ([`crate::archives::Program`]), answering one
    /// observation a step in the step order.
    ///
    /// One call for a whole program rather than one per container call, because neither
    /// `MTLBinaryArchive` nor `MTLDynamicLibrary` answers anything a single call can be compared
    /// at: what either one is *for* is a sequence — make it, add to it, write it, load it back,
    /// create from it — and the objects between the steps are the side's own, which no handle in
    /// this vocabulary names. The side keeps them by index, as the program's steps name them.
    ///
    /// A side answers `Err` only where it could not run the program at all (no scratch directory,
    /// no assembler). A call the program made that Metal or the runtime refused is an
    /// **observation**, not an error of this call: that is what makes a refusal one leaf of a
    /// swept table rather than the end of the case.
    fn archives(
        &mut self,
        program: &crate::archives::Program<'_>,
    ) -> Answer<Vec<crate::archives::Observation>> {
        let _ = program;
        Err(Failure::unavailable("archives"))
    }

    /// `newComputePipelineStateWithFunction:options:reflection:error:` under the raw
    /// `MTLPipelineOption` value `options`, answering whether a reflection came back and what it
    /// held ([`OptionOutcome`](crate::pipeline_options::OptionOutcome)).
    ///
    /// The option is a **raw** value because the sweep's subject is the whole of Metal's option
    /// word, the bits it does not define included, and a typed set that spelled only the three
    /// defined bits could not state the rest.
    ///
    /// One call for the creation and the reflection together, for the reason
    /// [`Side::compute_pipeline_bindings`] states: which getter to ask of a binding is decided by
    /// what an earlier getter answered, and whether to ask at all is decided by this creation.
    fn compute_pipeline_under_options(
        &mut self,
        library: Handle,
        entry: &str,
        options: u64,
    ) -> Answer<crate::pipeline_options::OptionOutcome> {
        let _ = (library, entry, options);
        Err(Failure::unavailable("compute_pipeline_under_options"))
    }

    /// `newRenderPipelineStateWithDescriptor:options:reflection:error:` of the pipeline `vertex`,
    /// `fragment` and `attachment` describe, under the raw `MTLPipelineOption` value `options`.
    ///
    /// The vertex and fragment binding lists alone, for the reason
    /// [`crate::cases::pipeline_options`] states: a drawing pipeline's other three stage lists are
    /// empty on both sides by construction, so a site for one would compare an empty list against
    /// an empty list.
    fn render_pipeline_under_options(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
        options: u64,
    ) -> Answer<crate::pipeline_options::OptionOutcome> {
        let _ = (library, vertex, fragment, attachment, options);
        Err(Failure::unavailable("render_pipeline_under_options"))
    }

    /// What an `MTLRenderPipelineFunctionsDescriptor` answers after `spec`'s writes
    /// ([`FunctionsDescriptorFacts`](crate::pipeline_functions::FunctionsDescriptorFacts)).
    ///
    /// The writes and the reads in one call, because the descriptor is mutable and what a case
    /// means is "the descriptor these writes made answers this": a side that kept the object
    /// between calls would be answering about a different descriptor than the one the case
    /// states.
    fn render_pipeline_functions_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::pipeline_functions::FunctionsDescriptorSpec<'_>,
    ) -> Answer<crate::pipeline_functions::FunctionsDescriptorFacts> {
        let _ = (library, spec);
        Err(Failure::unavailable(
            "render_pipeline_functions_descriptor_facts",
        ))
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` of the descriptor
    /// `spec` describes whole ([`ComputeStateSpec`](crate::pipeline_facts::ComputeStateSpec)).
    ///
    /// The state joins the same handle space as [`Side::compute_pipeline`]: one Metal object kind
    /// made from a function or from a descriptor.
    fn compute_pipeline_with_descriptor(
        &mut self,
        library: Handle,
        spec: &crate::pipeline_facts::ComputeStateSpec<'_>,
    ) -> Answer<Handle> {
        let _ = (library, spec);
        Err(Failure::unavailable("compute_pipeline_with_descriptor"))
    }

    /// What a compute pipeline state answers of itself
    /// ([`ComputeStateFacts`](crate::pipeline_facts::ComputeStateFacts)), with
    /// `imageblockMemoryLengthForDimensions:` asked for `imageblock`.
    ///
    /// The dimensions are the case's own rather than fixed, because the answer is a function of
    /// them: a case states the extent whose length it means.
    fn compute_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<crate::pipeline_facts::ComputeStateFacts> {
        let _ = (state, imageblock);
        Err(Failure::unavailable("compute_pipeline_facts"))
    }

    /// What a render pipeline state answers of itself
    /// ([`RenderStateFacts`](crate::pipeline_facts::RenderStateFacts)), with
    /// `imageblockMemoryLengthForDimensions:` asked for `imageblock`.
    ///
    /// [`Side::tile_facts`] is three of these getters read off a **tile** state, which is the
    /// other descriptor the same Metal object is made from. This is the drawing state's whole
    /// portable surface, and the two are kept apart because what each answers is decided by which
    /// descriptor made it.
    fn render_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<crate::pipeline_facts::RenderStateFacts> {
        let _ = (state, imageblock);
        Err(Failure::unavailable("render_pipeline_facts"))
    }

    /// What a render pipeline **descriptor** answers of itself
    /// ([`RenderDescriptorFacts`](crate::render_descriptors::RenderDescriptorFacts)): the whole of
    /// `MTLRenderPipelineDescriptor`'s own surface, in the shape
    /// [`spec`](crate::render_descriptors::DescriptorSpec) states.
    ///
    /// One call, for the reason [`Side::function_facts`] gives: the descriptor is mutable and the
    /// observation is what it answers **after** a stated sequence of setters, a `reset` among them,
    /// so a case cannot state it as independent calls over an object neither side's handle space
    /// holds. No pipeline is made from the descriptor, which is what keeps the family off the GPU.
    ///
    /// [`Side::render_pipeline_facts`] is the **state**'s side of the same descriptor: what the
    /// made object echoes back. This is the descriptor before any state exists, which is where
    /// Metal's defaults are.
    fn render_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::render_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::render_descriptors::RenderDescriptorFacts> {
        let _ = (library, spec);
        Err(Failure::unavailable("render_descriptor_facts"))
    }

    /// What a compute pipeline **descriptor** answers of itself
    /// ([`ComputeDescriptorFacts`](crate::compute_descriptors::ComputeDescriptorFacts)): the whole
    /// of `MTLComputePipelineDescriptor`'s own surface, in the shape
    /// [`spec`](crate::compute_descriptors::DescriptorSpec) states.
    ///
    /// One call, for the reason [`Side::render_descriptor_facts`] gives: the descriptor is mutable
    /// and the observation is what it answers **after** a stated sequence of setters, a `reset`
    /// among them, so a case cannot state it as independent calls over an object neither side's
    /// handle space holds. No pipeline is made from the descriptor, which is what keeps the family
    /// off the GPU.
    fn compute_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::compute_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::compute_descriptors::ComputeDescriptorFacts> {
        let _ = (library, spec);
        Err(Failure::unavailable("compute_descriptor_facts"))
    }

    /// What a tile render pipeline **descriptor** answers of itself
    /// ([`TileDescriptorFacts`](crate::tile_descriptors::TileDescriptorFacts)): the whole of
    /// `MTLTileRenderPipelineDescriptor`'s own surface, in the shape
    /// [`spec`](crate::tile_descriptors::DescriptorSpec) states.
    ///
    /// [`Side::tile_pipeline`] is the **state**'s side of the same descriptor: what a made tile
    /// pipeline then answers. This is the descriptor before any state exists, which is where
    /// Metal's defaults are.
    fn tile_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::tile_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::tile_descriptors::TileDescriptorFacts> {
        let _ = (library, spec);
        Err(Failure::unavailable("tile_descriptor_facts"))
    }

    /// What a mesh render pipeline **descriptor** answers of itself
    /// ([`MeshDescriptorFacts`](crate::mesh_descriptors::MeshDescriptorFacts)): the whole of
    /// `MTLMeshRenderPipelineDescriptor`'s implemented surface, in the shape
    /// [`spec`](crate::mesh_descriptors::DescriptorSpec) states.
    fn mesh_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::mesh_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::mesh_descriptors::MeshDescriptorFacts> {
        let _ = (library, spec);
        Err(Failure::unavailable("mesh_descriptor_facts"))
    }

    /// What a **render pass descriptor** answers of itself
    /// ([`PassDescriptorFacts`](crate::pass_descriptors::PassDescriptorFacts)):
    /// `MTLRenderPassDescriptor`, its three attachment descriptors and its colour-attachment
    /// array, in the shape [`spec`](crate::pass_descriptors::PassSpec) states.
    ///
    /// One call, for the reason [`Side::render_descriptor_facts`] gives: the descriptor is mutable
    /// and the observation is what it answers **after** a stated sequence of setters, so a case
    /// cannot state it as independent calls over objects neither side's handle space holds. No
    /// encoder is made from the descriptor and nothing is allocated, which is what keeps the family
    /// off the GPU.
    ///
    /// The runtime has no descriptor object: its `RenderPassDescriptor` is a value a consumer
    /// fills, so Metal's `renderPassDescriptor` is `RenderPassDescriptor::new` and an attachment
    /// set to nil is its `Default` again.
    fn pass_descriptor_facts(
        &mut self,
        spec: &crate::pass_descriptors::PassSpec,
    ) -> Answer<crate::pass_descriptors::PassDescriptorFacts> {
        let _ = spec;
        Err(Failure::unavailable("pass_descriptor_facts"))
    }

    /// `newFunctionWithName:` of `entry`, or `newFunctionWithName:constantValues:error:` where
    /// `constants` holds a value, and what the function then answers about **itself**
    /// ([`crate::function_facts::FunctionFacts`]).
    ///
    /// One call, because what it observes is a **tree** whose shape one getter decides: which of
    /// `vertexAttributes` and `stageInputAttributes` answers an array at all is a fact about the
    /// function type, and each element is an object with six getters of its own. A case cannot
    /// state that as independent calls.
    ///
    /// The function is made and read inside the call rather than kept in a pool of its own: every
    /// fact here is the function's, no case holds one past reading it, and the two sides then need
    /// no handle space for an object whose identity nothing compares.
    ///
    /// `constants` is the whole of `MTLFunctionConstantValues`' own surface, as three setters
    /// ([`ConstantAt`](crate::function_facts::ConstantAt)). A side whose constant values have no
    /// range form expands the range into one value an index, which is what the range setter means.
    fn function_facts(
        &mut self,
        library: Handle,
        entry: &str,
        constants: &[crate::function_facts::ConstantValue],
    ) -> Answer<crate::function_facts::FunctionFacts> {
        let _ = (library, entry, constants);
        Err(Failure::unavailable("function_facts"))
    }

    /// What the library answers about **itself**
    /// ([`crate::library_facts::LibraryFacts`]): `functionNames`, `type`, `installName`, and
    /// whether `device` is the case's own.
    ///
    /// One call for the four getters, because a library's own facts are read together and none of
    /// them takes an argument: the call is "ask this library about itself". No function is made
    /// and nothing is compiled, so the answer is the container's reading alone.
    fn library_facts(&mut self, library: Handle) -> Answer<crate::library_facts::LibraryFacts> {
        let _ = library;
        Err(Failure::unavailable("library_facts"))
    }

    /// `newFunctionWithDescriptor:error:` of the descriptor `descriptor` states, and what the
    /// function then answers about itself ([`crate::function_facts::FunctionFacts`]).
    ///
    /// The same answer [`Side::function_facts`] reads, from the other way of making a function, so
    /// the two are directly comparable: a descriptor naming a function with no constant values and
    /// no options must give the facts `newFunctionWithName:` gives, and a descriptor's `options`
    /// must come back out of the function it made.
    ///
    /// The descriptor is made, set and spent inside the call. Nothing compares its identity and
    /// no case holds one past the function it makes, so neither side needs a handle space for it.
    fn function_facts_from_descriptor(
        &mut self,
        library: Handle,
        descriptor: &crate::library_facts::FunctionDescriptorSpec,
    ) -> Answer<crate::function_facts::FunctionFacts> {
        let _ = (library, descriptor);
        Err(Failure::unavailable("function_facts_from_descriptor"))
    }

    /// Where `constantDataAtIndex:` of `index` points, as a byte offset inside the encoder's
    /// arguments.
    ///
    /// The call answers a CPU address, which is the host's own number and belongs in no
    /// recording. Its offset from the arguments' start is the layout fact the address stands for,
    /// and it is exactly comparable.
    fn constant_data_offset(&mut self, encoder: Handle, index: u32) -> Answer<u64> {
        let _ = (encoder, index);
        Err(Failure::unavailable("constant_data_offset"))
    }

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up per
    /// kind, one `dispatchThreadgroups:threadsPerThreadgroup:`, and `commit`.
    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle>;

    /// `waitUntilCompleted` and then `status`, as a variant of the status's name.
    fn wait(&mut self, command_buffer: Handle) -> Answer<Value>;

    /// `contents` of a shared buffer: `length` bytes from `offset`.
    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>>;

    /// The degrade notices the side has announced so far, by code, and taken away. Metal announces
    /// none, so its side answers an empty list.
    fn announced(&mut self) -> Vec<String>;

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` of the library's `entry`,
    /// under the stage-input descriptor `stage_input` ([`StageInput`]).
    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle>;

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up per
    /// kind, each of `stage_in` set in the order the case lists them, one
    /// `dispatchThreads:threadsPerThreadgroup:` of `grid` threads, and `commit`.
    ///
    /// `stage_in` is a list rather than one region because setting it twice is a case of its own:
    /// Metal's later call replaces the earlier, and a case states that by listing both.
    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle>;

    /// **One** command buffer with **one** compute encoder holding every command of `pass` in
    /// order, then `endEncoding`, `commit` and a bounded wait
    /// ([`ComputePass`](crate::compute_pass::ComputePass)).
    ///
    /// [`dispatch`](Side::dispatch) and [`dispatch_threads`](Side::dispatch_threads) are a fixed
    /// pass — pipeline, bindings at 0 up, one dispatch, commit — and four of
    /// `MTLComputeCommandEncoder`'s items are reachable that way. A case that states its commands
    /// reaches the rest: the inline bytes, the plural setters, the LOD-clamped sampler bindings,
    /// the dynamic threadgroup allocation, the residency declarations, the barriers, the
    /// threadgroup-grid dispatch and the indirect one.
    ///
    /// Every command is encoded whatever an earlier one answered, as
    /// [`mesh_pass`](Side::mesh_pass) and [`tile_pass`](Side::tile_pass) do, and the answer names
    /// which one refused. **The side makes nothing resident of its own**: a case that reaches a
    /// resource other than through a bound slot says so with a residency command, which is both
    /// what Metal requires and how those items are exercised.
    fn compute_pass(
        &mut self,
        pass: &crate::compute_pass::ComputePass<'_>,
    ) -> Answer<crate::compute_pass::ComputePassOutcome> {
        let _ = pass;
        Err(Failure::unavailable("compute_pass"))
    }

    /// The read-back of an indirect render command the CPU encoded, as the side's getters answer
    /// it: `getCommandType` and the draw. Metal exposes these privately only, so the Metal side
    /// answers [`Failure::unavailable`] and such a case's Metal answer is carried.
    ///
    /// `primitive_type` is the raw `MTLPrimitiveType` value, so a case can set one of Metal's
    /// private types; `indexed` chooses the indexed setter.
    fn indirect_draw_read_back(
        &mut self,
        primitive_type: u64,
        indexed: bool,
    ) -> Answer<IndirectReadBack> {
        let _ = (primitive_type, indexed);
        Err(Failure::unavailable("indirect_draw_read_back"))
    }

    /// Run a copy-and-fill program ([`crate::work`]) and answer what it read.
    ///
    /// The Metal side runs a program through public `objc2-metal` calls and refuses the two fills
    /// alone, because `fillTexture:` and `fillTextureColor:` are Apple SPI. The pixel format a
    /// case names is never the reason: it is a raw value, and Metal handles Apple's two-plane
    /// extended-range formats publicly. So a program that fills carries its Metal answer; a
    /// program of copies records on Metal. A side that runs no program at all answers
    /// [`Failure::unavailable`], which is this default.
    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        let _ = work;
        Err(Failure::unavailable("work"))
    }

    /// Run one full-screen fragment fill ([`FragmentFill`]) and answer the pass's status and the
    /// target's bytes.
    ///
    /// Metal's whole path here is public, so both sides run it: `newRenderPipelineStateWithDescriptor:
    /// error:` of the two functions over one color attachment of the case's format, a private
    /// render target of the case's extent, one render pass that loads nothing and stores, the
    /// fragment buffers set at 0 and up, one `drawPrimitives:`, then a blit of level 0 into a
    /// shared buffer the host reads.
    fn fragment_fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let _ = fill;
        Err(Failure::unavailable("fragment_fill"))
    }

    /// `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:` of the library's
    /// functions, over `functions.attachments` color attachments of [`TARGET_FORMAT`].
    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        let _ = (library, functions);
        Err(Failure::unavailable("mesh_pipeline"))
    }

    /// `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:` of the pipeline `spec`
    /// describes ([`MeshReflectionSpec`]), and **all five** stage lists the reflection answers
    /// ([`StageBindings`](crate::function_bindings::StageBindings)).
    ///
    /// The creation and the reading in one call, and the state kept by neither side, for the
    /// reason [`Side::tile_pipeline_bindings`] states: Metal answers a reflection only at the
    /// creation.
    ///
    /// This is the only call of either side that can answer `MTLBindingTypeObjectPayload`, and so
    /// the only one whose answer can hold `MTLObjectPayloadBinding`'s two fields
    /// ([`PayloadFacts`](crate::function_bindings::PayloadFacts)): a payload is an argument of an
    /// object or a mesh function and of nothing else.
    /// `options` is the raw `MTLPipelineOption` word and a refusal is an `Ok`, for the reasons
    /// [`Side::tile_pipeline_bindings`] states.
    fn mesh_pipeline_bindings(
        &mut self,
        library: Handle,
        spec: &MeshReflectionSpec<'_>,
        options: u64,
    ) -> Answer<crate::function_bindings::StageReflection> {
        let _ = (library, spec, options);
        Err(Failure::unavailable("mesh_pipeline_bindings"))
    }

    /// `newTextureWithDescriptor:` of a 2D [`TARGET_FORMAT`] render target of `width` by
    /// `height`, whose bytes the host reads.
    ///
    /// The target joins the same handle space as [`texture_2d`](Side::texture_2d), so
    /// [`read_texture`](Side::read_texture) reads it back and a texture has one pool however it
    /// was made. Only the descriptor differs: a target is a render target of
    /// [`TARGET_FORMAT`], not a shader-read-and-write texture of the case's format.
    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        let _ = (width, height);
        Err(Failure::unavailable("render_target"))
    }

    /// One command buffer with one render pass over `targets`, each cleared to zero and stored,
    /// whose commands are `steps` in order, then `endEncoding`, `commit` and a bounded wait.
    ///
    /// Every step is encoded, whatever an earlier one answered, because that is what the probe
    /// does: a refused draw leaves the pass and its other commands as they were, and the answer
    /// names which step refused ([`PassOutcome::refused`]).
    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        let _ = (targets, steps);
        Err(Failure::unavailable("mesh_pass"))
    }

    /// `newRenderPipelineStateWithMeshDescriptor:` as [`mesh_pipeline`] does, with
    /// `supportIndirectCommandBuffers` set, so an indirect command may set the state.
    ///
    /// [`mesh_pipeline`]: Side::mesh_pipeline
    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        let _ = (library, functions);
        Err(Failure::unavailable("mesh_pipeline_for_commands"))
    }

    /// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` of `commands` commands
    /// of both mesh draw kinds, binding its own pipeline state, one object buffer and one mesh
    /// buffer, then every command of `commands` encoded by the host in order.
    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        let _ = commands;
        Err(Failure::unavailable("mesh_command_buffer"))
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` of the library's `entry`
    /// with `supportIndirectCommandBuffers` set, so an indirect compute command may set the
    /// state, under the stage-input descriptor `stage_input` where the case names one.
    ///
    /// It is a call of its own rather than a flag on
    /// [`compute_pipeline`](Side::compute_pipeline), because the flag is a property of the state
    /// Metal checks at the setter rather than at the dispatch: a state made without it refuses to
    /// be set on a command, and a case that executes commands means the state it names to be one
    /// a command may hold.
    fn compute_pipeline_for_commands(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: Option<&StageInput>,
    ) -> Answer<Handle> {
        let _ = (library, entry, stage_input);
        Err(Failure::unavailable("compute_pipeline_for_commands"))
    }

    /// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` of the descriptor the
    /// case states whole ([`IcbDescriptor`]), of shared storage and of as many commands as
    /// `commands` holds, and then every command of `commands` encoded by the host in order
    /// ([`EncodedCommand`]).
    ///
    /// Every step of every command is encoded, whatever an earlier one answered, as the pass
    /// vocabularies do: a refused setter leaves the rest of the encoding as it was, and the answer
    /// names which step refused ([`CommandsMade::refused`]).
    ///
    /// [`IcbDescriptor`]: crate::indirect_commands::IcbDescriptor
    /// [`EncodedCommand`]: crate::indirect_commands::EncodedCommand
    /// [`CommandsMade::refused`]: crate::indirect_commands::CommandsMade::refused
    fn indirect_command_buffer(
        &mut self,
        descriptor: &crate::indirect_commands::IcbDescriptor,
        commands: &[crate::indirect_commands::EncodedCommand],
    ) -> Answer<crate::indirect_commands::CommandsMade> {
        let _ = (descriptor, commands);
        Err(Failure::unavailable("indirect_command_buffer"))
    }

    /// `size` of an indirect command buffer: the commands it holds.
    fn indirect_commands_size(&mut self, commands: Handle) -> Answer<u64> {
        let _ = commands;
        Err(Failure::unavailable("indirect_commands_size"))
    }

    /// `resetWithRange:` of `count` commands from `first`, which puts them back as a fresh
    /// buffer's are.
    fn reset_indirect_commands(&mut self, commands: Handle, first: u64, count: u64) -> Answer<()> {
        let _ = (commands, first, count);
        Err(Failure::unavailable("reset_indirect_commands"))
    }

    /// One command buffer with one render pass over `targets`, each cleared to zero and stored,
    /// which makes `used` resident for the object, mesh and fragment stages, executes every
    /// command of `commands`, then `endEncoding`, `commit` and a bounded wait.
    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        let _ = (targets, commands, used);
        Err(Failure::unavailable("execute_mesh_commands"))
    }

    /// A texture over `buffer`: over one plane of the surface the buffer stands for, or over the
    /// buffer itself ([`PlaneTexture`]).
    ///
    /// The buffer stands for the surface with its planes where the surface's lie, which is how the
    /// probe wrote these cases: a host without CoreVideo then runs them unchanged, and the bytes a
    /// fill wrote are the buffer's own.
    fn plane_texture(&mut self, buffer: Handle, over: &PlaneTexture) -> Answer<Handle> {
        let _ = (buffer, over);
        Err(Failure::unavailable("plane_texture"))
    }

    /// One command buffer with one blit encoder holding one fill of `texture` ([`Fill`]),
    /// committed and waited for.
    ///
    /// Metal's texture fills are private selectors, so Metal's own side answers
    /// [`Failure::unavailable`] here and a case that uses this carries its Metal answer.
    fn fill_texture(&mut self, texture: Handle, fill: &Fill) -> Answer<Value> {
        let _ = (texture, fill);
        Err(Failure::unavailable("fill_texture"))
    }

    /// `newRasterizationRateMapWithDescriptor:` of `request`.
    ///
    /// Nothing about a rate map reaches the GPU until a render pass takes one, and no case here
    /// opens such a pass, so the whole family is a CPU-side call: the descriptor is compiled, the
    /// map answers its queries, and its parameter data is copied by the CPU.
    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        let _ = request;
        Err(Failure::unavailable("rate_map"))
    }

    /// What the map answers: its label, screen size, granularity, layer count, parameter size and
    /// alignment, and `physicalSizeForLayer:` of each index in `layers`.
    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        let _ = (map, layers);
        Err(Failure::unavailable("rate_map_facts"))
    }

    /// `copyParameterDataToBuffer:offset:` into the buffer `into` at `offset`. The bytes are read
    /// back with [`Side::read_buffer`], so the copy itself observes nothing of its own.
    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        let _ = (map, into, offset);
        Err(Failure::unavailable("copy_rate_map_parameters"))
    }

    /// `mapScreenToPhysicalCoordinates:forLayer:` and `mapPhysicalToScreenCoordinates:forLayer:`
    /// at each of `probes`, in order.
    ///
    /// Both are CPU arithmetic over the map's own parameter data: no command buffer is made and no
    /// queue is reached. **Two shapes end Metal's process** at the screen-to-physical call and are
    /// a case's to avoid, never this method's to guess at: a map of no layers, and a map of a zero
    /// screen extent (both measured, SIGSEGV with no validation layer). A probe's layer index is
    /// the case's to keep inside the map's `layerCount`, for the reason
    /// [`crate::cases::rate_maps`] gives of `physicalSizeForLayer:` — past its own layer table
    /// Metal reads whatever lies there.
    fn rate_map_coordinates(
        &mut self,
        map: Handle,
        probes: &[CoordinateProbe],
    ) -> Answer<Vec<MappedCoordinates>> {
        let _ = (map, probes);
        Err(Failure::unavailable("rate_map_coordinates"))
    }

    /// Whether the map's `device` is the one this side made it on.
    ///
    /// The identity of a device is not a value a transcript can hold — a pointer is this process's
    /// and a name is the host's — so the comparable fact is whether the map points back at the
    /// device that compiled it, which is the whole of what `device` promises.
    fn rate_map_device_is_the_sides(&mut self, map: Handle) -> Answer<bool> {
        let _ = map;
        Err(Failure::unavailable("rate_map_device_is_the_sides"))
    }

    /// `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` of the library's tile
    /// function, over the color attachments `spec` names ([`TilePipelineSpec`]).
    ///
    /// The state joins the same handle space as [`mesh_pipeline`](Side::mesh_pipeline): a tile
    /// state and a mesh state are one Metal object kind (`MTLRenderPipelineState`), made from two
    /// descriptors.
    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        let _ = (library, spec);
        Err(Failure::unavailable("tile_pipeline"))
    }

    /// `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` of the pipeline `spec`
    /// describes, asking for binding information, and **all five** stage lists the reflection
    /// answers ([`StageBindings`](crate::function_bindings::StageBindings)).
    ///
    /// [`Side::tile_pipeline`] makes the same pipeline and keeps it by handle; this one asks the
    /// creation for its reflection and keeps nothing, because Metal answers a reflection **only**
    /// at the creation — `MTLRenderPipelineState` has no getter for one — which is the reason
    /// [`Side::compute_pipeline_bindings`] states for the compute half.
    ///
    /// The whole reflection rather than `tileBindings` alone, for the reason
    /// [`StageBindings`](crate::function_bindings::StageBindings) states: which stage Metal
    /// attributes a tile function's arguments to is the fact, and reading one list could not see
    /// a side that answered them under another's name.
    ///
    /// `options` is the **raw** `MTLPipelineOption` word, so a case states Metal's own value
    /// rather than a vocabulary of modes, which is the choice
    /// [`Side::compute_pipeline_reflection_under`] already makes for the compute creation. A
    /// refused creation is an `Ok` whose answer says it was refused
    /// ([`StageReflection::refused`](crate::function_bindings::StageReflection::refused)), not an
    /// `Err`: what an options value does is the question, and "this value is refused" is one of
    /// the answers the case compares.
    fn tile_pipeline_bindings(
        &mut self,
        library: Handle,
        spec: &TilePipelineSpec<'_>,
        options: u64,
    ) -> Answer<crate::function_bindings::StageReflection> {
        let _ = (library, spec, options);
        Err(Failure::unavailable("tile_pipeline_bindings"))
    }

    /// What a tile render pipeline state answers ([`TileFacts`]).
    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        let _ = state;
        Err(Failure::unavailable("tile_facts"))
    }

    /// `newComputePipelineStateWithFunction:options:reflection:error:` of the library's `entry`
    /// under the raw `MTLPipelineOption` bits `options`, and the whole reflection it answered
    /// ([`crate::pipeline_option_reflection`]).
    ///
    /// One call for the creation and the reading, because Metal answers a reflection **only** at
    /// the creation: there is no getter to ask the state for one afterwards. The options are the
    /// raw bits so the case states Metal's own value rather than a vocabulary of modes, which is
    /// the choice [`crate::device_queries`]' heap query already makes for `MTLResourceOptions`.
    ///
    /// A refused creation is an `Ok` whose answer says it was refused
    /// ([`UnderOptions::refused`](crate::pipeline_option_reflection::UnderOptions::refused)), not an `Err`:
    /// the case is asking what each options value does, and "this value is refused" is one of the
    /// answers it compares rather than a call the side could not make.
    fn compute_pipeline_reflection_under(
        &mut self,
        library: Handle,
        entry: &str,
        options: u64,
    ) -> Answer<crate::pipeline_option_reflection::UnderOptions> {
        let _ = (library, entry, options);
        Err(Failure::unavailable("compute_pipeline_reflection_under"))
    }

    /// One pipeline descriptor made **twice**: once through Metal's `completionHandler:` form of
    /// the creation `form` names, once through that form's synchronous sibling
    /// ([`crate::pipeline_handlers`]).
    ///
    /// One call for the pair, because what the case means is a comparison **between** the two
    /// paths of one side over one descriptor, which a case cannot state as two independent calls:
    /// the descriptor is built inside the call so that both paths are given the same value, and
    /// [`BothPaths::agree`](crate::pipeline_handlers::BothPaths::agree) is derived from the two
    /// deliveries rather than asserted by the side.
    ///
    /// Neither state is kept by handle. Every fact the case reads is read inside the call, no case
    /// holds either state afterwards, and nothing compares a pipeline's identity — so the two
    /// sides need no handle space for an object the family makes and spends.
    ///
    /// The handler is waited for on **its own channel with a bound**
    /// ([`WAIT`](crate::pipeline_handlers::WAIT)); a bound that runs out is recorded as
    /// [`Arrival::NeverArrived`](crate::pipeline_handlers::Arrival::NeverArrived) and never
    /// retried.
    fn pipeline_both_paths(
        &mut self,
        library: Handle,
        form: &crate::pipeline_handlers::Form<'_>,
    ) -> Answer<crate::pipeline_handlers::BothPaths> {
        let _ = (library, form);
        Err(Failure::unavailable("pipeline_both_paths"))
    }

    /// One command buffer with one render pass over `targets` in `tile`-sized tiles, each
    /// attachment **loaded and stored**, whose commands are `steps` in order, then `endEncoding`,
    /// `commit` and a bounded wait.
    ///
    /// Loaded, not cleared, because a tile function reads the cells it writes: a case states what
    /// the attachment held by writing it before the pass, and the pass's answer is what the tile
    /// function left there.
    ///
    /// Every step is encoded, whatever an earlier one answered, as [`Side::mesh_pass`] does, and
    /// the answer names which step refused ([`PassOutcome::refused`]).
    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        let _ = (targets, tile, steps);
        Err(Failure::unavailable("tile_pass"))
    }

    /// `heapTextureSizeAndAlignWithDescriptor:` of the descriptor `query` names
    /// ([`crate::texture_query`]).
    ///
    /// A pure getter: it compiles nothing, allocates nothing and reaches no GPU, so a case over it
    /// is safe on every device and its answer can be re-measured on the oracle as often as anybody
    /// asks.
    fn heap_texture_size_and_align(
        &mut self,
        query: &crate::texture_query::HeapTextureQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let _ = query;
        Err(Failure::unavailable("heap_texture_size_and_align"))
    }

    /// `heapBufferSizeAndAlignWithLength:options:` of the length and the raw `MTLResourceOptions`
    /// bits `query` names ([`crate::device_queries::HeapBufferQuery`]).
    ///
    /// A pure getter, as [`Side::heap_texture_size_and_align`] is: arithmetic over a length, with
    /// nothing allocated and no heap made.
    fn heap_buffer_size_and_align(
        &mut self,
        query: &crate::device_queries::HeapBufferQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let _ = query;
        Err(Failure::unavailable("heap_buffer_size_and_align"))
    }

    /// `supportsVertexAmplificationCount:` of `count`.
    ///
    /// A device query whose answer is a **model**: the runtime answers yes up to a constant its own
    /// documentation attributes to a Metal probe on this chip and on an M4, so it is the same on
    /// every device that enables multiview and is a contract with Metal rather than a reading of
    /// the device ([`crate::device_queries`] says which queries pass that test and which do not).
    fn supports_vertex_amplification_count(&mut self, count: u64) -> Answer<bool> {
        let _ = count;
        Err(Failure::unavailable("supports_vertex_amplification_count"))
    }

    /// `supportsRasterizationRateMapWithLayerCount:` of `layers`.
    ///
    /// A model as [`Side::supports_vertex_amplification_count`] is: the runtime answers yes for the
    /// layer counts Apple's map compile was measured with and reproduces on every device.
    fn supports_rasterization_rate_map_layers(&mut self, layers: u64) -> Answer<bool> {
        let _ = layers;
        Err(Failure::unavailable(
            "supports_rasterization_rate_map_layers",
        ))
    }

    /// `minimumLinearTextureAlignmentForPixelFormat:` of `format`.
    ///
    /// A model as the two above are: the runtime answers one alignment for every format a buffer
    /// can back, on every device, and refuses the formats Metal's own query asserts on. A case asks
    /// only formats the runtime answers a number for, which is what keeps Metal's assertion out of
    /// reach ([`crate::device_queries`]).
    fn minimum_linear_texture_alignment(
        &mut self,
        format: metal2vulkan::metal::PixelFormat,
    ) -> Answer<u64> {
        let _ = format;
        Err(Failure::unavailable("minimum_linear_texture_alignment"))
    }

    /// `convertSparsePixelRegions:toTileRegions:withTileSize:alignmentMode:numRegions:` of the
    /// regions, tile size and alignment mode `query` names, as the tile boxes it wrote.
    ///
    /// Arithmetic over the case's own numbers: nothing is allocated, no texture is sparse and no
    /// device memory is bound. Metal's one `numRegions` is the slices' single length, so the case
    /// hands one list and reads one back.
    fn convert_sparse_pixel_regions(
        &mut self,
        query: &crate::device_queries::RegionQuery,
    ) -> Answer<Vec<crate::device_queries::Box6>> {
        let _ = query;
        Err(Failure::unavailable("convert_sparse_pixel_regions"))
    }

    /// `convertSparseTileRegions:toPixelRegions:withTileSize:numRegions:` of the regions and tile
    /// size `query` names, as the texel boxes it wrote. The mode is not read: this direction takes
    /// none.
    fn convert_sparse_tile_regions(
        &mut self,
        query: &crate::device_queries::RegionQuery,
    ) -> Answer<Vec<crate::device_queries::Box6>> {
        let _ = query;
        Err(Failure::unavailable("convert_sparse_tile_regions"))
    }

    /// Whether `newTextureWithDescriptor:` of the descriptor `query` names answers a texture.
    ///
    /// The answer is a **fact about the descriptor**, not about the object: Metal answers nil for a
    /// descriptor it describes no texture of, and the runtime answers a typed `Nil` error for the
    /// same descriptor, which is the same fact in the two APIs' spellings. So both sides answer
    /// `false` there, and a side that refuses the call for any other reason answers its typed
    /// error, which compares as a refusal.
    ///
    /// A case asks this only of descriptors Metal makes no texture of, so nothing is allocated.
    /// `newTextureWithDescriptor:` of the descriptor `query` names, and every `MTLTexture` getter
    /// the texture it answers has ([`crate::buffer_texture::BufferTextureFacts`], which is the
    /// same set [`Side::buffer_texture`] reads of a texture over a buffer). `None` where plain
    /// Metal answers nil.
    ///
    /// This is the one call that reads a descriptor's **answered** fields rather than its
    /// footprint, and so the only one that can say what Metal made of the descriptor's own words:
    /// what usage a texture of `MTLTextureUsageUnknown` answers, what a usage of only the bits
    /// Metal drops answers, and what storage mode, level count and array length survive the
    /// crossing a case asked for. A size query can say none of that.
    ///
    /// The texture is kept, as every other texture of a side is, so a case may ask it nothing more
    /// and the handle space stays one pool.
    fn queried_texture_facts(
        &mut self,
        query: &crate::texture_query::HeapTextureQuery,
    ) -> Answer<Option<crate::buffer_texture::BufferTextureFacts>> {
        let _ = query;
        Err(Failure::unavailable("queried_texture_facts"))
    }

    fn makes_texture(&mut self, query: &crate::texture_query::HeapTextureQuery) -> Answer<bool> {
        let _ = query;
        Err(Failure::unavailable("makes_texture"))
    }

    /// `newTextureWithDescriptor:offset:bytesPerRow:` on a buffer of the request's own length and
    /// storage mode, and what the made texture answers of itself
    /// ([`crate::buffer_texture`]).
    ///
    /// The whole shape is one call because what it observes is a **rule**: whether Metal lays a
    /// texture of this descriptor over a buffer at that offset and that row pitch at all. A case
    /// cannot state that as a resource plus a step, because a resource that cannot be made stops a
    /// program, and this family's subject is exactly the shapes that cannot.
    ///
    /// It allocates no texels: the texture's memory **is** the buffer's, and the buffer is the
    /// request's own length. Nothing is encoded and nothing reaches the GPU.
    ///
    /// A request whose row pitch is below the texture's own row ends Metal's process, so the Metal
    /// side refuses it before making the call
    /// ([`crate::buffer_texture::ends_metals_process`]) and a case of that shape says
    /// [`MetalSource::did_not_return`].
    fn buffer_texture(
        &mut self,
        request: &crate::buffer_texture::BufferTextureRequest,
    ) -> Answer<crate::buffer_texture::BufferTextureFacts> {
        let _ = request;
        Err(Failure::unavailable("buffer_texture"))
    }

    /// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:`, or its `sparsePageSize:` form
    /// where the query names a page ([`crate::sparse_query`]).
    ///
    /// A pure getter, like [`heap_texture_size_and_align`](Side::heap_texture_size_and_align): it
    /// compiles nothing, allocates nothing, reaches no GPU and asks nothing of the device's sparse
    /// capabilities, so a case over it is safe on every device and answers on a venue that binds no
    /// sparse memory at all.
    fn sparse_tile_size(
        &mut self,
        query: &crate::sparse_query::SparseTileQuery,
    ) -> Answer<crate::sparse_query::TileSize> {
        let _ = query;
        Err(Failure::unavailable("sparse_tile_size"))
    }

    /// `sparseTileSizeInBytes`, or `sparseTileSizeInBytesForSparsePageSize:` where `page` names
    /// one: the bytes one sparse tile of that page takes.
    ///
    /// A pure getter, as [`sparse_tile_size`](Side::sparse_tile_size) is.
    fn sparse_tile_size_in_bytes(&mut self, page: crate::sparse_query::QueryPage) -> Answer<u64> {
        let _ = page;
        Err(Failure::unavailable("sparse_tile_size_in_bytes"))
    }

    /// `newHeapWithDescriptor:` of a `MTLHeapTypeSparse` heap the case describes
    /// ([`crate::sparse_query::SparseHeapSpec`]).
    ///
    /// The heap holds no memory of its own until a mapping takes a tile, so making one allocates
    /// nothing a GPU can fault on and encodes nothing. A device that cannot make one refuses typed,
    /// which is how a case finds out rather than by reading a device name.
    fn sparse_heap(&mut self, spec: &crate::sparse_query::SparseHeapSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("sparse_heap"))
    }

    /// What a heap answers of itself ([`crate::sparse_query::HeapFacts`]).
    fn heap_facts(&mut self, heap: Handle) -> Answer<crate::sparse_query::HeapFacts> {
        let _ = heap;
        Err(Failure::unavailable("heap_facts"))
    }

    /// `newTextureWithDescriptor:` on a heap, of the texture the case describes
    /// ([`crate::sparse_query::SparseTextureSpec`]).
    ///
    /// The texture joins the same handle space as [`texture_2d`](Side::texture_2d), so a texture
    /// has one pool however it was made. Nothing is written into it: a case that asks what a
    /// **fresh** sparse texture answers needs it untouched, and a case that writes one says so with
    /// a command of its own.
    fn heap_texture(
        &mut self,
        heap: Handle,
        spec: &crate::sparse_query::SparseTextureSpec,
    ) -> Answer<Handle> {
        let _ = (heap, spec);
        Err(Failure::unavailable("heap_texture"))
    }

    /// What a sparse texture answers of its tiles
    /// ([`crate::sparse_query::SparseTextureFacts`]).
    fn sparse_texture_facts(
        &mut self,
        texture: Handle,
    ) -> Answer<crate::sparse_query::SparseTextureFacts> {
        let _ = texture;
        Err(Failure::unavailable("sparse_texture_facts"))
    }

    /// One sparse mapping program: the heap, the textures over it, and every command of
    /// `program` encoded in order through `MTLResourceStateCommandEncoder` and
    /// `MTLBlitCommandEncoder` ([`crate::sparse_mapping`]).
    ///
    /// The whole program is one call because what it measures is a **sequence**: the heap's
    /// `usedSize` after a map, after an unmap and after a move is three numbers of one experiment,
    /// and a case that made them three independent calls would be measuring three heaps.
    ///
    /// Every command is encoded whatever an earlier one answered, as
    /// [`compute_pass`](Side::compute_pass) does, and the answer names which one refused. **Every
    /// region is in tiles**, bounded by the texture's own grid, and the side checks the venue's
    /// tile size against the program's stated one before it encodes anything: a venue whose tiles
    /// are another size answers a typed refusal rather than being handed a region past its grid.
    fn sparse_mapping(
        &mut self,
        program: &crate::sparse_mapping::SparseMappingProgram,
    ) -> Answer<crate::sparse_mapping::SparseMapped> {
        let _ = program;
        Err(Failure::unavailable("sparse_mapping"))
    }

    /// What one command buffer answers about itself
    /// ([`crate::command_buffer_facts::CommandBufferFacts`]): its `device` and its `commandQueue`,
    /// the four timing getters before and after it ran, and whether a
    /// `resourceStateCommandEncoder` could be opened on one.
    ///
    /// The timings are recorded as **orderings and signs**, never as times: a recording holding a
    /// machine's clock would decay on its next run. What is durable is whether each getter answers
    /// zero before the work ran, whether it is positive after, and the order of the four.
    fn command_buffer_facts(&mut self) -> Answer<crate::command_buffer_facts::CommandBufferFacts> {
        Err(Failure::unavailable("command_buffer_facts"))
    }

    /// A command queue's limit on uncompleted command buffers, by the probe both sides run
    /// ([`crate::queue_slots`]): the queue is made with `max` as its
    /// `maxCommandBufferCount` (`newCommandQueue` where `max` is `None`), and at most `cap` command
    /// buffers are made on a thread of its own.
    ///
    /// The whole experiment is one call because what it observes is an **order of events** — a make
    /// that did not return, another queue's make that did, and the first one answering once a slot
    /// came back — and a case cannot state that as a sequence of independent calls.
    ///
    /// Nothing is committed, so nothing reaches the GPU, and no make can be left unanswered: the
    /// probe releases a blocked make by dropping a command buffer held uncommitted.
    fn queue_slots(
        &mut self,
        max: Option<u32>,
        cap: u32,
    ) -> Answer<crate::queue_slots::QueueSlots> {
        let _ = (max, cap);
        Err(Failure::unavailable("queue_slots"))
    }

    /// `newTextureWithDescriptor:` of a private render target the case describes whole
    /// ([`TargetSpec`]).
    ///
    /// It joins the same handle space as [`texture_2d`](Side::texture_2d), so a kernel may bind it
    /// and [`RenderWork::ReadTexture`] may read it back: a texture has one pool however it was
    /// made.
    ///
    /// [`TargetSpec`]: crate::render::TargetSpec
    /// [`RenderWork::ReadTexture`]: crate::render::RenderWork::ReadTexture
    fn target(&mut self, spec: &crate::render::TargetSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("target"))
    }

    /// `newRenderPipelineStateWithDescriptor:error:` of a vertex-and-fragment pipeline the case
    /// describes whole ([`RenderPipelineSpec`]).
    ///
    /// [`RenderPipelineSpec`]: crate::render::RenderPipelineSpec
    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &crate::render::RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let _ = (library, spec);
        Err(Failure::unavailable("render_pipeline"))
    }

    /// `newDepthStencilStateWithDescriptor:` of the state `spec` names.
    ///
    /// The descriptor is `MTLDepthStencilDescriptor` as the translator's public API already names
    /// it, so a compare function and a stencil operation have one spelling across the two sides.
    /// Its own handle space: `MTLDepthStencilState` is its own Metal object, not another name for
    /// one of the pools above.
    fn depth_stencil_state(&mut self, spec: &DepthStencilDescriptor) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("depth_stencil_state"))
    }

    /// **One** command buffer holding every piece of `work` in order, `commit`, and a bounded
    /// wait ([`RenderWork`]).
    ///
    /// Every piece is encoded whatever an earlier one answered, as the probes did: a refused draw
    /// leaves the pass and its other commands as they were, and the answer names which piece
    /// refused ([`PassOutcome::refused`], one entry per piece of `work`).
    ///
    /// [`RenderWork`]: crate::render::RenderWork
    /// The host's write of every byte of the plane a target lies over
    /// ([`crate::surface::PlaneSpec`]): `IOSurfaceLock`, a copy of `bytes`, `IOSurfaceUnlock` on
    /// the Metal side, and the stand-in buffer's own bytes on the runtime's.
    ///
    /// It is its own call rather than [`replace_region`](Side::replace_region) because the plane is
    /// addressed as **the surface's bytes**, stride and padding included, not as the texture's
    /// texels: a case asserts that the bytes between the rows stayed as they were, which a region
    /// write cannot say.
    fn write_plane(&mut self, target: Handle, bytes: &[u8]) -> Answer<()> {
        let _ = (target, bytes);
        Err(Failure::unavailable("write_plane"))
    }

    /// Every byte of the plane a target lies over, read back by the host after the GPU is done
    /// with it: the whole stride of every row, so the bytes between the rows are observed too.
    fn read_plane(&mut self, target: Handle) -> Answer<Vec<u8>> {
        let _ = target;
        Err(Failure::unavailable("read_plane"))
    }

    fn render(&mut self, work: &[crate::render::RenderWork]) -> Answer<PassOutcome> {
        let _ = work;
        Err(Failure::unavailable("render"))
    }

    /// What a wait made from a command buffer's own handler, for that same command buffer,
    /// answered ([`crate::handler_waits`]).
    ///
    /// The whole experiment is one call because the fact it observes is only visible from inside a
    /// handler: a command buffer is made with one small fill, the handler `wait.from` names makes
    /// the wait `wait.wait` names on it, and the answer comes back on a channel. The answer is a
    /// [`Value`]: `Ok` where the wait returned, the typed error where it did not.
    ///
    /// **Metal has no side here, deliberately.** Metal does not come back from this call — the
    /// probe that measured it ended with it — so [`crate::metal_side::MetalSide`] leaves this
    /// default in place and must go on leaving it. A case over it says
    /// [`MetalSource::did_not_return`] and record mode refuses the row before anything is encoded.
    ///
    /// Our own leg cannot hang: the runtime refuses the wait at once, so the handler returns and
    /// the command buffer completes, and the side waits for it with a bound.
    fn handler_wait(&mut self, wait: crate::handler_waits::OwnWait) -> Answer<Value> {
        let _ = wait;
        Err(Failure::unavailable("handler_wait"))
    }

    /// `detachBacking` (Metal's private `MTLBuffer` method) of the no-copy buffer `buffer`, called
    /// from a completed handler of a command buffer whose blit fill named it, `after` into that
    /// handler: the whole program in one call — the queue, the command buffer, the fill, the
    /// handler, the commit and this thread's own bounded wait — and what the detach answered
    /// inside the handler ([`crate::host_detach`]).
    ///
    /// One call rather than several, because the fact is a **relation** between a handler and the
    /// work a command buffer still holds, and a case cannot state "from inside the handler" as
    /// calls whose order nothing fixes.
    ///
    /// `after` is what makes the answer comparable: Metal's release of what the command buffer
    /// used races the start of its handlers, so the first microseconds of a handler have no
    /// answer, and this call asks in the region past the race.
    ///
    /// Bounded on both legs, and it holds no GPU: the handler sleeps `after` and returns, and a
    /// command buffer that does not arrive inside the wait is recorded as the negative fact.
    fn detach_in_a_completed_handler(
        &mut self,
        buffer: Handle,
        after: std::time::Duration,
    ) -> Answer<crate::host_detach::DetachedInHandler> {
        let _ = (buffer, after);
        Err(Failure::unavailable("detachBacking"))
    }

    /// What a shared event's value did, by the rising-value probe ([`crate::shared_events`]):
    /// `newSharedEvent`, its label, the host's sets, two GPU signals, three host waits, and a
    /// signal of `u64::MAX`.
    ///
    /// One call rather than a dozen, because the facts are a **sequence** — a value that did not
    /// fall when a lower one was set — and a case cannot state that as independent calls whose
    /// order nothing fixes.
    /// `newSharedEventHandle` of a shared event, then `newSharedEventWithHandle:` of what it
    /// answered, and what the two events then say about themselves
    /// ([`EventHandleRoundTrip`](crate::shared_events::EventHandleRoundTrip)).
    ///
    /// One call, because the fact under test is a **relation** between two objects: that the event
    /// the device opens from the handle is the one the handle came from, which only a value set on
    /// one and read on the other can say. Two independent calls over a handle space would be two
    /// objects whose identity nothing compares.
    ///
    /// Nothing here reaches the GPU: every value is set and read by the host, and both waits are
    /// bounded.
    fn shared_event_handle_round_trip(
        &mut self,
        label: &str,
    ) -> Answer<crate::shared_events::EventHandleRoundTrip> {
        let _ = label;
        Err(Failure::unavailable("shared_event_handle_round_trip"))
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        let _ = label;
        Err(Failure::unavailable("shared_event_rises"))
    }

    /// What the cross-queue wait probe observed ([`crate::event_orders::CrossQueueWait`]): one
    /// queue's command buffer waits for an event the other's signals, and the waiting one is
    /// committed first.
    ///
    /// One call rather than six, because what it observes is an **order**: a status read between
    /// two commits, and then the bytes both command buffers wrote. A case cannot state "this status
    /// was read while the other command buffer had not been committed yet" as independent calls.
    fn event_cross_queue_wait(&mut self) -> Answer<crate::event_orders::CrossQueueWait> {
        Err(Failure::unavailable("event_cross_queue_wait"))
    }

    /// `signaledValue` of one shared event two queues' command buffers signalled with `values`, in
    /// that submission order, once both completed.
    ///
    /// One call, because the fact is about the pair: whichever signal executed last, the value a
    /// lower signal cannot lower is what the event holds.
    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        let _ = values;
        Err(Failure::unavailable("event_highest_signal"))
    }

    /// What the held-command-buffer probe observed
    /// ([`crate::event_orders::HeldCommandBuffer`]): a command buffer held by a shared event the
    /// host raises later, one committed behind it on its own queue, and one on another queue.
    ///
    /// One call, because every fact is an order — which command buffer went on while another was
    /// held, and which handler ran before which.
    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        Err(Failure::unavailable("held_command_buffer"))
    }

    /// What the signal-before-a-wait probe observed
    /// ([`crate::event_orders::SignalBeforeWait`]): one command buffer that fills, signals, waits
    /// for a gate, fills again and signals again.
    ///
    /// One call, because the fact is that the first signal landed **while** the wait was still
    /// pending, which is a relation between two moments of one command buffer's life.
    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        Err(Failure::unavailable("signal_before_a_wait"))
    }

    /// What the scheduled-handler order probe observed
    /// ([`crate::scheduled_handlers::ScheduledHandlerOrder`]): a command buffer whose scheduled
    /// handler is held inside itself while the probe reads the status and a wait for the scheduling
    /// on another thread.
    ///
    /// One call, because the fact is a relation between a handler, a status and a wait on a third
    /// thread, and holding the handler is the only way any of it is observable at all.
    fn scheduled_handler_order(
        &mut self,
    ) -> Answer<crate::scheduled_handlers::ScheduledHandlerOrder> {
        Err(Failure::unavailable("scheduled_handler_order"))
    }

    /// The handles a pipeline state gives, the visible-function table it makes, and what a
    /// dispatch through that table wrote ([`crate::visible_tables`]).
    ///
    /// One call, because the four objects a case here compares only exist in sequence: a handle
    /// comes from a state, a table comes from the same state, an entry takes a handle of that
    /// table's state, and the words come from a dispatch through the filled table. Asking them
    /// apart would be four handle spaces whose members could each belong to another case's state,
    /// and the one shape the family may never reach is exactly that
    /// (`FunctionTableProblem::HandleOfAnotherState`).
    ///
    /// `caller` holds `spec.entry`; `functions` holds the `[[visible]]` functions the stage links
    /// and the case takes handles of. Two libraries because Metal's own linking takes them so: the
    /// caller declares the visible functions and another module defines them.
    ///
    /// **Three shapes a case must not state**, each measured and each this method's caller's to
    /// avoid rather than this method's to guess at: a `stage` of no stage at all, where plain
    /// Metal faults ([`crate::visible_tables::StageAsked`] holds no such value); a
    /// `function_count` near the device's maximum, where Metal ended an M4's process; and a handle
    /// of a state other than the one that made the table.
    fn visible_function_table(
        &mut self,
        caller: Handle,
        functions: Handle,
        spec: &crate::visible_tables::TableSpec<'_>,
    ) -> Answer<crate::visible_tables::TableOutcome> {
        let _ = (caller, functions, spec);
        Err(Failure::unavailable("visible_function_table"))
    }

    /// `newHeapWithDescriptor:` of the automatic or placement heap `spec` names
    /// ([`crate::heap_query::HeapSpec`]).
    ///
    /// A plain allocation: nothing is encoded, dispatched or committed, and the heap's memory is
    /// the `spec`'s own size. A sparse heap is [`Side::sparse_heap`]'s, whose descriptor has other
    /// rules.
    fn heap(&mut self, spec: &crate::heap_query::HeapSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("heap"))
    }

    /// Every property `MTLHeap` declares about itself, read of the heap `heap` names
    /// ([`crate::heap_query::HeapOwnFacts`]).
    ///
    /// Pure getters: one call because a difference should name the property, not the object, and
    /// because reading them apart would be ten sites that could each name a different heap.
    fn heap_own_facts(&mut self, heap: Handle) -> Answer<crate::heap_query::HeapOwnFacts> {
        let _ = heap;
        Err(Failure::unavailable("heap_own_facts"))
    }

    /// `maxAvailableSizeWithAlignment:` of `alignment` on the heap `heap` names.
    ///
    /// A pure getter over Metal's own suballocator. A case asks only powers of two: an alignment
    /// of 0 is a shape nothing in this tree says Metal takes, and any other non-power-of-two is one
    /// the runtime refuses `Invalid` ([`crate::heap_query`]).
    fn heap_max_available_size(&mut self, heap: Handle, alignment: u64) -> Answer<u64> {
        let _ = (heap, alignment);
        Err(Failure::unavailable("heap_max_available_size"))
    }

    /// `setLabel:` of `label` on the heap `heap` names.
    ///
    /// It answers nothing: what the label became is read back through [`Side::heap_own_facts`], so
    /// a case states the set and the read as two steps and a recording holds both.
    fn set_heap_label(&mut self, heap: Handle, label: &str) -> Answer<()> {
        let _ = (heap, label);
        Err(Failure::unavailable("set_heap_label"))
    }

    /// `setPurgeableState:` of `state` on the heap `heap` names, as the raw `MTLPurgeableState`
    /// it answered — the state the heap was in **before** the call.
    ///
    /// A case asks this last of whatever it asks it of: `Empty` lets Metal discard the heap's
    /// bytes, and nothing is read through the heap afterwards but these answers.
    fn heap_purgeable_state(
        &mut self,
        heap: Handle,
        state: metal2vulkan_runtime::PurgeableState,
    ) -> Answer<u64> {
        let _ = (heap, state);
        Err(Failure::unavailable("heap_purgeable_state"))
    }

    /// `newBufferWithLength:options:` of an automatic heap, or
    /// `newBufferWithLength:options:offset:` of a placement one, as `spec` names
    /// ([`crate::heap_query::HeapBufferSpec`]), on the heap `heap` names.
    ///
    /// The options are the heap's own, which both APIs require: the runtime refuses any other
    /// `Invalid`, and a case that stated one would be a case about a descriptor Metal's validation
    /// stops.
    fn heap_buffer(
        &mut self,
        heap: Handle,
        spec: &crate::heap_query::HeapBufferSpec,
    ) -> Answer<Handle> {
        let _ = (heap, spec);
        Err(Failure::unavailable("heap_buffer"))
    }

    /// Every property `MTLResource` declares, read of the buffer, texture or view `of` names,
    /// where `heap` is the heap the case placed it in ([`crate::heap_query::ResourceFacts`]).
    ///
    /// `heap` is what makes `heap` and `heapOffset` readable: the facts say whether the resource
    /// answered **that** heap, which is the relation Metal settles, rather than an address.
    fn resource_facts(
        &mut self,
        of: crate::heap_query::QueriedResource,
        heap: Option<Handle>,
    ) -> Answer<crate::heap_query::ResourceFacts> {
        let _ = (of, heap);
        Err(Failure::unavailable("resource_facts"))
    }

    /// `setLabel:` of `label` on the resource `of` names, as [`Side::set_heap_label`] is for a
    /// heap.
    fn set_resource_label(
        &mut self,
        of: crate::heap_query::QueriedResource,
        label: &str,
    ) -> Answer<()> {
        let _ = (of, label);
        Err(Failure::unavailable("set_resource_label"))
    }

    /// `makeAliasable` on the resource `of` names. What it changed is read back through
    /// [`Side::resource_facts`] and the heap's own `usedSize`.
    fn make_resource_aliasable(&mut self, of: crate::heap_query::QueriedResource) -> Answer<()> {
        let _ = of;
        Err(Failure::unavailable("make_resource_aliasable"))
    }

    /// `setPurgeableState:` of `state` on the resource `of` names, as the raw `MTLPurgeableState`
    /// it answered, with the ordering [`Side::heap_purgeable_state`] states.
    fn resource_purgeable_state(
        &mut self,
        of: crate::heap_query::QueriedResource,
        state: metal2vulkan_runtime::PurgeableState,
    ) -> Answer<u64> {
        let _ = (of, state);
        Err(Failure::unavailable("resource_purgeable_state"))
    }

    /// One of `MTLTexture`'s four view creators, as `request` names
    /// ([`crate::texture_views::ViewRequest`]), on the texture `of` names.
    ///
    /// A view holds no memory of its own, so the call allocates no texels and reaches no queue. A
    /// request whose ranges or format Metal's validation stops is refused by the request's own
    /// guard before the call is made.
    fn texture_view(
        &mut self,
        of: Handle,
        request: &crate::texture_views::ViewRequest,
    ) -> Answer<Handle> {
        let _ = (of, request);
        Err(Failure::unavailable("texture_view"))
    }

    /// What the texture `texture` names answers about its place in a view chain and about sharing
    /// ([`crate::texture_views::ViewFacts`]), where `parent` is the texture the case made it a
    /// view of.
    ///
    /// `parent` is what makes `parentTexture` readable as a relation rather than an address: the
    /// facts say whether the texture answered **that** parent. `None` is "the case made this of no
    /// texture", where `parentTexture` answering anything at all is the difference.
    fn texture_view_facts(
        &mut self,
        texture: Handle,
        parent: Option<Handle>,
    ) -> Answer<crate::texture_views::ViewFacts> {
        let _ = (texture, parent);
        Err(Failure::unavailable("texture_view_facts"))
    }

    /// `newSharedTextureWithDescriptor:` of the descriptor `query` names, which is the only way
    /// `shareable` is ever true and the only receiver `newSharedTextureHandle` answers for.
    fn shared_texture(&mut self, query: &crate::texture_query::HeapTextureQuery) -> Answer<Handle> {
        let _ = query;
        Err(Failure::unavailable("shared_texture"))
    }

    /// `newSharedTextureHandle` of the texture `texture` names, what the handle answers of itself,
    /// and the texture `newSharedTextureWithHandle:` opens from it
    /// ([`crate::texture_views::SharedHandleFacts`]).
    ///
    /// One call, because a handle is observable only through what it opens: a case cannot hold one
    /// as a resource and ask it questions later, since neither API gives it a shape the suite's
    /// vocabulary names.
    ///
    /// A case asks this only of a **shareable** texture. What Metal does for one that is not is
    /// unmeasured here: the runtime models nil, and a case that asked and was wrong would end
    /// Metal's process.
    fn shared_texture_handle(
        &mut self,
        texture: Handle,
    ) -> Answer<crate::texture_views::SharedHandleFacts> {
        let _ = texture;
        Err(Failure::unavailable("shared_texture_handle"))
    }
}

/// A tile render pipeline a case makes: `MTLTileRenderPipelineDescriptor`'s facts the cases state.
///
/// The attachment formats are the case's own rather than one fixed format, because a tile
/// function's implicit imageblock **is** the pass's attachments: their formats decide the
/// imageblock's layout, and so what the function reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePipelineSpec<'a> {
    /// The tile function, by its name in the library.
    pub entry: &'a str,
    /// The color attachment formats, at attachment index 0 up in this order.
    pub attachments: &'a [PixelFormat],
    /// `maxTotalThreadsPerThreadgroup`: `None` is Metal's 0, the pipeline's own maximum.
    pub max_total_threads_per_threadgroup: Option<u32>,
    /// `threadgroupSizeMatchesTileSize`.
    pub threadgroup_size_matches_tile_size: bool,
}

impl<'a> TilePipelineSpec<'a> {
    /// The tile function `entry` over `attachments`, with Metal's defaults for the rest.
    pub const fn new(entry: &'a str, attachments: &'a [PixelFormat]) -> Self {
        Self {
            entry,
            attachments,
            max_total_threads_per_threadgroup: None,
            threadgroup_size_matches_tile_size: false,
        }
    }

    /// The descriptor with `maxTotalThreadsPerThreadgroup` set to `limit`.
    #[must_use]
    pub const fn limited(mut self, limit: u32) -> Self {
        self.max_total_threads_per_threadgroup = Some(limit);
        self
    }

    /// The descriptor with `threadgroupSizeMatchesTileSize` set.
    #[must_use]
    pub const fn tile_sized(mut self) -> Self {
        self.threadgroup_size_matches_tile_size = true;
        self
    }

    /// The words the case's input fingerprint covers: every field of the descriptor, so a tile
    /// pipeline that moves is a change to the case's inputs.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.entry.as_bytes().to_vec();
        for format in self.attachments {
            out.extend(format.as_raw().to_le_bytes());
        }
        out.extend(
            self.max_total_threads_per_threadgroup
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.push(u8::from(self.threadgroup_size_matches_tile_size));
        out
    }
}

/// What a tile render pipeline state answers. Each field is a Metal getter's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileFacts {
    /// `maxTotalThreadsPerThreadgroup`: Metal's 1024 where the device runs a threadgroup of 1024
    /// threads, lowered by the descriptor's own limit to the largest threadgroup of even sides
    /// within it.
    pub max_total_threads_per_threadgroup: u64,
    /// `threadgroupSizeMatchesTileSize`, which the state echoes from its descriptor.
    pub threadgroup_size_matches_tile_size: bool,
    /// `imageblockSampleLength`: the attachments' texels packed. `None` where the side answered a
    /// typed error for it rather than a length.
    pub imageblock_sample_length: Option<u64>,
}

impl TileFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "max_total_threads_per_threadgroup",
                Observed::value(Value::Uint(self.max_total_threads_per_threadgroup)),
            ))
            .with(Node::leaf(
                "threadgroup_size_matches_tile_size",
                Observed::value(Value::Bool(self.threadgroup_size_matches_tile_size)),
            ))
            .with(Node::leaf(
                "imageblock_sample_length",
                Observed::value(match self.imageblock_sample_length {
                    Some(length) => Value::Variant {
                        name: "Some".to_owned(),
                        payload: Box::new(Value::Uint(length)),
                    },
                    None => Value::unit_variant("None"),
                }),
            ))
    }
}

/// One command a case encodes into a tile render pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileStep {
    /// `setRenderPipelineState:` of a tile state.
    Pipeline(Handle),
    /// `setTileBuffer:offset:atIndex:`.
    TileBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setTileBytes:length:atIndex:`.
    TileBytes { bytes: Vec<u8>, index: u32 },
    /// `setTileTexture:atIndex:`.
    TileTexture { texture: Handle, index: u32 },
    /// `dispatchThreadsPerTile:`.
    DispatchThreadsPerTile([u32; 3]),
}

impl TileStep {
    /// The site name the step's answer is emitted at: the call it makes, so a refusal reads as the
    /// Metal call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            TileStep::Pipeline(_) => "setRenderPipelineState",
            TileStep::TileBuffer { .. } => "setTileBuffer",
            TileStep::TileBytes { .. } => "setTileBytes",
            TileStep::TileTexture { .. } => "setTileTexture",
            TileStep::DispatchThreadsPerTile(_) => "dispatchThreadsPerTile",
        }
    }
}

/// One layer of a rasterization-rate map's descriptor
/// (`MTLRasterizationRateLayerDescriptor`), as its quality samples along each axis.
///
/// Both axes hold at least one sample, because Metal's
/// `initWithSampleCount:horizontal:vertical:` answers nil for an axis of none (verified while
/// porting this family). A descriptor of **no layers** is the shape whose `layerCount` is 0, and a
/// case says that by holding no layer at all rather than by a layer with an empty axis.
#[derive(Clone, Debug, PartialEq)]
pub struct RateMapLayer {
    pub horizontal: Vec<f32>,
    pub vertical: Vec<f32>,
}

impl RateMapLayer {
    /// A layer of one quality sample on each axis.
    pub fn uniform(horizontal: f32, vertical: f32) -> Self {
        Self {
            horizontal: vec![horizontal],
            vertical: vec![vertical],
        }
    }
}

/// How the descriptor a rate map is compiled from is **built**.
///
/// Metal's descriptor is an object with four ways to reach the same state, and a program that
/// reaches it one way must get the map it would have got another way. The runtime has no descriptor
/// object — its [`RateMapDescriptor`](metal2vulkan::bind::RateMapDescriptor) is a plain struct a
/// consumer fills — so a form is a **Metal-side construction path** whose answer the runtime's one
/// struct has to match. That is what makes a form a comparison rather than a readback: nothing of
/// the descriptor's own surface is compared, because our side would only be reading back the struct
/// the case just handed it. What is compared is what the compiled map **derives**.
///
/// [`RateMapForm::Holed`] is the one form whose state the runtime's struct cannot express, and that
/// is the measurement: Metal's `layerCount` counts the layers from index 0 **contiguously**, so a
/// descriptor holding a layer at index 1 and nil at index 0 answers 0 and compiles a map of no
/// layers (measured on an Apple M3 Max under macOS 27.0.1, where `layerAtIndex:1` is still
/// non-nil). A consumer of the runtime cannot build that shape, so the case says a holed descriptor
/// is, to Metal, a descriptor of no layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateMapForm {
    /// `rasterizationRateMapDescriptorWithScreenSize:` then `setLayer:atIndex:` per layer. The
    /// form every case of this family used before the others existed.
    ScreenSizeThenSetLayer,
    /// `rasterizationRateMapDescriptorWithScreenSize:layer:`, the one-layer convenience. A request
    /// of any other layer count cannot take this form.
    ScreenSizeAndLayer,
    /// `rasterizationRateMapDescriptorWithScreenSize:layerCount:layers:`, the whole descriptor in
    /// one call. `objc2-metal` 0.3.2 does not bind it, so the Metal side sends the selector by
    /// hand.
    ScreenSizeLayerCountLayers,
    /// `rasterizationRateMapDescriptorWithScreenSize:` then `layers[i] = layer`
    /// (`MTLRasterizationRateLayerArray`'s `setObject:atIndexedSubscript:`), which Apple's header
    /// calls identical to `setLayer:atIndex:`.
    LayersSubscript,
    /// The request's layers set from index 1 up, with index 0 left nil: the shape whose
    /// `layerCount` Metal answers 0 for. The runtime's side builds a descriptor of **no** layers,
    /// which is what the case claims the two are.
    Holed,
}

impl RateMapForm {
    /// Whether this form can build a descriptor of `layers` layers. The one-layer convenience
    /// takes exactly one; the holed form needs a layer to leave a hole under.
    pub fn takes(self, layers: usize) -> bool {
        match self {
            RateMapForm::ScreenSizeAndLayer => layers == 1,
            RateMapForm::Holed => layers >= 1,
            _ => true,
        }
    }

    /// The name the form's site carries.
    pub fn at(self) -> &'static str {
        match self {
            RateMapForm::ScreenSizeThenSetLayer => "form-set-layer-at-index",
            RateMapForm::ScreenSizeAndLayer => "form-with-screen-size-and-layer",
            RateMapForm::ScreenSizeLayerCountLayers => "form-with-layer-count-and-layers",
            RateMapForm::LayersSubscript => "form-layers-subscript",
            RateMapForm::Holed => "form-holed-at-index-zero",
        }
    }
}

/// A rasterization-rate map's descriptor, as both sides take it.
#[derive(Clone, Copy, Debug)]
pub struct RateMapRequest<'a> {
    /// `screenSize`'s width and height in screen pixels.
    pub screen: [u32; 2],
    /// `layers`, from layer 0.
    pub layers: &'a [RateMapLayer],
    /// `label`, which the map keeps.
    pub label: Option<&'a str>,
    /// How the Metal side builds the descriptor ([`RateMapForm`]). The runtime has one way, so
    /// this moves only Metal's leg — which is the point: every form must compile one map.
    pub form: RateMapForm,
}

impl<'a> RateMapRequest<'a> {
    /// A request taking the form every case of this family took before the others existed.
    pub fn new(screen: [u32; 2], layers: &'a [RateMapLayer]) -> RateMapRequest<'a> {
        RateMapRequest {
            screen,
            layers,
            label: None,
            form: RateMapForm::ScreenSizeThenSetLayer,
        }
    }
}

/// A coordinate to map, and the layer to map it in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinateProbe {
    /// The screen coordinate `mapScreenToPhysicalCoordinates:forLayer:` is asked at.
    pub screen: [f32; 2],
    /// The physical coordinate `mapPhysicalToScreenCoordinates:forLayer:` is asked at.
    pub physical: [f32; 2],
    /// The layer both are asked in, which a case keeps inside the map's own `layerCount`.
    pub layer: u64,
}

/// What a map's two coordinate maps answered at one probe, by their bits.
///
/// A rate map's coordinate maps are `f32` arithmetic over the parameter data, so an answer is kept
/// as bits: a signed zero and a one-ulp difference are both real, and the runtime's own claim is
/// that it reproduces Apple's bit for bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappedCoordinates {
    /// `mapScreenToPhysicalCoordinates:forLayer:` of the probe's screen coordinate.
    pub screen_to_physical: [u32; 2],
    /// `mapPhysicalToScreenCoordinates:forLayer:` of the probe's physical coordinate.
    pub physical_to_screen: [u32; 2],
}

/// What a rasterization-rate map answers. Each field is a Metal getter's answer, and a size's
/// depth is kept because Metal's is 1 where its own documentation says 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RateMapFacts {
    pub label: Option<String>,
    pub screen_size: [u64; 3],
    pub physical_granularity: [u64; 3],
    pub layer_count: u64,
    /// `physicalSizeForLayer:` of each index the case asked about, in that order.
    pub physical_sizes: Vec<[u64; 3]>,
    pub parameter_size: u64,
    pub parameter_align: u64,
}

impl RateMapFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let size = |axes: &[u64; 3]| {
            Value::fields([
                ("width", Value::Uint(axes[0])),
                ("height", Value::Uint(axes[1])),
                ("depth", Value::Uint(axes[2])),
            ])
        };
        let label = match &self.label {
            Some(label) => Value::Variant {
                name: "Some".to_owned(),
                payload: Box::new(Value::Lines(vec![label.clone()])),
            },
            None => Value::unit_variant("None"),
        };
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf("label", Observed::value(label)))
            .with(Node::leaf(
                "screen_size",
                Observed::value(size(&self.screen_size)),
            ))
            .with(Node::leaf(
                "physical_granularity",
                Observed::value(size(&self.physical_granularity)),
            ))
            .with(Node::leaf(
                "layer_count",
                Observed::value(Value::Uint(self.layer_count)),
            ))
            .with(Node::leaf(
                "parameter_size",
                Observed::value(Value::Uint(self.parameter_size)),
            ))
            .with(Node::leaf(
                "parameter_align",
                Observed::value(Value::Uint(self.parameter_align)),
            ));
        for (at, axes) in self.physical_sizes.iter().enumerate() {
            node = node.with(Node::leaf(
                format!("physical_size_for_layer/{at}"),
                Observed::value(size(axes)),
            ));
        }
        node
    }
}

/// One render command the host encodes into an indirect command buffer: the command's own pipeline
/// state, the buffers it binds, and its one mesh draw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshCommand {
    /// The state the command sets, from [`Side::mesh_pipeline_for_commands`].
    pub pipeline: Handle,
    /// The object buffers the command binds, each at its index.
    pub object_buffers: Vec<(Handle, u32)>,
    /// The mesh buffers the command binds, each at its index.
    pub mesh_buffers: Vec<(Handle, u32)>,
    /// The command's draw: `drawMeshThreadgroups` of a grid, or `drawMeshThreads` of threads.
    pub draw: CommandDraw,
}

/// An indirect render command's mesh draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandDraw {
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    Threadgroups {
        groups: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    Threads {
        threads: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
}

/// A texture a case makes over a buffer: over one plane of the surface the buffer stands for, or
/// over the buffer itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneTexture {
    /// The surface's four-character OSType, `None` for a texture over the buffer itself.
    pub ostype: Option<[u8; 4]>,
    /// The texture's raw `MTLPixelFormat`.
    pub format: u32,
    /// The texture's extent in texels.
    pub size: [u32; 2],
    /// The plane of the surface the texture covers.
    pub plane: u32,
    /// Each of the surface's two planes: its first byte in the buffer and its `bytesPerRow`. A
    /// texture over the buffer itself uses the first alone.
    pub planes: [(u64, u64); 2],
}

/// One fill of a texture, as the probe encoded it.
#[derive(Clone, Debug, PartialEq)]
pub enum Fill {
    /// `fillTexture:level:slice:region:pattern:` (Apple SPI) of these pattern bytes.
    Pattern {
        level: u32,
        slice: u32,
        /// The region as `x`, `y`, `width`, `height`.
        region: [u32; 4],
        pattern: Vec<u8>,
    },
    /// `fillTexture:level:slice:region:color:pixelFormat:` (Apple SPI) of this color packed in
    /// this raw `MTLPixelFormat`.
    Color {
        level: u32,
        slice: u32,
        region: [u32; 4],
        /// The color's four channels.
        color: [f64; 4],
        /// The raw `MTLPixelFormat` the color packs in.
        packed_into: u32,
    },
}

impl Fill {
    /// The site name the fill's answer is emitted at: the call it makes.
    pub fn call(&self) -> &'static str {
        match self {
            Fill::Pattern { .. } => "fillTexture",
            Fill::Color { .. } => "fillTextureColor",
        }
    }

    /// The words the fill's parameters fingerprint as.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Fill::Pattern {
                level,
                slice,
                region,
                pattern,
            } => {
                out.extend(b"pattern");
                out.extend(level.to_le_bytes());
                out.extend(slice.to_le_bytes());
                for axis in region {
                    out.extend(axis.to_le_bytes());
                }
                out.extend(pattern);
            }
            Fill::Color {
                level,
                slice,
                region,
                color,
                packed_into,
            } => {
                out.extend(b"color");
                out.extend(level.to_le_bytes());
                out.extend(slice.to_le_bytes());
                for axis in region {
                    out.extend(axis.to_le_bytes());
                }
                for channel in color {
                    out.extend(channel.to_le_bytes());
                }
                out.extend(packed_into.to_le_bytes());
            }
        }
        out
    }
}

/// One full-screen fragment fill: a classic vertex-and-fragment pipeline, a private render target
/// of its own, one draw of three vertices, and the target's bytes read back through a blit.
///
/// This is the shape every store, clear and load question of a render pass is written in: the
/// vertex function computes the full-screen triangle from its `[[vertex_id]]`, so the pass needs no
/// vertex buffer, and what is observed is the bytes the fragment function's return value stored.
/// The target is **private** and read back by a blit into a shared buffer, because that is how the
/// probes measured it and storage mode is part of what a padded format's answer means.
#[derive(Clone, Copy, Debug)]
pub struct FragmentFill<'a> {
    /// The library both functions come from.
    pub library: Handle,
    /// The vertex function, which takes no buffer.
    pub vertex: &'a str,
    /// The fragment function, whose return value the attachment stores.
    pub fragment: &'a str,
    /// The attachment's and the target's pixel format.
    pub format: PixelFormat,
    /// The target's extent in texels.
    pub size: [u32; 2],
    /// Bytes a row of the read-back, which the case states because the pitch is part of what it
    /// means.
    pub row_pitch: u32,
    /// The buffers the pass sets as fragment buffers 0 and up, in order.
    pub buffers: &'a [Handle],
    /// The vertices the one `drawPrimitives:` of `MTLPrimitiveTypeTriangle` draws.
    pub vertices: u32,
}

/// What a [`FragmentFill`] answered: the pass's command-buffer status and the target's bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentFilled {
    /// The render pass's command-buffer status, as a variant of its name.
    pub status: Value,
    /// The target's level 0, row after row at the fill's row pitch.
    pub bytes: Vec<u8>,
}

/// The color format every mesh case's attachments take: `MTLPixelFormatRGBA32Uint`, whose texel is
/// four 32-bit words, so what a shader stores is read back as the words it stored and a diff names
/// the word rather than a byte.
pub const TARGET_FORMAT: &str = "RGBA32Uint";

/// The bytes one [`TARGET_FORMAT`] texel takes.
pub const TARGET_TEXEL: u32 = 16;

/// The functions of a mesh render pipeline, as a case names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshFunctions<'a> {
    /// The object function, where the pipeline has one.
    pub object: Option<&'a str>,
    pub mesh: &'a str,
    pub fragment: &'a str,
    /// The color attachments of [`TARGET_FORMAT`] the pipeline writes.
    pub attachments: u32,
}

/// A mesh render pipeline a case makes **for its reflection**: the three stages, and the colour
/// attachment formats.
///
/// Separate from [`MeshFunctions`], which is what the drawing families state, and the difference
/// is the attachments. `MeshFunctions` takes a **count** and the sides fill that many attachments
/// with [`TARGET_FORMAT`], because a family that draws reads its target back and
/// [`TARGET_FORMAT`] is the format those readers share. A reflection case draws nothing, so it has
/// no reason to want that format and one strong reason not to: [`TARGET_FORMAT`] is an integer
/// format, and a fragment function returning `float4` over an integer attachment is a pipeline
/// **Metal refuses** ("output of type float4 is not compatible with a MTLPixelFormatRGBA32Uint
/// color attachment", measured on an M3 Max under Metal 27.0.1). A reflection case given that
/// count would compare a refusal against a refusal at every site and reflect nothing.
///
/// So this spec names its formats the way [`TilePipelineSpec`] already names its own, and a case
/// states a format its fragment function's output is compatible with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshReflectionSpec<'a> {
    /// The object function, where the pipeline has one.
    pub object: Option<&'a str>,
    /// The mesh function.
    pub mesh: &'a str,
    /// The fragment function.
    pub fragment: &'a str,
    /// The colour attachment formats, at attachment index 0 up in this order.
    pub attachments: &'a [PixelFormat],
}

impl MeshReflectionSpec<'_> {
    /// The words a case's input fingerprint covers: every stage and every format, so a pipeline
    /// that moves is a change to the case's inputs.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.mesh.as_bytes().to_vec();
        out.extend(self.fragment.as_bytes());
        out.extend(self.object.unwrap_or("").as_bytes());
        for format in self.attachments {
            out.extend(format.as_raw().to_le_bytes());
        }
        out
    }
}

/// One command a case encodes into a render pass.
///
/// The stage and scope fields are the raw `MTLRenderStages` and `MTLBarrierScope` bits, so the
/// vocabulary names neither API's types and a case reads as the Metal call it makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshStep {
    /// `setRenderPipelineState:`.
    Pipeline(Handle),
    /// `setObjectBuffer:offset:atIndex:`.
    ObjectBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setObjectBytes:length:atIndex:`.
    ObjectBytes { bytes: Vec<u8>, index: u32 },
    /// `setMeshBuffer:offset:atIndex:`.
    MeshBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setFragmentBuffer:offset:atIndex:`.
    FragmentBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawThreadgroups {
        groups: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawThreads {
        threads: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:
    /// threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawIndirect {
        buffer: Handle,
        offset: u64,
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `memoryBarrierWithScope:afterStages:beforeStages:`.
    BarrierScope { scope: u64, after: u64, before: u64 },
    /// `memoryBarrierWithResources:count:afterStages:beforeStages:` over buffers.
    BarrierResources {
        buffers: Vec<Handle>,
        after: u64,
        before: u64,
    },
}

impl MeshStep {
    /// The site name the step's answer is emitted at: the call it makes, so a refusal reads as the
    /// Metal call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            MeshStep::Pipeline(_) => "setRenderPipelineState",
            MeshStep::ObjectBuffer { .. } => "setObjectBuffer",
            MeshStep::ObjectBytes { .. } => "setObjectBytes",
            MeshStep::MeshBuffer { .. } => "setMeshBuffer",
            MeshStep::FragmentBuffer { .. } => "setFragmentBuffer",
            MeshStep::DrawThreadgroups { .. } => "drawMeshThreadgroups",
            MeshStep::DrawThreads { .. } => "drawMeshThreads",
            MeshStep::DrawIndirect { .. } => "drawMeshThreadgroupsWithIndirectBuffer",
            MeshStep::BarrierScope { .. } => "memoryBarrierWithScope",
            MeshStep::BarrierResources { .. } => "memoryBarrierWithResources",
        }
    }
}

/// `MTLBarrierScopeBuffers`.
pub const SCOPE_BUFFERS: u64 = 0x1;

/// `MTLRenderStageVertex`.
pub const STAGE_VERTEX: u64 = 0x1;

/// `MTLRenderStageFragment`.
pub const STAGE_FRAGMENT: u64 = 0x2;

/// `MTLRenderStageObject`.
pub const STAGE_OBJECT: u64 = 0x8;

/// `MTLRenderStageMesh`.
pub const STAGE_MESH: u64 = 0x10;

/// What one render pass came to.
#[derive(Clone, Debug, PartialEq)]
pub struct PassOutcome {
    /// What each step of the pass answered, in the order the case gave them: `None` where the call
    /// answered, the typed error where it did not. A side whose calls cannot refuse (Metal's
    /// encoder methods return nothing) answers `None` throughout, so a refusal of ours compares
    /// against Metal's answer and reads as a refusal rather than a divergence.
    pub refused: Vec<Option<Failure>>,
    /// The command buffer's status after the bounded wait, as the variant of its name.
    pub status: Value,
    /// The command buffer's error after it completed, where it reported one.
    pub error: Option<Value>,
}

/// What an indirect render command's getters answered, beside what the whole record holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndirectReadBack {
    /// `getCommandType`'s answer, as its bits.
    pub command_type: u64,
    /// The primitive type the draw getter answered, or `None` where it answered no draw.
    pub answered_primitive_type: Option<u64>,
    /// The primitive type the whole record's read-back keeps.
    pub recorded_primitive_type: Option<u64>,
}

/// Where a case's Metal answer comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalSource {
    /// Record mode runs the case on Metal.
    Run,
    /// Metal has no public call for what the case reads, or running it on Metal is not safe, so
    /// its answer is carried from a measurement this tree already holds: the module whose table it
    /// came from, and whether record mode may replace it.
    Carried {
        from: &'static str,
        re_record: ReRecord,
        /// The chip the carried measurement was taken on, which is the oracle's unless the case
        /// says otherwise ([`MetalSource::measured_on`]).
        measured_on: &'static str,
    },
    /// **Metal did not return from the call.** There is nothing to carry and nothing to run.
    ///
    /// A row of this shape has no value and no behaviour on Metal's side: the probe that found it
    /// out ended Metal's process, so the lane it came from holds no line for it either — it is not
    /// recorded-only material, it simply has no answer. What the case pins is the one fact there
    /// is, that Metal never came back from the call, against our typed refusal at that site
    /// ([`Verdict::MetalNeverReturned`](crate::verdict::Verdict::MetalNeverReturned)).
    ///
    /// **It is never produced by running the shape.** The transcript is stated by the case
    /// ([`Case::carried_transcript`]) from a probe that already died, and record mode refuses the
    /// row outright rather than taking an answer from it: this is the one source whose Metal leg
    /// would end the process, and the suite's standing rule is that no case may do that.
    DidNotReturn {
        /// Where the probe that died is written down.
        from: &'static str,
        /// The call Metal did not return from, as the case names the site.
        call: &'static str,
        /// What happened, in the probe's own words: how Metal went away.
        says: &'static str,
    },
}

impl MetalSource {
    /// A carried answer record mode may replace by running the case on Metal.
    pub const fn carried(from: &'static str) -> Self {
        Self::Carried {
            from,
            re_record: ReRecord::Allowed,
            measured_on: crate::recording::ORACLE_CHIP,
        }
    }

    /// A carried answer record mode must refuse to replace, because running the case on Metal is
    /// not safe.
    pub const fn carried_only(from: &'static str, reason: &'static str) -> Self {
        Self::Carried {
            from,
            re_record: ReRecord::Forbidden { reason },
            measured_on: crate::recording::ORACLE_CHIP,
        }
    }

    /// The same carried answer, measured on `chip` rather than on the oracle this round records
    /// on.
    ///
    /// A case says this where its table predates the oracle — the two families carrying an Apple
    /// M4's values — so the recording holds the chip in a field a reader and the report can see,
    /// instead of in the `from` prose where nothing could.
    pub const fn measured_on(self, chip: &'static str) -> Self {
        match self {
            Self::Carried {
                from, re_record, ..
            } => Self::Carried {
                from,
                re_record,
                measured_on: chip,
            },
            Self::Run => Self::Run,
            source @ Self::DidNotReturn { .. } => source,
        }
    }

    /// A row whose Metal leg does not come back: `call` is the call, `says` is how Metal went
    /// away, and `from` is where the probe that found that out is written down.
    pub const fn did_not_return(
        from: &'static str,
        call: &'static str,
        says: &'static str,
    ) -> Self {
        Self::DidNotReturn { from, call, says }
    }

    /// Why record mode must not run the case on Metal, where it must not.
    pub const fn forbidden(&self) -> Option<&'static str> {
        match self {
            MetalSource::Carried {
                re_record: ReRecord::Forbidden { reason },
                ..
            } => Some(reason),
            // Record mode must never run this one: the call it would make is the call Metal does
            // not come back from. The case's own words are the reason a refusal prints.
            MetalSource::DidNotReturn { says, .. } => Some(says),
            MetalSource::Run | MetalSource::Carried { .. } => None,
        }
    }
}

/// Whether record mode may run a case on Metal.
///
/// A case whose Metal leg faults the GPU, trips its watchdog or ends the process carries its
/// answer and forbids re-recording. Record mode refuses such a case by this flag, so a later
/// re-record of a whole selection cannot crash the GPU or the process by walking into one. Our own
/// leg of such a case is safe and compares normally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReRecord {
    Allowed,
    Forbidden { reason: &'static str },
}

/// A case of the suite.
pub trait Case {
    /// The case's name: its recording's file name, and its row in the report.
    fn name(&self) -> &'static str;

    /// Where the case's Metal answer comes from.
    fn metal_source(&self) -> MetalSource;

    /// The shaders the case runs, as committed fixtures; empty where it runs none.
    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        Ok(Vec::new())
    }

    /// The fingerprint of everything the case feeds its run: the bytes it writes into each buffer
    /// and texture before it runs, its specialization constants, and the parameters it sets.
    ///
    /// The suite emits this itself, at the reserved site [`INPUTS_SITE`], and the comparison checks
    /// it before any output. A case whose inputs are not the recording's is
    /// [`Unrecorded`](crate::verdict::Verdict::Unrecorded), never a pass: its Metal answer was
    /// taken from other bytes. Without it a recording decays in silence the first time a fixture
    /// or a generator changes.
    fn inputs(&self) -> Result<Vec<Input>, String>;

    /// Set the case up on `side`, run it, and emit the sites it observes. A call that did not
    /// answer stops the run with an [`Abort`]; the sites emitted before it still compare.
    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort>;

    /// What the case loosens, and what justifies each. Empty is the default, and the default is
    /// what a case should keep: comparison is exact.
    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    /// How many runs record mode takes the case's Metal answer over.
    ///
    /// The default ([`crate::run::RECORD_RUNS`]) is enough for a case whose behaviour is settled:
    /// the runs then all fold into one outcome and the recording says so. A case whose work Metal
    /// orders by nothing must ask for more, because its recording's value is the **set** of
    /// outcomes with their frequencies, and a set sampled three times states a frequency nobody
    /// should read. Asking for more runs never invents one: the recording holds the outcomes this
    /// many runs actually produced.
    fn record_runs(&self) -> u32 {
        crate::run::RECORD_RUNS
    }

    /// The transcript the case's carried measurement states, and the runs it stands on, for a case
    /// whose [`MetalSource`] is [`MetalSource::Carried`]. `None` for a case record mode runs on
    /// Metal.
    ///
    /// A carried answer is the case stating what our own measurement of this chip found, in the
    /// shape the case emits, so it compares through the same path a recorded run does.
    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(None)
    }
}

/// The reserved site the suite emits a case's input fingerprint at.
pub const INPUTS_SITE: &str = "inputs";

/// One input a case feeds its run, and its digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    /// What the input is, as the case names it: `buffer:0`, `constant:width`, `texture:albedo`.
    pub name: String,
    pub digest: crate::transcript::Digest,
}

impl Input {
    /// The input `name`, fingerprinted by the SHA-256 of `bytes`.
    pub fn bytes(name: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            name: name.into(),
            digest: crate::transcript::Digest::sha256(bytes),
        }
    }
}

/// The node the suite emits for a case's inputs.
pub fn inputs_node(inputs: &[Input]) -> Node {
    Node::branch(
        INPUTS_SITE,
        inputs
            .iter()
            .map(|input| {
                Node::leaf(
                    input.name.clone(),
                    Observed::absent().digested(input.digest.clone()),
                )
            })
            .collect(),
    )
}

/// The site a call that answered an object emits, as a carried measurement states it: the call
/// answered, and nothing of what it answered. A carried transcript holds the same sites a run
/// emits, so it states this one too.
pub fn answered() -> Observed {
    Observed::value(Value::ok(Value::Absent))
}

/// Emit a site for `answer` and go on with what it answered, or stop the run.
///
/// `observed` says what the site records of a successful answer: `Value::Absent` for a call that
/// answers an object, whose identity is not observable.
pub fn call<T>(
    sites: &mut Sites,
    at: &str,
    answer: Answer<T>,
    observed: impl FnOnce(&T) -> Value,
) -> Result<T, Abort> {
    match answer {
        Ok(value) => {
            sites.value(at, Value::ok(observed(&value)));
            Ok(value)
        }
        Err(failure) => {
            sites.push(Node::leaf(at, Observed::value(failure.recorded())));
            // A device's honest gap is recorded at the site, not only where it stops the case.
            // The whole-case `LacksFeature` verdict comes from an abort, so without this a case
            // that carries on past an unsupported call would have that site read as a refusal of
            // ours — the one thing the verdict exists to prevent.
            if let Some(needs) = &failure.lacks {
                sites.device_gap(at, needs.clone());
            }
            Err(Abort {
                at: at.to_owned(),
                failure,
            })
        }
    }
}

/// [`call`] for a **sweep that carries on** past a call that did not answer: the site records the
/// answer either way, the venue's own gap is recorded with it, and nothing aborts.
///
/// A sweep over many queries wants one refused query to compare as a refusal at its own site while
/// the rest stay readable, which an abort would lose. Written by hand that reads
/// `Err(failure) => failure.recorded()`, and then [`Failure::lacks`] — whether the typed refusal is the
/// **venue's** honest gap rather than ours — is silently dropped, so a device that simply cannot
/// run one query of the sweep has that site scored as a refusal of ours. That is the one thing the
/// `lacks-feature` verdict exists to prevent, and it was happening on lavapipe for every
/// multisampled sparse query. Use this rather than matching, so the fact cannot be forgotten.
pub fn probe<T>(
    sites: &mut Sites,
    at: &str,
    answer: Answer<T>,
    observed: impl FnOnce(&T) -> Value,
) -> Option<T> {
    match answer {
        Ok(value) => {
            sites.value(at, Value::ok(observed(&value)));
            Some(value)
        }
        Err(failure) => {
            sites.push(Node::leaf(at, Observed::value(failure.recorded())));
            if let Some(needs) = &failure.lacks {
                sites.device_gap(at, needs.clone());
            }
            None
        }
    }
}

/// [`call`] for a call that answers an object: the site records that it answered, and nothing of
/// what it answered.
pub fn made<T>(sites: &mut Sites, at: &str, answer: Answer<T>) -> Result<T, Abort> {
    call(sites, at, answer, |_| Value::Absent)
}
