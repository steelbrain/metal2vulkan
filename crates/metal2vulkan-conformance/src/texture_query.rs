//! A texture descriptor as both sides take it, for the calls that only *ask* about one.
//!
//! `MTLDevice`'s `heapTextureSizeAndAlignWithDescriptor:` is a pure getter: it compiles nothing,
//! allocates nothing and reaches no GPU, and its answer is Metal's own layout arithmetic. That is
//! exactly the shape a conformance case wants — the whole descriptor space is reachable from the
//! CPU, every answer is two numbers, and the whole family can be re-measured on the oracle as often
//! as anybody asks.
//!
//! It is also the family whose numbers were, until now, held by a comment: `runtime_lane`'s
//! `heap_metal_sizes` carries tables of what an Apple M4 answered, with nothing mechanical tying
//! them to a Metal run. A case over this type takes the answer from Metal.
//!
//! The vocabulary is Metal's own enumerations as the translator's public API names them, so a value
//! a case writes reaches `MTLTextureType`, `MTLPixelFormat` and `MTLTextureUsage` by its raw
//! ordinal and reaches the runtime's descriptor unchanged. A case may therefore name one of Apple's
//! private formats (`Unspecialized`, the 4:2:2 pair, the two-plane extended-range pair) and one of
//! Apple's private usage bits as readily as a public one.

use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The storage mode a case asks a resource for: three of Metal's four modes, and the suite's one
/// spelling of them.
///
/// It lives here because the queried descriptor was the first thing to need it, and it is used
/// wherever a case states where a resource lives — a queried descriptor
/// ([`HeapTextureQuery::storage`]) and a copy-and-fill program's buffers, heaps and textures
/// ([`crate::work`]). One spelling rather than a second enumeration or a `private` flag per type,
/// so a case that sweeps the modes sweeps one list.
///
/// All four modes. `Memoryless` was the last to arrive, and what a case asks **of** one is the
/// descriptor rather than the contents: its heap footprint, whether Metal makes a texture of it at
/// all, and what the texture answers about itself. Nothing reads a memoryless texture's texels,
/// because there are none to read outside a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// `MTLStorageModeShared`.
    Shared,
    /// `MTLStorageModeManaged`: the device's own copy, which the host reaches through an explicit
    /// synchronization. A case that reads such a resource reads it through a copy into a `Shared`
    /// one, which needs none.
    Managed,
    /// `MTLStorageModePrivate`.
    Private,
    /// `MTLStorageModeMemoryless`: a texture whose bytes exist only inside the pass that renders
    /// it, so no host call and no copy reaches them.
    ///
    /// Metal takes it on a plain `MTLTextureDescriptor` of **any** usage, not only a render
    /// target: measured on an Apple M3 Max under macOS 27.0.1, a 64 by 64 `RGBA8Unorm` descriptor
    /// of `ShaderRead`, of `ShaderWrite`, of `RenderTarget`, of their union and of `Unknown` each
    /// answers a texture and a heap footprint, in every texture type but `TextureBuffer`. So
    /// "memoryless means render target" is not a rule plain Metal enforces at the descriptor, and
    /// a case over it is a case about the footprint and the getters.
    ///
    /// A **heap** of this mode is a different question and is refused:
    /// [`crate::heap_query::HeapSpec::is_well_formed`] says so.
    Memoryless,
}

impl Storage {
    /// The word the site's name carries.
    pub fn word(self) -> &'static str {
        match self {
            Storage::Shared => "shared",
            Storage::Managed => "managed",
            Storage::Private => "private",
            Storage::Memoryless => "memoryless",
        }
    }
}

/// A descriptor a case asks Metal about: every `MTLTextureDescriptor` field the heap size query
/// reads.
///
/// `allowGPUOptimizedContents` is a field, and was not always: it is left at Metal's default, on,
/// everywhere but the one family that is **about** it
/// ([`crate::cases::texture_descriptor_pairs`]), because with it off the runtime announces
/// `optimized-depth-stencil-texture` on MoltenVK for a depth or stencil texture, and a case whose
/// subject is Metal's arithmetic should not carry a device's announcement as well. That is why the
/// family that turns it off states a **colour** format in every such row, and why the field
/// fingerprints and names itself only where it is off — a descriptor that leaves it at Metal's
/// default fingerprints exactly as it did before the field existed, so every recording taken
/// before it stays the answer to the inputs it was taken from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapTextureQuery {
    pub texture_type: TextureType,
    pub format: PixelFormat,
    /// `width`, `height` and `depth`, in texels.
    pub size: [u32; 3],
    /// `mipmapLevelCount`.
    pub levels: u32,
    /// `sampleCount`.
    pub samples: u32,
    /// `arrayLength`.
    pub array_length: u32,
    pub usage: TextureUsage,
    pub storage: Storage,
    /// `allowGPUOptimizedContents`, at Metal's own default of `true` unless a case says otherwise.
    pub gpu_optimized: bool,
}

impl HeapTextureQuery {
    /// A 2D descriptor of `format`, `width` by `height`, one level, one sample, one slice, read by
    /// shaders, of `storage`.
    pub const fn texture_2d(format: PixelFormat, size: [u32; 2], storage: Storage) -> Self {
        Self {
            texture_type: TextureType::Type2D,
            format,
            size: [size[0], size[1], 1],
            levels: 1,
            samples: 1,
            array_length: 1,
            usage: TextureUsage::SHADER_READ,
            storage,
            gpu_optimized: true,
        }
    }

    /// The descriptor with `texture_type`, and the `height`, `depth` and `arrayLength` that type
    /// takes: a 1D or buffer texture one row tall, an array type `slices` slices, a 3D texture
    /// `depth` texels deep.
    #[must_use]
    pub const fn of_type(mut self, texture_type: TextureType, depth: u32, slices: u32) -> Self {
        self.texture_type = texture_type;
        match texture_type {
            TextureType::Type1D | TextureType::Type1DArray | TextureType::TextureBuffer => {
                self.size[1] = 1;
            }
            TextureType::Type3D => self.size[2] = depth,
            _ => {}
        }
        self.array_length = slices;
        self
    }

    /// The descriptor with `levels` mipmap levels.
    #[must_use]
    pub const fn with_levels(mut self, levels: u32) -> Self {
        self.levels = levels;
        self
    }

    /// The descriptor with `usage`.
    #[must_use]
    pub const fn with_usage(mut self, usage: TextureUsage) -> Self {
        self.usage = usage;
        self
    }

    /// The descriptor with `samples` samples.
    #[must_use]
    pub const fn with_samples(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    /// The descriptor with `allowGPUOptimizedContents` off, which is what parts a texture Metal
    /// keeps losslessly compressed from one it does not.
    #[must_use]
    pub const fn plain_contents(mut self) -> Self {
        self.gpu_optimized = false;
        self
    }

    /// The descriptor in `storage`.
    #[must_use]
    pub const fn stored(mut self, storage: Storage) -> Self {
        self.storage = storage;
        self
    }

    /// The descriptor of `format`.
    #[must_use]
    pub const fn of_format(mut self, format: PixelFormat) -> Self {
        self.format = format;
        self
    }

    /// Why this descriptor must never be handed to Metal, where it must not: the shapes
    /// `-[MTLTextureDescriptorInternal validateWithDevice:]` fails its assertion for, which ends
    /// Metal's process rather than answering nil.
    ///
    /// One list, decided from the descriptor's own fields, and reached with no device at all —
    /// which is the only proof a guard against ending a process can have. The Metal side calls it
    /// before every call it makes of a queried descriptor, so a row of a recorded table cannot
    /// reach Metal around it.
    ///
    /// Both shapes were **measured** on an Apple M3 Max under macOS 27.0.1, plain Metal with no
    /// API validation layer, in a throwaway process forked per shape, and Metal's own words are
    /// what each arm reports:
    ///
    /// - a `mipmapLevelCount` past 1 on `Type1D` or `Type1DArray`: *"requests a mipmapLevelCount
    ///   of 2 but the texture type MTLTextureType1D requires that mipmapLevelCount is 1"*. A
    ///   `TextureBuffer` of the same one row is **not** in this list: it takes a
    ///   `mipmapLevelCount` of 2 and answers 2 back, which is the per-type difference the
    ///   interaction family is about;
    /// - an `arrayLength` past 1 on a type that is not an array type: *"requests an arrayLength of
    ///   3, but MTLTextureType2D requires that arrayLength is 1"*, and the same for `Cube`,
    ///   `Type3D` and `TextureBuffer`;
    /// - a `mipmapLevelCount` past 1 on a **multisample** type;
    /// - a `mipmapLevelCount` past the **full chain** of the descriptor's largest axis,
    ///   `floor(log2(max(width, height, depth))) + 1`. Measured per type: a 64 by 64 2D descriptor
    ///   takes 7 and ends the process at 8; an 8 by 8 by 64 3D one takes 7 and ends it at 8; a 16
    ///   by 16 by 256 one takes 9 and ends it at 10. So a 3D descriptor's **depth counts toward
    ///   the level count Metal allows**, which is the opposite of how it counts toward the chain's
    ///   bytes — where the full chain is its width and height's alone, and the family's own case
    ///   is what records that. A `TextureBuffer` is held to the same bound and passes it: its one
    ///   row of 64 texels takes a `mipmapLevelCount` of 7 and answers 7 back, which is the whole
    ///   difference from `Type1D`.
    ///
    /// Formats, usages and sample counts are **not** here. The shapes Metal stops over those are
    /// named where the tables that avoid them are written
    /// ([`crate::cases::heap_metal_sizes`], [`crate::cases::format_footprints`]), because each is a
    /// fact about one value rather than about the descriptor's own shape.
    pub fn forbidden(&self) -> Option<String> {
        let one_dimensional = matches!(
            self.texture_type,
            TextureType::Type1D | TextureType::Type1DArray
        );
        if one_dimensional && self.levels > 1 {
            return Some(format!(
                "{:?} requires a mipmapLevelCount of 1, not {}",
                self.texture_type, self.levels
            ));
        }
        let array = matches!(
            self.texture_type,
            TextureType::Type1DArray
                | TextureType::Type2DArray
                | TextureType::Type2DMultisampleArray
                | TextureType::CubeArray
        );
        if !array && self.array_length > 1 {
            return Some(format!(
                "{:?} requires an arrayLength of 1, not {}",
                self.texture_type, self.array_length
            ));
        }
        let multisample = matches!(
            self.texture_type,
            TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray
        );
        if multisample && self.levels > 1 {
            return Some(format!(
                "{:?} requires a mipmapLevelCount of 1, not {}",
                self.texture_type, self.levels
            ));
        }
        let largest = self.size.iter().copied().max().unwrap_or(1).max(1);
        let full_chain = u32::BITS - largest.leading_zeros();
        if self.levels > full_chain {
            return Some(format!(
                "a largest axis of {largest} has a full chain of {full_chain} levels, not {}",
                self.levels
            ));
        }
        None
    }

    /// The site name the query's answer sits at: every field that can change the answer, so two
    /// descriptors of one family never share a site and a reader of a diff knows which descriptor
    /// moved without counting rows.
    pub fn site(&self) -> String {
        format!(
            "{}/{:?}/{}x{}x{}/levels-{}/samples-{}/slices-{}/usage-{:#x}/{}{}",
            self.format.name(),
            self.texture_type,
            self.size[0],
            self.size[1],
            self.size[2],
            self.levels,
            self.samples,
            self.array_length,
            self.usage.as_raw(),
            self.storage.word(),
            // Named only where it is off, so a descriptor at Metal's default sits at the site it
            // sat at before this field existed.
            match self.gpu_optimized {
                true => "",
                false => "/plain-contents",
            },
        )
    }

    /// The words the descriptor fingerprints as, for a case's input fingerprint: every field, so a
    /// descriptor that moves is a change to the case's inputs rather than a silent re-point of the
    /// recording.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.texture_type.as_raw().to_le_bytes());
        out.extend(self.format.as_raw().to_le_bytes());
        for axis in self.size {
            out.extend(axis.to_le_bytes());
        }
        out.extend(self.levels.to_le_bytes());
        out.extend(self.samples.to_le_bytes());
        out.extend(self.array_length.to_le_bytes());
        out.extend(self.usage.as_raw().to_le_bytes());
        // The two modes that were here first keep their bytes, so a recording taken before the
        // third existed is still the answer to the inputs it was taken from.
        out.extend(match self.storage {
            Storage::Shared => [0u8],
            Storage::Private => [1u8],
            Storage::Managed => [2u8],
            Storage::Memoryless => [3u8],
        });
        // Written only where `allowGPUOptimizedContents` is off, so a descriptor that leaves it at
        // Metal's default fingerprints as it did before the field existed and every recording
        // taken then stays the answer to the inputs it was taken from.
        if !self.gpu_optimized {
            out.extend(b"plain-contents");
        }
        out
    }
}

/// `MTLSizeAndAlign`: what the heap size query answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeAndAlign {
    pub size: u64,
    pub align: u64,
}

impl SizeAndAlign {
    /// The answer as the fields a site records: both numbers, so a diff names which of the two
    /// moved.
    pub fn value(&self) -> crate::transcript::Value {
        crate::transcript::Value::fields([
            ("size", crate::transcript::Value::Uint(self.size)),
            ("align", crate::transcript::Value::Uint(self.align)),
        ])
    }
}
