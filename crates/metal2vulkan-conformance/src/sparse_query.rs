//! A sparse tile query as both sides take it, for `MTLDevice`'s two sparse tile getters.
//!
//! `sparseTileSizeWithTextureType:pixelFormat:sampleCount:` and its `sparsePageSize:` form answer
//! the tile a sparse texture maps in, in texels; `sparseTileSizeInBytes` and its
//! `ForSparsePageSize:` form answer that tile's bytes. All four are **pure getters**: they compile
//! nothing, allocate nothing, reach no GPU and need no sparse-capable device, so a case over them
//! is safe on every venue and comparable on every venue — including one with no sparse binding at
//! all, which is what makes this the first sparse family the suite can carry end to end.
//!
//! The runtime answers them from a model of Metal's rule (`sparse::tile_texels`), the same on every
//! device, whose comment says it was measured on an Apple M4 and whose only possible oracle is
//! Metal. `runtime_lane::sparse_textures` asserts exactly three of its answers —
//! `(Type2D, R32Float, 1)`, `(Cube, R32Float, 1)` and `(Type2DMultisample, RGBA8Unorm, 2 and 4)` —
//! so the rule's two interesting claims, that the answer does not depend on the texture type and
//! that it halves the tile's sides as the block's bytes double, have never been checked against
//! Metal over a sweep. A case over this type checks them.
//!
//! The vocabulary is Metal's own enumerations as the translator's public API names them, so a value
//! a case writes reaches `MTLTextureType`, `MTLPixelFormat` and `MTLSparsePageSize` by its raw
//! ordinal and reaches the runtime's call unchanged.

use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime::SparsePageSize;

/// The page a query asks in: one of Metal's three, or none, which is the getter without a page
/// argument (whose answer Metal documents as the 16-KiB page's).
///
/// `None` is a query of its own rather than a spelling of `Some(Size16)`, because the two are two
/// Metal selectors and the suite's business is what each selector answers. A release where they
/// part fails the case rather than passing under a rule nobody re-measured.
pub type QueryPage = Option<SparsePageSize>;

/// The word a page carries in a site's name.
pub fn page_word(page: QueryPage) -> &'static str {
    match page {
        None => "default-page",
        Some(SparsePageSize::Size16) => "page-16",
        Some(SparsePageSize::Size64) => "page-64",
        Some(SparsePageSize::Size256) => "page-256",
        // `SparsePageSize` is `#[non_exhaustive]`: a page Metal gains is named by its raw value
        // rather than silently sharing a site with one of these.
        Some(page) => match page.raw() {
            101 => "page-16",
            102 => "page-64",
            103 => "page-256",
            _ => "page-unknown",
        },
    }
}

/// The bytes a page fingerprints as.
pub fn page_words(page: QueryPage) -> Vec<u8> {
    match page {
        None => vec![0u8; 9],
        Some(page) => {
            let mut out = vec![1u8];
            out.extend(page.raw().to_le_bytes());
            out
        }
    }
}

/// One sparse tile query: the three arguments of
/// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:`, and the page it asks in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseTileQuery {
    pub texture_type: TextureType,
    pub format: PixelFormat,
    /// `sampleCount`. A case states one the texture type takes: 1 for a single-sampled type, and
    /// for a multisample type a count the oracle's device holds.
    pub samples: u32,
    pub page: QueryPage,
}

impl SparseTileQuery {
    /// The query of `texture_type`, `format` and `samples` in the page `page`.
    pub const fn new(
        texture_type: TextureType,
        format: PixelFormat,
        samples: u32,
        page: QueryPage,
    ) -> Self {
        Self {
            texture_type,
            format,
            samples,
            page,
        }
    }

    /// The site name the query's answer sits at: every argument, so two queries of one family
    /// never share a site and a reader of a diff knows which query moved without counting rows.
    pub fn site(&self) -> String {
        format!(
            "{}/{:?}/samples-{}/{}",
            self.format.name(),
            self.texture_type,
            self.samples,
            page_word(self.page),
        )
    }

    /// The words the query fingerprints as, for a case's input fingerprint: every argument, so a
    /// query that moves is a change to the case's inputs rather than a silent re-point of the
    /// recording.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.texture_type.as_raw().to_le_bytes());
        out.extend(self.format.as_raw().to_le_bytes());
        out.extend(self.samples.to_le_bytes());
        out.extend(page_words(self.page));
        out
    }
}

/// `MTLSize`: what a sparse tile getter answers, in texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

impl TileSize {
    /// The answer as the fields a site records: all three axes, so a diff names which one moved.
    pub fn value(&self) -> crate::transcript::Value {
        crate::transcript::Value::fields([
            ("width", crate::transcript::Value::Uint(self.width)),
            ("height", crate::transcript::Value::Uint(self.height)),
            ("depth", crate::transcript::Value::Uint(self.depth)),
        ])
    }
}

/// A sparse heap a case makes: `MTLHeapDescriptor` with `MTLHeapTypeSparse`, as the cases need it.
///
/// The storage is always `Private`: Metal stops any other for a sparse heap even without its
/// validation layer, so it is not a field a case may set wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseHeapSpec {
    /// `size`, in bytes. Metal rounds it up to a whole page.
    pub size: u64,
    /// `sparsePageSize`.
    pub page: SparsePageSize,
}

impl SparseHeapSpec {
    /// A heap of `size` bytes in 16-KiB pages, which is the descriptor's default and the one the
    /// lane uses.
    pub const fn new(size: u64) -> Self {
        Self {
            size,
            page: SparsePageSize::Size16,
        }
    }

    /// The heap with `page` as its `sparsePageSize`.
    #[must_use]
    pub const fn in_page(mut self, page: SparsePageSize) -> Self {
        self.page = page;
        self
    }

    /// The word the site's name carries.
    pub fn word(&self) -> String {
        format!("heap-{}-{}", self.size, page_word(Some(self.page)))
    }

    /// The bytes the heap fingerprints as.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.size.to_le_bytes().to_vec();
        out.extend(page_words(Some(self.page)));
        out
    }
}

/// What a heap answers: the three getters a case reads of one.
///
/// `currentAllocatedSize` is deliberately not here. A sparse heap holds no memory of its own, so
/// Metal's answer for it is an implementation's accounting rather than a fact about the heap, and
/// comparing it would pin our allocator to Apple's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapFacts {
    /// `type`, by its raw `MTLHeapType` value.
    pub heap_type: u64,
    /// `size`: the descriptor's size rounded up to a whole page.
    pub size: u64,
    /// `usedSize`: the bytes the heap's resources have taken, which for a sparse heap is its
    /// mapped tiles.
    pub used_size: u64,
}

impl HeapFacts {
    /// The facts as the fields a site records, so a diff names which of the three moved.
    pub fn value(&self) -> crate::transcript::Value {
        crate::transcript::Value::fields([
            ("type", crate::transcript::Value::Uint(self.heap_type)),
            ("size", crate::transcript::Value::Uint(self.size)),
            ("used_size", crate::transcript::Value::Uint(self.used_size)),
        ])
    }
}

/// A texture a case makes from a sparse heap: `MTLTextureDescriptor`'s fields a sparse texture's
/// tail arithmetic reads.
///
/// The storage is always the heap's (`Private`), because Metal stops any other for a heap's
/// texture even without its validation layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseTextureSpec {
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
    /// Whether the texture is a render target as well as shader-read, which a multisampled sparse
    /// texture must be.
    pub render_target: bool,
    /// `allowGPUOptimizedContents`: whether Metal may keep the texture losslessly compressed.
    /// Metal's default, `YES`, lets it compress this shader-read texture, and a tile a move
    /// rebound then reads back whole for only a fraction of its texels
    /// (`probes/sparse_move_compression.m`), so the rows that read moved texels turn it off.
    pub gpu_optimized: bool,
}

impl SparseTextureSpec {
    /// A single-sampled 2D texture of `format`, `width` by `height`, `levels` levels, read by
    /// shaders.
    pub const fn texture_2d(format: PixelFormat, size: [u32; 2], levels: u32) -> Self {
        Self {
            texture_type: TextureType::Type2D,
            format,
            size: [size[0], size[1], 1],
            levels,
            samples: 1,
            array_length: 1,
            render_target: false,
            gpu_optimized: true,
        }
    }

    /// The texture with `texture_type`, and the `depth` and `arrayLength` that type takes.
    #[must_use]
    pub const fn of_type(mut self, texture_type: TextureType, depth: u32, slices: u32) -> Self {
        self.texture_type = texture_type;
        if matches!(texture_type, TextureType::Type3D) {
            self.size[2] = depth;
        }
        self.array_length = slices;
        self
    }

    /// The texture at `samples` samples, as a render target, which is what a multisampled sparse
    /// texture needs.
    #[must_use]
    pub const fn multisampled(mut self, samples: u32) -> Self {
        self.samples = samples;
        self.render_target = true;
        self
    }

    /// The texture with `allowGPUOptimizedContents` off, which keeps Metal from compressing it.
    #[must_use]
    pub const fn unoptimized(mut self) -> Self {
        self.gpu_optimized = false;
        self
    }

    /// The site name the texture's answers sit at: every field that can change the tail
    /// arithmetic.
    pub fn site(&self) -> String {
        format!(
            "{}/{:?}/{}x{}x{}/levels-{}/samples-{}/slices-{}{}{}",
            self.format.name(),
            self.texture_type,
            self.size[0],
            self.size[1],
            self.size[2],
            self.levels,
            self.samples,
            self.array_length,
            if self.render_target { "/target" } else { "" },
            if self.gpu_optimized {
                ""
            } else {
                "/unoptimized"
            },
        )
    }

    /// The bytes the texture fingerprints as. An unoptimized texture adds a byte where Metal's
    /// default adds none, so the fingerprints taken before the field existed still hold.
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
        out.push(u8::from(self.render_target));
        if !self.gpu_optimized {
            out.push(0xA0);
        }
        out
    }
}

/// What a sparse texture answers of its tiles: the three getters Metal gives one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseTextureFacts {
    /// `isSparse`.
    pub is_sparse: bool,
    /// `firstMipmapInTail`. Metal answers a level that may be past the texture's own level count.
    pub first_mipmap_in_tail: u64,
    /// `tailSizeInBytes`: 0 for a texture without a tail.
    pub tail_size_in_bytes: u64,
}

impl SparseTextureFacts {
    /// The facts as the fields a site records, so a diff names which of the three moved.
    pub fn value(&self) -> crate::transcript::Value {
        crate::transcript::Value::fields([
            ("is_sparse", crate::transcript::Value::Bool(self.is_sparse)),
            (
                "first_mipmap_in_tail",
                crate::transcript::Value::Uint(self.first_mipmap_in_tail),
            ),
            (
                "tail_size_in_bytes",
                crate::transcript::Value::Uint(self.tail_size_in_bytes),
            ),
        ])
    }
}
