//! A texture laid over a buffer's bytes, as both sides take it:
//! `newTextureWithDescriptor:offset:bytesPerRow:` on an `MTLBuffer`.
//!
//! The call is public on both sides and it allocates no texels of its own — the texture's memory
//! **is** the buffer's — so a case over it is cheap, reaches no GPU, and can be re-measured on the
//! oracle as often as anybody asks. What it answers is Metal's own rule table for the shape: which
//! descriptors Metal lays over a buffer at all, which offsets and row pitches it takes, and which
//! of the descriptor's own fields the made texture keeps against which it takes the buffer's.
//!
//! The descriptor is [`crate::texture_query::HeapTextureQuery`], the suite's one spelling of an
//! `MTLTextureDescriptor`, so a buffer-backed texture and a heap-size query cannot disagree about
//! what a case's descriptor says. Three fields are the call's own and the request holds them:
//! the buffer's length and storage mode, the offset into it, and the bytes between the texture's
//! rows.
//!
//! # The row pitch is stated, never derived
//!
//! [`BufferTextureRequest::row_bytes`] is the texture's own row in bytes, and the **case** states
//! it. Neither side can be asked for it — `MTLTexture` answers no bytes-per-texel, and a format
//! table of ours would be a second spelling to keep in step — and it is the field everything here
//! turns on, so it is part of what a case means and part of its input fingerprint.
//!
//! # The hazard this module owns
//!
//! A row pitch **below the texture's own row** is the one shape of this family that is not safe to
//! hand to Metal: the AGX driver fails the assertion `bytes_per_row >= used_bytes_per_row` and ends
//! the process. So the refusal lives in [`ends_metals_process`], decided from the request's own
//! shape, and `metal_side::MetalSide` calls it before it makes the call. A case of such a
//! shape says [`MetalSource::did_not_return`](crate::case::MetalSource::did_not_return) and record
//! mode refuses the row before anything is made; the guard is the second line, so that a row
//! somebody adds to a recorded table by mistake is refused by the side rather than by a convention.

use crate::texture_query::{HeapTextureQuery, Storage};
use crate::transcript::{Node, NodeName, Observed, Value};

/// A texture a case asks a buffer for: `newTextureWithDescriptor:offset:bytesPerRow:`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferTextureRequest {
    /// The buffer's length in bytes, which a side makes zeroed.
    pub buffer_length: u64,
    /// Where the buffer's memory lives. The made texture reports **the buffer's** resource options
    /// whatever the descriptor names, which is one of the facts this family records.
    pub buffer_storage: Storage,
    /// The descriptor, by every field `MTLTextureDescriptor` carries that the call reads.
    pub descriptor: HeapTextureQuery,
    /// `offset`: the texture's first byte in the buffer.
    pub offset: u64,
    /// `bytesPerRow`: the bytes between the texture's rows.
    pub bytes_per_row: u64,
    /// The texture's own row in bytes — its width in texels times the format's bytes a texel — as
    /// the **case** states it.
    ///
    /// Stated rather than derived: `MTLTexture` answers no block or texel size, so neither side can
    /// be asked, and a row pitch below this number is the shape that ends Metal's process
    /// ([`ends_metals_process`]).
    pub row_bytes: u64,
}

impl BufferTextureRequest {
    /// A texture buffer of `format` and `width` texels over a buffer of `buffer_length` shared
    /// bytes, from `offset`, its row `bytes_per_row` bytes, with `texel` bytes a texel.
    pub const fn texture_buffer(
        buffer_length: u64,
        format: metal2vulkan::metal::PixelFormat,
        width: u32,
        texel: u64,
        offset: u64,
        bytes_per_row: u64,
    ) -> Self {
        Self {
            buffer_length,
            buffer_storage: Storage::Shared,
            descriptor: HeapTextureQuery::texture_2d(format, [width, 1], Storage::Shared).of_type(
                metal2vulkan::metal::TextureType::TextureBuffer,
                1,
                1,
            ),
            offset,
            bytes_per_row,
            row_bytes: width as u64 * texel,
        }
    }

    /// A 2D texture of `format`, `size` texels, over a buffer of `buffer_length` shared bytes from
    /// `offset`, its rows `bytes_per_row` apart, with `texel` bytes a texel.
    pub const fn texture_2d(
        buffer_length: u64,
        format: metal2vulkan::metal::PixelFormat,
        size: [u32; 2],
        texel: u64,
        offset: u64,
        bytes_per_row: u64,
    ) -> Self {
        Self {
            buffer_length,
            buffer_storage: Storage::Shared,
            descriptor: HeapTextureQuery::texture_2d(format, size, Storage::Shared),
            offset,
            bytes_per_row,
            row_bytes: size[0] as u64 * texel,
        }
    }

    /// The same request with the descriptor's own usage set to `usage`.
    #[must_use]
    pub const fn used(mut self, usage: metal2vulkan::metal::TextureUsage) -> Self {
        self.descriptor = self.descriptor.with_usage(usage);
        self
    }

    /// The same request whose **descriptor** names the storage mode `storage`, the buffer's
    /// staying as it is: the shape that asks which of the two the made texture reports.
    #[must_use]
    pub const fn descriptor_storage(mut self, storage: Storage) -> Self {
        self.descriptor.storage = storage;
        self
    }

    /// The same request over a buffer of `storage`.
    #[must_use]
    pub const fn over(mut self, storage: Storage) -> Self {
        self.buffer_storage = storage;
        self
    }

    /// The same request with `levels` mipmap levels in its descriptor.
    #[must_use]
    pub const fn with_levels(mut self, levels: u32) -> Self {
        self.descriptor = self.descriptor.with_levels(levels);
        self
    }

    /// The site the request's answer sits at: the descriptor's own site and the three numbers the
    /// call adds, so no two requests of a table share a site and a reader of a diff knows which
    /// row moved without counting.
    pub fn site(&self) -> String {
        format!(
            "{}/over-{}-{}/offset-{}/pitch-{}/row-{}",
            self.descriptor.site(),
            self.buffer_length,
            self.buffer_storage.word(),
            self.offset,
            self.bytes_per_row,
            self.row_bytes,
        )
    }

    /// The words the request fingerprints as: the descriptor's own, then the call's three numbers
    /// and the stated row.
    ///
    /// Keyed by the call it makes and the numbers it hands that call, never by a positional
    /// discriminant, so another field of another family renumbers nothing here.
    pub fn words(&self) -> Vec<u8> {
        let mut out = b"newTextureWithDescriptor:offset:bytesPerRow:".to_vec();
        out.extend(self.descriptor.words());
        out.extend(self.buffer_length.to_le_bytes());
        out.extend(match self.buffer_storage {
            Storage::Shared => [0u8],
            Storage::Private => [1u8],
            Storage::Managed => [2u8],
            Storage::Memoryless => [3u8],
        });
        out.extend(self.offset.to_le_bytes());
        out.extend(self.bytes_per_row.to_le_bytes());
        out.extend(self.row_bytes.to_le_bytes());
        out
    }
}

/// Why a request must never be handed to Metal, where it must not.
///
/// One list, decided from the **request's own shape**, and the only such list: the Metal side calls
/// it in the one place that makes the call, so a row of a recorded table cannot reach Metal around
/// it. A test reaches it with no device at all, which is the only proof such a guard can have.
///
/// The one shape: a row pitch below the texture's own row. The AGX driver fails
/// `bytes_per_row >= used_bytes_per_row` and ends the process, so the suite's standing rule — no
/// case may end Metal's process — forbids the call outright. The runtime refuses it typed
/// (`BufferTextureProblem::RowPitch`), which is what a case of this shape pins.
///
/// A pitch **at or above** the row is refused by nothing here: a texture buffer takes any pitch
/// that covers its texels (measured: 300 bytes over a 256-byte row), and a 2D texture's own rules
/// about alignment are Metal's to answer, not ours to pre-empt.
pub fn ends_metals_process(request: &BufferTextureRequest) -> Option<String> {
    (request.bytes_per_row < request.row_bytes).then(|| {
        format!(
            "bytesPerRow {} is below the texture's own row of {} bytes, which fails AGX's \
             assertion `bytes_per_row >= used_bytes_per_row` and ends Metal's process",
            request.bytes_per_row, request.row_bytes
        )
    })
}

/// What a texture made over a buffer answers of itself: every `MTLTexture` getter both sides have,
/// so a difference names the field rather than the object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferTextureFacts {
    /// `textureType`, by its raw `MTLTextureType`.
    pub texture_type: u64,
    /// `pixelFormat`, by its raw `MTLPixelFormat`.
    pub pixel_format: u64,
    pub width: u64,
    pub height: u64,
    pub depth: u64,
    pub mipmap_level_count: u64,
    pub array_length: u64,
    pub sample_count: u64,
    /// `usage`, by its raw `MTLTextureUsage`.
    pub usage: u64,
    /// `storageMode`, by its raw `MTLStorageMode`.
    pub storage_mode: u64,
    /// `cpuCacheMode`, by its raw `MTLCPUCacheMode`.
    pub cpu_cache_mode: u64,
    /// `hazardTrackingMode`, by its raw `MTLHazardTrackingMode`.
    pub hazard_tracking_mode: u64,
    /// `allowGPUOptimizedContents`.
    pub gpu_optimized: bool,
}

impl BufferTextureFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape and a
    /// difference names the one getter that moved.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let number = |value: u64| Observed::value(Value::Uint(value));
        Node::branch(name, Vec::new())
            .with(Node::leaf("texture_type", number(self.texture_type)))
            .with(Node::leaf("pixel_format", number(self.pixel_format)))
            .with(Node::leaf("width", number(self.width)))
            .with(Node::leaf("height", number(self.height)))
            .with(Node::leaf("depth", number(self.depth)))
            .with(Node::leaf(
                "mipmap_level_count",
                number(self.mipmap_level_count),
            ))
            .with(Node::leaf("array_length", number(self.array_length)))
            .with(Node::leaf("sample_count", number(self.sample_count)))
            .with(Node::leaf("usage", number(self.usage)))
            .with(Node::leaf("storage_mode", number(self.storage_mode)))
            .with(Node::leaf("cpu_cache_mode", number(self.cpu_cache_mode)))
            .with(Node::leaf(
                "hazard_tracking_mode",
                number(self.hazard_tracking_mode),
            ))
            .with(Node::leaf(
                "gpu_optimized",
                Observed::value(Value::Bool(self.gpu_optimized)),
            ))
    }
}
