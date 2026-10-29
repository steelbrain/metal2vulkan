//! The copy-and-fill vocabulary: a case's resources and the steps it encodes, as data.
//!
//! The five lanes this batch ports all say the same kind of thing — make a texture, write known
//! bytes into it, run one blit, read every byte back — so the vocabulary they need is one
//! declarative program rather than sixteen trait methods. A case states its [`Work`]; a side runs
//! it and answers what it read ([`Done`]). The case body stays a description of the probe, which
//! is what makes a port readable beside the lane it came from.
//!
//! A side with no public call for a step answers [`crate::case::Failure::unavailable`] for that
//! step. On the Metal side that is the **fill** and the fill alone: `fillTexture:` and
//! `fillTextureColor:` are Apple SPI (`metal_work`). The *formats* a case names are not
//! private — a pixel format is a raw value in `MTLTextureDescriptor`, and Metal makes, writes,
//! blits and reads back Apple's two-plane extended-range 550 and 551 through public calls like any
//! other. So a program that fills carries its Metal answer
//! ([`crate::case::MetalSource::Carried`]); a program of copies records on Metal.
//!
//! A program's resources are made before its steps run, and its steps may change that: a case that
//! asks what *freshly allocated* memory holds says [`Step::Release`] and then [`Step::Make`], so
//! the second allocation really happens after the first was freed ([`crate::cases::fresh_contents`]).
//! Without those two steps such a case would declare two resources that were both live at once and
//! would prove nothing about reused memory.
//!
//! Nothing here names a device, a vendor or an environment variable. A resource the venue cannot
//! make comes back as the runtime's typed refusal, and the suite reads the venue's honest gap out
//! of that refusal rather than out of a device name.

/// A rectangle of a texture: an origin and a size, in texels, with depth so a case can state a
/// region of no images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub origin: [u64; 3],
    pub size: [u64; 3],
}

impl Rect {
    /// A 2D rectangle of one image.
    pub const fn new(x: u64, y: u64, width: u64, height: u64) -> Self {
        Self {
            origin: [x, y, 0],
            size: [width, height, 1],
        }
    }

    /// A rectangle with every axis stated.
    pub const fn new_3d(origin: [u64; 3], size: [u64; 3]) -> Self {
        Self { origin, size }
    }
}

pub use crate::texture_query::Storage;

/// How a texture is shaped. The format is its raw `MTLPixelFormat`, because a case states the
/// probe's number and some of those numbers have no name in a public header. A number with no
/// public name is still a number Metal takes: `MTLTextureDescriptor`'s pixel format is a raw
/// value, so both sides make, write, blit and read such a texture through public calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSpec {
    /// The raw `MTLPixelFormat`.
    pub format: u32,
    pub kind: TextureKind,
    pub width: u64,
    pub height: u64,
    /// `depth`: a 3D texture's images. 1 for every other kind, as Metal's descriptor has it.
    pub depth: u64,
    pub levels: u64,
    /// `arrayLength`, which is 1 for a cube: Metal counts a cube's six faces as its slices and not
    /// as its array length, so a case over a cube reads six slices out of a spec whose
    /// `slices` is 1 ([`TextureSpec::faces`]).
    pub slices: u64,
    pub samples: u64,
    /// Where the texture's own memory lives ([`Storage`]).
    pub storage: Storage,
    /// Whether the texture is a render target as well as read and viewed.
    pub render_target: bool,
    /// `allowGPUOptimizedContents`: whether the GPU may keep the texture in a layout of its own.
    /// Metal's descriptor has it on, so a spec has it on, and a case that means a plain layout
    /// says so ([`TextureSpec::plain`]).
    pub gpu_optimized: bool,
    /// `usage`, as the raw `MTLTextureUsage` bits, where the case states it whole. `None` is the
    /// usage `render_target` implies: shader-read, and a render target that may be viewed in
    /// another format where it holds.
    ///
    /// A case states this where the usage is part of what it means: Metal refuses a depth texture
    /// of shader-write usage, and a render target that may be viewed is laid out differently from
    /// one that may not.
    pub usage: Option<u32>,
}

impl TextureSpec {
    /// A one-level 2D texture read by a shader.
    pub const fn texture_2d(format: u32, width: u64, height: u64) -> Self {
        Self {
            format,
            kind: TextureKind::Type2D,
            width,
            height,
            depth: 1,
            levels: 1,
            slices: 1,
            samples: 1,
            storage: Storage::Private,
            render_target: false,
            gpu_optimized: true,
            usage: None,
        }
    }

    /// The same texture in the raw `MTLTextureUsage` bits `usage`.
    #[must_use]
    pub const fn used(mut self, usage: u32) -> Self {
        self.usage = Some(usage);
        self
    }

    /// The same texture without `allowGPUOptimizedContents`.
    #[must_use]
    pub const fn plain(mut self) -> Self {
        self.gpu_optimized = false;
        self
    }

    /// The same texture with `levels` mip levels.
    #[must_use]
    pub const fn levels(mut self, levels: u64) -> Self {
        self.levels = levels;
        self
    }

    /// The same texture as a 2D array of `slices`.
    #[must_use]
    pub const fn array(mut self, slices: u64) -> Self {
        self.kind = TextureKind::Type2DArray;
        self.slices = slices;
        self
    }

    /// The same texture as a cube of `side` texels a face.
    ///
    /// Its `slices` stays 1, because that is `arrayLength` and a cube's array length is 1; its six
    /// faces are the slices a copy names ([`TextureSpec::faces`]).
    #[must_use]
    pub const fn cube(mut self, side: u64) -> Self {
        self.kind = TextureKind::Cube;
        self.width = side;
        self.height = side;
        self.slices = 1;
        self
    }

    /// The same texture as a 3D one of `depth` images.
    #[must_use]
    pub const fn volume(mut self, depth: u64) -> Self {
        self.kind = TextureKind::Type3D;
        self.depth = depth;
        self.slices = 1;
        self
    }

    /// The slices a copy may name: a cube's six faces, otherwise the array length.
    pub const fn faces(&self) -> u64 {
        match self.kind {
            TextureKind::Cube => 6,
            _ => self.slices,
        }
    }

    /// The same texture in shared storage.
    #[must_use]
    pub const fn shared(mut self) -> Self {
        self.storage = Storage::Shared;
        self
    }

    /// The same texture in managed storage: the device's own copy, read back through a copy into a
    /// shared resource rather than by a host call.
    #[must_use]
    pub const fn managed(mut self) -> Self {
        self.storage = Storage::Managed;
        self
    }

    /// The same texture in the storage mode `storage`, for a case that sweeps the modes.
    #[must_use]
    pub const fn stored(mut self, storage: Storage) -> Self {
        self.storage = storage;
        self
    }

    /// The same texture as a multisample one of `samples`, a render target that may be viewed in
    /// another format (what a per-sample read-back needs).
    #[must_use]
    pub const fn multisample(mut self, samples: u64, slices: u64) -> Self {
        self.kind = if slices > 1 {
            TextureKind::Type2DMultisampleArray
        } else {
            TextureKind::Type2DMultisample
        };
        self.samples = samples;
        self.slices = if slices > 1 { slices } else { 1 };
        self.render_target = true;
        self
    }
}

/// Which kind of texture a spec names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureKind {
    Type2D,
    Type2DArray,
    /// A cube of six faces, each `width` by `height`. Its faces are the slices a copy names, and
    /// its `arrayLength` is 1.
    Cube,
    /// A 3D texture of `depth` images, whose levels halve the depth as well as the width and the
    /// height.
    Type3D,
    Type2DMultisample,
    Type2DMultisampleArray,
    /// A texture over a buffer's bytes; its width is its texels and its height is 1.
    TextureBuffer,
}

/// The `MTLBlitOption` a copy takes: which plane of a combined format it moves, or PVRTC's
/// row-linear block order.
///
/// It is one enum rather than a bit set because Metal's own options are mutually exclusive in
/// practice — a depth plane and a PVRTC row order cannot both apply to one format — and a case
/// states the one it means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    /// `MTLBlitOptionNone`: every byte of the texture's own format, with no option.
    Whole,
    /// `MTLBlitOptionDepthFromDepthStencil`: the depth plane of a combined format.
    Depth,
    /// `MTLBlitOptionStencilFromDepthStencil`: the stencil plane of a combined format.
    Stencil,
    /// `MTLBlitOptionRowLinearPVRTC`: a PVRTC texture's blocks lie row by row, `bytes_per_row`
    /// apart, instead of in the block order Metal otherwise uses.
    RowLinearPvrtc,
}

impl Plane {
    /// The `MTLBlitOption` enumerator the plane names.
    ///
    /// One name for the fingerprint below and for the site a case records its read at, so the
    /// bytes a recording holds and the bytes it was keyed by carry the same spelling and the
    /// suite's option-space axis reads the option off the recording.
    pub fn option_name(&self) -> &'static str {
        match self {
            Plane::Whole => "MTLBlitOptionNone",
            Plane::Depth => "MTLBlitOptionDepthFromDepthStencil",
            Plane::Stencil => "MTLBlitOptionStencilFromDepthStencil",
            Plane::RowLinearPvrtc => "MTLBlitOptionRowLinearPVRTC",
        }
    }

    /// The words a case's input fingerprint covers: the `MTLBlitOption`'s own name, and **nothing
    /// at all** for [`Plane::Whole`].
    ///
    /// `Whole` is Metal's own `MTLBlitOptionNone`, so a copy at the default fingerprints to
    /// exactly the bytes it did before a plane could be stated. That is what keeps every
    /// recording already committed reading as fresh, and it is stated here, once, rather than in
    /// each case that states a plane: a fingerprint that missed a new field would make a stale
    /// recording read as a fresh one, which is worse than the field being absent.
    pub fn words(&self) -> Vec<u8> {
        match self {
            Plane::Whole => Vec::new(),
            named => named.option_name().as_bytes().to_vec(),
        }
    }
}

/// A resource a case makes. Each is named by its index in [`Work::resources`], so a step refers to
/// one by position and nothing in a transcript holds an object's identity.
#[derive(Clone, Debug, PartialEq)]
pub enum Resource {
    /// A buffer holding `bytes`, in shared storage so the host reads it back.
    Buffer { bytes: Vec<u8> },
    /// A buffer of `length` bytes that **nothing has written**, in the storage mode `storage`.
    ///
    /// Its own variant rather than a flag on [`Resource::Buffer`], because that one is made *with*
    /// its bytes: a case that passed it `vec![0; n]` and then claimed to read the zeros back would
    /// be reading what it had just written, which is a row that cannot fail. So a case that wants
    /// known bytes in such a buffer writes them with a [`Step::FillBuffer`] or a copy, and that
    /// write is part of what the case means.
    ///
    /// **What a fresh buffer holds is not a Metal promise.** A case that reads one unwritten is
    /// asking what this Metal and this runtime in fact do, which is a question worth recording and
    /// is what [`crate::cases::fresh_contents`] asks; it is never a guarantee a case may lean on
    /// elsewhere. Such a case gets its teeth from the order of its steps — fill one of this shape,
    /// [`Step::Release`] it, [`Step::Make`] another — rather than from an assumption.
    UnwrittenBuffer { length: u64, storage: Storage },
    /// A placement heap of `size` bytes.
    Heap { size: u64, storage: Storage },
    /// A buffer over `length` bytes of `heap` from `offset`.
    HeapBuffer {
        heap: usize,
        length: u64,
        offset: u64,
    },
    /// A texture of its own memory.
    Texture { spec: TextureSpec },
    /// A placement heap exactly as large as one texture of `spec` needs, as Metal sizes it: what
    /// a probe that placed one texture in a heap of Metal's size made.
    HeapForTexture { spec: TextureSpec, storage: Storage },
    /// A texture placed in `heap` at `offset`.
    HeapTexture {
        heap: usize,
        spec: TextureSpec,
        offset: u64,
    },
    /// A texture over `buffer`'s bytes from `offset`, its rows `bytes_per_row` apart.
    TextureOverBuffer {
        buffer: usize,
        spec: TextureSpec,
        offset: u64,
        bytes_per_row: u64,
    },
    /// `newFence`: a fence two of a command buffer's encoders order their work by
    /// ([`Step::UpdateFence`], [`Step::WaitForFence`]).
    ///
    /// It holds no memory, reaches no GPU on its own and is never read, so a program that makes
    /// one allocates nothing a device can fault on.
    Fence,
    /// A view of `texture` in `format`, over `level_count` levels from `level` and `slice_count`
    /// slices from `slice`, as a 2D multisample texture where `flatten`.
    ///
    /// The ranges are stated rather than assumed because the two families that view a texture mean
    /// opposite things by one: a per-sample read-back wants one slice of one level, while a
    /// compressed texture read through its sibling format wants every level and every slice, so
    /// that the view's own copies reach the whole chain.
    View {
        texture: usize,
        format: u32,
        level: u64,
        level_count: u64,
        slice: u64,
        slice_count: u64,
        flatten: bool,
    },
}

/// One step of a case's program.
///
/// The copies and fills are encoded into one blit encoder; [`Step::Commit`] closes it, commits the
/// command buffer and waits for it. A read is answered in the order the steps asked for it,
/// whether it is a host call or a read-back the device had to run first.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// `fillTexture:`'s private fill of `pattern`.
    FillTexture {
        texture: usize,
        level: u64,
        slice: u64,
        region: Rect,
        pattern: Vec<u8>,
    },
    /// `fillTexture:`'s private fill of a color packed in `packed_in`, or in the texture's own
    /// format where it is `None`.
    FillTextureColor {
        texture: usize,
        level: u64,
        slice: u64,
        region: Rect,
        color: [f64; 4],
        packed_in: Option<u32>,
    },
    CopyBufferToTexture {
        buffer: usize,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
        size: [u64; 3],
        texture: usize,
        slice: u64,
        level: u64,
        origin: [u64; 3],
        plane: Plane,
    },
    CopyTextureToBuffer {
        texture: usize,
        slice: u64,
        level: u64,
        origin: [u64; 3],
        size: [u64; 3],
        buffer: usize,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
        plane: Plane,
    },
    CopyTextureToTexture {
        from: usize,
        from_slice: u64,
        from_level: u64,
        from_origin: [u64; 3],
        size: [u64; 3],
        to: usize,
        to_slice: u64,
        to_level: u64,
        to_origin: [u64; 3],
    },
    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:
    /// destinationSlice:destinationLevel:destinationOrigin:` — the selector that takes **no**
    /// `MTLBlitOption`.
    ///
    /// Its own step rather than a [`Plane`] of [`Step::CopyBufferToTexture`], because Metal binds
    /// two selectors here and a case states which of them it means. Passing
    /// `MTLBlitOptionNone` to the optioned form is the same request written the other way, and
    /// whether the two agree is exactly what a case over this can ask.
    CopyBufferToTextureWithoutOptions {
        buffer: usize,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
        size: [u64; 3],
        texture: usize,
        slice: u64,
        level: u64,
        origin: [u64; 3],
    },
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:
    /// destinationOffset:destinationBytesPerRow:destinationBytesPerImage:` — the selector that
    /// takes **no** `MTLBlitOption`, the read-back twin of
    /// [`Step::CopyBufferToTextureWithoutOptions`].
    CopyTextureToBufferWithoutOptions {
        texture: usize,
        slice: u64,
        level: u64,
        origin: [u64; 3],
        size: [u64; 3],
        buffer: usize,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    },
    /// `copyFromTexture:toTexture:`: every level and slice the two textures share, with no region
    /// of either stated. Which levels Metal pairs, and which slices, is the whole subject.
    CopyWholeTexture { from: usize, to: usize },
    /// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:
    /// sliceCount:levelCount:`: whole levels and whole slices, by count rather than by region.
    ///
    /// A count of zero copies nothing wherever its slices and levels lie, which is a step a case
    /// may state on purpose.
    CopyTextureSlices {
        from: usize,
        from_slice: u64,
        from_level: u64,
        to: usize,
        to_slice: u64,
        to_level: u64,
        slice_count: u64,
        level_count: u64,
    },
    /// `generateMipmapsForTexture:`: the levels under level 0, each filtered from the one above.
    ///
    /// The one step of this vocabulary whose answer is a **filter's** rather than a move's, so a
    /// case over it is kept apart from the cases that move bytes: an implementation whose filter
    /// rounds otherwise diverges here and nowhere else.
    GenerateMipmaps { texture: usize },
    /// `optimizeContentsForGPUAccess:`, or `optimizeContentsForGPUAccess:slice:level:` where `at`
    /// names a slice and a level.
    ///
    /// One step for the two selectors, because they are one request of a case — let the
    /// implementation lay these texels out for the GPU — differing only in how much of the texture
    /// it names.
    OptimizeForGpuAccess {
        texture: usize,
        at: Option<(u64, u64)>,
    },
    /// `optimizeContentsForCPUAccess:`, or `optimizeContentsForCPUAccess:slice:level:` where `at`
    /// names a slice and a level.
    OptimizeForCpuAccess {
        texture: usize,
        at: Option<(u64, u64)>,
    },
    /// `synchronizeResource:`: the host sees what the GPU wrote into a `Managed` resource once the
    /// command buffer completes.
    ///
    /// It takes a buffer or a texture, as Metal's own `id<MTLResource>` does.
    SynchronizeResource { resource: usize },
    /// `synchronizeTexture:slice:level:`: as [`Step::SynchronizeResource`], for one slice of one
    /// level of a `Managed` texture.
    SynchronizeTexture {
        texture: usize,
        slice: u64,
        level: u64,
    },
    CopyBufferToBuffer {
        from: usize,
        from_offset: u64,
        to: usize,
        to_offset: u64,
        length: u64,
    },
    /// `fillBuffer:range:value:`: every byte of `[offset, offset + length)` takes `value`.
    ///
    /// A fill of no bytes is a step of its own and not nothing: plain Metal skips it wherever its
    /// range lies, and whether we do is a case.
    FillBuffer {
        buffer: usize,
        offset: u64,
        length: u64,
        value: u8,
    },
    /// A render pass over one slice that clears it and draws nothing: how Metal writes a known
    /// value into a multisample texture, which no CPU copy reaches.
    ClearPass {
        texture: usize,
        slice: u64,
        clear: Clear,
    },
    /// `replaceRegion:`: the host writes `bytes` into the texture.
    ReplaceRegion {
        texture: usize,
        level: u64,
        slice: u64,
        region: Rect,
        bytes: Vec<u8>,
        /// The rows of bytes the call hands, which for a block-compressed format are the region's
        /// **block** rows and not its texel rows.
        ///
        /// A case states it because neither side can derive it: `MTLTexture` answers no block
        /// extent, and the difference is the whole subject of the compressed families. An
        /// uncompressed case states the region's height, which is what it always meant.
        rows: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    },
    /// `getBytes:`: the host reads the region into `length` bytes, which answers a read.
    GetBytes {
        texture: usize,
        level: u64,
        slice: u64,
        region: Rect,
        bytes_per_row: u64,
        bytes_per_image: u64,
        length: usize,
        /// The byte every one of those `length` bytes holds before the call.
        ///
        /// A case that reads at a pitch wider than its rows asks two questions at once — the bytes
        /// the call wrote, and whether it left the padding between the rows alone — and the second
        /// needs a fill the call would never write. Zero is a poor one, because a call that wrote
        /// zeros would pass; a case that means the question states a distinctive byte.
        before: u8,
    },
    /// A read of a buffer's bytes, which answers a read.
    ReadBuffer {
        buffer: usize,
        offset: u64,
        length: usize,
    },
    /// Every sample of a multisample texture, read by a dispatch of the committed fixture's
    /// kernel, one thread a sample: the texels in rows, each texel's samples in order, `texel_bytes`
    /// each. It answers a read.
    ReadSamples {
        texture: usize,
        kernel: &'static str,
        width: u32,
        height: u32,
        samples: u32,
        /// The bytes of one sample the read keeps: the kernel writes whole words, and a format of
        /// fewer bytes keeps its own.
        texel_bytes: usize,
        /// The words the kernel writes per sample.
        words: usize,
    },
    /// Release the resource at this index: the side drops its reference to it, and a later step
    /// naming it is refused until a [`Step::Make`] makes it again.
    ///
    /// This is the teeth of a case about what a new resource holds. A program's resources are made
    /// before its steps run, so a case cannot ask "what does memory that was just freed read as?"
    /// by declaring two resources: the second was allocated while the first was still live. It has
    /// to say *release this one, now make another*, and this is the first half of saying it.
    ///
    /// **It is refused while a command buffer is open**, because releasing a resource an
    /// uncommitted command buffer refers to is a program whose own order is unclear, and because
    /// the release only frees anything once the work that used it has completed. A case therefore
    /// states a [`Step::Commit`] before it releases, and the refusal says so rather than the case
    /// silently proving less than it reads.
    Release { resource: usize },
    /// Make the resource at this index again, from its own declaration in [`Work::resources`].
    ///
    /// The other half of the teeth: made **after** the [`Step::Release`] of the resource it
    /// replaces, so a side that handed the freed memory back as it lies would read the old bytes.
    /// A resource that is still made is refused rather than replaced, so a program cannot leak one
    /// by mistake.
    ///
    /// A resource made over another — a texture over a buffer, a buffer or texture in a heap —
    /// reads its own declaration again, so a case remakes the thing underneath first and then what
    /// stands on it, in that order.
    Make { resource: usize },
    /// `endEncoding` of the blit encoder, **without** committing: the steps after it open a
    /// second encoder on the same command buffer.
    ///
    /// A fence orders one encoder's work against another's, so a case over
    /// [`Step::UpdateFence`] cannot be written at all while a program has one encoder a command
    /// buffer. [`Step::Commit`] is not that step: it closes the command buffer too, and two
    /// command buffers is a different program — the queue orders them anyway, so a fence between
    /// them would prove nothing.
    EndEncoder,
    /// `updateFence:` of the blit encoder: the fence is updated once this encoder's commands are
    /// done.
    UpdateFence { fence: usize },
    /// `waitForFence:` of the blit encoder: this encoder's commands wait for the fence's last
    /// update before them in the queue's commit order.
    WaitForFence { fence: usize },
    /// Close the blit encoder, commit the command buffer and wait for it: its status is recorded
    /// and every read-back the steps before it asked for is answered.
    Commit,
}

/// What a [`Step::ClearPass`] clears.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Clear {
    Color([f64; 4]),
    Depth(f64),
    Stencil(u32),
    /// A combined depth and stencil attachment, both cleared in one pass.
    DepthStencil(f64, u32),
}

/// A case's program: the resources it makes, in order, and the steps it runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Work {
    pub resources: Vec<Resource>,
    pub steps: Vec<Step>,
    /// The fixture whose kernels a [`Step::ReadSamples`] runs, where the program holds one.
    pub kernels: Option<&'static str>,
}

impl Work {
    /// An empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `resource`, answering its index.
    pub fn add(&mut self, resource: Resource) -> usize {
        self.resources.push(resource);
        self.resources.len() - 1
    }

    /// Add `step`.
    pub fn step(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// Add a buffer holding `bytes`, answering its index.
    pub fn buffer(&mut self, bytes: Vec<u8>) -> usize {
        self.add(Resource::Buffer { bytes })
    }

    /// Add a buffer of `length` bytes in `storage` that nothing has written, answering its index.
    pub fn unwritten(&mut self, length: u64, storage: Storage) -> usize {
        self.add(Resource::UnwrittenBuffer { length, storage })
    }

    /// How many reads the program asks for: a step that answers one.
    pub fn reads(&self) -> usize {
        self.steps.iter().filter(|step| step.reads()).count()
    }
}

impl Step {
    /// Whether the step answers a read.
    pub fn reads(&self) -> bool {
        matches!(
            self,
            Step::GetBytes { .. } | Step::ReadBuffer { .. } | Step::ReadSamples { .. }
        )
    }
}

/// What running a [`Work`] answered: the reads, in the order the steps asked for them, and each
/// command buffer's status, in commit order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Done {
    pub reads: Vec<Vec<u8>>,
    pub statuses: Vec<String>,
}

impl Done {
    /// Read `index`, or an error naming what the program asked for.
    pub fn read(&self, index: usize) -> Result<&[u8], String> {
        self.reads.get(index).map(Vec::as_slice).ok_or_else(|| {
            format!(
                "the run answered {} reads, not {}",
                self.reads.len(),
                index + 1
            )
        })
    }
}
