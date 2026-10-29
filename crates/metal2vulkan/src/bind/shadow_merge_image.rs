//! The merge of a shadowed texture's writes into its buffer's bytes, reading the shadow itself.
//!
//! [`shadow_merge_module`](super::shadow_merge_module) compares a texture's texels kept before a
//! command with the same texels copied out of its shadow after it. [`shadow_merge_image_module`]
//! reads the texels after the command from the shadow instead, so no copy of them is made, through
//! one of two kinds of view ([`ShadowMergeRead`]):
//!
//! - **An unsigned-integer view** of the texel's width ([`ShadowMergeRead::Words`]): Vulkan
//!   states that "Views of compatible formats will have the same mapping between texel
//!   coordinates and memory locations irrespective of the format, with only the interpretation of
//!   the bit pattern changing" (`VkImageViewCreateInfo`), and an unsigned-integer view returns its
//!   components' bits unconverted, so the words a fetch returns are the words a copy of the texel
//!   writes (a 4-byte texel's `R8G8B8A8_UINT` bytes packed into its word, as
//!   [`shadow_copy_back_module`](super::shadow_copy_back_module) packs them).
//! - **Its own view of an 8-bit unsigned normalized format** ([`ShadowMergeRead::B8G8R8A8Unorm`],
//!   [`ShadowMergeRead::R8G8B8A8Unorm`]), which needs no other format on the image. Vulkan
//!   defines "the conversion from an unsigned normalized fixed-point value c to the corresponding
//!   floating-point value f" as `f = c / (2^b - 1)` ("Fixed-Point Data Conversions"), here
//!   `c / 255`, and `OpImageFetch` takes no sampler and filters nothing. The kernel rebuilds each
//!   component's byte as `RoundEven(f · 255)` (`OpFMul` correctly rounded, `RoundEven` and
//!   `OpConvertFToU` of a whole number exact): the `f32` nearest `c / 255` lies within `2^-25` of
//!   it (`c / 255` is at most 1), so `f · 255` lies within `255 · 2^-25 + 2^-17`, below `2^-16`,
//!   of `c`, and even a conversion only as accurate as Vulkan's floor for floating-point
//!   computation ("about 1 part in 10^5", "Floating-Point Computation") leaves it within 0.003 of
//!   `c`: the rounding returns `c`, for every byte. The bytes go into the texel's word in memory
//!   order, byte `i` at bits `8i` onward: `B8G8R8A8_UNORM` holds B in byte 0, G in 1, R in 2 and A
//!   in 3, and `R8G8B8A8_UNORM` R, G, B and A, so the word is the one a copy of the texel writes.
//!
//! **The module** (one per read, [`ShadowMergeRead`]). One `GLCompute` entry `main` with local
//! size ([`ShadowMergeImageModule::LOCAL_SIZE`], 1, 1), [`ShadowMergeImageModule::PUSH_BYTES`] of
//! `COMPUTE` push constants at offset 0 (the region's shape, [`ShadowMergeImageModule::shape`]),
//! and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`ShadowMergeImageModule::IMAGE_BINDING`] | the shadow's sampled view of the module's format ([`ShadowMergeRead`]), in the `GENERAL` layout |
//! | [`ShadowMergeImageModule::KEPT_BINDING`] | the region's texels before the command, row after row, a storage buffer |
//! | [`ShadowMergeImageModule::CURRENT_BINDING`] | the bytes the region lies in, a storage buffer merged in place |
//!
//! The shape's words, in order: `origin`, the region's first texel `x0` in its low 16 bits and
//! `y0` in its high 16; `width`, its texels a row; `row_words`, the current buffer's rows apart in
//! words; and `first_word`, the word of the region's first texel in the current binding.
//! A row of the region is `width · n` words, `n` the module's texel width in 4-byte words.
//! Invocation `l` of workgroup `(g, y)` takes the row's words `w = 4Ng + Nk + l` for `k` from 0 to
//! 3 ([`ShadowMergeImageModule::WORDS`], `N` [`ShadowMergeImageModule::LOCAL_SIZE`]), one after
//! the other. Each `w` below `width · n` fetches texel `(x0 + w / n, y0 + y)` of level 0, takes
//! its word `w % n`, and compares it with word `y · width · n + w` of the kept buffer; where they
//! differ, word `first_word + y · row_words + w` of the current buffer takes, in each byte where
//! they differ, the fetched word's byte, and keeps its own in the others. Where they are equal it
//! reads and writes nothing of the current buffer; a `w` past the row's words does nothing. A
//! dispatch takes `ceil(width · n / (LOCAL_SIZE · WORDS))` by the region's height workgroups
//! ([`ShadowMergeImageModule::groups`]). No word between the region's rows or past its texels is
//! read or written.
//!
//! Words across the invocations, not texels: as in
//! [`shadow_merge_module`](super::shadow_merge_module) each word's merge
//! reads and writes that word alone, so which invocation takes which word is free, and `4Ng + Nk
//! + l` puts the `k`th store of consecutive invocations at consecutive words. Spreading texels
//! instead would leave `4n - 1` words between one store's lanes, which on an RTX 5080 Laptop GPU
//! costs hundreds of times the time over imported host memory (the measurements in that
//! module's docs). A texel wider than a word is fetched once per word of it,
//! `n` lanes fetching the one texel, where loads of every shape on that device cost the same.
//!
//! **Kept in an image** ([`shadow_merge_kept_image_module`]): the same module, but for its kept
//! binding, a sampled image of the module's unsigned-integer format ([`ShadowTexelView`]) in the
//! `GENERAL` layout, of at least the shadow's extent, which holds the region's texels before the
//! command at their own coordinates (a copy of the shadow's region into it, `vkCmdCopyImage`).
//! Each texel `x` fetches texel `(x0 + x, y0 + y)` of both images and compares their words as
//! above. Both views being of one format, the kept words are the shadow's bits at the copy:
//! "Copies are done layer-by-layer starting with the baseArrayLayer member of srcSubresource ...
//! the texel data is copied" bit for bit between compatible formats (`vkCmdCopyImage`).
//!
//! **Synchronization** (consumer): the shadow's last writes → `SHADER_SAMPLED_READ` in
//! `COMPUTE_SHADER`, the kept buffer's copy → `SHADER_STORAGE_READ` (the kept image's copy →
//! `SHADER_SAMPLED_READ`), and the bytes' earlier accesses → `SHADER_STORAGE_READ` and
//! `SHADER_STORAGE_WRITE` there, before the dispatch; its writes → the bytes' next use.
//!
//! **Whole blocks** (`ShadowMergeStores::Blocks`): as `bind::shadow_merge`'s module docs say,
//! the `LOCAL_SIZE` words of a row a workgroup takes in one step are one block, marked in
//! workgroup memory where a word of it changed, and every word of a marked block is stored, none
//! of an unmarked one. A word past the row reads the row's first texel, and counts as unchanged.

use std::sync::Arc;

use super::shadow_copy_back::{fetched_words, ShadowTexelView};
use super::shadow_merge::{
    barrier, block_flags, merge_word, u32_value, words_buffer, Blocks, ShadowMergeStores,
};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, GlslOp, ImageOperands, ImageType,
    ImageUsage, IoDir, MemAccess, StorageClass, Type, TypedModule,
};

/// How [`shadow_merge_image_module`] reads the shadow's texels after the command (the module
/// docs): the view its image binding is, and how the kernel takes a texel's words from a fetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShadowMergeRead {
    /// The shadow's unsigned-integer view of its texel's width: the texel's bits, unconverted.
    Words(ShadowTexelView),
    /// The shadow's own `B8G8R8A8_UNORM` view: each component's byte rebuilt from its value,
    /// B, G, R and A in memory order.
    B8G8R8A8Unorm,
    /// The shadow's own `R8G8B8A8_UNORM` view: each component's byte rebuilt from its value,
    /// R, G, B and A in memory order.
    R8G8B8A8Unorm,
}

impl ShadowMergeRead {
    /// Every read.
    pub const ALL: [ShadowMergeRead; 6] = [
        ShadowMergeRead::Words(ShadowTexelView::R32Uint),
        ShadowMergeRead::Words(ShadowTexelView::R8G8B8A8Uint),
        ShadowMergeRead::Words(ShadowTexelView::R32G32Uint),
        ShadowMergeRead::Words(ShadowTexelView::R32G32B32A32Uint),
        ShadowMergeRead::B8G8R8A8Unorm,
        ShadowMergeRead::R8G8B8A8Unorm,
    ];

    /// The texel's 4-byte words: 1, 2 or 4.
    pub fn words(self) -> u32 {
        match self {
            ShadowMergeRead::Words(view) => view.words(),
            ShadowMergeRead::B8G8R8A8Unorm | ShadowMergeRead::R8G8B8A8Unorm => 1,
        }
    }
}

/// The crate-owned image-reading merge module of one read on one device, validated like any
/// output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowMergeImageModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl ShadowMergeImageModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The shadow's view of the module's read, a sampled image.
    pub const IMAGE_BINDING: u32 = 0;
    /// The region's texels before the command: a storage buffer, row after row, or (the
    /// [`shadow_merge_kept_image_module`] form) a sampled image holding them at their own
    /// coordinates.
    pub const KEPT_BINDING: u32 = 1;
    /// The buffer's bytes after the command, a storage buffer the dispatch merges into.
    pub const CURRENT_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: [`WORDS`](Self::WORDS) words each, of one row.
    pub const LOCAL_SIZE: u32 = 64;
    /// The words one invocation merges, one after the other.
    pub const WORDS: u32 = 4;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the region's shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of a region whose first texel is `[x0, y0]` (each below 65536), `width`
    /// texels a row, whose rows lie `row_words` apart in the current binding from its word
    /// `first_word` (the module docs).
    pub fn shape([x0, y0]: [u32; 2], width: u32, row_words: u32, first_word: u32) -> [u8; 16] {
        let words = [(x0 & 0xffff) | (y0 << 16), width, row_words, first_word];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The workgroups of a dispatch of `read`'s module over a region `[width, height]` texels:
    /// its rows are `width` times `read`'s words each.
    pub fn groups(read: ShadowMergeRead, [width, height]: [u32; 2]) -> [u32; 2] {
        let words = width.saturating_mul(read.words());
        [words.div_ceil(Self::LOCAL_SIZE * Self::WORDS), height]
    }
}

/// The binding symbol of the shadow's view in the typed module.
const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the kept buffer in the typed module.
const KEPT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the kept image in the typed module of the kept-image form.
const KEPT_IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 1,
};
/// The binding symbol of the current buffer in the typed module.
const CURRENT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

/// Where the merge reads the texels before the command (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kept {
    /// A storage buffer, row after row: the symbol `KEPT`.
    Buffer,
    /// A sampled image of an unsigned-integer view, at their own coordinates: the symbol
    /// `KEPT_IMAGE`.
    Image(ShadowTexelView),
}

/// What a fetch through a read's view returns (the module docs).
#[derive(Clone, Copy)]
enum Texel {
    /// An unsigned-integer view's components, its bits.
    Bits(ShadowTexelView),
    /// An 8-bit unsigned normalized view's components, whose bytes lie in memory in this order
    /// of components, byte 0 first.
    Unorm([u32; 4]),
}

/// What a fetch through `read`'s view returns.
fn texel_of(read: ShadowMergeRead) -> Texel {
    match read {
        ShadowMergeRead::Words(view) => Texel::Bits(view),
        ShadowMergeRead::B8G8R8A8Unorm => Texel::Unorm([2, 1, 0, 3]),
        ShadowMergeRead::R8G8B8A8Unorm => Texel::Unorm([0, 1, 2, 3]),
    }
}

/// How the kernel takes a fetched texel's words (the module docs).
#[derive(Clone, Copy)]
enum Rebuild {
    /// An unsigned-integer view's texel, a `uvec4`: its bits.
    Words(ShadowTexelView),
    /// An 8-bit unsigned normalized view's texel, a `vec4` of `f32`: each component times
    /// `scale` (`vec4(255)`), rounded into a `uvec4` (the type `uvec4`), its components taken in
    /// memory `order`.
    Unorm {
        scale: Value,
        uvec4: TypeId,
        order: [u32; 4],
    },
}

impl Rebuild {
    /// The words of a texel a fetch through the shadow's view returned (`texel`), as a copy of
    /// the texel writes them (the module docs).
    fn words(self, f: &mut FunctionBuilder<'_>, texel: Value) -> BResult<Vec<Value>> {
        match self {
            Rebuild::Words(view) => fetched_words(f, texel, view),
            Rebuild::Unorm {
                scale,
                uvec4,
                order,
            } => {
                // c = RoundEven(f · 255), component by component, then the bytes in memory
                // order, packed as an `R8G8B8A8_UINT` view's are.
                let scaled = f.binary(BinOp::FMul, texel, scale)?;
                let rounded = f.glsl(GlslOp::RoundEven, &[scaled])?;
                let bytes = f.convert(ConvOp::ConvertFToU, rounded, uvec4)?;
                let bytes = match order {
                    [0, 1, 2, 3] => bytes,
                    order => f.vector_shuffle(bytes, bytes, &order)?,
                };
                fetched_words(f, bytes, ShadowTexelView::R8G8B8A8Uint)
            }
        }
    }
}

/// The merge module of `read` reading the kept texels from `kept` as a typed module, before
/// serialization: the module [`shadow_merge_image_module`] (or, the kept texels in an image,
/// [`shadow_merge_kept_image_module`]) serializes, which the CPU interpreter runs. Its bindings
/// are the symbols `IMAGE`, `KEPT` (or `KEPT_IMAGE`) and `CURRENT`, and its push field the shape,
/// a `uvec4` (the module docs).
fn typed_module(read: ShadowMergeRead, kept: Kept, blocks: bool) -> BResult<TypedModule> {
    let words = read.words();
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    // An unsigned-integer view's texels are `uvec4`s, an unsigned normalized view's `vec4`s.
    let texel = texel_of(read);
    let sampled_type = match texel {
        Texel::Bits(_) => u32_,
        Texel::Unorm(_) => b.ty(Type::Float { width: 32 })?,
    };
    let image_type = |b: &mut ModuleBuilder, sampled_type| {
        b.ty(Type::Image(ImageType {
            sampled_type,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage: ImageUsage::Sampled,
            format: crate::spv::ImageFormat::Unknown,
        }))
    };
    let shadow_type = image_type(&mut b, sampled_type)?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, shadow_type)?;
    let kept = match kept {
        Kept::Buffer => KeptAt::Buffer(words_buffer(&mut b, KEPT, u32_)?),
        Kept::Image(view) => {
            let kept_type = image_type(&mut b, u32_)?;
            let kept = b.resource(KEPT_IMAGE, StorageClass::UniformConstant, kept_type)?;
            KeptAt::Image(kept, view)
        }
    };
    let current = words_buffer(&mut b, CURRENT, u32_)?;
    let block_flags = match blocks {
        true => Some(block_flags(&mut b, u32_, ShadowMergeImageModule::WORDS)?),
        false => None,
    };
    let group = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let local = b.builtin(crate::spv::BuiltIn::LocalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let low = u32_value(&mut f, 0xffff)?;
    let sixteen = u32_value(&mut f, 16)?;
    let mut lanes = Vec::with_capacity(words as usize);
    for lane in 0..words {
        lanes.push(u32_value(&mut f, lane)?);
    }
    let rebuild = match texel {
        Texel::Bits(view) => Rebuild::Words(view),
        Texel::Unorm(order) => {
            let m = f.module();
            let vec4 = m.ty(Type::Vector {
                elem: sampled_type,
                len: 4,
            })?;
            let full = m.constant(Constant::Scalar {
                ty: sampled_type,
                bits: ConstBits(u64::from(255f32.to_bits())),
            })?;
            let scale = m.constant(Constant::Composite {
                ty: vec4,
                parts: [full; 4].into(),
            })?;
            Rebuild::Unorm {
                scale: m.const_value(scale)?,
                uvec4,
                order,
            }
        }
    };
    let group = f.module().global_value(group)?;
    let group = f.load(group, MemAccess::default())?;
    let g = f.composite_extract(group, &[0])?;
    // The region's row: the workgroup's y, the local size along y being 1.
    let y = f.composite_extract(group, &[1])?;
    let local = f.module().global_value(local)?;
    let local = f.load(local, MemAccess::default())?;
    let l = f.composite_extract(local, &[0])?;
    let image = f.module().global_value(image)?;
    let kept = match kept {
        KeptAt::Buffer(kept) => KeptAt::Buffer(f.module().global_value(kept)?),
        KeptAt::Image(kept, view) => KeptAt::Image(f.module().global_value(kept)?, view),
    };
    let current = f.module().global_value(current)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let origin = f.composite_extract(shape, &[0])?;
    let width = f.composite_extract(shape, &[1])?;
    let row_words = f.composite_extract(shape, &[2])?;
    let first_word = f.composite_extract(shape, &[3])?;
    let x0 = f.binary(BinOp::BitwiseAnd, origin, low)?;
    let y0 = f.binary(BinOp::ShiftRightLogical, origin, sixteen)?;
    let ty = f.binary(BinOp::IAdd, y0, y)?;
    // The row's first word in the current buffer, and the row's words: `width` texels of `words`
    // words each.
    let row = f.binary(BinOp::IMul, y, row_words)?;
    let row = f.binary(BinOp::IAdd, first_word, row)?;
    let row_count = match words {
        1 => width,
        words => {
            let n = u32_value(&mut f, words)?;
            f.binary(BinOp::IMul, width, n)?
        }
    };
    // A row word's texel and its word of that texel, where a texel is more than one word (1, 2
    // or 4, so a shift and a mask).
    let split = match words {
        1 => None,
        words => Some((
            u32_value(&mut f, words.trailing_zeros())?,
            u32_value(&mut f, words - 1)?,
        )),
    };
    // The workgroup's first word of the row, plus the invocation's place in it: its words lie
    // `LOCAL_SIZE` apart from there, so each of the four stores below falls on consecutive words
    // across consecutive invocations (the module docs).
    let span = ShadowMergeImageModule::WORDS * ShadowMergeImageModule::LOCAL_SIZE;
    let span = u32_value(&mut f, span)?;
    let base = f.binary(BinOp::IMul, g, span)?;
    let base = f.binary(BinOp::IAdd, base, l)?;
    if let Some(block_flags) = block_flags {
        // Whole blocks: each step's words, the workgroup's `LOCAL_SIZE` consecutive words of the
        // row, are one block, stored whole where any of them changed (`bind::shadow_merge`'s
        // module docs).
        let flags = Blocks::new(&mut f, block_flags, l)?;
        let mut taken = Vec::new();
        for k in 0..ShadowMergeImageModule::WORDS {
            let w = match k {
                0 => base,
                k => {
                    let step = u32_value(&mut f, k * ShadowMergeImageModule::LOCAL_SIZE)?;
                    f.binary(BinOp::IAdd, base, step)?
                }
            };
            let inside = f.binary(BinOp::ULessThan, w, row_count)?;
            // A word past the row reads the row's first, and counts as unchanged.
            let read = f.select(inside, w, zero)?;
            let (x, lane) = match split {
                None => (read, None),
                Some((shift, mask)) => (
                    f.binary(BinOp::ShiftRightLogical, read, shift)?,
                    Some(f.binary(BinOp::BitwiseAnd, read, mask)?),
                ),
            };
            let tx = f.binary(BinOp::IAdd, x0, x)?;
            let coord = f.composite_construct(uvec2, &[tx, ty])?;
            let loaded = f.load(image, MemAccess::default())?;
            let ops = ImageOperands {
                lod: Some(zero),
                ..ImageOperands::default()
            };
            let texel = f.fetch(loaded, coord, &ops)?;
            let texel = rebuild.words(&mut f, texel)?;
            let after = one_word(&mut f, &texel, lane, &lanes)?;
            let before = match kept {
                KeptAt::Image(kept, view) => {
                    let kept = f.load(kept, MemAccess::default())?;
                    let texel = f.fetch(kept, coord, &ops)?;
                    let words = fetched_words(&mut f, texel, view)?;
                    one_word(&mut f, &words, lane, &lanes)?
                }
                KeptAt::Buffer(kept) => {
                    let kept_row = f.binary(BinOp::IMul, y, row_count)?;
                    let place = f.binary(BinOp::IAdd, kept_row, read)?;
                    let p = f.access_chain(kept, &[zero, place])?;
                    f.load(p, MemAccess::default())?
                }
            };
            let differ = f.binary(BinOp::BitwiseXor, before, after)?;
            let differ = f.select(inside, differ, zero)?;
            flags.mark(&mut f, k, differ)?;
            taken.push((w, inside, after, differ));
        }
        barrier(&mut f)?;
        for (k, (w, inside, after, differ)) in (0..).zip(taken) {
            let store = flags.stores(&mut f, k, inside)?;
            let write: ArmFn<'_, ()> = Box::new(move |f, _| {
                let place = f.binary(BinOp::IAdd, row, w)?;
                let to = f.access_chain(current, &[zero, place])?;
                merge_word(f, to, after, differ)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), store, write, None)?;
        }
    } else {
        // The invocation's words, one after the other, each as one invocation took it alone.
        for k in 0..ShadowMergeImageModule::WORDS {
            let w = match k {
                0 => base,
                k => {
                    let step = u32_value(&mut f, k * ShadowMergeImageModule::LOCAL_SIZE)?;
                    f.binary(BinOp::IAdd, base, step)?
                }
            };
            let inside = f.binary(BinOp::ULessThan, w, row_count)?;
            let lanes = lanes.clone();
            let merge: ArmFn<'_, ()> = Box::new(move |f, _| {
                let (x, lane) = match split {
                    None => (w, None),
                    Some((shift, mask)) => (
                        f.binary(BinOp::ShiftRightLogical, w, shift)?,
                        Some(f.binary(BinOp::BitwiseAnd, w, mask)?),
                    ),
                };
                let tx = f.binary(BinOp::IAdd, x0, x)?;
                let coord = f.composite_construct(uvec2, &[tx, ty])?;
                let image = f.load(image, MemAccess::default())?;
                let ops = ImageOperands {
                    lod: Some(zero),
                    ..ImageOperands::default()
                };
                let texel = f.fetch(image, coord, &ops)?;
                let texel = rebuild.words(f, texel)?;
                let after = one_word(f, &texel, lane, &lanes)?;
                // The word before the command: in the kept image, the texel's at its own
                // coordinates; in the kept buffer, the row's word, row after row.
                let before = match kept {
                    KeptAt::Image(kept, view) => {
                        let kept = f.load(kept, MemAccess::default())?;
                        let texel = f.fetch(kept, coord, &ops)?;
                        let words = fetched_words(f, texel, view)?;
                        one_word(f, &words, lane, &lanes)?
                    }
                    KeptAt::Buffer(kept) => {
                        let kept_row = f.binary(BinOp::IMul, y, row_count)?;
                        let place = f.binary(BinOp::IAdd, kept_row, w)?;
                        let p = f.access_chain(kept, &[zero, place])?;
                        f.load(p, MemAccess::default())?
                    }
                };
                let differ = f.binary(BinOp::BitwiseXor, before, after)?;
                let moved = f.binary(BinOp::INotEqual, differ, zero)?;
                // A word the command left as it was is neither read nor written.
                let write: ArmFn<'_, ()> = Box::new(move |f, _| {
                    // The word at its row's place in the current buffer.
                    let place = f.binary(BinOp::IAdd, row, w)?;
                    let to = f.access_chain(current, &[zero, place])?;
                    merge_word(f, to, after, differ)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                f.if_(&mut (), moved, write, None)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), inside, merge, None)?;
        }
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: ShadowMergeImageModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The word a row word took of its texel's `words`: the texel's only word where it is one word
/// (`lane` `None`), else the one `lane` names, by a chain of selects over the texel's words
/// (`lanes` holding each word's index).
fn one_word(
    f: &mut FunctionBuilder<'_>,
    words: &[Value],
    lane: Option<Value>,
    lanes: &[Value],
) -> BResult<Value> {
    let Some(lane) = lane else {
        return Ok(words[0]);
    };
    let mut chosen = words[0];
    for (&word, &index) in words.iter().zip(lanes).skip(1) {
        let is = f.binary(BinOp::IEqual, lane, index)?;
        chosen = f.select(is, word, chosen)?;
    }
    Ok(chosen)
}

/// A kept binding: the buffer's or the image's global (or, inside the function, value), the
/// image's with its view.
#[derive(Clone, Copy)]
enum KeptAt<T> {
    Buffer(T),
    Image(T, ShadowTexelView),
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ShadowMergeImageModule::SET),
        Symbol::Binding(s) if *s == IMAGE => Some(ShadowMergeImageModule::IMAGE_BINDING),
        Symbol::Binding(s) if *s == KEPT || *s == KEPT_IMAGE => {
            Some(ShadowMergeImageModule::KEPT_BINDING)
        }
        Symbol::Binding(s) if *s == CURRENT => Some(ShadowMergeImageModule::CURRENT_BINDING),
        _ => None,
    }
}

/// The image-reading merge module of `device` reading the shadow's texels as `read` does and
/// storing as `stores` says:
/// built with the typed builder, checked by the owned verifier against the device's profile
/// (requirements included), serialized, and numbered with the fixed bindings of
/// [`ShadowMergeImageModule`].
pub fn shadow_merge_image_module(
    device: &Device,
    read: ShadowMergeRead,
    stores: ShadowMergeStores,
) -> Result<ShadowMergeImageModule, Error> {
    module_of(device, read, Kept::Buffer, stores)
}

/// The image-reading merge module of `device` reading texels through the unsigned-integer
/// `view`, the kept texels a sampled image of `view`'s format holding them at their own
/// coordinates (the module docs): as [`shadow_merge_image_module`] of
/// [`ShadowMergeRead::Words`], its [`ShadowMergeImageModule::KEPT_BINDING`] that image.
pub fn shadow_merge_kept_image_module(
    device: &Device,
    view: ShadowTexelView,
    stores: ShadowMergeStores,
) -> Result<ShadowMergeImageModule, Error> {
    module_of(
        device,
        ShadowMergeRead::Words(view),
        Kept::Image(view),
        stores,
    )
}

/// [`shadow_merge_image_module`] or [`shadow_merge_kept_image_module`], by `kept`.
fn module_of(
    device: &Device,
    read: ShadowMergeRead,
    kept: Kept,
    stores: ShadowMergeStores,
) -> Result<ShadowMergeImageModule, Error> {
    let profile = device.profile();
    let m = typed_module(read, kept, stores == ShadowMergeStores::Blocks)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the image-reading shadow merge needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(ShadowMergeImageModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        texel_of, typed_module, Kept, ShadowMergeImageModule, ShadowMergeRead, ShadowTexelView,
        Texel, CURRENT, IMAGE, KEPT, KEPT_IMAGE,
    };
    use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// Every form of the module: each read with its kept texels in a buffer, and each
    /// unsigned-integer view's with them in an image too.
    fn forms() -> Vec<(ShadowMergeRead, Kept)> {
        let mut forms: Vec<_> = (ShadowMergeRead::ALL.into_iter())
            .map(|read| (read, Kept::Buffer))
            .collect();
        for view in ShadowTexelView::ALL {
            forms.push((ShadowMergeRead::Words(view), Kept::Image(view)));
        }
        forms
    }

    #[test]
    fn the_image_merge_kernels_are_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            for (read, kept) in forms() {
                for blocks in [false, true] {
                    assert_valid(typed_module(read, kept, blocks).expect("module"), &p);
                }
            }
        }
    }

    /// Texel `(x, y)`'s word `i` of the image the tests read: `0x0100_0000 · (i + 1) +
    /// 0x1000 · y + x`.
    fn texel_word(x: u32, y: u32, i: u32) -> u32 {
        0x0100_0000 * (i + 1) + 0x1000 * y + x
    }

    /// The components a fetch of the texel of `words` returns through a view of `texel`: an
    /// `R8G8B8A8_UINT` view's its word's bytes; another unsigned-integer view's its words (the
    /// components a view lacks 0, and alpha 1); an 8-bit unsigned normalized view's the `f32`
    /// nearest `c / 255 · scale` for its byte `c` in memory order (`scale` 1 an exact
    /// conversion), as `f32` bits.
    fn components(texel: Texel, words: &[u32], scale: f64) -> [u32; 4] {
        match texel {
            Texel::Bits(ShadowTexelView::R8G8B8A8Uint) => words[0].to_le_bytes().map(u32::from),
            Texel::Bits(_) => {
                let mut components = [0, 0, 0, 1];
                components[..words.len()].copy_from_slice(words);
                components
            }
            Texel::Unorm(order) => {
                let mut components = [0; 4];
                for (byte, &component) in words[0].to_le_bytes().into_iter().zip(&order) {
                    let value = f64::from(byte) / 255.0 * scale;
                    components[component as usize] = (value as f32).to_bits();
                }
                components
            }
        }
    }

    /// The `slot` image of `[width, height]` texels whose words are `word(x, y, i)`, as a fetch
    /// through `texel`'s view returns them ([`components`], an exact conversion).
    fn image_of(
        slot: crate::pipeline::sym::ResourceSlot,
        [width, height]: [u32; 2],
        texel: Texel,
        word: impl Fn(u32, u32, u32) -> u32,
    ) -> Image {
        let n = match texel {
            Texel::Bits(view) => view.words(),
            Texel::Unorm(_) => 1,
        };
        let mut image = Image::new(slot, [width, height, 1], 1, 1, 1);
        for y in 0..height {
            for x in 0..width {
                let words: Vec<u32> = (0..n).map(|i| word(x, y, i)).collect();
                image.texels[(y * width + x) as usize] = components(texel, &words, 1.0);
            }
        }
        image
    }

    /// The `[width, height]` image the tests read through `read`, of texels whose words are
    /// [`texel_word`].
    fn image(extent: [u32; 2], read: ShadowMergeRead) -> Image {
        image_of(IMAGE, extent, texel_of(read), texel_word)
    }

    fn bytes_of(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn words_of(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }

    /// The current buffer's words after one dispatch over `region` (`[x0, y0, width, height]`) of
    /// `shadow` read through `read`, `kept` its region's words before (in a buffer, row after row,
    /// or, by `form`, in an image at their own coordinates), rows `row_words` apart in `current`
    /// from `first_word`, in both invocation orders and both forms of the stores (whole blocks and
    /// changed words, which give the same bytes where no other writer runs beside the kernel).
    fn merged(
        shadow: &Image,
        (read, form): (ShadowMergeRead, Kept),
        region: [u32; 4],
        kept: &[u32],
        (row_words, first_word): (u32, u32),
        current: &[u32],
    ) -> Vec<u32> {
        let [x0, y0, width, height] = region;
        let extent = [shadow.extent[0], shadow.extent[1]];
        let shape = ShadowMergeImageModule::shape([x0, y0], width, row_words, first_word);
        let words = read.words();
        // The kept image: the region's kept words at their coordinates, and outside it words no
        // texel of the read image has, which the merge must not read.
        let kept_image = |view: ShadowTexelView| {
            image_of(KEPT_IMAGE, extent, Texel::Bits(view), |x, y, i| {
                let inside = (x0..x0 + width).contains(&x) && (y0..y0 + height).contains(&y);
                if inside {
                    kept[(((y - y0) * width + (x - x0)) * words + i) as usize]
                } else {
                    0x5a5a_5a5a ^ texel_word(x, y, i)
                }
            })
        };
        let mut out = Vec::new();
        for (blocks, order) in [false, true]
            .into_iter()
            .flat_map(|blocks| [Order::Forward, Order::Reverse].map(|order| (blocks, order)))
        {
            let m = typed_module(read, form, blocks).expect("module");
            let buffer = |slot, words: &[u32]| Buffer {
                slot: Some(slot),
                address: None,
                bytes: bytes_of(words),
            };
            let mut buffers = vec![buffer(CURRENT, current)];
            let mut images = vec![shadow.clone()];
            match form {
                Kept::Buffer => buffers.insert(0, buffer(KEPT, kept)),
                Kept::Image(view) => images.push(kept_image(view)),
            }
            let res = Resources {
                buffers,
                push: shape.to_vec(),
                spec: Default::default(),
            };
            let [gx, gy] = ShadowMergeImageModule::groups(read, [width, height]);
            let cfg = Config {
                order,
                workgroups: [gx, gy, 1],
                ..Config::default()
            };
            let (outcome, images) = run_with_images(&m, &cfg, res, images).expect("run");
            // Whole blocks take two workgroup barriers (the clear, the marks); words none.
            let barriers = if blocks { 2 } else { 0 };
            assert!(outcome.invocations.iter().all(|i| i.barriers == barriers));
            assert_eq!(&images[0], shadow, "the view is only read");
            match form {
                Kept::Buffer => {
                    assert_eq!(outcome.buffers[0], bytes_of(kept), "kept is only read");
                }
                Kept::Image(view) => assert_eq!(images[1], kept_image(view), "kept is only read"),
            }
            out.push(words_of(outcome.buffers.last().expect("current")));
        }
        assert!(
            out.iter().all(|o| *o == out[0]),
            "order and form independent"
        );
        out.swap_remove(0)
    }
    /// Whole blocks: a block of a row with no changed word reads and writes no word of the
    /// current buffer, and a word of a changed block the command left keeps the current
    /// buffer's. The current buffer holds the row's first block alone (`LOCAL_SIZE` words), so an
    /// access to any other block of the row traps (out of bounds) and fails the run, through
    /// every read, the kept words in a buffer or in an image.
    #[test]
    fn whole_blocks_touch_no_word_of_an_unchanged_block_of_a_row() {
        for (read, form) in forms() {
            let words = read.words();
            let region = [10, 2, 270, 1];
            let [x0, y0, width, _] = region;
            // Kept: the fetched words but texel 3's word 0, whose low byte alone differs.
            let mut kept = Vec::new();
            for x in 0..width {
                for i in 0..words {
                    let word = texel_word(x0 + x, y0, i);
                    kept.push(if x == 3 && i == 0 { word ^ 0xff } else { word });
                }
            }
            let current: Vec<u32> = (0..ShadowMergeImageModule::LOCAL_SIZE)
                .map(|i| 0xa5a5_0000 | i)
                .collect();
            let got = merged(
                &image([300, 7], read),
                (read, form),
                region,
                &kept,
                (width * words, 0),
                &current,
            );
            let mut want = current.clone();
            let at = (3 * words) as usize;
            want[at] = (current[at] & !0xff) | (texel_word(x0 + 3, y0, 0) & 0xff);
            assert_eq!(got, want, "{read:?} {form:?}");
        }
    }

    /// Each byte the texture changed since the keep lands at its row's place; a word whose kept
    /// and fetched words are equal keeps the current buffer's (another writer's write there
    /// stays); a word with one byte changed takes that byte alone; and no word between the rows,
    /// before the first texel or past the last changes: through every read, across workgroups
    /// and over a row whose words are not a multiple of `WORDS * LOCAL_SIZE`, so the last
    /// workgroup's invocations run past the row, the kept words in a buffer or in an image.
    #[test]
    fn the_bytes_the_texture_changed_merge_at_their_rows_alone() {
        for (read, form) in forms() {
            let words = read.words();
            let extent = [300, 7];
            // 270 texels a row: 270, 540 or 1080 words of it, by the read's texel, none of them
            // a multiple of the 256 words a workgroup takes.
            let region = [10, 2, 270, 4];
            let [x0, y0, width, height] = region;
            let row_words = 5 + width * words;
            let first_word = 3;
            let len = (first_word + height * row_words + 2) as usize;
            let current: Vec<u32> = (0..len as u32).map(|i| 0xa5a5_0000 | i).collect();
            // Kept: the fetched words, but word 0 of texel (x, y) where (x + y) % 3 == 0, whose
            // low byte alone differs, and every word of texel (x, y) where (x + y) % 3 == 1,
            // which differs in every byte.
            let mut kept = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    for i in 0..words {
                        let word = texel_word(x0 + x, y0 + y, i);
                        kept.push(match (x + y) % 3 {
                            0 if i == 0 => word ^ 0xff,
                            1 => !word,
                            _ => word,
                        });
                    }
                }
            }
            let got = merged(
                &image(extent, read),
                (read, form),
                region,
                &kept,
                (row_words, first_word),
                &current,
            );
            let mut want = current.clone();
            for y in 0..height {
                for x in 0..width {
                    for i in 0..words {
                        let at = (first_word + y * row_words + x * words + i) as usize;
                        let word = texel_word(x0 + x, y0 + y, i);
                        want[at] = match (x + y) % 3 {
                            0 if i == 0 => (current[at] & !0xff) | (word & 0xff),
                            1 => word,
                            _ => current[at],
                        };
                    }
                }
            }
            assert_eq!(got, want, "{read:?} {form:?}");
        }
    }

    /// An 8-bit unsigned normalized read rebuilds every byte value in every channel: through
    /// components the `f32` nearest `c / 255` (an exact conversion), and that value off by one
    /// part in 10^5 either way (Vulkan's floor for the accuracy of floating-point computation),
    /// the merged words are the texels' bytes in memory order.
    #[test]
    fn an_unsigned_normalized_read_rebuilds_every_byte_in_every_channel() {
        // Texel x's byte j is (x + 61 j) mod 256: each byte value once in each byte of the row.
        let width = 256;
        let word = |x: u32| u32::from_le_bytes([0, 1, 2, 3].map(|j| ((x + 61 * j) % 256) as u8));
        let want: Vec<u32> = (0..width).map(word).collect();
        for read in [
            ShadowMergeRead::B8G8R8A8Unorm,
            ShadowMergeRead::R8G8B8A8Unorm,
        ] {
            let texel = texel_of(read);
            for scale in [1.0, 1.0 - 1e-5, 1.0 + 1e-5] {
                let mut shadow = Image::new(IMAGE, [width, 1, 1], 1, 1, 1);
                for x in 0..width {
                    shadow.texels[x as usize] = components(texel, &[word(x)], scale);
                }
                // Every byte kept differs, so every byte merged is the texel's.
                let kept: Vec<u32> = want.iter().map(|w| !w).collect();
                let got = merged(
                    &shadow,
                    (read, Kept::Buffer),
                    [0, 0, width, 1],
                    &kept,
                    (width, 0),
                    &vec![0; width as usize],
                );
                assert_eq!(got, want, "{read:?} at {scale}");
            }
        }
    }
}
