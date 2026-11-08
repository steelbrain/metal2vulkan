//! A shadowed texture's texels copied in from its buffer's rows, and kept, by one compute
//! dispatch.
//!
//! A texture over a buffer whose rows the device cannot lay as Metal does lives in a shadow, an
//! optimally tiled image copied in from the buffer's bytes before a command and back after it
//! (`shadow_copy_back`). Where another writer may write the buffer's bytes while the command runs,
//! the shadow's texels are also *kept* after the copy in, in a scratch buffer laid out row after
//! row, so the copy back can merge only the words the command changed (`shadow_merge`). The copy
//! in is `vkCmdCopyBufferToImage` on every device and the keep a copy or a dispatch of
//! [`shadow_copy_back_module`](super::shadow_copy_back_module) after it. Where a device copies a
//! shadow back faster by a dispatch (a cost the runtime measures per device), a consumer may take
//! both with one dispatch of [`shadow_copy_in_module`] instead: it reads each texel's words from
//! the buffer once, stores them through an unsigned-integer storage view of the shadow, and
//! stores them in the kept part. The shadow holds the bytes the copy writes: Vulkan states that
//! "Views of compatible formats will have the same mapping between texel coordinates and memory
//! locations irrespective of the format, with only the interpretation of the bit pattern
//! changing" (`VkImageViewCreateInfo`), and an unsigned-integer view stores its components' bits
//! unconverted (an `R8G8B8A8_UINT` view's components a 4-byte texel's bytes in memory order, as
//! [`shadow_copy_back_module`](super::shadow_copy_back_module) reads them); the kept part holds
//! the same words, as the keep after the copy would store them.
//!
//! **The module** (one per view, [`ShadowTexelView`]). One `GLCompute` entry `main` with local
//! size ([`ShadowCopyInModule::LOCAL_SIZE`], 1, 1), [`ShadowCopyInModule::PUSH_BYTES`] of
//! `COMPUTE` push constants at offset 0 (the region's shape, [`ShadowCopyInModule::shape`]), and
//! in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`ShadowCopyInModule::IMAGE_BINDING`] | the shadow's storage view of the module's format ([`ShadowTexelView`]), in the `GENERAL` layout |
//! | [`ShadowCopyInModule::BYTES_BINDING`] | a storage buffer over the buffer's bytes, a whole number of 4-byte words, only read |
//! | [`ShadowCopyInModule::KEPT_BINDING`] | a storage buffer over the kept part, from its first word |
//!
//! The shape's words, in order: `origin`, the region's first texel `x0` in its low 16 bits and
//! `y0` in its high 16; `width`, its texels a row; `row_words`, the buffer's rows apart in words;
//! and `first_word`, the word of the region's first texel in the bytes binding. Invocation
//! `(x, y)` with `x` below `width` reads the `n` words (`n` the module's texel width) at words
//! `first_word + y · row_words + x · n` onward, stores them as texel `(x0 + x, y0 + y)` of level 0
//! (through an `R8G8B8A8_UINT` view, byte `i` its component `i`) and at words `(y · width + x) · n`
//! onward of the kept part; one at `x` past `width` does nothing. A texel of 1 or 2 bytes
//! (`R8_UINT`, `R16_UINT`) is one word of both buffers, its bits low ([`ShadowTexelView`]): the
//! invocation keeps the word's low byte or bytes, zero above them, for the texel and the kept
//! part alike, so its bytes binding holds a word per texel, not a narrow texture's own rows. A dispatch takes `ceil(width / LOCAL_SIZE)` by the region's height workgroups
//! ([`ShadowCopyInModule::groups`]). No texel outside the region and no kept word past its texels
//! is written, and nothing is written to the bytes.
//!
//! **Synchronization** (consumer): the bytes' last writes → `SHADER_STORAGE_READ` in
//! `COMPUTE_SHADER`, and the shadow's and the kept part's earlier accesses →
//! `SHADER_STORAGE_WRITE` there, before the dispatch; its writes → the shadow's and the kept part's
//! next uses.

use std::sync::Arc;

use super::shadow_copy_back::{stored_texel, ShadowTexelView};
use super::shadow_merge::{u32_value, words_buffer};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::{
    BinOp, Dim, ExecMode, ExecModel, ImageOperands, ImageType, ImageUsage, IoDir, MemAccess,
    StorageClass, TexelMemory, Type, TypedModule,
};

/// The crate-owned copy-in module of one texel width on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowCopyInModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl ShadowCopyInModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The shadow's unsigned-integer view, a storage image the dispatch writes the texels into.
    pub const IMAGE_BINDING: u32 = 0;
    /// The buffer's bytes, a storage buffer the dispatch reads the texels' words from.
    pub const BYTES_BINDING: u32 = 1;
    /// The kept part, a storage buffer the dispatch writes the texels' words into, row after row.
    pub const KEPT_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: one texel each, of one row.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the region's shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of a region whose first texel is `[x0, y0]` (each below 65536), `width`
    /// texels a row, whose rows lie `row_words` apart in the bytes binding from its word
    /// `first_word` (the module docs).
    pub fn shape([x0, y0]: [u32; 2], width: u32, row_words: u32, first_word: u32) -> [u8; 16] {
        let words = [(x0 & 0xffff) | (y0 << 16), width, row_words, first_word];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The workgroups of a dispatch over a region `[width, height]` texels.
    pub fn groups([width, height]: [u32; 2]) -> [u32; 2] {
        [width.div_ceil(Self::LOCAL_SIZE), height]
    }
}

/// The binding symbol of the shadow's view in the typed module.
const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the buffer's bytes in the typed module.
const BYTES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the kept part in the typed module.
const KEPT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

/// The copy-in module of `view` as a typed module, before
/// serialization: the module [`shadow_copy_in_module`] serializes, which the CPU interpreter runs.
/// Its bindings are the symbols `IMAGE`, `BYTES` and `KEPT`, and its push field the shape, a
/// `uvec4` (the module docs).
fn typed_module(view: ShadowTexelView) -> BResult<TypedModule> {
    let words = view.words();
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
    let image_type = b.ty(Type::Image(ImageType {
        sampled_type: u32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Storage,
        format: view.storage_format(),
    }))?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, image_type)?;
    let bytes = words_buffer(&mut b, BYTES, u32_)?;
    let kept = words_buffer(&mut b, KEPT, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let low = u32_value(&mut f, 0xffff)?;
    let sixteen = u32_value(&mut f, 16)?;
    let n = u32_value(&mut f, words)?;
    let mut lanes = Vec::with_capacity(words as usize);
    for lane in 0..words {
        lanes.push(u32_value(&mut f, lane)?);
    }
    let mask = match view.word_mask() {
        u32::MAX => None,
        mask => Some(u32_value(&mut f, mask)?),
    };
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let image = f.module().global_value(image)?;
    let bytes = f.module().global_value(bytes)?;
    let kept = f.module().global_value(kept)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let origin = f.composite_extract(shape, &[0])?;
    let width = f.composite_extract(shape, &[1])?;
    let row_words = f.composite_extract(shape, &[2])?;
    let first_word = f.composite_extract(shape, &[3])?;
    let inside = f.binary(BinOp::ULessThan, x, width)?;
    let copy: ArmFn<'_, ()> = Box::new(move |f, _| {
        let row = f.binary(BinOp::IMul, y, row_words)?;
        let column = f.binary(BinOp::IMul, x, n)?;
        let at = f.binary(BinOp::IAdd, first_word, row)?;
        let at = f.binary(BinOp::IAdd, at, column)?;
        let kept_row = f.binary(BinOp::IMul, y, width)?;
        let kept_texel = f.binary(BinOp::IAdd, kept_row, x)?;
        let kept_at = f.binary(BinOp::IMul, kept_texel, n)?;
        let mut texel = Vec::with_capacity(lanes.len());
        for &index in &lanes {
            let place = f.binary(BinOp::IAdd, at, index)?;
            let from = f.access_chain(bytes, &[zero, place])?;
            let word = f.load(from, MemAccess::default())?;
            // A 1- or 2-byte texel's word holds its bits alone: those above them are dropped.
            let word = match mask {
                Some(mask) => f.binary(BinOp::BitwiseAnd, word, mask)?,
                None => word,
            };
            let place = f.binary(BinOp::IAdd, kept_at, index)?;
            let to = f.access_chain(kept, &[zero, place])?;
            f.store(to, word, MemAccess::default())?;
            texel.push(word);
        }
        let x0 = f.binary(BinOp::BitwiseAnd, origin, low)?;
        let y0 = f.binary(BinOp::ShiftRightLogical, origin, sixteen)?;
        let tx = f.binary(BinOp::IAdd, x0, x)?;
        let ty = f.binary(BinOp::IAdd, y0, y)?;
        let coord = f.composite_construct(uvec2, &[tx, ty])?;
        let texel = stored_texel(f, uvec4, view, &texel)?;
        let image = f.load(image, MemAccess::default())?;
        f.write(
            image,
            coord,
            texel,
            &ImageOperands::default(),
            TexelMemory::default(),
        )?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, copy, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: ShadowCopyInModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ShadowCopyInModule::SET),
        Symbol::Binding(s) if *s == IMAGE => Some(ShadowCopyInModule::IMAGE_BINDING),
        Symbol::Binding(s) if *s == BYTES => Some(ShadowCopyInModule::BYTES_BINDING),
        Symbol::Binding(s) if *s == KEPT => Some(ShadowCopyInModule::KEPT_BINDING),
        _ => None,
    }
}

/// The copy-in module of `device` storing texels through `view`: built with the typed builder,
/// checked by the owned verifier against the device's profile (requirements included),
/// serialized, and numbered with the fixed bindings of [`ShadowCopyInModule`]. Refused,
/// `Unsupported` (feature `Format`) naming what the device lacks, for the `R32G32_UINT`,
/// `R8_UINT` and `R16_UINT` views without `shaderStorageImageExtendedFormats` (their `Rg32ui`,
/// `R8ui` and `R16ui`).
pub fn shadow_copy_in_module(
    device: &Device,
    view: ShadowTexelView,
) -> Result<ShadowCopyInModule, Error> {
    let profile = device.profile();
    let m = typed_module(view)?;
    profile
        .check(m.requirements())
        .map_err(|c| Error::Unsupported(Unsupported::needs(Feature::Format, c)))?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the shadow copy in needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(ShadowCopyInModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, ShadowCopyInModule, ShadowTexelView, BYTES, IMAGE, KEPT};
    use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// Valid on every committed profile that has what it needs: the `R32G32_UINT`, `R8_UINT`
    /// and `R16_UINT` kernels' `Rg32ui`, `R8ui` and `R16ui` views need extended storage image
    /// formats, which the Vulkan 1.2 floor lacks.
    #[test]
    fn the_copy_in_kernels_are_valid_on_every_committed_profile() {
        let extended = [
            ShadowTexelView::R32G32Uint,
            ShadowTexelView::R8Uint,
            ShadowTexelView::R16Uint,
        ];
        for (_, p) in committed_profiles().expect("profiles") {
            for view in ShadowTexelView::ALL {
                let m = typed_module(view).expect("module");
                if extended.contains(&view) && p.check(m.requirements()).is_err() {
                    continue;
                }
                assert_valid(m, &p);
            }
        }
    }

    /// A `[width, height]` image, every component of every texel `fill`.
    fn image([width, height]: [u32; 2], fill: u32) -> Image {
        let mut image = Image::new(IMAGE, [width, height, 1], 1, 1, 1);
        for texel in image.texels.iter_mut() {
            *texel = [fill; 4];
        }
        image
    }

    /// The image and the kept part after one dispatch over `region` (`[x0, y0, width, height]`)
    /// of the `extent` image stored through `view` from `bytes` words, rows `row_words` apart
    /// from `first_word`, in both invocation orders; the bytes are checked unchanged.
    fn copied_in(
        extent: [u32; 2],
        view: ShadowTexelView,
        region: [u32; 4],
        (row_words, first_word): (u32, u32),
        bytes: &[u32],
        kept_words: usize,
    ) -> (Image, Vec<u32>) {
        let m = typed_module(view).expect("module");
        let [x0, y0, width, height] = region;
        let shape = ShadowCopyInModule::shape([x0, y0], width, row_words, first_word);
        let to_bytes =
            |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: Some(BYTES),
                        address: None,
                        bytes: to_bytes(bytes),
                    },
                    Buffer {
                        slot: Some(KEPT),
                        address: None,
                        bytes: to_bytes(&vec![0xdead_beef; kept_words]),
                    },
                ],
                push: shape.to_vec(),
                spec: Default::default(),
            };
            let [gx, gy] = ShadowCopyInModule::groups([width, height]);
            let cfg = Config {
                order,
                workgroups: [gx, gy, 1],
                ..Config::default()
            };
            let (outcome, mut images) =
                run_with_images(&m, &cfg, res, vec![image(extent, 7)]).expect("run");
            assert_eq!(
                outcome.buffers[0],
                to_bytes(bytes),
                "the bytes are only read"
            );
            let kept: Vec<u32> = outcome.buffers[1]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            out.push((images.swap_remove(0), kept));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// Every texel of a region is its row's words (through an `R8G8B8A8_UINT` view its bytes, one
    /// a component; through a 1- or 2-byte texel's view its word's bits, zero above them), in the
    /// image and row after row in the kept part, and no texel outside it and no kept word past its
    /// texels changes: through every view, across workgroups.
    #[test]
    fn a_region_lands_in_the_image_and_the_kept_part_alone() {
        for view in ShadowTexelView::ALL {
            let words = view.words();
            let extent = [90, 7];
            // The region's rows 5 + 70 · words words apart, from word 3.
            let region = [10, 2, 70, 4];
            let row_words = 5 + 70 * words;
            let first_word = 3;
            let len = (first_word + 4 * row_words + 2) as usize;
            let bytes: Vec<u32> = (0..len as u32).map(|i| 0xa5a5_0000 | i).collect();
            let kept_words = (70 * 4 * words + 3) as usize;
            let (got, kept) = copied_in(
                extent,
                view,
                region,
                (row_words, first_word),
                &bytes,
                kept_words,
            );
            let mut want = image(extent, 7);
            let mut want_kept = vec![0xdead_beef; kept_words];
            for y in 0..4 {
                for x in 0..70 {
                    let (tx, ty) = (10 + x, 2 + y);
                    let mut texel = [0, 0, 0, 0];
                    for i in 0..words {
                        let word = bytes[(first_word + y * row_words + x * words + i) as usize];
                        // A 1- or 2-byte texel keeps its word's bits alone.
                        let word = word & view.word_mask();
                        texel[i as usize] = word;
                        want_kept[((y * 70 + x) * words + i) as usize] = word;
                    }
                    if view == ShadowTexelView::R8G8B8A8Uint {
                        texel = texel[0].to_le_bytes().map(u32::from);
                    }
                    want.texels[(ty * extent[0] + tx) as usize] = texel;
                }
            }
            assert_eq!(got, want, "{view:?}: the image");
            assert_eq!(kept, want_kept, "{view:?}: the kept part");
        }
    }

    /// The origin's two halves: a texel past 255 in both coordinates is written at its place.
    #[test]
    fn the_origin_reaches_far_texels() {
        let extent = [300, 260];
        let region = [290, 257, 3, 2];
        let bytes: Vec<u32> = (0..8).map(|i| 0x100 + i).collect();
        for view in [ShadowTexelView::R32Uint, ShadowTexelView::R8G8B8A8Uint] {
            let (got, kept) = copied_in(extent, view, region, (4, 1), &bytes, 6);
            let at = |x: u32, y: u32| {
                let texel = got.texels[(y * extent[0] + x) as usize];
                match view {
                    ShadowTexelView::R8G8B8A8Uint => u32::from_le_bytes(texel.map(|b| b as u8)),
                    _ => texel[0],
                }
            };
            assert_eq!(
                [at(290, 257), at(291, 257), at(292, 257)],
                [0x101, 0x102, 0x103],
                "{view:?}"
            );
            assert_eq!(
                [at(290, 258), at(291, 258), at(292, 258)],
                [0x105, 0x106, 0x107],
                "{view:?}"
            );
            assert_eq!(kept, vec![0x101, 0x102, 0x103, 0x105, 0x106, 0x107]);
        }
    }
}
