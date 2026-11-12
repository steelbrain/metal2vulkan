//! A texture that a draw samples where its own render pass renders it (an attached read,
//! `texture/attached_reads.rs`), held in two images of its own, its faces, that take turns as its
//! truth: a pass that renders the whole texture without loading it renders into the face that
//! does not hold the truth, while its draws read the one that does, so no copy goes before it.
//!
//! - **The faces.** The texture's own image (`X`) and a second image made with the same plan
//!   (`Y`), in memory of its own, made at the texture's first attached read in the place of the
//!   shadow that read would make (`texture/usage_shadow.rs`, `Need::AttachedRead`): the texture
//!   presents one of them ([`Texture::storage`](super::Texture)), and the other is its attached
//!   reads' shadow, copied in from the one presented as any image's shadow is (`shadow.rs`,
//!   `ShadowSource::Image`), each direction its own copies. So the memory is what the shadow took.
//! - **Which textures** ([`Texture::faces_fit`]): a texture made by `newTextureWithDescriptor:`
//!   (no view, not shareable, in no heap, over no buffer or surface), `Private`, its hazards
//!   tracked, 2D of one level, one slice and one sample, presented whole in its image's format,
//!   whose image samples, renders and copies both ways, never handed out, with no storage form, no
//!   decoded twin and no conversion. A texture of several levels keeps its copy: a pass that
//!   renders one level would leave the other face's other levels stale, and one view cannot present
//!   levels of two images.
//! - **The turn** (`render/face_swap.rs`): at the open of a render pass whose one color attachment
//!   renders the whole texture (level 0, slice 0, the pass's extent the texture's), loaded `Clear`
//!   or `DontCare`, no resolve or depth or stencil attachment of it, the texture presents its other
//!   face from then on ([`Texture::turn_face`]). The pass renders into it; its draws read the face
//!   presented until then, which holds the texels from before the pass, exactly what the copy in
//!   ahead of the pass's load actions holds: the pass's first instance takes no copy
//!   (`InstanceShadows::ahead`, `shadow::ShadowSync::hold_ahead`), counted
//!   (`WorkCounts::attached_read_swaps`, `attached_read_swap_reads`). A draw after a barrier splits
//!   the pass as before (`render.rs`), and its copy in moves the new face's texels into the old,
//!   which the first instance no longer reads. Every command after the turn names the new face, so
//!   no copy follows the pass either.
//! - **When.** Only where nothing else can name the face presented until then: the opening command
//!   buffer is the device's only uncommitted one, the scheduler holds no batch, no command buffer
//!   that may read the heap sets is incomplete and this one bound none (as a storage form's switch,
//!   `texture/storage_form.rs`), the texture turned in no command buffer whose turning pass has not
//!   been submitted, and it turned in this one never yet. A new command buffer counts itself before
//!   it reads the device's turning flag ([`Device::face_turn_wait`]), which is up from before the
//!   turn reads the count until the face is stored, so it either stopped the turn or sees the new
//!   face. The turn writes the texture's heap set elements again (`heap_sets::rewrite`): its ID
//!   reads the face presented, and nothing pending reads the elements meanwhile.
//! - **A command buffer made meanwhile.** One made after the turn, before the turning command
//!   buffer's pass is submitted, names the new face, which holds the truth only once that pass ran.
//!   Where one of its batches is submitted first and names the new face (or binds the heap sets, or
//!   reaches memory no use names), its submission copies the old face into the new before the
//!   batch, and the new into the old after it where it may write the new
//!   ([`FaceSwaps::fixes`], `command_buffer/submit.rs`), counted
//!   (`WorkCounts::attached_read_swap_fixes`): so it reads the truth, and the turning pass's reads
//!   see what it wrote. Where the turning command buffer goes uncommitted, or commits nothing of
//!   its work ([`FaceSwaps::ran_nothing`]), the next batch to name the new face takes the same
//!   copies, once. Where its pass is submitted, every later batch is after it.
//! - **Frozen** ([`Texture::freeze_faces`]): a view of the texture, its `VkImage` handed out, a
//!   binding beyond its usage (whose shadow names one face as its truth for good), a storage form,
//!   or a purgeable state end its turns: it presents the face it presents then from then on, the
//!   other kept as its attached reads' shadow. A kept recording of indirect commands that named the
//!   face no longer presented runs no more ([`Device::names_old_face`], `kept_commands.rs`).
//! - **Exact** in every case: a pass's read sees the texels from before it, a later command the
//!   pass's, as with the copy. Nothing is announced.

use super::usage_shadow::UsageShadow;
use super::{texture_aspects, Image, Owner, Storage, Texture, TextureShared};
use crate::device::lock;
use crate::error::Error;
use crate::heap_sets;
use crate::resource::ResourceKey;
use crate::shadow::{Backing, ImageTruth, ShadowCopies, ShadowSource};
use crate::shadow_merge::ScratchSpares;
use crate::shadow_standing::StandingAcross;
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use crate::tracker::{Summary, UseRange};
use crate::values::StorageMode;
use crate::Device;
use ash::vk;
use metal2vulkan::metal::format::ycbcr_writes_stored;
use metal2vulkan::metal::TextureType;
use metal2vulkan::vk::{ImageType, ImageUsageFlags};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// A texture's two faces (the module docs).
pub(super) struct Faces {
    /// The second face, `Y`: an image of the texture's plan in memory of its own.
    other: Storage,
    /// The face the texture presents: 0 its own storage, 1 `other`.
    face: AtomicU8,
    /// The attached reads' shadow while each face is presented: the other face, copied in from
    /// the one presented.
    shadows: [Arc<UsageShadow>; 2],
    /// Its turns are over for good (the module's **Frozen**).
    frozen: AtomicBool,
    /// Held while it turns or freezes.
    turning: Mutex<()>,
}

impl Faces {
    /// The face presented.
    fn face(&self) -> usize {
        usize::from(self.face.load(Ordering::Acquire))
    }

    /// The attached reads' shadow now.
    pub(super) fn shadow(&self) -> Arc<UsageShadow> {
        Arc::clone(&self.shadows[self.face()])
    }

    /// The storage presented where it is the second face, `None` for the texture's own.
    pub(super) fn presented(&self) -> Option<&Storage> {
        (self.face() == 1).then_some(&self.other)
    }

    /// The second face's image, which the texture's slot elements may name.
    pub(super) fn other_image(&self) -> Option<&Arc<Image>> {
        match &self.other {
            Storage::Image(image) => Some(image),
            _ => None,
        }
    }
}

/// Why a pass that renders a texture an attached read may sample did not turn its faces (the
/// module docs), as `WorkCounts::attached_read_unswapped_load` and kin count the copies in such
/// passes took instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unturned {
    /// The pass loads the color (`Load`).
    Load,
    /// The texture has several levels, slices or samples, or is not 2D.
    Levels,
    /// The color renders part of the texture: another level or slice, an extent short of the
    /// texture's, the texture rendered by several colors or resolved into, or its depth or
    /// stencil attachment.
    Area,
    /// The texture is no image of its own the runtime made: over a buffer's bytes or a surface,
    /// in a heap, shareable, sparse, the caller's, a view, or not `Private` and tracked.
    Storage,
    /// Its turns are over (the module's **Frozen**).
    Frozen,
    /// No attached read made its faces yet: the texture's first.
    First,
    /// Something else may name it meanwhile: another uncommitted command buffer, a batch the
    /// scheduler holds, a heap ticket out, a heap set this command buffer bound, a turn whose
    /// pass is not submitted, or a turn earlier in this command buffer.
    Busy,
    /// The pass takes no turn of any texture: an encoding again, color groups, a
    /// rasterization-rate map, or a log for a later encoding.
    Pass,
    /// A swapped pass's copies after its first instance: a read after a barrier, whose copy in
    /// the turn does not take off.
    Split,
}

impl Unturned {
    /// How many causes there are.
    pub(crate) const COUNT: usize = 9;
}

/// What a turn leaves its pass: the attached reads' shadow its first instance reads without a
/// copy in (the module docs).
pub(crate) struct Turned {
    pub(crate) copies: Arc<ShadowCopies>,
}

/// Where a render pass opens in its command buffer, for its turns (`CommandBuffer::turn_point`):
/// the command buffer's address, which names it while it lives; the place among its ended
/// encoders' parts that the pass's encoder's first part takes; whether a command of it bound a
/// heap set; and whether an ended encoder of it keeps a log that a rebind encodes again
/// (`replay.rs`), whose commands would then name the face presented after the turn.
pub(crate) struct TurnPoint {
    pub(crate) owner: usize,
    pub(crate) part: usize,
    pub(crate) bound_heaps: bool,
    pub(crate) logs_kept: bool,
}

impl Texture {
    /// Whether the texture may take faces (the module's **Which textures**): and no binding beyond
    /// its usage made it a shadow of its own before an attached read could.
    pub(super) fn faces_fit(&self) -> bool {
        let shared = &self.0;
        let Storage::Image(image) = &shared.storage else {
            return false;
        };
        let own = matches!(
            image.owner,
            Owner::Runtime {
                shareable: false,
                ..
            }
        );
        let plan = image.plan();
        let usage = ImageUsageFlags::SAMPLED
            | ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST;
        let record = &shared.record;
        let private = matches!(image.options.storage_mode(), Ok(StorageMode::Private));
        own && shared.parent.is_none()
            && shared.heap.is_none()
            && shared.moved.get().is_none()
            && shared.planes_in_place.is_none()
            && private
            && image.identity.tracked
            && plan.image_type == ImageType::TYPE_2D
            && (plan.levels, plan.layers, plan.samples.as_raw()) == (1, 1, 1)
            && plan.usage.contains(usage)
            && record.texture_type == TextureType::Type2D
            && (record.base_level, record.level_count) == (0, 1)
            && (record.base_layer, record.layer_count) == (0, 1)
            && record.read_as.is_none()
            && record.presented.rotation == 0
            && record.pixel_format == plan.pixel_format
            && !ycbcr_writes_stored(record.pixel_format)
            && image.decoded.is_none()
            && !image.handed_out()
            && image.form.plan().is_none()
            && image.objects.count() == 1
            && shared.usage_shadow.get().is_none()
    }

    /// Whether the texture has faces (the module docs).
    pub(crate) fn has_faces(&self) -> bool {
        self.0.faces.get().is_some()
    }

    /// Why the texture may not take faces now, where it may not
    /// ([`faces_fit`](Self::faces_fit)): [`Unturned::Levels`] for its shape, several levels,
    /// slices or samples, or another type than 2D; [`Unturned::Storage`] for the rest.
    pub(crate) fn unfit_for_faces(&self) -> Option<Unturned> {
        let record = &self.0.record;
        let shape = record.texture_type == TextureType::Type2D
            && (record.level_count, record.layer_count, record.samples) == (1, 1, 1);
        match (shape, self.faces_fit()) {
            (false, _) => Some(Unturned::Levels),
            (true, false) => Some(Unturned::Storage),
            (true, true) => None,
        }
    }

    /// The texture's faces, made now (the module docs): its second image, and the copies between
    /// the two. The attached reads' shadow while the texture presents its own image.
    pub(super) fn make_faces(&self) -> Result<Arc<UsageShadow>, Error> {
        let Storage::Image(own) = &self.0.storage else {
            return Err(crate::error::internal(
                "faces of a texture without an image",
            ));
        };
        let device = self.device();
        let record = &self.0.record;
        let layout = TextureLayout {
            texture_type: record.texture_type,
            pixel_format: own.plan().pixel_format,
            extent: record.image_extent,
            levels: 1,
            layers: 1,
            samples: 1,
        };
        let mut descriptor = TextureDescriptor::texture_2d(own.plan().pixel_format, 1, 1, false);
        descriptor.resource_options = own.options;
        descriptor.allow_gpu_optimized_contents = own.allow_gpu_optimized_contents;
        let hazard = own.options.hazard_tracking_mode()?;
        let other = device
            .new_image(own.plan().clone(), &layout, &descriptor, hazard)?
            .shared();
        let into_other = Arc::new(face_copies(own, &other, record.image_extent));
        let into_own = Arc::new(face_copies(&other, own, record.image_extent));
        let shadows = [
            Arc::new(UsageShadow::face(Arc::clone(&other), into_other)),
            Arc::new(UsageShadow::face(Arc::clone(own), into_own)),
        ];
        let faces = Faces {
            other: Storage::Image(other),
            face: AtomicU8::new(0),
            shadows,
            frozen: AtomicBool::new(false),
            turning: Mutex::new(()),
        };
        let first = Arc::clone(&faces.shadows[0]);
        let keys = [
            own.identity.key,
            faces
                .other_image()
                .map_or(own.identity.key, |o| o.identity.key),
        ];
        if self.0.faces.set(faces).is_err() {
            return Err(crate::error::internal("a texture's faces made twice"));
        }
        (device.shared().face_swaps).faced(keys, Arc::downgrade(&self.0));
        device.shared().counts.texture_faces_made();
        Ok(first)
    }

    /// The face the texture presents: 0 its own image (a texture without faces too), 1 its
    /// second.
    pub(crate) fn face(&self) -> u8 {
        (self.0.faces.get()).map_or(0, |faces| faces.face.load(Ordering::Acquire))
    }

    /// End the texture's turns, where it has faces (the module's **Frozen**): it presents the
    /// face it presents now for good.
    pub(crate) fn freeze_faces(&self) {
        if let Some(faces) = self.root().0.faces.get() {
            let _turning = lock(&faces.turning);
            faces.frozen.store(true, Ordering::Release);
        }
    }

    /// Whether the texture has faces whose turns are over.
    pub(crate) fn faces_frozen(&self) -> Option<bool> {
        (self.0.faces.get()).map(|faces| faces.frozen.load(Ordering::Acquire))
    }

    /// Present the texture's other face from now on, for a pass opening at `point` of its
    /// command buffer (the module's **The turn** and **When**); the caller checked the pass. The
    /// shadow the pass's first instance reads without a copy in, or why it did not turn.
    pub(crate) fn turn_face(&self, point: &TurnPoint) -> Result<Turned, Unturned> {
        let Some(faces) = self.0.faces.get() else {
            return Err(match self.faces_fit() {
                true => Unturned::First,
                false => Unturned::Storage,
            });
        };
        let _turning = lock(&faces.turning);
        if faces.frozen.load(Ordering::Acquire) {
            return Err(Unturned::Frozen);
        }
        let device = self.device();
        let shared = device.shared();
        let swaps = &shared.face_swaps;
        if point.bound_heaps || shared.uncommitted.count() != 1 || swaps.named(&self.0) {
            return Err(Unturned::Busy);
        }
        swaps.turning.store(true, Ordering::SeqCst);
        let quiet = shared.uncommitted.count() == 1
            && !device.is_lost()
            && shared.scheduler.holds_none()
            && !heap_sets::tickets_out(device);
        if !quiet {
            swaps.turning.store(false, Ordering::SeqCst);
            return Err(Unturned::Busy);
        }
        let from = faces.face();
        let to = 1 - from;
        let image = |face: usize| -> Arc<Image> {
            let storage = match face {
                0 => &self.0.storage,
                _ => &faces.other,
            };
            match storage {
                Storage::Image(image) => Arc::clone(image),
                _ => unreachable!("a face is an image"),
            }
        };
        let (base, turned) = (image(from), image(to));
        faces.face.store(to as u8, Ordering::Release);
        // Its ID reads the face presented: written again while nothing pending reads it.
        let id = self.0.id.id().0;
        if id != 0 {
            heap_sets::rewrite(device, id);
        }
        // A kept list of resident textures' shadows is listed again (`shadow::ResidentShadows`).
        (shared.usage_shadows_made).fetch_add(1, Ordering::Release);
        swaps.begin(SwapRecord {
            owner: point.owner,
            part: point.part,
            texture: Arc::downgrade(&self.0),
            base,
            turned,
            stale: AtomicBool::new(false),
        });
        swaps.turning.store(false, Ordering::SeqCst);
        shared.counts.attached_read_swap();
        Ok(Turned {
            copies: Arc::clone(faces.shadows[to].copies()),
        })
    }

    /// The copies of the attached reads' shadow of each face, where the texture has faces: a
    /// command buffer's carry of either goes at a turn (`render/face_swap.rs`).
    pub(crate) fn face_copies(&self) -> Option<[Arc<ShadowCopies>; 2]> {
        let faces = self.0.faces.get()?;
        Some([
            Arc::clone(faces.shadows[0].copies()),
            Arc::clone(faces.shadows[1].copies()),
        ])
    }
}

/// The copies from the face `truth` into the face `shadow` of a texture of `extent` (one level,
/// one slice), as an image's shadow's are (`texture/view_shadow.rs`).
fn face_copies(truth: &Arc<Image>, shadow: &Arc<Image>, extent: [u32; 3]) -> ShadowCopies {
    let aspects = texture_aspects(truth.plan());
    let range = UseRange::Image {
        aspects,
        levels: 0..1,
        layers: 0..1,
    };
    let layers = vk::ImageSubresourceLayers {
        aspect_mask: aspects,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let [width, height, _] = extent;
    let regions = vec![vk::ImageCopy {
        src_subresource: layers,
        src_offset: vk::Offset3D::default(),
        dst_subresource: layers,
        dst_offset: vk::Offset3D::default(),
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }];
    let image = Arc::clone(truth) as Arc<dyn ImageTruth>;
    // A tracked private image in memory of its own: only the runtime's commands write it
    // (`Texture::faces_fit`), so its shadow may stand across command buffers.
    let standing = StandingAcross::over_image(Arc::clone(&image), true);
    ShadowCopies {
        source: ShadowSource::Image {
            image,
            key: truth.identity.key,
            range: range.clone(),
            shadow: shadow.raw.get(),
            regions,
            through: None,
        },
        shadow_key: shadow.identity.key,
        shadow_range: range,
        tracked: shadow.identity.tracked,
        drops_writes: false,
        levels: None,
        backing: Backing::PrivateImage,
        merge_scratch: ScratchSpares::default(),
        stood: AtomicU64::new(0),
        standing,
    }
}

/// A turn whose pass may not have run yet (the module's **A command buffer made meanwhile**).
struct SwapRecord {
    /// The address of the command buffer whose pass turned the texture, and its part that holds
    /// the pass ([`TurnPoint`]).
    owner: usize,
    part: usize,
    texture: Weak<TextureShared>,
    /// The face that holds the truth until the pass runs, and the face presented from the turn.
    base: Arc<Image>,
    turned: Arc<Image>,
    /// The pass will not run (its command buffer went uncommitted or ran nothing): the next batch
    /// that names the turned face takes the copies, and the record goes.
    stale: AtomicBool,
}

/// The device's turns whose passes may not have run (the module docs).
#[derive(Default)]
pub(crate) struct FaceSwaps {
    /// Up while a texture turns (the module's **When**).
    turning: AtomicBool,
    /// How many records there are: 0 wherever no turn waits, which a batch's submission reads
    /// alone.
    pending: AtomicUsize,
    records: Mutex<Vec<SwapRecord>>,
    /// The faces' keys of the textures with faces, each with its texture, and how many there are
    /// ([`Device::names_old_face`]).
    faced: Mutex<Vec<([ResourceKey; 2], Weak<TextureShared>)>>,
    faced_count: AtomicUsize,
}

/// The copies a batch's submission records around it for a turn (the module docs): from the
/// face that holds the truth into the one the batch names before it, and back after it where
/// `back`.
pub(crate) struct FaceFix {
    pub(crate) from: Arc<Image>,
    pub(crate) to: Arc<Image>,
    pub(crate) back: bool,
}

impl FaceFix {
    /// The texels each copy of it moves: the whole face.
    pub(crate) fn texels(&self) -> u64 {
        let [width, height, _] = self.from.plan().extent;
        u64::from(width) * u64::from(height)
    }

    /// The fix's uses: read `from` and write `to`, then the reverse where `back`.
    pub(crate) fn uses(&self) -> (Summary<ResourceKey>, Option<Summary<ResourceKey>>) {
        let read = crate::tracker::Access::new(
            vk::PipelineStageFlags2::ALL_TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let write = crate::tracker::Access::new(
            vk::PipelineStageFlags2::ALL_TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let face_use = |image: &Image, writes: bool| {
            let range = UseRange::Image {
                aspects: texture_aspects(image.plan()),
                levels: 0..1,
                layers: 0..1,
            };
            let identity = image.identity;
            match writes {
                true => crate::tracker::Use::writes(identity.key, identity.tracked, range, write),
                false => crate::tracker::Use::reads(identity.key, identity.tracked, range, read),
            }
        };
        let front =
            Summary::outside_encoders(vec![face_use(&self.from, false), face_use(&self.to, true)]);
        let back = self.back.then(|| {
            Summary::outside_encoders(vec![face_use(&self.to, false), face_use(&self.from, true)])
        });
        (front, back)
    }

    /// Record the copy from `from` into `to` (`forward`) or back on `command_buffer`, with a
    /// barrier from everything before it and to everything after it.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; both images are live and rest in
    /// `GENERAL`.
    pub(crate) unsafe fn record(
        &self,
        device: &crate::device::functions::Vk,
        command_buffer: vk::CommandBuffer,
        forward: bool,
    ) {
        let (from, to) = match forward {
            true => (&self.from, &self.to),
            false => (&self.to, &self.from),
        };
        let aspects = texture_aspects(from.plan());
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: aspects,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let [width, height, _] = from.plan().extent;
        let region = vk::ImageCopy {
            src_subresource: layers,
            src_offset: vk::Offset3D::default(),
            dst_subresource: layers,
            dst_offset: vk::Offset3D::default(),
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        // SAFETY: the caller's contract.
        unsafe {
            crate::tracker::Barrier::everything().record(device, command_buffer);
            device.cmd_copy_image(
                command_buffer,
                from.raw.get(),
                vk::ImageLayout::GENERAL,
                to.raw.get(),
                vk::ImageLayout::GENERAL,
                &[region],
            );
            crate::tracker::Barrier::everything().record(device, command_buffer);
        }
    }
}

impl FaceSwaps {
    /// Note the faces `keys` of `texture`, just made; those of textures gone go.
    fn faced(&self, keys: [ResourceKey; 2], texture: Weak<TextureShared>) {
        let mut faced = lock(&self.faced);
        faced.retain(|(_, texture)| texture.strong_count() > 0);
        faced.push((keys, texture));
        self.faced_count.store(faced.len(), Ordering::Release);
    }

    /// Keep `record`, a turn just made.
    fn begin(&self, record: SwapRecord) {
        let mut records = lock(&self.records);
        records.push(record);
        self.pending.store(records.len(), Ordering::SeqCst);
    }

    /// Whether a record names `texture`: a turn of it whose pass may not have run.
    fn named(&self, texture: &TextureShared) -> bool {
        if self.pending.load(Ordering::SeqCst) == 0 {
            return false;
        }
        let records = lock(&self.records);
        (records.iter()).any(|record| std::ptr::eq(record.texture.as_ptr(), texture))
    }

    /// The command buffer at `owner` will run none of its work: its went uncommitted, or its commit
    /// ran nothing (the module docs). Its records go stale.
    pub(crate) fn ran_nothing(&self, owner: usize) {
        if self.pending.load(Ordering::SeqCst) == 0 {
            return;
        }
        let records = lock(&self.records);
        for record in records.iter().filter(|record| record.owner == owner) {
            record.stale.store(true, Ordering::Release);
        }
    }

    /// Whether a turn's pass may not have run: one atomic read, each batch's submission's gate.
    pub(crate) fn any(&self) -> bool {
        self.pending.load(Ordering::SeqCst) > 0
    }

    /// The copies the batch of parts `parts` of the command buffer at `owner`, whose summaries
    /// are `summaries`, takes
    /// around it for the turns whose passes have not run (the module docs), `unnamed` where it
    /// may reach memory no use names or binds the heap sets; `ran` says whether a part of it
    /// runs anything. A record of `owner` whose part is among `parts` goes where that part runs:
    /// its pass is submitted now; where it runs nothing, the record goes stale.
    pub(crate) fn fixes<'a>(
        &self,
        owner: usize,
        (parts, ran): (std::ops::Range<usize>, &dyn Fn(usize) -> bool),
        summaries: impl Iterator<Item = &'a Summary<ResourceKey>> + Clone,
        unnamed: bool,
    ) -> Vec<FaceFix> {
        if !self.any() {
            return Vec::new();
        }
        let mut records = lock(&self.records);
        let mut fixes = Vec::new();
        records.retain(|record| {
            let own = record.owner == owner;
            if own && !record.stale.load(Ordering::Acquire) {
                // Its own batches before the pass name the face from before the turn, which
                // holds the truth; the pass's batch runs the pass, unless the pass's part runs
                // nothing.
                if !parts.contains(&record.part) {
                    return true;
                }
                if ran(record.part) {
                    return false;
                }
                record.stale.store(true, Ordering::Release);
                return true;
            }
            let stale = record.stale.load(Ordering::Acquire);
            let key = record.turned.identity.key;
            let names = |summary: &Summary<ResourceKey>| summary.keys().any(|k| k == key);
            if !unnamed && !summaries.clone().any(names) {
                return true;
            }
            // The pass reads the face from before the turn after this batch, where it is still to
            // run: it takes what the batch wrote too.
            let writes = |summary: &Summary<ResourceKey>| summary.written_keys().any(|k| k == key);
            let back = !stale && (unnamed || summaries.clone().any(writes));
            fixes.push(FaceFix {
                from: Arc::clone(&record.base),
                to: Arc::clone(&record.turned),
                back,
            });
            // A stale record's pass never runs: the face turned to holds the truth from here on.
            !stale
        });
        // A record whose texture went needs no copy: nothing names its faces again.
        records.retain(|record| record.texture.strong_count() > 0);
        self.pending.store(records.len(), Ordering::SeqCst);
        fixes
    }
}

impl Device {
    /// A new command buffer, counted, waits out a texture's turn under way (the module's
    /// **When**): microseconds, as a turn holds the flag only while it stores the face.
    pub(crate) fn face_turn_wait(&self) {
        let swaps = &self.shared().face_swaps;
        while swaps.turning.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
    }

    /// Whether `keys`, a kept recording's, name a face its texture no longer presents (the
    /// module's **Frozen**): such a recording runs no more.
    pub(crate) fn names_old_face(&self, keys: impl IntoIterator<Item = ResourceKey>) -> bool {
        let swaps = &self.shared().face_swaps;
        if swaps.faced_count.load(Ordering::Acquire) == 0 {
            return false;
        }
        let faced = lock(&swaps.faced);
        keys.into_iter().any(|key| {
            (faced.iter())
                .filter(|(faces, _)| faces.contains(&key))
                .filter_map(|(_, texture)| texture.upgrade().map(Texture))
                .any(|texture| texture.image().is_ok_and(|image| image.identity.key != key))
        })
    }
}
