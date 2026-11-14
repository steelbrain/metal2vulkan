//! A multisampled depth texture held in the attachment layout between the render passes that
//! render it, on a device whose driver renders such a depth faster out of `GENERAL`
//! (`DriverFact::MultisampledDepthFasterOutOfGeneral`).
//!
//! - **Why.** On such a driver (ANV on the Iris Xe of RPL-P) a multisampled depth renders without
//!   its hierarchical depth in `GENERAL`, where every image of the runtime rests, and a move back
//!   to `GENERAL` after a pass costs a full resolve of the depth (about 1 ms at 1920x1080 with 4
//!   samples). A pass may hold the depth in `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` for its instances
//!   (`render_target::depth_layout`); this keeps it there between passes too, until a command
//!   needs it in `GENERAL`.
//! - **Which.** A device's own `Private` texture ([`eligible`]): multisampled, of a depth or
//!   depth/stencil format, not memoryless (`TRANSIENT_ATTACHMENT`). No host
//!   copy reaches it (Metal refuses the CPU a private texture's bytes), and never a heap's, a
//!   caller's image or a texture over a buffer, whose images other resources reach. Its `VkImage`
//!   is its own for good. The registry ([`HeldImages`]) knows each by its key from its making to
//!   its image's drop.
//! - **Where the moves go.** Only between encoders, in the device's submission order: the
//!   scheduler keeps each image's layout as the batches submitted so far leave it
//!   ([`HeldImage::current`]), and before an encoder that names the image (its commands' uses and
//!   bindings, `EncoderCore::named_keys`) moves it into the layout the encoder needs, in the
//!   encoder's command buffer of barriers (`command_buffer/submit.rs`): [`HELD`] for a render
//!   encoder that holds it, `GENERAL` for any other. A move is a write the hazard tracker orders
//!   as every write ([`HeldImage::transition_use`]), waits for other `VkQueue`s included. Nothing
//!   inside an encoder moves the image, so the order its commands are recorded in never matters.
//!   An image no encoder names stays as the last left it.
//! - **Which encoder holds.** A render encoder whose depth attachment is such a texture holds it
//!   only where nothing else in it touches the image: no command of its own named it before its
//!   pass's end, no step of its pass uses it but as the attachment, and its pass runs none of the
//!   steps that read or write the depth apart from its attachment (`render.rs`, the hold's
//!   decision). Its instances then render the depth in [`HELD`] with no move around them. Any other
//!   encoder keeps the depth in `GENERAL`, a render encoder of it moving it around each instance
//!   as before. A use other than the attachment's recorded after a hold is an internal error
//!   (`tracker::EncoderUses::broke_hold`).
//! - **What a program sees.** The texels: a layout transition keeps them ("The contents of any
//!   portion of an image subresource that is transitioned from one layout to another are
//!   preserved", unless the old layout is `UNDEFINED`; Vulkan's synchronization chapter). A read
//!   through a binding the program made registers its use (a slot's texture, or one made resident
//!   by `use_resource`), which moves the image to `GENERAL` before that encoder, whatever its
//!   usage says. A texture reached through an argument buffer without `use_resource` registers
//!   nothing, which Metal's contract does not allow.

use super::super::key_hash::KeyMap;
use super::super::resource::ResourceKey;
use super::super::tracker::{Access, ImagePlace, Use, UseRange};
use super::super::Device;
use super::image_aspects;
use super::wrapped::Transition;
use crate::image_plan::ImagePlan;
use ash::vk;
use metal2vulkan::device::DriverFact;
use metal2vulkan::vk::{ImageAspectFlags, ImageLayout, ImageUsageFlags};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// The layout a holding render encoder renders the depth in, and the image rests in after it.
pub(crate) const HELD: ImageLayout = ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

/// Whether a device's own texture of `plan` is held between passes on `device` (the module docs).
pub(crate) fn eligible(device: &Device, plan: &ImagePlan) -> bool {
    let measured = (device.properties().driver_facts)
        .contains(&DriverFact::MultisampledDepthFasterOutOfGeneral);
    let depth = plan
        .format
        .block()
        .is_some_and(|block| block.depth.is_some());
    measured
        && depth
        && plan.samples.as_raw() > 1
        && !plan.usage.contains(ImageUsageFlags::TRANSIENT_ATTACHMENT)
}

/// One held image: its whole subresource range, and its layout as the batches submitted so far
/// leave it.
pub(crate) struct HeldImage {
    key: ResourceKey,
    raw: vk::Image,
    aspects: ImageAspectFlags,
    levels: u32,
    layers: u32,
    /// `GENERAL` from its first move on (`texture/first_move.rs`), which the device's first
    /// submission after its making records before any command. The scheduler's thread reads and
    /// writes it as it submits, under its lock.
    current: Mutex<ImageLayout>,
}

impl HeldImage {
    /// The held image `raw` of the texture known by `key`, planned as `plan`.
    pub(crate) fn of_plan(key: ResourceKey, raw: vk::Image, plan: &ImagePlan) -> HeldImage {
        let aspects = image_aspects(plan.format);
        HeldImage::new(key, raw, aspects, plan.levels, plan.layers)
    }

    /// The held image `raw` of the texture known by `key`, of `levels` and `layers` of
    /// `aspects`.
    pub(crate) fn new(
        key: ResourceKey,
        raw: vk::Image,
        aspects: ImageAspectFlags,
        levels: u32,
        layers: u32,
    ) -> HeldImage {
        HeldImage {
            key,
            raw,
            aspects,
            levels,
            layers,
            current: Mutex::new(ImageLayout::GENERAL),
        }
    }

    /// The layout the batches submitted so far leave the image in.
    fn current(&self) -> ImageLayout {
        *self.current.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A move of the whole image from `old` to `new`.
    fn transition(&self, old: ImageLayout, new: ImageLayout) -> Transition {
        Transition {
            image: self.raw,
            aspects: self.aspects,
            old,
            new,
        }
    }

    /// A move's use of the whole image, for the hazard tracker: a write in every stage.
    fn transition_use(&self) -> Use<ResourceKey> {
        Use::writes(
            self.key,
            true,
            UseRange::Image {
                aspects: self.aspects.into(),
                levels: 0..self.levels,
                layers: 0..self.layers,
                place: ImagePlace::WHOLE,
            },
            Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
        )
    }
}

/// The held images of one device, by key (the module docs).
#[derive(Default)]
pub(crate) struct HeldImages {
    images: Mutex<KeyMap<ResourceKey, Arc<HeldImage>>>,
    /// How many there are: none on a device without the fact, which then plans nothing.
    count: AtomicUsize,
}

impl HeldImages {
    /// `image` is held from now on.
    pub(crate) fn insert(&self, image: HeldImage) {
        let mut images = self.lock();
        images.insert(image.key, Arc::new(image));
        self.count.store(images.len(), Ordering::Relaxed);
    }

    /// The image known by `key` went.
    pub(crate) fn remove(&self, key: ResourceKey) {
        if self.count.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut images = self.lock();
        images.remove(&key);
        self.count.store(images.len(), Ordering::Relaxed);
    }

    /// Whether the texture known by `key` is held.
    pub(crate) fn holds(&self, key: ResourceKey) -> bool {
        self.count.load(Ordering::Relaxed) > 0 && self.lock().contains_key(&key)
    }

    /// The moves of a batch whose encoders, in order, named `encoders`' keys and held the keys
    /// beside them (`EncoderRecord`): for each encoder, the moves before it, and their uses.
    /// Nothing is allocated where the batch moves nothing, and nothing is looked at where the
    /// device holds no image.
    pub(crate) fn moves<'a>(
        &self,
        encoders: impl ExactSizeIterator<Item = (&'a BTreeSet<ResourceKey>, &'a [ResourceKey])>,
    ) -> BatchMoves {
        let count = encoders.len();
        let mut moves = BatchMoves::none();
        if self.count.load(Ordering::Relaxed) == 0 {
            return moves;
        }
        let images: Vec<Arc<HeldImage>> = self.lock().values().cloned().collect();
        let mut layouts: Vec<ImageLayout> = images.iter().map(|image| image.current()).collect();
        for (at, (named, held)) in encoders.enumerate() {
            for (index, image) in images.iter().enumerate() {
                if !named.contains(&image.key) && !held.contains(&image.key) {
                    continue;
                }
                let need = need(held.contains(&image.key));
                let current = layouts[index];
                if current != need {
                    if moves.transitions.is_empty() {
                        moves.transitions = vec![Vec::new(); count];
                        moves.uses = vec![Vec::new(); count];
                    }
                    moves.transitions[at].push(image.transition(current, need));
                    moves.uses[at].push(image.transition_use());
                    moves.released += usize::from(current == HELD);
                    layouts[index] = need;
                }
            }
        }
        moves.settled = images
            .into_iter()
            .zip(layouts)
            .filter(|(image, layout)| image.current() != *layout)
            .collect();
        moves
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, KeyMap<ResourceKey, Arc<HeldImage>>> {
        self.images.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// The layout an encoder that names a held image needs it in: [`HELD`] where it holds it, else
/// `GENERAL`.
pub(crate) fn need(held: bool) -> ImageLayout {
    match held {
        true => HELD,
        false => ImageLayout::GENERAL,
    }
}

/// The held images' moves of one batch ([`HeldImages::moves`]).
pub(crate) struct BatchMoves {
    /// For each encoder, the moves before it, in the order its command buffer of barriers records
    /// them; empty where the batch moves nothing.
    transitions: Vec<Vec<Transition>>,
    /// For each encoder, its moves' uses, which the hazard tracker plans before it; empty where
    /// the batch moves nothing.
    uses: Vec<Vec<Use<ResourceKey>>>,
    /// How many of the moves take an image out of [`HELD`].
    pub(crate) released: usize,
    /// Each image whose layout the batch changes, with the layout it leaves it in.
    settled: Vec<(Arc<HeldImage>, ImageLayout)>,
}

impl BatchMoves {
    /// No move.
    fn none() -> BatchMoves {
        BatchMoves {
            transitions: Vec::new(),
            uses: Vec::new(),
            released: 0,
            settled: Vec::new(),
        }
    }

    /// Whether the batch moves nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// The moves before the batch's encoder `at`.
    pub(crate) fn transitions(&self, at: usize) -> &[Transition] {
        self.transitions.get(at).map_or(&[], Vec::as_slice)
    }

    /// The uses of the moves before the batch's encoder `at`.
    pub(crate) fn uses(&self, at: usize) -> &[Use<ResourceKey>] {
        self.uses.get(at).map_or(&[], Vec::as_slice)
    }

    /// The batch was submitted: each image is in the layout it leaves it in, for the next.
    pub(crate) fn submitted(&self) {
        for (image, layout) in &self.settled {
            *image.current.lock().unwrap_or_else(|p| p.into_inner()) = *layout;
        }
    }
}
