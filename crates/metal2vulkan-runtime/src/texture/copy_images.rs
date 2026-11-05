//! The images the runtime's multisample copies go through (`sample_interleave.rs`: an interleave
//! copy's wide and scratch images; `sample_copy.rs`: its draws' images), lent again wherever no
//! use of one can overlap another's.
//!
//! - **Within a copy** ([`Reused`]): a later step of a copy (a band of an interleave copy; the
//!   next aspect or slice of a heap texture's moves) takes an image an earlier step of it made
//!   that [covers] the one it needs, where no image of the step took it already. A copy's
//!   steps are recorded one after another into one command buffer, each of their commands after a
//!   barrier from every command before it to every command after it (`Barrier::everything`, in
//!   `InterleaveCopy::record` and `SampleCopy::record`), and a step writes each texel of an image
//!   it reads before it reads it: so a later step's use of an image follows an earlier one's
//!   whole, and reads nothing it left. Before, each band made images of its own, so a copy held
//!   the images of all its bands at once, about twice the texture's bytes, and its draws' too,
//!   which lie at the texture's coordinates, each band's from the origin to its end.
//! - **Between the copies of a queue** ([`CopyImages`]): a heap's multisample texture's moves
//!   between its image and Metal's tiles (`texture/heap_tiled.rs`) are planned as an encoder
//!   takes a copy, and take their images from those the encoder's queue keeps, made by an earlier
//!   copy of any texture of the queue, as its queue keeps a merge's kept images
//!   (`kept_image.rs`). Each use of them is one copy's steps, recorded at once into a
//!   `VkCommandBuffer` of the queue's `VkQueue`, each of their commands after such a barrier, whose
//!   first synchronization scope is every command before it in submission order on that queue,
//!   in earlier command buffers too ("The first synchronization scope and access scope of each
//!   memory dependency defined by pDependencyInfo are applied to operations that occurred
//!   earlier in submission order", `vkCmdPipelineBarrier2`): so two uses never overlap, in
//!   whatever order their command buffers go, and neither reads what the other left. The moves'
//!   other objects (their descriptor sets and framebuffers) are the use's own, which its command
//!   buffer holds until it completes, with the images it took. Before, a texture planned its
//!   moves as it was made and held their images for its life, each way: a 4096 x 2160
//!   `Depth32Float_Stencil8` texture of 4 samples (170 MiB) held about 1.2 GiB of them.
//! - **Covered**: an image serves a use of an image of the same descriptor and plan but a smaller
//!   extent, the one of the fewest bytes first: each use names its region in it (its copies'
//!   regions, its draws' framebuffer and the texels they fetch at their own coordinates) and
//!   reads no texel past it. Not a storage image: the split's dispatch is bounded by its wide
//!   image's size (`bind::sample_interleave`), so it takes one of exactly its size.
//! - **Bounded**: a queue keeps at most a sixty-fourth of the device's largest device-local heap of
//!   images, as a GPU pool's block (`gpu_pools::device_share`), letting go of those a use took
//!   least recently first; one let go of goes once the command buffers that used it complete. A
//!   copy whose images pass the bound makes them again at its next use.
//! - **Not cleared**: a use writes each texel it reads, as above.
//!
//! `WorkCounts::heap_sample_move_images_made` and `heap_sample_move_images_lent` count the images
//! the moves made and those their queue lent.

use super::super::device::lock;
use super::super::texture_descriptor::TextureDescriptor;
use super::super::Device;
use super::{ImageMaker, ImagePlan, Owner, Storage, Texture};
use metal2vulkan::vk::ImageUsageFlags;
use std::sync::Mutex;

/// What a kept image is kept by: its descriptor and plan, which a texture taking it needs alike.
type ImageKey = (TextureDescriptor, ImagePlan);

/// Whether an image of `have` serves where one of `want` is needed (the module docs): the same
/// descriptor and plan, or the same but for an extent at least as large where `want` is no
/// storage image.
fn covers(have: &ImageKey, want: &ImageKey) -> bool {
    let ((descriptor, plan), (wanted, planned)) = (have, want);
    if have == want {
        return true;
    }
    let larger = |have: &[u32; 3], want: &[u32; 3]| have.iter().zip(want).all(|(h, w)| h >= w);
    let sized = TextureDescriptor {
        width: wanted.width,
        height: wanted.height,
        ..*descriptor
    };
    let extent = ImagePlan {
        extent: planned.extent,
        ..plan.clone()
    };
    !planned.usage.contains(ImageUsageFlags::STORAGE)
        && sized == *wanted
        && extent == *planned
        && larger(&plan.extent, &planned.extent)
}

/// The bytes of memory of `texture`'s image, where the runtime placed it in memory of its own; 0
/// for any other.
fn image_bytes(texture: &Texture) -> u64 {
    match &texture.0.storage {
        Storage::Image(image) => match &image.owner {
            Owner::Runtime { placement, .. } => placement.size(),
            _ => 0,
        },
        _ => 0,
    }
}

/// A maker that lends each later step of a copy an image an earlier step of it took (the module
/// docs) that covers the one it needs, where no image of the step took it already, the one of the
/// fewest bytes; it takes the others from `inner`, which holds each as it says.
/// [`next`](Self::next) begins a step.
pub(crate) struct Reused<'a> {
    inner: &'a dyn ImageMaker,
    /// Each image taken so far, its bytes, and whether the current step took it.
    taken: Mutex<Vec<(ImageKey, Texture, u64, bool)>>,
}

impl<'a> Reused<'a> {
    pub(crate) fn new(inner: &'a dyn ImageMaker) -> Reused<'a> {
        Reused {
            inner,
            taken: Mutex::new(Vec::new()),
        }
    }

    /// Begin the next step: each image taken so far may be taken again.
    pub(crate) fn next(&self) {
        for (_, _, _, busy) in lock(&self.taken).iter_mut() {
            *busy = false;
        }
    }
}

impl ImageMaker for Reused<'_> {
    fn device(&self) -> &Device {
        self.inner.device()
    }

    fn lend(&self, descriptor: &TextureDescriptor, plan: &ImagePlan) -> Option<Texture> {
        let key = (*descriptor, plan.clone());
        {
            let mut taken = lock(&self.taken);
            let free = (taken.iter_mut())
                .filter(|(of, _, _, busy)| !*busy && covers(of, &key))
                .min_by_key(|(_, _, bytes, _)| *bytes);
            if let Some((_, texture, _, busy)) = free {
                *busy = true;
                return Some(texture.clone());
            }
        }
        let texture = self.inner.lend(descriptor, plan)?;
        let bytes = image_bytes(&texture);
        // Kept by the plan it was lent for, whose extent it covers.
        let key = (texture_key(&texture, descriptor)).unwrap_or(key);
        lock(&self.taken).push((key, texture.clone(), bytes, true));
        Some(texture)
    }

    fn hold(&self, texture: Texture, descriptor: &TextureDescriptor, plan: &ImagePlan) {
        self.inner.hold(texture.clone(), descriptor, plan);
        let bytes = image_bytes(&texture);
        lock(&self.taken).push(((*descriptor, plan.clone()), texture, bytes, true));
    }
}

/// The key of `texture`, an image the runtime made of a descriptor as `descriptor` but for its
/// width and height, which are its plan's: what it covers.
fn texture_key(texture: &Texture, descriptor: &TextureDescriptor) -> Option<ImageKey> {
    let plan = texture.image_plan().ok()?;
    let [width, height, _] = plan.extent;
    let descriptor = TextureDescriptor {
        width: width.into(),
        height: height.into(),
        ..*descriptor
    };
    Some((descriptor, plan.clone()))
}

/// The images one command queue keeps for its heap textures' moves (the module docs), each with
/// the use that took it last. A leaf lock: the images let go of go after it.
#[derive(Default)]
pub(crate) struct CopyImages(Mutex<Kept<ImageKey, Texture>>);

impl CopyImages {
    /// The maker of one use of the images: the moves of the copies an encoder takes at once, the
    /// steps of each taking what the earlier ones took through a [`Reused`] over it.
    pub(crate) fn maker<'a>(&'a self, device: &'a Device) -> UseMaker<'a> {
        UseMaker {
            device,
            kept: self,
            stamp: lock(&self.0).begin(),
            taken: Mutex::new(Vec::new()),
        }
    }
}

/// One use's maker over its queue's images ([`CopyImages::maker`]): it lends the queue's image of
/// the fewest bytes that covers the one needed and the use has not taken, and keeps each it
/// makes, within the device's bound (the module docs). The caller holds the textures until the
/// command buffer that records the use completes.
pub(crate) struct UseMaker<'a> {
    device: &'a Device,
    kept: &'a CopyImages,
    stamp: u64,
    /// The images the use took.
    taken: Mutex<Vec<Texture>>,
}

impl ImageMaker for UseMaker<'_> {
    fn device(&self) -> &Device {
        self.device
    }

    fn lend(&self, descriptor: &TextureDescriptor, plan: &ImagePlan) -> Option<Texture> {
        let key = (*descriptor, plan.clone());
        let mut taken = lock(&self.taken);
        let fits = |have: &ImageKey| covers(have, &key);
        let texture = lock(&self.kept.0).lend(fits, &taken, self.stamp)?;
        taken.push(texture.clone());
        self.device.shared().counts.heap_sample_move_image(true);
        Some(texture)
    }

    fn hold(&self, texture: Texture, descriptor: &TextureDescriptor, plan: &ImagePlan) {
        self.device.shared().counts.heap_sample_move_image(false);
        lock(&self.taken).push(texture.clone());
        let bytes = image_bytes(&texture);
        let key = (*descriptor, plan.clone());
        let limit = self.device.shared().copy_images_bytes;
        let gone = lock(&self.kept.0).keep(key, texture, bytes, self.stamp, limit);
        drop(gone);
    }
}

/// The images a queue keeps ([`CopyImages`]), each of its key and bytes, by the use that took it
/// last, and the uses so far, the clock those read.
struct Kept<K, T> {
    images: Vec<Entry<K, T>>,
    uses: u64,
}

impl<K, T> Default for Kept<K, T> {
    fn default() -> Kept<K, T> {
        Kept {
            images: Vec::new(),
            uses: 0,
        }
    }
}

/// A kept image, its key and bytes, and the use that took it last.
struct Entry<K, T> {
    key: K,
    image: T,
    bytes: u64,
    used: u64,
}

impl<K, T: Clone + PartialEq> Kept<K, T> {
    /// Begin a use: its stamp.
    fn begin(&mut self) -> u64 {
        self.uses += 1;
        self.uses
    }

    /// The image of the fewest bytes whose key `fits` but those of `taken`, taken by the use
    /// `stamp`.
    fn lend(&mut self, fits: impl Fn(&K) -> bool, taken: &[T], stamp: u64) -> Option<T> {
        let entry = (self.images.iter_mut())
            .filter(|entry| fits(&entry.key) && !taken.contains(&entry.image))
            .min_by_key(|entry| entry.bytes)?;
        entry.used = stamp;
        Some(entry.image.clone())
    }

    /// Keep `image` of `key`, `bytes` long, made for the use `stamp`; the images let go of to keep
    /// the rest within `limit` bytes, those used least recently first (the caller drops them
    /// without the lock).
    fn keep(&mut self, key: K, image: T, bytes: u64, stamp: u64, limit: u64) -> Vec<T> {
        self.images.push(Entry {
            key,
            image,
            bytes,
            used: stamp,
        });
        let mut gone = Vec::new();
        while self.bytes() > limit {
            let oldest = (0..self.images.len()).min_by_key(|&at| self.images[at].used);
            let Some(at) = oldest else {
                break;
            };
            gone.push(self.images.remove(at).image);
        }
        gone
    }

    /// The bytes of the images kept.
    fn bytes(&self) -> u64 {
        self.images.iter().map(|entry| entry.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::Kept;

    /// A use takes the image of the fewest bytes whose key fits, never one it took, and none
    /// where no key fits.
    #[test]
    fn a_use_takes_the_smallest_image_that_fits_it_has_not_taken() {
        let mut kept = Kept::<u32, u32>::default();
        let first = kept.begin();
        assert!(kept.keep(7, 70, 30, first, 100).is_empty());
        assert!(kept.keep(7, 71, 10, first, 100).is_empty());
        assert!(kept.keep(8, 80, 10, first, 100).is_empty());
        let second = kept.begin();
        let seven = |key: &u32| *key == 7;
        assert_eq!(kept.lend(seven, &[], second), Some(71));
        assert_eq!(kept.lend(seven, &[71], second), Some(70));
        assert_eq!(kept.lend(seven, &[70, 71], second), None);
        assert_eq!(kept.lend(|key| *key == 9, &[], second), None);
        assert_eq!(kept.lend(|key| *key >= 7, &[70, 71], second), Some(80));
    }

    /// Past the limit the images used least recently go first, the use's own last; the bytes kept
    /// never pass the limit once one is let go of.
    #[test]
    fn past_the_limit_the_images_used_least_recently_go() {
        let mut kept = Kept::<u32, u32>::default();
        let first = kept.begin();
        assert!(kept.keep(1, 10, 40, first, 100).is_empty());
        let second = kept.begin();
        assert!(kept.keep(2, 20, 40, second, 100).is_empty());
        let third = kept.begin();
        assert_eq!(kept.lend(|key| *key == 1, &[], third), Some(10));
        // Image 20 is the one used least recently now.
        assert_eq!(kept.keep(3, 30, 40, third, 100), vec![20]);
        assert_eq!(kept.bytes(), 80);
        // One past the limit alone goes too, once the rest went.
        assert_eq!(kept.keep(4, 40, 200, kept.uses, 100), vec![10, 30, 40]);
        assert_eq!(kept.bytes(), 0);
    }
}
