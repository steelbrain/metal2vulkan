//! How a device's buffers and images are shared among the queue families it submits to.
//!
//! A device that takes queues of several families (every family equivalent to the first, as
//! MoltenVK's four of one queue each: `device/create.rs`, `queue_plan`) submits each logical
//! queue's command buffers to its own queue, so any buffer or image may be used by queues of
//! every family. Vulkan keeps an exclusive resource's contents only in the family that owns it,
//! and the runtime transfers no ownership: every buffer and image it makes is concurrent among
//! all the device's families. A device of one family keeps them exclusive. MoltenVK reads no
//! sharing mode, so the concurrent form costs nothing there.
//!
//! Where the device reserved a staging queue (`device/staging.rs`: a copy engine, or a queue of
//! compute work beside the device's own), every buffer it makes is concurrent among its families
//! and the staging queue's too: a new buffer's zeros or
//! staged bytes made at its hand-out run there (`Buffer::settle_first_work`), behind no command of
//! the program's, and no buffer changes owner. Its images keep the device's families (but those
//! that copy on the staging queue, `Device::staging_sharing`).

use ash::vk;

/// The sharing of a device's buffers and images: its queue families, the primary first.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sharing<'a> {
    families: &'a [u32],
    /// The families its buffers are shared among: `families`, then the staging queue's where the
    /// device has one (the module docs).
    buffers: &'a [u32],
}

impl<'a> Sharing<'a> {
    /// The sharing of a device that submits to `families` (at least one).
    pub(crate) fn new(families: &'a [u32]) -> Sharing<'a> {
        Sharing {
            families,
            buffers: families,
        }
    }

    /// The sharing of a device that submits to `families` (at least one), whose buffers are
    /// shared among `buffers`: `families`, then its staging queue's (the module docs).
    pub(crate) fn with_buffers(families: &'a [u32], buffers: &'a [u32]) -> Sharing<'a> {
        Sharing { families, buffers }
    }

    /// The queue families it shares among.
    pub(crate) fn families(self) -> &'a [u32] {
        self.families
    }

    /// Whether resources are concurrent: the device submits to more than one family.
    fn concurrent(self) -> bool {
        self.families.len() > 1
    }

    /// `info` with the device's sharing of buffers: concurrent among the buffers' families where
    /// they are more than one.
    pub(crate) fn buffer<'b>(self, info: vk::BufferCreateInfo<'b>) -> vk::BufferCreateInfo<'b>
    where
        'a: 'b,
    {
        match self.buffers.len() > 1 {
            true => info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(self.buffers),
            false => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
        }
    }

    /// `info` with the device's sharing.
    pub(crate) fn image<'b>(self, info: vk::ImageCreateInfo<'b>) -> vk::ImageCreateInfo<'b>
    where
        'a: 'b,
    {
        match self.concurrent() {
            true => info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(self.families),
            false => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
        }
    }

    /// Whether a resource of the caller's, made `shared` among `families` (those of a concurrent
    /// one; ignored for an exclusive one), may be used on every queue of the device: on a device
    /// of one family, an exclusive resource (the caller's family is the device's) or a concurrent
    /// one among the device's family; on a device of several, a concurrent one among all of
    /// them, as the runtime transfers no ownership.
    pub(crate) fn admits(self, shared: vk::SharingMode, families: &[u32]) -> bool {
        match shared == vk::SharingMode::CONCURRENT {
            true => self.families.iter().all(|family| families.contains(family)),
            false => !self.concurrent(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Sharing;
    use ash::vk;

    #[test]
    fn one_family_keeps_resources_exclusive() {
        let sharing = Sharing::new(&[0]);
        let info = sharing.buffer(vk::BufferCreateInfo::default());
        assert!(info.sharing_mode == vk::SharingMode::EXCLUSIVE);
        assert_eq!(info.queue_family_index_count, 0);
        let info = sharing.image(vk::ImageCreateInfo::default());
        assert!(info.sharing_mode == vk::SharingMode::EXCLUSIVE);
        assert!(sharing.admits(vk::SharingMode::EXCLUSIVE, &[]));
        assert!(sharing.admits(vk::SharingMode::CONCURRENT, &[0, 2]));
        assert!(!sharing.admits(vk::SharingMode::CONCURRENT, &[1, 2]));
    }

    #[test]
    fn several_families_make_resources_concurrent_among_all() {
        let families = [0, 1, 2, 3];
        let sharing = Sharing::new(&families);
        let info = sharing.buffer(vk::BufferCreateInfo::default());
        assert!(info.sharing_mode == vk::SharingMode::CONCURRENT);
        assert_eq!(info.queue_family_index_count, 4);
        let info = sharing.image(vk::ImageCreateInfo::default());
        assert!(info.sharing_mode == vk::SharingMode::CONCURRENT);
        assert_eq!(info.queue_family_index_count, 4);
        assert!(!sharing.admits(vk::SharingMode::EXCLUSIVE, &[]));
        assert!(!sharing.admits(vk::SharingMode::CONCURRENT, &[0, 1, 2]));
        assert!(sharing.admits(vk::SharingMode::CONCURRENT, &[3, 2, 1, 0]));
    }

    #[test]
    fn a_staging_family_makes_buffers_concurrent_and_leaves_images_exclusive() {
        let buffers = [0, 2];
        let sharing = Sharing::with_buffers(&buffers[..1], &buffers);
        let info = sharing.buffer(vk::BufferCreateInfo::default());
        assert!(info.sharing_mode == vk::SharingMode::CONCURRENT);
        assert_eq!(info.queue_family_index_count, 2);
        let info = sharing.image(vk::ImageCreateInfo::default());
        assert!(info.sharing_mode == vk::SharingMode::EXCLUSIVE);
        assert!(sharing.admits(vk::SharingMode::EXCLUSIVE, &[]));
    }
}
