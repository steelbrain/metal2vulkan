//! A texture's label and purgeable state (`MTLResource`'s `label` and `setPurgeableState:` on
//! `MTLTexture`), as Metal answered them on an Apple M3 Max and an M4:
//!
//! - **Labels** are each texture object's own. A new texture's and a view's start `None`, and
//!   setting one changes no other object's (a view's not its texture's, nor the reverse). A
//!   texture opened from a shared texture's handle starts with that texture's label at the open,
//!   and the two change apart after it.
//! - **Purgeable state** is kept, as a buffer's is (`attributes.rs`), only by a texture with an
//!   allocation of its own:
//!   - a shareable texture (`Device::new_shared_texture`) of any size and kind, which shares one
//!     state with every texture opened from its handles;
//!   - any other texture the device made of its own memory whose allocation is at least
//!     `PURGEABLE_ALLOCATION` (16 KiB), by the runtime's model of Apple's layout
//!     (`texture_allocation`): Metal suballocates the smaller ones.
//!
//!   `Empty` keeps the texels, as Metal's did in every storage mode; the runtime never purges.
//!   Every other texture answers `NonVolatile` to every request and keeps its state: a smaller
//!   texture, a view (of a shareable texture too: a view's answers never follow its texture's), a
//!   texture over a buffer (whose own state the texture does not change), a memoryless texture
//!   (whose `allocatedSize` is 0), and a texture over the caller's image, as a buffer over the
//!   caller's memory does.
//! - **A texture in a heap** (a heap's texture, a view of one, a texture over a heap's buffer, a
//!   sparse texture) answers `NonVolatile` to every request, a query included, and changes
//!   nothing, neither its texels nor its heap's state, as plain Metal answers where its
//!   validation stops each ("Setting purgeability state directly on heap sub-allocated resources
//!   is not allowed"), as a buffer in a heap does.

use super::super::attributes::{set_purgeable, texture_allocation, PURGEABLE_ALLOCATION};
use super::super::buffer::Buffer;
use super::super::buffer_texture::TexelsHome;
use super::super::copy_plan::block_extent;
use super::super::device::lock;
use super::super::error::Error;
use super::super::values::StorageMode;
use super::super::PurgeableState;
use super::{Image, Owner, Storage, Texture};
use metal2vulkan::vk::ImageType;
use std::sync::Mutex;

/// Where a texture's purgeable state is kept, for a texture whose state Metal keeps.
enum Kept<'a> {
    /// An image's, of a texture the device made of its own memory.
    Image(&'a Mutex<PurgeableState>),
    /// The buffer a shareable texture buffer's texels are the bytes of, made for it.
    Texels(&'a Buffer),
}

impl Texture {
    /// `setLabel:`: this texture object's label; no other object's changes (the module docs).
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.label) = Some(label.to_owned());
    }

    /// `label`: `None` until one is set, except for a texture opened from a shared texture's
    /// handle, which starts with that texture's label at the open.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.label).clone()
    }

    /// `setPurgeableState:`: the state before the call, or the state now for `KeepCurrent`,
    /// where Metal keeps the texture's state (a shareable texture and the textures opened from
    /// its handles, sharing one; a texture of the device's own memory whose allocation is at
    /// least 16 KiB); `NonVolatile` to every request otherwise, the state unchanged. `Empty`
    /// keeps the texels, as Metal's do, and the runtime never purges (the module docs).
    ///
    /// A texture in a heap, a view of one, and a texture over a heap's buffer answer
    /// `NonVolatile` to every request and change nothing, as plain Metal answers where its
    /// validation stops each, a query included.
    pub fn set_purgeable_state(&self, state: PurgeableState) -> Result<PurgeableState, Error> {
        if self.heap().is_some() {
            return Ok(PurgeableState::NonVolatile);
        }
        Ok(match self.kept() {
            Some(Kept::Image(cell)) => {
                let mut current = lock(cell);
                let (answer, next) = set_purgeable(*current, state);
                *current = next;
                answer
            }
            Some(Kept::Texels(buffer)) => buffer.attributes().set_purgeable_state(state),
            None => PurgeableState::NonVolatile,
        })
    }

    /// Where the texture's purgeable state is kept: `None` for a texture Metal answers
    /// `NonVolatile` for (the module docs).
    fn kept(&self) -> Option<Kept<'_>> {
        if self.0.parent.is_some() {
            return None;
        }
        match self.storage() {
            Storage::Image(image) => match &image.owner {
                Owner::Runtime {
                    shareable,
                    purgeable,
                    ..
                } if *shareable || own_allocation(image) => Some(Kept::Image(purgeable)),
                _ => None,
            },
            Storage::Texels(texels) if texels.home == TexelsHome::Shared => {
                Some(Kept::Texels(&texels.buffer))
            }
            Storage::Texels(_) | Storage::Alias { .. } | Storage::Shadow { .. } => None,
        }
    }
}

/// Whether Metal gives the texture of `image`, of the device's own memory, an allocation of its
/// own: not memoryless, and at least `PURGEABLE_ALLOCATION` by `texture_allocation`.
fn own_allocation(image: &Image) -> bool {
    if matches!(image.options.storage_mode(), Ok(StorageMode::Memoryless)) {
        return false;
    }
    let plan = image.plan();
    let block_bytes = plan.format.block().map_or(0, |block| u64::from(block.size));
    let bytes = texture_allocation(
        block_extent(plan.pixel_format).unwrap_or([1, 1]),
        block_bytes,
        plan.extent,
        plan.image_type == ImageType::TYPE_3D,
        plan.levels,
        plan.layers,
        plan.samples.as_raw(),
    );
    bytes >= PURGEABLE_ALLOCATION
}
