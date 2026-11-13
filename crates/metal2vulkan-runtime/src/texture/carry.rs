//! What a device's recovery carries of the program's textures (`device/carry.rs`): each texture
//! made by a call the recovery can make again, the same texture object on the new `VkDevice`.
//!
//! - **What is made again.** A texture of `Device::new_texture` of any storage mode (its image
//!   new memory, its contents lost: it reads as a new texture of its descriptor reads, as no copy
//!   of its texels but its image's in the lost `VkDevice`'s memory exists; a texture buffer's
//!   made over its texels' buffer where the recovery carried that, so it reads what the buffer
//!   holds), one over a buffer the recovery carried (`Buffer::new_texture` and its kin: made over the buffer's bytes there, so it
//!   reads what the buffer holds), and a view of either (`new_texture_view` and its kin: of its
//!   parent's parts there). The call is made again on the new generation as the runtime's own
//!   texture ([`Remake`]), which takes no resource ID, and holds the texture's parts there
//!   (`TextureShared::next`): its handles reach them from then on (`Texture::s`). The texture
//!   object stays what it was: its `ObjectId`, resource ID, label, purgeable state, parent and
//!   handles; its ID's slot in the new generation's texture table and heap sets names its parts
//!   there (`heap_sets::place`). What its uses made of its image (views, shadows, faces,
//!   snapshots, memoryless backings, decoded twins) is made again there as its uses need it.
//! - **What stays the lost generation's**, refused where it meets the next
//!   (`RuntimeGap::MadeBeforeRecovery`), by kind ([`TextureLeft`]): a heap's texture or one over a
//!   heap's buffer, a texture over the caller's `VkImage`, a sparse texture, a shareable texture
//!   and one opened from its handles, a texture over a buffer the recovery left, and a view of
//!   any of them.
//! - **Kept as made.** Each texture of the program's is kept by its generation's registry as it
//!   is made (`Carry::texture`), held weakly; a call the recovery makes again sets how
//!   ([`Device::kept_texture`]), and carries the texture itself where the recovery took the
//!   generation's objects before that (`Device::carry_late`).
//! - **Its parts on the lost generation** stay with the texture until it goes: its images there
//!   and the lost `VkDevice` with them.

use super::super::buffer::Buffer;
use super::super::buffer_texture::{texture_buffer_over, TexelsHome};
use super::super::degradation::NoticesFor;
use super::super::heap_sets;
use super::super::linear_plan::BufferTextureCall;
use super::super::texture_descriptor::TextureDescriptor;
use super::super::values::StorageMode;
use super::super::view_plan::ViewRequest;
use super::super::ycbcr::IoSurfaceYCbCr;
use super::super::Device;
use super::{IdUse, Initial, Owner, Storage, Texture, TextureShared};
use std::sync::atomic::Ordering;
use std::sync::Weak;

/// How a recovery makes a program's texture again (`TextureShared::remake`; the module docs):
/// the call that made it, made again on the next generation as a texture of the runtime's own.
pub(crate) enum Remake {
    /// `Device::new_texture` of `descriptor` (its options resolved), starting as `initial`.
    Own {
        descriptor: TextureDescriptor,
        initial: Initial,
    },
    /// A texture over `buffer` (`Buffer::new_texture` and its kin): `call`, made through `form`,
    /// of `descriptor` at `offset` with `bytes_per_row`, over a surface of `ycbcr` facts where
    /// given.
    Over {
        buffer: Buffer,
        call: &'static str,
        descriptor: TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
        form: BufferTextureCall,
        ycbcr: Option<IoSurfaceYCbCr>,
    },
    /// A view of the texture's parent: `call` of `request`.
    View {
        call: &'static str,
        request: ViewRequest,
    },
}

/// What a recovery made again of a texture it carried ([`Texture::carry`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Remade {
    /// A `Shared` texture of its own image or texels: its contents lost.
    LostShared,
    /// A `Managed` one.
    LostManaged,
    /// A `Private` or memoryless one.
    LostPrivate,
    /// A texture over a buffer the recovery carried (a texture buffer of `Device::new_texture`
    /// over its texels' buffer too): it reads what the buffer holds.
    Over,
    /// A view, of its parent's parts there.
    View,
}

impl Remade {
    /// A texture of `descriptor` made again with its contents lost, by its storage mode.
    fn lost(descriptor: &TextureDescriptor) -> Remade {
        match descriptor.resource_options.storage_mode() {
            Ok(StorageMode::Shared) => Remade::LostShared,
            Ok(StorageMode::Managed) => Remade::LostManaged,
            _ => Remade::LostPrivate,
        }
    }
}

/// Why a texture stays the lost generation's ([`Texture::carry`]; the module docs). A view
/// stays where its parent stays, and for its parent's reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextureLeft {
    /// A heap's texture, or one over a heap's buffer.
    Heap,
    /// Over the caller's `VkImage` (`texture/wrapped.rs`), an object of the lost `VkDevice`.
    Caller,
    /// A sparse heap's texture.
    Sparse,
    /// A shareable texture (`Device::new_shared_texture`), or one opened from its handles, whose
    /// memory another process may import.
    Shared,
    /// Over a buffer the recovery left.
    Buffer,
    /// Its making on the new generation failed.
    Failed,
    /// Made by a call no recovery makes again.
    Other,
}

/// A texture of the program's, held weakly by its generation's registry (`Carry::texture`).
pub(crate) struct WeakTexture(Weak<TextureShared>);

impl WeakTexture {
    /// `texture`, held weakly.
    pub(crate) fn of(texture: &Texture) -> WeakTexture {
        WeakTexture(texture.downgrade())
    }

    /// Whether the texture lives.
    pub(crate) fn lives(&self) -> bool {
        self.0.strong_count() > 0
    }

    /// The texture, while it lives.
    pub(crate) fn upgrade(&self) -> Option<Texture> {
        Texture::upgrade(&self.0)
    }
}

impl Device {
    /// `texture`, a program's texture this generation made by a call a recovery makes again,
    /// kept with how (`remake`; the module docs): carried by its maker to the generation that
    /// serves where this generation's recovery took its objects before then, read under the
    /// lock the take sets that under, as a heap's maker carries it (`Device::carry_late`).
    pub(crate) fn kept_texture(&self, texture: Texture, remake: Remake) -> Texture {
        // Set once: the call that made it.
        let _ = texture.0.remake.set(remake);
        if self.shared().carry.late_texture() {
            self.carry_late(|serving| texture.carry(serving));
        }
        texture
    }
}

impl Texture {
    /// Whether the texture is a view of another (`parentTexture`).
    pub(crate) fn is_view(&self) -> bool {
        self.0.parent.is_some()
    }

    /// Carry the texture, made on a generation of its device that a recovery leaves, to `to`,
    /// the generation that recovery made (the module docs): its call made again there as the
    /// runtime's own texture, whose parts the texture's handles reach from then on, its ID's
    /// slot there naming them; a view's parent first. What was made again, `None` where it was
    /// carried there already; or why it stays the lost generation's.
    pub(crate) fn carry(&self, to: &Device) -> Result<Option<Remade>, TextureLeft> {
        const CALL: &str = "new_texture";
        let generation = std::ptr::from_ref(to.shared()).addr();
        // Every carried object's parts this reads (the buffer a texture is over, a view's parent,
        // the texture's own as its slot is written) are those on `to`, where it has some, whatever
        // command buffer's work or held parts the thread is inside.
        let _scope = NoticesFor::of_generation(generation);
        // The generation it was carried to last: a later recovery carries it on from there.
        let mut last: &TextureShared = &self.0;
        while let Some(next) = last.next.get() {
            last = &*next.0;
        }
        if last.generation == generation {
            // Carried already, by its maker or as a view's parent.
            return Ok(None);
        }
        let Some(remake) = self.0.remake.get() else {
            return Err(self.left());
        };
        let (made, remade) = match remake {
            Remake::Own {
                descriptor,
                initial,
            } => match self.texels_carried_to(to) {
                // A texture buffer, over its texels' buffer where the recovery carried it.
                Some(buffer) => {
                    let made = (descriptor.check(to.properties().texture_rules()))
                        .map_err(|refused| refused.error(CALL))
                        .and_then(|layout| {
                            let (id_use, home) = (IdUse::Runtime, TexelsHome::Buffer);
                            texture_buffer_over(buffer, descriptor, &layout, id_use, home)
                        });
                    (made, Remade::Over)
                }
                None => {
                    let made = to.texture_as(CALL, descriptor, (IdUse::Runtime, *initial));
                    (made, Remade::lost(descriptor))
                }
            },
            Remake::Over {
                buffer,
                call,
                descriptor,
                offset,
                bytes_per_row,
                form,
                ycbcr,
            } => {
                if buffer.heap().is_some() {
                    return Err(TextureLeft::Heap);
                }
                if !buffer.generation().same(to) {
                    return Err(TextureLeft::Buffer);
                }
                let over = (*offset, *bytes_per_row);
                let made = buffer.texture_over_again(call, descriptor, over, *form, *ycbcr);
                (made, Remade::Over)
            }
            Remake::View { call, request } => {
                let Some(parent) = &self.0.parent else {
                    return Err(TextureLeft::Other);
                };
                parent.carry(to)?;
                (parent.view_as(call, request, IdUse::Runtime), Remade::View)
            }
        };
        let made = made.map_err(|_| TextureLeft::Failed)?;
        if !self.is_view() {
            self.carry_purgeable_state(&made);
        }
        if last.next.set(made).is_err() {
            // Another carry of it to `to` came first.
            return Ok(None);
        }
        self.0.carried.store(true, Ordering::Release);
        // Among the program's resources there, its ID's slot there naming its parts there.
        if !self.is_view() {
            to.shared().owners.texture(self);
        }
        if self.0.id.id().0 != 0 {
            heap_sets::place(to, self);
        }
        // Kept there for a later recovery to carry on, as a texture made there is.
        if to.shared().carry.texture(self) {
            to.carry_late(|serving| self.carry(serving));
        }
        Ok(Some(remade))
    }

    /// The buffer the texels of the texture, a texture buffer of `Device::new_texture`, lie in,
    /// where the recovery carried it to `to`; `None` for any other texture, or where it left it.
    fn texels_carried_to(&self, to: &Device) -> Option<Buffer> {
        match self.storage() {
            Storage::Texels(texels)
                if texels.home == TexelsHome::Buffer && texels.buffer.generation().same(to) =>
            {
                Some(texels.buffer.clone())
            }
            _ => None,
        }
    }

    /// Why the texture, which no call a recovery makes again made, stays the lost generation's.
    fn left(&self) -> TextureLeft {
        if self.root().0.heap.is_some() {
            return TextureLeft::Heap;
        }
        match self.storage() {
            Storage::Image(image) => match &image.owner {
                Owner::Heap(_) | Owner::HeapShadow(_) | Owner::HeapUnaliased(_) => {
                    TextureLeft::Heap
                }
                Owner::Caller(_) => TextureLeft::Caller,
                Owner::Sparse(_) => TextureLeft::Sparse,
                Owner::Runtime {
                    shareable: true, ..
                } => TextureLeft::Shared,
                Owner::Buffer { .. } => TextureLeft::Buffer,
                Owner::Runtime { .. } | Owner::ViewShadow(_) => TextureLeft::Other,
            },
            Storage::Texels(texels) if texels.home == TexelsHome::Shared => TextureLeft::Shared,
            Storage::Texels(texels)
            | Storage::Alias { texels, .. }
            | Storage::Shadow { texels, .. } => match texels.buffer.heap() {
                Some(_) => TextureLeft::Heap,
                None => TextureLeft::Buffer,
            },
        }
    }
}
