//! A shadow left unmade until a reach needs it: a program's texture over a buffer's bytes whose
//! image the device refused for a reason that leaves its rows in place (`texture/texel_rows.rs`)
//! is a shadow (`shadow.rs`) to every reach but a slot of a function that only reads, writes or
//! asks the size of it, which binds its rows. So its shadow is planned and checked as the texture
//! is made (`Device::storage_over_buffer`: its image's plan and usage, its copies' views, how it
//! starts), and made at the first reach of the texture's storage ([`Texture::storage`]): a slot
//! that binds the shadow, a residency (`use_resource`), a pass's attachment, a blit of its image, a
//! view, its `VkImage` handed out, a CPU access. A texture only ever bound through its rows makes
//! none (`WorkCounts::shadows_left_unmade`, `shadows_made_at_use`).
//!
//! - **Which.** A texture whose rows are in place (`TextureFacts::texel_rows`), its shadow not one
//!   that waits for its pages' in-place verdict (`texture/storage_form.rs`): every other shadow is
//!   made with its texture.
//! - **Until it is made** the texture holds no image. Its resource ID holds no slot in the
//!   device's texture table and heap sets ([`heap_sets::place`] places it as its shadow is made),
//!   so a read through its ID that no command named reads the neutral, where a shadow made with the
//!   texture read whatever its image held: nothing copies into a shadow no command names
//!   (`shadow.rs`). A use of the texture names the key planned with its shadow
//!   ([`ShadowRecipe::identity`]), so a use before the making and one after name one resource; it
//!   has no copies, as nothing is copied into or out of a shadow before a command uses it.
//! - **The making** is the texture's making's, once, under the texture's lock: the image as its
//!   memory lies (or as the plan says it starts), its views, its copies, its notice
//!   (`shadowed-texture`) and shadow log line, its counts; then its slot is placed and a detach of
//!   its bytes reaches its image (`NotedTexture`), as at a texture's making. A slot that binds it
//!   copies the bytes in before its command, as before the first command that uses any shadow:
//!   the copy is a command of the encoder's own at that point (`encoder.rs`), so it reads what the
//!   commands before it wrote through the rows, ordered after them (a rows binding's use is a use
//!   of the bytes, `Texture::rows_use`). A kept list of resident textures' shadows is listed
//!   again (`DeviceShared::usage_shadows_made`, `shadow::ResidentShadows`).
//! - **Out of memory** at the making fails the call that reached it, with the error the texture's
//!   making would have given; the texture stays unmade, and a later reach makes it again.

use super::{Storage, Texture};
use crate::buffer_texture::Texels;
use crate::device::lock;
use crate::error::Error;
use crate::heap_sets;
use crate::image_plan::ImagePlan;
use crate::resource::Identity;
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{ShadowMergeRead, ShadowTexelView};
use metal2vulkan::vk::Format;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

/// A texture over a buffer's shadow as its making planned and checked it, to make
/// (`Device::made_shadow`).
pub(in crate::texture) struct ShadowRecipe {
    /// The call that made the texture, which its notice names.
    pub(in crate::texture) call: &'static str,
    /// The buffer's bytes the texture is over.
    pub(in crate::texture) texels: Arc<Texels>,
    pub(in crate::texture) descriptor: TextureDescriptor,
    pub(in crate::texture) layout: TextureLayout,
    /// The shadow image's plan, its undeclared usage and its views' usage in it.
    pub(in crate::texture) plan: ImagePlan,
    /// Its copies' views: the integer view a dispatch copies back through
    /// (`shadow_copy_back.rs`), or the view of its own format its merges read
    /// (`shadow_merge.rs`).
    pub(in crate::texture) views: (
        Option<(Format, ShadowTexelView, bool)>,
        Option<ShadowMergeRead>,
    ),
    /// The aspects its copies move.
    pub(in crate::texture) aspects: vk::ImageAspectFlags,
    /// How its image starts, where not as its memory lies.
    pub(in crate::texture) start: Option<super::Start>,
    /// The image's identity, its key taken as the texture is made: a use of the texture names
    /// it before the image is made too.
    pub(in crate::texture) identity: Identity,
    /// Whether it is a device copy of host pages, which degrades nothing.
    pub(in crate::texture) device_copy: bool,
    /// Why the texture is a shadow: its notice's reason.
    pub(in crate::texture) why: String,
}

/// What a texture over a buffer is made over (`Device::storage_over_buffer`): its storage, or the
/// shadow it plans, to make.
pub(in crate::texture) enum Over {
    Storage(Storage),
    Shadow(Box<ShadowRecipe>),
}

/// A shadow left unmade (the module docs): its recipe, and its storage once made.
pub(in crate::texture) struct LazyShadow {
    pub(in crate::texture) recipe: ShadowRecipe,
    made: OnceLock<Storage>,
    /// Held while the shadow is made, so one reach makes it.
    making: Mutex<()>,
}

impl LazyShadow {
    pub(in crate::texture) fn new(recipe: ShadowRecipe) -> LazyShadow {
        LazyShadow {
            recipe,
            made: OnceLock::new(),
            making: Mutex::new(()),
        }
    }
}

/// A texture's storage as its parts hold it: made with the texture, or a shadow left unmade until
/// a reach makes it (the module docs).
pub(in crate::texture) enum StorageCell {
    Made(Storage),
    Lazy(Box<LazyShadow>),
}

/// A texture's storage as made so far ([`Texture::stored`]): made, or its shadow's recipe while it
/// is unmade.
pub(in crate::texture) enum Stored<'a> {
    Made(&'a Storage),
    Unmade(&'a ShadowRecipe),
}

impl StorageCell {
    /// Its storage, where made.
    pub(in crate::texture) fn made(&self) -> Option<&Storage> {
        match self {
            StorageCell::Made(storage) => Some(storage),
            StorageCell::Lazy(lazy) => lazy.made.get(),
        }
    }

    /// As made so far.
    pub(in crate::texture) fn stored(&self) -> Stored<'_> {
        match self {
            StorageCell::Made(storage) => Stored::Made(storage),
            StorageCell::Lazy(lazy) => match lazy.made.get() {
                Some(storage) => Stored::Made(storage),
                None => Stored::Unmade(&lazy.recipe),
            },
        }
    }

    /// The generation its objects are of, or a shadow's left unmade would be of: its bytes'.
    pub(in crate::texture) fn generation(&self) -> &Device {
        match self.stored() {
            Stored::Made(storage) => storage.generation(),
            Stored::Unmade(recipe) => recipe.texels.generation(),
        }
    }

    /// Whether its image's `VkImage` was handed out; never one not made.
    pub(in crate::texture) fn handed_out(&self) -> bool {
        self.made().is_some_and(Storage::handed_out)
    }

    /// Where an automatic or placement heap's texture's image lies in its heap; never a shadow's.
    pub(in crate::texture) fn heap_place(&self) -> Option<&super::heap::HeapPlace> {
        self.made().and_then(Storage::heap_place)
    }

    /// The texture objects over an image or texels of the texture's own; never a shadow's.
    pub(in crate::texture) fn objects(&self) -> Option<&crate::address::ObjectsToken> {
        self.made().and_then(Storage::objects)
    }
}

impl Texture {
    /// Whether the texture's shadow is left unmade (the module docs).
    pub(crate) fn shadow_unmade(&self) -> bool {
        matches!(self.stored(), Stored::Unmade(_))
    }

    /// Make the texture's shadow where it is left unmade (the module docs): at a reach that
    /// names the texture's image to the GPU before its later questions of the image ask it made
    /// (a slot that binds the image, a residency, a pass's attachment, an execution of commands
    /// that may bind it). Refused as its making is.
    pub(crate) fn made_for_use(&self) -> Result<(), Error> {
        self.storage().map(drop)
    }

    /// The texture's storage as made so far, its shadow's recipe while that is left unmade (the
    /// module docs): what a reach that must not make it reads (its rows' binding, its placement,
    /// its facts, its shape and options).
    pub(in crate::texture) fn stored(&self) -> Stored<'_> {
        match self.presented() {
            Some(storage) => Stored::Made(storage),
            None => self.s().storage.stored(),
        }
    }

    /// Its shadow left unmade, made (the module docs): the storage [`Texture::storage`] answers.
    /// Refused as its making is.
    pub(in crate::texture) fn made_lazily(&self) -> Result<&Storage, Error> {
        let parts = self.s();
        let lazy = match &parts.storage {
            super::StorageCell::Made(storage) => return Ok(storage),
            super::StorageCell::Lazy(lazy) => lazy,
        };
        let storage = {
            let _making = lock(&lazy.making);
            if let Some(made) = lazy.made.get() {
                return Ok(made);
            }
            let device = lazy.recipe.texels.generation();
            let storage = device.made_shadow(&lazy.recipe)?;
            lazy.made.get_or_init(|| storage)
        };
        let device = storage.generation();
        device.shared().counts.shadow_made_at_use();
        // A kept list of the resident textures' shadows is listed again.
        (device.shared().usage_shadows_made).fetch_add(1, Ordering::Release);
        // A detach of its bytes reaches its image, and its slot holds it from now on, as at a
        // texture's making.
        self.note_over_bytes();
        if self.0.id.id().0 != 0 {
            heap_sets::place(device, self);
        }
        Ok(storage)
    }
}
