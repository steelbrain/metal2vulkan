//! The shadow log: a default-off diagnostic, `METAL2VULKAN_LOG_SHADOWS` (`metal2vulkan::env_vars`),
//! that names each shadow a device makes of a texture, and why no image over its truth serves.
//!
//! - **What it prints.** One line per shadow as it is made, through the device's message sink at
//!   `Info` severity (id name `metal2vulkan-shadows`), else on stderr: the Metal call that made
//!   it (`new_texture_with_iosurface_plane` for an IOSurface plane's, the binding's or the pass's
//!   call for a shadow made at its first use), the texture's format, size, levels and usage, and
//!   its truth: a buffer's bytes (the buffer's storage mode, length and object ID,
//!   `Buffer::object_id`, and the texels' offset and row pitch) or the texture's own image; then
//!   what made it a shadow. Two textures over the same bytes name the same object ID and offset.
//! - **Which shadows.** A texture over a buffer's bytes made a shadow (with the reason its notice
//!   gives, `shadowed-texture`, announced once a device); a usage shadow, for a binding beyond
//!   the texture's usage or for a draw that samples a color attachment of its own render pass
//!   instance (`texture/usage_shadow.rs`), naming whether the texture's own image aliases the
//!   bytes and took the feedback loop usage; a rotated texture's image of its stored frame and
//!   its twin (`texture/rotation.rs`); a two-plane extended-range texture's
//!   (`texture/surface_two_planes.rs`); a depth or stencil texture's tiled twin over a surface
//!   plane (`texture/surface_depth.rs`); and a YCbCr texture's converted image and its render
//!   shadow (`texture/ycbcr_image.rs`; its write shadow is a usage shadow). A device copy is no shadow it names, nor is a heap texture's
//!   (its own notice names it), a view's shadow of an image or a texture buffer's staging.
//! - **What a shadowed texture's first draw or dispatch reads.** One line per shadowed texture
//!   (one whose reads go through a shadow of its truth, a usage shadow or two faces that take
//!   turns), at the first draw or dispatch whose pipeline uses a slot that binds it, after the
//!   command's bindings made the shadow it reads: the slot, whether the function fetches, samples
//!   or writes it, and for each sampler it samples it through, a sampler state's minification,
//!   magnification and mipmap filters (Metal's names) or a constexpr sampler's (Vulkan's names,
//!   as the layout bakes it). A texture's line names it once in the process, at whichever command
//!   bound it first ([`drew`], [`dispatched`]).
//! - **A read twin not in place.** One line per texture over a buffer with a read twin
//!   (`read_twin.rs`) whose twin texture is not in place over the twin's bytes, as its making
//!   leaves it: the texture, its buffer, and why, the twin's placement (its query's answer) where
//!   the making made a shadow, whose own line (`make_twin_texture made ...`) names the twin's bytes
//!   and the shadow's cause in full, or the making's error ([`twin_not_in_place`]).
//! - **Off, it costs nothing.** The switch is read once; no line is formatted, and a draw or a
//!   dispatch names no texture.

use super::argument_table::ArgumentTable;
use super::buffer_texture::Texels;
use super::device::lock;
use super::driver_facts::inform_through;
use super::texture_descriptor::{TextureDescriptor, TextureLayout};
use super::texture_placement::TexturePlacement;
use super::{Buffer, Device, Error, Texture};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{
    BindingSource, ComputePipeline, PipelineInterface, PipelineLayout, RenderPipeline, SamplerRef,
    SlotClass, SlotInterface, SlotRef, Stage, StageModule,
};
use metal2vulkan::vk;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// The id name of the log's messages.
const ID_NAME: &str = "metal2vulkan-shadows";

/// Whether the log is on: read once.
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(metal2vulkan::env_vars::log_shadows) || turned_on()
}

/// Whether a test turned the log on in its process ([`turn_on`]).
#[cfg(test)]
fn turned_on() -> bool {
    TURNED_ON.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether a test turned the log on: never outside tests.
#[cfg(not(test))]
fn turned_on() -> bool {
    false
}

/// A test's switch of the log ([`turn_on`]).
#[cfg(test)]
static TURNED_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn the log on for the rest of the test's process, as `METAL2VULKAN_LOG_SHADOWS` does (a
/// device test's).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn turn_on() {
    TURNED_ON.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// What a line names of a shadow's texture: its format, width and height, levels and usage.
pub(super) struct Shadowed {
    pub(super) pixel_format: PixelFormat,
    pub(super) extent: [u32; 2],
    pub(super) levels: u32,
    pub(super) usage: TextureUsage,
}

impl Shadowed {
    /// A texture of `descriptor`, checked as `layout`.
    pub(super) fn of(descriptor: &TextureDescriptor, layout: &TextureLayout) -> Shadowed {
        let [width, height, _] = layout.extent;
        Shadowed {
            pixel_format: descriptor.pixel_format,
            extent: [width, height],
            levels: layout.levels,
            usage: descriptor.usage,
        }
    }
}

/// Where the log is on, say that `call` made a shadow of a texture of `descriptor` (checked as
/// `layout`) over `texels`, for `why` (the module docs).
pub(super) fn made(
    device: &Device,
    call: &str,
    (descriptor, layout): (&TextureDescriptor, &TextureLayout),
    texels: &Texels,
    why: &str,
) {
    made_of(
        device,
        call,
        &Shadowed::of(descriptor, layout),
        Some(texels),
        why,
    );
}

/// Where the log is on, say that `call` made a shadow of `texture`, over `texels` where its truth
/// is a buffer's bytes, else over the texture's own image, for `why` (the module docs).
pub(super) fn made_of(
    device: &Device,
    call: &str,
    texture: &Shadowed,
    texels: Option<&Texels>,
    why: &str,
) {
    if !enabled() {
        return;
    }
    let [width, height] = texture.extent;
    let truth = match texels {
        Some(texels) => {
            let buffer = &texels.buffer;
            let storage = match buffer.resource_options().storage_mode() {
                Ok(mode) => format!("{mode:?}"),
                Err(_) => "unknown".to_owned(),
            };
            format!(
                "a {storage} buffer, object {}, of {} bytes at offset {} with rows of {} bytes",
                buffer.object_id().get(),
                buffer.length(),
                texels.layout.offset,
                texels.layout.row_pitch,
            )
        }
        None => "its own image".to_owned(),
    };
    let text = format!(
        "metal2vulkan: shadow: {call} made a {:?} texture of {width}x{height} and {} levels, \
         usage {:?}, over {truth}, a shadow: {why}",
        texture.pixel_format, texture.levels, texture.usage,
    );
    inform_through(device.shared().instance.sink(), ID_NAME, text);
}

/// Where the log is on, name each shadowed texture that `stages`' tables bind at a slot a function
/// of `compiled`, a draw's pipeline, uses, where no line named it before (the module's **What a
/// shadowed texture's first draw or dispatch reads**).
pub(crate) fn drew(device: &Device, compiled: &RenderPipeline, stages: &[(Stage, &ArgumentTable)]) {
    if !enabled() {
        return;
    }
    let pipeline = (&compiled.interface, &compiled.stages[..], &compiled.layout);
    first_binds(device, "draw", pipeline, stages);
}

/// Where the log is on, name each shadowed texture that `table` binds at a slot the kernel of
/// `compiled`, a dispatch's pipeline, uses, where no line named it before (the module's **What a
/// shadowed texture's first draw or dispatch reads**).
pub(crate) fn dispatched(device: &Device, compiled: &ComputePipeline, table: &ArgumentTable) {
    if !enabled() {
        return;
    }
    let pipeline = (
        &compiled.interface,
        std::slice::from_ref(&compiled.stage),
        &compiled.layout,
    );
    first_binds(device, "dispatch", pipeline, &[(Stage::Kernel, table)]);
}

/// The lines of [`drew`] and [`dispatched`]: each shadowed texture `stages`' tables bind at a slot
/// `pipeline`'s functions use, named once in the process, at the first `command` that binds it.
fn first_binds(
    device: &Device,
    command: &str,
    (interface, modules, layout): (&PipelineInterface, &[StageModule], &PipelineLayout),
    stages: &[(Stage, &ArgumentTable)],
) {
    /// The object IDs of the textures a line named.
    static NAMED: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    for &(stage, table) in stages {
        for (&index, texture) in &table.textures {
            let Some(kind) = shadowed(texture) else {
                continue;
            };
            let slot = SlotRef {
                stage,
                class: SlotClass::Texture,
                index,
            };
            let Some(SlotInterface::Texture(used)) = interface.slots.get(&slot) else {
                continue;
            };
            let id = texture.object_id().get();
            if !lock(NAMED.get_or_init(Mutex::default)).insert(id) {
                continue;
            }
            let access = used.access;
            let uses = [
                (access.read, "fetches"),
                (access.sample, "samples"),
                (access.write, "writes"),
            ];
            let uses: Vec<&str> = (uses.iter())
                .filter_map(|&(used, name)| used.then_some(name))
                .collect();
            let samplers = match access.sample {
                true => format!("; {}", samplers_named((modules, layout), table, slot)),
                false => String::new(),
            };
            let text = format!(
                "metal2vulkan: shadow: the first {command} that binds the {:?} texture of {}x{} \
                 and {} levels, object {id}, through {kind}, at {stage:?} texture {index}: its \
                 function {} it{samplers}",
                texture.pixel_format(),
                texture.width(),
                texture.height(),
                texture.mipmap_level_count(),
                uses.join(" and "),
            );
            inform_through(device.shared().instance.sink(), ID_NAME, text);
        }
    }
}

/// What a draw's reads of `texture` go through, where they go through a shadow: its faces
/// (`texture/faces.rs`), a usage shadow (`texture/usage_shadow.rs`) or a shadow of its truth.
fn shadowed(texture: &Texture) -> Option<&'static str> {
    if texture.has_faces() {
        return Some("two faces that take turns");
    }
    if texture.usage_shadow_copies().is_some() {
        return Some("a usage shadow");
    }
    texture.shadow().map(|_| "a shadow of its truth")
}

/// The samplers the function of a pipeline's `modules` samples the texture at `slot` through, each
/// with its filters: a sampler state bound in `table` by Metal's names, a constexpr sampler by the
/// Vulkan sampler `layout` bakes.
fn samplers_named(
    (modules, layout): (&[StageModule], &PipelineLayout),
    table: &ArgumentTable,
    slot: SlotRef,
) -> String {
    let stage = match slot.stage {
        Stage::Vertex => vk::ShaderStage::VERTEX,
        Stage::Fragment => vk::ShaderStage::FRAGMENT,
        Stage::Kernel => vk::ShaderStage::COMPUTE,
        Stage::Object => vk::ShaderStage::TASK_EXT,
        Stage::Mesh => vk::ShaderStage::MESH_EXT,
        _ => return "its samplers unknown".to_owned(),
    };
    let reads = (modules.iter())
        .find(|module| module.stage == stage)
        .and_then(|module| module.level_reads.iter().find(|r| r.slot == slot.index));
    let Some(reads) = reads else {
        return "its samplers unknown".to_owned();
    };
    let mut named: Vec<String> = Vec::new();
    for sampled in &reads.sampled {
        let name = match sampled.sampler {
            SamplerRef::Slot(index) => match table.samplers.get(&index) {
                Some((state, _)) => {
                    let d = state.descriptor();
                    format!(
                        "sampler {index}: min {:?}, mag {:?}, mip {:?}",
                        d.min_filter, d.mag_filter, d.mip_filter
                    )
                }
                None => format!("sampler {index}: none bound"),
            },
            SamplerRef::Constexpr(words) => {
                let baked = (layout.sets.iter())
                    .flat_map(|set| set.bindings.iter())
                    .find_map(|binding| match binding.source {
                        BindingSource::CombinedTextureSampler {
                            texture, sampler, ..
                        } if texture == slot && sampler.sampler == SamplerRef::Constexpr(words) => {
                            binding.immutable_sampler
                        }
                        _ => None,
                    });
                match baked {
                    Some(desc) => format!(
                        "a constexpr sampler: min {:?}, mag {:?}, mipmap mode {:?}",
                        desc.min_filter, desc.mag_filter, desc.mipmap_mode
                    ),
                    None => format!("a constexpr sampler {:#x}:{:#x}", words[0], words[1]),
                }
            }
            other => format!("{other:?}"),
        };
        if !named.contains(&name) {
            named.push(name);
        }
    }
    match named.is_empty() {
        true => "its samplers unknown".to_owned(),
        false => format!("through {}", named.join("; ")),
    }
}

/// Where the log is on, say that the read twin of `texture`, a texture of `descriptor` over
/// `buffer` at `offset` with rows of `bytes_per_row` bytes, is not in place over the twin's bytes
/// (the module's **A read twin not in place**): `made` is the making's answer, a shadow over the
/// twin, `twin` (`Ok`), or its error; a shadow's placement is asked of the twin again, as its query
/// answers it.
pub(super) fn twin_not_in_place(
    texture: &Texture,
    (buffer, twin): (&Buffer, &Buffer),
    (descriptor, offset, bytes_per_row): (&TextureDescriptor, u64, u64),
    made: Result<(), &Error>,
) {
    if !enabled() {
        return;
    }
    let why = match made {
        Ok(()) => match twin.texture_placement(descriptor, offset, bytes_per_row) {
            Ok(placement @ TexturePlacement::Shadow(_)) => format!(
                "its making made a shadow over the twin, object {}, {placement:?}: {}",
                twin.object_id().get(),
                placement.words()
            ),
            Ok(placement) => format!(
                "its making made a shadow over the twin, object {}; asked again, {placement:?}",
                twin.object_id().get()
            ),
            Err(error) => format!(
                "its making made a shadow over the twin, object {}; asked again, refused: {error}",
                twin.object_id().get()
            ),
        },
        Err(error) => format!("its making was refused: {error}"),
    };
    let text = format!(
        "metal2vulkan: shadow: the read twin of the {:?} texture of {}x{} and {} levels, object \
         {}, over the buffer object {} at offset {offset} with rows of {bytes_per_row} bytes, is \
         not in place: {why}",
        texture.pixel_format(),
        texture.width(),
        texture.height(),
        texture.mipmap_level_count(),
        texture.object_id().get(),
        buffer.object_id().get(),
    );
    inform_through(buffer.generation().shared().instance.sink(), ID_NAME, text);
}
