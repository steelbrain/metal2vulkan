//! Color attachments a render pass does not store: a texture over a buffer keeps the buffer's
//! bytes, and a texture's own image (`DontCare`) its earlier texels.
//!
//! **Metal's answer** (the DontCare-store probe, an M3 Max under macOS 27.0.1, each case 20 times
//! alike): after a pass whose color attachment over a buffer is stored `DontCare`, the buffer
//! holds the bytes it held before the pass in every texel; Metal writes neither the rendered nor
//! the clear value, and no byte between the rows. So it was in 660 passes over a texture at its
//! first use, 72 after a pass that stored it (which keep that pass's bytes), 72 beside a second
//! attachment stored with `Store` (which holds the rendered value), 98 that clear and draw
//! nothing, and 24 over IOSurfaces: 16x16, 333x77, 1307x512 at rows of 5248 bytes, and 2048x2048,
//! tight and padded; `BGRA8Unorm`, `RGBA8Unorm`, `BGRA8Unorm_sRGB`, `RGBA16Float` and `R8Unorm`;
//! shared, no-copy and IOSurface buffers; every load action; a draw over the whole texture, half
//! of it, a corner, or none. And so it was with a memory barrier between two draws (the barrier
//! probe, the same GPU, 20 times alike): over buffers, render targets, textures or the buffer
//! alone, from the fragment stage to the vertex or the fragment stage, where a Vulkan pass splits
//! or MoltenVK restarts its Metal pass. A `Store` there leaves the rendered texels, and outside the
//! buffer's (`Load`) or the clear value (`Clear`). The one exception: the first pass in a process
//! to overflow Apple's tiler (some 2 to 4 million triangle-tile entries) leaves the rendered
//! texels, in every texel, all or nothing; the driver then grows its tiler's buffer for the rest of
//! the process. No call predicts or detects it, and the runtime does not model it: its answer is
//! the buffer's bytes, exactly. `MultisampleResolve`, which stores nothing of the attachment,
//! leaves it the same way: a texture over a buffer is never multisampled, so such an attachment is
//! a single-sampled one whose texels are copied into its resolve texture (`resolve.rs`).
//!
//! - **A shadowed texture** (`shadow.rs`) renders into its shadow, which the pass copies back only
//!   where it stores it (`render/shadows.rs`): the buffer keeps its bytes. So does a planar
//!   surface's depth or stencil texture, which renders through its tiled twin, a shadow
//!   (`texture/surface_depth.rs`), and any texture over a buffer the pass renders through a
//!   stand-in (`attachment_stand_in.rs`), which copies back only what Metal stores.
//! - **A texture whose image lies over its buffer** (`Texture::image_over_buffer`: a linear image
//!   over the buffer's memory, an ordinary buffer's, a no-copy buffer's host pages, a dma-buf's,
//!   a surface's, a texture buffer's attachment twin among them; or, on MoltenVK, a Metal texture
//!   over a no-copy or surface buffer's `MTLBuffer`) would have the pass render into the buffer's
//!   bytes. Where the pass does not store it, it renders through the image its texture keeps for
//!   the queue instead (`attachment_stand_in::kept_backing`, [`Cause::KeptBytes`]): a `Load`
//!   copies the buffer's texels over the render area into that image before the pass; a `Clear`
//!   or a `DontCare` load is the image's own first operation, by the rules a `DontCare` load
//!   follows elsewhere; nothing is copied back, and the image's own store is `DONT_CARE`
//!   (`render_target.rs`): nothing reads what a pass leaves there. A pass that stores the
//!   attachment renders it in place, as before, at no cost. Announced once (`dont-care-stand-in`,
//!   performance), and counted per pass (`WorkCounts::kept_bytes_passes`).
//! - **Handed to Metal's own `DontCare`**: on a driver that stores `DONT_CARE` as Metal's own on
//!   the same GPU (`DriverFact::DontCareStoreIsMetals`, MoltenVK on the M4 and the M3 Max), a pass
//!   recorded as one render pass instance with nothing that restarts the driver's Metal pass
//!   hands the attachment to Metal, which keeps the buffer's bytes itself (`HandedToMetal`): it
//!   renders in place, at no cost, and is neither announced nor counted. A pass split into
//!   instances, or restarted, stores the attachment between them, so it renders through the kept
//!   image as on any other driver. One that runs one more instance after its last to resolve
//!   (`render.rs`), which this cannot see, stays in place, and `dont-care-store` announces it.
//! - **Decided as the pass ends** ([`settle`]), once its store actions are known (a late one from
//!   `set_color_store_action`, and `Unknown`, which stores as `DontCare`): the draws record against
//!   the compatible pass, which a stand-in of the attachment's format and samples keeps, and the
//!   framebuffer is made then. But where a draw of the pass took a view of the attachment's image
//!   (a framebuffer fetch reading it in place, as an input attachment or a storage image, its
//!   sample planes, a draw rendering it in another format through a view, its extended-range
//!   shadow), where the pass executes render commands (whose draws record against the encoder's
//!   layout as the execution was encoded, `render/executions.rs`), where the device renders the
//!   texture's image only beside other linear ones (`VK_NV_linear_color_attachment`'s rule,
//!   `linear_attachment.rs`) and a draw took a view of another such attachment in place, which
//!   would then have to render apart, and where its image or its format copies no texels and the
//!   pass loads it: there the pass renders in place, and `dont-care-store` announces the texels it
//!   rendered there, Metal's answer where its tiler overflows.
//!
//! **A texture's own image** keeps its earlier texels too. Metal's documentation leaves them open
//! (`MTLStoreAction.dontCare`: "You need to assume that GPU discarded the texture's contents"), and
//! what Apple's GPU leaves is measured: the attachment's texels from before the pass, the pass's
//! clear and its draws both gone (the conformance case `pass-sequence-color-actions`, an M3 Max
//! under macOS 27.0.1, eight runs alike, whose two passes over one texture the host wrote first
//! read the host's texels back after a `DontCare` store under either load action of the first and
//! either load action and render area of the second; and `dont_care_stores`' passes on MoltenVK,
//! which hands the store to Metal). Vulkan's `DONT_CARE` leaves the texels undefined, and a driver
//! may leave neither answer: NVIDIA's left a pass's clear under its draw. So a pass that stores a
//! color attachment `DontCare` renders it, where the texture's image is its own (not a buffer's
//! bytes, nor a shadow of them), through a backing its queue keeps for its passes of the
//! attachment's shape (`attachment_stand_in::pooled_backing`, [`Cause::KeptTexels`]): a `Load`
//! copies the texture's texels over the render area into it before the pass (what the pass's
//! blending and tile dispatches read there), a `Clear` or a `DontCare` load is its own first
//! operation, as above, and nothing is copied back. Its store
//! is `DONT_CARE` (`render_target.rs`). Not where the pass hands the attachment to Metal's own
//! `DontCare` (`HandedToMetal`, at no cost), nor in the cases above that render in place, nor for a
//! memoryless texture (none of its texels outlive a pass in Metal), nor one the pass resolves
//! (`MultisampleResolve`, which `resolve::keep_samples` keeps where its resolve reads the samples),
//! nor where a format of the device's makes no stand-in image or, loaded, copies none: those
//! render in place, and `dont-care-store` announces them. The texture keeps its texels, Metal's
//! answer, so nothing is announced; each such pass is counted (`WorkCounts::kept_texels_passes`).
//! The cost is the backing's memory, one image of the attachment's shape per queue (bounded with
//! the queue's memoryless backings, `attachment_stand_in::BackingPool`), and a `Load`'s copy.

use super::attachment_stand_in::{self, Cause, StandIn};
use super::command_buffer::CommandBuffer;
use super::degradation::Degradation;
use super::error::{Error, RuntimeGap};
use super::linear_attachment;
use super::render_pass::ColorTarget;
use super::render_values::{LoadAction, StoreAction};

/// Whether a pass stores nothing of `color`: `DontCare`, or `MultisampleResolve` (as `Unknown`
/// stores once the pass ends).
fn unstored(color: &ColorTarget) -> bool {
    matches!(
        color.store,
        StoreAction::DontCare | StoreAction::MultisampleResolve
    )
}

/// Whether `color`, a texture's own image (neither over a buffer's bytes nor a shadow of them), is
/// one a pass renders through a backing so that it keeps its texels (the module docs): stored
/// `DontCare`, resolving nothing, and not memoryless.
fn keeps_texels(color: &ColorTarget) -> bool {
    color.store == StoreAction::DontCare
        && !color.memoryless
        && color.resolve.is_none()
        && color.copied_resolve.is_none()
        && !color.target.texture.over_buffer()
}

/// Render each color attachment among `colors`, a pass's as it ends, that the pass does not
/// store and whose texture's image lies over its buffer's bytes, through the image its texture
/// keeps for `maker`'s queue, and each the pass stores `DontCare` whose texture's image is its
/// own, through a backing the queue keeps (the module docs): but where the pass hands it to the
/// driver's `DONT_CARE` as Metal's own `DontCare` (`handed`, `HandedToMetal`), a draw took a view
/// of its image (`viewed`), the pass executes render commands (`executes`), or another color in
/// place is bound by `VK_NV_linear_color_attachment`'s rule and the pass cannot render it apart
/// (`taken`, as `linear_attachment::settle` asks it). The first announced; both counted.
pub(super) fn settle(
    maker: &CommandBuffer,
    colors: &mut [Option<ColorTarget>],
    executes: bool,
    handed: impl Fn(usize) -> bool,
    viewed: impl Fn(usize) -> bool,
    taken: impl Fn(usize) -> bool,
) -> Result<(), Error> {
    if executes {
        return Ok(());
    }
    let device = maker.device();
    // The colors a draw took a view of that render in place under the extension's rule: one
    // rendered through a kept image beside them would leave them to render apart, which
    // `linear_attachment::settle` refuses.
    let mut held = Vec::new();
    for (index, color) in colors.iter().enumerate() {
        let Some(color) = color else {
            continue;
        };
        if color.stand_in.is_none()
            && taken(index)
            && linear_attachment::bound(device, &color.target)?
        {
            held.push(index);
        }
    }
    let queue = maker.command_queue().id();
    let (mut kept, mut kept_texels) = (false, false);
    for index in 0..colors.len() {
        // The images the pass renders its other colors into, which no backing of this one is.
        let taken_images: Vec<_> = (colors.iter().flatten())
            .filter(|color| color.stand_in.is_some())
            .map(|color| color.target.texture.clone())
            .collect();
        let Some(color) = colors[index].as_mut() else {
            continue;
        };
        let over_buffer = color.target.texture.image_over_buffer();
        if !unstored(color)
            || color.stand_in.is_some()
            || !(over_buffer || keeps_texels(color))
            || handed(index)
            || viewed(index)
            || held.iter().any(|&other| other != index)
        {
            continue;
        }
        let loads = color.load == LoadAction::Load;
        if loads && attachment_stand_in::check_copies(&color.target).is_err() {
            continue;
        }
        if !over_buffer {
            let own = color.target.clone();
            // A format the device makes no stand-in of keeps the pass in place.
            let backing =
                match attachment_stand_in::pooled_backing(maker, &own, &taken_images, loads) {
                    Ok(backing) => backing,
                    Err(Error::NotImplemented(RuntimeGap::Render)) => continue,
                    Err(error) => return Err(error),
                };
            color.target = backing;
            color.stand_in = Some(StandIn {
                own,
                cause: Cause::KeptTexels,
                area: None,
            });
            kept_texels = true;
            continue;
        }
        let own = color.target.clone();
        // A format the device copies no image of keeps the pass in place.
        let backing = match attachment_stand_in::kept_backing(maker, &own, queue, true) {
            Ok(backing) => backing,
            Err(Error::NotImplemented(RuntimeGap::Render)) => continue,
            Err(error) => return Err(error),
        };
        let (format, store) = (own.texture.pixel_format(), color.store);
        device.degrade(Degradation::DontCareStandIn, || {
            (
                format!(
                    "a {format:?} texture over a buffer's bytes, color attachment {index} of a \
                     render pass"
                ),
                format!(
                    "the pass stores it with `{store:?}`, after which Metal's texture over a \
                     buffer holds the buffer's bytes, and the texture's image lies over them, so \
                     the pass renders into an image the texture keeps"
                ),
            )
        });
        color.target = backing;
        color.stand_in = Some(StandIn {
            own,
            cause: Cause::KeptBytes,
            area: None,
        });
        held.retain(|&other| other != index);
        kept = true;
    }
    if kept {
        device.shared().counts.kept_bytes_pass();
    }
    if kept_texels {
        device.shared().counts.kept_texels_pass();
    }
    Ok(())
}
