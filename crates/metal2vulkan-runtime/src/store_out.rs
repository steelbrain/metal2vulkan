//! A render pass's store-out into memory whose GPU writes are dropped (Metal's private option bit
//! 24, [`ResourceOptions::PRIVATE_BIT`](crate::ResourceOptions::PRIVATE_BIT), or pages mapped
//! read-only): Metal's GPU faults the command buffer there, and the runtime fails it so where the
//! memory is a copy of the pages (`copied.rs`), which would take the store and drop it.
//!
//! - **What Metal does.** Measured on the M3 Max under macOS 27.0.1, each shape against a
//!   writable control that stores: a pass that stores out into such memory ends its command
//!   buffer with `MTLCommandBufferErrorPageFault`, its event signals firing and the pages
//!   unchanged. Each store-out does: a colour attachment over a texture over such a buffer at
//!   `storeAction = Store`; a multisample attachment resolving into such a texture
//!   (`MultisampleResolve`, the descriptor's action or one the encoder set); and a depth
//!   attachment in a heap of the bit at `CustomSampleDepthStore`, the descriptor's or the
//!   encoder's. An attachment there that the pass does not store out (`DontCare`, or a
//!   multisample one at `MultisampleResolve` into other memory) completes (`read_only_view.rs`).
//!   The command buffers committed after the fault run: on the same queue at once and 100 ms on,
//!   and on a new queue, those that read the faulted buffer and its texture too, which read the
//!   pages' bytes.
//! - **The runtime.** A pass whose store actions, as they settle at its end, store an
//!   attachment's or a resolve texture's texels into such memory faults as its command buffer
//!   commits ([`faults`], `EncoderCore::fault`): the pass and every encoder after it run nothing,
//!   every event signal fires, the command buffer completes with
//!   [`CommandBufferError::PageFault`](crate::CommandBufferError::PageFault), and the fault counts
//!   against its queue as every page fault does (`command_buffer/revocation.rs`). The attachment
//!   counted is the program's texture, never the image a pass renders it through (a stand-in, the
//!   separate depth and stencil textures' scratch image), nor a buffer's read twin
//!   (`read_twin.rs`), which no attachment names. On macOS such memory is imported through a
//!   read-only view of its pages (`read_only_view.rs`), where Metal's own GPU faults the store.

use super::buffer::Buffer;
use super::device::DeviceShared;
use super::render_pass::{ColorTarget, DepthStencilTarget, PassLayout, Target};
use super::render_values::StoreAction;
use super::texture::{Texture, TextureMemory};

/// Whether the pass of `layout`, its store actions settled, stores out into memory whose GPU
/// writes are dropped, where that memory is a copy of the pages (the module docs): Metal's GPU
/// faults its command buffer there. Never on a test's device that turned the fault off
/// ([`unfaulted`]).
pub(super) fn faults(device: &DeviceShared, layout: &PassLayout) -> bool {
    if super::read_only_view::VIEWS_DROP_WRITES || unfaulted(device) {
        return false;
    }
    let colors = layout.colors.iter().flatten().any(color_faults);
    let depth_stencil = (layout.depth_stencil.as_ref())
        .filter(|ds| !ds.hidden)
        .is_some_and(depth_stencil_faults);
    colors || depth_stencil
}

/// Whether a color attachment stores its texels, or resolves them, into such memory.
fn color_faults(color: &ColorTarget) -> bool {
    let own = color.stand_in.as_ref().map_or(&color.target, |s| &s.own);
    let stores = matches!(
        color.store,
        StoreAction::Store | StoreAction::StoreAndMultisampleResolve
    );
    let resolved = (color.resolve.as_ref())
        .map(|resolve| resolve.copy_to.as_ref().unwrap_or(&resolve.attachment))
        .or(color.copied_resolve.as_ref());
    (stores && drops_writes(&own.texture))
        || (color.store.resolves() && resolved.is_some_and(|into| drops_writes(&into.texture)))
}

/// Whether an aspect of the depth/stencil attachment stores its texels, or resolves them, into
/// such memory.
fn depth_stencil_faults(ds: &DepthStencilTarget) -> bool {
    let own: &Target = ds.stand_in.as_ref().map_or(&ds.target, |s| &s.own);
    let (depth, stencil) = match &ds.separate {
        Some(separate) => (separate.depth.as_ref(), Some(&separate.stencil)),
        None => (Some(own), Some(own)),
    };
    let resolves = ds.resolve.is_some();
    let depth_faults = ds.depth.is_some_and(|aspect| {
        let stores = matches!(
            aspect.store,
            StoreAction::Store
                | StoreAction::StoreAndMultisampleResolve
                | StoreAction::CustomSampleDepthStore
        );
        let into = ds.depth_resolve_target.as_ref();
        (stores && depth.is_some_and(|target| drops_writes(&target.texture)))
            || (resolves
                && aspect.store.resolves()
                && into.is_some_and(|into| drops_writes(&into.texture)))
    });
    let stencil_faults = ds.stencil.is_some_and(|aspect| {
        let stores = matches!(
            aspect.store,
            StoreAction::Store | StoreAction::StoreAndMultisampleResolve
        );
        let into = ds.stencil_resolve_target.as_ref();
        (stores && stencil.is_some_and(|target| drops_writes(&target.texture)))
            || (resolves
                && aspect.store.resolves()
                && into.is_some_and(|into| drops_writes(&into.texture)))
    });
    depth_faults || stencil_faults
}

/// Whether the GPU's writes to the memory `texture`'s texels lie in are dropped: the buffer it
/// is over, or its heap's own bytes, of bit 24 or over pages mapped read-only.
fn drops_writes(texture: &Texture) -> bool {
    match texture.memory() {
        TextureMemory::Heap(heap) => heap.bytes().is_some_and(Buffer::drops_writes),
        TextureMemory::Buffer(buffer) => buffer.drops_writes(),
        TextureMemory::Own(_) => false,
    }
}

/// Whether a test's device turned the fault off, to show what it keeps (the negative control of
/// `tests/store_out_device.rs`).
#[cfg(test)]
fn unfaulted(device: &DeviceShared) -> bool {
    device
        .store_out_unfaulted
        .load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether a test's device turned the fault off: never outside the tests.
#[cfg(not(test))]
fn unfaulted(_: &DeviceShared) -> bool {
    false
}
