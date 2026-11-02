//! What a draw's shape reads of its pass's layout ([`LayoutView`]), and those values as one
//! ([`LayoutFacts`]), which a drawing state's kept shapes are found by (`render/draw_shape.rs`).
//!
//! The view holds the layout behind a field no other module reaches: the shape's derivations and
//! the encoder's call checks read the layout through its methods alone, and [`LayoutView::facts`]
//! reads every one of them. Two passes whose facts are equal give a draw the same shape. A method
//! added here that reads more of the layout adds that to the facts.

use crate::error::{Error, RuntimeGap};
use crate::render_pass::{PassLayout, MAX_COLOR_ATTACHMENTS};
use crate::tile_pass::TilePass;
use metal2vulkan::metal::PixelFormat;

/// A pass's layout as a draw's shape reads it (the module docs).
#[derive(Clone, Copy)]
pub(in crate::render) struct LayoutView<'a>(&'a PassLayout);

/// The values a [`LayoutView`] reads: no object, so two passes of the same values (the same pass
/// encoded again) give equal facts.
#[derive(Clone, Copy, PartialEq)]
pub(in crate::render) struct LayoutFacts {
    formats: [Option<PixelFormat>; MAX_COLOR_ATTACHMENTS],
    depth_stencil: Option<DepthStencilFacts>,
    samples: u32,
    extent: [u32; 2],
    tile: TilePass,
}

/// What a draw's shape reads of a pass's depth and stencil ([`LayoutView::depth_stencil`]).
#[derive(Clone, Copy, PartialEq)]
pub(in crate::render) struct DepthStencilFacts {
    /// Whether Metal's depth attachment has a texture.
    pub(in crate::render) depth: bool,
    /// Whether they are the hidden ones a draw gave the pass (`render_pass::DepthStencilTarget`).
    pub(in crate::render) hidden: bool,
    /// The Metal pixel format of the depth the pass renders.
    pub(in crate::render) depth_format: Option<PixelFormat>,
}

impl<'a> LayoutView<'a> {
    /// The view of `layout`.
    pub(in crate::render) fn of(layout: &'a PassLayout) -> LayoutView<'a> {
        LayoutView(layout)
    }

    /// Each color attachment's pixel format by index, `None` where the pass has none.
    pub(in crate::render) fn formats(self) -> [Option<PixelFormat>; MAX_COLOR_ATTACHMENTS] {
        self.0.pass_formats()
    }

    /// The pass's depth and stencil, where it has an attachment for them.
    pub(in crate::render) fn depth_stencil(self) -> Option<DepthStencilFacts> {
        self.0.depth_stencil.as_ref().map(|ds| DepthStencilFacts {
            depth: ds.depth.is_some(),
            hidden: ds.hidden,
            depth_format: ds.depth_format,
        })
    }

    /// The samples every attachment has.
    pub(in crate::render) fn samples(self) -> u32 {
        self.0.samples
    }

    /// The pass's tile memory and the encoder's tile size.
    pub(in crate::render) fn tile(self) -> TilePass {
        self.0.tile
    }

    /// The pass's tile size, where Metal's is measured (`PassLayout::tile_size`).
    pub(in crate::render) fn tile_size(self) -> Result<[u32; 2], Error> {
        self.tile()
            .tile
            .ok_or(Error::NotImplemented(RuntimeGap::TileAtSampleCount))
    }

    /// The pixels a pass of tiles of `tile` covers: whole tiles over the render area
    /// (`PassLayout::tile_area`).
    pub(in crate::render) fn tile_area(self, tile: [u32; 2]) -> [u32; 2] {
        let [ew, eh] = self.0.extent;
        [
            ew.div_ceil(tile[0]).saturating_mul(tile[0]),
            eh.div_ceil(tile[1]).saturating_mul(tile[1]),
        ]
    }

    /// Every value the methods above read.
    pub(in crate::render) fn facts(self) -> LayoutFacts {
        LayoutFacts {
            formats: self.formats(),
            depth_stencil: self.depth_stencil(),
            samples: self.samples(),
            extent: self.0.extent,
            tile: self.tile(),
        }
    }
}
