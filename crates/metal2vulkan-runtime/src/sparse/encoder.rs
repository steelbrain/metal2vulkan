//! `MTLResourceStateCommandEncoder`: an encoder of sparse texture mapping updates.
//!
//! It records updates, not commands: each is checked against the textures and against the
//! device's sparse units when it is encoded, and the scheduler applies them in submission order
//! as one `vkQueueBindSparse` before the encoders after it (`bind.rs`). The encoder starts a
//! batch of its own, unless the batch so far holds only such encoders. An indirect update's
//! regions are GPU memory the work before it may write: the batch waits for a host step, which
//! reads them once that work completed (`indirect.rs`).

use super::mapping::{MetalUnit, UnitUpdate};
use super::texture::SparseImage;
use super::{tile_texels, SparseProblem};
use crate::buffer::Buffer;
use crate::command_buffer::CommandBuffer;
use crate::encoder::{OpenGuard, Recorded};
use crate::error::{invalid, Error, Problem, RuntimeGap};
use crate::fence::Fence;
use crate::resource::ResourceKey;
use crate::texture::Texture;
use crate::tracker::Summary;
use crate::used_resources::EncoderKind;
use crate::values::{Origin, Region, Size};
use metal2vulkan::bind::TileMapping;
use std::marker::PhantomData;

/// One update a resource state encoder recorded, which the scheduler applies in order. The
/// command buffer holds the textures and buffers until it completes.
#[derive(Clone)]
pub(crate) enum SparseUpdate {
    /// Map or unmap Metal units of a texture.
    Units {
        texture: Texture,
        update: UnitUpdate,
    },
    /// Move the mapping of each pair's source unit of `source` to its destination unit of
    /// `destination`, where that one is unmapped.
    Move {
        source: Texture,
        destination: Texture,
        pairs: Vec<(MetalUnit, MetalUnit)>,
    },
    /// An indirect update, whose regions the host step reads from `buffer` at `offset`
    /// (`indirect.rs`); the step replaces it with the updates they state.
    Indirect {
        texture: Texture,
        mode: TileMapping,
        buffer: Buffer,
        offset: u64,
    },
}

impl SparseUpdate {
    /// Whether the host step reads it before its batch is submitted.
    pub(crate) fn needs_step(&self) -> bool {
        matches!(self, SparseUpdate::Indirect { .. })
    }
}

/// `MTLResourceStateCommandEncoder`. `Send`, not `Sync`.
pub struct ResourceStateCommandEncoder {
    /// Dropped before `end_encoding` ran, it marks the command buffer abandoned.
    open: OpenGuard,
    updates: Vec<SparseUpdate>,
    /// The fences it waits for, and those it updates, in call order.
    waits: Vec<ResourceKey>,
    fence_updates: Vec<ResourceKey>,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl std::fmt::Debug for ResourceStateCommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceStateCommandEncoder")
            .field("updates", &self.updates.len())
            .finish()
    }
}

impl CommandBuffer {
    /// `resourceStateCommandEncoder`: an encoder of sparse texture mapping updates.
    ///
    /// Refused as every encoder is: the command buffer committed, or an encoder open or abandoned
    /// (`Invalid`).
    pub fn resource_state_command_encoder(&self) -> Result<ResourceStateCommandEncoder, Error> {
        // The last render encoder's end, where its call left it to the device's lowering thread.
        self.settle_end();
        self.refuse_protected_state_encoder()?;
        let mut state = self.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid("resource_state_command_encoder", problem));
        }
        // Made under the protection options set now (`command_buffer/protection.rs`).
        state.protected_made();
        state.encoder_open = true;
        drop(state);
        (self.device().shared().identifiers).encoder_opened(EncoderKind::ResourceState);
        Ok(ResourceStateCommandEncoder {
            open: OpenGuard {
                command_buffer: self.clone(),
                ended: false,
            },
            updates: Vec::new(),
            waits: Vec::new(),
            fence_updates: Vec::new(),
            _not_sync: PhantomData,
        })
    }
}

impl ResourceStateCommandEncoder {
    /// `updateTextureMapping:mode:region:mipLevel:slice:`: map or unmap the tiles `region` (in
    /// tiles) of `level` and `slice` of sparse `texture`, after the encoders before this one and
    /// before those after it, as Metal does: at `firstMipmapInTail` the slice's whole tail whatever
    /// the region; nothing past it, nor for a region of no tiles; a tile already as asked stays as
    /// it is; a heap that fills maps the tiles in order until it is full.
    ///
    /// The device's sparse units need not lie in the update's tiles: a 16-KiB tile maps on a
    /// device whose blocks are 64 KiB, its block bound while any of its tiles is mapped and the
    /// block's unmapped tiles kept at zero.
    ///
    /// Through a view of a sparse texture (`new_texture_view`), the update maps the texture's own
    /// tiles as Metal's does (a Metal sparse probe on an M3 Max and an M4): `level` names the
    /// texture's level whatever the view's levels, past them too (a view of levels 1 and 2 mapping
    /// its level 2 maps the texture's level 2), and `slice` the view's slice, its texture's slice
    /// past the view's first, past the view's slices too (a view of slices 1 and 2 mapping its
    /// slice 2 maps the texture's slice 3); a view of another pixel format of the same tile maps
    /// the same tiles. Plain Metal; its validation stops a level or slice past the view's.
    ///
    /// A texture that is not sparse records nothing, as plain Metal maps nothing in it (an M4 under
    /// macOS 27.0.1: its texels kept, the command buffer completed; Metal's
    /// validation stops it).
    ///
    /// A level or slice past the texture's own, and a region past a tiled level's tiles, are
    /// **taken**, mapping the tiles inside the texture and nothing else, as plain Metal maps
    /// arguments its own validation layer stops (`UnitGrid::clipped_region`, where the
    /// measurement is): level 9 of a nine-level texture and an origin past the level's tiles map
    /// nothing, and a region running past them maps the level's own tiles. A slice past the
    /// texture's maps none, where plain Metal's `usedSize` took one page for it.
    ///
    /// Refused, recording nothing: a texture of another device (`Invalid`); a view of a pixel
    /// format whose tile is not its texture's (`NotImplemented(SparseHeaps)`: not measured).
    pub fn update_texture_mapping(
        &mut self,
        texture: &Texture,
        mode: TileMapping,
        region: Region,
        level: u64,
        slice: u64,
    ) -> Result<(), Error> {
        self.update_texture_mappings(texture, mode, &[region], &[level], &[slice])
    }

    /// `updateTextureMappings:mode:regions:mipLevels:slices:numRegions:`: each region in order, as
    /// [`update_texture_mapping`](Self::update_texture_mapping) maps it.
    ///
    /// Refused, recording nothing, as one region is, for any of them, and for lists of unequal
    /// lengths (`Invalid`, `SparseProblem::Lengths`).
    pub fn update_texture_mappings(
        &mut self,
        texture: &Texture,
        mode: TileMapping,
        regions: &[Region],
        levels: &[u64],
        slices: &[u64],
    ) -> Result<(), Error> {
        const CALL: &str = "update_texture_mappings";
        if levels.len() != regions.len() || slices.len() != regions.len() {
            return Err(invalid(CALL, Problem::Sparse(SparseProblem::Lengths)));
        }
        let Some(sparse) = self.sparse(CALL, texture)? else {
            return Ok(());
        };
        let first = first_slice(texture, sparse)?;
        let mut units = Vec::new();
        for ((region, &level), &slice) in regions.iter().zip(levels).zip(slices) {
            let slice = slice.saturating_add(first);
            units.extend(sparse.grid.clipped_region(level, slice, region));
        }
        if !units.is_empty() {
            self.updates.push(SparseUpdate::Units {
                texture: texture.clone(),
                update: UnitUpdate { mode, units },
            });
        }
        Ok(())
    }

    /// `updateTextureMapping:mode:indirectBuffer:indirectBufferOffset:`: map or unmap the regions
    /// `indirect_buffer` holds at `indirect_buffer_offset` when the update runs, after what the
    /// work before it wrote there, as Metal reads them: a 32-bit count, then that
    /// many `MTLMapIndirectArguments` (each 32-bit: the region's origin and size in tiles, the
    /// level and the slice), each mapped as [`update_texture_mapping`](Self::update_texture_mapping)
    /// maps it. An offset that is not a multiple of 4 reads at the multiple of 4 below it, as
    /// plain Metal reads it (an M4 under macOS 27.0.1; its validation takes it).
    ///
    /// The host reads the regions once the work before the update completed (the
    /// `host-read-mapping-arguments` performance degrade, announced once). Arguments Metal's
    /// validation would stop map as plain Metal maps them (the M3 Max under macOS 27.0.1): a
    /// region past its level's tiles maps the tiles inside them; one whose origin is past them,
    /// of no size on an axis, or of a level or slice past the texture's, maps nothing; and the
    /// regions the count names past the buffer map nothing. Differs: Metal's `usedSize` over a
    /// region of a slice past the texture's varies (a tile more beside other regions, none
    /// alone, and an unmap of it can take the count below zero); the runtime counts none.
    ///
    /// A texture that is not sparse records nothing, the buffer unread, as plain Metal maps
    /// nothing in it (as [`update_texture_mapping`](Self::update_texture_mapping)'s). Through a
    /// view the regions name the texture's own tiles as a direct update's do (the levels the
    /// texture's, the slices the view's).
    ///
    /// Refused, recording nothing: a texture or buffer of another device, or a count past the
    /// buffer of a sparse texture (`Invalid`); a view of a pixel format whose tile is not its
    /// texture's (`NotImplemented(SparseHeaps)`: not measured).
    pub fn update_texture_mapping_indirect(
        &mut self,
        texture: &Texture,
        mode: TileMapping,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        const CALL: &str = "update_texture_mapping_indirect";
        let offset = indirect_buffer_offset - indirect_buffer_offset % 4;
        if !indirect_buffer
            .device()
            .same(self.open.command_buffer.device())
        {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        let Some(sparse) = self.sparse(CALL, texture)? else {
            return Ok(());
        };
        first_slice(texture, sparse)?;
        let length = indirect_buffer.length();
        if offset.checked_add(4).is_none_or(|end| end > length) {
            let problem = SparseProblem::IndirectArguments {
                offset,
                bytes: 4,
                length,
            };
            return Err(invalid(CALL, Problem::Sparse(problem)));
        }
        self.updates.push(SparseUpdate::Indirect {
            texture: texture.clone(),
            mode,
            buffer: indirect_buffer.clone(),
            offset,
        });
        Ok(())
    }

    /// `moveTextureMappingsFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:
    /// destinationSlice:destinationLevel:destinationOrigin:`: move each mapped tile of the
    /// `source_size` tiles at `source_origin` of `source_level` and `source_slice` of `source` to
    /// the tile at the same place from `destination_origin` of `destination_level` and
    /// `destination_slice` of `destination`, where that tile is unmapped, as Metal does
    /// (and a Metal sparse probe on the M3 Max and an M4): a tile whose destination is mapped stays
    /// where it is, and the heaps' `usedSize` does not change. The tiles move one at a time in
    /// Metal's order (x, then y, then z), each where its destination is unmapped as it moves, so a
    /// move within a texture whose regions overlap moves only what lands on unmapped tiles (tiles 0
    /// to 2 onto 1 to 3: only tile 2, onto 3). At `firstMipmapInTail` the slice's whole tail moves,
    /// onto the other side's tail or onto one tile of a tiled level, and one tile onto a whole
    /// tail, as Metal's validation takes and Metal moves them; past it nothing moves, as plain
    /// Metal (validation stops it). A level or slice past a texture's own, and a source size past
    /// a tiled level's tiles, are **taken**, moving the tiles inside both textures and nothing
    /// else: plain Metal takes all three and leaves both heaps' `usedSize` where it was, measured
    /// beside the update's (`UnitGrid::clipped_region`); which tiles it moved there is not
    /// measured, so moving the clipped pairs is a near-equivalent. Through a view, each side names its texture's own tiles as
    /// [`update_texture_mapping`](Self::update_texture_mapping)'s does (the levels the texture's,
    /// the slices the view's; validation stops a region past the view's own level). Plain Metal
    /// moves between textures of different heaps and of different formats, tiles of different sizes
    /// included (its validation stops different formats); so does the runtime, tile for tile. The
    /// destination heap does not count the tiles it gains, as Metal's does not. Each moved tile's
    /// texels go with it, as Metal's (a Metal sparse probe on the M3 Max and an M4): the runtime
    /// copies them, so the device's sparse units need not move whole; between formats of different
    /// texels, and between a tail and a tile, the destination reads zero, as its bytes would be
    /// another layout's (a near-equivalent: Metal's read those bytes).
    ///
    /// Between two textures that are not sparse it records nothing, as plain Metal moves nothing
    /// (an M4: both textures' texels kept).
    ///
    /// Refused, recording nothing: a texture of another device, a move between a sparse texture
    /// and one that is not (`NotSparse`, not measured) (`Invalid`); a view of a pixel format whose
    /// tile is not its texture's (`NotImplemented(SparseHeaps)`: not measured).
    pub fn move_texture_mappings(
        &mut self,
        source: &Texture,
        source_slice: u64,
        source_level: u64,
        source_origin: Origin,
        source_size: Size,
        destination: &Texture,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: Origin,
    ) -> Result<(), Error> {
        const CALL: &str = "move_texture_mappings";
        let (from, to) = match (self.sparse(CALL, source)?, self.sparse(CALL, destination)?) {
            (Some(from), Some(to)) => (from, to),
            (None, None) => return Ok(()),
            _ => return Err(invalid(CALL, Problem::Sparse(SparseProblem::NotSparse))),
        };
        let source_slice = source_slice.saturating_add(first_slice(source, from)?);
        let destination_slice = destination_slice.saturating_add(first_slice(destination, to)?);
        let from_region = Region {
            origin: source_origin,
            size: source_size,
        };
        let to_region = Region {
            origin: destination_origin,
            size: source_size,
        };
        let clipped = |image: &SparseImage, level, slice, region: &Region| {
            image.grid.clipped_region(level, slice, region)
        };
        let sources = clipped(from, source_level, source_slice, &from_region);
        let destinations = clipped(to, destination_level, destination_slice, &to_region);
        let pairs: Vec<(MetalUnit, MetalUnit)> = sources.into_iter().zip(destinations).collect();
        if !pairs.is_empty() {
            self.updates.push(SparseUpdate::Move {
                source: source.clone(),
                destination: destination.clone(),
                pairs,
            });
        }
        Ok(())
    }

    /// `updateFence:`: the fence is updated after the encoder's updates are bound.
    ///
    /// Refused (`Invalid`): a fence of another device.
    pub fn update_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        if !fence.device().same(self.open.command_buffer.device()) {
            return Err(invalid("update_fence", Problem::ForeignObject));
        }
        let key = fence.key();
        if !self.fence_updates.contains(&key) {
            self.fence_updates.push(key);
        }
        Ok(())
    }

    /// `waitForFence:`: the encoder's updates are bound after the fence's last update before
    /// them in the queue's commit order (a bind waits for everything submitted before it, which
    /// holds that update). A wait for a fence this encoder updated waits for the update before
    /// the encoder, as the other encoders' waits do, and the updates are bound, as plain Metal
    /// binds them (an M4 under macOS 27.0.1; its validation stops the wait).
    ///
    /// Refused (`Invalid`): a fence of another device.
    pub fn wait_for_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        const CALL: &str = "wait_for_fence";
        if !fence.device().same(self.open.command_buffer.device()) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        let key = fence.key();
        if !self.waits.contains(&key) {
            self.waits.push(key);
        }
        Ok(())
    }

    /// `endEncoding`: the updates go to the command buffer, in a batch that binds them before
    /// the encoders after this one.
    pub fn end_encoding(self) -> Result<(), Error> {
        let ResourceStateCommandEncoder {
            mut open,
            updates,
            waits,
            fence_updates,
            ..
        } = self;
        // The textures it maps, and the buffers an indirect update reads, are its used
        // resources (`used_resources.rs`).
        let keys = updates.iter().flat_map(|update| match update {
            SparseUpdate::Units { texture, .. } => [Some(texture.use_key()), None],
            SparseUpdate::Move {
                source,
                destination,
                ..
            } => [Some(source.use_key()), Some(destination.use_key())],
            SparseUpdate::Indirect {
                texture, buffer, ..
            } => [Some(texture.use_key()), Some(buffer.identity().key)],
        });
        let keys = keys.flatten().collect();
        let mut state = open.command_buffer.state();
        // Its updates take no pages of a buffer over host pages (`rebind.rs`): an indirect one
        // reads its buffer in its host step.
        let encoder = state.encoder_ended(
            EncoderKind::ResourceState,
            keys,
            None,
            crate::rebind::Taken::default(),
        );
        state.end_sparse_encoder(
            &open.command_buffer,
            Recorded {
                encoder,
                raws: Vec::new(),
                segments: Vec::new(),
                summary: Summary::of_fences(waits, fence_updates),
                sparse: updates,
                twin_writes: Vec::new(),
            },
        );
        drop(state);
        open.ended = true;
        Ok(())
    }

    /// The sparse image of `texture`, a sparse texture or a view of one; `None` for a texture
    /// that is not sparse, whose update plain Metal records as nothing (module docs). Refused
    /// (`Invalid`): a texture of another device.
    fn sparse<'a>(
        &self,
        call: &'static str,
        texture: &'a Texture,
    ) -> Result<Option<&'a SparseImage>, Error> {
        if !texture.device().same(self.open.command_buffer.device()) {
            return Err(invalid(call, Problem::ForeignObject));
        }
        Ok(texture.sparse_image())
    }
}

/// The slice of its sparse texture `sparse` that slice 0 of `texture` names: a view's first slice,
/// 0 for the texture itself. Through a view an update or a move names its texture's own tiles, as
/// Metal's does (a Metal sparse probe on an M3 Max and an M4): the level is the texture's whatever
/// the view's levels, past them too (a view of levels 1 and 2 mapping its level 2 maps the
/// texture's level 2, and a view of level 0 mapping level 1 the texture's level 1); the slice is
/// the view's, its texture's past the view's first, past the view's slices too (a view of slices 1
/// and 2 mapping its slice 2 maps the texture's slice 3); and a view of another pixel format of the
/// same tile (`R32Uint`, `RGBA8Unorm` and `RG16Float` views of an `R32Float` texture) names the
/// same tiles. Plain Metal; its validation stops a level or slice past the view's and a region past
/// the view's own level, and takes the rest.
///
/// Refused (`NotImplemented(SparseHeaps)`): a view of a pixel format whose tile is not the
/// texture's (not measured).
fn first_slice(texture: &Texture, sparse: &SparseImage) -> Result<u64, Error> {
    let record = texture.record();
    let shape = &sparse.grid.shape;
    if record.pixel_format != shape.pixel_format
        && tile_texels(record.pixel_format, shape.samples, sparse.page.bytes()) != sparse.grid.tile
    {
        return Err(Error::NotImplemented(RuntimeGap::SparseHeaps));
    }
    Ok(u64::from(record.base_layer))
}
