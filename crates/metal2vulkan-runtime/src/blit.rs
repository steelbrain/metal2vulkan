//! `BlitCommandEncoder` (`MTLBlitCommandEncoder`) and its buffer commands; the texture commands are
//! step 5's (`blit_texture.rs`).
//!
//! - **Order.** A blit encoder runs its commands in encoded order over every resource: each command
//!   goes through [`BlitCommandEncoder::command`] with what it reads and writes, which records the
//!   barrier those uses need after the encoder's earlier commands.
//! - **Copies.** `vkCmdCopyBuffer` copies any byte range, as Metal's copy does. Metal moves
//!   overlapping ranges of one buffer as `memmove` does (measured), while Vulkan forbids them, so
//!   such a copy goes through a scratch buffer.
//! - **Fills.** `vkCmdFillBuffer` writes only whole 32-bit words at offsets that are multiples of
//!   4, while Metal fills any byte range with one byte value: [`fill_pieces`] splits a fill into
//!   the words and the at most 3 bytes before and after them, which are copied from 4 bytes of the
//!   value ([`fill_word`]) in a small pattern buffer of the encoder. The words go as one
//!   `vkCmdFillBuffer` (`word_fill.rs`).
//! - **Heap shadows** (`shadow.rs`). A heap's texture held as a shadow of the heap's bytes (a
//!   depth or stencil texture, or one of a heap over host pages, `texture/heap.rs`) is an image to
//!   a blit: a command that uses its image copies it in from the bytes first where the encoder has
//!   not, and what a command writes through it is copied back by `shadow.rs`'s rules, as the
//!   encoder ends at the latest. A copy that writes every texel of a one-level, one-slice such
//!   texture replaces them (`shadow::overwritten`): it takes no copy in and no keep. A blit writes
//!   one destination, so none writes such a texture's bytes two ways.

use super::blit_fill::FillTiles;
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::degradation::Degradation;
use super::encoder::{EncoderCore, Replayed};
use super::error::{invalid, Error, Problem};
use super::fresh_unwritten::FreshRead;
use super::replay::{BlitCall, CallLog, Replay, Resource};
use super::resource::{ResourceKey, ResourceRef};
use super::shadow::{self, Covered, ShadowCopies};
use super::texture::Texture;
use super::tracker::{Access, Use, UseRange};
use super::used_resources::EncoderKind;
use super::values::{ResourceOptions, StorageMode};
use crate::device::functions::Vk;
use ash::vk;
use std::ops::Range;
use std::sync::Arc;

/// How a fill writes a byte range.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct FillPieces {
    /// The bytes before the words, fewer than 4.
    pub(super) head: Range<u64>,
    /// The whole 32-bit words at multiples of 4, for `vkCmdFillBuffer`; empty when the range holds
    /// none.
    pub(super) words: Range<u64>,
    /// The bytes after the words, fewer than 4.
    pub(super) tail: Range<u64>,
}

/// The pieces of a fill of the bytes `range` (offsets in the `VkBuffer`).
pub(super) fn fill_pieces(range: Range<u64>) -> FillPieces {
    if range.is_empty() {
        return FillPieces::default();
    }
    // The first word boundary in the range (or its end), and the last one at or after it.
    let words_start = range
        .start
        .checked_next_multiple_of(4)
        .map_or(range.end, |at| at.min(range.end));
    let words_end = (range.end / 4 * 4).max(words_start);
    FillPieces {
        head: range.start..words_start,
        words: words_start..words_end,
        tail: words_end..range.end,
    }
}

/// The 32-bit word whose four bytes are all `value`: `vkCmdFillBuffer`'s data, and the copy source
/// of a fill's head and tail.
pub(super) fn fill_word(value: u8) -> u32 {
    u32::from_ne_bytes([value; 4])
}

/// Every blit command runs in the transfer stage.
pub(super) const TRANSFER: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::TRANSFER;

/// A blit command's read of `range` of `buffer`.
pub(super) fn reads(buffer: &Buffer, range: Range<u64>) -> Use<ResourceKey> {
    reads_by(
        buffer,
        range,
        Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ),
    )
}

/// A blit command's write of `range` of `buffer`.
pub(super) fn writes(buffer: &Buffer, range: Range<u64>) -> Use<ResourceKey> {
    writes_by(
        buffer,
        range,
        Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
    )
}

/// A blit command's read of `range` of `buffer` as `read` (a copy of rows that dispatches reads
/// in a shader, `row_copies.rs`).
pub(super) fn reads_by(buffer: &Buffer, range: Range<u64>, read: Access) -> Use<ResourceKey> {
    let (identity, range) = buffer.tracked(range);
    Use::reads(identity.key, identity.tracked, UseRange::Bytes(range), read)
}

/// A blit command's write of `range` of `buffer` as `write` ([`reads_by`]).
pub(super) fn writes_by(buffer: &Buffer, range: Range<u64>, write: Access) -> Use<ResourceKey> {
    let (identity, range) = buffer.tracked(range);
    Use::writes(
        identity.key,
        identity.tracked,
        UseRange::Bytes(range),
        write,
    )
}

/// `offset..offset + size` when it lies in a buffer of `length` bytes.
pub(super) fn bytes_in(
    call: &'static str,
    offset: u64,
    size: u64,
    length: u64,
) -> Result<Range<u64>, Error> {
    match offset.checked_add(size) {
        Some(end) if end <= length => Ok(offset..end),
        _ => Err(out_of_bounds(call, offset, size, length)),
    }
}

/// The refusal of `size` bytes from `offset` past a buffer of `length` bytes.
fn out_of_bounds(call: &'static str, offset: u64, size: u64, length: u64) -> Error {
    invalid(
        call,
        Problem::RangeOutOfBounds {
            offset,
            size,
            length,
        },
    )
}

/// One region of `vkCmdCopyBuffer`.
fn region(source: u64, destination: u64, size: u64) -> vk::BufferCopy {
    vk::BufferCopy {
        src_offset: source,
        dst_offset: destination,
        size,
    }
}

/// The pattern buffer's length: 4 bytes of each byte value.
const PATTERN_LENGTH: u64 = 4 * 256;

/// A blit encoder (`MTLBlitCommandEncoder`): `Send`, not `Sync`; `end_encoding` consumes it. Its commands run in encoded
/// order, over every resource.
pub struct BlitCommandEncoder {
    core: EncoderCore,
    /// 4 bytes of each byte value at offset `4 * value`, the source of an unaligned fill's ends;
    /// made at the encoder's first such fill and retained at its end.
    pattern: Option<Buffer>,
    /// The copies of the heap shadows among the textures the encoder's blits named
    /// ([`BlitCommandEncoder::see`]), each once.
    shadows: Vec<Arc<ShadowCopies>>,
    /// Its calls so far, where the device had a buffer over host pages as it opened
    /// (`replay.rs`): `None` where it logs nothing, or no longer.
    log: Option<CallLog<BlitCall>>,
    /// The protection options it was made under where they end Metal's process at its first
    /// command (`command_buffer/protection.rs`): each command is refused.
    aborts: Option<u64>,
    /// The tiles its plain color fills cleared, which its later ones copy from again
    /// (`blit_fill.rs`). An encoding again starts with none, so it clears them again.
    fill_tiles: FillTiles,
}

impl std::fmt::Debug for BlitCommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BlitCommandEncoder")
    }
}

impl CommandBuffer {
    /// `blitCommandEncoder`. Refused while another encoder of this command buffer is open,
    /// after one was abandoned, and after commit.
    pub fn blit_command_encoder(&self) -> Result<BlitCommandEncoder, Error> {
        let core = EncoderCore::open(self, "blit_command_encoder", EncoderKind::Blit, true)?;
        Ok(BlitCommandEncoder {
            core,
            pattern: None,
            shadows: Vec::new(),
            log: (self.device().shared().rebinds.any())
                .then(|| self.device().shared().scheduler.spare_logs().blit()),
            aborts: self.protected_blit(),
            fill_tiles: FillTiles::default(),
        })
    }
}

impl BlitCommandEncoder {
    /// An encoder that encodes an ended one of `command_buffer` again (`replay.rs`): it logs
    /// nothing.
    pub(super) fn replaying(command_buffer: &CommandBuffer) -> Result<BlitCommandEncoder, Error> {
        Ok(BlitCommandEncoder {
            core: EncoderCore::replaying(command_buffer, true)?,
            pattern: None,
            shadows: Vec::new(),
            log: None,
            aborts: None,
            fill_tiles: FillTiles::default(),
        })
    }

    /// End an encoder made by [`replaying`](Self::replaying): what it recorded.
    pub(super) fn finish_replay(self) -> Result<Replayed, Error> {
        let BlitCommandEncoder {
            mut core, pattern, ..
        } = self;
        if let Some(pattern) = pattern {
            core.retain(pattern);
        }
        core.finish_replay()
    }

    /// Run `run`, a command, logged first as `call` makes it of the log, where the encoder logs
    /// (`replay.rs`): where it is refused, its log notes it so (`CallLog::refuse`), and an
    /// encoding again refuses it again and goes on. Refused before it is logged, for every
    /// command, where the encoder was made under protection options that end Metal's process at
    /// its first command (`Invalid`, `Problem::ProtectedEncoder`, `command_buffer/protection.rs`).
    pub(super) fn logged<T>(
        &mut self,
        call: impl FnOnce(&mut CallLog<BlitCall>) -> BlitCall,
        run: impl FnOnce(&mut Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.refuse_aborted()?;
        let at = match self.log.as_mut().filter(|log| !log.muted()) {
            Some(log) => {
                let call = call(log);
                let extra = call.extra();
                Some(log.push(call, extra))
            }
            None => None,
        };
        let ran = run(self);
        if let (Err(_), Some(at), Some(log)) = (&ran, at, &mut self.log) {
            log.refuse(at);
        }
        ran
    }

    /// Refuse a command where the encoder was made under protection options that end Metal's
    /// process at its first command (`Invalid`, `Problem::ProtectedEncoder`,
    /// `command_buffer/protection.rs`): a command whose first step comes before its log.
    pub(super) fn refuse_aborted(&self) -> Result<(), Error> {
        match self.aborts {
            Some(options) => Err(invalid(
                "blit_command_encoder",
                Problem::ProtectedEncoder { options },
            )),
            None => Ok(()),
        }
    }

    /// Run `call`, a public call one of the encoder's own makes, unlogged: the call that makes
    /// it is logged, and encoded again makes it again.
    pub(super) fn unlogged<T>(&mut self, call: impl FnOnce(&mut Self) -> T) -> T {
        if let Some(log) = &mut self.log {
            log.mute(true);
        }
        let made = call(self);
        if let Some(log) = &mut self.log {
            log.mute(false);
        }
        made
    }

    /// The encoder's core.
    pub(super) fn core(&mut self) -> &mut EncoderCore {
        &mut self.core
    }

    /// The tiles its plain color fills keep (`blit_fill.rs`).
    pub(super) fn fill_tiles(&mut self) -> &mut FillTiles {
        &mut self.fill_tiles
    }

    /// Record one blit command that reads and writes `uses`: the copies into the heap shadows
    /// it uses that the encoder has not copied in, and the barrier the uses need after the
    /// encoder's earlier commands, then the command through `record`. Every blit command goes
    /// through here once its checks passed, and its uses name every byte it touches, untracked
    /// memory's too (`EncoderCore::command_named`). Refused only as `EncoderCore::plan_shadows`
    /// is, for a merge, which no blit needs (the module docs).
    pub(super) fn command(
        &mut self,
        uses: &[Use<ResourceKey>],
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) -> Result<(), Error> {
        self.command_in(TRANSFER, uses, record)
    }

    /// [`command`](Self::command) for a blit command that runs in `stages`: a multisampled copy
    /// by draws (`sample_copy.rs`) also renders.
    pub(super) fn command_in(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) -> Result<(), Error> {
        self.command_in_reading(stages, uses, (&[], &[]), record)
    }

    /// [`command`](Self::command) of a blit command that reads the boxes `reads` of its source
    /// texture (`FreshUnwritten::reads`): a new texture's unwritten pattern is announced only
    /// where a box reaches it (`EncoderCore::command_named`); and writes every texel of the
    /// boxes `covered` of its destination's image: a heap shadow whose every texel they cover
    /// takes no copy in and no keep, its copy back moving what the command wrote
    /// (`shadow::overwritten`).
    pub(super) fn command_reading(
        &mut self,
        uses: &[Use<ResourceKey>],
        boxes: (&[FreshRead], &[Covered]),
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) -> Result<(), Error> {
        self.command_in_reading(TRANSFER, uses, boxes, record)
    }

    /// [`command_in`](Self::command_in) reading the boxes `reads` and writing every texel of
    /// the boxes `covered` ([`command_reading`](Self::command_reading)).
    pub(super) fn command_in_reading(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        (reads, covered): (&[FreshRead], &[Covered]),
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) -> Result<(), Error> {
        let mut shadows = shadow::copies_used(self.shadows.iter(), uses);
        shadow::overwritten(&mut shadows, uses, covered);
        let copies = self.core.plan_shadows(&shadows, uses)?;
        self.core.shadow_copies(&copies);
        self.core.command_named(stages, uses, reads, record);
        self.core.shadows_used(&shadows, uses, &[]);
        Ok(())
    }

    /// Note `texture`, which a blit names: where it is a heap shadow, whose image the blit uses
    /// (a texture over a buffer blits on its bytes, `blit_texture.rs`), its commands take its
    /// copies.
    pub(super) fn see(&mut self, texture: &Texture) {
        if texture.over_buffer() {
            return;
        }
        if let Some(copies) = texture.shadow() {
            if !self.shadows.iter().any(|seen| Arc::ptr_eq(seen, copies)) {
                self.shadows.push(Arc::clone(copies));
            }
        }
    }

    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`: `size` bytes, at any
    /// offsets, as on Metal.
    ///
    /// Overlapping ranges of one buffer are moved as `memmove` moves them, as Metal does, and so
    /// are those of two no-copy buffers made over the same bytes, which lie in one `VkBuffer`.
    /// Vulkan forbids such a copy, so it costs more: a scratch buffer of `size` bytes outside the
    /// Metal address space, cleared by a fill this call waits for, and a second copy of the range
    /// through it.
    ///
    /// A size of 0 copies nothing, wherever its offsets lie, and the encoder's other commands
    /// run, as plain Metal does (Metal's validation stops it).
    ///
    /// A no-copy source buffer is read past its length to the end of its host pages, the
    /// caller's bytes, as plain Metal reads them (its validation stops it).
    ///
    /// Refused (`Invalid`): a range past either buffer (but that read), a buffer of another
    /// device.
    pub fn copy_from_buffer_to_buffer(
        &mut self,
        source_buffer: &Buffer,
        source_offset: u64,
        destination_buffer: &Buffer,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                BlitCall::CopyBufferToBuffer(
                    source_buffer.clone(),
                    source_offset,
                    destination_buffer.clone(),
                    destination_offset,
                    size,
                )
            },
            |this| {
                const CALL: &str = "copy_from_buffer_to_buffer";
                this.core.check_device(CALL, source_buffer.device())?;
                this.core.check_device(CALL, destination_buffer.device())?;
                if size == 0 {
                    return Ok(());
                }
                // A no-copy buffer's source reads its pages past its length.
                let from = bytes_in(CALL, source_offset, size, source_buffer.copy_reach())
                    .map_err(|_| {
                        out_of_bounds(CALL, source_offset, size, source_buffer.length())
                    })?;
                let to = bytes_in(CALL, destination_offset, size, destination_buffer.length())?;
                // Read as a blit's: a no-copy buffer's import released while idle is imported again
                // (`host_residency.rs`).
                let (source, source_at) = this.core.blit_at(source_buffer, source_offset)?;
                let (destination, destination_at) =
                    this.core.blit_at(destination_buffer, destination_offset)?;
                // One buffer's bytes, or two no-copy buffers' made over the same bytes, which lie
                // in one import (`host_aliases.rs`): one `VkBuffer` from one byte 0.
                let one_bytes = source == destination
                    && source_at - source_offset == destination_at - destination_offset;
                let overlap = one_bytes && from.start < to.end && to.start < from.end;
                if overlap {
                    let scratch = this
                        .core
                        .device()
                        .own_buffer(size, ResourceOptions::private())?;
                    let (through, through_at) = scratch.at(0);
                    this.command(
                        &[reads(source_buffer, from), writes(&scratch, 0..size)],
                        |device, raw| {
                            // SAFETY: the encoder's command buffer is recording; the buffers are
                            // live and retained until it completes, and the regions lie inside
                            // them.
                            unsafe {
                                device.cmd_copy_buffer(
                                    raw,
                                    source,
                                    through,
                                    &[region(source_at, through_at, size)],
                                );
                            }
                        },
                    )?;
                    this.command(
                        &[reads(&scratch, 0..size), writes(destination_buffer, to)],
                        |device, raw| {
                            // SAFETY: as above.
                            unsafe {
                                device.cmd_copy_buffer(
                                    raw,
                                    through,
                                    destination,
                                    &[region(through_at, destination_at, size)],
                                );
                            }
                        },
                    )?;
                    this.core.retain(scratch);
                    this.core
                        .device()
                        .degrade(Degradation::OverlappingBufferCopy, || {
                            let object = format!(
                                "a copy of {size} bytes from offset {source_offset} to offset \
                     {destination_offset} of one buffer"
                            );
                            (
                                object,
                                "the ranges overlap, which Vulkan's copy forbids".to_owned(),
                            )
                        });
                } else {
                    this.command(
                        &[reads(source_buffer, from), writes(destination_buffer, to)],
                        |device, raw| {
                            // SAFETY: as above; the regions do not overlap.
                            unsafe {
                                device.cmd_copy_buffer(
                                    raw,
                                    source,
                                    destination,
                                    &[region(source_at, destination_at, size)],
                                );
                            }
                        },
                    )?;
                }
                Ok(())
            },
        )
    }

    /// `fillBuffer:range:value:`: every byte of `range` becomes `value`, for any range, as on
    /// Metal.
    ///
    /// The range's whole words are filled as one `vkCmdFillBuffer` (`word_fill.rs`); the at most
    /// 3 bytes at each end are copied from the encoder's pattern buffer (1 KiB, made at its first
    /// unaligned fill).
    ///
    /// An empty range fills nothing, wherever it lies, and the encoder's other commands run, as
    /// plain Metal does (Metal's validation stops it).
    ///
    /// Refused (`Invalid`): a range past the buffer, a buffer of another device.
    pub fn fill_buffer(
        &mut self,
        buffer: &Buffer,
        range: Range<u64>,
        value: u8,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::FillBuffer(buffer.clone(), range.clone(), value),
            |this| {
                const CALL: &str = "fill_buffer";
                this.core.check_device(CALL, buffer.device())?;
                let size = range.end.saturating_sub(range.start);
                if size == 0 {
                    return Ok(());
                }
                let range = bytes_in(CALL, range.start, size, buffer.length())?;
                let metal = (range.start, range.end);
                // The words and ends are the `VkBuffer`'s: a no-copy buffer's start in it need not
                // be a multiple of 4 (`Buffer::base`). Read as a blit's (`host_residency.rs`).
                let (destination, start) = this.core.blit_at(buffer, range.start)?;
                let pieces = fill_pieces(start..start + size);
                let mut uses = vec![writes(buffer, range)];
                let pattern = if pieces.head.is_empty() && pieces.tail.is_empty() {
                    None
                } else {
                    let pattern = this.pattern()?;
                    let at = 4 * u64::from(value);
                    uses.push(reads(&pattern, at..at + 4));
                    Some(pattern.at(at))
                };
                let split = pattern.is_some();
                let owner = this.core.device().clone();
                this.command(&uses, |device, raw| {
                    let ends = [&pieces.head, &pieces.tail];
                    let ends = ends.iter().filter(|end| !end.is_empty());
                    let from = pattern.map_or(0, |(_, at)| at);
                    let regions: Vec<vk::BufferCopy> = ends
                        .map(|end| region(from, end.start, end.end - end.start))
                        .collect();
                    // SAFETY: the encoder's command buffer is recording; the buffers are live and
                    // retained until it completes; the words start and end at multiples of 4 inside
                    // the buffer, and the ends' regions lie inside both buffers.
                    unsafe {
                        if let Some((pattern, _)) = pattern {
                            device.cmd_copy_buffer(raw, pattern, destination, &regions);
                        }
                        crate::word_fill::record(
                            &owner,
                            raw,
                            destination,
                            pieces.words.clone(),
                            fill_word(value),
                        );
                    }
                })?;
                if split {
                    this.core
                        .device()
                        .degrade(Degradation::SplitBufferFill, || {
                            let object =
                                format!("a fill of bytes {}..{} of a buffer", metal.0, metal.1);
                            let reason = format!(
                    "they lie at bytes {start}..{} of the buffer's VkBuffer, and Vulkan fills only \
                     whole 4-byte words at multiples of 4",
                    start + size
                );
                            (object, reason)
                        });
                }
                Ok(())
            },
        )
    }

    /// The encoder's pattern buffer, made on first use: 4 bytes of each byte value at offset
    /// `4 * value`, written through its mapping (host-coherent write-combined `Shared` memory,
    /// seen by the device at the commit's submission).
    fn pattern(&mut self) -> Result<Buffer, Error> {
        if let Some(pattern) = &self.pattern {
            return Ok(pattern.clone());
        }
        let pattern = self.core.device().write_combined_buffer(PATTERN_LENGTH)?;
        let Some(contents) = pattern.contents() else {
            return Err(Error::Vulkan(super::error::VkResultCode(
                vk::Result::ERROR_MEMORY_MAP_FAILED.as_raw(),
            )));
        };
        let bytes: Vec<u8> = (0..=u8::MAX).flat_map(|value| [value; 4]).collect();
        // SAFETY: the mapping covers the buffer's `PATTERN_LENGTH` bytes, which nothing else uses
        // yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        self.pattern = Some(pattern.clone());
        Ok(pattern)
    }

    /// `synchronizeResource:`: the CPU sees the GPU's writes to a `Managed` resource once the
    /// command buffer completes. Every command buffer ends with a barrier to the host. On a device
    /// with unified memory a managed buffer is host-coherent memory the GPU uses itself, and a
    /// texture's CPU access copies from its image, so nothing more is recorded. On one without, a
    /// managed buffer's bytes, or a texture's over one, are copied to its host copy, which
    /// `contents` maps (`managed.rs`), after the encoder's earlier commands.
    ///
    /// On a resource of another storage mode it records nothing, as plain Metal does (RT2
    /// `managed.synchronize.shared`, `.private`: `Shared` and `Private` buffers
    /// and textures; Metal's validation stops it).
    ///
    /// Refused (`Invalid`): a resource of another device.
    pub fn synchronize_resource(&mut self, resource: ResourceRef<'_>) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::SynchronizeResource(Resource::one(resource)),
            |this| {
                const CALL: &str = "synchronize_resource";
                this.core.check_device(CALL, resource.device())?;
                if resource.storage_mode() != StorageMode::Managed {
                    return Ok(());
                }
                let (buffer, range) = match resource {
                    ResourceRef::Buffer(buffer) => (buffer, 0..buffer.length()),
                    ResourceRef::Texture(texture) => match texture.buffer_texels() {
                        // Where the GPU or the CPU finds the texels (an IOSurface plane's two
                        // layouts), in the buffer: those past its length lie in the texture's
                        // `Shared` tail, which takes no synchronization (`buffer_texture.rs`).
                        Some(texels) => {
                            let span = texels.span();
                            let end = span.end.min(texels.buffer.length());
                            (&texels.buffer, span.start.min(end)..end)
                        }
                        None => return Ok(()),
                    },
                    ResourceRef::IndirectCommandBuffer(_) => return Ok(()),
                };
                let Some(host) = buffer.host_copy() else {
                    return Ok(());
                };
                if range.is_empty() {
                    return Ok(());
                }
                let (own, start) = buffer.at(range.start);
                let copy = region(start, range.start, range.end - range.start);
                this.command(&[reads(buffer, range)], |device, raw| {
                    // SAFETY: the encoder's command buffer is recording; the buffer and its host
                    // copy are live and retained until it completes, and the range lies in both
                    // (the host copy holds the buffer's bytes from its byte 0).
                    unsafe { device.cmd_copy_buffer(raw, own, host, &[copy]) };
                })?;
                this.core.retain_shared(buffer.held());
                Ok(())
            },
        )
    }

    /// `endEncoding`.
    pub fn end_encoding(self) -> Result<(), Error> {
        // The core's shadows copy back what the blits wrote through heap shadows. The fill tiles
        // are their command buffer's, which holds them until it completes.
        let BlitCommandEncoder {
            mut core,
            pattern,
            shadows: _,
            log,
            aborts: _,
            fill_tiles: _,
        } = self;
        if let Some(pattern) = pattern {
            core.retain(pattern);
        }
        core.end_with_replay(EncoderKind::Blit, None, log.map(Replay::Blit))
    }
}
