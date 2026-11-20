//! Tiled textures over a buffer of more than one level
//! (`newTiledTextureWithDescriptor:offset:bytesPerRow:`, Apple's SPI, `docs/REPLAY-STREAM.md`
//! section 10), played as x86 Macs were measured to do: an ordinary texture of the descriptor
//! whose first
//! level takes a copy of the buffer's bytes from the offset, rows `bytesPerRow` apart, and whose
//! later levels are then made from the first as `generateMipmapsForTexture:` makes them, at the
//! first use after the making. The take is that copy and generation in a blit encoder of a command
//! buffer of the play's own, committed on the queue of the command buffer whose encoder used the
//! texture first, just before it and after its copies from the guest's RAM: it reads the bytes as
//! they are when that command buffer runs. A format whose mipmaps the runtime refuses or cannot
//! filter keeps its later levels. This work is the play's stand-in for the call, counted apart from
//! the stream's (`tiled_*`).
//!
//! The take runs ahead of the whole command buffer, not just ahead of the encoder that used the
//! texture first, so what an earlier encoder of that command
//! buffer writes into the bytes or the texture lands after it; a texture an encoder names only
//! through an argument buffer with no use named is not seen; and a `replaceRegion` of the first
//! level, which writes the bytes and asks the next use for another take, is refused.

use super::super::args::{index, refuse, uint};
use super::{failed, objects_named, unaddressed, NewRuntime, Object};
use crate::replay::player::{Made, PlayError};
use crate::replay::stream::Call;
use crate::replay::value::Value;
use metal2vulkan_runtime as rt;
use std::collections::{HashMap, VecDeque};

/// A tiled texture's take: its texture, and its buffer, offset and row pitch.
#[derive(Clone)]
struct Take {
    texture: rt::Texture,
    buffer: rt::Buffer,
    offset: u64,
    bytes_per_row: u64,
}

/// What the play's stand-in did (`counted`).
#[derive(Default)]
struct Counts {
    textures: u64,
    takes: u64,
    take_bytes: u64,
    generations: u64,
    levels_kept: u64,
    gpu_nanos: u64,
}

/// The tiled textures, their takes owed and the command buffers of the play's own that ran them.
#[derive(Default)]
pub(super) struct Tiled {
    /// Each tiled texture, and each view of one, by object: the tiled texture's object.
    of: HashMap<u64, u64>,
    /// Each live tiled texture: its take, and whether its next use takes it.
    textures: HashMap<u64, (Take, bool)>,
    /// The takes each command buffer not yet committed owes, in the order its encoders first used
    /// their textures.
    takes: HashMap<u64, Vec<Take>>,
    /// The command buffers of the play's own not yet seen complete, for their GPU time.
    timing: VecDeque<rt::CommandBuffer>,
    counts: Counts,
}

impl Tiled {
    /// Whether no tiled texture lives.
    pub(super) fn is_empty(&self) -> bool {
        self.of.is_empty()
    }

    /// Object `id`'s release: a tiled texture, a view of one, or a command buffer released
    /// uncommitted, which runs none of its takes.
    pub(super) fn released(&mut self, id: u64) {
        if self.of.remove(&id) == Some(id) {
            self.textures.remove(&id);
        }
        self.takes.remove(&id);
    }

    /// Adds the GPU time of the command buffers of the play's own that completed since the last
    /// call.
    fn timed(&mut self) {
        while let Some(front) = self.timing.front() {
            match front.status() {
                rt::CommandBufferStatus::Completed => {
                    let (start, end) = (front.gpu_start_time(), front.gpu_end_time());
                    self.counts.gpu_nanos += end.saturating_sub(start);
                }
                rt::CommandBufferStatus::Error => {}
                _ => return,
            }
            self.timing.pop_front();
        }
    }

    /// The executed counters.
    pub(super) fn counted(&mut self) -> [(&'static str, u64); 6] {
        self.timed();
        let c = &self.counts;
        [
            ("tiled_textures", c.textures),
            ("tiled_takes", c.takes),
            ("tiled_take_bytes", c.take_bytes),
            ("tiled_mipmap_generations", c.generations),
            ("tiled_levels_kept", c.levels_kept),
            ("tiled_gpu_nanos", c.gpu_nanos),
        ]
    }

    /// The report's line.
    pub(super) fn line(&self) -> String {
        let c = &self.counts;
        format!(
            "tiled textures over a buffer {}: {} takes \
             at a first use, each a copy of the buffer's bytes into the first level ({} bytes) \
             and a generation of the later levels ({} generated, {} kept as the format's are), \
             in command buffers of the play's own; {} ns of GPU time",
            c.textures, c.takes, c.take_bytes, c.generations, c.levels_kept, c.gpu_nanos
        )
    }
}

impl NewRuntime {
    /// `newTiledTextureWithDescriptor:offset:bytesPerRow:` on `buffer` of a descriptor of more
    /// than one level: an ordinary texture of `descriptor`, which takes the buffer's bytes at its
    /// first use. A type other than 2D is refused.
    pub(super) fn tiled_texture(
        &mut self,
        call: &Call,
        buffer: &rt::Buffer,
        descriptor: &rt::TextureDescriptor,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        if descriptor.texture_type != rt::TextureType::Type2D {
            return Err(refuse(
                call,
                "a tiled texture of more than one level, not 2D",
            ));
        }
        let (offset, bytes_per_row) = (uint(call, 1)?, uint(call, 2)?);
        let texture = self.new_texture(call, descriptor, None, false)?;
        let take = Take {
            texture: texture.clone(),
            buffer: buffer.clone(),
            offset,
            bytes_per_row,
        };
        self.tiled.of.insert(call.result, call.result);
        self.tiled.textures.insert(call.result, (take, true));
        self.tiled.counts.textures += 1;
        unaddressed(Object::Texture(texture))
    }

    /// Before `call` plays: a view of a tiled texture is noted as one; the first use of a tiled
    /// texture since its making, an encoder's command or a render pass's attachment that names
    /// it or a view of it, owes its take to the command buffer; a `replaceRegion` of a tiled
    /// texture's first level is refused.
    pub(super) fn tiled_named(&mut self, call: &Call) -> Result<(), PlayError> {
        if let Some(&tiled) = self.tiled.of.get(&call.receiver) {
            if call.selector.starts_with("newTextureView") {
                self.tiled.of.insert(call.result, tiled);
            }
            if call.selector.starts_with("replaceRegion:") && index(call, 1)? == 0 {
                let why = "a replaceRegion of a tiled texture's first level, which writes its \
                           buffer's bytes";
                return Err(refuse(call, why));
            }
        }
        let mut named = Vec::new();
        let command_buffer = match self.owners.get(&call.receiver) {
            Some(&command_buffer) => {
                objects_named(&call.args, &mut named);
                command_buffer
            }
            None if &*call.selector == "renderCommandEncoderWithDescriptor:" => {
                let pass = call.args.first().and_then(Value::as_object);
                if let Some(Object::Pass(pass)) = pass.and_then(|id| self.objects.get(&id)) {
                    named.extend(pass.textures());
                }
                call.receiver
            }
            None => return Ok(()),
        };
        for id in named {
            let Some(texture) = self.tiled.of.get(&id) else {
                continue;
            };
            let Some((take, owed)) = self.tiled.textures.get_mut(texture) else {
                continue;
            };
            if *owed {
                *owed = false;
                let takes = self.tiled.takes.entry(command_buffer).or_default();
                takes.push(take.clone());
            }
        }
        Ok(())
    }

    /// Before command buffer `call.receiver` commits, after its copies from the guest's RAM: the
    /// takes it owes, in a command buffer of the play's own committed now, ahead of it on its
    /// queue.
    pub(super) fn tiled_take(&mut self, call: &Call) -> Result<(), PlayError> {
        let Some(takes) = self.tiled.takes.remove(&call.receiver) else {
            return Ok(());
        };
        let queue = self.queue_for(call)?;
        let command_buffer = queue.command_buffer().map_err(failed(call))?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(failed(call))?;
        for take in &takes {
            let d = *take.texture.descriptor();
            let raw = u64::from(d.pixel_format.as_raw());
            let query = rt::TextureLevelQuery::plane_2d(raw, d.width.into(), d.height.into());
            let level = self.device.texture_level_info(&query);
            // The rows of texels, or of blocks, the first level holds.
            let rows = level.bytes_per_image / level.bytes_per_row.max(1);
            let layout = rt::BufferLayout {
                offset: take.offset,
                bytes_per_row: take.bytes_per_row,
                bytes_per_image: take.bytes_per_row * rows,
            };
            let first = rt::TexturePlace {
                slice: 0,
                level: 0,
                origin: [0; 3],
            };
            let size = [d.width, d.height, 1];
            (blit.copy_from_buffer(&take.buffer, layout, size, &take.texture, first))
                .map_err(failed(call))?;
            let counts = &mut self.tiled.counts;
            counts.takes += 1;
            counts.take_bytes += level.bytes_per_image;
            match blit.generate_mipmaps(&take.texture) {
                Ok(()) => counts.generations += 1,
                Err(
                    rt::Error::Refused(rt::Refusal::MipmapFormat { .. })
                    | rt::Error::Unsupported(rt::Gap::MipmapFilter { .. }),
                ) => counts.levels_kept += 1,
                Err(error) => return Err(failed(call)(error)),
            }
        }
        blit.end_encoding();
        self.tiled.timing.push_back(command_buffer.clone());
        self.commit(&queue, command_buffer, 0).map_err(failed(call))
    }
}
