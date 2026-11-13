//! What a render stage's argument table binds, as a render encoder's checks at the call read it
//! (`render/call_checks.rs`): each buffer index's buffer, inline bytes or visible-function table,
//! and each setter's refusals, the table's own (`argument_table.rs`) made over these facts with
//! its own helpers. The lowering's table holds the same after the same calls.

use crate::argument_table::{
    check_bound_buffer, check_index, check_range, BUFFER_SLOTS, INLINE_BYTES_LIMIT, SAMPLER_SLOTS,
    TEXTURE_SLOTS,
};
use crate::buffer::Buffer;
use crate::device::Device;
use crate::encoder::check_device;
use crate::error::{invalid, Error, Problem};
use crate::function_table::VisibleFunctionTable;
use crate::sampler::SamplerState;
use crate::texture::Texture;

/// What one buffer index binds.
#[derive(Clone, Default)]
enum Bound {
    #[default]
    Nothing,
    /// A buffer at an offset.
    Buffer(Buffer, u64),
    /// Inline bytes.
    Bytes,
    /// A visible-function table.
    Table,
    /// The null table an offset made of a table (`ArgumentTable::null_tables`).
    Null,
}

/// One render stage's table's facts (the module docs).
pub(super) struct TableFacts {
    buffers: [Bound; BUFFER_SLOTS as usize],
}

impl Default for TableFacts {
    fn default() -> TableFacts {
        TableFacts {
            buffers: std::array::from_fn(|_| Bound::Nothing),
        }
    }
}

impl TableFacts {
    /// Let go of every binding: a new pass's table.
    pub(super) fn clear(&mut self) {
        self.buffers.fill(Bound::Nothing);
    }

    /// Whether a buffer bound holds more than `bytes` past its bind offset.
    pub(super) fn holds_past(&self, bytes: u64) -> bool {
        self.buffers.iter().any(|bound| match bound {
            Bound::Buffer(buffer, offset) => buffer.length().saturating_sub(*offset) > bytes,
            _ => false,
        })
    }

    /// Whether a visible-function table, or the null table an offset made of one, is bound.
    pub(super) fn holds_tables(&self) -> bool {
        (self.buffers.iter()).any(|bound| matches!(bound, Bound::Table | Bound::Null))
    }

    /// `setBuffer:offset:atIndex:` and its stride form, as `ArgumentTable::set_buffer_with_stride`
    /// refuses it.
    pub(super) fn set_buffer(
        &mut self,
        device: &Device,
        call: &'static str,
        (buffer, offset): (Option<&Buffer>, u64),
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        if let Some(buffer) = buffer {
            check_bound_buffer(device, call, (buffer, offset), false)?;
        }
        self.buffers[index as usize] = match buffer {
            Some(buffer) => Bound::Buffer(buffer.clone(), offset),
            None => Bound::Nothing,
        };
        Ok(())
    }

    /// `setBuffers:offsets:withRange:` and its stride form (`strides`, their count), as
    /// `ArgumentTable::set_buffers_with_strides` refuses it.
    pub(super) fn set_buffers(
        &mut self,
        device: &Device,
        call: &'static str,
        (buffers, offsets, strides): (&[Option<Buffer>], &[u64], Option<usize>),
        first: u32,
    ) -> Result<(), Error> {
        if buffers.len() != offsets.len() || strides.is_some_and(|s| s != buffers.len()) {
            return Err(invalid(call, Problem::LengthMismatch));
        }
        check_range(call, first, buffers.len(), BUFFER_SLOTS)?;
        for (buffer, &offset) in buffers.iter().zip(offsets) {
            if let Some(buffer) = buffer {
                check_bound_buffer(device, call, (buffer, offset), false)?;
            }
        }
        for ((buffer, &offset), index) in buffers.iter().zip(offsets).zip(first..) {
            self.buffers[index as usize] = match buffer {
                Some(buffer) => Bound::Buffer(buffer.clone(), offset),
                None => Bound::Nothing,
            };
        }
        Ok(())
    }

    /// `setBufferOffset:atIndex:` and its stride form, as
    /// `ArgumentTable::set_buffer_offset_with_stride` takes and refuses it.
    pub(super) fn set_buffer_offset(
        &mut self,
        call: &'static str,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        let bound = &mut self.buffers[index as usize];
        match bound {
            // Inline bytes then read as zeros, at any offset, until the index is bound again.
            Bound::Bytes => Ok(()),
            // A table reads as the null table until the index is bound again.
            Bound::Table => {
                *bound = Bound::Null;
                Ok(())
            }
            Bound::Buffer(buffer, at) => {
                let length = buffer.length();
                if offset > length {
                    return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
                }
                *at = offset;
                Ok(())
            }
            // Nothing bound keeps no offset, in a render stage's table as in a compute one: the
            // index stays unbound and the draw reads zeros, as plain Metal's does
            // (`ArgumentTable::set_buffer_offset_with_stride`).
            Bound::Nothing | Bound::Null => Ok(()),
        }
    }

    /// `setBytes:length:atIndex:` and its forms, of `length` bytes, as `ArgumentTable::set_bytes`
    /// refuses it.
    pub(super) fn set_bytes(
        &mut self,
        call: &'static str,
        length: usize,
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        let length = length as u64;
        if length > INLINE_BYTES_LIMIT {
            return Err(invalid(
                call,
                Problem::InlineBytesLength {
                    length,
                    limit: INLINE_BYTES_LIMIT,
                },
            ));
        }
        self.buffers[index as usize] = Bound::Bytes;
        Ok(())
    }

    /// `setVisibleFunctionTables:withBufferRange:` and its one-table form, as
    /// `ArgumentTable::set_function_tables` refuses it.
    pub(super) fn set_function_tables<'a>(
        &mut self,
        device: &Device,
        call: &'static str,
        tables: impl ExactSizeIterator<Item = Option<&'a VisibleFunctionTable>> + Clone,
        first: u32,
    ) -> Result<(), Error> {
        check_range(call, first, tables.len(), BUFFER_SLOTS)?;
        for table in tables.clone().flatten() {
            check_device(device, call, table.generation())?;
        }
        for (table, index) in tables.zip(first..) {
            self.buffers[index as usize] = match table {
                Some(_) => Bound::Table,
                None => Bound::Nothing,
            };
        }
        Ok(())
    }
}

/// `setTextures:withRange:` and its one-texture form, as `ArgumentTable::set_textures` refuses
/// it: no table fact changes.
pub(super) fn check_textures<'a>(
    device: &Device,
    call: &'static str,
    textures: impl ExactSizeIterator<Item = Option<&'a Texture>> + Clone,
    first: u32,
) -> Result<(), Error> {
    check_range(call, first, textures.len(), TEXTURE_SLOTS)?;
    for texture in textures.flatten() {
        check_device(device, call, texture.generation())?;
    }
    Ok(())
}

/// `setSamplerStates:withRange:` and its forms (`clamps`, the minimums' and maximums' counts), as
/// `ArgumentTable::set_samplers` refuses it: no table fact changes.
pub(super) fn check_samplers<'a>(
    device: &Device,
    call: &'static str,
    samplers: impl ExactSizeIterator<Item = Option<&'a SamplerState>> + Clone,
    clamps: Option<(usize, usize)>,
    first: u32,
) -> Result<(), Error> {
    if let Some((min, max)) = clamps {
        if min != samplers.len() || max != samplers.len() {
            return Err(invalid(call, Problem::LengthMismatch));
        }
    }
    check_range(call, first, samplers.len(), SAMPLER_SLOTS)?;
    for sampler in samplers.flatten() {
        check_device(device, call, sampler.generation())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `setBufferOffset:atIndex:` at an index that binds nothing, and at one whose
    /// visible-function table an earlier offset made the null table, keeps no offset and leaves
    /// the index as it was, as the stage table itself does
    /// (`ArgumentTable::take_buffer_offset`): plain Metal keeps no offset there and the draw reads
    /// zeros, byte for byte as it reads an index never bound (an Apple M3 Max under macOS 27.0.1,
    /// that method's own doc).
    #[test]
    fn an_offset_at_an_unbound_index_keeps_nothing() {
        let mut facts = TableFacts::default();
        for index in [0u32, 7, 30] {
            for offset in [0u64, 64, u64::MAX] {
                assert_eq!(
                    facts.set_buffer_offset("set_vertex_buffer_offset", offset, index),
                    Ok(()),
                    "nothing bound at {index}, offset {offset}"
                );
            }
        }
        // Nothing is bound, so nothing reads past any offset and no table is bound.
        assert!(!facts.holds_past(0));
        assert!(!facts.holds_tables());
        // The null table an offset made of a visible-function table stays the null table.
        facts.buffers[3] = Bound::Null;
        assert_eq!(
            facts.set_buffer_offset("set_vertex_buffer_offset", 64, 3),
            Ok(())
        );
        assert!(matches!(facts.buffers[3], Bound::Null));
        assert!(facts.holds_tables());
        // An index past the table is still refused, so the test reads a real check.
        assert!(facts
            .set_buffer_offset("set_vertex_buffer_offset", 0, BUFFER_SLOTS)
            .is_err());
    }
}
