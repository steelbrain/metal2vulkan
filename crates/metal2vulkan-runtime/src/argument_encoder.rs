//! Argument encoders (`MTLArgumentEncoder`).
//!
//! An encoder writes the arguments of an argument buffer at the places a
//! [`bind::ArgumentBufferLayout`](metal2vulkan::bind::ArgumentBufferLayout) gives: the layout of the
//! argument buffer a function takes (`Function::new_argument_encoder_with_buffer_index`), the
//! layout of a pointer member's argument buffer (`new_argument_encoder_for_buffer_at_index`), the
//! layout of argument descriptors (`Device::new_argument_encoder_with_arguments`), or the layout a
//! reflected buffer binding states (`Device::new_argument_encoder_with_buffer_binding`). Each
//! argument is the 8 bytes, little endian, a program could write itself: a buffer's Metal
//! address plus the offset, and every other object's `gpu_resource_id`; `None` writes 0. So
//! encoded and hand-written argument buffers reach the translated functions alike, through the
//! resident and handle tables. An encoder makes nothing resident: as on Metal, the program calls
//! `use_resource(s)` or `use_heap(s)` for what the function reaches.
//!
//! The bytes are written at once, through the argument buffer's mapping: a `Shared` or `Managed`
//! buffer, as Metal's validation requires. Writing them while a command reads them races, as on
//! Metal.
//!
//! **What Metal's validation stops is refused** (`Invalid`, [`ArgumentEncoderProblem`]), and
//! nothing is written; measured on an M3 Max and an M4 with `MTL_DEBUG_LAYER=1` (probes), where
//! Metal without validation writes regardless, or crashes:
//! - a setter or `constant_data` before an argument buffer is set;
//! - an argument buffer of `Private` storage, or one whose `offset + encoded_length` passes its
//!   length (`RangeOutOfBounds`); an unaligned offset is taken, as Metal takes it;
//! - a setter's index of an argument of another type ("Trying to set a Texture at index 0 but the
//!   argument buffer has a Buffer at this index"), and one no argument has where the 8 bytes at
//!   the arguments' offset would pass the argument buffer's end; a range form checks every index
//!   before writing any.
//!
//! **Where plain Metal gives a deterministic, measured answer, the encoder gives it, even where
//! the debug layer asserts** (the orchestrator's parity rule):
//! - `constant_data` at an index no argument has answers the arguments' start (offset 0), and at
//!   an argument of another type that argument's own offset, as plain Metal's
//!   `constantDataAtIndex:` did on an M4; `MTL_DEBUG_LAYER=1` asserts at both
//!   ("Index 2 does not correspond to any valid member of the argument buffer");
//! - every setter writes its word at the arguments' offset 0 for an index no argument has,
//!   whatever argument lies there, a range's words in turn, so the last stands, and in a layout
//!   of fewer than 8 bytes past its encoded length (an M4);
//! - `set_buffer` with no buffer writes 0 whatever the offset ("offset(16) must be 0 when buffer
//!   is nil");
//! - a sampler state made without `support_argument_buffers` is written as any other, and a
//!   kernel samples through it as its descriptor says ("Sampler state did not have
//!   supportArgumentBuffers flag set on creation"; and the probe 7 for an ID
//!   written by hand);
//! - a compute or render pipeline state made without `support_indirect_command_buffers`, a tile
//!   pipeline's among them, is written as its resource ID, which is 0 ("Setting a pipeline that
//!   does not have supportIndirectCommandBuffers = YES is invalid"). An indirect command a kernel
//!   encodes with that word names no state, as one encoded from `None` does: Metal's GPU faulted
//!   running one, and the runtime's execution runs nothing for it;
//! - an indirect command buffer of no command types is written as its resource ID, as a buffer of
//!   command types is. Its command types decide nothing about the word: the ID is the buffer's
//!   place among the device's indirect command buffers, so two buffers encoded in one process take
//!   two IDs whatever their command types, and the first of them takes 1 whichever it is (an M3
//!   Max, macOS 27.0.1, build 26A434, the validation layer off, one shape per process, four
//!   processes a shape, all four alike: a buffer of no command types and a `Draw` buffer encoded
//!   side by side into one argument buffer took `0100000000000000` and `0200000000000000` in the
//!   order they were made, either order, and so did two `Draw` buffers and two buffers of no
//!   command types; `probes/icb_argument_encode_and_copy_destinations.m`). A kernel that reads
//!   such a word reaches the buffer: `command_buffer::size()` answered 4 for a buffer of 4
//!   commands and no command types, and `render_command(cb, 0).reset()` ran with no fault and no
//!   error, where the same reset on a `Draw` buffer removed the draw the CPU had set (the same
//!   probe, four processes a shape). A render pass's execution of such a buffer paints nothing,
//!   before the kernel and after it.
//!
//! Where plain Metal's result is undefined (memory-dependent, a crash, or unmeasured), the call is
//! refused, as validation refuses it: the list above;
//! - a buffer offset at or past the buffer's length (`OffsetOutOfRange`);
//! - an object of another device (`ForeignObject`).
//!
//! **What the runtime does not build has no setter**: the runtime has no intersection function
//! tables or acceleration structures (ray tracing is out of scope), and its depth-stencil states
//! have no resource ID, so their arguments keep the bytes the buffer holds. A render pipeline
//! state's resource ID is its ID among the states made with `supportIndirectCommandBuffers`, as a
//! compute state's is.

use super::buffer::Buffer;
use super::device::device_error;
use super::error::{internal, invalid, nil, Error, Problem};
use super::function_table::VisibleFunctionTable;
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::library::Function;
use super::pipeline::{ComputePipelineState, RenderPipelineState};
use super::sampler::SamplerState;
use super::texture::Texture;
use super::{Device, StorageMode};
use crate::derived::DerivedMap;
use metal2vulkan::bind::{
    ArgumentBufferLayout, ArgumentDescriptor, ArgumentLayoutError, ArgumentPlace, ArgumentType,
    ARGUMENT_HANDLE_BYTES,
};
use metal2vulkan::reflection::BufferBinding;
use metal2vulkan::Budget;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

/// What an argument encoder call did wrong (`Problem::ArgumentEncoder`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgumentEncoderProblem {
    /// The layout cannot be made: no argument buffer at the function's buffer index, an array of
    /// arrays, or descriptors Metal stops (`bind::ArgumentLayoutError`). Boxed, so a [`Problem`]
    /// stays small: an [`Error`] holds one in each of two kinds.
    ///
    /// Basis: unknown (the layout's own refusals carry their reasons in
    /// `bind::ArgumentLayoutError`, which this column does not read, so Metal's answer for each is
    /// not settled here).
    Layout(Box<ArgumentLayoutError>),
    /// `new_argument_encoder_for_buffer_at_index`: the buffer at `index` points to no argument
    /// buffer, for which Metal answers nil ("No argument buffer at that index"), an
    /// [`Error::Nil`](crate::Error::Nil).
    ///
    /// Basis: metal-documented (Metal's validation says "No argument buffer at that index", for
    /// which it answers nil).
    NoNestedArgumentBuffer {
        /// The argument's index.
        index: u32,
    },
    /// A setter or `constant_data` with no argument buffer set ("No argument buffer is set").
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal's process ends, SIGSEGV,
    /// at a setter with no argument buffer set, where `constantDataAtIndex:` answers a null
    /// pointer).
    NoArgumentBuffer,
    /// No argument of the layout has `index` ("Index 15 does not correspond to any valid member
    /// of the argument buffer", "index (1000) is outside of the valid index range"), where the 8
    /// bytes at the arguments' offset would pass the argument buffer's end, and for a range whose
    /// indices pass `u32::MAX` (`index` is then `u32::MAX`). Elsewhere every setter writes there,
    /// as plain Metal does.
    ///
    /// Basis: metal-takes (an Apple M3 Max, macOS 27.0.1: plain Metal takes
    /// `setBuffer:offset:atIndex:15` on a layout holding index 0 alone, writes nothing and returns,
    /// its validation alone saying "index (15) is outside of the valid index range `[0, 0]`").
    NoArgument {
        /// The index given.
        index: u32,
    },
    /// The argument at `index` is of type `found`, the call writes `expected`.
    ///
    /// Basis: metal-takes (an Apple M3 Max, macOS 27.0.1: plain Metal takes `setTexture:atIndex:0`
    /// where the layout says a pointer, and returns).
    ArgumentType {
        /// The argument's index.
        index: u32,
        /// The type the call writes.
        expected: ArgumentType,
        /// The argument's type.
        found: ArgumentType,
    },
}

fn problem(call: &'static str, problem: ArgumentEncoderProblem) -> Error {
    invalid(call, Problem::ArgumentEncoder(problem))
}

/// A layout refusal of `call`: the compile layer's own errors as the device's, the rest as what
/// Metal stops.
fn layout_error(call: &'static str, error: ArgumentLayoutError) -> Error {
    match error {
        ArgumentLayoutError::Compile(error) => device_error(error),
        other => problem(call, ArgumentEncoderProblem::Layout(Box::new(other))),
    }
}

/// An argument encoder (`MTLArgumentEncoder`): a layout and the argument buffer it writes into.
/// Cheap to clone (the layout is shared); `Send + Sync`.
#[derive(Clone, Debug)]
pub struct ArgumentEncoder {
    device: Device,
    layout: Arc<ArgumentBufferLayout>,
    /// The argument buffer and the offset of the arguments in it.
    target: Option<(Buffer, u64)>,
    label: Option<String>,
}

impl Device {
    /// `newArgumentEncoderWithArguments:`: an encoder of the layout `arguments` give
    /// (`ArgumentBufferLayout::from_descriptors`). Refused (`Invalid`,
    /// [`ArgumentEncoderProblem::Layout`]): no arguments, a data type no argument has, and a
    /// constant block alignment on a constant that does not start its block, as Metal raises.
    pub fn new_argument_encoder_with_arguments(
        &self,
        arguments: &[ArgumentDescriptor],
    ) -> Result<ArgumentEncoder, Error> {
        const CALL: &str = "new_argument_encoder_with_arguments";
        let layout = ArgumentBufferLayout::from_descriptors(arguments)
            .map_err(|error| layout_error(CALL, error))?;
        Ok(ArgumentEncoder::new(self, Arc::new(layout)))
    }

    /// `newArgumentEncoderWithBufferBinding:`: an encoder of the argument buffer a reflected
    /// buffer binding describes, a library's or a pipeline's, laid out from its struct type as
    /// Metal lays it out (`ArgumentBufferLayout::from_buffer_binding`): a library's binding gives
    /// the function's layout, and a pipeline's, whose arrays of textures, samplers and pointers,
    /// and of structs of them only, reflect stride 0, an encoder that writes every element of such an array at element
    /// 0's place. A plain buffer of a struct gives the encoder Metal makes of it, each constant
    /// member 8 bytes aligned to 8 at its offset (an M3 Max's answer; an M4's is never longer or
    /// more aligned), and a binding with no struct type (a plain buffer of a scalar, vector or
    /// matrix) Metal's encoder of no bytes, alignment 1; Metal's API validation lets each through.
    /// Refused: an array of arrays (`Invalid`, [`ArgumentEncoderProblem::Layout`]), as for
    /// [`Function::new_argument_encoder_with_buffer_index`]; a member whose encoder Metal's was
    /// not measured for (`Unsupported`), such as a plain buffer's pointer member.
    pub fn new_argument_encoder_with_buffer_binding(
        &self,
        binding: &BufferBinding,
    ) -> Result<ArgumentEncoder, Error> {
        const CALL: &str = "new_argument_encoder_with_buffer_binding";
        let layout = ArgumentBufferLayout::from_buffer_binding(binding)
            .map_err(|error| layout_error(CALL, error))?;
        Ok(ArgumentEncoder::new(self, Arc::new(layout)))
    }
}

impl Function {
    /// `newArgumentEncoderWithBufferIndex:`: an encoder of the argument buffer the function takes
    /// at buffer index `index`, as its AIR lays it out
    /// (`library::Function::argument_buffer_layout`, decoded once per function and index).
    /// Refused: no argument buffer at the index, or an array of arrays (`Invalid`,
    /// [`ArgumentEncoderProblem::Layout`]), as Metal stops or raises (a plain buffer at the index
    /// fails Metal's API validation); the decode's errors as a pipeline
    /// creation's (`Compile`, `Unsupported`).
    pub fn new_argument_encoder_with_buffer_index(
        &self,
        index: u32,
    ) -> Result<ArgumentEncoder, Error> {
        const CALL: &str = "new_argument_encoder_with_buffer_index";
        let layout = DerivedMap::or_derive(self.argument_layouts(), index, || {
            self.compile_function()
                .argument_buffer_layout(index, self.constants(), &Budget::standard())
                .map(Arc::new)
                .map_err(|error| layout_error(CALL, error))
        })?;
        Ok(ArgumentEncoder::new(self.device(), layout))
    }
}

/// The offset of the arguments in an argument buffer of `length` bytes:
/// `start_offset + element · encoded_length`, refused where the arguments pass the buffer's end
/// (`RangeOutOfBounds`, as Metal's validation: "offset + encodedLength should be smaller or
/// equal to the buffer length").
pub(super) fn target_offset(
    layout: &ArgumentBufferLayout,
    length: u64,
    start_offset: u64,
    element: u64,
) -> Result<u64, Problem> {
    let size = layout.encoded_length;
    let out_of_bounds = |offset| Problem::RangeOutOfBounds {
        offset,
        size,
        length,
    };
    let offset = element
        .checked_mul(size)
        .and_then(|bytes| bytes.checked_add(start_offset))
        .ok_or(out_of_bounds(u64::MAX))?;
    match offset.checked_add(size) {
        Some(end) if end <= length => Ok(offset),
        _ => Err(out_of_bounds(offset)),
    }
}

/// The place of argument `index` of `layout` for a write of type `ty`, refused where the layout
/// has no such argument or one of another type.
pub(super) fn argument_place(
    layout: &ArgumentBufferLayout,
    index: u32,
    ty: ArgumentType,
) -> Result<ArgumentPlace, ArgumentEncoderProblem> {
    let place = layout
        .argument(index)
        .ok_or(ArgumentEncoderProblem::NoArgument { index })?;
    if place.ty != ty {
        return Err(ArgumentEncoderProblem::ArgumentType {
            index,
            expected: ty,
            found: place.ty,
        });
    }
    Ok(place)
}

/// The offset in the arguments that `constantDataAtIndex:` answers for `index` of `layout`, as
/// plain Metal answers it (an M4): the offset of the argument at `index`, a
/// constant's or one of any other type, and 0 where no argument of the layout has `index`. The
/// debug layer asserts at the last two.
pub(super) fn constant_offset(layout: &ArgumentBufferLayout, index: u32) -> u64 {
    layout.argument(index).map_or(0, |place| place.offset)
}

/// The word `setBuffer:offset:atIndex:` writes for a buffer of Metal address `address` and
/// `length` bytes (or none) at `offset`: the address plus the offset, or 0 for none whatever the
/// offset, as plain Metal writes it (an M4; Metal's validation stops an
/// offset other than 0 there). Refused as Metal's validation refuses: an offset at or past the
/// length.
pub(super) fn buffer_word(buffer: Option<(u64, u64)>, offset: u64) -> Result<u64, Problem> {
    match buffer {
        None => Ok(0),
        Some((_, length)) if offset >= length => Err(Problem::OffsetOutOfRange { offset, length }),
        // An address is below 2^48 (`address.rs`) and the offset below the buffer's length.
        Some((address, _)) => Ok(address.wrapping_add(offset)),
    }
}

impl ArgumentEncoder {
    fn new(device: &Device, layout: Arc<ArgumentBufferLayout>) -> ArgumentEncoder {
        ArgumentEncoder {
            device: device.clone(),
            layout,
            target: None,
            label: None,
        }
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `label`: `None` until one is set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// `setLabel:`.
    pub fn set_label(&mut self, label: &str) {
        self.label = Some(label.to_owned());
    }

    /// `encodedLength`: the bytes the encoder's arguments take in an argument buffer.
    pub fn encoded_length(&self) -> u64 {
        self.layout.encoded_length
    }

    /// `alignment`.
    pub fn alignment(&self) -> u64 {
        self.layout.alignment
    }

    /// The layout the encoder writes (a runtime extension): where each argument goes.
    pub fn layout(&self) -> &ArgumentBufferLayout {
        &self.layout
    }

    /// `setArgumentBuffer:offset:`: write the arguments into `buffer` from `offset` on; `None`
    /// sets no buffer, and a later write is refused until one is set. Refused (`Invalid`): a
    /// buffer of another device, a `Private` one (`StorageMode`), and one whose `offset +
    /// encoded_length` passes its length (`RangeOutOfBounds`), as Metal's validation stops them.
    pub fn set_argument_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
    ) -> Result<(), Error> {
        self.target("set_argument_buffer", buffer, offset, 0)
    }

    /// `setArgumentBuffer:startOffset:arrayElement:`: the arguments of element `array_element`
    /// of an array of argument buffers from `start_offset`, at `start_offset + array_element ·
    /// encoded_length`. Refused as [`set_argument_buffer`](Self::set_argument_buffer).
    pub fn set_argument_buffer_start_offset_array_element(
        &mut self,
        buffer: Option<&Buffer>,
        start_offset: u64,
        array_element: u64,
    ) -> Result<(), Error> {
        self.target(
            "set_argument_buffer_start_offset_array_element",
            buffer,
            start_offset,
            array_element,
        )
    }

    fn target(
        &mut self,
        call: &'static str,
        buffer: Option<&Buffer>,
        start_offset: u64,
        element: u64,
    ) -> Result<(), Error> {
        let Some(buffer) = buffer else {
            self.target = None;
            return Ok(());
        };
        if !buffer.device().same(&self.device) {
            return Err(invalid(call, Problem::ForeignObject));
        }
        if buffer.storage_mode() == StorageMode::Private || buffer.contents().is_none() {
            return Err(invalid(call, Problem::StorageMode));
        }
        let offset = target_offset(&self.layout, buffer.length(), start_offset, element)
            .map_err(|problem| invalid(call, problem))?;
        self.target = Some((buffer.clone(), offset));
        Ok(())
    }

    /// The argument buffer's mapping at the arguments' offset, refused when none is set.
    fn mapped(&self, call: &'static str) -> Result<NonNull<u8>, Error> {
        let (buffer, offset) = self
            .target
            .as_ref()
            .ok_or_else(|| problem(call, ArgumentEncoderProblem::NoArgumentBuffer))?;
        let contents = buffer
            .contents()
            .ok_or_else(|| internal("an argument buffer lost its mapping"))?;
        // `target` checked `offset + encoded_length` against the buffer's length.
        NonNull::new(contents.as_ptr().wrapping_add(*offset as usize))
            .ok_or_else(|| internal("an argument buffer's mapping is null"))
    }

    /// The place of argument `index` for a write of type `ty`, inside the encoded length.
    fn place(
        &self,
        call: &'static str,
        index: u32,
        ty: ArgumentType,
    ) -> Result<ArgumentPlace, Error> {
        let place = argument_place(&self.layout, index, ty).map_err(|p| problem(call, p))?;
        match place.offset.checked_add(place.size) {
            Some(end) if end <= self.layout.encoded_length => Ok(place),
            _ => Err(internal("an argument past its layout's encoded length")),
        }
    }

    /// The offset of argument `index` for a write of type `ty`: the argument's, or, where no
    /// argument of the layout has `index`, the arguments' offset 0, whatever argument lies there,
    /// as plain Metal writes every setter's word, a range's each in turn (an M4; no other byte
    /// changed; Metal's debug layer asserts). The 8 bytes pass the encoded length of a layout of
    /// fewer, as Metal's did in a layout of 4; where they would pass the argument buffer's end,
    /// which holds memory Metal's result depends on, the refusal stands (`NoArgument`).
    fn offset(&self, call: &'static str, index: u32, ty: ArgumentType) -> Result<u64, Error> {
        let inside = |(buffer, at): &(Buffer, u64)| {
            at.checked_add(ARGUMENT_HANDLE_BYTES)
                .is_some_and(|end| end <= buffer.length())
        };
        if self.layout.argument(index).is_none() && self.target.as_ref().is_some_and(inside) {
            return Ok(0);
        }
        Ok(self.place(call, index, ty)?.offset)
    }

    /// Write `words[k]` at argument `first + k`, each of type `ty` (at its [`offset`](Self::offset)):
    /// every index checked before any is written, and the words written in order, so the last of
    /// those at one offset stands.
    fn write(
        &self,
        call: &'static str,
        ty: ArgumentType,
        first: u32,
        words: &[u64],
    ) -> Result<(), Error> {
        let at = self.mapped(call)?;
        let mut offsets = Vec::with_capacity(words.len());
        for k in 0..words.len() {
            let index = u32::try_from(k)
                .ok()
                .and_then(|k| first.checked_add(k))
                .ok_or_else(|| {
                    problem(call, ArgumentEncoderProblem::NoArgument { index: u32::MAX })
                })?;
            offsets.push(self.offset(call, index, ty)?);
        }
        // The bytes written, from the mapping's first: reported once written.
        let mut wrote: Option<Range<u64>> = None;
        for (offset, word) in offsets.into_iter().zip(words) {
            let end = offset + ARGUMENT_HANDLE_BYTES;
            wrote = Some(wrote.map_or(offset..end, |w| w.start.min(offset)..w.end.max(end)));
            let bytes = word.to_le_bytes();
            debug_assert_eq!(bytes.len() as u64, ARGUMENT_HANDLE_BYTES);
            // SAFETY: `mapped` is the argument buffer's mapping at the arguments' offset, whose
            // `encoded_length` bytes lie inside the buffer (`target`), and `offset` keeps
            // `offset + 8` inside the buffer: inside those bytes (`place`), or 0 where the
            // arguments' offset plus 8 is at most the buffer's length. A command that reads or
            // writes them meanwhile races, as on Metal.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    at.as_ptr().add(offset as usize),
                    bytes.len(),
                )
            };
        }
        // A write of host pages a consumer logs is the runtime's own, which its log does not see:
        // reported, so the copies of the pages take it (`dirty_pages.rs`).
        if let (Some((buffer, _)), Some(wrote)) = (&self.target, wrote) {
            let first = at.as_ptr() as u64;
            buffer.wrote_host(first + wrote.start..first + wrote.end);
        }
        Ok(())
    }

    /// Refuse an object of another device.
    fn check_device(&self, call: &'static str, device: &Device) -> Result<(), Error> {
        if device.same(&self.device) {
            Ok(())
        } else {
            Err(invalid(call, Problem::ForeignObject))
        }
    }

    fn buffer_word(
        &self,
        call: &'static str,
        buffer: Option<&Buffer>,
        offset: u64,
    ) -> Result<u64, Error> {
        if let Some(buffer) = buffer {
            self.check_device(call, buffer.device())?;
        }
        buffer_word(buffer.map(|b| (b.gpu_address().0, b.length())), offset)
            .map_err(|p| invalid(call, p))
    }

    /// `setBuffer:offset:atIndex:`: the buffer's Metal address plus `offset`, or 0 for `None`
    /// whatever the offset; at an index no argument has, at the arguments' offset 0, as plain
    /// Metal writes both (the module docs). Refused (`Invalid`): as the module docs list, and an
    /// offset at or past the buffer's length (`OffsetOutOfRange`).
    pub fn set_buffer(
        &self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_buffer";
        let word = self.buffer_word(CALL, buffer, offset)?;
        self.write(CALL, ArgumentType::Buffer, index, &[word])
    }

    /// `setBuffers:offsets:withRange:`: `buffers[k]` at `offsets[k]` at index `first + k`.
    /// Refused as [`set_buffer`](Self::set_buffer) is for each, and when the slices' lengths
    /// differ (`LengthMismatch`); nothing is written then.
    pub fn set_buffers(
        &self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_buffers";
        if buffers.len() != offsets.len() {
            return Err(invalid(CALL, Problem::LengthMismatch));
        }
        let words = buffers
            .iter()
            .zip(offsets)
            .map(|(buffer, offset)| self.buffer_word(CALL, *buffer, *offset))
            .collect::<Result<Vec<_>, _>>()?;
        self.write(CALL, ArgumentType::Buffer, first, &words)
    }

    fn texture_word(&self, call: &'static str, texture: Option<&Texture>) -> Result<u64, Error> {
        match texture {
            None => Ok(0),
            Some(texture) => {
                self.check_device(call, texture.device())?;
                Ok(texture.gpu_resource_id().0)
            }
        }
    }

    /// `setTexture:atIndex:`: the texture's resource ID, or 0 for `None`; at an index no argument
    /// has, at the arguments' offset 0, as plain Metal writes it (the module docs).
    pub fn set_texture(&self, texture: Option<&Texture>, index: u32) -> Result<(), Error> {
        const CALL: &str = "set_texture";
        let word = self.texture_word(CALL, texture)?;
        self.write(CALL, ArgumentType::Texture, index, &[word])
    }

    /// `setTextures:withRange:`: `textures[k]` at index `first + k`.
    pub fn set_textures(&self, textures: &[Option<&Texture>], first: u32) -> Result<(), Error> {
        const CALL: &str = "set_textures";
        let words = textures
            .iter()
            .map(|texture| self.texture_word(CALL, *texture))
            .collect::<Result<Vec<_>, _>>()?;
        self.write(CALL, ArgumentType::Texture, first, &words)
    }

    fn sampler_word(
        &self,
        call: &'static str,
        sampler: Option<&SamplerState>,
    ) -> Result<u64, Error> {
        match sampler {
            None => Ok(0),
            Some(sampler) => {
                self.check_device(call, sampler.device())?;
                Ok(sampler.gpu_resource_id().0)
            }
        }
    }

    /// `setSamplerState:atIndex:`: the sampler state's resource ID, or 0 for `None`. A state made
    /// without `support_argument_buffers` is written too, and a kernel samples through it, as
    /// plain Metal does (Metal's API validation stops it).
    pub fn set_sampler_state(
        &self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_sampler_state";
        let word = self.sampler_word(CALL, sampler)?;
        self.write(CALL, ArgumentType::Sampler, index, &[word])
    }

    /// `setSamplerStates:withRange:`: `samplers[k]` at index `first + k`.
    pub fn set_sampler_states(
        &self,
        samplers: &[Option<&SamplerState>],
        first: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_sampler_states";
        let words = samplers
            .iter()
            .map(|sampler| self.sampler_word(CALL, *sampler))
            .collect::<Result<Vec<_>, _>>()?;
        self.write(CALL, ArgumentType::Sampler, first, &words)
    }

    fn compute_word(
        &self,
        call: &'static str,
        state: Option<&ComputePipelineState>,
    ) -> Result<u64, Error> {
        match state {
            None => Ok(0),
            Some(state) => {
                self.check_device(call, state.device())?;
                Ok(state.gpu_resource_id().0)
            }
        }
    }

    /// `setComputePipelineState:atIndex:`: the state's resource ID, or 0 for `None`; a state made
    /// without `support_indirect_command_buffers` writes its ID, 0, as plain Metal writes it (the
    /// module docs).
    pub fn set_compute_pipeline_state(
        &self,
        state: Option<&ComputePipelineState>,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_compute_pipeline_state";
        let word = self.compute_word(CALL, state)?;
        self.write(CALL, ArgumentType::ComputePipelineState, index, &[word])
    }

    /// `setComputePipelineStates:withRange:`: `states[k]` at index `first + k`.
    pub fn set_compute_pipeline_states(
        &self,
        states: &[Option<&ComputePipelineState>],
        first: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_compute_pipeline_states";
        let words = states
            .iter()
            .map(|state| self.compute_word(CALL, *state))
            .collect::<Result<Vec<_>, _>>()?;
        self.write(CALL, ArgumentType::ComputePipelineState, first, &words)
    }

    fn render_word(
        &self,
        call: &'static str,
        state: Option<&RenderPipelineState>,
    ) -> Result<u64, Error> {
        match state {
            None => Ok(0),
            Some(state) => {
                self.check_device(call, state.device())?;
                Ok(state.gpu_resource_id().0)
            }
        }
    }

    /// `setRenderPipelineState:atIndex:`: the state's resource ID, or 0 for `None`; a state made
    /// without `support_indirect_command_buffers`, a tile pipeline's among them, writes its ID,
    /// 0, as plain Metal writes it (the module docs).
    pub fn set_render_pipeline_state(
        &self,
        state: Option<&RenderPipelineState>,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_render_pipeline_state";
        let word = self.render_word(CALL, state)?;
        self.write(CALL, ArgumentType::RenderPipelineState, index, &[word])
    }

    /// `setRenderPipelineStates:withRange:`: `states[k]` at index `first + k`.
    pub fn set_render_pipeline_states(
        &self,
        states: &[Option<&RenderPipelineState>],
        first: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_render_pipeline_states";
        let words = states
            .iter()
            .map(|state| self.render_word(CALL, *state))
            .collect::<Result<Vec<_>, _>>()?;
        self.write(CALL, ArgumentType::RenderPipelineState, first, &words)
    }

    fn command_buffer_word(
        &self,
        call: &'static str,
        buffer: Option<&IndirectCommandBuffer>,
    ) -> Result<u64, Error> {
        match buffer {
            None => Ok(0),
            Some(buffer) => {
                self.check_device(call, buffer.device())?;
                Ok(buffer.gpu_resource_id().0)
            }
        }
    }

    /// `setIndirectCommandBuffer:atIndex:`: the buffer's resource ID (`gpu_resource_id`, a small
    /// integer of the buffers' own space, as Metal writes it), or 0 for `None`. A buffer of no
    /// command types encodes its own ID as any other buffer's, as plain Metal encodes it: its
    /// command types decide nothing about the word (the module docs).
    pub fn set_indirect_command_buffer(
        &self,
        buffer: Option<&IndirectCommandBuffer>,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_indirect_command_buffer";
        let word = self.command_buffer_word(CALL, buffer)?;
        self.write(CALL, ArgumentType::IndirectCommandBuffer, index, &[word])
    }

    /// `setIndirectCommandBuffers:withRange:`: `buffers[k]` at index `first + k`.
    pub fn set_indirect_command_buffers(
        &self,
        buffers: &[Option<&IndirectCommandBuffer>],
        first: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_indirect_command_buffers";
        let words = buffers
            .iter()
            .map(|buffer| self.command_buffer_word(CALL, *buffer))
            .collect::<Result<Vec<_>, _>>()?;
        self.write(CALL, ArgumentType::IndirectCommandBuffer, first, &words)
    }

    fn table_word(
        &self,
        call: &'static str,
        table: Option<&VisibleFunctionTable>,
    ) -> Result<u64, Error> {
        match table {
            None => Ok(0),
            Some(table) => {
                self.check_device(call, table.device())?;
                Ok(table.gpu_resource_id().0)
            }
        }
    }

    /// `setVisibleFunctionTable:atIndex:`: the table's resource ID, or 0 for `None` (resolved for a
    /// dispatch of the state that made the table).
    pub fn set_visible_function_table(
        &self,
        table: Option<&VisibleFunctionTable>,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_visible_function_table";
        let word = self.table_word(CALL, table)?;
        self.write(CALL, ArgumentType::VisibleFunctionTable, index, &[word])
    }

    /// `setVisibleFunctionTables:withRange:`: `tables[k]` at index `first + k`.
    pub fn set_visible_function_tables(
        &self,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_visible_function_tables";
        let words = tables
            .iter()
            .map(|table| self.table_word(CALL, *table))
            .collect::<Result<Vec<_>, _>>()?;
        self.write(CALL, ArgumentType::VisibleFunctionTable, first, &words)
    }

    /// `constantDataAtIndex:`: the CPU address of the constant at `index` in the argument
    /// buffer, for the program to write its bytes. As plain Metal answers (the module docs): at an
    /// argument of another type, that argument's address, and at an index no argument has, the
    /// arguments' start (offset 0); the debug layer asserts at both. Refused (`Invalid`): no
    /// argument buffer set.
    pub fn constant_data(&self, index: u32) -> Result<NonNull<u8>, Error> {
        const CALL: &str = "constant_data";
        let at = self.mapped(CALL)?;
        let offset = constant_offset(&self.layout, index);
        if offset > self.layout.encoded_length {
            return Err(internal("a constant past its layout's encoded length"));
        }
        // The constant lies inside the encoded length (the layout's), inside the mapping
        // (`target`); offset 0 is the arguments' start.
        NonNull::new(at.as_ptr().wrapping_add(offset as usize))
            .ok_or_else(|| internal("an argument buffer's mapping is null"))
    }

    /// `newArgumentEncoderForBufferAtIndex:`: an encoder of the argument buffer the buffer
    /// argument at `index` points to, with no argument buffer set.
    ///
    /// Every index that names no nested argument buffer answers `Nil`
    /// ([`NoNestedArgumentBuffer`](ArgumentEncoderProblem::NoNestedArgumentBuffer)), because plain
    /// Metal answers **nil** for each of them rather than raising: measured on an Apple M3 Max
    /// under macOS 27.0.1 over one layout's own indices — a plain `device float *`, a pointer to a
    /// struct of constants, an inline struct of a texture and a sampler, a texture, a sampler, an
    /// array of textures, an array of samplers, an array of structs, an indirect command buffer, a
    /// compute pipeline state, and an index past the layout — and an encoder only for a pointer
    /// whose pointee holds a resource (`metal2vulkan-conformance`'s
    /// `probes/argument_descriptor_types.m`, held by its `argument-encoder-cross-nested` case).
    /// So the index's **reason** for having no nested buffer changes nothing a program sees, and
    /// an `Invalid` here would refuse what plain Metal answers.
    pub fn new_argument_encoder_for_buffer_at_index(
        &self,
        index: u32,
    ) -> Result<ArgumentEncoder, Error> {
        const CALL: &str = "new_argument_encoder_for_buffer_at_index";
        let nested = self
            .layout
            .argument(index)
            .filter(|place| place.ty == ArgumentType::Buffer)
            .and_then(|place| place.nested)
            .ok_or_else(|| {
                let problem = ArgumentEncoderProblem::NoNestedArgumentBuffer { index };
                nil(CALL, Problem::ArgumentEncoder(problem))
            })?;
        Ok(ArgumentEncoder::new(&self.device, nested))
    }
}
