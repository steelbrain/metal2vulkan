//! Visible-function tables of compute pipeline states (`MTLVisibleFunctionTable`).
//!
//! A kernel reads a `visible_function_table` argument as the words `metal2vulkan` lays out
//! ([`function_table_header`], then one word per entry at [`function_table_entry_offset`]), at the
//! address its buffer index's slot pushes. A table is those words in a Shared block the runtime
//! takes, written through the block's mapping, so a dispatch reads the entries as they stand when
//! it runs: an entry set after the dispatch is encoded and before its command buffer commits is the
//! one it calls. An entry holds the word of a function handle a state answered
//! ([`ComputePipelineState::function_handle`]), and a call through it calls the function that has
//! that word among the linked functions of the state the dispatch runs (the translator's switch
//! over them); a null entry, or a word none of them has, calls nothing and yields zero. A table or
//! a handle another state answered calls whichever function has its word among the dispatch's
//! state's, or nothing.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use ash::vk;
use metal2vulkan::bind::{function_table_entry_offset, function_table_header};
use metal2vulkan::pipeline::{FunctionHandle, LinkedFunction};

use crate::buffer::{BufferInner, BufferObject, MAX_BUFFER_LENGTH};
use crate::counters::Site;
use crate::error::{Error, Refusal};
use crate::memory::{Block, Mapping};
use crate::pipeline::ComputePipelineState;
use crate::MemoryClass;

/// `MTLVisibleFunctionTableDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisibleFunctionTableDescriptor {
    /// `functionCount`: the table's entries, 0 by default.
    pub function_count: u64,
}

impl VisibleFunctionTableDescriptor {
    /// A descriptor of `function_count` entries.
    pub fn new(function_count: u64) -> VisibleFunctionTableDescriptor {
        VisibleFunctionTableDescriptor { function_count }
    }
}

/// `MTLVisibleFunctionTable` of a compute pipeline state
/// ([`ComputePipelineState::new_visible_function_table`]), which a compute encoder binds at a
/// buffer index
/// ([`ComputeCommandEncoder::set_visible_function_table`](crate::ComputeCommandEncoder::set_visible_function_table)).
/// A cheap handle: clones share one table, and a command buffer that binds it keeps its words
/// until it retires. Its words are a buffer of the runtime's own, counted among the device's
/// buffers ([`Gauges::buffers`](crate::Gauges::buffers)), in a Shared block of its own, 64 KiB at
/// the least (the runtime's smallest block).
#[derive(Clone)]
pub struct VisibleFunctionTable {
    pub(crate) words: Arc<BufferInner>,
    mapping: Mapping,
    count: u32,
}

impl ComputePipelineState {
    /// `functionHandleWithFunction:`: the handle of `function` (the library function with the
    /// constant values and name it was made with) among the kernel's linked functions, numbered
    /// as `LinkedFunctions::function_handle` numbers them; `None`, Metal's nil, for a function the
    /// kernel does not link in `functions` or `binary_functions` (a private function, the kernel
    /// itself, any other). A state whose kernel the consumer compiled (`StageSource::Compiled`)
    /// holds no linked functions and answers `None` for every function.
    pub fn function_handle(&self, function: &LinkedFunction) -> Option<FunctionHandle> {
        self.inner.compiled.linked.function_handle(function)
    }

    /// `newVisibleFunctionTableWithDescriptor:`: a table of `descriptor.function_count` null
    /// entries, whatever the kernel links.
    ///
    /// Err: [`Refusal::BufferTooLong`] for a table whose words pass
    /// [`MAX_BUFFER_LENGTH`]; [`Error::Quiesced`]; what taking the
    /// words' block returns ([`Device::take_block`](crate::Device::take_block)).
    pub fn new_visible_function_table(
        &self,
        descriptor: &VisibleFunctionTableDescriptor,
    ) -> Result<VisibleFunctionTable, Error> {
        let core = self.inner.core();
        if core.closed() {
            return Err(Error::Quiesced);
        }
        let entries = descriptor.function_count.saturating_mul(4);
        let length = function_table_entry_offset(0).saturating_add(entries);
        if length > MAX_BUFFER_LENGTH {
            return Err(Error::Refused(Refusal::BufferTooLong {
                length,
                max: MAX_BUFFER_LENGTH,
            }));
        }
        // Within `MAX_BUFFER_LENGTH`, so the count is a `u32`.
        let count = descriptor.function_count as u32;
        let memory = core.take_memory(MemoryClass::Shared, length, Site::SharedTake)?;
        let block = Arc::new(Block::new(memory, Arc::clone(core)));
        // A Shared block is mapped for its life; an unmapped one is a failed map.
        let Some(mapping) = block.memory.mapping else {
            return Err(Error::Vulkan {
                call: "vkMapMemory",
                result: vk::Result::ERROR_MEMORY_MAP_FAILED,
            });
        };
        // SAFETY: the block is the table's alone and holds `length` bytes from its mapping.
        unsafe { mapping.get().as_ptr().write_bytes(0, length as usize) };
        let words = BufferInner::new(core, block, 0, length, Some(mapping), None, None);
        let table = VisibleFunctionTable {
            words,
            mapping,
            count,
        };
        for (at, word) in function_table_header(count).into_iter().enumerate() {
            table.write(at as u64 * 4, word);
        }
        Ok(table)
    }
}

impl VisibleFunctionTable {
    /// `setFunction:atIndex:`: entry `index` calls `function`, or is null (`None`).
    ///
    /// Err: [`Refusal::FunctionTableIndex`] for an index at or past the table's size.
    pub fn set_function(&self, function: Option<FunctionHandle>, index: u32) -> Result<(), Error> {
        self.set_functions(&[function], index)
    }

    /// `setFunctions:withRange:`: entry `first + i` calls `functions[i]`, or is null.
    ///
    /// Err: [`Refusal::FunctionTableIndex`] for a range that passes the table's size, which sets
    /// none of them.
    pub fn set_functions(
        &self,
        functions: &[Option<FunctionHandle>],
        first: u32,
    ) -> Result<(), Error> {
        let end = u64::from(first) + functions.len() as u64;
        if end > u64::from(self.count) {
            return Err(Error::Refused(Refusal::FunctionTableIndex {
                index: end.saturating_sub(1),
                count: self.count,
            }));
        }
        for (index, function) in (first..).zip(functions) {
            let word = function.map_or(FunctionHandle::NULL_WORD, FunctionHandle::word);
            self.write(function_table_entry_offset(index), word);
        }
        Ok(())
    }

    /// `gpuResourceID`: the device address of the table's words (Metal states no value). A kernel
    /// that reads a table through an argument buffer is not built: its state is refused
    /// ([`Gap::KernelInput`](crate::Gap::KernelInput)).
    pub fn gpu_resource_id(&self) -> u64 {
        self.words.gpu_address()
    }

    /// Writes `word` at byte `offset` of the table's words.
    fn write(&self, offset: u64, word: u32) {
        // SAFETY: `offset` is a multiple of 4 inside the table's words, which lie at the start of
        // its block's mapping, aligned past 4, for the table's life; every write to them is atomic,
        // and a dispatch reading a word as it is written reads the old or the new one.
        let at = unsafe { mapping_word(self.mapping, offset) };
        at.store(word.to_le(), Ordering::Relaxed);
    }
}

/// The word at byte `offset` of `mapping`.
///
/// # Safety
///
/// `offset` is a multiple of 4 inside the mapped memory, which lives past the reference's use and
/// is written only atomically meanwhile.
unsafe fn mapping_word<'a>(mapping: Mapping, offset: u64) -> &'a AtomicU32 {
    // SAFETY: as the caller states.
    unsafe { AtomicU32::from_ptr(mapping.at(offset).as_ptr().cast::<u32>()) }
}
