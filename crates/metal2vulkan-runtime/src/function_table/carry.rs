//! What a device's recovery carries of the program's visible function tables
//! (`device/carry.rs`): each the same object on the new `VkDevice`, after the pipeline state that
//! made it, as Metal keeps its objects across its GPU's recovery.
//!
//! - **The words.** The table's buffer of words is a buffer of the runtime's own whose `contents`
//!   pointer no program holds, carried as such (`Buffer::carry_runtime`): in the runtime's own
//!   pages, the same pages there, the words as they are; in pooled host-visible memory (a
//!   write-combined buffer lies in device-local memory where the device maps such), new memory
//!   there, written again from the host's copy of the words (`TableShared::writes`), so its
//!   entries call the functions they called. The CPU's writes of words wait for the carry.
//! - **What stays.** The object, its `gpuResourceID` (the device's line's, `device/line.rs`'s
//!   `Carried`), its entries and its label; it is listed among the new generation's tables, which
//!   an argument buffer of the state's commands names (`BindState::function_tables`).
//! - **Left** the lost generation's, refused where it meets the next
//!   (`RuntimeGap::MadeBeforeRecovery`): a table whose state stays there (its making failed, or
//!   the generations compile otherwise, `pipeline/carry.rs`), and one whose words' carry fails.

use super::{TableShared, VisibleFunctionTable};
use crate::device::carry::{Contents, Left};
use crate::device::lock;
use crate::Device;
use std::sync::{Arc, Weak};

/// A visible function table, held weakly by its generation's registry for a recovery to carry
/// (`Carry::function_table`).
pub(crate) struct WeakFunctionTable(Weak<TableShared>);

impl WeakFunctionTable {
    /// The table, while the program holds it.
    pub(crate) fn upgrade(&self) -> Option<VisibleFunctionTable> {
        self.0.upgrade().map(VisibleFunctionTable)
    }

    /// Whether the table lives.
    pub(crate) fn lives(&self) -> bool {
        self.0.strong_count() > 0
    }
}

impl VisibleFunctionTable {
    /// The table, held weakly.
    pub(crate) fn downgrade(&self) -> WeakFunctionTable {
        WeakFunctionTable(Arc::downgrade(&self.0))
    }

    /// Carry the table, made on a generation of its device that a recovery leaves, to `to`, the
    /// generation that recovery made, after the state that made it (the module docs): what
    /// became of its words' memory, the words as they were either way. `Kept` where it was
    /// carried there already.
    pub(crate) fn carry(&self, to: &Device) -> Result<Contents, Left> {
        let shared = &self.0;
        if shared.words.generation().same(to) {
            return Ok(Contents::Kept);
        }
        if !shared.state.generation().same(to) {
            return Err(Left::Failed);
        }
        let written = lock(&shared.writes);
        let contents = shared.words.carry_runtime(to)?;
        if let Contents::Lost(_) = contents {
            let Some(mapped) = shared.words.mapped_contents() else {
                return Err(Left::Failed);
            };
            for (at, word) in written.iter().enumerate() {
                // SAFETY: the words' new memory is host-visible and mapped, `byte_len(count)`
                // bytes, four for each word of the host copy, its start aligned past 4; no
                // command reads them before the next generation serves, and the CPU's writes
                // wait (`writes`, held).
                unsafe {
                    mapped
                        .as_ptr()
                        .add(at * 4)
                        .cast::<u32>()
                        .write_volatile(word.to_le());
                }
            }
        }
        drop(written);
        lock(&to.shared().function_tables).insert(self.gpu_resource_id().0, Arc::downgrade(shared));
        // Kept there for a later recovery to carry on.
        if to.shared().carry.function_table(self) {
            to.carry_late(|serving| self.carry(serving));
        }
        Ok(contents)
    }
}
