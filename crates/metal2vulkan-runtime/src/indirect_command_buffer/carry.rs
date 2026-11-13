//! What a device's recovery carries of the program's indirect command buffers
//! (`device/carry.rs`): each the same object on the new `VkDevice`, as Metal keeps its objects
//! across its GPU's recovery.
//!
//! - **The records.** Its storage, and the buffer of the slots past its records ([`Overflow`]),
//!   are buffers of the runtime's own whose `contents` pointer no program holds, carried as such
//!   (`Buffer::carry_runtime`): in the runtime's own pages, where every such buffer lies on a
//!   device that imports host pointers (`fresh_pages.rs`), the same pages there, the records as
//!   the CPU and the GPU left them, as Metal keeps them across its GPU's restart; in pooled
//!   host-visible memory, on a device that imports none or where the import failed, new memory
//!   there, the records lost with the `VkDevice`, whose memory Vulkan makes undefined (vkspec
//!   5.2.3, "Lost Device"), written again as a new buffer's: its header, every command reset, its
//!   slots past them zero. The CPU's writes of the records under `pipelines` wait for the carry;
//!   a setter's write without it (`Batch`: one that adds nothing to what the records may hold)
//!   does not: in the runtime's own pages it writes the pages the carry keeps, and in pooled
//!   memory one meanwhile races the carry. A host read outside a command buffer's work holds the
//!   storage's parts it reads.
//! - **What stays.** The object, its records ID, address, `gpuResourceID`, `uniqueIdentifier` and
//!   label, which are the device's line's rather than a generation's (`device/line.rs`'s
//!   `Carried`); the threadgroup lengths past the last command's, which the host keeps; what its
//!   records may hold (`written_commands.rs`) and the CPU's writes counted.
//! - **What is made again.** The program each state the CPU set on a command runs: a live state's
//!   its program there, which the state's carry made before (`pipeline/carry.rs`); a released
//!   state's made again there from its compiled pipeline, as is each program the buffer keeps of
//!   the released states its GPU-written records may name (`released_programs.rs`). One whose
//!   making there fails stays the lost generation's, refused where bound
//!   (`RuntimeGap::MadeBeforeRecovery`). Its kept recordings of chunks (`kept_commands.rs`), the
//!   lost generation's command buffers, are let go of; its executions there record them again.
//!   Where the GPU may write its records, the new generation's keepers list it
//!   (`Keepers::add`), so a state released there is kept for it too.
//! - **Left** the lost generation's, refused where it meets the next: a buffer whose storage's
//!   carry fails (the making of its memory there).

use super::{CommandState, IcbShared, IndirectCommandBuffer};
use crate::device::carry::{Contents, Left};
use crate::device::lock;
use crate::Device;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

/// An indirect command buffer, held weakly by its generation's registry for a recovery to carry
/// (`Carry::indirect_command_buffer`).
pub(crate) struct WeakIndirectCommandBuffer(Weak<IcbShared>);

impl WeakIndirectCommandBuffer {
    /// The buffer, while the program holds it.
    pub(crate) fn upgrade(&self) -> Option<IndirectCommandBuffer> {
        self.0.upgrade().map(IndirectCommandBuffer)
    }

    /// Whether the buffer lives.
    pub(crate) fn lives(&self) -> bool {
        self.0.strong_count() > 0
    }
}

impl IndirectCommandBuffer {
    /// The buffer, held weakly.
    pub(crate) fn downgrade(&self) -> WeakIndirectCommandBuffer {
        WeakIndirectCommandBuffer(Arc::downgrade(&self.0))
    }

    /// Carry the buffer, made on a generation of its device that a recovery leaves, to `to`, the
    /// generation that recovery made, after its pipeline states (the module docs): what became of
    /// its records. `Kept` where it was carried there already.
    pub(crate) fn carry(&self, to: &Device) -> Result<Contents, Left> {
        let shared = &self.0;
        if shared.storage.generation().same(to) {
            return Ok(Contents::Kept);
        }
        let mut commands = lock(&shared.pipelines);
        // Each command's next set of its state checked again against the generation it meets,
        // under the lock, whether the carry succeeds or not.
        commands.states.forget_keys();
        // The slots past the records first: a buffer whose storage stays the lost generation's
        // stays whole there, its device that of its storage (`generation`).
        if let Some(overflow) = &shared.overflow {
            overflow.storage.carry_runtime(to)?;
        }
        let contents = shared.storage.carry_runtime(to)?;
        if let Contents::Lost(_) = contents {
            // Every command reset, as a new buffer's: the states the CPU set on them go.
            self.write_header().map_err(|_| Left::Failed)?;
            commands.states.clear_all();
        }
        for state in commands.states.entries_mut() {
            *state = state.carried(to);
        }
        commands.kept.clear();
        commands.step_kept.clear();
        drop(commands);
        lock(&shared.kept).carry(to);
        if shared.keeping.load(Ordering::Acquire) {
            to.shared().keepers.add(&shared.kept);
        }
        // Kept there for a later recovery to carry on.
        if to.shared().carry.indirect_command_buffer(self) {
            to.carry_late(|serving| self.carry(serving));
        }
        Ok(contents)
    }
}

impl CommandState {
    /// The state on `to`, the generation a recovery made: a live state's program there, a
    /// released state's made again there, or the lost one where its making there fails (the
    /// module docs).
    fn carried(&self, to: &Device) -> CommandState {
        match self {
            CommandState::Compute(named, program, stamp, can) => {
                let program = match named.upgrade() {
                    Some(state) => state.program(),
                    None => program.made_again(to).unwrap_or_else(|_| program.clone()),
                };
                CommandState::Compute(named.clone(), program, *stamp, *can)
            }
            CommandState::Render(named, program, stamp, can) => {
                let program = match named.upgrade().and_then(|state| state.program()) {
                    Some(program) => program,
                    None => program.made_again(to).unwrap_or_else(|_| program.clone()),
                };
                CommandState::Render(named.clone(), program, *stamp, *can)
            }
        }
    }
}
