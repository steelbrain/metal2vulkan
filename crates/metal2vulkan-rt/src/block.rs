//! The consumer's handle of a block ([`Block`]): one claim, which its clones share and whose last
//! handle's drop gives the block back.

use crate::device::{lock, Core};
use crate::error::Error;
use crate::memory::{self, BlockInfo};
use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, Mutex, PoisonError, Weak};

/// No Metal counterpart: rt's own handle of a block of memory the consumer places buffers in, as
/// a placement `MTLHeap`'s buffers sit in its memory
/// ([`Device::take_block`](crate::Device::take_block)), or as a no-copy buffer's bytes are the
/// memory it was made over (`Device::import_dma_buf` on Unix,
/// [`Device::import_host_memory`](crate::Device::import_host_memory)), or as another API reaches
/// it by a handle ([`Device::new_exported_block`](crate::Device::new_exported_block)). It derefs
/// to the block's [`BlockInfo`], plain data.
///
/// Clones share one claim. Dropping the last handle gives the block back, as
/// [`Device::release_block`](crate::Device::release_block) does: the block's memory lives until
/// the last handle and the last buffer, command buffer or upload using it are gone, and is then
/// a free range of its allocation; an imported block's memory is freed, and then its release runs;
/// an exported block's memory is freed.
/// While a handle lives, the block's number names it to
/// [`Device::new_buffer`](crate::Device::new_buffer) and the other calls that take one; once the
/// last is gone, they answer [`Error::NoSuchBlock`](crate::Error::NoSuchBlock). The device's
/// teardown frees every allocation, blocks still claimed included, each after the consumer's
/// [`BlockMappings::unmap`](crate::BlockMappings::unmap) of it returned; a handle dropped after
/// that does nothing.
#[derive(Clone)]
pub struct Block {
    claim: Arc<Claim>,
}

/// One claim of a block, which every clone of its handle shares.
pub(crate) struct Claim {
    info: BlockInfo,
    /// The device, which the block holds while the slot does.
    core: Weak<Core>,
    /// The block, until the last handle drops or the device's teardown takes it.
    block: Mutex<Option<Arc<memory::Block>>>,
}

impl Block {
    /// A handle of a new claim of `block`, whose info is `info`, on `core`'s device; the device
    /// keeps the claim by the block's number ([`Core::claims`]).
    pub(crate) fn claim(core: &Arc<Core>, info: BlockInfo, block: memory::Block) -> Block {
        let claim = Arc::new(Claim {
            info,
            core: Arc::downgrade(core),
            block: Mutex::new(Some(Arc::new(block))),
        });
        lock(&core.claims).insert(info.id, Arc::downgrade(&claim));
        Block { claim }
    }

    /// Whether the block is of `core`'s device.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Weak::as_ptr(&self.claim.core) == Arc::as_ptr(core)
    }

    /// Its device and the block, while the claim holds it.
    ///
    /// Err: [`Error::Quiesced`] once the device's teardown took the block.
    pub(crate) fn held(&self) -> Result<(Arc<Core>, Arc<memory::Block>), Error> {
        let core = self.claim.core.upgrade().ok_or(Error::Quiesced)?;
        let block = self.claim.block().ok_or(Error::Quiesced)?;
        Ok((core, block))
    }
}

impl Claim {
    /// The block, while the claim holds it.
    pub(crate) fn block(&self) -> Option<Arc<memory::Block>> {
        lock(&self.block).clone()
    }

    /// At the device's teardown: the block, which the claim no longer holds.
    pub(crate) fn take(&self) -> Option<Arc<memory::Block>> {
        lock(&self.block).take()
    }
}

impl Deref for Block {
    type Target = BlockInfo;

    fn deref(&self) -> &BlockInfo {
        &self.claim.info
    }
}

impl fmt::Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Block").field(&self.claim.info).finish()
    }
}

impl Drop for Claim {
    /// The last handle is gone: the device forgets the block's number and gives the block back,
    /// as [`Device::release_block`](crate::Device::release_block) does, unless its teardown took
    /// the block first.
    fn drop(&mut self) {
        let slot = self.block.get_mut().unwrap_or_else(PoisonError::into_inner);
        let (Some(block), Some(core)) = (slot.take(), self.core.upgrade()) else {
            return;
        };
        lock(&core.claims).remove(&self.info.id);
        core.work.release_block(&core, &block);
    }
}
