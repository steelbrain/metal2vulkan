//! A shared dma-buf imported ahead and kept ([`Device::import_shared_dma_buf`], a runtime
//! extension: Metal has none). A file a caller's source shares among several buffers' and heaps'
//! chunks (`DmaBufChunk`) is imported once per device generation and freed with the last buffer
//! or heap bound in it (`dma_buf.rs`, **Shared files**), so a guest whose buffers over one file
//! come and go one at a time imports it again for each, and RADV maps it again at each import,
//! the next submission waiting for the page tables (`WorkCounts::dma_buf_imports`). A handle
//! ([`SharedDmaBufImport`]) imports the file on its caller's thread and keeps it in the device's
//! map ([`DmaBufFiles::keep`](super::DmaBufFiles::keep)): while it lives, the last buffer to let go leaves the import, and
//! each later piece joins it (`WorkCounts::dma_buf_pieces_joined`).
//!
//! - **One map, one hold.** A keep is a count in the file's entry of the generation's map, which
//!   holds the import while the count is positive and the file imported: the import live as the
//!   handle is made, or the one entered next. So a kept file has one import whatever the
//!   handles, buffers and heaps, and the bound is one kept import per file a live handle names.
//! - **Recovery re-imports lazily.** A handle is kept by the generation it keeps its file on,
//!   weakly (`device/carry.rs`), and the recovery carries it before the buffers: it keeps the file
//!   on the new generation, which imports it at the first buffer that needs it, the entry
//!   keeping that import, and lets go of its keep on the lost one, whose import goes with the
//!   last object left there. Nothing is imported on the recovering thread.
//! - **Nothing kept** where the device imports no dma-buf for a buffer: the handle holds nothing,
//!   and says so (`Debug`).

use super::{DmaBufFile, Keep, Unkept};
use crate::buffer::Modes;
use crate::buffer_descriptor::{host_class, no_copy_options};
use crate::device::lock;
use crate::error::Error;
use crate::exit_guard;
use crate::host_memory::{not_importable, Importer};
use crate::memory::MemoryClass;
use crate::release;
use crate::values::ResourceOptions;
use crate::Device;
use ash::vk;
use std::convert::Infallible;
use std::os::fd::OwnedFd;
use std::sync::{Arc, Mutex};

/// A shared dma-buf's import kept on its device (a runtime extension: Metal has none), made by
/// [`Device::import_shared_dma_buf`]: while it lives, the device's one import of the file stays
/// imported whether or not a buffer or heap is bound in it, so each no-copy buffer, heap or rebind
/// over a piece of the file joins it rather than importing the file again. Dropped (on any
/// thread), it lets go of its keep, and the import goes with the last buffer or heap bound in it,
/// or at once where none is. `Debug` says whether it holds an import now: never where its device
/// keeps nothing (it imports no dma-buf for a buffer), and after the device's recovery not until
/// a buffer on the new device imports the file.
#[must_use = "the import is kept only while the handle lives"]
pub struct SharedDmaBufImport(Arc<KeptFile>);

impl std::fmt::Debug for SharedDmaBufImport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedDmaBufImport")
            .field("imported", &self.0.imported())
            .finish()
    }
}

/// A handle's keep of its file ([`SharedDmaBufImport`]), shared with the recovery's registry
/// (`device/carry.rs`), which holds it weakly.
pub(crate) struct KeptFile {
    file: Arc<OwnedFd>,
    /// The generation that keeps the file; `None` where the device keeps nothing.
    on: Mutex<Option<KeptOn>>,
}

/// A keep of `file` on one generation of its device, let go of as it drops.
struct KeptOn {
    device: Device,
    file: Arc<OwnedFd>,
}

impl Drop for KeptOn {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard; while the process exits the import
        // stays to its end.
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        self.device.unkeep_file(&self.file);
        drop(busy);
    }
}

impl KeptFile {
    /// Whether the generation that keeps the file holds an import of it for the keep now.
    fn imported(&self) -> bool {
        let on = lock(&self.on);
        (on.as_ref()).is_some_and(|kept| kept.device.shared().dma_buf_files.kept(&self.file))
    }

    /// Carry the keep to `to`, its device's next generation, by the recovery that made it or by
    /// its maker (`device/carry.rs`): `to` keeps the file from now on, unimported until a buffer
    /// there imports it, and the keep on the generation it leaves is let go of, its import going
    /// with the last object left there.
    pub(crate) fn carry(self: &Arc<Self>, to: &Device) {
        let left = {
            let mut on = lock(&self.on);
            match on.as_ref() {
                None => return,
                // Carried already, by its maker.
                Some(kept) if kept.device.same(to) => return,
                Some(_) => {}
            }
            to.keep_here(&self.file);
            on.replace(KeptOn {
                device: to.clone(),
                file: Arc::clone(&self.file),
            })
        };
        // Let go of with no lock held.
        drop(left);
        // Kept for `to`'s recovery in turn, or carried on where that took `to`'s keeps already
        // (a late carry, `Device::carry_late`).
        if to.shared().carry.kept_file(self) {
            to.carry_late(|serving| {
                self.carry(serving);
                Ok::<(), Infallible>(())
            });
        }
    }
}

impl Device {
    /// A runtime extension (Metal has none): import `file`, a dma-buf a caller's source shares
    /// among the chunks of several no-copy buffers and heaps (`DmaBufChunk`, the same `Arc`), now,
    /// on the calling thread, and keep the import while the returned handle lives. Without it the
    /// device imports such a file at the first buffer over a piece of it and frees the import with
    /// the last, so buffers that come and go one at a time import it again each time; with it,
    /// each one joins the kept import (`WorkCounts::dma_buf_pieces_joined`) and none imports the
    /// file (`WorkCounts::dma_buf_imports`). Kept imports are counted:
    /// `WorkCounts::dma_buf_shared_imports_kept`.
    ///
    /// The import is the one the device's first buffer over a piece of the file would make: the
    /// whole dma-buf, dedicated to nothing, of the best memory type of `options`' class among the
    /// types the file and a no-copy buffer take: `Shared` storage, whatever the CPU cache mode,
    /// or `Managed`, as a no-copy buffer of those options is. Where the device holds an import of
    /// the file already, the handle keeps that one. The import is made with no lock held; one a
    /// buffer makes meanwhile on another thread is the one kept, this one freed. Each kept file
    /// is one import on the device, whatever the handles of it: at most one per file a live handle
    /// names.
    ///
    /// The handle imports the file whether or not the device's no-copy buffers over those pages
    /// take the dma-buf route: a device that imports the pages' pointer never calls their source.
    /// There a kept import serves nothing, and RADV lists it in every submission (one listed
    /// dma-buf more, as each import is): make one only for files whose buffers take dma-bufs. A
    /// device that imports no dma-buf for a buffer (it enables no dma-buf route) keeps nothing:
    /// the handle holds nothing, as its `Debug` says.
    ///
    /// After the device's recovery the handle keeps the file on the new device, which imports it
    /// at the first buffer over a piece of it, as without a handle, and keeps that import from
    /// then on; the import on the lost device goes with the last object left there. Nothing is
    /// imported again as the device recovers.
    ///
    /// Refused: `Private` or `Memoryless` storage, as a no-copy buffer's, and an undefined option
    /// (`Invalid`); `file` not a dma-buf, a file whose types take none of the class's that a
    /// no-copy buffer takes, or one the driver refuses to import (`Unsupported`, `HostMemory`), as
    /// a buffer over a piece of it is refused, which then becomes a copy of its pages; memory the
    /// driver has no room for (`OutOfMemory`); a lost device.
    pub fn import_shared_dma_buf(
        &self,
        file: &Arc<OwnedFd>,
        options: ResourceOptions,
    ) -> Result<SharedDmaBufImport, Error> {
        self.at_serving(|device| device.import_shared_dma_buf_now(file, options))
    }

    /// [`Device::import_shared_dma_buf`] on the generation that serves (`device/line.rs`).
    fn import_shared_dma_buf_now(
        &self,
        file: &Arc<OwnedFd>,
        options: ResourceOptions,
    ) -> Result<SharedDmaBufImport, Error> {
        const CALL: &str = "import_shared_dma_buf";
        let class = host_class(CALL, Modes::of(no_copy_options(options).0)?)?;
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let shared = self.shared();
        let importer = (shared.host_import.as_ref().ok())
            .filter(|importer| importer.imports_dma_bufs())
            .filter(|_| self.imports_dma_buf_buffers(vk::BufferCreateFlags::empty()));
        let on = match importer {
            Some(importer) => {
                self.keep_file(importer, file, class)?;
                Some(KeptOn {
                    device: self.clone(),
                    file: Arc::clone(file),
                })
            }
            None => None,
        };
        let kept = on.is_some();
        let cell = Arc::new(KeptFile {
            file: Arc::clone(file),
            on: Mutex::new(on),
        });
        // Kept for the device's recovery; one made after its recovery took this generation's
        // keeps is carried there, by its maker (`device/carry.rs`).
        if kept && shared.carry.kept_file(&cell) {
            self.carry_late(|serving| {
                cell.carry(serving);
                Ok::<(), Infallible>(())
            });
        }
        Ok(SharedDmaBufImport(cell))
    }

    /// Keep `file` on this generation, imported: the live import, or a new one of `class`
    /// ([`import_file`](Device::import_file)) of a type a no-copy buffer takes, entered and so
    /// held by the entry; the keep let go of where the import is refused.
    fn keep_file(
        &self,
        importer: &Importer,
        file: &Arc<OwnedFd>,
        class: MemoryClass,
    ) -> Result<(), Error> {
        let shared = self.shared();
        if !self.keep_here(file) {
            return Ok(());
        }
        // The types a no-copy buffer made for dma-bufs takes, which Vulkan keeps the same for
        // every buffer of the same flags, usage and handle types; a sparse one takes them all
        // where the device splits a request (`Device::sparse_granule`).
        let made = (self
            .dma_buf_buffer_requirements(importer.page, vk::BufferCreateFlags::empty()))
        .ok_or_else(not_importable)
        .and_then(|plain| self.import_file(importer, file, plain.memory_type_bits, class));
        let made = match made {
            Ok(made) => made,
            Err(error) => {
                self.unkeep_file(file);
                return Err(error);
            }
        };
        // Entered, or another thread's entered meanwhile: either way the entry holds it, as it
        // was entered while this keep lasted.
        let hold = match self.enter_file(made) {
            Ok(entered) => {
                shared.counts.dma_buf_imported(1);
                entered
            }
            Err(live) => live,
        };
        self.release_files(shared.dma_buf_files.let_go(std::iter::once(hold)));
        Ok(())
    }

    /// Keep `file` on this generation ([`DmaBufFiles::keep`](super::DmaBufFiles::keep)): whether
    /// it is unimported there, so that the import entered next is kept.
    fn keep_here(&self, file: &Arc<OwnedFd>) -> bool {
        let shared = self.shared();
        match shared.dma_buf_files.keep(file) {
            Keep::Imported { first } => {
                if first {
                    shared.counts.dma_buf_shared_import_kept(true);
                }
                false
            }
            Keep::Unimported => true,
        }
    }

    /// Let go of a keep of `file` on this generation: with the last, the entry's hold of its
    /// import, freed where it was the last hold.
    fn unkeep_file(&self, file: &Arc<OwnedFd>) {
        let shared = self.shared();
        let Unkept { let_go, last } = shared.dma_buf_files.unkeep(file);
        if let_go {
            shared.counts.dma_buf_shared_import_kept(false);
        }
        self.release_files(last.into_iter().collect());
    }

    /// Free `last`, shared files' imports whose last hold went: no buffer is bound in them and no
    /// command uses them.
    fn release_files(&self, last: Vec<DmaBufFile>) {
        if last.is_empty() {
            return;
        }
        let shared = self.shared();
        for file in &last {
            // SAFETY: the import's last hold went: nothing is bound to it and no command uses it.
            unsafe { release::object(shared.vk(), file.memory) };
        }
        shared.counts.dma_buf_memories_freed(false, last.len());
        shared.counts.dma_buf_shared_imports_freed(last.len());
    }
}
