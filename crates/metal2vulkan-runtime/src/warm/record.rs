//! [`PipelineRecord`] and its making: what a device hands
//! [`DeviceOptions::pipeline_records`](crate::DeviceOptions::pipeline_records) at each pipeline
//! state's creation.
//!
//! - **Once per creation.** Each `new_compute_pipeline_state*`, `new_render_pipeline_state*`,
//!   `new_mesh_render_pipeline_state*`, `new_tile_render_pipeline_state*` and
//!   `new_*_pipeline_state_with_additional_binary_functions` call that makes a state hands the
//!   hook one record, on the creating thread, after the state is made and before the call
//!   returns, as the program's own code (a panic in the hook is caught). So does a twin the
//!   runtime makes of a program's render state where a draw first needs one (its one-sample or
//!   factor-scaling form, `GraphicsState::single_sample`), on the thread that makes it: its
//!   translations are the program's too. A creation that fails, a binary archive's add, and a
//!   state a warm pass makes ([`Device::warm`]) hand it nothing. Without a hook no record is made.
//! - **What it holds.** The creation's descriptor as the program gave it, every field, with each
//!   stage's function by its library and table index, its constants as supplied, its linked
//!   functions and its dynamic libraries; and how many of each stage's last binary functions were
//!   added without the stage's support flag. Its libraries are named by their content hash, and
//!   their bytes are kept in the device's shader cache directory (`shader_cache/libraries.rs`),
//!   once per library, so a record is small and a VM's records over one library hold it once.
//!   Nothing a later process would compute again is in it: no key, translation, variant or
//!   pipeline, so a record outlives every translator change (`codec.rs`).
//! - **Exact or none.** A record is read back from its own bytes as it is made, and handed over
//!   only where that reading equals the creation's descriptor; one that cannot be written (a stage
//!   the compile layer compiled already) or read back is counted
//!   ([`WarmStatistics::records_unmade`](super::WarmStatistics::records_unmade)) and not handed
//!   over.

use super::codec::{self, Pipeline, Unwritable};
use crate::device::Device;
use metal2vulkan::library::Library;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// No Metal counterpart: one pipeline state creation, as
/// [`DeviceOptions::pipeline_records`](crate::DeviceOptions::pipeline_records) hands it over, to
/// make the same translations and `VkPipeline`s again in a later process
/// ([`Device::warm`]). Opaque; cheap to clone; `Send + Sync`.
///
/// It holds the creation's descriptor and its stages' functions, constants, linked functions and
/// dynamic libraries; its libraries are named by their content hash, their bytes kept in the
/// device's shader cache directory, so a record warms only on a device over the same directory.
/// [`to_bytes`](Self::to_bytes) and [`from_bytes`](Self::from_bytes) carry it between processes;
/// equal records have equal bytes, so a consumer may keep each once.
#[derive(Clone)]
pub struct PipelineRecord {
    bytes: Arc<[u8]>,
    /// The libraries its table names, in its order, where it was made in this process; empty for
    /// one read from bytes, whose libraries the warm pass reads from the directory.
    libraries: Arc<[Library]>,
}

impl PipelineRecord {
    /// The record of `pipeline`, read back from its own bytes and found equal (the module docs).
    pub(crate) fn of(pipeline: &Pipeline) -> Result<PipelineRecord, Unwritable> {
        let (bytes, libraries) = codec::encode(pipeline)?;
        match codec::decode(&bytes, &libraries) {
            Ok(read) if read == *pipeline => Ok(PipelineRecord {
                bytes: bytes.into(),
                libraries: libraries.into(),
            }),
            _ => Err(Unwritable),
        }
    }

    /// No Metal counterpart: the record as bytes, which [`from_bytes`](Self::from_bytes) reads in
    /// a later process. They name the record format; its libraries' bytes stay in the shader
    /// cache directory.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bytes.to_vec()
    }

    /// No Metal counterpart: the record `bytes` hold, `None` where they hold none: damaged, cut,
    /// not a record, or a record of another format, which this build does not read (the format
    /// moves only where the record's bytes change meaning, never with a translator or runtime
    /// change alone, so a consumer may keep records across updates and drop those this returns
    /// `None` for).
    pub fn from_bytes(bytes: &[u8]) -> Option<PipelineRecord> {
        codec::is_record(bytes).then(|| PipelineRecord {
            bytes: bytes.into(),
            libraries: Arc::from([]),
        })
    }

    /// The libraries it was made with in this process, in its table's order; empty for a record
    /// read from bytes.
    pub(crate) fn libraries(&self) -> &[Library] {
        &self.libraries
    }

    /// Its bytes.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl PartialEq for PipelineRecord {
    fn eq(&self, other: &PipelineRecord) -> bool {
        self.bytes == other.bytes
    }
}

impl Eq for PipelineRecord {}

impl std::hash::Hash for PipelineRecord {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}

impl std::fmt::Debug for PipelineRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineRecord")
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

/// The hook a device hands its records to.
pub(crate) type RecordHook = Arc<dyn Fn(PipelineRecord) + Send + Sync>;

/// A device's records: its hook, and the counts of what it made.
pub(crate) struct Recording {
    hook: Option<RecordHook>,
    /// Records handed to the hook.
    pub(crate) made: AtomicU64,
    /// Creations the hook was set for whose record could not be made.
    pub(crate) unmade: AtomicU64,
}

impl Recording {
    /// A device's recording to `hook`, none where `None`.
    pub(crate) fn new(hook: Option<RecordHook>) -> Recording {
        Recording {
            hook,
            made: AtomicU64::new(0),
            unmade: AtomicU64::new(0),
        }
    }
}

impl Device {
    /// Hand the device's hook the record of `pipeline`, a creation that made its state (the
    /// module docs): nothing where the device has no hook or the warm pass made the state.
    /// `pipeline` is asked for only where a record is made.
    pub(crate) fn record_pipeline(&self, pipeline: impl FnOnce() -> Pipeline) {
        if super::warming() {
            return;
        }
        let shared = self.shared();
        let recording = &shared.pipeline_recording;
        let Some(hook) = &recording.hook else {
            return;
        };
        let record = match PipelineRecord::of(&pipeline()) {
            Ok(record) => record,
            Err(Unwritable) => {
                recording.unmade.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        for library in record.libraries() {
            shared.shader_cache.keep_library(library);
        }
        recording.made.fetch_add(1, Ordering::Relaxed);
        crate::exit_guard::program(|| hook(record));
    }
}
