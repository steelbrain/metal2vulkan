//! Pipeline records and the warm pass: what a device hands
//! [`DeviceOptions::pipeline_records`](crate::DeviceOptions::pipeline_records) at each pipeline
//! state's creation (`record.rs`, `codec.rs`), for a consumer to keep and hand a later process's
//! device ([`Device::warm`]), which makes the same translations and `VkPipeline`s before its
//! program asks for them (`pass.rs`).

pub(crate) mod codec;
mod pass;
mod record;

pub(crate) use codec::Pipeline;
/// A warm pass's progress, which the budget's tests make one of.
#[cfg(test)]
pub(crate) use pass::Pass;
/// The id name of the warning of a device's first unread record, which `tests::warm` listens for.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use pass::UNREAD;
pub(crate) use pass::{added_pipeline_bytes, pass, translation_only, warming, within, Warmer};
pub use record::PipelineRecord;
pub(crate) use record::{RecordHook, Recording};

use crate::device::Device;
use std::sync::atomic::Ordering;

/// No Metal counterpart: a device's pipeline records and warm passes since it was made
/// ([`Device::warm_statistics`]).
///
/// The record counts of a warm pass add up, once its calls are done, to the records handed to
/// [`Device::warm`]: each was warmed, translated only, failed, unread, not warmable or not run.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WarmStatistics {
    /// Records handed to [`DeviceOptions::pipeline_records`](crate::DeviceOptions::pipeline_records):
    /// one per creation that made its state.
    pub records_made: u64,
    /// Creations the hook was set for whose record was not made: a stage the compile layer had
    /// compiled already, a value the record format cannot hold, or bytes that did not read back as
    /// the creation's descriptor (`warm/codec.rs`).
    pub records_unmade: u64,
    /// Libraries those records name that the device wrote to its shader cache directory, each
    /// once per directory (`shader_cache/libraries.rs`).
    pub libraries_kept: u64,
    /// Records a warm pass made the creation of: its translations and its predicted
    /// `VkPipeline`s.
    pub records_warmed: u64,
    /// Records a warm pass made the translations of alone: a layout that bakes an immutable
    /// sampler, which a warm pass makes none of.
    pub records_translated: u64,
    /// Records whose creation failed on the warm pass, dropped.
    pub records_failed: u64,
    /// Records a warm pass could not read: of another format, damaged, or naming a library the
    /// shader cache directory no longer holds. The device's first is announced
    /// (`metal2vulkan-warm-record-unread`); the pass goes on with the next record.
    pub records_unread: u64,
    /// Mesh render pipeline records, which a warm pass does not make: that creation keeps no
    /// translation on disk and makes no `VkPipeline`.
    pub records_not_warmable: u64,
    /// Records a warm pass did not reach: its byte budget was spent, the process began exiting,
    /// or no thread could be started.
    pub records_not_run: u64,
    /// The driver's pipeline cache data the warm passes' makings added, in bytes: what
    /// [`Device::warm`]'s `byte_budget` bounds.
    pub pipeline_bytes: u64,
}

impl Device {
    /// No Metal counterpart: the device's pipeline records and warm passes since it was made.
    pub fn warm_statistics(&self) -> WarmStatistics {
        let shared = self.shared();
        let read = |counter: &std::sync::atomic::AtomicU64| counter.load(Ordering::Relaxed);
        let mut statistics = WarmStatistics {
            records_made: read(&shared.pipeline_recording.made),
            records_unmade: read(&shared.pipeline_recording.unmade),
            libraries_kept: shared.shader_cache.libraries_kept(),
            ..WarmStatistics::default()
        };
        shared.warmer.read(&mut statistics);
        statistics
    }
}
