//! Pipeline records and the warm pass: what a device hands
//! [`DeviceOptions::pipeline_records`](crate::DeviceOptions::pipeline_records) at each pipeline
//! state's creation (`record.rs`, `codec.rs`), for a consumer to keep and hand a later process's
//! device ([`Device::warm`]), which makes the same translations and `VkPipeline`s before its
//! program asks for them (`pass.rs`).

pub(crate) mod codec;
mod pass;
mod record;

pub(crate) use codec::Pipeline;
/// The gate in front of a record's making, which `tests::warm` closes.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use pass::gate;
/// A warm pass's progress, which the budget's tests make one of.
#[cfg(test)]
pub(crate) use pass::Pass;
/// The id name of the warning of a device's first unread record, which `tests::warm` listens for.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use pass::UNREAD;
pub(crate) use pass::{
    added_pipeline_bytes, pass, translation_only, waiting, warming, within, Warmer,
};
/// The id name of the warning of a device's first failed creation of each kind, and the kind it
/// names, which `tests::warm` listens for.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use pass::{failure_kind, FAILED};
pub use record::PipelineRecord;
pub(crate) use record::{RecordHook, Recording};

use crate::device::Device;
use std::sync::atomic::Ordering;

/// No Metal counterpart (a runtime extension): a device's pipeline records and warm passes since
/// it was made, and the record a pass is making now ([`Device::warm_statistics`]).
///
/// The record counts of a warm pass add up, once its calls are done, to the records handed to
/// [`Device::warm`]: each was warmed, translated only, failed, unread, not warmable or not run.
/// While a record is made, they add up to its place ([`record_in_making`](Self::record_in_making)),
/// and [`record_in_making_nanos`](Self::record_in_making_nanos) says how long its making has
/// worked, so a consumer can tell a making that works from one that waits.
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
    /// The `VkPipeline`s warm passes made for the dispatches their compute records name (each
    /// region of each dispatch, planned as `Device::warm` documents, and its residual form where
    /// that differs: a storage view read at its bind offset, on a profile whose storage views
    /// declare the key), on top of what the creations predict.
    pub dispatch_pipelines_warmed: u64,
    /// Records a warm pass made the translations of alone: a layout that bakes an immutable
    /// sampler, which a warm pass makes none of.
    pub records_translated: u64,
    /// Records whose creation failed on the warm pass, dropped: the device's first of each kind
    /// of failure is announced (`metal2vulkan-warm-record-failed`), naming the kind and nothing
    /// of the record's shaders.
    pub records_failed: u64,
    /// Of [`records_failed`](Self::records_failed), those a pass answered from the failure an
    /// earlier pass over the same shader cache directory kept for the record, on the same
    /// device, driver, profile, runtime version and translator: a refusal for good (a kernel
    /// past what its driver compiles intact, or past the private memory a thread holds), so the
    /// record's creation was not made again (`shader_cache/warm_refusals.rs`). A transient
    /// failure is never kept.
    pub records_failed_before: u64,
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
    /// [`Device::warm`]'s `byte_budget` bounds. A making adds what it writes to its record past
    /// the data the record's cache was loaded with or last wrote, so a making whose data the
    /// directory holds already adds nothing: a pass over records an earlier process made over the
    /// same directory, whose creations wrote their data, adds nothing for them, and its budget
    /// binds only where a making writes data the directory did not hold
    /// (`shader_cache/driver.rs`).
    pub pipeline_bytes: u64,
    /// The record a warm pass is making now, if any: its place among every record handed to
    /// this device's [`Device::warm`] calls, in the order handed (the first call's first is 0),
    /// the place the six record counts' sum names while it is made. `None` between records,
    /// and while no pass runs (on a lost generation's line too).
    pub record_in_making: Option<u64>,
    /// How long the making of `record_in_making` has worked, in nanoseconds, up to now: the
    /// time its read and translation ran and its driver calls held their budget turns
    /// (`pipeline/budget.rs`). The time it waited for a turn, or for the large-making words
    /// limit, behind the process's other makings is left out, so a record starved by a
    /// program's storm never grows it. 0 where `record_in_making` is None.
    pub record_in_making_nanos: u64,
}

impl Device {
    /// No Metal counterpart: the device's pipeline records and warm passes since it was made, and
    /// the record a pass is making now.
    pub fn warm_statistics(&self) -> WarmStatistics {
        self.current().warm_statistics_now()
    }

    /// [`Device::warm_statistics`] on the generation that serves (`device/line.rs`).
    pub(crate) fn warm_statistics_now(&self) -> WarmStatistics {
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
