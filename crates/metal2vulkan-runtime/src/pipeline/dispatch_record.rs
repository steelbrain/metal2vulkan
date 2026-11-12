//! A compute state's dispatches in its pipeline records (`warm/record.rs`), and their making on
//! a warm pass (`warm/pass.rs`).
//!
//! - **What a creation cannot predict.** A compute state's creation makes its first dispatch's
//!   `VkPipeline` only where its descriptor requires a threadgroup size (`dispatch_making.rs`): the
//!   local size and the threadgroup memory lengths are the dispatch's own calls, so a state that
//!   requires none makes each region's pipeline at its first dispatch, and the creation's record
//!   makes none of them on a later warm pass.
//! - **A record per new dispatch.** Where the device hands records over, a state's direct dispatch
//!   (`dispatchThreadgroups:` or `dispatchThreads:`) whose regions' pipelines were not all made
//!   hands the hook one more record ([`ComputePipelineState::record_dispatch`]): the creation's
//!   descriptor, as its own record holds it, and the dispatch ([`RecordedDispatch`]: its
//!   threadgroup, its grid's ragged edge where it is a `dispatchThreads:`, and the encoder's
//!   threadgroup memory lengths). Each such dispatch once per state, at most
//!   [`DISPATCH_RECORDS`] of them; an indirect dispatch, a dispatch of a threadgroup range and a
//!   tile dispatch hand over none. A record is immutable, so a consumer keeps each once and needs
//!   no rule for which supersedes which.
//! - **Made as the program's dispatch makes it** ([`ComputePipelineState::warm_dispatch`]). A warm
//!   pass that creates such a record's state then plans the dispatch as a creation's prediction
//!   plans its first one (nothing bound but Metal's default sampler at each runtime sampler slot
//!   that does not compare), with the recorded threadgroup memory lengths, settles its variant as
//!   a dispatch does (compiling it where it is not the base), and makes each region's `VkPipeline`
//!   on the pass's thread, below every other making: its translation lands in the shader cache and
//!   its driver data in its record, under the names a later dispatch of the same shape asks for. A
//!   dispatch whose bindings specialize its regions otherwise (a bound texture's format the
//!   variant reads, a uniform view read at its bind offset) asks for other names, and makes its
//!   own as before.

use super::dispatch_making::predicted_bindings;
use super::{BufferForms, ComputePipelineState, RegionKey};
use crate::device::lock;
use crate::warm::codec::RecordedDispatch;
use metal2vulkan::bind::{BindState, Dispatch};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, Stage};
use std::collections::BTreeSet;
use std::sync::Mutex;

/// The most dispatches of one state that hand the hook a record: past them a state's new
/// dispatches make their pipelines as before, unrecorded.
pub(crate) const DISPATCH_RECORDS: usize = 16;

/// What a compute state keeps to record its dispatches: its creation as its record holds it (the
/// descriptor the program gave, and how many of its kernel's last binary functions were added
/// without the support flag), and the dispatches it recorded.
pub(crate) struct DispatchRecords {
    creation: (ComputePipelineDescriptor, usize),
    recorded: Mutex<BTreeSet<RecordedDispatch>>,
}

impl DispatchRecords {
    /// The records of a state created as `creation` says.
    pub(crate) fn new(creation: (ComputePipelineDescriptor, usize)) -> DispatchRecords {
        DispatchRecords {
            creation,
            recorded: Mutex::new(BTreeSet::new()),
        }
    }
}

/// `dispatch`, a program's, as a record keeps it (the module docs), over the threadgroup memory
/// lengths `state` binds: `None` for a dispatch no record keeps (indirect, or of a threadgroup
/// range).
fn recorded(dispatch: &Dispatch, state: &BindState) -> Option<RecordedDispatch> {
    let (threads_per_group, threads) = match *dispatch {
        Dispatch::Threadgroups {
            threads_per_group, ..
        } => (threads_per_group, None),
        Dispatch::Threads {
            threads,
            threads_per_group,
        } => {
            // Each axis one whole threadgroup and the grid's ragged edge: the regions, and so the
            // pipelines, of any grid of that edge.
            let axis = |a: usize| match (threads[a], threads_per_group[a]) {
                (n, per) if n <= per || per == 0 => n,
                (n, per) if n % per == 0 => per,
                (n, per) => per + n % per,
            };
            (threads_per_group, Some([axis(0), axis(1), axis(2)]))
        }
        _ => return None,
    };
    let threadgroup_lengths = (state.stages.get(&Stage::Kernel))
        .map(|table| {
            table
                .threadgroup_lengths
                .iter()
                .map(|(i, n)| (*i, *n))
                .collect()
        })
        .unwrap_or_default();
    Some(RecordedDispatch {
        threads_per_group,
        threads,
        threadgroup_lengths,
    })
}

/// The Metal dispatch `recorded` stands for: one threadgroup, or its grid.
fn dispatch_of(recorded: &RecordedDispatch) -> Dispatch {
    let threads_per_group = recorded.threads_per_group;
    match recorded.threads {
        Some(threads) => Dispatch::Threads {
            threads,
            threads_per_group,
        },
        None => Dispatch::Threadgroups {
            groups: [1, 1, 1],
            threads_per_group,
        },
    }
}

impl ComputePipelineState {
    /// Hand the device's record hook the record of `dispatch`, a direct dispatch of this state
    /// over `state` whose regions' pipelines were not all made, where the state records its
    /// dispatches and has not recorded this one, nor [`DISPATCH_RECORDS`] others (the module
    /// docs).
    pub(crate) fn record_dispatch(&self, dispatch: &Dispatch, state: &BindState) {
        let Some(records) = self.0.dispatch_records.get() else {
            return;
        };
        let Some(recorded) = recorded(dispatch, state) else {
            return;
        };
        {
            let mut seen = lock(&records.recorded);
            if seen.len() >= DISPATCH_RECORDS || !seen.insert(recorded.clone()) {
                return;
            }
        }
        let (descriptor, added) = &records.creation;
        self.0.device.record_pipeline(|| {
            crate::warm::Pipeline::Compute(descriptor.clone(), *added, vec![recorded])
        });
    }

    /// On a warm pass's thread, make the `VkPipeline` of each region of `recorded`, a dispatch a
    /// record of this state names, as the module docs say: how many were made. A dispatch the
    /// plan refuses makes none, and a making that fails is not counted; neither fails the record.
    pub(crate) fn warm_dispatch(&self, recorded: &RecordedDispatch) -> u64 {
        let device = &self.0.device;
        let dispatch = dispatch_of(recorded);
        let mut state = predicted_bindings(&self.0.base.pipeline);
        let lengths = &mut state
            .stages
            .entry(Stage::Kernel)
            .or_default()
            .threadgroup_lengths;
        lengths.extend(recorded.threadgroup_lengths.iter().copied());
        let Ok((variant, plan)) = self.settle(&state.shape(), &dispatch, BufferForms::Bound) else {
            return 0;
        };
        let Ok(mut commands) = plan.instantiate_dispatch(&state, &dispatch) else {
            return 0;
        };
        variant.keyed_regions(device, &mut commands);
        let made = (commands.regions.iter())
            .filter(|region| variant.make_region(&RegionKey::of(region)).is_ok())
            .count();
        made as u64
    }
}
