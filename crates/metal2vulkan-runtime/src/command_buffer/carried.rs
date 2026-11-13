//! What a command buffer's batches carried into a `vkQueueSubmit` that failed, for the line it
//! logs (`submit_failure.rs`): the command buffer as the runtime's notices name it, and the
//! resources its encoders' uses name and what they made resident, each as what owns its memory
//! (`used_resources.rs`). Read only once the call failed, under the locks its submitter holds.

use super::{CommandBufferShared, State};
use crate::submit_failure::{Carried, TypeFlags};
use crate::used_resources::{failure_facts, resident_keys};
use std::collections::BTreeSet;
use std::fmt::Write;
use std::ops::Range;

impl CommandBufferShared {
    /// Add the command buffer, `state` its state, to `carried`: named by its identity and as the
    /// runtime's notices name it, with its batch `index` where given; the role of each
    /// `VkCommandBuffer` of the encoders it recorded at `encoders` (every one it recorded where
    /// that is `None`); and the resources those encoders name, or, where it is `None`, every ended
    /// encoder of it, stated with `types`' flags for their memory types.
    pub(crate) fn carry(
        &self,
        state: &State,
        (index, encoders): (Option<usize>, Option<Range<usize>>),
        carried: &mut Carried,
        types: &TypeFlags,
    ) {
        let queue = self.queue.label();
        let mut named = format!("#{} {}", self.id.get(), state.named(queue.as_deref()));
        if let Some(index) = index {
            let batches = state.batches.all().len();
            let _ = write!(named, ", its batch {index} of {batches}");
        }
        carried.buffers.push(named);
        let recorded = match &encoders {
            Some(encoders) => &state.recorded[encoders.clone()],
            None => &state.recorded[..],
        };
        let mut keys = BTreeSet::new();
        for part in recorded {
            let kind = match state.ended.get(part.encoder) {
                Some(record) => format!("{:?}", record.kind),
                None => "open".to_owned(),
            };
            let role = format!("encoder {} {kind} of #{}", part.encoder, self.id.get());
            carried
                .roles
                .extend(part.raws.iter().map(|&raw| (raw, role.clone())));
            if encoders.is_some() {
                keys.extend(part.summary.keys());
            }
        }
        let ended = match encoders {
            Some(_) => (recorded.iter())
                .filter_map(|part| state.ended.get(part.encoder))
                .collect(),
            None => state.ended.iter().collect::<Vec<_>>(),
        };
        for record in ended {
            keys.extend(&record.keys);
            if let Some(residency) = &record.residency {
                resident_keys(residency, &mut keys);
            }
        }
        let shared = self.device.shared();
        failure_facts(
            shared,
            &keys,
            types,
            (&mut carried.resources, &mut carried.stated),
        );
    }
}
