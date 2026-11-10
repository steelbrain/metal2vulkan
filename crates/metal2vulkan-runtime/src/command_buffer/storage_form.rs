//! A commit that takes the waiting fetched render targets' storage forms
//! (`texture/storage_form.rs`): where the committing command buffer allows it, a last part of it
//! copies each target's image into its storage form, and the switch rides on the command buffer
//! ([`Riding`]) until its scheduler settles it at the submission of its last batch.

use super::{lock, CommandBuffer, CommandBufferShared, State};
use crate::encoder::{EncoderCore, Recorded};
use crate::texture::storage_form::Riding;

impl State {
    /// Whether its commit may take a switch (`texture/storage_form.rs`): it may be committed, it
    /// holds a slot of its queue (so it counts among the device's uncommitted command buffers)
    /// and ended an encoder; every encoder of it may run; it waits for no event or binary
    /// semaphore and leaves no segment and no sparse update to a host step, so the scheduler
    /// submits all of it once its submission checks (if it has any) ran; and it bound no heap
    /// set.
    fn takes_storage_forms(&self) -> bool {
        self.refusal().is_none()
            && self.slot.is_some()
            && !self.ended.is_empty()
            && self.fails_from.is_none()
            && self.faulted.is_empty()
            && !self.heaps.bound()
            && self.recorded.iter().all(|part| {
                part.segments.is_empty() && part.sparse.iter().all(|update| !update.needs_step())
            })
            && self
                .batches
                .all()
                .iter()
                .all(|batch| batch.waits.is_empty() && batch.binary_waits.is_empty())
    }
}

impl CommandBuffer {
    /// Take the switch of the waiting textures' storage forms at this commit (with an `injected`
    /// failure or not), where the command buffer and the device allow it now: their copies
    /// appended to the command buffer as its last part, and the switch riding on it. Whether it
    /// did; nothing appended (one atomic read) where no texture waits.
    pub(super) fn take_storage_forms(&self, injected: bool) -> bool {
        let device = &self.0.device;
        if injected || !device.storage_forms_wanted() {
            return false;
        }
        let parts = {
            let state = self.state();
            if !state.takes_storage_forms() {
                return false;
            }
            state.recorded.len()
        };
        let Some(switch) = device.storage_form_switch() else {
            return false;
        };
        let Ok(mut core) = EncoderCore::segment(self, true) else {
            device.put_back_switch(switch);
            return false;
        };
        let prepared = switch.prepare(device, &mut core);
        if prepared.is_empty() {
            drop(core);
            device.give_up_switch(switch, prepared);
            return false;
        }
        let Ok(recorded) = core.end_segment() else {
            device.give_up_switch(switch, prepared);
            return false;
        };
        let mut state = self.state();
        // Nothing else ended meanwhile (a program that encodes on another thread as it commits).
        let encoder = state.ended.len().checked_sub(1);
        let (Some(encoder), true) = (
            encoder,
            state.recorded.len() == parts && state.takes_storage_forms(),
        ) else {
            drop(state);
            // The images made go unused, never submitted.
            drop(recorded);
            device.give_up_switch(switch, prepared);
            return false;
        };
        state.end_encoder(
            self,
            Recorded {
                encoder,
                raws: recorded.raws,
                barriered: None,
                segments: Vec::new(),
                summary: recorded.summary,
                sparse: Vec::new(),
                twin_writes: Vec::new(),
            },
        );
        state.retained.extend(recorded.retained.into_entry());
        // What its commands took is the encoder's, as the switch's commands are.
        if let Some(record) = state.ended.get_mut(encoder) {
            record.taken.join(recorded.taken);
        }
        state.storage_form = Some(Riding::new(device, switch, prepared));
        true
    }

    /// The commit refused the command buffer, on which a switch rode: given up, nothing swaps.
    pub(super) fn give_up_storage_forms(&self) {
        let riding = self.state().storage_form.take();
        // With the lock free: its drop gives the switch up under the device's switch lock.
        drop(riding);
    }
}

impl CommandBufferShared {
    /// Its last batch was submitted, or it failed (the scheduler, under its lock): settle the
    /// switch riding on it (`texture/storage_form.rs`), the verdict whether every encoder of it
    /// ran (no check failed a submission of it, and its submission did not fail). Whether one
    /// rode on it.
    pub(crate) fn settle_storage_forms(&self) -> bool {
        let _for = self.notices_for();
        let state = lock(&self.state);
        let Some(riding) = &state.storage_form else {
            return false;
        };
        riding.settle(state.failure.is_none());
        true
    }
}
