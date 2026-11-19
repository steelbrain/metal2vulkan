//! The rt side's fence experiments ([`crate::fence_orders`]), made on the runtime's fences, blit
//! encoders, events and handlers, each queue the probe's own, as `newCommandQueue` makes them.
//!
//! The runtime keeps no hazard-tracking mode: X and Y are its buffers whichever mode the shape
//! names, and what the runtime orders without a fence is its own answer, which the comparison
//! reads. The one command buffer held is held by a shared event a guard raises wherever the probe
//! returns, a refusal after its commit included.

use super::events::{of_rt, settled, status, Raise};
use super::RtSide;
use crate::case::{Answer, Failure};
use crate::event_orders::{Status, COMPLETION};
use crate::fence_orders::{
    written_words, FenceOrder, FenceShape, BYTES, HELD_BOUND, LONG_FILLS, LONG_FILL_BYTES, WRITTEN,
};
use metal2vulkan_rt as rt;
use std::sync::mpsc;

/// The long job on `encoder`: [`LONG_FILLS`] fills of `long`, no loop on the GPU.
fn long_job(encoder: &mut rt::BlitCommandEncoder, long: &rt::Buffer) -> Answer<()> {
    for fill in 0..LONG_FILLS {
        (encoder.fill_buffer(long, 0..LONG_FILL_BYTES, fill as u8)).map_err(of_rt)?;
    }
    Ok(())
}

/// The bytes of a Shared buffer's first [`BYTES`], read where nothing writes them any more.
fn words_written(buffer: &rt::Buffer) -> Answer<u64> {
    let contents = (buffer.contents())
        .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
    // SAFETY: the probe made the buffer `BYTES` bytes long and the host reaches them.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), BYTES as usize) };
    Ok(written_words(bytes))
}

impl RtSide<'_> {
    /// [`crate::case::Side::fence_order`].
    pub(super) fn fence_probe(&mut self, shape: FenceShape) -> Answer<FenceOrder> {
        let device = &self.on.device;
        let (updating_queue, waiting_queue) = (self.own_queue()?, self.own_queue()?);
        let x = self.scratch(rt::MemoryClass::Shared, BYTES)?;
        let y = self.scratch(rt::MemoryClass::Shared, BYTES)?;
        let long = self.scratch(rt::MemoryClass::Private, LONG_FILL_BYTES)?;
        let fence = device.new_fence();
        let gate = device.new_shared_event().map_err(of_rt)?;
        // The busy command buffers ahead of the updater, each a long job, committed at once.
        let mut busy = Vec::new();
        if shape.busy() > 0 {
            for _ in 0..shape.busy() {
                let ahead = updating_queue.command_buffer().map_err(of_rt)?;
                let mut encoder = ahead.blit_command_encoder().map_err(of_rt)?;
                long_job(&mut encoder, &long)?;
                encoder.end_encoding();
                ahead.commit().map_err(of_rt)?;
                busy.push(ahead);
            }
        }
        let updater = updating_queue.command_buffer().map_err(of_rt)?;
        if shape == FenceShape::HeldUpdater {
            (updater.encode_wait_for_event(gate.as_event(), 1)).map_err(of_rt)?;
        }
        let mut encoder = updater.blit_command_encoder().map_err(of_rt)?;
        if matches!(shape, FenceShape::Ordered | FenceShape::EmptyWaiter) {
            long_job(&mut encoder, &long)?;
        }
        (encoder.fill_buffer(&x, 0..BYTES, WRITTEN)).map_err(of_rt)?;
        if shape.untracked() {
            encoder.update_fence(&fence).map_err(of_rt)?;
        }
        encoder.end_encoding();
        // Raised wherever the probe returns from here, so no refusal leaves the updater held.
        let release = Raise(&gate);
        updater.commit().map_err(of_rt)?;
        let waiter = waiting_queue.command_buffer().map_err(of_rt)?;
        let mut encoder = waiter.blit_command_encoder().map_err(of_rt)?;
        if shape.untracked() {
            encoder.wait_for_fence(&fence).map_err(of_rt)?;
        }
        if shape != FenceShape::EmptyWaiter {
            (encoder.copy_from_buffer_to_buffer(&x, 0, &y, 0, BYTES)).map_err(of_rt)?;
        }
        encoder.end_encoding();
        // The waiter's completed handler counts X's words as it runs: what the updater had
        // written by the waiter's completion.
        let (sender, counted) = mpsc::channel();
        let x_address = (x.contents())
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?
            .as_ptr() as usize;
        (waiter.add_completed_handler(move |_: &rt::CommandBuffer| {
            // SAFETY: X is `BYTES` bytes of shared memory the probe keeps alive until it has
            // waited for this handler or its bound; a read races no CPU write.
            let bytes =
                unsafe { std::slice::from_raw_parts(x_address as *const u8, BYTES as usize) };
            let _ = sender.send(written_words(bytes));
        }))
        .map_err(of_rt)?;
        waiter.commit().map_err(of_rt)?;
        let mut waiter_completed_before_the_release = None;
        let mut updater_before_the_release = None;
        let mut source_words = None;
        if shape == FenceShape::HeldUpdater {
            let completed = counted.recv_timeout(HELD_BOUND);
            updater_before_the_release = Some(status(&updater));
            waiter_completed_before_the_release = Some(completed.is_ok());
            source_words = completed.ok();
        }
        drop(release);
        let source_words = match source_words {
            Some(words) => Some(words),
            None => counted.recv_timeout(COMPLETION).ok(),
        };
        let waiter_at_the_end = match source_words {
            Some(_) => status(&waiter),
            None => Status::NeverSettled,
        };
        let updater_at_the_end = settled(&updater, COMPLETION);
        for ahead in &busy {
            settled(ahead, COMPLETION);
        }
        Ok(FenceOrder {
            waiter_completed_before_the_release,
            updater_before_the_release,
            source_words_when_the_waiter_completed: (shape == FenceShape::EmptyWaiter)
                .then_some(source_words)
                .flatten(),
            copied_words: match shape {
                FenceShape::EmptyWaiter => None,
                _ => Some(words_written(&y)?),
            },
            updater_at_the_end,
            waiter_at_the_end,
        })
    }
}
