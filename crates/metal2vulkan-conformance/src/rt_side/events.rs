//! The rt side's event and handler probes: the orders of [`crate::event_orders`] and
//! [`crate::scheduled_handlers`], made as the old runtime's side makes them, on the runtime's
//! events, command buffers and handlers.
//!
//! Each probe makes its own queues, as `newCommandQueue` does. The runtime's waits have no bound,
//! so a wait for a command buffer here is [`waited_for`]'s, on a thread of its own: one that never
//! returns is recorded as the probe's negative fact rather than hanging the suite. A command
//! buffer held by an event the host raises is released before any refusal after its commit is
//! answered, so no refusal leaves it held.

use super::{waited_for, RtSide, QUEUE_COMMAND_BUFFERS};
use crate::case::{Answer, Failure};
use crate::event_orders::{
    CrossQueueWait, HeldCommandBuffer, SignalBeforeWait, Status, WaitBehindRunningSignal,
    COMPLETED, COMPLETION, HALF, HELD_FOR, LONG_FILLS, LONG_FILL_BYTES, RUNNING_WINDOW, SCHEDULED,
    SCRATCH, SIGNALLING_BYTE, SIGNAL_WAIT, WAITING_BYTE,
};
use crate::scheduled_handlers::{Returned, ScheduledHandlerOrder, ARRIVAL, HOLD_LIMIT, WINDOW};
use crate::shared_events::{EventRise, WAIT_MS};
use metal2vulkan_rt as rt;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// A typed error of the runtime as the failure it records.
fn of_rt(error: rt::Error) -> Failure {
    Failure::of_rt(&error)
}

/// The command buffer's status now, as the probes read it.
fn status(command_buffer: &rt::CommandBuffer) -> Status {
    Status::read(format!("{:?}", command_buffer.status()))
}

/// The command buffer's status once its wait returned within `bound`, or that it never did.
fn settled(command_buffer: &rt::CommandBuffer, bound: Duration) -> Status {
    waited_for(command_buffer, bound).map_or(Status::NeverSettled, Status::read)
}

/// One blit encoder on `command_buffer` filling `range` of `buffer` with `byte`.
fn fill(
    command_buffer: &rt::CommandBuffer,
    buffer: &rt::Buffer,
    range: Range<u64>,
    byte: u8,
) -> Answer<()> {
    let mut blit = command_buffer.blit_command_encoder().map_err(of_rt)?;
    blit.fill_buffer(buffer, range, byte).map_err(of_rt)?;
    blit.end_encoding();
    Ok(())
}

/// The first `length` bytes of a Shared buffer, read once the command buffers that wrote them
/// were waited for.
fn read(buffer: &rt::Buffer, length: u64) -> Answer<Vec<u8>> {
    let contents = (buffer.contents())
        .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
    // SAFETY: the probe made the buffer `length` bytes long, the host reaches them, and the caller
    // waited for the command buffers that wrote them.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) };
    Ok(bytes.to_vec())
}

/// Raises a shared event to 1 when dropped: the command buffer it holds is released wherever the
/// probe returns, a refusal past its commit included.
struct Raise<'a>(&'a rt::SharedEvent);

impl Drop for Raise<'_> {
    fn drop(&mut self) {
        self.0.set_signaled_value(1);
    }
}

/// The names a command buffer's handlers logged, in the order they were called, each also sent so
/// the probe waits for the one it reads after rather than guessing when the log is whole, as the
/// old runtime's side keeps it.
struct HandlerLog {
    called: Arc<Mutex<Vec<String>>>,
    sender: mpsc::Sender<String>,
    arrived: mpsc::Receiver<String>,
}

impl HandlerLog {
    fn new() -> Self {
        let (sender, arrived) = mpsc::channel();
        Self {
            called: Arc::default(),
            sender,
            arrived,
        }
    }

    /// A handler that logs itself as `name`.
    fn handler(&self, name: &'static str) -> impl FnOnce(&rt::CommandBuffer) + Send + 'static {
        let (called, sender) = (Arc::clone(&self.called), self.sender.clone());
        move |_: &rt::CommandBuffer| {
            if let Ok(mut called) = called.lock() {
                called.push(name.to_owned());
            }
            let _ = sender.send(name.to_owned());
        }
    }

    /// Wait until the handler named `name` has been called, or `bound` runs out.
    fn wait_for(&self, name: &str, bound: Duration) {
        let deadline = Instant::now() + bound;
        loop {
            if (self.called.lock()).is_ok_and(|called| called.iter().any(|held| held == name)) {
                return;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || self.arrived.recv_timeout(left).is_err() {
                return;
            }
        }
    }

    /// The names logged so far, in order.
    fn names(&self) -> Vec<String> {
        (self.called.lock())
            .map(|called| called.clone())
            .unwrap_or_default()
    }
}

impl RtSide<'_> {
    /// A queue of the probe's own, as `newCommandQueue`.
    fn own_queue(&self) -> Answer<rt::CommandQueue> {
        (self.on.device.new_command_queue(QUEUE_COMMAND_BUFFERS)).map_err(of_rt)
    }

    /// A buffer of `length` bytes in a block of `class` of its own: a Shared one zeroed, as
    /// `newBufferWithLength:options:` zeroes, and a Private one, which nothing reads, not.
    fn scratch(&self, class: rt::MemoryClass, length: u64) -> Answer<rt::Buffer> {
        let device = &self.on.device;
        let block = device.take_block(class, length).map_err(of_rt)?;
        let zeroed = (class == rt::MemoryClass::Shared).then_some(0);
        (device.new_buffer(block.id, 0, length, zeroed)).map_err(of_rt)
    }

    /// [`crate::case::Side::event_cross_queue_wait`].
    pub(super) fn cross_queue_wait(&mut self) -> Answer<CrossQueueWait> {
        let (waiting_queue, signalling_queue) = (self.own_queue()?, self.own_queue()?);
        let event = self.on.device.new_event().map_err(of_rt)?;
        let scratch = self.scratch(rt::MemoryClass::Shared, SCRATCH)?;
        let waiting = waiting_queue.command_buffer().map_err(of_rt)?;
        waiting.encode_wait_for_event(&event, 1).map_err(of_rt)?;
        fill(&waiting, &scratch, 0..HALF, WAITING_BYTE)?;
        let signalling = signalling_queue.command_buffer().map_err(of_rt)?;
        fill(&signalling, &scratch, HALF..SCRATCH, SIGNALLING_BYTE)?;
        signalling.encode_signal_event(&event, 1).map_err(of_rt)?;
        // The waiting one first: its signal is not committed yet.
        waiting.commit().map_err(of_rt)?;
        let waiting_after_its_commit = status(&waiting);
        signalling.commit().map_err(of_rt)?;
        let signalling_at_the_end = settled(&signalling, COMPLETION);
        let waiting_at_the_end = settled(&waiting, COMPLETION);
        Ok(CrossQueueWait {
            waiting_after_its_commit,
            waiting_at_the_end,
            signalling_at_the_end,
            bytes: read(&scratch, SCRATCH)?,
        })
    }

    /// [`crate::case::Side::shared_event_rises`]: the host's sets, the GPU's signals and the
    /// host's bounded waits on one shared event of the runtime's. The runtime keeps no label, so
    /// the side states none and the label is a refused field (the consumer's own).
    pub(super) fn event_rise(&mut self) -> Answer<EventRise> {
        let bound = Duration::from_millis(WAIT_MS);
        let queue = self.own_queue()?;
        let event = self.on.device.new_shared_event().map_err(of_rt)?;
        let initial = event.signaled_value();
        // The host's sets: a lower value leaves the event where it was.
        event.set_signaled_value(5);
        event.set_signaled_value(3);
        event.set_signaled_value(5);
        let after_host_sets = event.signaled_value();
        let signal = |value: u64| -> Answer<u64> {
            let command_buffer = queue.command_buffer().map_err(of_rt)?;
            (command_buffer.encode_signal_event(event.as_event(), value)).map_err(of_rt)?;
            command_buffer.commit().map_err(of_rt)?;
            if waited_for(&command_buffer, COMPLETION).is_none() {
                return Err(Failure::named(
                    "NeverCompleted",
                    format!("the command buffer signalling {value}"),
                ));
            }
            Ok(event.signaled_value())
        };
        let wait = |value: u64| (event.wait_until_signaled_value(value, bound)).map_err(of_rt);
        let after_gpu_signals = [signal(3)?, signal(5)?];
        let waits = [wait(3)?, wait(5)?, wait(6)?];
        let after_max_signal = signal(u64::MAX)?;
        let waited_for_max = wait(u64::MAX)?;
        Ok(EventRise {
            label: Err(Failure::unavailable("label")),
            initial,
            after_host_sets,
            after_gpu_signals,
            waits,
            after_max_signal,
            waited_for_max,
        })
    }

    /// [`crate::case::Side::event_highest_signal`].
    pub(super) fn highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        let queues = [self.own_queue()?, self.own_queue()?];
        let event = self.on.device.new_shared_event().map_err(of_rt)?;
        let scratch = self.scratch(rt::MemoryClass::Private, SCRATCH)?;
        let mut committed = Vec::new();
        for (queue, &value) in queues.iter().zip(&values) {
            let command_buffer = queue.command_buffer().map_err(of_rt)?;
            fill(&command_buffer, &scratch, 0..SCRATCH, value as u8)?;
            (command_buffer.encode_signal_event(event.as_event(), value)).map_err(of_rt)?;
            command_buffer.commit().map_err(of_rt)?;
            committed.push(command_buffer);
        }
        for command_buffer in &committed {
            if waited_for(command_buffer, COMPLETION).is_none() {
                return Err(Failure::named(
                    "NeverCompleted",
                    "a signalling command buffer",
                ));
            }
        }
        Ok(event.signaled_value())
    }

    /// [`crate::case::Side::held_command_buffer`].
    pub(super) fn held_command_buffer_probe(&mut self) -> Answer<HeldCommandBuffer> {
        let (held_queue, other_queue) = (self.own_queue()?, self.own_queue()?);
        let event = self.on.device.new_shared_event().map_err(of_rt)?;
        let scratch = self.scratch(rt::MemoryClass::Private, SCRATCH)?;
        let log = HandlerLog::new();
        let held = held_queue.command_buffer().map_err(of_rt)?;
        (held.encode_wait_for_event(event.as_event(), 1)).map_err(of_rt)?;
        fill(&held, &scratch, 0..SCRATCH, 1)?;
        (held.add_scheduled_handler(log.handler(SCHEDULED))).map_err(of_rt)?;
        (held.add_completed_handler(log.handler(COMPLETED))).map_err(of_rt)?;
        held.commit().map_err(of_rt)?;
        let raise = Raise(&event);
        let behind = held_queue.command_buffer().map_err(of_rt)?;
        fill(&behind, &scratch, 0..SCRATCH, 2)?;
        behind.commit().map_err(of_rt)?;
        let other = other_queue.command_buffer().map_err(of_rt)?;
        fill(&other, &scratch, 0..SCRATCH, 3)?;
        other.commit().map_err(of_rt)?;
        let other_queue = settled(&other, COMPLETION);
        // The window the negative facts are false in.
        std::thread::sleep(HELD_FOR);
        let held_while_waiting = status(&held);
        let behind_while_waiting = status(&behind);
        let handlers_while_waiting = log.names();
        drop(raise);
        // The held command buffer's completed handler is the last of its handlers to run: once it
        // has, the log is whole and the status settled.
        log.wait_for(COMPLETED, COMPLETION);
        let handlers_at_the_end = log.names();
        let held_at_the_end = match handlers_at_the_end.iter().any(|name| name == COMPLETED) {
            true => status(&held),
            false => Status::NeverSettled,
        };
        Ok(HeldCommandBuffer {
            other_queue,
            held_while_waiting,
            behind_while_waiting,
            handlers_while_waiting,
            handlers_at_the_end,
            held_at_the_end,
            behind_at_the_end: settled(&behind, COMPLETION),
        })
    }

    /// [`crate::case::Side::event_wait_behind_a_running_signal`].
    pub(super) fn wait_behind_a_running_signal(
        &mut self,
        shared: bool,
    ) -> Answer<WaitBehindRunningSignal> {
        let device = &self.on.device;
        let (signalling_queue, waiting_queue) = (self.own_queue()?, self.own_queue()?);
        let (event, shared_event) = match shared {
            true => {
                let made = device.new_shared_event().map_err(of_rt)?;
                (made.as_event().clone(), Some(made))
            }
            false => (device.new_event().map_err(of_rt)?, None),
        };
        let event = &event;
        // The companion the signaller signals just before the event: still 0 when read, last, it
        // says the event's signal had not landed when the rest was read.
        let landed = device.new_shared_event().map_err(of_rt)?;
        let long = self.scratch(rt::MemoryClass::Private, LONG_FILL_BYTES)?;
        let scratch = self.scratch(rt::MemoryClass::Shared, SCRATCH)?;
        let signalling = signalling_queue.command_buffer().map_err(of_rt)?;
        let mut blit = signalling.blit_command_encoder().map_err(of_rt)?;
        for at in 0..LONG_FILLS {
            (blit.fill_buffer(&long, 0..LONG_FILL_BYTES, at as u8)).map_err(of_rt)?;
        }
        blit.end_encoding();
        (signalling.encode_signal_event(landed.as_event(), 1)).map_err(of_rt)?;
        signalling.encode_signal_event(event, 1).map_err(of_rt)?;
        let waiting = waiting_queue.command_buffer().map_err(of_rt)?;
        waiting.encode_wait_for_event(event, 1).map_err(of_rt)?;
        fill(&waiting, &scratch, 0..SCRATCH, WAITING_BYTE)?;
        let log = HandlerLog::new();
        (waiting.add_scheduled_handler(log.handler(SCHEDULED))).map_err(of_rt)?;
        (waiting.add_completed_handler(log.handler(COMPLETED))).map_err(of_rt)?;
        // The signaller first, so its signal is submitted ahead of the wait.
        signalling.commit().map_err(of_rt)?;
        waiting.commit().map_err(of_rt)?;
        std::thread::sleep(RUNNING_WINDOW);
        let waiting_while_it_runs = status(&waiting);
        let waiting_handlers_while_it_runs = log.names();
        let value_while_it_runs = shared_event.as_ref().map(rt::SharedEvent::signaled_value);
        let signalling_while_it_runs = status(&signalling);
        // Last: 0 says nothing above was read after the event's signal landed.
        let landed_before_the_reading = landed.signaled_value() != 0;
        log.wait_for(COMPLETED, COMPLETION);
        let waiting_handlers_at_the_end = log.names();
        let waiting_at_the_end = match waiting_handlers_at_the_end.iter().any(|n| n == COMPLETED) {
            true => status(&waiting),
            false => Status::NeverSettled,
        };
        Ok(WaitBehindRunningSignal {
            landed_before_the_reading,
            waiting_while_it_runs,
            waiting_handlers_while_it_runs,
            value_while_it_runs,
            signalling_while_it_runs,
            waiting_handlers_at_the_end,
            waiting_at_the_end,
            signalling_at_the_end: settled(&signalling, COMPLETION),
            bytes: read(&scratch, SCRATCH)?,
        })
    }

    /// [`crate::case::Side::signal_before_a_wait`].
    pub(super) fn signal_before_wait(&mut self) -> Answer<SignalBeforeWait> {
        let queue = self.own_queue()?;
        let event = || self.on.device.new_shared_event().map_err(of_rt);
        let (before, gate, after) = (event()?, event()?, event()?);
        let scratch = self.scratch(rt::MemoryClass::Private, SCRATCH)?;
        let command_buffer = queue.command_buffer().map_err(of_rt)?;
        fill(&command_buffer, &scratch, 0..SCRATCH, 1)?;
        (command_buffer.encode_signal_event(before.as_event(), 1)).map_err(of_rt)?;
        (command_buffer.encode_wait_for_event(gate.as_event(), 1)).map_err(of_rt)?;
        fill(&command_buffer, &scratch, 0..SCRATCH, 2)?;
        (command_buffer.encode_signal_event(after.as_event(), 1)).map_err(of_rt)?;
        command_buffer.commit().map_err(of_rt)?;
        let raise = Raise(&gate);
        let before_landed = (before.wait_until_signaled_value(1, SIGNAL_WAIT)).map_err(of_rt)?;
        // The window the two negative facts are false in.
        std::thread::sleep(HELD_FOR);
        let after_while_pending = after.signaled_value();
        let status_while_pending = status(&command_buffer);
        drop(raise);
        let status_at_the_end = settled(&command_buffer, COMPLETION);
        Ok(SignalBeforeWait {
            before_landed,
            after_while_pending,
            status_while_pending,
            after_at_the_end: after.signaled_value(),
            status_at_the_end,
        })
    }

    /// [`crate::case::Side::scheduled_handler_order`]. The handler is held inside itself, bounded
    /// by [`HOLD_LIMIT`], while the probe reads the status and a `wait_until_scheduled` runs on a
    /// thread of its own, which a wait that never returns leaves behind.
    pub(super) fn scheduled_handler_order_probe(&mut self) -> Answer<ScheduledHandlerOrder> {
        let queue = self.own_queue()?;
        let scratch = self.scratch(rt::MemoryClass::Private, SCRATCH)?;
        let command_buffer = queue.command_buffer().map_err(of_rt)?;
        fill(&command_buffer, &scratch, 0..SCRATCH, 1)?;
        let order: Arc<Mutex<Vec<String>>> = Arc::default();
        let (entered, entering) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let in_order = Arc::clone(&order);
        (command_buffer.add_scheduled_handler(move |own: &rt::CommandBuffer| {
            if let Ok(mut order) = in_order.lock() {
                order.push(SCHEDULED.to_owned());
            }
            let _ = entered.send(format!("{:?}", own.status()));
            let _ = released.recv_timeout(HOLD_LIMIT);
        }))
        .map_err(of_rt)?;
        let (done, completing) = mpsc::channel();
        let in_order = Arc::clone(&order);
        (command_buffer.add_completed_handler(move |own: &rt::CommandBuffer| {
            if let Ok(mut order) = in_order.lock() {
                order.push(COMPLETED.to_owned());
            }
            let _ = done.send(format!("{:?}", own.status()));
        }))
        .map_err(of_rt)?;
        command_buffer.commit().map_err(of_rt)?;
        let inside_the_handler =
            (entering.recv_timeout(ARRIVAL)).map_or(Status::NeverSettled, Status::read);
        let while_the_handler_ran = status(&command_buffer);
        let returned = Arc::new(AtomicBool::new(false));
        let (answered, answer) = mpsc::channel();
        let (waited, in_wait) = (command_buffer.clone(), Arc::clone(&returned));
        std::thread::spawn(move || {
            let answer = waited.wait_until_scheduled();
            in_wait.store(true, Ordering::SeqCst);
            let _ = answered.send(answer.err().map(of_rt));
        });
        // The window the negative fact is false in.
        std::thread::sleep(WINDOW);
        let wait_returned_while_the_handler_ran = returned.load(Ordering::SeqCst);
        let _ = release.send(());
        let wait_answered = match answer.recv_timeout(ARRIVAL) {
            Ok(None) => Returned::Ok,
            Ok(Some(failure)) => Returned::Err(failure.recorded()),
            Err(_) => Returned::NotWithinTheWindow,
        };
        let inside_the_completed_handler =
            (completing.recv_timeout(ARRIVAL)).map_or(Status::NeverSettled, Status::read);
        Ok(ScheduledHandlerOrder {
            inside_the_handler,
            while_the_handler_ran,
            wait_returned_while_the_handler_ran,
            wait_answered,
            inside_the_completed_handler,
            handler_order: order.lock().map(|order| order.clone()).unwrap_or_default(),
        })
    }
}
