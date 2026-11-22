//! Pieces, under a device cfg on every device of the cfg's kind: a command buffer's compute work
//! goes to the device's `VkQueue` in pieces, each expected to run for at most `GROUP_RUN`
//! (`crate::piece`), each a submission that waits for the device's room. A kernel's rate here is
//! the test's own input (`Rate::set_for_test`), set before each recording, so where a piece ends
//! does not depend on how fast the device runs. Behind the busy gate (`Busy::new`) nothing
//! submitted completes until the test opens it, so the device holds the two pieces its room
//! takes and the rest wait: a command buffer of another queue goes between two pieces, one that
//! shares a resource or a fence with a cut command buffer goes after its last, the queue's own
//! next waits for every piece, a cut command buffer is scheduled at its first piece and kept by an
//! abandon, and an idle queue's commit waits for no more than the two pieces on the device
//! behind five queues' backlogs. Without the gate: the cuts counted, a kernel of unknown rate in
//! a piece of its own whose run sets its rate, a dispatch past the bound uncut, a blit encoder
//! after a full piece cut, and a command buffer of one piece, committed behind another queue's
//! long one, taking a rate that holds none of the long one's run.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{
        each_device, fill, finished, shared_event, within, word, Arena, Busy, BODY_RUN,
    };
    use crate::tests::compute::on_device::{add, step};
    use crate::tests::gpu_span::on_device::{dispatches_for, long, words, WORDS};
    use crate::{
        Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputeCommandEncoder,
        ComputePipelineState, Device, DispatchType,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A kernel's rate of 3 ms a threadgroup, in picoseconds: one dispatch of one threadgroup
    /// fits a piece of 4 ms (`GROUP_RUN`), two do not, and the device's room takes two such
    /// pieces, not three (`submit::room`).
    const RATE: u64 = 3_000_000_000;

    fn queue(device: &Device) -> CommandQueue {
        device.new_command_queue(64).expect("a queue")
    }

    /// A compute encoder of `count` dispatches of `state` (`x = x * 2 + 1`), each of one
    /// threadgroup, over `buffer`, left open.
    fn steps(
        command_buffer: &CommandBuffer,
        state: &ComputePipelineState,
        buffer: &Buffer,
        count: u32,
    ) -> ComputeCommandEncoder {
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(state).expect("a state");
        encoder.set_buffer(Some(buffer), 0, 0).expect("a buffer");
        for _ in 0..count {
            (encoder.dispatch_threadgroups([1, 1, 1], [64, 1, 1])).expect("a dispatch");
        }
        encoder
    }

    /// A command buffer of `queue` with `count` steps over `buffer`, `state`'s rate set to
    /// `rate` first, committed.
    fn cut(
        queue: &CommandQueue,
        state: &ComputePipelineState,
        rate: u64,
        buffer: &Buffer,
        count: u32,
    ) -> CommandBuffer {
        state.inner.compiled.rate.set_for_test(rate);
        let command_buffer = queue.command_buffer().expect("a command buffer");
        steps(&command_buffer, state, buffer, count).end_encoding();
        command_buffer.commit().expect("a commit");
        command_buffer
    }

    /// The word `count` steps leave in a word that held 0.
    fn stepped(count: u32) -> u32 {
        (1 << count) - 1
    }

    /// Of the queues' submissions so far, in order, which were `of`'s queue's.
    fn order(device: &Device, of: &CommandBuffer) -> Vec<bool> {
        device.core().work.submitted_by(of.queue())
    }

    /// The queues' submissions so far.
    fn sent(device: &Device, queue: &CommandQueue) -> usize {
        order(device, &queue.command_buffer().expect("a command buffer")).len()
    }

    /// Waits for `command_buffer` to complete, as Metal's would, with no error.
    fn completed(name: &str, command_buffer: &CommandBuffer) {
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// The cuts: at 3 ms a threadgroup six dispatches go in six pieces, five cuts, each its own
    /// submission; at 1.5 ms, two to a piece, three pieces and two cuts. Every step lands.
    #[test]
    fn every_device_cuts_a_command_buffer_into_pieces_of_the_bound() {
        each_device("pieces_cuts", |name, _, device| {
            let mut arena = Arena::new(&device);
            let state = step(&device);
            let queue = queue(&device);
            for (rate, pieces) in [(RATE, 6), (RATE / 2, 3)] {
                let buffer = arena.buffer();
                let (from, cuts) = (sent(&device, &queue), device.counters().metrics().cuts);
                let command_buffer = cut(&queue, &state, rate, &buffer, 6);
                completed(name, &command_buffer);
                assert_eq!(
                    (
                        device.counters().metrics().cuts - cuts,
                        order(&device, &command_buffer).len() - from,
                        word(&buffer),
                    ),
                    (pieces - 1, pieces as usize, stepped(6)),
                    "{name}: at {rate} ps a threadgroup, the cuts, the submissions and the word"
                );
            }
        });
    }

    /// A command buffer of another queue goes between two pieces: A's command buffer of four
    /// pieces has two on the device, the room's two, and B's fill, committed to an idle queue,
    /// goes before A's third, counted once (`Metrics::pieces_passed`). Every step and the fill
    /// land.
    #[test]
    fn every_device_runs_another_queues_command_buffer_between_two_pieces() {
        each_device("pieces_passed", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (stepped_buffer, filled) = (arena.buffer(), arena.buffer());
            let state = step(&device);
            let (a, b) = (queue(&device), queue(&device));
            let busy = Busy::new(&device);
            let from = sent(&device, &a);
            let pieces = cut(&a, &state, RATE, &stepped_buffer, 4);
            within(name, || order(&device, &pieces).len() == from + 2);
            let passed = device.counters().metrics().pieces_passed;
            let passer = b.command_buffer().expect("a command buffer");
            fill(&passer, &filled, 7);
            passer.commit().expect("a commit");
            within(name, || passer.status() != CommandBufferStatus::Committed);
            busy.open();
            completed(name, &pieces);
            completed(name, &passer);
            assert_eq!(
                order(&device, &passer)[from..],
                [false, false, true, false, false],
                "{name}: A's two pieces, B's fill, A's last two"
            );
            assert_eq!(
                (
                    device.counters().metrics().pieces_passed - passed,
                    word(&stepped_buffer),
                    word(&filled),
                ),
                (1, stepped(4), 7),
                "{name}: the passes, the steps and the fill"
            );
        });
    }

    /// A command buffer of another queue that reads what a cut command buffer committed before
    /// it writes goes after its last piece, held once (`Metrics::shared_holds`), though its queue
    /// is idle, and reads every step.
    #[test]
    fn every_device_runs_a_read_of_a_cut_command_buffers_write_after_its_last_piece() {
        each_device("pieces_shared", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (written, read) = (arena.buffer(), arena.buffer());
            let state = step(&device);
            let (a, b) = (queue(&device), queue(&device));
            let busy = Busy::new(&device);
            let from = sent(&device, &a);
            let pieces = cut(&a, &state, RATE, &written, 4);
            within(name, || order(&device, &pieces).len() == from + 2);
            let holds = device.counters().metrics().shared_holds;
            let reader = b.command_buffer().expect("a command buffer");
            let mut blit = reader.blit_command_encoder().expect("an encoder");
            (blit.copy_from_buffer_to_buffer(&written, 0, &read, 0, 256)).expect("a copy");
            blit.end_encoding();
            reader.commit().expect("a commit");
            within(name, || {
                device.counters().metrics().shared_holds == holds + 1
            });
            busy.open();
            completed(name, &pieces);
            completed(name, &reader);
            assert_eq!(
                order(&device, &reader)[from..],
                [false, false, false, false, true],
                "{name}: A's four pieces, then B's read"
            );
            assert_eq!(word(&read), stepped(4), "{name}: the read sees every step");
        });
    }

    /// A queue's next command buffer waits for every piece of its cut one: its copy reads every
    /// step.
    #[test]
    fn every_device_runs_a_queues_next_command_buffer_after_every_piece() {
        each_device("pieces_own_order", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (written, read) = (arena.buffer(), arena.buffer());
            let state = step(&device);
            let a = queue(&device);
            let busy = Busy::new(&device);
            let from = sent(&device, &a);
            let pieces = cut(&a, &state, RATE, &written, 4);
            within(name, || order(&device, &pieces).len() == from + 2);
            let next = a.command_buffer().expect("a command buffer");
            let mut blit = next.blit_command_encoder().expect("an encoder");
            (blit.copy_from_buffer_to_buffer(&written, 0, &read, 0, 256)).expect("a copy");
            blit.end_encoding();
            next.commit().expect("a commit");
            busy.open();
            completed(name, &pieces);
            completed(name, &next);
            assert_eq!(
                (order(&device, &next).len() - from, word(&read)),
                (5, stepped(4)),
                "{name}: four pieces and the copy, which sees every step"
            );
        });
    }

    /// A wait on another queue for a fence a cut command buffer updates after its last
    /// dispatch goes after that last piece, held once (`Metrics::shared_holds`), though it
    /// shares no buffer with it.
    #[test]
    fn every_device_runs_a_fence_wait_after_the_last_piece_of_its_update() {
        each_device("pieces_fence", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (written, filled) = (arena.buffer(), arena.buffer());
            let state = step(&device);
            let fence = device.new_fence();
            let (a, b) = (queue(&device), queue(&device));
            let busy = Busy::new(&device);
            state.inner.compiled.rate.set_for_test(RATE);
            let updater = a.command_buffer().expect("a command buffer");
            let mut encoder = steps(&updater, &state, &written, 4);
            encoder.update_fence(&fence).expect("an update");
            encoder.end_encoding();
            let from = order(&device, &updater).len();
            updater.commit().expect("a commit");
            within(name, || order(&device, &updater).len() == from + 2);
            let holds = device.counters().metrics().shared_holds;
            let waiter = b.command_buffer().expect("a command buffer");
            let mut blit = waiter.blit_command_encoder().expect("an encoder");
            blit.wait_for_fence(&fence).expect("a wait");
            (blit.fill_buffer(&filled, 0..4, 9)).expect("a fill");
            blit.end_encoding();
            waiter.commit().expect("a commit");
            within(name, || {
                device.counters().metrics().shared_holds == holds + 1
            });
            busy.open();
            completed(name, &updater);
            completed(name, &waiter);
            assert_eq!(
                (order(&device, &waiter)[from..].to_vec(), word(&filled)),
                (vec![false, false, false, false, true], 0x0909_0909),
                "{name}: A's four pieces, then B's wait and fill"
            );
        });
    }

    /// The dispatches of a kernel of unknown rate share a piece of their own: three of one such
    /// kernel and then one of another make two pieces, one cut, and each piece's run sets its
    /// kernel's rate. A dispatch past the bound is not cut, and a second after it is.
    #[test]
    fn every_device_puts_a_kernel_of_unknown_rate_in_a_piece_of_its_own() {
        each_device("pieces_unknown_rate", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (x, a, b, o) = (
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
            );
            let (stepping, adding) = (step(&device), add(&device));
            let rate = |state: &ComputePipelineState| state.inner.compiled.rate.expected(1);
            assert_eq!((rate(&stepping), rate(&adding)), (None, None), "{name}");
            let queue = queue(&device);
            let cuts = || device.counters().metrics().cuts;
            let before = cuts();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = steps(&command_buffer, &stepping, &x, 3);
            encoder
                .set_compute_pipeline_state(&adding)
                .expect("a state");
            for (slot, buffer) in [&a, &b, &o].into_iter().enumerate() {
                (encoder.set_buffer(Some(buffer), 0, slot as u32)).expect("a buffer");
            }
            (encoder.dispatch_threadgroups([1, 1, 1], [64, 1, 1])).expect("a dispatch");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            completed(name, &command_buffer);
            assert_eq!(
                cuts() - before,
                1,
                "{name}: the steps' piece, then the add's"
            );
            assert!(
                rate(&stepping).is_some() && rate(&adding).is_some(),
                "{name}: each kernel's rate set by its piece's run"
            );
            for (count, cut_before) in [(1, 0), (2, 1)] {
                let before = cuts();
                completed(name, &cut(&queue, &stepping, 5 * RATE, &x, count));
                assert_eq!(cuts() - before, cut_before, "{name}: {count} of 15 ms");
            }
            assert_eq!(word(&x), stepped(6), "{name}");
        });
    }

    /// A blit encoder that opens after a piece past the bound begins the next piece, and one
    /// after a piece within it does not.
    #[test]
    fn every_device_cuts_before_a_blit_encoder_after_a_full_piece() {
        each_device("pieces_blit", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (x, filled) = (arena.buffer(), arena.buffer());
            let state = step(&device);
            let queue = queue(&device);
            for (rate, cuts) in [(5 * RATE / 3, 1), (RATE / 3, 0)] {
                state.inner.compiled.rate.set_for_test(rate);
                let before = device.counters().metrics().cuts;
                let command_buffer = queue.command_buffer().expect("a command buffer");
                steps(&command_buffer, &state, &x, 1).end_encoding();
                let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
                (blit.fill_buffer(&filled, 0..4, 5)).expect("a fill");
                blit.end_encoding();
                command_buffer.commit().expect("a commit");
                completed(name, &command_buffer);
                assert_eq!(
                    device.counters().metrics().cuts - before,
                    cuts,
                    "{name}: after a piece of {rate} ps"
                );
            }
            assert_eq!(
                (word(&x), word(&filled)),
                (stepped(2), 0x0505_0505),
                "{name}"
            );
        });
    }

    /// A cut command buffer is scheduled as its first piece is staged, its scheduled handler
    /// run, while two of its four pieces wait for the device's room.
    #[test]
    fn every_device_schedules_a_cut_command_buffer_at_its_first_piece() {
        each_device("pieces_scheduled", |name, _, device| {
            let mut arena = Arena::new(&device);
            let x = arena.buffer();
            let state = step(&device);
            let queue = queue(&device);
            let busy = Busy::new(&device);
            state.inner.compiled.rate.set_for_test(RATE);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            steps(&command_buffer, &state, &x, 4).end_encoding();
            let scheduled = Arc::new(AtomicUsize::new(0));
            let counted = Arc::clone(&scheduled);
            (command_buffer.add_scheduled_handler(move |_| {
                counted.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("a handler");
            let from = order(&device, &command_buffer).len();
            command_buffer.commit().expect("a commit");
            within(name, || order(&device, &command_buffer).len() == from + 2);
            within(name, || scheduled.load(Ordering::SeqCst) == 1);
            assert_eq!(
                (
                    command_buffer.status(),
                    order(&device, &command_buffer).len() - from
                ),
                (CommandBufferStatus::Scheduled, 2),
                "{name}: scheduled, with two pieces submitted and two waiting for room"
            );
            busy.open();
            completed(name, &command_buffer);
            assert_eq!(word(&x), stepped(4), "{name}");
        });
    }

    /// An abandon of a queue whose cut command buffer is scheduled with pieces still waiting for
    /// room ends nothing: every piece runs, and the command buffer completes.
    #[test]
    fn every_device_keeps_a_scheduled_cut_command_buffer_through_an_abandon() {
        each_device("pieces_abandon", |name, _, device| {
            let mut arena = Arena::new(&device);
            let x = arena.buffer();
            let state = step(&device);
            let queue = queue(&device);
            let busy = Busy::new(&device);
            let command_buffer = cut(&queue, &state, RATE, &x, 4);
            within(name, || {
                command_buffer.status() == CommandBufferStatus::Scheduled
            });
            queue.abandon();
            busy.open();
            completed(name, &command_buffer);
            assert_eq!(word(&x), stepped(4), "{name}: every piece ran");
        });
    }

    /// The shape of five queues with backlogs and a sixth, idle, committing one command buffer.
    /// Each of the five submits a fill, the first piece of a cut command buffer whose next waits
    /// for an event's value, and holds behind it a second of four pieces. The value set, the
    /// device takes two pieces, its room, and the sixth queue, which last submitted before all of
    /// them and expects a run of 2 ms, too long to go behind two pieces of 3 ms, commits a fill.
    /// It goes at the device's next room, before any backlog's next piece, though the three queues
    /// with nothing in flight hold command buffers committed before it: the pieces submitted
    /// ahead of it are the two on the device when it committed. Every step lands.
    #[test]
    fn every_device_runs_an_idle_queues_commit_behind_two_pieces_of_five_backlogs() {
        each_device("pieces_backlogs", |name, _, device| {
            let mut arena = Arena::new(&device);
            let state = step(&device);
            let event = shared_event(&device);
            let queues: Vec<CommandQueue> = (0..5).map(|_| queue(&device)).collect();
            let buffers: Vec<Buffer> = (0..5).map(|_| arena.buffer()).collect();
            let (idle, filled) = (queue(&device), arena.buffer());
            let warm = idle.command_buffer().expect("a command buffer");
            fill(&warm, &filled, 0);
            warm.commit().expect("a commit");
            completed(name, &warm);
            warm.queue().set_run_for_test(2 * BODY_RUN);
            state.inner.compiled.rate.set_for_test(RATE);
            let base = order(&device, &warm).len();
            let mut backlogs = Vec::new();
            for (queue, buffer) in queues.iter().zip(&buffers) {
                let head = queue.command_buffer().expect("a command buffer");
                fill(&head, buffer, 0);
                (head.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
                steps(&head, &state, buffer, 4).end_encoding();
                head.commit().expect("a commit");
                let next = queue.command_buffer().expect("a command buffer");
                steps(&next, &state, buffer, 4).end_encoding();
                next.commit().expect("a commit");
                backlogs.extend([head, next]);
            }
            let work = &device.core().work;
            within(name, || {
                order(&device, &warm).len() == base + 5
                    && (backlogs.iter()).all(|backlog| work.in_flight(backlog.queue()).0 == 0)
            });
            let busy = Busy::new(&device);
            let from = base + 5;
            event.set_signaled_value(1);
            within(name, || order(&device, &warm).len() == from + 2);
            let late = idle.command_buffer().expect("a command buffer");
            fill(&late, &filled, 7);
            late.commit().expect("a commit");
            within(name, || late.status() != CommandBufferStatus::Committed);
            busy.open();
            completed(name, &late);
            for command_buffer in &backlogs {
                completed(name, command_buffer);
            }
            let submitted = order(&device, &late);
            let ahead = submitted[from..].iter().position(|&late| late);
            println!("{name}: {ahead:?} pieces submitted ahead of the idle queue's fill");
            assert_eq!(
                ahead,
                Some(2),
                "{name}: the pieces submitted ahead of the fill"
            );
            assert_eq!(
                submitted.len() - from,
                41,
                "{name}: forty pieces and the fill"
            );
            for buffer in &buffers {
                assert_eq!(word(buffer), stepped(8), "{name}");
            }
            assert_eq!(word(&filled), 7, "{name}");
        });
    }

    /// A command buffer of one piece takes no pair of timestamps past its own, and its kernel's
    /// rate from its own two, which hold no other submission's run: one dispatch of `add`
    /// committed to a queue right behind another queue's command buffer of about 20 ms of
    /// dispatches, uncut (its kernel's rate the test's own), submitted while the long one is on
    /// the device (its submission seen before the long one's GPU end, and its own end after it),
    /// takes a rate under twice the rate it takes alone. The ratio is printed. A start written
    /// before the long one ends would put its tail in the rate.
    #[test]
    fn every_device_rates_a_one_piece_command_buffer_without_the_run_ahead_of_it() {
        each_device("pieces_rate_behind", |name, _, device| {
            let (long_state, add) = (step(&device), add(&device));
            let words = words(&device);
            let mut arena = Arena::new(&device);
            let slots = [arena.buffer(), arena.buffer(), arena.buffer()];
            let (first, second) = (queue(&device), queue(&device));
            let count = dispatches_for(name, &first, &long_state, &words, 20_000_000);
            // One dispatch of `add` on the second queue, its rate unknown, so its run sets it.
            let one = || {
                add.inner.compiled.rate.set_for_test(u64::MAX);
                let command_buffer = second.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&add).expect("a state");
                for (index, buffer) in (0..).zip(&slots) {
                    encoder
                        .set_buffer(Some(buffer), 0, index)
                        .expect("a buffer");
                }
                (encoder.dispatch_threadgroups([1, 1, 1], [64, 1, 1])).expect("a dispatch");
                encoder.end_encoding();
                command_buffer
            };
            let rate = || (add.inner.compiled.rate.expected(1000)).expect("a rate");
            // The first meets every first use; the second is the rate alone.
            for _ in 0..2 {
                let alone = one();
                alone.commit().expect("a commit");
                completed(name, &alone);
            }
            let alone = rate();
            // Recorded, it holds its own pair of timestamps alone (`Building::stamp_piece`).
            let stamps = &device.core().stamps;
            let out = stamps.out_for_test();
            let recorded = one();
            assert_eq!(
                stamps.out_for_test() - out,
                1,
                "{name}: the pairs a command buffer of one piece takes"
            );
            recorded.commit().expect("a commit");
            completed(name, &recorded);
            long_state.inner.compiled.rate.set_for_test(1);
            let ahead = long(&first, &long_state, &words, count, WORDS);
            let behind = one();
            let second_sent = || {
                order(&device, &behind)
                    .into_iter()
                    .filter(|&own| own)
                    .count()
            };
            let before = second_sent();
            ahead.commit().expect("a commit");
            behind.commit().expect("a commit");
            assert_ne!(
                ahead.status(),
                CommandBufferStatus::Completed,
                "{name}: the long command buffer still runs as the short one is committed"
            );
            // The short one's submission, seen on the host clock no earlier than it went.
            within(name, || second_sent() > before);
            let sent_at = crate::host_time();
            completed(name, &ahead);
            completed(name, &behind);
            let (start, end) = (ahead.gpu_start_time(), ahead.gpu_end_time());
            assert!(
                sent_at < end && behind.gpu_end_time() >= end,
                "{name}: the short one, submitted at {sent_at}, behind the long one's \
                 {start}..{end}: it ends at {}",
                behind.gpu_end_time()
            );
            let behind_rate = rate();
            let ratio = behind_rate as f64 / alone.max(1) as f64;
            println!(
                "pieces_rate_behind: {name}: alone {alone} ps a threadgroup, behind a {} us \
                 command buffer, submitted {} us before its GPU end, {behind_rate} ps: ratio \
                 {ratio:.2}",
                (end - start.min(end)) / 1000,
                (end - sent_at) / 1000
            );
            assert!(
                behind_rate < alone.saturating_mul(2),
                "{name}: the rate behind the long command buffer, {behind_rate} ps, against \
                 {alone} ps alone"
            );
        });
    }
}
