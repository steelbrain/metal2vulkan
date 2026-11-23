//! The parameter ring, on every device of the cfg's kind: the block-maker thread makes the next
//! block once a take leaves fewer chunks free than a block holds, without the ring's lock, so a
//! give-back on another thread goes on meanwhile; a take that finds no chunk free while it makes
//! the block waits for it, or for a chunk given back, counted as a wait for the block-maker
//! thread; no block is made on a caller's thread; and chunks given back are taken again before
//! any further block. Where the block-maker thread makes none, a take that makes a block makes it
//! without the ring's lock, and takes that meet while it is made make one block between them. An
//! ask made once the ring has the block-maker thread's answer to the last, before the thread is
//! done with it, is answered too. Every wait of these tests is bounded: a lost step fails with
//! its message.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::{lock, Core};
    use crate::memory::MakerTest;
    use crate::ring::{Writes, CHUNK, SPARE};
    use crate::tests::command::on_device::{
        each_device, each_device_with, within, Release, PATIENCE,
    };
    use crate::DeviceOptions;
    use std::sync::{mpsc, Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    /// How long a take that waits is given before the test looks.
    const HOLD: Duration = Duration::from_millis(250);

    /// `chunks` chunks taken: each write of a chunk's bytes takes one of its own.
    fn taken(core: &Core, chunks: usize) -> Writes {
        let bytes = vec![7u8; CHUNK as usize];
        let mut writes = Writes::default();
        for _ in 0..chunks {
            writes.write(core, &bytes).expect("a ring write");
        }
        writes
    }

    /// One chunk taken on a thread of its own.
    fn take_one(core: &Arc<Core>) -> thread::JoinHandle<Writes> {
        let core = Arc::clone(core);
        thread::spawn(move || taken(&core, 1))
    }

    /// What `thread` returned, once it has within the tests' patience.
    fn joined(what: &str, thread: thread::JoinHandle<Writes>) -> Writes {
        within(what, || thread.is_finished());
        thread.join().unwrap_or_else(|_| panic!("{what} panicked"))
    }

    /// The test's first meeting with the thread parked at `park`, which comes within the tests'
    /// patience, and the release of that thread. The meeting waits on a thread of its own, so a
    /// thread that never parks fails the test rather than hanging it.
    fn met(what: &str, park: &Arc<Barrier>) -> Release {
        let (done, meeting) = mpsc::channel();
        let at = Arc::clone(park);
        thread::spawn(move || {
            at.wait();
            let _ = done.send(());
        });
        let meeting = meeting.recv_timeout(PATIENCE);
        meeting.unwrap_or_else(|_| panic!("timed out waiting for {what}"));
        Release::of(park)
    }

    /// The ring's blocks and free chunks, its lock free.
    fn chunks(core: &Core) -> (usize, usize) {
        core.ring.chunks().expect("the ring's lock is free")
    }

    /// The ring's blocks and free chunks once the block-maker thread has answered every ask. Where
    /// it makes none, its answer to a take's ask (`Ring::unasked`) takes the ring's lock on its own
    /// thread, at whatever time the host's scheduler runs it, so a read before that answer may
    /// find the lock held.
    fn settled(core: &Core) -> (usize, usize) {
        core.pool.await_internals();
        chunks(core)
    }

    /// The options of a device whose block-maker thread makes no internal.
    fn making_none() -> DeviceOptions {
        DeviceOptions {
            maker: MakerTest::Fail,
            ..DeviceOptions::default()
        }
    }

    #[test]
    fn the_block_maker_thread_makes_the_ring_s_next_block_ahead_of_the_takes() {
        each_device("ring_ahead", |name, _, device| {
            let core = device.core();
            let ring = &core.ring;
            let counters = device.counters();
            assert_eq!(
                chunks(core),
                (1, SPARE),
                "{name}: the first block, every chunk free"
            );
            // The take that leaves fewer than a block's chunks free asks for the next block; the
            // block-maker thread is parked where it makes it.
            let park = Arc::new(Barrier::new(2));
            *lock(&ring.park) = Some(Arc::clone(&park));
            let first = taken(core, 1);
            let mut release = met("the block-maker thread's make", &park);
            *lock(&ring.park) = None;
            assert_eq!(
                ring.chunks(),
                Some((1, SPARE - 1)),
                "{name}: the ring's lock is held while a block is made"
            );
            // Every free chunk taken, one of them apart to give back while the block is made.
            let held = taken(core, SPARE - 2);
            let spare = taken(core, 1);
            assert_eq!(chunks(core), (1, 0), "{name}: every chunk taken");
            let waits = counters.metrics().maker_waits;
            let waiter = take_one(core);
            thread::sleep(HOLD);
            assert!(
                !waiter.is_finished(),
                "{name}: a take with no chunk free went on while the block is made"
            );
            // A give-back on this thread goes on while the block is made, and serves the take.
            spare.give_back(core);
            let waited = joined("the take served by the give-back", waiter);
            assert_eq!(
                counters.metrics().maker_waits,
                waits + 1,
                "{name}: the take's wait for the block-maker thread"
            );
            release.now();
            core.pool.await_internals();
            assert_eq!(chunks(core), (2, SPARE), "{name}: the block made");
            for writes in [first, held, waited] {
                writes.give_back(core);
            }
            // Chunks given back are taken again before any further block: takes that leave a
            // block's chunks free ask for none; the next take asks for one.
            let (blocks, free) = chunks(core);
            let again = taken(core, free - SPARE);
            core.pool.await_internals();
            assert_eq!(chunks(core), (blocks, SPARE), "{name}: no block asked");
            let past = taken(core, 1);
            core.pool.await_internals();
            assert_eq!(chunks(core).0, blocks + 1, "{name}: one block past");
            again.give_back(core);
            past.give_back(core);
            let site = counters.metrics().ring_block_allocations;
            assert_eq!(site.count, 0, "{name}: a block made on a caller's thread");
        });
    }

    #[test]
    fn where_the_block_maker_thread_makes_none_a_take_makes_the_block_off_the_lock() {
        each_device_with("ring_off_the_lock", making_none, |name, _, device| {
            let core = device.core();
            let ring = &core.ring;
            if settled(core).1 == 0 {
                taken(core, 1).give_back(core);
            }
            // Every free chunk taken, one of them apart to give back while a block is made.
            let (before, free) = settled(core);
            let held = taken(core, free - 1);
            let spare = taken(core, 1);
            assert_eq!(settled(core), (before, 0), "{name}: every chunk taken");
            let park = Arc::new(Barrier::new(2));
            *lock(&ring.park) = Some(Arc::clone(&park));
            let maker = take_one(core);
            // The take is parked where it makes the block.
            let mut release = met("the take that makes the block", &park);
            *lock(&ring.park) = None;
            assert_eq!(
                ring.chunks(),
                Some((before, 0)),
                "{name}: the ring's lock is held while a block is made"
            );
            let waiter = take_one(core);
            // A give-back on this thread goes on while the block is made.
            spare.give_back(core);
            release.now();
            let made = joined("the take that made the block", maker);
            let waited = joined("the take that met it", waiter);
            assert_eq!(
                settled(core).0,
                before + 1,
                "{name}: takes that met made one block between them"
            );
            for writes in [made, waited, held] {
                writes.give_back(core);
            }
            // A fill past the ring's chunks makes one block; chunks given back are taken again.
            let (blocks, free) = settled(core);
            let past = [
                taken(core, free),
                joined("a take past the ring's chunks", take_one(core)),
            ];
            assert_eq!(settled(core).0, blocks + 1, "{name}: one block past");
            for writes in past {
                writes.give_back(core);
            }
            let again = taken(core, settled(core).1);
            assert_eq!(
                settled(core),
                (blocks + 1, 0),
                "{name}: chunks given back are taken before a block is made"
            );
            again.give_back(core);
            let site = device.counters().metrics().ring_block_allocations;
            assert_eq!(
                site.count,
                (blocks + 1) as u64,
                "{name}: each block made on a caller's thread"
            );
        });
    }

    #[test]
    fn an_ask_made_once_the_ring_has_the_block_maker_thread_s_answer_is_answered() {
        each_device_with("ring_ask_in_answer", making_none, |name, _, device| {
            let core = device.core();
            let ring = &core.ring;
            if settled(core).1 == 0 {
                taken(core, 1).give_back(core);
            }
            let blocks = settled(core).0;
            assert!(!ring.asked(), "{name}: no ask waits");
            // The take that leaves fewer than a block's chunks free asks; the block-maker
            // thread's answer, that it makes none, is parked once the ring has it.
            let park = Arc::new(Barrier::new(2));
            *lock(&ring.park) = Some(Arc::clone(&park));
            let first = taken(core, 1);
            let mut release = met("the block-maker thread's answer", &park);
            *lock(&ring.park) = None;
            // The ring has the answer: the next take asks again, before the thread is done.
            let second = taken(core, 1);
            assert!(ring.asked(), "{name}: the second take asked");
            release.now();
            let free = settled(core).1;
            assert!(
                !ring.asked(),
                "{name}: the ask made while the block-maker thread answered the last went \
                 unanswered: a take that finds no chunk free waits for it with nothing coming"
            );
            // A take past the ring's chunks makes the block, on its own thread.
            let rest = taken(core, free);
            let past = joined("a take past the ring's chunks", take_one(core));
            assert_eq!(settled(core).0, blocks + 1, "{name}: one block past");
            for writes in [first, second, rest, past] {
                writes.give_back(core);
            }
        });
    }
}
