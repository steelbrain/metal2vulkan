//! The parameter ring, on every device of the cfg's kind: the block-maker thread makes the next
//! block once a take leaves fewer chunks free than a block holds, without the ring's lock, so a
//! give-back on another thread goes on meanwhile; a take that finds no chunk free while it makes
//! the block waits for it, or for a chunk given back, counted as a wait for the block-maker
//! thread; no block is made on a caller's thread; and chunks given back are taken again before
//! any further block. Where the block-maker thread makes none, a take that makes a block makes it
//! without the ring's lock, and takes that meet while it is made make one block between them.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::{lock, Core};
    use crate::memory::MakerTest;
    use crate::ring::{Writes, CHUNK, SPARE};
    use crate::tests::command::on_device::{each_device, each_device_with};
    use crate::DeviceOptions;
    use std::sync::{Arc, Barrier};
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

    /// The ring's blocks and free chunks, its lock free.
    fn chunks(core: &Core) -> (usize, usize) {
        core.ring.chunks().expect("the ring's lock is free")
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
            park.wait();
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
            let waited = waiter.join().expect("the waiter");
            assert_eq!(
                counters.metrics().maker_waits,
                waits + 1,
                "{name}: the take's wait for the block-maker thread"
            );
            park.wait();
            core.pool.settle();
            assert_eq!(chunks(core), (2, SPARE), "{name}: the block made");
            for writes in [first, held, waited] {
                writes.give_back(core);
            }
            // Chunks given back are taken again before any further block: takes that leave a
            // block's chunks free ask for none; the next take asks for one.
            let (blocks, free) = chunks(core);
            let again = taken(core, free - SPARE);
            core.pool.settle();
            assert_eq!(chunks(core), (blocks, SPARE), "{name}: no block asked");
            let past = taken(core, 1);
            core.pool.settle();
            assert_eq!(chunks(core).0, blocks + 1, "{name}: one block past");
            again.give_back(core);
            past.give_back(core);
            let site = counters.metrics().ring_block_allocations;
            assert_eq!(site.count, 0, "{name}: a block made on a caller's thread");
        });
    }

    #[test]
    fn where_the_block_maker_thread_makes_none_a_take_makes_the_block_off_the_lock() {
        let options = || DeviceOptions {
            maker: MakerTest::Fail,
            ..DeviceOptions::default()
        };
        each_device_with("ring_off_the_lock", options, |name, _, device| {
            let core = device.core();
            let ring = &core.ring;
            if chunks(core).1 == 0 {
                taken(core, 1).give_back(core);
            }
            // Every free chunk taken, one of them apart to give back while a block is made.
            let (before, free) = chunks(core);
            let held = taken(core, free - 1);
            let spare = taken(core, 1);
            assert_eq!(chunks(core), (before, 0), "{name}: every chunk taken");
            let park = Arc::new(Barrier::new(2));
            *lock(&ring.park) = Some(Arc::clone(&park));
            let maker = take_one(core);
            // The take is parked where it makes the block.
            park.wait();
            *lock(&ring.park) = None;
            assert_eq!(
                ring.chunks(),
                Some((before, 0)),
                "{name}: the ring's lock is held while a block is made"
            );
            let waiter = take_one(core);
            // A give-back on this thread goes on while the block is made.
            spare.give_back(core);
            park.wait();
            let made = maker.join().expect("the maker");
            let waited = waiter.join().expect("the waiter");
            assert_eq!(
                chunks(core).0,
                before + 1,
                "{name}: takes that met made one block between them"
            );
            for writes in [made, waited, held] {
                writes.give_back(core);
            }
            // A fill past the ring's chunks makes one block; chunks given back are taken again.
            let (blocks, free) = chunks(core);
            let past = taken(core, free + 1);
            assert_eq!(chunks(core).0, blocks + 1, "{name}: one block past");
            past.give_back(core);
            let again = taken(core, chunks(core).1);
            assert_eq!(
                chunks(core),
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
}
