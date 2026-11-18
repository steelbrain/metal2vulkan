//! The parameter ring, on every device of the cfg's kind: a take that makes a block makes it
//! without the ring's lock, so a give-back on another thread goes on meanwhile; takes that meet
//! while a block is being made make one block between them; and a fill past the ring's chunks
//! makes the next block, while chunks given back are taken again before any further block.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::{lock, Core};
    use crate::ring::{Writes, CHUNK};
    use crate::tests::command::on_device::each_device;
    use std::sync::{Arc, Barrier};
    use std::thread;

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
    fn a_ring_block_is_made_off_the_lock_and_once_for_takes_that_meet() {
        each_device("ring_off_the_lock", |name, _, device| {
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
            // The maker is parked where it makes the block.
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
        });
    }
}
