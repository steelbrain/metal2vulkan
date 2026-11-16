//! Device objects shared by their exact content: one object per distinct content, held by the
//! device's map and shared by every variant that uses it. Who destroys them, and through what, is
//! the rule.
//!
//! - **Exact.** An object is found by its whole content (the map hashes it, then compares it):
//!   two variants share one only where the content that made it is the same.
//! - **The device's, destroyed by the map alone**. An object keeps no device functions and has no
//!   `Drop` that reaches Vulkan: a holder's [`Share`] (a variant's objects, a kept instance's,
//!   `kept.rs`) is a count, never a destructor. Only the map destroys an object
//!   ([`Owned::destroy`], through the device it is handed, `DeviceShared::vk`), on one of three
//!   paths, each returning what it let go ([`Gone`]), destroyed once the map's lock is released:
//!   - [`ContentKeyed::settle`], in a program's call, once no share of the object is left;
//!   - [`ContentKeyed::sweep`], on any thread, for the objects of a host step's or a pool piece's
//!     own (below) that nothing else holds;
//!   - [`ContentKeyed::drain`], at the device's teardown (`DeviceShared::drop`), after the kept
//!     variants': everything left.
//!
//!   So no object is destroyed before the device's teardown while anything else shares it (a new
//!   share is taken from the map, under its lock, or cloned from a live one, so none appears
//!   after the check), and none after it: the map is the device's, so a destruction always has a
//!   live device. The check reads the object's count of shares, with an Acquire that orders every
//!   use a released holder made (a pool piece's `vkCreate*Pipelines` from a module, on its thread)
//!   before the destruction, as each share's Release drop publishes it; at the teardown the
//!   device's own last drop does (every holder's use came before it let go of its device, or of
//!   what held its share). A share left at the teardown is one whose holder is dropping its own
//!   `Device` right then (a variant whose last device handle goes before its fields), and a share
//!   is used only through its holder's device, which is gone: dropping it calls nothing. The
//!   order of a holder's fields therefore decides nothing (the earlier rule, a destroying `Drop`
//!   of the object's own, gave).
//! - **A settle visits only what was released.** The share whose drop leaves an object of the map
//!   with none queues the object's place in the order of insertion on its map's release queue (a
//!   leaf lock: nothing else is taken under it), once until a settle takes it. A settle takes the
//!   queue, and destroys each queued object that still has no share, checked under the map's lock,
//!   where alone a share can be taken anew; one a lookup took a share of since stays, and is queued
//!   again when that share goes. So a settle lets go of exactly the objects whose shares all went,
//!   which a walk of every object found, in the same order, at the same calls; but its work is the
//!   objects released since the last settle, not every object the device holds, and one with
//!   nothing released takes the queue's lock and finds it empty. The device's modules, layouts,
//!   sampler sets, graphics pipeline libraries and linked pipelines are all such maps
//!   (`objects.rs`, `shared_layouts.rs`, `libraries.rs`): a compute state's creation settles them
//!   all, the render libraries among them, and walks none.
//! - **On the program's threads, in the program's order** (the owner's HITCH rule 1). Only a
//!   program's call ([`program_call`]: a thread of the program's or one running its code, outside
//!   a host step and a pool piece) inserts or settles, each a debug assertion; a host step or a
//!   pool piece only looks up ([`ContentKeyed::get`]) and, on a miss, makes an object of its own
//!   that no lookup finds. Settling removes in the order of insertion. What remains timing's is
//!   only whether a command buffer let its variants go before a settle, as for the kept variants.
//! - **The exception: an object of a host step's own.** No lookup finds it, so it changes nothing
//!   another variant takes; the map keeps it apart (`own`) only to destroy it. For a variant a
//!   command buffer's host step made (a device-generated replay's, `generated_execute.rs`, which
//!   keeps no instance), the handler thread releases the variant's layouts and samplers as the
//!   command buffer releases its segment, and its own objects go at the next sweep or settle (the
//!   next variant made, on any thread) or at the teardown.

use crate::device::lock;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Whether this call is the program's own (the module docs): a thread of the program's, or a
/// runtime thread running the program's code (`exit_guard::in_program_call`), and neither a
/// command buffer's host step nor a pool piece. Only such a call changes a device's maps.
pub(crate) fn program_call() -> bool {
    crate::exit_guard::in_program_call()
        && !crate::exit_guard::in_host_step()
        && !super::pool::in_piece()
}

/// An object a device's map holds (the module docs): it keeps no device functions, and nothing
/// but the map destroys it, through its owner `C` (the device's shared state, `DeviceShared`).
pub(crate) trait Owned<C: ?Sized> {
    /// Destroy its Vulkan objects through `owner`: once, by the map, when nothing else shares it
    /// or at the device's teardown.
    fn destroy(&self, owner: &C);
}

/// A map's release queue (the module docs): the places in its order of insertion of the objects
/// whose last share went since its last settle, each once. Its lock is a leaf: a share's drop
/// takes it holding any other, and nothing is taken under it.
type Released = Mutex<BTreeSet<u64>>;

/// One object, with the count of its holders' shares.
struct Held<T> {
    object: T,
    /// The [`Share`]s of it that live.
    shares: AtomicUsize,
    /// Its place in its map's order of insertion.
    at: u64,
    /// The release queue of the map whose lookups find it; `None` for an object no lookup finds
    /// (a host step's own, which a sweep looks at, or one no map holds).
    released: Option<Arc<Released>>,
}

/// A holder's share of an object of a device's map (the module docs): a count the map reads,
/// never a destructor. Its drop that leaves an object of the map with no share queues it for the
/// map's next settle.
pub(crate) struct Share<T>(Arc<Held<T>>);

impl<T> Share<T> {
    /// A new share of `held`.
    fn of(held: &Arc<Held<T>>) -> Share<T> {
        held.shares.fetch_add(1, Ordering::Relaxed);
        Share(Arc::clone(held))
    }

    /// The share of `object`, which no map holds: dropping it destroys nothing, as there is
    /// nothing to destroy (a layout's empty sampler set).
    pub(crate) fn unmapped(object: T) -> Share<T> {
        let held = Held {
            object,
            shares: AtomicUsize::new(0),
            at: 0,
            released: None,
        };
        Share::of(&Arc::new(held))
    }
}

impl<T> Clone for Share<T> {
    fn clone(&self) -> Share<T> {
        // As `Arc::clone`: a live share holds the count above zero, so the new one needs no
        // ordering of its own.
        Share::of(&self.0)
    }
}

impl<T> Drop for Share<T> {
    fn drop(&mut self) {
        // Release: every use made through this share comes before the settle that reads the
        // count it leaves (its Acquire).
        if self.0.shares.fetch_sub(1, Ordering::Release) != 1 {
            return;
        }
        // The last share: queued for the map's settle, which checks the count again under the
        // queue's lock and the map's. A count read here as zero and taken back up by a lookup
        // before that settle leaves the object in the map.
        if let Some(released) = &self.0.released {
            lock(released).insert(self.0.at);
        }
    }
}

impl<T> std::ops::Deref for Share<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0.object
    }
}

/// The device's objects of one kind, by their content `K`.
pub(crate) struct ContentKeyed<K, T> {
    /// Each object found by lookups, by its content.
    entries: HashMap<K, Arc<Held<T>>>,
    /// Each such object's content, by its place in the order of insertion.
    keys: BTreeMap<u64, K>,
    /// The objects of a host step's or a pool piece's own, which no lookup finds, in the order
    /// they were made: held only until a sweep finds nothing else sharing them.
    own: Vec<Arc<Held<T>>>,
    /// The places of the objects whose last share went since the last settle.
    released: Arc<Released>,
    next: u64,
    /// The objects the settles looked at (`tests::content_keyed`'s check of a settle's work).
    #[cfg(test)]
    looked_at: u64,
}

impl<K, T> Default for ContentKeyed<K, T> {
    fn default() -> Self {
        ContentKeyed {
            entries: HashMap::new(),
            keys: BTreeMap::new(),
            own: Vec::new(),
            released: Arc::default(),
            next: 0,
            #[cfg(test)]
            looked_at: 0,
        }
    }
}

/// What a map let go of, in its order: destroyed through the owner once the map's lock is
/// released ([`Gone::destroy`]).
#[must_use = "the map let these go: destroy them through the device"]
pub(crate) struct Gone<T>(Vec<Arc<Held<T>>>);

impl<T> Gone<T> {
    /// Destroy each, in order, through `owner`.
    pub(crate) fn destroy<C: ?Sized>(self, owner: &C)
    where
        T: Owned<C>,
    {
        for held in &self.0 {
            held.object.destroy(owner);
        }
    }
}

impl<K: Hash + Eq + Clone, T> ContentKeyed<K, T> {
    /// A share of the object of `key`, if the map holds one.
    pub(crate) fn get(&self, key: &K) -> Option<Share<T>> {
        self.entries.get(key).map(Share::of)
    }

    /// Hold `object` as the one of `key` (a program's call made it; none was there): a share of
    /// it.
    fn insert(&mut self, key: K, object: T) -> Share<T> {
        debug_assert!(
            program_call(),
            "a device's map changed outside a program's call"
        );
        let at = self.next;
        self.next += 1;
        let held = Arc::new(Held {
            object,
            shares: AtomicUsize::new(0),
            at,
            released: Some(Arc::clone(&self.released)),
        });
        let share = Share::of(&held);
        self.keys.insert(at, key.clone());
        self.entries.insert(key, held);
        share
    }

    /// A share of the object of `key`, else of one `make` makes now (and whether it was made):
    /// found by later lookups where `program` (a program's call made it), else the caller's own
    /// (a host step's or a pool piece's), which the map holds apart until a sweep.
    pub(crate) fn get_or_make<E>(
        &mut self,
        key: K,
        program: bool,
        make: impl FnOnce() -> Result<T, E>,
    ) -> Result<(Share<T>, bool), E> {
        if let Some(object) = self.get(&key) {
            return Ok((object, false));
        }
        let object = make()?;
        let share = match program {
            true => self.insert(key, object),
            false => {
                let held = Arc::new(Held {
                    object,
                    shares: AtomicUsize::new(0),
                    at: 0,
                    released: None,
                });
                self.own.push(Arc::clone(&held));
                Share::of(&held)
            }
        };
        Ok((share, true))
    }

    /// Let go of every object found by lookups whose last share went since the last settle and
    /// that none was taken of since, in the order they were inserted, then of the objects of a
    /// host step's own that nothing holds: in a program's call. Only the queued objects are
    /// looked at (the module docs).
    pub(crate) fn settle(&mut self) -> Gone<T> {
        debug_assert!(
            program_call(),
            "a device's map settled outside a program's call"
        );
        let mut gone = Vec::new();
        {
            // Held while the counts are read: a share whose drop leaves none after this read
            // queues its object again once the lock is free, for the next settle; one that left
            // none before it is read as none here.
            let mut released = lock(&self.released);
            while let Some(at) = released.pop_first() {
                #[cfg(test)]
                {
                    self.looked_at += 1;
                }
                // Gone already: queued again after a settle took it, or drained.
                let Some(key) = self.keys.get(&at) else {
                    continue;
                };
                let Some(held) = self.entries.get(key) else {
                    continue;
                };
                // Acquire: every use by a holder that let its share go happens before the
                // destruction (the module docs). A share taken since by a lookup keeps it.
                if held.shares.load(Ordering::Acquire) != 0 {
                    continue;
                }
                if let Some(key) = self.keys.remove(&at) {
                    if let Some(held) = self.entries.remove(&key) {
                        gone.push(held);
                    }
                }
            }
        }
        gone.extend(self.unheld_own());
        Gone(gone)
    }

    /// Let go of the objects of a host step's or a pool piece's own that nothing else holds: on
    /// any thread, as no lookup finds them.
    pub(crate) fn sweep(&mut self) -> Gone<T> {
        Gone(self.unheld_own())
    }

    /// The own objects nothing else holds, removed, in the order they were made.
    fn unheld_own(&mut self) -> Vec<Arc<Held<T>>> {
        let mut gone = Vec::new();
        self.own.retain(|held| {
            // As in `settle`: no share left, with an Acquire. No lookup finds an own object, so
            // none is taken anew.
            if held.shares.load(Ordering::Acquire) != 0 {
                return true;
            }
            gone.push(Arc::clone(held));
            false
        });
        gone
    }

    /// How many objects it holds, those found by lookups and the own ones (a bench's look).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn len(&self) -> usize {
        self.entries.len() + self.own.len()
    }

    /// Every object, those found by lookups in the order they were inserted, then the own ones:
    /// the device's teardown, whatever still shares them. A share left then queues nothing that
    /// is destroyed again: its drop finds its place gone from the map.
    pub(crate) fn drain(&mut self) -> Gone<T> {
        let mut all = Vec::with_capacity(self.entries.len() + self.own.len());
        while let Some((_, key)) = self.keys.pop_first() {
            all.extend(self.entries.remove(&key));
        }
        all.append(&mut self.own);
        lock(&self.released).clear();
        Gone(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// An object whose destruction its owner, a log, records; dropping it records nothing.
    struct Recorded(u32);

    /// The objects destroyed, in order.
    type Log = Mutex<Vec<u32>>;

    impl Owned<Log> for Recorded {
        fn destroy(&self, log: &Log) {
            log.lock().unwrap().push(self.0);
        }
    }

    fn destroyed(log: &Log) -> Vec<u32> {
        log.lock().unwrap().clone()
    }

    /// The object of `n`, made for the program where none is.
    fn made(map: &mut ContentKeyed<u32, Recorded>, n: u32) -> Share<Recorded> {
        map.get_or_make(n, true, || Ok::<_, ()>(Recorded(n)))
            .unwrap()
            .0
    }

    /// The objects a settle of `map` looked at.
    fn settled(map: &mut ContentKeyed<u32, Recorded>, log: &Log) -> u64 {
        let before = map.looked_at;
        map.settle().destroy(log);
        map.looked_at - before
    }

    #[test]
    fn objects_leave_only_when_the_map_alone_holds_them_in_insertion_order() {
        let log = Log::default();
        let mut map: ContentKeyed<Vec<u32>, Recorded> = ContentKeyed::default();
        let mut make = |key: Vec<u32>, n| {
            map.get_or_make(key, true, || Ok::<_, ()>(Recorded(n)))
                .unwrap()
                .0
        };
        let a = make(vec![7, 7], 1);
        let b = make(vec![8], 2);
        let c = make(vec![9], 3);
        // The same content finds the same object; other content none.
        assert!(std::ptr::eq(&*map.get(&vec![7, 7]).unwrap(), &*a));
        assert!(map.get(&vec![7]).is_none());
        // Held by a variant: nothing leaves, nothing is destroyed.
        map.settle().destroy(&log);
        assert_eq!(map.entries.len(), 3);
        // Released by their holders: dropping a share destroys nothing; the settle's result,
        // destroyed through the owner, destroys them in the order of insertion, whatever the
        // order of release.
        drop(c);
        drop(a);
        let taken = map.get(&vec![8]).unwrap();
        drop(b);
        assert!(destroyed(&log).is_empty());
        let gone = map.settle();
        assert!(destroyed(&log).is_empty());
        gone.destroy(&log);
        assert_eq!(destroyed(&log), vec![1, 3]);
        // A holder that took the object from the map keeps it there.
        assert_eq!(map.entries.len(), 1);
        drop(taken);
        map.drain().destroy(&log);
        assert_eq!(destroyed(&log), vec![1, 3, 2]);
        assert_eq!(map.entries.len(), 0);
    }

    #[test]
    fn only_the_program_s_objects_enter_the_map() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        let made = |n| move || Ok::<_, ()>(Recorded(n));
        // A host step's miss: made, its own, found by no lookup.
        let (own, fresh) = map.get_or_make(5, false, made(5)).unwrap();
        assert!(fresh);
        assert_eq!(map.entries.len(), 0);
        assert!(map.get(&5).is_none());
        // The program's: made once, then found.
        let (first, fresh) = map.get_or_make(6, true, made(6)).unwrap();
        assert!(fresh);
        let (again, fresh) = map.get_or_make(6, false, made(60)).unwrap();
        assert!(!fresh);
        assert!(std::ptr::eq(&*first, &*again));
        // A failed making leaves nothing.
        assert!(map
            .get_or_make(7, true, || Err::<Recorded, ()>(()))
            .is_err());
        assert!(map
            .get_or_make(8, false, || Err::<Recorded, ()>(()))
            .is_err());
        assert_eq!((map.entries.len(), map.own.len()), (1, 1));
        drop((first, again, own));
        map.settle().destroy(&log);
        assert_eq!(destroyed(&log), vec![6, 5]);
    }

    /// A host step's own object goes at a sweep, on any thread, once nothing else holds it; a
    /// sweep leaves the map's objects alone.
    #[test]
    fn a_sweep_destroys_only_unheld_own_objects() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        let (program, _) = map
            .get_or_make(1, true, || Ok::<_, ()>(Recorded(1)))
            .unwrap();
        let (held, _) = map
            .get_or_make(2, false, || Ok::<_, ()>(Recorded(2)))
            .unwrap();
        let (released, _) = map
            .get_or_make(3, false, || Ok::<_, ()>(Recorded(3)))
            .unwrap();
        drop((program, released));
        map.sweep().destroy(&log);
        assert_eq!(destroyed(&log), vec![3]);
        assert_eq!((map.entries.len(), map.own.len()), (1, 1));
        drop(held);
        map.sweep().destroy(&log);
        assert_eq!(destroyed(&log), vec![3, 2]);
        assert_eq!(
            map.entries.len(),
            1,
            "the program's object waits for a settle"
        );
    }

    /// The teardown destroys every object, the program's and the own ones, whatever still shares
    /// them: a share left then (its holder dropping its device right then) destroys nothing as it
    /// goes, so each object is destroyed once, with the device.
    #[test]
    fn the_teardown_destroys_everything_once_and_a_late_share_destroys_nothing() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        let (late, _) = map
            .get_or_make(1, true, || Ok::<_, ()>(Recorded(1)))
            .unwrap();
        let (unheld, _) = map
            .get_or_make(2, true, || Ok::<_, ()>(Recorded(2)))
            .unwrap();
        let (late_own, _) = map
            .get_or_make(3, false, || Ok::<_, ()>(Recorded(3)))
            .unwrap();
        drop(unheld);
        map.drain().destroy(&log);
        assert_eq!(destroyed(&log), vec![1, 2, 3]);
        assert_eq!((map.entries.len(), map.own.len()), (0, 0));
        drop((late, late_own));
        drop(map);
        assert_eq!(
            destroyed(&log),
            vec![1, 2, 3],
            "a late share destroyed nothing"
        );
    }

    /// A share that outlives its map (a holder dropping its device after the teardown) goes with
    /// no panic, and leaves nothing behind: neither its object nor the map's release queue.
    #[test]
    fn a_share_dropped_after_its_map_leaves_nothing() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        let late = made(&mut map, 1);
        let copy = late.clone();
        let object = Arc::downgrade(&late.0);
        let queue = Arc::downgrade(&map.released);
        map.drain().destroy(&log);
        drop(map);
        drop(copy);
        drop(late);
        assert_eq!(destroyed(&log), vec![1], "destroyed once, at the teardown");
        assert!(object.upgrade().is_none(), "the object was freed");
        assert!(queue.upgrade().is_none(), "the release queue was freed");
    }

    /// The bound this fixes: a settle looks only at the objects released since the last, however
    /// many the map holds, and at none where none was released.
    #[test]
    fn a_settle_looks_only_at_what_was_released() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        let mut held: Vec<Share<Recorded>> = (0..10_000).map(|n| made(&mut map, n)).collect();
        assert_eq!(
            settled(&mut map, &log),
            0,
            "nothing released, nothing looked at"
        );
        let released = held.swap_remove(4321);
        drop(released);
        assert_eq!(settled(&mut map, &log), 1, "one released, one looked at");
        assert_eq!(destroyed(&log), vec![4321]);
        assert_eq!(settled(&mut map, &log), 0);
        assert_eq!(map.entries.len(), 9_999);
        drop(held);
        assert_eq!(settled(&mut map, &log), 9_999);
        assert_eq!(map.entries.len(), 0);
    }

    /// A share taken from the map after the object's last went, before the settle, keeps the
    /// object: the settle looks at it and leaves it, and its next release queues it again. Taken
    /// and let go again before the settle, it is queued once.
    #[test]
    fn a_share_taken_again_before_the_settle_keeps_the_object() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        drop(made(&mut map, 1));
        let again = map.get(&1).unwrap();
        assert_eq!(settled(&mut map, &log), 1);
        assert!(destroyed(&log).is_empty(), "the share taken again keeps it");
        assert_eq!(map.entries.len(), 1);
        drop(again);
        drop(map.get(&1).unwrap());
        drop(map.get(&1).unwrap());
        assert_eq!(settled(&mut map, &log), 1, "queued once");
        assert_eq!(destroyed(&log), vec![1]);
        assert!(map.get(&1).is_none());
    }

    /// A thread of the runtime's (`exit_guard::spawn`, as every thread the runtime starts is),
    /// running `f`: a holder's, as a handler thread or a pool piece is, outside the program's
    /// calls.
    fn holder_thread<T: Send + 'static>(
        f: impl FnOnce() -> T + Send + 'static,
    ) -> std::thread::JoinHandle<T> {
        crate::exit_guard::spawn(std::thread::Builder::new(), f).unwrap()
    }

    /// The same across threads: one thread lets its share go, another takes one from the map
    /// before the settle and holds it through it; the object stays until that share goes, then
    /// leaves at the next settle.
    #[test]
    fn a_share_taken_on_another_thread_before_the_settle_keeps_the_object() {
        let log = Log::default();
        let map = Arc::new(Mutex::new(ContentKeyed::<u32, Recorded>::default()));
        let first = made(&mut lock(&map), 1);
        holder_thread(move || drop(first)).join().unwrap();
        let (taken, give_back) = std::sync::mpsc::channel::<()>();
        let holder = {
            let map = Arc::clone(&map);
            holder_thread(move || {
                let share = lock(&map).get(&1).unwrap();
                taken.send(()).unwrap();
                // Held until the main thread's settle is done.
                std::thread::park();
                drop(share);
            })
        };
        give_back.recv().unwrap();
        lock(&map).settle().destroy(&log);
        assert!(
            destroyed(&log).is_empty(),
            "the other thread's share keeps it"
        );
        holder.thread().unpark();
        holder.join().unwrap();
        lock(&map).settle().destroy(&log);
        assert_eq!(destroyed(&log), vec![1]);
    }

    /// An object whose destruction marks it, and a log of each destroyed once.
    struct Marked {
        id: u32,
        destroyed: AtomicBool,
    }

    impl Owned<Log> for Marked {
        fn destroy(&self, log: &Log) {
            assert!(
                !self.destroyed.swap(true, Ordering::SeqCst),
                "destroyed twice"
            );
            log.lock().unwrap().push(self.id);
        }
    }

    /// Holders' threads that take shares from the map and let them go while the program's
    /// thread makes the objects again and settles the map, as a pool piece's or a handler
    /// thread's holders race a program's settle: no object is destroyed while a share of it
    /// lives, none twice, and every one made by the end. The program's thread lets its own share
    /// go as it makes each object, so a holder's share taken before the next settle is what keeps
    /// it: the race between a last drop and a settle.
    #[test]
    fn settles_racing_shares_destroy_nothing_held() {
        const THREADS: usize = 4;
        const HOLDS: u32 = 2_000;
        const KEYS: u32 = 8;
        let log = Log::default();
        let map = Arc::new(Mutex::new(ContentKeyed::<u32, Marked>::default()));
        let running = Arc::new(AtomicUsize::new(THREADS));
        let takers: Vec<_> = (0..THREADS as u32)
            .map(|t| {
                let (map, running) = (Arc::clone(&map), Arc::clone(&running));
                holder_thread(move || {
                    let (mut held, mut n) = (0, t);
                    while held < HOLDS {
                        n = n.wrapping_add(1);
                        let Some(share) = lock(&map).get(&(n % KEYS)) else {
                            continue;
                        };
                        let copy = share.clone();
                        drop(share);
                        assert!(
                            !copy.destroyed.load(Ordering::SeqCst),
                            "destroyed while held"
                        );
                        held += 1;
                    }
                    running.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        // The program's thread: each key's object made again where a settle let it go, its own
        // share let go at once; the settle under a lock of its own, so holders take shares
        // between the two.
        let mut made = 0u32;
        while running.load(Ordering::SeqCst) > 0 {
            {
                let mut map = lock(&map);
                for key in 0..KEYS {
                    let (share, fresh) = map
                        .get_or_make(key, true, || {
                            Ok::<_, ()>(Marked {
                                id: made,
                                destroyed: AtomicBool::new(false),
                            })
                        })
                        .unwrap();
                    made += u32::from(fresh);
                    drop(share);
                }
            }
            std::thread::yield_now();
            let gone = lock(&map).settle();
            gone.destroy(&log);
        }
        for taker in takers {
            taker.join().unwrap();
        }
        lock(&map).settle().destroy(&log);
        let mut destroyed = log.lock().unwrap().clone();
        destroyed.sort_unstable();
        assert_eq!(
            destroyed,
            (0..made).collect::<Vec<_>>(),
            "each made, destroyed once"
        );
        assert_eq!(lock(&map).entries.len(), 0);
    }
}
