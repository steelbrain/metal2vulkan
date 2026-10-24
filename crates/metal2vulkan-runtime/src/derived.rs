//! A derivation an object holds once, and the count that holds it to once.
//!
//! A Metal object answers a getter from what it already made. Several of ours answer from a
//! derivation over the object's inputs -- a decode of a function's AIR, a reflection over its
//! bindings -- and a getter that derives per call pays the whole of it per call, where Metal pays
//! nothing. Where every input of a derivation is fixed when the object is made, the object holds
//! the answer and derives it once: [`Derived`] for one answer, [`DerivedMap`] for one per key.
//!
//! # Why each memo counts its derivations
//!
//! A memo is proven by a **count**, never by a time. However many getters read a derivation and
//! however often, the object must derive it once, which is a fact of the structure and holds on
//! any host under any load; a ratio against Metal is a measurement of two machines and moves with
//! both. So every memo here counts the derivations that ran, and one table-driven test reads the
//! count for a list of getters (`tests::derived_once`). A getter added to an object joins that
//! list as a row.
//!
//! The count is of this object alone and of nothing that ran before it in the process: it needs no
//! quiet host, no warm cache and no ordering between tests.

use crate::device::lock;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// One derivation an object holds, with how many times it has been derived.
///
/// `Default` is "not derived yet", which is what an object is made with: making an object derives
/// nothing, as making Metal's does not.
#[derive(Debug)]
pub(crate) struct Derived<T> {
    held: Option<T>,
    derived: u32,
}

impl<T> Default for Derived<T> {
    fn default() -> Derived<T> {
        Derived {
            held: None,
            derived: 0,
        }
    }
}

impl<T: Clone> Derived<T> {
    /// The held answer, else `derive`'s, kept.
    ///
    /// `derive` runs with no lock held, so a slow derivation blocks no other reader; two readers
    /// that race both derive and the first to finish is kept, which answers both the same value
    /// and is why the count is of derivations that **ran**. A refusal is not kept: a derivation
    /// refused under a fixed budget refuses the same way again, so the error path is left exactly
    /// as it was without a memo.
    pub(crate) fn or_derive<E>(
        memo: &Mutex<Derived<T>>,
        derive: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if let Some(held) = lock(memo).held.clone() {
            return Ok(held);
        }
        let made = derive()?;
        let mut memo = lock(memo);
        memo.derived += 1;
        Ok(memo.held.get_or_insert(made).clone())
    }

    /// How many times the answer has been derived: 1 however many getters have read it.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn derived(memo: &Mutex<Derived<T>>) -> u32 {
        lock(memo).derived
    }
}

/// One derivation an object holds per key -- an argument buffer's index, a function's name -- with
/// how many times it has derived any of them. [`Derived`]'s rules, per key.
#[derive(Debug)]
pub(crate) struct DerivedMap<K, V> {
    held: BTreeMap<K, V>,
    derived: u32,
}

impl<K, V> Default for DerivedMap<K, V> {
    fn default() -> DerivedMap<K, V> {
        DerivedMap {
            held: BTreeMap::new(),
            derived: 0,
        }
    }
}

impl<K: Ord, V: Clone> DerivedMap<K, V> {
    /// The answer held for `key`, else `derive`'s, kept under it.
    pub(crate) fn or_derive<E>(
        memo: &Mutex<DerivedMap<K, V>>,
        key: K,
        derive: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        if let Some(held) = lock(memo).held.get(&key).cloned() {
            return Ok(held);
        }
        let made = derive()?;
        let mut memo = lock(memo);
        memo.derived += 1;
        Ok(memo.held.entry(key).or_insert(made).clone())
    }

    /// How many times an answer has been derived, over every key: one per key read, however many
    /// getters read each.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn derived(memo: &Mutex<DerivedMap<K, V>>) -> u32 {
        lock(memo).derived
    }
}
