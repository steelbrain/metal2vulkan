//! Dense `u32` ids and the arena that owns what they index.
//!
//! Every IR entity lives in an [`Arena`] and is referred to by a typed id; ids of different kinds
//! never mix. Ids follow insertion order, which the decoder makes bitcode order, so identical bytes
//! produce identical ids. Lookups return `Option`: the core's panic-lint policy forbids unchecked
//! indexing, and [`super::Module::new`] proves every stored id in range once.

use std::fmt;
use std::marker::PhantomData;

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};

/// A dense arena index.
pub trait Idx: Copy + Eq + Ord + fmt::Debug {
    /// The id at a `usize` position, if it fits the `u32` index space.
    fn from_index(index: usize) -> Option<Self>;
    /// The dense index.
    fn index(self) -> usize;
}

macro_rules! ir_id {
    ($(#[$doc:meta])* $name:ident, $prefix:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            /// The id with this dense index.
            pub const fn new(index: u32) -> $name {
                $name(index)
            }

            /// The raw `u32` index.
            pub const fn raw(self) -> u32 {
                self.0
            }
        }

        impl Idx for $name {
            fn from_index(index: usize) -> Option<$name> {
                u32::try_from(index).ok().map($name::new)
            }

            fn index(self) -> usize {
                self.raw() as usize
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
    };
}

ir_id!(
    /// An interned type ([`super::Type`]).
    TypeId,
    "t"
);
ir_id!(
    /// An interned constant ([`super::Constant`]).
    ConstId,
    "c"
);
ir_id!(
    /// A module-level global variable.
    GlobalId,
    "g"
);
ir_id!(
    /// A function (definition or declaration).
    FuncId,
    "f"
);
ir_id!(
    /// A basic block within one function body. Block 0 is the entry.
    BlockId,
    "bb"
);
ir_id!(
    /// An instruction within one function body (void instructions included).
    InstId,
    "i"
);
ir_id!(
    /// A function parameter position.
    ArgIdx,
    "a"
);
ir_id!(
    /// A metadata node, string, or value.
    MdId,
    "md"
);
ir_id!(
    /// An interned byte string (symbol names, metadata strings). Diagnostics and ABI only: no
    /// product decision may branch on its contents except stable `air.*`/`llvm.*` ABI keys.
    Sym,
    "s"
);

/// A dense arena of `T` indexed by `I`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Arena<I, T> {
    items: Vec<T>,
    _id: PhantomData<fn(I) -> I>,
}

impl<I: Idx, T> Arena<I, T> {
    /// An empty arena.
    pub fn new() -> Arena<I, T> {
        Arena {
            items: Vec::new(),
            _id: PhantomData,
        }
    }

    /// Append `value`, charging the meter for the capacity it adds before reserving it.
    pub fn push(&mut self, value: T, meter: &mut Meter) -> Result<I, Error> {
        let id = I::from_index(self.items.len()).ok_or_else(|| {
            Error::Internal(InternalError::new(
                InternalKind::Invariant,
                "arena index space exhausted",
            ))
        })?;
        meter.grow(&mut self.items, 1)?;
        self.items.push(value);
        Ok(id)
    }

    /// Make room for exactly `additional` more elements, charging the meter for the capacity it
    /// adds ([`Meter::reserve_exact`]), where the caller knows the arena's final length: the
    /// pushes that fill it then charge nothing and leave no doubling slack.
    pub fn reserve_exact(&mut self, additional: usize, meter: &mut Meter) -> Result<(), Error> {
        meter.reserve_exact(&mut self.items, additional)
    }

    /// The heap bytes the arena's capacity holds: what [`Arena::push`] and
    /// [`Arena::reserve_exact`] charged for it, since the capacity only grows by those charged
    /// steps.
    pub fn capacity_bytes(&self) -> usize {
        crate::budget::vec_bytes(&self.items)
    }

    /// The element at `id`, if it exists.
    pub fn get(&self, id: I) -> Option<&T> {
        self.items.get(id.index())
    }

    /// The mutable element at `id`, if it exists.
    pub fn get_mut(&mut self, id: I) -> Option<&mut T> {
        self.items.get_mut(id.index())
    }

    /// Whether `id` is in range.
    pub fn contains(&self, id: I) -> bool {
        id.index() < self.items.len()
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Every `(id, element)` pair in id order.
    ///
    /// Every index fits `I` because [`Arena::push`] is the only way to grow an arena.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (I, &T)> + '_ {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(i, t)| Some((I::from_index(i)?, t)))
    }

    /// Every id in order.
    pub fn ids(&self) -> impl DoubleEndedIterator<Item = I> + '_ {
        (0..self.items.len()).filter_map(I::from_index)
    }

    /// Every element in id order.
    pub fn values(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }

    /// Every element, mutably, in id order.
    pub fn values_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.items.iter_mut()
    }

    /// Consume the arena into its elements in id order.
    pub fn into_vec(self) -> Vec<T> {
        self.items
    }
}

impl<I: Idx, T> Default for Arena<I, T> {
    fn default() -> Arena<I, T> {
        Arena::new()
    }
}

impl<I: Idx, T: fmt::Debug> fmt::Debug for Arena<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}
