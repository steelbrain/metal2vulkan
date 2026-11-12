//! Types and the type interner.
//!
//! Pointers carry only their address space: pointee types are erased at decode and kept in the
//! decoder's decode-only `PointeeTable`. Memory uses the explicit access types of loads, stores,
//! and GEPs plus the [`super::DataLayout`]. Erasure also makes the type graph acyclic, so the
//! interner can require every referenced type to exist already: a child id is always smaller than
//! its parent's, and whole-table passes (layout, printing) run in one forward sweep with no
//! recursion.

use std::fmt;

use super::id::{Arena, Sym, TypeId};
use super::intern::FxHashMap;
use crate::budget::Meter;
use crate::error::{Error, Feature, Limit, Unsupported};
use crate::error::{InputError, InputErrorKind, Location, VerifyError};

/// An AIR type. Interned: equal types have equal [`TypeId`]s.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// `void`: no value.
    Void,
    /// `label`: the type of a basic block.
    Label,
    /// `metadata`: the type of a metadata operand to an intrinsic call.
    Metadata,
    /// An integer of 1..=64 bits. Widths such as `i2`/`i40` exist in the corpus; lowering decides
    /// their support.
    Int(u8),
    /// `half`: IEEE binary16.
    Half,
    /// `bfloat`: the 16-bit brain floating-point format (8 exponent bits, 7 mantissa bits).
    BFloat,
    /// `float`: IEEE binary32.
    Float,
    /// `double`: IEEE binary64.
    Double,
    /// A fixed vector of `len >= 1` integer, floating-point, or pointer elements.
    Vector {
        /// The element type.
        elem: TypeId,
        /// The number of lanes.
        len: u32,
    },
    /// An array of a sized element type.
    Array {
        /// The element type.
        elem: TypeId,
        /// The number of elements.
        len: u64,
    },
    /// A literal or identified struct. `name` is for diagnostics and keeps identified structs
    /// nominal (as in LLVM); no decision may branch on it.
    Struct {
        /// The field types, in order.
        fields: Box<[TypeId]>,
        /// A packed struct (`<{ ... }>`): no padding between fields, alignment 1.
        packed: bool,
        /// The identified struct's name; `None` for a literal struct.
        name: Option<Sym>,
    },
    /// An identified struct without a body (for example a texture handle's pointee in older
    /// inputs). Never interpreted by name: resource identity comes from the metadata ABI.
    OpaqueStruct {
        /// The struct's name, when the bitcode gives one.
        name: Option<Sym>,
    },
    /// A pointer into `space`; the pointee is erased.
    Ptr {
        /// The address space pointed into.
        space: AddrSpace,
    },
    /// A function type. `vararg` is representable but refused where used.
    Function {
        /// The return type; `void` when the function returns nothing.
        ret: TypeId,
        /// The parameter types, in order.
        params: Box<[TypeId]>,
        /// Whether the function takes variable arguments (`...`).
        vararg: bool,
    },
}

impl Type {
    /// The width in bits of a floating-point type.
    pub fn float_bits(&self) -> Option<u32> {
        match self {
            Type::Half | Type::BFloat => Some(16),
            Type::Float => Some(32),
            Type::Double => Some(64),
            _ => None,
        }
    }

    /// Whether this is a floating-point scalar.
    pub fn is_float(&self) -> bool {
        self.float_bits().is_some()
    }

    /// Whether this may be a vector element.
    pub fn is_vector_element(&self) -> bool {
        matches!(self, Type::Int(_) | Type::Ptr { .. }) || self.is_float()
    }

    /// Whether values of this type have a size (and so may be stored, loaded, and passed).
    pub fn is_sized(&self) -> bool {
        matches!(
            self,
            Type::Int(_)
                | Type::Half
                | Type::BFloat
                | Type::Float
                | Type::Double
                | Type::Vector { .. }
                | Type::Array { .. }
                | Type::Struct { .. }
                | Type::Ptr { .. }
        )
    }

    /// Whether this is an aggregate (struct or array).
    pub fn is_aggregate(&self) -> bool {
        matches!(self, Type::Array { .. } | Type::Struct { .. })
    }

    /// The types this type refers to directly.
    pub fn children(&self) -> Vec<TypeId> {
        match self {
            Type::Vector { elem, .. } | Type::Array { elem, .. } => vec![*elem],
            Type::Struct { fields, .. } => fields.to_vec(),
            Type::Function { ret, params, .. } => std::iter::once(*ret)
                .chain(params.iter().copied())
                .collect(),
            Type::Void
            | Type::Label
            | Type::Metadata
            | Type::Int(_)
            | Type::Half
            | Type::BFloat
            | Type::Float
            | Type::Double
            | Type::OpaqueStruct { .. }
            | Type::Ptr { .. } => Vec::new(),
        }
    }
}

/// An AIR address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AddrSpace {
    /// `thread`: memory private to one thread (locals and the stack).
    Thread = 0,
    /// `device`: read-write buffer memory every thread can reach.
    Device = 1,
    /// `constant`: read-only buffer memory every thread can reach.
    Constant = 2,
    /// `threadgroup`: memory shared by the threads of one threadgroup.
    Threadgroup = 3,
    /// `threadgroup_imageblock`: a tile's imageblock, a region of threadgroup memory.
    ThreadgroupImageblock = 4,
    /// `object_data`: an object function's payload, which the mesh functions it launches read
    /// (`[[payload]]`).
    ObjectData = 6,
    /// A mesh function's output handle (`metal::mesh<..>`, an opaque struct).
    MeshOutput = 7,
    /// MSL 4.1's generic address space: a pointer declared with no address space, which a device,
    /// thread or threadgroup pointer converts to (`addrspacecast`) and keeps pointing into its
    /// space; `air.is_{device,thread,threadgroup}_pointer` asks which. Canonicalization gives each
    /// generic pointer the one space every pointer it may hold comes from (`canon::generic`).
    Generic = 8,
}

impl AddrSpace {
    /// Every address space.
    #[cfg(test)]
    pub const ALL: [AddrSpace; 8] = [
        AddrSpace::Thread,
        AddrSpace::Device,
        AddrSpace::Constant,
        AddrSpace::Threadgroup,
        AddrSpace::ThreadgroupImageblock,
        AddrSpace::ObjectData,
        AddrSpace::MeshOutput,
        AddrSpace::Generic,
    ];

    /// The address space numbered `n` in LLVM IR.
    pub fn from_raw(n: u32) -> Result<AddrSpace, Unsupported> {
        match n {
            0 => Ok(AddrSpace::Thread),
            1 => Ok(AddrSpace::Device),
            2 => Ok(AddrSpace::Constant),
            3 => Ok(AddrSpace::Threadgroup),
            4 => Ok(AddrSpace::ThreadgroupImageblock),
            6 => Ok(AddrSpace::ObjectData),
            7 => Ok(AddrSpace::MeshOutput),
            8 => Ok(AddrSpace::Generic),
            _ => Err(Unsupported::not_implemented(Feature::AddressSpace(n))),
        }
    }

    /// The LLVM IR number of this address space.
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

impl fmt::Display for AddrSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "addrspace({})", self.raw())
    }
}

/// The module's type interner.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TypeTable {
    types: Arena<TypeId, Type>,
    index: FxHashMap<Type, TypeId>,
    /// Whether each type is a pointer or an aggregate holding one ([`TypeTable::holds_pointer`]),
    /// by id: read from its children's at interning, which precede it.
    pointers: Arena<TypeId, bool>,
}

/// A malformed-type error at module scope.
pub(crate) fn malformed_type() -> Error {
    Error::InvalidInput(InputError::new(
        InputErrorKind::Verify(VerifyError::MalformedType),
        Location::module(),
    ))
}

impl TypeTable {
    /// An empty table.
    pub fn new() -> TypeTable {
        TypeTable::default()
    }

    /// Intern `ty`, checking it is well formed: children exist, integer widths are 1..=64,
    /// vector elements are scalars, aggregate members are sized, and function signatures take
    /// and return value types. The table is capped at [`Limit::TypeCount`].
    pub fn intern(&mut self, ty: Type, meter: &mut Meter) -> Result<TypeId, Error> {
        if let Some(&id) = self.index.get(&ty) {
            return Ok(id);
        }
        self.check(&ty)?;
        meter.cap(Limit::TypeCount, self.types.len() as u64 + 1)?;
        // Stored twice: once in the arena, once as the lookup key.
        let payload = std::mem::size_of::<TypeId>().saturating_mul(ty.children().len());
        meter.charge(payload.saturating_mul(2))?;
        meter.charge(std::mem::size_of::<Type>())?;
        let pointer = match &ty {
            Type::Ptr { .. } => true,
            Type::Struct { fields, .. } => fields.iter().any(|f| self.holds_pointer(*f)),
            Type::Array { elem, .. } | Type::Vector { elem, .. } => self.holds_pointer(*elem),
            _ => false,
        };
        let id = self.types.push(ty.clone(), meter)?;
        self.pointers.push(pointer, meter)?;
        self.index.insert(ty, id);
        Ok(id)
    }

    fn check(&self, ty: &Type) -> Result<(), Error> {
        let child = |id: TypeId| self.types.get(id).ok_or_else(malformed_type);
        let ok = match ty {
            Type::Int(bits) => (1..=64).contains(bits),
            Type::Vector { elem, len } => *len >= 1 && child(*elem)?.is_vector_element(),
            Type::Array { elem, .. } => child(*elem)?.is_sized(),
            Type::Struct { fields, .. } => {
                let mut ok = true;
                for &f in fields.iter() {
                    ok &= child(f)?.is_sized();
                }
                ok
            }
            Type::Function { ret, params, .. } => {
                let ret = child(*ret)?;
                let mut ok = matches!(ret, Type::Void) || ret.is_sized();
                for &p in params.iter() {
                    let p = child(p)?;
                    ok &= p.is_sized() || matches!(p, Type::Metadata);
                }
                ok
            }
            Type::Void
            | Type::Label
            | Type::Metadata
            | Type::Half
            | Type::BFloat
            | Type::Float
            | Type::Double
            | Type::OpaqueStruct { .. }
            | Type::Ptr { .. } => true,
        };
        if ok {
            Ok(())
        } else {
            Err(malformed_type())
        }
    }

    /// The id of `ty` if it is interned.
    pub fn lookup(&self, ty: &Type) -> Option<TypeId> {
        self.index.get(ty).copied()
    }

    /// The type at `id`.
    pub fn get(&self, id: TypeId) -> Option<&Type> {
        self.types.get(id)
    }

    /// Whether `id` is in range.
    pub fn contains(&self, id: TypeId) -> bool {
        self.types.contains(id)
    }

    /// The number of types.
    pub(crate) fn len(&self) -> usize {
        self.types.len()
    }

    /// Every `(id, type)` in id order; children precede parents.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (TypeId, &Type)> + '_ {
        self.types.iter()
    }

    /// Whether `id` is a pointer or an aggregate (a struct, array, or vector) that holds one at
    /// any depth. Each type's answer is kept at interning, from its children's, so asking costs
    /// one lookup however its members share types.
    pub fn holds_pointer(&self, id: TypeId) -> bool {
        self.pointers.get(id).copied().unwrap_or(false)
    }

    /// Whether `id` is sized.
    pub fn is_sized(&self, id: TypeId) -> bool {
        self.get(id).is_some_and(Type::is_sized)
    }

    /// The element type of a vector, or `id` itself for a non-vector.
    pub fn scalar(&self, id: TypeId) -> TypeId {
        match self.get(id) {
            Some(Type::Vector { elem, .. }) => *elem,
            _ => id,
        }
    }

    /// The lane count of a vector type.
    pub fn lanes(&self, id: TypeId) -> Option<u32> {
        match self.get(id) {
            Some(Type::Vector { len, .. }) => Some(*len),
            _ => None,
        }
    }

    /// The integer width of `id` or of its vector element.
    pub fn int_bits(&self, id: TypeId) -> Option<u8> {
        match self.get(self.scalar(id)) {
            Some(Type::Int(bits)) => Some(*bits),
            _ => None,
        }
    }

    /// The floating-point width of `id` or of its vector element.
    pub fn float_bits(&self, id: TypeId) -> Option<u32> {
        self.get(self.scalar(id)).and_then(Type::float_bits)
    }

    /// The address space of a pointer type, or of a vector of pointers' element.
    pub fn ptr_space(&self, id: TypeId) -> Option<AddrSpace> {
        match self.get(self.scalar(id)) {
            Some(Type::Ptr { space }) => Some(*space),
            _ => None,
        }
    }
}
