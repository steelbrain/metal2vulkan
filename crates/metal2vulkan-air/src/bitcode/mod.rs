//! Bitcode records of module versions 1 and 2 (relative value ids) to `air::ir`.
//!
//! [`decode`] turns one bitcode module (raw, or in the `0x0B17C0DE` wrapper that metallib
//! function entries carry) into a verified [`Decoded`] module plus its AIR metadata ABI
//! (`air::abi`) and its decode-only [`PointeeTable`]. [`decode_module`] returns the undecorated [`ModuleTables`]: identification,
//! the names (version 2's string table, version 1's module symbol table), the type table (typed pointers kept by bitcode index, erased in the IR),
//! attribute groups and lists, globals, function records, module constants, metadata and
//! metadata kinds, sync-scope names, the module value numbering, and every function body.
//!
//! Function blocks are decoded where the module block holds them, against the module tables
//! decoded before them, and associated with the defined functions in record order; once the
//! module block ends the association is checked against `VSTOFFSET` and the symbol table's
//! function offsets. Calls to markers (`llvm.lifetime.*` and the like) are dropped as they are
//! read. Two things translation does not read are kept only when a decode asks
//! ([`DecodeOptions`]): the enum attributes the IR does not interpret ([`EnumAttrTable`]), for a
//! writer that hands the module back as written; and the removal of a marker declaration no use
//! survives, with what only it used, the rest renumbered in order (`markers`), so that how an LLVM
//! major spells a marker does not reach a decoded module that is compared across majors.
//!
//! Records outside the accepted subset are typed errors: a code LLVM defines but the typed model
//! does not cover is `Unsupported` (an input construct with its code), a code LLVM does not define
//! is `UnknownRecord`/`UnknownBlock`, and a record whose operands do not fit its code is
//! `MalformedRecord`. Nothing panics on malformed input. Every input-scaled container is charged
//! to the [`Meter`] before it is reserved, and no reservation is sized from an untrusted count.

mod attrs;
mod codes;
mod constants;
mod decoded;
mod enum_attrs;
mod function;
#[cfg(test)]
mod function_tests;
mod markers;
mod metadata;
mod module;
mod stream;
#[cfg(test)]
mod tests;
mod types;

pub use attrs::AttrList;
pub use decoded::decode_with;
pub use decoded::{decode, DecodeOptions, Decoded, PointeeTable};
pub use enum_attrs::EnumAttrTable;
pub use enum_attrs::{EnumAttrs, SiteEnums};
pub use metadata::MdKinds;
pub use types::BitcodeTypes;

use crate::budget::Meter;
use crate::error::Error;
use crate::ir::{ConstId, FuncId, ModuleParts, SyncScope};

/// One module-level value (global, function, or module constant) in bitcode value-id order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModuleValue {
    /// The value as an IR constant (globals and functions as `GlobalAddr`/`FuncAddr`); `None`
    /// for a marker declaration decode removed because no use of it survives (`markers`).
    pub id: Option<ConstId>,
    /// The bitcode type index of the value itself, when the record names it: every module
    /// constant, a global in the older pointer-typed spelling, and a function whose record names
    /// a pointer to its function type. `None` when the record names only the addressed object's
    /// type (an explicit-type global, a function record naming its function type directly).
    pub bc_type: Option<u32>,
    /// For a global or function, the bitcode type index of the object it addresses: a global's
    /// value type, a function's function type (after the typed-pointer indirection). `None` for
    /// constants. Keeps typed-pointer pointees for decode-only checks: distinct bitcode types
    /// that differ only in pointees erase to one IR `TypeId`, so a function's parameter pointees
    /// are `BitcodeTypes::params(object_bc_type)` mapped through `BitcodeTypes::pointee`, never a
    /// lookup by the erased `Function::ty`.
    pub object_bc_type: Option<u32>,
}

/// The module-level tables of one bitcode module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleTables {
    /// Symbols, types, constants, globals (initializers resolved), functions with their bodies,
    /// metadata, named metadata, triple, source file name, and data layout.
    pub parts: ModuleParts,
    /// The bitcode type table: erased IR type and typed-pointer pointee per bitcode index.
    pub types: BitcodeTypes,
    /// The module value numbering (value id = index).
    pub values: Vec<ModuleValue>,
    /// `PARAMATTR` lists (call sites refer to them by `index + 1`).
    pub attr_lists: Vec<AttrList>,
    /// The instruction attachment kinds the IR keeps.
    pub md_kinds: MdKinds,
    /// Sync-scope names by id: `Some` for LLVM's built-in scopes, `None` for any other.
    pub sync_scopes: Vec<Option<SyncScope>>,
    /// Functions with bodies, in record order (the order of their function blocks).
    pub defined: Vec<FuncId>,
    /// Function blocks decoded (one per defined function).
    pub function_blocks: u32,
    /// Module `VALUE_SYMTAB` function entries: `(function, word offset of its block)`.
    pub fn_entries: Vec<(FuncId, u64)>,
    /// The enum attributes the IR does not interpret, by position, when the decode asked for
    /// them ([`DecodeOptions::enum_attrs`]); empty otherwise.
    pub enum_attrs: EnumAttrTable,
}

impl ModuleTables {
    /// The IR constant for module value id `id`.
    #[cfg(test)]
    pub fn value(&self, id: u64) -> Option<ConstId> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.values.get(i))
            .and_then(|v| v.id)
    }
}

/// Decode the module tables and function bodies of a bitcode stream (starting with
/// `BC 0xC0DE`, as located by `air::container`), keeping what `opts` asks for beyond what
/// translation reads. The result is not verified; [`decode`] is.
pub fn decode_module(
    stream: &[u8],
    opts: DecodeOptions,
    meter: &mut Meter,
) -> Result<ModuleTables, Error> {
    module::decode(stream, opts, meter)
}
