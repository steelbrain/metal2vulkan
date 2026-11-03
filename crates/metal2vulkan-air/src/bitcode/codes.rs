//! Numeric block ids and record codes from LLVM's `LLVMBitCodes.h`, complete through LLVM 23.1
//! (`llvmorg-23.1.2`).
//!
//! Only the codes the reader dispatches on are named. A code outside a block's table is either a
//! known LLVM construct the typed model does not cover (`Unsupported`) or not LLVM at all
//! (`UnknownRecord`); the per-block decoders decide which.

// ---- Block ids (`enum BlockIDs`, FIRST_APPLICATION_BLOCKID = 8) ----
pub(crate) const MODULE_BLOCK: u32 = 8;
pub(crate) const PARAMATTR_BLOCK: u32 = 9;
pub(crate) const PARAMATTR_GROUP_BLOCK: u32 = 10;
pub(crate) const CONSTANTS_BLOCK: u32 = 11;
pub(crate) const FUNCTION_BLOCK: u32 = 12;
pub(crate) const IDENTIFICATION_BLOCK: u32 = 13;
pub(crate) const VALUE_SYMTAB_BLOCK: u32 = 14;
pub(crate) const METADATA_BLOCK: u32 = 15;
pub(crate) const METADATA_ATTACHMENT_BLOCK: u32 = 16;
pub(crate) const TYPE_BLOCK: u32 = 17;
pub(crate) const USELIST_BLOCK: u32 = 18;
pub(crate) const OPERAND_BUNDLE_TAGS_BLOCK: u32 = 21;
pub(crate) const METADATA_KIND_BLOCK: u32 = 22;
pub(crate) const STRTAB_BLOCK: u32 = 23;
pub(crate) const SYMTAB_BLOCK: u32 = 25;
pub(crate) const SYNC_SCOPE_NAMES_BLOCK: u32 = 26;

// ---- IDENTIFICATION ----
pub(crate) const IDENTIFICATION_STRING: u32 = 1;
pub(crate) const IDENTIFICATION_EPOCH: u32 = 2;

// ---- MODULE ----
pub(crate) const MODULE_VERSION: u32 = 1;
pub(crate) const MODULE_TRIPLE: u32 = 2;
pub(crate) const MODULE_DATALAYOUT: u32 = 3;
pub(crate) const MODULE_SECTIONNAME: u32 = 5;
pub(crate) const MODULE_GLOBALVAR: u32 = 7;
pub(crate) const MODULE_FUNCTION: u32 = 8;
pub(crate) const MODULE_VSTOFFSET: u32 = 13;
pub(crate) const MODULE_SOURCE_FILENAME: u32 = 16;
pub(crate) const MODULE_HASH: u32 = 17;
/// `MODULE_CODE_GUIDLIST` (LLVM 23): `[n x (high 32 bits, low 32 bits)]`, one global-value GUID
/// per module value id (0 = none assigned).
pub(crate) const MODULE_GUIDLIST: u32 = 20;
/// Module codes LLVM defines that the typed model refuses: ASM, DEPLIB, ALIAS_OLD, PURGEVALS,
/// GCNAME, COMDAT, ALIAS, METADATA_VALUES_UNUSED, IFUNC, ASM_PROPERTY (LLVM 23: a property of the
/// module asm that follows it).
pub(crate) const MODULE_UNSUPPORTED: &[u32] = &[4, 6, 9, 10, 11, 12, 14, 15, 18, 19];

// ---- PARAMATTR / PARAMATTR_GROUP ----
/// `PARAMATTR_CODE_ENTRY_OLD`: a list spelled without attribute groups; refused.
pub(crate) const PARAMATTR_ENTRY_OLD: u32 = 1;
pub(crate) const PARAMATTR_ENTRY: u32 = 2;
pub(crate) const PARAMATTR_GRP_ENTRY: u32 = 3;

// ---- TYPE_BLOCK_NEW ----
pub(crate) const TYPE_NUMENTRY: u32 = 1;
pub(crate) const TYPE_VOID: u32 = 2;
pub(crate) const TYPE_FLOAT: u32 = 3;
pub(crate) const TYPE_DOUBLE: u32 = 4;
pub(crate) const TYPE_LABEL: u32 = 5;
pub(crate) const TYPE_OPAQUE: u32 = 6;
pub(crate) const TYPE_INTEGER: u32 = 7;
pub(crate) const TYPE_POINTER: u32 = 8;
pub(crate) const TYPE_HALF: u32 = 10;
pub(crate) const TYPE_ARRAY: u32 = 11;
pub(crate) const TYPE_VECTOR: u32 = 12;
pub(crate) const TYPE_METADATA: u32 = 16;
pub(crate) const TYPE_STRUCT_ANON: u32 = 18;
pub(crate) const TYPE_STRUCT_NAME: u32 = 19;
pub(crate) const TYPE_STRUCT_NAMED: u32 = 20;
pub(crate) const TYPE_FUNCTION: u32 = 21;
pub(crate) const TYPE_BFLOAT: u32 = 23;
pub(crate) const TYPE_OPAQUE_POINTER: u32 = 25;
/// Type codes LLVM defines that the typed model refuses: FUNCTION_OLD, X86_FP80, FP128,
/// PPC_FP128, X86_MMX, TOKEN, X86_AMX, TARGET_TYPE, BYTE (LLVM 23).
pub(crate) const TYPE_UNSUPPORTED: &[u32] = &[9, 13, 14, 15, 17, 22, 24, 26, 27];

// ---- CONSTANTS ----
pub(crate) const CST_SETTYPE: u32 = 1;
pub(crate) const CST_NULL: u32 = 2;
pub(crate) const CST_UNDEF: u32 = 3;
pub(crate) const CST_INTEGER: u32 = 4;
pub(crate) const CST_WIDE_INTEGER: u32 = 5;
pub(crate) const CST_FLOAT: u32 = 6;
pub(crate) const CST_AGGREGATE: u32 = 7;
pub(crate) const CST_STRING: u32 = 8;
pub(crate) const CST_CSTRING: u32 = 9;
pub(crate) const CST_CE_BINOP: u32 = 10;
pub(crate) const CST_CE_CAST: u32 = 11;
pub(crate) const CST_CE_GEP_OLD: u32 = 12;
pub(crate) const CST_CE_INBOUNDS_GEP: u32 = 20;
pub(crate) const CST_DATA: u32 = 22;
/// `INLINEASM_OLD`: `[sideeffect|alignstack, asmstr, conststr]`, typed by its pointer's pointee.
pub(crate) const CST_INLINEASM_OLD: u32 = 18;
/// `INLINEASM_OLD2`: as `INLINEASM_OLD`, with the dialect bit.
pub(crate) const CST_INLINEASM_OLD2: u32 = 23;
pub(crate) const CST_CE_GEP_WITH_INRANGE_INDEX_OLD: u32 = 24;
pub(crate) const CST_POISON: u32 = 26;
/// `INLINEASM_OLD3`: as `INLINEASM_OLD2`, with the unwind bit.
pub(crate) const CST_INLINEASM_OLD3: u32 = 28;
/// `INLINEASM`: `[fnty, sideeffect|alignstack|asmdialect|unwind, asmstr, conststr]` (LLVM 13+).
pub(crate) const CST_INLINEASM: u32 = 30;
pub(crate) const CST_CE_GEP: u32 = 32;
/// Constant codes LLVM defines that the typed model refuses: CE_SELECT, CE_EXTRACTELT,
/// CE_INSERTELT, CE_SHUFFLEVEC, CE_CMP, CE_SHUFVEC_EX, BLOCKADDRESS, CE_UNOP,
/// DSO_LOCAL_EQUIVALENT, NO_CFI_VALUE, CE_GEP_WITH_INRANGE, PTRAUTH, PTRAUTH2, BYTE and WIDE_BYTE
/// (LLVM 23: constants of the refused byte type).
pub(crate) const CST_UNSUPPORTED: &[u32] =
    &[13, 14, 15, 16, 17, 19, 21, 25, 27, 29, 31, 33, 34, 35, 36];

// ---- METADATA ----
pub(crate) const METADATA_VALUE: u32 = 2;
pub(crate) const METADATA_NODE: u32 = 3;
pub(crate) const METADATA_NAME: u32 = 4;
pub(crate) const METADATA_DISTINCT_NODE: u32 = 5;
pub(crate) const METADATA_KIND: u32 = 6;
pub(crate) const METADATA_NAMED_NODE: u32 = 10;
pub(crate) const METADATA_STRINGS: u32 = 35;
pub(crate) const METADATA_GLOBAL_DECL_ATTACHMENT: u32 = 36;
pub(crate) const METADATA_INDEX_OFFSET: u32 = 38;
pub(crate) const METADATA_INDEX: u32 = 39;
/// Legacy metadata codes LLVM defines that the typed model refuses: STRING_OLD, KIND (in the
/// `METADATA` block rather than `METADATA_KIND`), OLD_NODE, OLD_FN_NODE.
pub(crate) const METADATA_UNSUPPORTED: &[u32] = &[1, 6, 8, 9];
/// Debug-info records: each defines one metadata id, decoded as `Md::Debug`. LOCATION,
/// GENERIC_DEBUG, SUBRANGE … FIXED_POINT_TYPE, excluding the non-DI codes in between.
pub(crate) const METADATA_DEBUG: &[u32] = &[
    7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    37, 40, 41, 44, 45, 46, 47, 48, 49,
];

// ---- METADATA_ATTACHMENT ----
pub(crate) const METADATA_ATTACHMENT: u32 = 11;

// ---- FUNCTION ----
pub(crate) const FUNC_DECLAREBLOCKS: u32 = 1;
pub(crate) const FUNC_BINOP: u32 = 2;
pub(crate) const FUNC_CAST: u32 = 3;
pub(crate) const FUNC_EXTRACTELT: u32 = 6;
pub(crate) const FUNC_INSERTELT: u32 = 7;
pub(crate) const FUNC_SHUFFLEVEC: u32 = 8;
/// `INST_CMP`: read exactly like `INST_CMP2` (LLVM's reader treats them alike).
pub(crate) const FUNC_CMP: u32 = 9;
pub(crate) const FUNC_RET: u32 = 10;
pub(crate) const FUNC_BR: u32 = 11;
pub(crate) const FUNC_SWITCH: u32 = 12;
pub(crate) const FUNC_UNREACHABLE: u32 = 15;
pub(crate) const FUNC_PHI: u32 = 16;
pub(crate) const FUNC_ALLOCA: u32 = 19;
pub(crate) const FUNC_LOAD: u32 = 20;
pub(crate) const FUNC_EXTRACTVAL: u32 = 26;
pub(crate) const FUNC_INSERTVAL: u32 = 27;
pub(crate) const FUNC_CMP2: u32 = 28;
pub(crate) const FUNC_VSELECT: u32 = 29;
pub(crate) const FUNC_DEBUG_LOC_AGAIN: u32 = 33;
pub(crate) const FUNC_CALL: u32 = 34;
pub(crate) const FUNC_DEBUG_LOC: u32 = 35;
pub(crate) const FUNC_FENCE: u32 = 36;
pub(crate) const FUNC_LOADATOMIC: u32 = 41;
pub(crate) const FUNC_GEP: u32 = 43;
pub(crate) const FUNC_STORE: u32 = 44;
pub(crate) const FUNC_STOREATOMIC: u32 = 45;
pub(crate) const FUNC_CMPXCHG: u32 = 46;
pub(crate) const FUNC_UNOP: u32 = 56;
pub(crate) const FUNC_FREEZE: u32 = 58;
pub(crate) const FUNC_ATOMICRMW: u32 = 59;
pub(crate) const FUNC_BLOCKADDR_USERS: u32 = 60;
/// Instruction codes LLVM defines outside the closed instruction set: GEP_OLD, SELECT (the
/// pre-`VSELECT` spelling), INVOKE, VAARG, STORE_OLD, INBOUNDS_GEP_OLD, INDIRECTBR, CMPXCHG_OLD,
/// ATOMICRMW_OLD, RESUME, LANDINGPAD_OLD, STOREATOMIC_OLD, LANDINGPAD, CLEANUPRET, CATCHRET,
/// CATCHPAD, CLEANUPPAD, CATCHSWITCH, OPERAND_BUNDLE, CALLBR.
pub(crate) const FUNC_UNSUPPORTED: &[u32] = &[
    4, 5, 13, 23, 24, 30, 31, 37, 38, 39, 40, 42, 47, 48, 49, 50, 51, 52, 55, 57,
];
/// Debug records (`DEBUG_RECORD_VALUE` … `DEBUG_RECORD_LABEL`, LLVM 19+): debug info, which the IR
/// does not keep, like `DEBUG_LOC`.
pub(crate) const FUNC_DEBUG_RECORDS: &[u32] = &[61, 62, 63, 64, 65];

// ---- VALUE_SYMTAB ----
pub(crate) const VST_ENTRY: u32 = 1;
pub(crate) const VST_BBENTRY: u32 = 2;
pub(crate) const VST_FNENTRY: u32 = 3;
/// `VST_CODE_COMBINED_ENTRY`: a summary-index entry, which a module symbol table never needs;
/// refused.
pub(crate) const VST_COMBINED_ENTRY: u32 = 5;

// ---- STRTAB / SYNC_SCOPE_NAMES ----
pub(crate) const STRTAB_BLOB: u32 = 1;
pub(crate) const SYNC_SCOPE_NAME: u32 = 1;
