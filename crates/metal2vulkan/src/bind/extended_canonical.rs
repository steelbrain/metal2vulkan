//! The canonical encoding that Metal's lossless compression writes of `BGRA10_XR` and
//! `BGRA10_XR_sRGB` texels, written in place over a buffer's texels by a compute dispatch (the
//! runtime's `compressed_blit.rs`, under the policy's `extended_range_blits` `Canonical`).
//!
//! **The encoding.** A texel is four 16-bit fields, blue, green, red and alpha in that order (two
//! little-endian words, blue and green, then red and alpha), each a 10-bit code in its top bits.
//! A blit that reads or writes a texture Metal keeps losslessly compressed writes each field with
//! its low six bits cleared and the alpha code clamped to `[384, 894]`
//! ([`extended_range_canonical`]); the texel reads the same value either way.
//!
//! **The module.** One `GLCompute` entry `main` with local size
//! ([`ExtendedCanonicalModule::LOCAL_SIZE`], 1, 1), [`ExtendedCanonicalModule::PUSH_BYTES`] of
//! `COMPUTE` push constants at offset 0 (the texels' shape, [`ExtendedCanonicalModule::shape`]),
//! and in set 0 one storage buffer ([`ExtendedCanonicalModule::TEXELS_BINDING`]) over the texels'
//! bytes, a whole number of 4-byte words.
//!
//! The shape's words, in order: `first`, the binding's word of the first texel; `width`, the
//! texels a row; `row_words` and `image_words`, the rows and the images apart in words.
//! Invocation `(x, y, z)` with `x` below `width` writes the canonical encoding of the two words at
//! `first + z · image_words + y · row_words + 2x` over them; one past `width` does nothing. A
//! dispatch takes `ceil(width / LOCAL_SIZE)` by the rows by the images workgroups
//! ([`ExtendedCanonicalModule::groups`]). No word between the rows, the images or past them is
//! written.
//!
//! **Synchronization** (consumer): the texels' earlier writes → `SHADER_STORAGE_READ` and
//! `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the dispatch; its writes → the texels' next
//! use.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, GlslOp, IoDir, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};

/// The canonical encoding of one `BGRA10_XR` texel's words `[blue | green << 16, red | alpha <<
/// 16]`: each field's low six bits cleared, the alpha code clamped to `[384, 894]` (the module
/// docs).
pub fn extended_range_canonical([low, high]: [u32; 2]) -> [u32; 2] {
    let alpha = (high >> 22).clamp(384, 894);
    [low & 0xFFC0_FFC0, (high & 0xFFC0) | (alpha << 22)]
}

/// The crate-owned canonical-encoding module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtendedCanonicalModule {
    /// SPIR-V 1.5 words, entry point `main`, its binding numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl ExtendedCanonicalModule {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The texels' bytes, a storage buffer the dispatch reads and writes in place.
    pub const TEXELS_BINDING: u32 = 0;
    /// Invocations of one workgroup, along x: one texel each, of one row.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the texels' shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of rows of `width` texels, the first texel at word `first` of the binding,
    /// the rows `row_words` and the images `image_words` apart (the module docs).
    pub fn shape(first: u32, width: u32, row_words: u32, image_words: u32) -> [u8; 16] {
        let words = [first, width, row_words, image_words];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The workgroups of a dispatch over `images` images of `rows` rows of `width` texels.
    pub fn groups(width: u32, rows: u32, images: u32) -> [u32; 3] {
        [width.div_ceil(Self::LOCAL_SIZE), rows, images]
    }
}

/// The binding symbol of the texels' bytes in the typed module.
const TEXELS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The canonical-encoding module as a typed module, before serialization: the module
/// [`extended_canonical_module`] serializes, which the CPU interpreter runs. Its binding is the
/// symbol `TEXELS`, and its push field the shape, a `uvec4` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    let texels = words_buffer(&mut b, TEXELS, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let z = f.composite_extract(invocation, &[2])?;
    let texels = f.module().global_value(texels)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let first = f.composite_extract(shape, &[0])?;
    let width = f.composite_extract(shape, &[1])?;
    let row_words = f.composite_extract(shape, &[2])?;
    let image_words = f.composite_extract(shape, &[3])?;
    let inside = f.binary(BinOp::ULessThan, x, width)?;
    let encode: ArmFn<'_, ()> = Box::new(move |f, _| {
        let one = u32_value(f, 1)?;
        let at = f.binary(BinOp::IMul, z, image_words)?;
        let at = f.binary(BinOp::IAdd, first, at)?;
        let row = f.binary(BinOp::IMul, y, row_words)?;
        let at = f.binary(BinOp::IAdd, at, row)?;
        let texel = f.binary(BinOp::ShiftLeftLogical, x, one)?;
        let low_at = f.binary(BinOp::IAdd, at, texel)?;
        let high_at = f.binary(BinOp::IAdd, low_at, one)?;
        let low_ptr = f.access_chain(texels, &[zero, low_at])?;
        let high_ptr = f.access_chain(texels, &[zero, high_at])?;
        let low = f.load(low_ptr, MemAccess::default())?;
        let high = f.load(high_ptr, MemAccess::default())?;
        // Each field's low six bits cleared; the alpha code clamped to [384, 894].
        let fields = u32_value(f, 0xFFC0_FFC0)?;
        let low = f.binary(BinOp::BitwiseAnd, low, fields)?;
        let red_field = u32_value(f, 0xFFC0)?;
        let red = f.binary(BinOp::BitwiseAnd, high, red_field)?;
        let shift = u32_value(f, 22)?;
        let alpha = f.binary(BinOp::ShiftRightLogical, high, shift)?;
        let least = u32_value(f, 384)?;
        let most = u32_value(f, 894)?;
        let alpha = f.glsl(GlslOp::UClamp, &[alpha, least, most])?;
        let alpha = f.binary(BinOp::ShiftLeftLogical, alpha, shift)?;
        let high = f.binary(BinOp::BitwiseOr, red, alpha)?;
        f.store(low_ptr, low, MemAccess::default())?;
        f.store(high_ptr, high, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, encode, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: ExtendedCanonicalModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ExtendedCanonicalModule::SET),
        Symbol::Binding(s) if *s == TEXELS => Some(ExtendedCanonicalModule::TEXELS_BINDING),
        _ => None,
    }
}

/// The canonical-encoding module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed binding of [`ExtendedCanonicalModule`].
pub fn extended_canonical_module(device: &Device) -> Result<ExtendedCanonicalModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::RequirementsExceedProfile,
            format!("the extended-range canonical encoding needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(ExtendedCanonicalModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{extended_range_canonical, typed_module, ExtendedCanonicalModule, TEXELS};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_canonical_kernel_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(typed_module().expect("module"), &p);
        }
    }

    /// Every word of a buffer after one dispatch over two images of three rows of five texels,
    /// the first texel at word 2, rows 14 words apart and images 50: each texel inside the shape
    /// its canonical encoding, every other word as it was, in both invocation orders.
    #[test]
    fn the_kernel_writes_each_texel_s_canonical_encoding_and_nothing_else() {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let words: Vec<u32> = (0..110).map(|_| next()).collect();
        let (first, width, row_words, image_words) = (2u32, 5u32, 14u32, 50u32);
        let mut want = words.clone();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..width {
                    let at = (first + z * image_words + y * row_words + 2 * x) as usize;
                    let [low, high] = extended_range_canonical([want[at], want[at + 1]]);
                    (want[at], want[at + 1]) = (low, high);
                }
            }
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = typed_module().expect("module");
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                workgroups: ExtendedCanonicalModule::groups(width, 3, 2),
                order,
                ..Config::default()
            };
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(TEXELS),
                    address: None,
                    bytes: bytes.clone(),
                }],
                push: ExtendedCanonicalModule::shape(first, width, row_words, image_words).to_vec(),
                spec: Default::default(),
            };
            let out = run(&m, &cfg, res).expect("run");
            let got: Vec<u32> = (out.buffers[0].as_chunks::<4>().0.iter())
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            assert_eq!(got, want, "{order:?}");
        }
    }

    /// The encoding clears each field's six low bits and clamps the alpha code alone.
    #[test]
    fn the_encoding_clears_the_low_bits_and_clamps_alpha() {
        assert_eq!(
            extended_range_canonical([0xFFFF_FFFF, 0xFFFF_FFFF]),
            [0xFFC0_FFC0, 0xDF80_FFC0]
        );
        assert_eq!(extended_range_canonical([0, 0]), [0, 384 << 22]);
        assert_eq!(
            extended_range_canonical([0, 600 << 22 | 0x3F]),
            [0, 600 << 22]
        );
    }
}
