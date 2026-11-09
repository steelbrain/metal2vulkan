//! 32-bit float depths clamped into `[0, 1]` in place over a buffer's words by a compute
//! dispatch, before a copy takes them into a depth image of a device that takes no other
//! (the runtime's `depth_clamp.rs`).
//!
//! **The clamp** ([`clamped_depth`]), on a depth's bits, so no denormal is flushed: a NaN of
//! either sign is 1.0 (`0x3f800000`), `-0.0` stays, every other value with its sign set is 0,
//! a value above 1.0 (`+inf` among them) is 1.0, and every other value stays. A copy into a depth
//! image of a device without `VK_EXT_depth_range_unrestricted` takes no value outside `[0, 1]`
//! (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), and such a device stores one clamped; the
//! heap's tiled texels clamp as they move into their image by this same clamp
//! ([`TiledLevelMove::clamps_depth`](super::TiledLevelMove::clamps_depth)).
//!
//! **The module.** One `GLCompute` entry `main` with local size
//! ([`DepthClampModule::LOCAL_SIZE`], 1, 1), [`DepthClampModule::PUSH_BYTES`] of `COMPUTE` push
//! constants at offset 0 (the depths' shape, [`DepthClampModule::shape`]), and in set 0 one
//! storage buffer ([`DepthClampModule::DEPTHS_BINDING`]) over the depths' bytes, a whole number of
//! 4-byte words.
//!
//! The shape's words, in order: `first`, the binding's word of the first depth; `width`, the
//! depths a row; `row_words` and `image_words`, the rows and the images apart in words.
//! Invocation `(x, y, z)` with `x` below `width` writes the clamp of the word at
//! `first + z · image_words + y · row_words + x` over it; one past `width` does nothing. A
//! dispatch takes `ceil(width / LOCAL_SIZE)` by the rows by the images workgroups
//! ([`DepthClampModule::groups`]). No word between the rows, the images or past them is written,
//! and the consumer's shape places no two depths in one word.
//!
//! **Synchronization** (consumer): the depths' earlier writes → `SHADER_STORAGE_READ` and
//! `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the dispatch; its writes → the copy's
//! `TRANSFER_READ`.

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
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// The bits of 32-bit float depth `bits` clamped into `[0, 1]` (the module docs): a NaN 1.0,
/// `-0.0` as it is, another negative value 0, a value above 1.0 1.0, and the rest as they are.
pub const fn clamped_depth(bits: u32) -> u32 {
    const ONE: u32 = 0x3f80_0000;
    const SIGN: u32 = 0x8000_0000;
    if bits & !SIGN > 0x7f80_0000 {
        ONE
    } else if bits == SIGN {
        bits
    } else if bits > SIGN {
        0
    } else if bits > ONE {
        ONE
    } else {
        bits
    }
}

/// The crate-owned depth-clamp module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthClampModule {
    /// SPIR-V 1.5 words, entry point `main`, its binding numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl DepthClampModule {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The depths' bytes, a storage buffer the dispatch reads and writes in place.
    pub const DEPTHS_BINDING: u32 = 0;
    /// Invocations of one workgroup, along x: one depth each, of one row.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the depths' shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of rows of `width` depths, the first depth at word `first` of the binding,
    /// the rows `row_words` and the images `image_words` apart (the module docs).
    pub fn shape(first: u32, width: u32, row_words: u32, image_words: u32) -> [u8; 16] {
        let words = [first, width, row_words, image_words];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The workgroups of a dispatch over `images` images of `rows` rows of `width` depths.
    pub fn groups(width: u32, rows: u32, images: u32) -> [u32; 3] {
        [width.div_ceil(Self::LOCAL_SIZE), rows, images]
    }
}

/// The binding symbol of the depths' bytes in the typed module.
const DEPTHS: ResourceSlot = ResourceSlot {
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

/// The word `v`, a 32-bit float depth's bits, clamped into `[0, 1]` as [`clamped_depth`] clamps
/// it, compared as the word's bits.
pub(crate) fn clamped_depth_word(f: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
    let zero = u32_value(f, 0)?;
    let one = u32_value(f, 0x3f80_0000)?;
    let sign = u32_value(f, 0x8000_0000)?;
    let infinity = u32_value(f, 0x7f80_0000)?;
    let magnitude = u32_value(f, 0x7fff_ffff)?;
    let abs = f.binary(BinOp::BitwiseAnd, v, magnitude)?;
    let nan = f.binary(BinOp::UGreaterThan, abs, infinity)?;
    // Above 1.0's bits: a positive above 1.0, and every value with its sign bit, of which -0.0
    // alone stays.
    let above = f.binary(BinOp::UGreaterThan, v, one)?;
    let signed = f.binary(BinOp::UGreaterThanEqual, v, sign)?;
    let minus_zero = f.binary(BinOp::IEqual, v, sign)?;
    let r = f.select(above, one, v)?;
    let r = f.select(signed, zero, r)?;
    let r = f.select(minus_zero, v, r)?;
    f.select(nan, one, r)
}

/// The depth-clamp module as a typed module, before serialization: the module
/// [`depth_clamp_module`] serializes, which the CPU interpreter runs. Its binding is the symbol
/// `DEPTHS`, and its push field the shape, a `uvec4` (the module docs).
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
    let depths = words_buffer(&mut b, DEPTHS, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let z = f.composite_extract(invocation, &[2])?;
    let depths = f.module().global_value(depths)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let first = f.composite_extract(shape, &[0])?;
    let width = f.composite_extract(shape, &[1])?;
    let row_words = f.composite_extract(shape, &[2])?;
    let image_words = f.composite_extract(shape, &[3])?;
    let inside = f.binary(BinOp::ULessThan, x, width)?;
    let clamp: ArmFn<'_, ()> = Box::new(move |f, _| {
        let at = f.binary(BinOp::IMul, z, image_words)?;
        let at = f.binary(BinOp::IAdd, first, at)?;
        let row = f.binary(BinOp::IMul, y, row_words)?;
        let at = f.binary(BinOp::IAdd, at, row)?;
        let at = f.binary(BinOp::IAdd, at, x)?;
        let ptr = f.access_chain(depths, &[zero, at])?;
        let word = f.load(ptr, MemAccess::default())?;
        let word = clamped_depth_word(f, word)?;
        f.store(ptr, word, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, clamp, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: DepthClampModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(DepthClampModule::SET),
        Symbol::Binding(s) if *s == DEPTHS => Some(DepthClampModule::DEPTHS_BINDING),
        _ => None,
    }
}

/// The depth-clamp module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed binding of [`DepthClampModule`].
pub fn depth_clamp_module(device: &Device) -> Result<DepthClampModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::RequirementsExceedProfile,
            format!("the depth clamp needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(DepthClampModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{clamped_depth, typed_module, DepthClampModule, DEPTHS};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_depth_clamp_kernel_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(typed_module().expect("module"), &p);
        }
    }

    /// The clamp of each kind of value: inside `[0, 1]` (a denormal and `-0.0` among them) as it
    /// is, a NaN of either sign 1.0, an infinity or a value above 1.0 1.0, and every other
    /// negative value 0.
    #[test]
    fn the_clamp_keeps_unit_range_bits_and_clamps_the_rest() {
        let one = 1.0f32.to_bits();
        for (bits, want) in [
            (0, 0),
            ((-0.0f32).to_bits(), (-0.0f32).to_bits()),
            (1, 1),
            (0.5f32.to_bits(), 0.5f32.to_bits()),
            (one, one),
            (1.5f32.to_bits(), one),
            (f32::INFINITY.to_bits(), one),
            (0x7fc0_0000, one),
            (0x7f80_0001, one),
            (0xffc0_0000, one),
            ((-1.0f32).to_bits(), 0),
            (f32::NEG_INFINITY.to_bits(), 0),
            (0x8000_0001, 0),
            (0xeeee_eeee, 0),
        ] {
            assert_eq!(clamped_depth(bits), want, "{bits:#010x}");
        }
    }

    /// Every word of a buffer after one dispatch over two images of three rows of seventy depths,
    /// across workgroups, the first depth at word 2, rows 75 words apart and images 230: each
    /// depth inside the shape its clamp, every other word as it was, in both invocation orders.
    #[test]
    fn the_kernel_clamps_each_depth_of_the_shape_and_nothing_else() {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let (first, width, row_words, image_words) = (2u32, 70u32, 75u32, 230u32);
        // Random words, and every eighth a depth inside [0, 1], which stays.
        let words: Vec<u32> = (0..470)
            .map(|i| match i % 8 {
                0 => (i as f32 / 470.0).to_bits(),
                _ => next(),
            })
            .collect();
        let mut want = words.clone();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..width {
                    let at = (first + z * image_words + y * row_words + x) as usize;
                    want[at] = clamped_depth(want[at]);
                }
            }
        }
        assert_ne!(want, words, "the words hold depths the clamp changes");
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = typed_module().expect("module");
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                workgroups: DepthClampModule::groups(width, 3, 2),
                order,
                ..Config::default()
            };
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(DEPTHS),
                    address: None,
                    bytes: bytes.clone(),
                }],
                push: DepthClampModule::shape(first, width, row_words, image_words).to_vec(),
                spec: Default::default(),
            };
            let out = run(&m, &cfg, res).expect("run");
            let got: Vec<u32> = (out.buffers[0].as_chunks::<4>().0.iter())
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            assert_eq!(got, want, "{order:?}");
        }
    }
}
