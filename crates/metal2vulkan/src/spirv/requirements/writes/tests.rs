//! [`writes_memory`] on hand-built fragment modules.

use super::writes_memory;
use crate::bind::status::{StatusBits, StatusWord};
use crate::pipeline::sym::{
    PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, Stage, SyntheticBinding,
};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ConvOp, Dim, ExecModel, ImageOperands,
    ImageType, ImageUsage, LayoutRule, MemAccess, Scope, Semantics, StorageClass, TexelMemory,
    Type, TypedModule,
};

/// What a fragment writes beside the status words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Guest {
    /// Nothing: it loads through a slot's guest base.
    Reads,
    /// A store 16 bytes past a slot's guest base, a `uvec2` push member whose address is a
    /// bitcast, so the trace stops at the bitcast.
    Stores,
    /// A store in the miss count's own shape, `ConvertUToPtr(IAdd(load of a u64 push member,
    /// constant))`, through the push block's `RootTableAddress` member: the trace reaches the push
    /// block's load, and only the member index tells it from the status address.
    StoresThroughAnotherAddress,
}

fn int(b: &mut ModuleBuilder, width: u8) -> TypeId {
    b.ty(Type::Int {
        width,
        signed: false,
    })
    .expect("int")
}

fn k(f: &mut FunctionBuilder<'_>, ty: TypeId, v: u64) -> Value {
    let c = f
        .module()
        .constant(Constant::Scalar {
            ty,
            bits: ConstBits(v),
        })
        .expect("constant");
    f.module().const_value(c).expect("value")
}

/// Push member `index`'s value.
fn field(f: &mut FunctionBuilder<'_>, push: GlobalId, index: u32) -> Value {
    let pb = f.module().global_value(push).expect("push");
    let i = f.u32_const(index).expect("index");
    let i = f.module().const_value(i).expect("index");
    let p = f.access_chain(pb, &[i]).expect("member");
    f.load(p, MemAccess::default()).expect("load")
}

/// The module of fragment entry `main`.
fn fragment(mut b: ModuleBuilder, main: FuncId) -> TypedModule {
    b.entry(ExecModel::Fragment, main, Vec::new())
        .expect("entry");
    b.finalize().expect("finalize")
}

/// A fragment that counts a resident miss as `lower::access::dynamic` does (`ResidentMisses += 1`
/// and `Status |= RESIDENT_MISS`, relaxed device atomics at a constant offset from the push
/// block's `ExecutionStatusAddress`), then reads through a slot's guest base and writes as `guest`
/// says. Its push members follow the kind order: `RootTableAddress`, `ExecutionStatusAddress`, then
/// the slot's `SlotGuestBase`.
fn miss_counting(guest: Guest) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32);
    let u64_ = int(&mut b, 64);
    let root = b
        .push_member(PushField::RootTableAddress, u64_, 0)
        .expect("root field");
    let status = b
        .push_member(PushField::ExecutionStatusAddress, u64_, 8)
        .expect("status field");
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Buffer,
        index: 0,
    };
    // `SlotGuestBase` is a `uvec2`, low word first.
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 }).expect("uvec2");
    let base = b
        .push_member(PushField::SlotGuestBase(slot), uvec2, 16)
        .expect("base field");
    let push = b.push_global().expect("push block");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("main");
    let at = field(&mut f, push, status.index);
    for (word, op, value) in [
        (StatusWord::ResidentMisses, AtomicOp::IAdd, 1),
        (
            StatusWord::Status,
            AtomicOp::Or,
            StatusBits::RESIDENT_MISS.0,
        ),
    ] {
        let off = k(&mut f, u64_, u64::from(word.byte_offset()));
        let a = f.binary(BinOp::IAdd, at, off).expect("add");
        let p = f.device_place(a, u32_).expect("place");
        let v = k(&mut f, u32_, u64::from(value));
        f.atomic(op, p, Scope::Device, Semantics::RELAXED, v)
            .expect("atomic");
    }
    let words = field(&mut f, push, base.index);
    let addr = f.convert(ConvOp::Bitcast, words, u64_).expect("address");
    let sixteen = k(&mut f, u64_, 16);
    let addr = f.binary(BinOp::IAdd, addr, sixteen).expect("add");
    let p = f.device_place(addr, u32_).expect("place");
    let access = MemAccess {
        aligned: Some(4),
        ..MemAccess::default()
    };
    let v = f.load(p, access).expect("load");
    match guest {
        Guest::Reads => {}
        Guest::Stores => f.store(p, v, access).expect("store"),
        Guest::StoresThroughAnotherAddress => {
            let other = field(&mut f, push, root.index);
            let eight = k(&mut f, u64_, 8);
            let a = f.binary(BinOp::IAdd, other, eight).expect("add");
            let q = f.device_place(a, u32_).expect("place");
            f.store(q, v, access).expect("store");
        }
    }
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    fragment(b, main)
}

/// A fragment whose only write is an atomic on word `Status` of the `Synthetic(ExecutionStatus)`
/// storage buffer, the status buffer's binding.
fn status_binding() -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32);
    let rta = b
        .ty(Type::RuntimeArray {
            elem: u32_,
            stride: Some(4),
        })
        .expect("array");
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
    let block = b
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
        index: 0,
    };
    let g = b
        .resource(slot, StorageClass::StorageBuffer, block)
        .expect("binding");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("main");
    let gv = f.module().global_value(g).expect("binding");
    let zero = k(&mut f, u32_, 0);
    let word = k(&mut f, u32_, u64::from(StatusWord::Status.index()));
    let p = f.access_chain(gv, &[zero, word]).expect("word");
    let bit = k(&mut f, u32_, u64::from(StatusBits::RESIDENT_MISS.0));
    f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)
        .expect("atomic");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    fragment(b, main)
}

/// The `ExecutionStatus` buffer's words are the product's. A fragment whose only writes are the
/// resident-miss count's, through the push block's status address, or a status word through the
/// status binding writes no memory of its own. One that also stores through a slot's guest base (a
/// `uvec2` push member bitcast to an address, whose trace ends at the bitcast) does, and so does
/// one that stores in the count's own shape through the push block's other `u64` address,
/// `RootTableAddress`, whose trace reaches the push block's load and is told from the status
/// address by its member index alone.
#[test]
fn the_status_words_are_not_the_functions_writes() {
    assert!(!writes_memory(&miss_counting(Guest::Reads)).expect("reads"));
    assert!(writes_memory(&miss_counting(Guest::Stores)).expect("stores"));
    assert!(writes_memory(&miss_counting(Guest::StoresThroughAnotherAddress)).expect("root"));
    assert!(!writes_memory(&status_binding()).expect("binding"));
}

/// A fragment whose only write is an image write of a `uint` 2D storage image bound as `kind`.
fn image_write(kind: ResourceKind) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32);
    let ty = b
        .ty(Type::Image(ImageType {
            sampled_type: u32_,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage: ImageUsage::Storage,
            format: crate::spv::ImageFormat::R8ui,
        }))
        .expect("image");
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind,
        index: 0,
    };
    let g = b
        .resource(slot, StorageClass::UniformConstant, ty)
        .expect("binding");
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 }).expect("uvec2");
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 }).expect("uvec4");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("main");
    let gv = f.module().global_value(g).expect("binding");
    let image = f.load(gv, MemAccess::default()).expect("image");
    let zero = k(&mut f, u32_, 0);
    let coord = f.composite_construct(uvec2, &[zero, zero]).expect("coord");
    let texel = f
        .composite_construct(uvec4, &[zero, zero, zero, zero])
        .expect("texel");
    f.write(
        image,
        coord,
        texel,
        &ImageOperands::default(),
        TexelMemory::Private { volatile: false },
    )
    .expect("write");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    fragment(b, main)
}

/// The shader-side stencil's shadow is the product's, so its writes do not make the function run
/// per sample; a write of another storage image (the function's texture) does.
#[test]
fn the_shadow_stencils_writes_are_not_the_functions() {
    let shadow = ResourceKind::Synthetic(SyntheticBinding::ShadowStencil);
    assert!(!writes_memory(&image_write(shadow)).expect("shadow"));
    let imageblock = ResourceKind::Synthetic(SyntheticBinding::Imageblock);
    assert!(writes_memory(&image_write(imageblock)).expect("another image"));
}

/// A color's interlocked storage form is the product's, carrying Metal's color attachment write,
/// so its writes are not the function's; a texture's write is.
#[test]
fn a_colors_storage_form_writes_are_not_the_functions() {
    let color = ResourceKind::TileAttachment(2);
    assert!(!writes_memory(&image_write(color)).expect("color"));
    assert!(writes_memory(&image_write(ResourceKind::Texture)).expect("a texture"));
}

/// A module that stores a record at a slot the draw computes: a `vec4` at `address + slot * 32`
/// and an `f32` 16 bytes past it, `slot` an input's value (not a constant), `address` the push
/// block's member of `address`.
fn record_store(address: PushField) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32);
    let u64_ = int(&mut b, 64);
    let member = b.push_member(address, u64_, 0).expect("address field");
    let slot = b
        .push_member(PushField::ViewportCount, u32_, 8)
        .expect("slot");
    let push = b.push_global().expect("push block");
    let f32_ = b.ty(Type::Float { width: 32 }).expect("f32");
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 }).expect("vec4");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("main");
    let base = field(&mut f, push, member.index);
    let slot = field(&mut f, push, slot.index);
    let slot = f.convert(ConvOp::UConvert, slot, u64_).expect("wide");
    let bytes = k(&mut f, u64_, 32);
    let offset = f.binary(BinOp::IMul, slot, bytes).expect("offset");
    let record = f.binary(BinOp::IAdd, base, offset).expect("record");
    let p = f.device_place(record, vec4).expect("place");
    let zero = k(&mut f, f32_, 0);
    let clip = f
        .composite_construct(vec4, &[zero, zero, zero, zero])
        .expect("clip");
    let aligned = |n| MemAccess {
        aligned: Some(n),
        ..MemAccess::default()
    };
    f.store(p, clip, aligned(16)).expect("store");
    let sixteen = k(&mut f, u64_, 16);
    let at = f.binary(BinOp::IAdd, record, sixteen).expect("add");
    let p = f.device_place(at, f32_).expect("place");
    f.store(p, zero, aligned(4)).expect("store");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    fragment(b, main)
}

/// Stores at the push block's `ExecutionStatusAddress` plus an offset that is not a constant are
/// the product's; the same stores from another `u64` address member (`RootTableAddress`, whose
/// trace reaches the push block's load) are the function's.
#[test]
fn stores_past_the_status_address_are_not_the_functions_writes() {
    assert!(!writes_memory(&record_store(PushField::ExecutionStatusAddress)).expect("status"));
    assert!(writes_memory(&record_store(PushField::RootTableAddress)).expect("another address"));
}
