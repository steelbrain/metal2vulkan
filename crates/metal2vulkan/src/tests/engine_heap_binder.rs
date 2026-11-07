//! The binder half of the handle domain. No device is opened.
//!
//! Heap pipelines go through the public compile API on `moltenvk-m3`, `ComputePipeline::plan` or
//! `RenderPipeline::plan`, and `instantiate_dispatch` or `instantiate_draw`:
//!
//! - One element per heap member (used image textures by token, then bound image textures),
//!   shared by every class array: element 0 and every member a class does not hold are the
//!   class's neutral; a held member is written through its view (sampled or storage, one layout per
//!   member). Texture buffers, used buffers, and used textures reaching no stage with a table are
//!   not members. A member bound at a slot shares one layout with the slot's own writes (`GENERAL`
//!   when either is a storage image) and one use.
//! - Each stage's handle table has one entry per member reaching it plus the null entry; per
//!   draw or dispatch its bytes are the members' nonzero resource ids, deduplicated, sorted, after
//!   leading null entries. `lookup_host` finds each id's element in them, and so does a call of
//!   the emitted search (`memory::handles::define_lookup`) on the CPU interpreter, fed the
//!   binder's own bytes: the end-to-end path bind state, table bytes, emitted search, element.
//!   Each entry's flags are its member's out-of-bounds-alpha flag (its pixel format has no
//!   alpha), which the search's alpha form carries in the handle's bit 31.
//! - More members than the heap arrays hold is `HeapTooLarge`; a storage member the heap's
//!   formatless class cannot view is a typed refusal.
//! - A heap texture sampled through a constexpr sampler reads a set-0 heap sampler, the layout's
//!   immutable sampler, which the plan does not write; through a runtime sampler, a heap sampler
//!   the plan writes with the sampler bound at its slot. An opaque-black border is checked against
//!   every member the sampled arrays hold, and a linear filter against the view format of every
//!   member of its stage the sampled float arrays hold.
//! - A sampler loaded from an argument buffer reads the set-1 sampler array of its use at the
//!   element its stage's sampler table gives: the plan writes twice the shape's capacity class of
//!   `ArgumentSampler` elements, the table bytes list `BindState::argument_samplers` by id with
//!   their state word (`metal::sampler::SamplerWord`), the emitted search's entry form finds each
//!   id's entry, and a sampler made or released replans only when the class or the set of shapes
//!   changes; listed again as a command buffer commits, the states fit the plan up to its entries,
//!   a new shape where the plan's checks pass.
//! - Loads of one argument-buffer member share a heap group, so a read of the member types its
//!   queries (never the MSL type name); a group only queried is declared `Float` with
//!   `RuntimeFact::HeapNumericType`, and its class reports a member of another numeric type
//!   (`IncompatibleHeapEntry`).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{
    AccessClass, ArgumentSampler, BindState, BindingPlan, BoundSampler, BoundTexture,
    DescriptorBytes, Dispatch, DrawCall, HeapSource, NeutralResource, PlanDiagnostic,
    ResourceClass, ResourceUse, SamplerDescriptor, StageTable, TextureFacts, UseTarget,
    UsedResource, WriteSource,
};
use crate::budget::Budget;
use crate::device::requirements::FormatUsage;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Capability, Feature, Needs, Unsupported};
use crate::library::{FunctionArtifact, Library};
use crate::memory::handles::{
    declare_table, define_lookup, lookup_host, Lookup, FLAG_OUT_OF_BOUNDS_ALPHA_ONE,
    FLAG_TEXEL_CLASS, HANDLE_ALPHA_ONE, TABLE_ENTRY_BYTES,
};
use crate::memory::leaf::k32;
use crate::memory::views;
use crate::metal::format::storage_image_choice;
use crate::metal::sampler::SamplerWord;
use crate::metal::{
    PixelFormat, RenderStages, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    TextureType, TextureUsage,
};
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{
    HandleClass, ImageAccess, ImageHandleClass, ResourceKind, ResourceSlot, SampledType, SlotClass,
    SlotRef, Stage, SyntheticBinding,
};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, AssembleError,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, RenderPipeline,
    RenderPipelineDescriptor, StageSource,
};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::{ExecMode, ExecModel, MemAccess, Type};
use crate::vk;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

fn assembled(name: &str, r: Result<Vec<u8>, AssembleError>) -> Option<Vec<u8>> {
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-heap-binder-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    assembled(name, r)
}

/// The committed `moltenvk-m3` profile as a device.
fn m3() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn compute(dev: &Device, bc: &[u8], entry: &str) -> ComputePipeline {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    dev.compile_compute_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{entry}: {e}"))
}

/// `argument_buffer_read` of the owned `handles` fixture: one sampled `float` 2D heap class.
fn read_pipeline(dev: &Device) -> Option<ComputePipeline> {
    let bc = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll")))?;
    Some(compute(dev, &bc, "argument_buffer_read"))
}

/// A used texture: token `token`, `ty`/`pf`, `usage`, resource id `id`.
fn used(
    token: u32,
    ty: TextureType,
    pf: PixelFormat,
    usage: TextureUsage,
    id: Option<u64>,
) -> UsedResource {
    let mut facts = TextureFacts::new(ty, pf);
    facts.usage = usage;
    if matches!(
        ty,
        TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray
    ) {
        facts.sample_count = 4;
    }
    let mut u = UsedResource::new(ResourceClass::Texture(facts), token);
    u.resource_id = id;
    u
}

fn read() -> TextureUsage {
    TextureUsage::SHADER_READ
}

fn bound(pf: PixelFormat, id: Option<u64>) -> BoundTexture {
    BoundTexture::Texture {
        facts: TextureFacts::new(TextureType::Type2D, pf),
        resource_id: id,
    }
}

fn slot(stage: Stage, index: u32) -> SlotRef {
    SlotRef {
        stage,
        class: SlotClass::Texture,
        index,
    }
}

/// The heap writes of `plan`'s float array, by element ([`heap_writes_of`]).
fn heap_writes(plan: &BindingPlan) -> Vec<&WriteSource> {
    heap_writes_of(plan, SampledType::Float)
}

/// The heap writes of `plan`, by element: one heap array; but for the classes of its shape of
/// other numeric types beside a sampled class (the unsigned integer twin a float read or query
/// declares, and the classes a read or query of a member of another numeric class reads it
/// through), where the array is the one whose neutral at element 0 is of numeric type `own`: a
/// sampled one, or where there is none a storage one.
fn heap_writes_of(plan: &BindingPlan, own: SampledType) -> Vec<&WriteSource> {
    let heap: Vec<_> = plan.descriptors.iter().filter(|d| d.set == 1).collect();
    let numeric = |binding: u32, storage: bool| {
        heap.iter().find_map(|d| match d.source {
            WriteSource::Neutral(NeutralResource::Image {
                sampled_type,
                storage_format,
                ..
            }) if d.binding == binding
                && d.array_element == 0
                && storage_format.is_some() == storage =>
            {
                Some(sampled_type)
            }
            _ => None,
        })
    };
    let bindings: std::collections::BTreeSet<u32> = heap.iter().map(|d| d.binding).collect();
    let of = |storage: bool| -> Vec<u32> {
        bindings
            .iter()
            .copied()
            .filter(|&b| numeric(b, storage) == Some(own))
            .collect()
    };
    let own: Vec<u32> = match bindings.len() {
        1 => bindings.iter().copied().collect(),
        _ => match of(false) {
            sampled if sampled.is_empty() => of(true),
            sampled => sampled,
        },
    };
    assert_eq!(own.len(), 1, "one heap array: {bindings:?}");
    let mut w: Vec<_> = heap.into_iter().filter(|d| d.binding == own[0]).collect();
    w.sort_by_key(|d| d.array_element);
    for (e, d) in w.iter().enumerate() {
        assert_eq!(d.array_element as usize, e, "one write per element");
    }
    w.into_iter().map(|d| &d.source).collect()
}

/// The `(set, binding, bytes_len)` of `stage`'s handle-table write.
fn table_write(plan: &BindingPlan, stage: Stage) -> (u32, u32, u32) {
    plan.descriptors
        .iter()
        .find_map(|d| match d.source {
            WriteSource::HandleTable {
                stage: s,
                bytes_len,
            } if s == stage => Some((d.set, d.binding, bytes_len)),
            _ => None,
        })
        .expect("a handle-table write")
}

/// The entries `(id, element)` of the table bytes at `(set, binding)`, checking the flags word.
fn table(bytes: &[DescriptorBytes], set: u32, binding: u32) -> Vec<(u64, u32)> {
    entries(bytes, set, binding)
        .into_iter()
        .map(|(id, element, _)| (id, element))
        .collect()
}

/// The entries `(id, element, flags)` of the table bytes at `(set, binding)`: a texture entry's
/// flags hold its out-of-bounds-alpha flag and its texel class's index alone, and the null entry's
/// none.
fn entries(bytes: &[DescriptorBytes], set: u32, binding: u32) -> Vec<(u64, u32, u32)> {
    let b = &bytes
        .iter()
        .find(|b| b.set == set && b.binding == binding)
        .expect("table bytes")
        .bytes;
    assert_eq!(b.len() % TABLE_ENTRY_BYTES as usize, 0);
    b.chunks(TABLE_ENTRY_BYTES as usize)
        .map(|e| {
            let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().expect("word"));
            let id = u64::from(w(0)) | (u64::from(w(1)) << 32);
            assert_eq!(
                w(3) & !(FLAG_OUT_OF_BOUNDS_ALPHA_ONE | FLAG_TEXEL_CLASS),
                0,
                "other flags are reserved"
            );
            if id == 0 {
                assert_eq!(w(3), 0, "the null entry has no flags");
            }
            (id, w(2), w(3))
        })
        .collect()
}

/// The source and view format of a held element, or `None` for a neutral.
fn held(w: &WriteSource) -> Option<(HeapSource, vk::Format, vk::ImageLayout)> {
    match w {
        WriteSource::HeapEntry { source, view, .. } => Some((*source, view.format, view.layout)),
        WriteSource::Neutral(_) => None,
        other => panic!("a heap write {other:?}"),
    }
}

const KEYS: u32 = 8;

/// A kernel that looks up the [`KEYS`] ids (low, high word pairs) at the start of the buffer at
/// slot 1 in the handle table at slot 0 by calling the emitted search function, as every heap
/// handle load does, and writes each handle (in `form`) after them. The table is an entry array
/// read as `u64` ids when `int64`, else a word view.
fn lookup_module(int64: bool, form: Lookup) -> crate::spirv::ir::TypedModule {
    let sym = |index| ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    };
    let mut b = ModuleBuilder::new();
    let t = declare_table(&mut b, sym(0), int64).expect("table");
    let io = views::declare_at(&mut b, sym(1), views::ViewKind::Storage).expect("io");
    let search = define_lookup(&mut b, t, form).expect("search");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    for i in 0..KEYS {
        let at_lo = k32(&mut f, 2 * i).expect("k");
        let at_hi = k32(&mut f, 2 * i + 1).expect("k");
        let lo = views::read_word(&mut f, io, at_lo).expect("lo");
        let hi = views::read_word(&mut f, io, at_hi).expect("hi");
        let e = f
            .call(search, &[lo, hi])
            .expect("call")
            .expect("the search returns the element");
        let at = k32(&mut f, 2 * KEYS + i).expect("k");
        let p = views::word_ptr(&mut f, io, at)
            .expect("ptr")
            .expect("a storage view is writable");
        f.store(p, e, MemAccess::default()).expect("store");
    }
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("module")
}

/// The elements the emitted search finds for `keys` in the table `bytes`, on the interpreter:
/// the same through an entry array and a word view.
fn emitted_lookup(bytes: &[u8], keys: &[u64]) -> Vec<u32> {
    emitted_lookup_in(bytes, keys, Lookup::Element)
}

/// The handles the emitted search in `form` finds for `keys` in the table `bytes`, on the
/// interpreter: the same through an entry array and a word view.
fn emitted_lookup_in(bytes: &[u8], keys: &[u64], form: Lookup) -> Vec<u32> {
    let wide = emitted_lookup_as(bytes, keys, true, form);
    assert_eq!(
        wide,
        emitted_lookup_as(bytes, keys, false, form),
        "u64 and word ids"
    );
    wide
}

fn emitted_lookup_as(bytes: &[u8], keys: &[u64], int64: bool, form: Lookup) -> Vec<u32> {
    let m = lookup_module(int64, form);
    let sym = |index| ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    };
    let mut out = Vec::new();
    for chunk in keys.chunks(KEYS as usize) {
        let mut io = Vec::new();
        for i in 0..KEYS as usize {
            let id = chunk.get(i).copied().unwrap_or(0);
            io.extend((id as u32).to_le_bytes());
            io.extend(((id >> 32) as u32).to_le_bytes());
        }
        io.extend(vec![0xAA; 4 * KEYS as usize]);
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(sym(0)),
                    address: None,
                    bytes: bytes.to_vec(),
                },
                Buffer {
                    slot: Some(sym(1)),
                    address: None,
                    bytes: io,
                },
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let o = run(&m, &Config::default(), res).expect("run");
        let io = &o.buffers[1];
        for i in 0..chunk.len() {
            let at = 4 * (2 * KEYS as usize + i);
            out.push(u32::from_le_bytes(io[at..at + 4].try_into().expect("w")));
        }
    }
    out
}

const A: u64 = 0x0000_0001_0000_0010;
const B: u64 = 0x0000_0001_0000_0020;
const C: u64 = 0x0000_0002_0000_0000;
const D: u64 = 0x0000_0000_ffff_fff0;
const E: u64 = 0xffff_ffff_ffff_fff0;
const F: u64 = 0x0000_0003_0000_0001;

/// A sampled heap: every used image texture and every bound one is a member with one element;
/// the class holds exactly the `float` 2D `ShaderRead` members through their sampled views; the
/// table's bytes resolve every resident id to its element, by `lookup_host` and by the emitted
/// search on the interpreter, and every other id (0, a texture buffer's, an absent one) to 0.
#[test]
fn a_sampled_heap_writes_each_member_and_its_table() {
    let dev = m3();
    let Some(pipe) = read_pipeline(&dev) else {
        return;
    };
    let mut state = BindState {
        used_resources: vec![
            used(
                1,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(A),
            ),
            // Another numeric type, no shader read, another type: other classes' members.
            used(
                2,
                TextureType::Type2D,
                PixelFormat::R32Uint,
                read(),
                Some(B),
            ),
            used(
                3,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
                Some(C),
            ),
            used(
                4,
                TextureType::Type2DArray,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(D),
            ),
            UsedResource::new(ResourceClass::Buffer, 5),
            // No resource id, the null id, and the id of token 1 again.
            used(
                6,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                None,
            ),
            used(
                7,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(0),
            ),
            used(
                8,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(A),
            ),
            // A texture buffer is no member.
            used(
                9,
                TextureType::TextureBuffer,
                PixelFormat::R32Float,
                read(),
                Some(E),
            ),
        ],
        ..Default::default()
    };
    let mut k = StageTable::default();
    k.textures
        .insert(3, bound(PixelFormat::BGRA8Unorm, Some(F)));
    state.stages.insert(Stage::Kernel, k);

    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    // Members: tokens 1, 2, 3, 4, 6, 7, 8 (elements 1..=7), then [[texture(3)]] (element 8).
    let (set, binding, len) = table_write(&plan, Stage::Kernel);
    assert_eq!(set, 0);
    assert_eq!(len, 9 * TABLE_ENTRY_BYTES);
    let w = heap_writes(&plan);
    assert_eq!(w.len(), 9);
    let neutral = match w[0] {
        WriteSource::Neutral(n @ NeutralResource::Image { .. }) => *n,
        other => panic!("element 0 {other:?}"),
    };
    assert!(matches!(
        neutral,
        NeutralResource::Image {
            sampled_type: SampledType::Float,
            storage_format: None,
            ..
        }
    ));
    let rgba = vk::Format::R8G8B8A8_UNORM;
    let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    let expected = [
        None,
        Some((HeapSource::Used(1), rgba, ro)),
        None,
        None,
        None,
        Some((HeapSource::Used(6), rgba, ro)),
        Some((HeapSource::Used(7), rgba, ro)),
        Some((HeapSource::Used(8), rgba, ro)),
        Some((
            HeapSource::Bound(slot(Stage::Kernel, 3)),
            vk::Format::B8G8R8A8_UNORM,
            ro,
        )),
    ];
    for (e, (got, want)) in w.iter().zip(expected).enumerate() {
        assert_eq!(held(got), want, "element {e}");
        if want.is_none() {
            assert_eq!(**got, WriteSource::Neutral(neutral), "element {e}");
        }
    }
    for got in &w {
        if let WriteSource::HeapEntry { view, class, .. } = got {
            assert_eq!(view.view_type, vk::ImageViewType::TYPE_2D);
            assert_eq!(view.aspect, vk::ImageAspectFlags::COLOR);
            assert!(matches!(class, HandleClass::Image(c) if c.access == ImageAccess::Sampled));
        }
    }
    // A typed class reports no member of another numeric type.
    assert!(plan.diagnostics.is_empty(), "{:?}", plan.diagnostics);
    // The bound member is used through the heap: read, sampled, in the kernel.
    assert!(plan.uses.iter().any(|u| {
        u.resource == UseTarget::Slot(slot(Stage::Kernel, 3))
            && u.access.read
            && !u.access.write
            && u.stages == vk::PipelineStageFlags2::COMPUTE_SHADER
    }));

    let cmds = plan
        .instantiate_dispatch(
            &state,
            &Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [1, 1, 1],
            },
        )
        .expect("dispatch");
    let t = table(&cmds.descriptor_bytes, set, binding);
    // Five resident ids (token 8's is token 1's, the lower element wins), after four leading
    // null entries that pad the table to its planned nine.
    assert_eq!(
        t,
        vec![
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (D, 4),
            (A, 1),
            (B, 2),
            (C, 3),
            (F, 8),
        ]
    );
    let keys = [A, B, C, D, F, 0, E, A + 1, u64::MAX, 1];
    let want = [1, 2, 3, 4, 8, 0, 0, 0, 0, 0];
    for (id, e) in keys.iter().zip(want) {
        assert_eq!(lookup_host(&t, *id), e, "id {id:#x}");
    }
    let bytes = &cmds
        .descriptor_bytes
        .iter()
        .find(|b| b.set == set && b.binding == binding)
        .expect("bytes")
        .bytes;
    assert_eq!(emitted_lookup(bytes, &keys), want.to_vec());
    // Each entry's flags are its member's out-of-bounds alpha: token 2's `R32Uint` has no alpha,
    // every other member's format has one, and the null entries have none (beside its texel class,
    // which `table_entries_state_each_members_texel_class` reads).
    let flag = FLAG_OUT_OF_BOUNDS_ALPHA_ONE;
    let flags: Vec<u32> = entries(&cmds.descriptor_bytes, set, binding)
        .iter()
        .map(|e| e.2 & !FLAG_TEXEL_CLASS)
        .collect();
    assert_eq!(flags, vec![0, 0, 0, 0, 0, 0, flag, 0, 0]);
    // The search's flagged form carries the found entry's alpha flag in the handle's bit 31; a
    // miss is the null handle without it.
    let alpha: Vec<u32> = want
        .iter()
        .map(|&e| if e == 2 { e | HANDLE_ALPHA_ONE } else { e })
        .collect();
    let flagged = Lookup::Flagged {
        alpha: true,
        class: false,
        red_blue: false,
    };
    assert_eq!(emitted_lookup_in(bytes, &keys, flagged), alpha);
    // The table's element of an id is the element the plan wrote that texture to.
    for (source, id) in [
        (HeapSource::Used(1), A),
        (HeapSource::Bound(slot(Stage::Kernel, 3)), F),
    ] {
        let e = w
            .iter()
            .position(|x| held(x).is_some_and(|h| h.0 == source))
            .expect("held");
        assert_eq!(lookup_host(&t, id), e as u32);
    }
}

/// The plan is total where the pipeline compiles: every committed profile with the sampled heap
/// plans one resident texture and instantiates its table.
#[test]
fn heap_plans_are_total_on_the_committed_profiles() {
    let Some(bc) = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll"))) else {
        return;
    };
    let mut planned = 0;
    for (stem, _) in crate::spirv::testing::committed_profiles().expect("profiles") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let lib = Library::from_air(&bc).expect("library");
        let f = lib.function("argument_buffer_read").expect("entry");
        let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
        let Ok(pipe) = dev.compile_compute_pipeline(&d, &Budget::standard()) else {
            continue;
        };
        let state = BindState {
            used_resources: vec![used(
                1,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(A),
            )],
            ..Default::default()
        };
        let plan = pipe
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let cmds = plan
            .instantiate_dispatch(
                &state,
                &Dispatch::Threadgroups {
                    groups: [1, 1, 1],
                    threads_per_group: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (set, binding, _) = table_write(&plan, Stage::Kernel);
        assert_eq!(
            table(&cmds.descriptor_bytes, set, binding),
            vec![(0, 0), (A, 1)],
            "{stem}"
        );
        planned += 1;
    }
    assert!(planned > 0);
}

/// The most members a `moltenvk-m3` heap holds: its capacity (`TABLE_CAPACITY`) less the null
/// element.
const LARGE_HEAP: u32 = 65_535;

/// Planning and instantiating are linear in the heap's members: every member's resource id is
/// found once (never by rescanning the bind state per member), so a full heap of [`LARGE_HEAP`]
/// used textures plans and instantiates inside a bound a per-member scan of the state cannot
/// meet (quadratic: 14 s at half this size in a debug build, about 4 times that here). Not a
/// benchmark: the bound is an order of magnitude above the linear cost.
#[test]
fn a_large_heap_plans_and_instantiates_in_linear_time() {
    let dev = m3();
    let Some(pipe) = read_pipeline(&dev) else {
        return;
    };
    // Distinct, unordered, nonzero ids.
    let id = |t: u32| (u64::from(t).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 1) | 1;
    let state = BindState {
        used_resources: (1..=LARGE_HEAP)
            .map(|t| {
                used(
                    t,
                    TextureType::Type2D,
                    PixelFormat::RGBA8Unorm,
                    read(),
                    Some(id(t)),
                )
            })
            .collect(),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let planned = start.elapsed();
    let cmds = plan
        .instantiate_dispatch(
            &state,
            &Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [1, 1, 1],
            },
        )
        .expect("dispatch");
    let instantiated = start.elapsed() - planned;
    eprintln!("{LARGE_HEAP} members: plan {planned:?}, instantiate {instantiated:?}");
    let (set, binding, _) = table_write(&plan, Stage::Kernel);
    let t = table(&cmds.descriptor_bytes, set, binding);
    assert_eq!(t.len(), LARGE_HEAP as usize + 1);
    assert_eq!(t[0], (0, 0));
    for token in [1, 2, LARGE_HEAP / 2, LARGE_HEAP] {
        assert_eq!(lookup_host(&t, id(token)), token, "token {token}");
    }
    let bound = std::time::Duration::from_secs(5);
    assert!(
        planned < bound && instantiated < bound,
        "plan {planned:?}, instantiate {instantiated:?}"
    );
}

/// The members plus the null element must fit the heap arrays' count: one more member than the
/// capacity allows is `HeapTooLarge`, and a state of another shape is an internal error.
#[test]
fn the_heap_capacity_bounds_the_members() {
    let dev = m3();
    let Some(mut pipe) = read_pipeline(&dev) else {
        return;
    };
    let heap = pipe.layout.sets[1]
        .bindings
        .iter_mut()
        .find(|b| matches!(b.source, BindingSource::Heap(_)))
        .expect("heap array");
    heap.count = 3;
    let members = |n: u32| BindState {
        used_resources: (1..=n)
            .map(|t| {
                used(
                    t,
                    TextureType::Type2D,
                    PixelFormat::RGBA8Unorm,
                    read(),
                    Some(u64::from(t)),
                )
            })
            .collect(),
        ..Default::default()
    };
    let fits = members(2);
    let plan = pipe
        .plan(&dev, &fits.shape())
        .expect("two members fit three elements");
    assert_eq!(heap_writes(&plan).len(), 3);
    assert_eq!(
        pipe.plan(&dev, &members(3).shape()),
        Err(BindError::HeapTooLarge { count: 4, limit: 3 })
    );
    // Instantiating with a member whose facts differ from the planned shape is internal.
    let mut other = fits.clone();
    other.used_resources[1] = used(
        2,
        TextureType::Type2D,
        PixelFormat::R32Uint,
        read(),
        Some(2),
    );
    assert!(matches!(
        plan.instantiate_dispatch(
            &other,
            &Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [1, 1, 1],
            },
        ),
        Err(BindError::Internal(_))
    ));
}

const STORAGE_KERNEL: &str = r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T, %struct.T }
define void @heap_write(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %pu = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 1, i32 0
  %u = load ptr addrspace(1), ptr addrspace(2) %pu, align 8
  %odd = trunc i32 %gid to i1
  %h = select i1 %odd, ptr addrspace(1) %t, ptr addrspace(1) %u
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %h, <2 x i32> %c, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 2)
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_write, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"t", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"texture2d<float, write>", !"u", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"u"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A storage heap holds the `ShaderWrite` members its formatless class can view (the direct
/// format, `GENERAL`), never a depth texture or a read-only one; a writable member whose storage
/// choice needs a conversion view is a typed refusal, since the class's one declaration cannot
/// convert per element.
#[test]
fn a_storage_heap_holds_writable_textures() {
    let dev = m3();
    let Some(bc) = assemble("heap_write", STORAGE_KERNEL) else {
        return;
    };
    let pipe = compute(&dev, &bc, "heap_write");
    let rw = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let write = TextureUsage::SHADER_WRITE;
    let mut state = BindState {
        used_resources: vec![
            used(1, TextureType::Type2D, PixelFormat::RGBA8Unorm, rw, Some(A)),
            used(
                2,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(B),
            ),
            used(
                3,
                TextureType::Type2D,
                PixelFormat::Depth32Float,
                write,
                Some(C),
            ),
            used(
                4,
                TextureType::Type2D,
                PixelFormat::R32Float,
                write,
                Some(D),
            ),
        ],
        ..Default::default()
    };
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let w = heap_writes(&plan);
    let general = vk::ImageLayout::GENERAL;
    assert!(matches!(
        w[0],
        WriteSource::Neutral(NeutralResource::Image {
            storage_format: Some(vk::Format::R32_SFLOAT),
            ..
        })
    ));
    assert_eq!(
        w.iter().map(|x| held(x)).collect::<Vec<_>>(),
        vec![
            None,
            Some((HeapSource::Used(1), vk::Format::R8G8B8A8_UNORM, general)),
            None,
            None,
            Some((HeapSource::Used(4), vk::Format::R32_SFLOAT, general)),
        ]
    );
    // An sRGB texture written through the heap: the plan follows its storage choice.
    state.used_resources.push(used(
        5,
        TextureType::Type2D,
        PixelFormat::RGBA8UnormSRGB,
        write,
        Some(E),
    ));
    let direct = storage_image_choice(PixelFormat::RGBA8UnormSRGB, dev.profile())
        .is_ok_and(|c| c.class == StorageImageClass::Identity && c.alias.is_none());
    match pipe.plan(&dev, &state.shape()) {
        Ok(_) => assert!(direct),
        Err(BindError::Unsupported(u)) => {
            assert!(!direct);
            assert_eq!(u.feature, Feature::ArgumentBufferTextures);
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        Err(e) => panic!("{e}"),
    }
}

/// `argument_buffer_depth_read` of the owned `handles` fixture: a depth texture read through the
/// float 2D sampled class, and a texture written through the storage class. The sampled array holds
/// the depth texture through its depth-aspect view and the storage array the writable one; each
/// holds the other's element as its neutral.
#[test]
fn a_depth_member_is_held_by_the_float_class_through_its_depth_view() {
    let dev = m3();
    let Some(bc) = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll"))) else {
        return;
    };
    let pipe = compute(&dev, &bc, "argument_buffer_depth_read");
    let state = BindState {
        used_resources: vec![
            used(
                1,
                TextureType::Type2D,
                PixelFormat::Depth32Float,
                read(),
                Some(A),
            ),
            used(
                2,
                TextureType::Type2D,
                PixelFormat::RGBA32Float,
                TextureUsage::SHADER_WRITE,
                Some(B),
            ),
        ],
        ..Default::default()
    };
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let heap = &pipe.layout.sets[1].bindings;
    assert_eq!(heap.len(), 2);
    for b in heap {
        let BindingSource::Heap(HandleClass::Image(class)) = b.source else {
            panic!("{:?}", b.source);
        };
        let mut w: Vec<_> = plan
            .descriptors
            .iter()
            .filter(|d| d.set == 1 && d.binding == b.binding)
            .collect();
        w.sort_by_key(|d| d.array_element);
        let views: Vec<_> = w
            .iter()
            .map(|d| match &d.source {
                WriteSource::HeapEntry { source, view, .. } => Some((*source, view.aspect)),
                WriteSource::Neutral(_) => None,
                other => panic!("{other:?}"),
            })
            .collect();
        match class.access {
            ImageAccess::Sampled => {
                assert_eq!(class.sampled_type, SampledType::Float);
                assert!(!class.depth_compare);
                assert_eq!(
                    views,
                    vec![
                        None,
                        Some((HeapSource::Used(1), vk::ImageAspectFlags::DEPTH)),
                        None
                    ]
                );
            }
            _ => assert_eq!(
                views,
                vec![
                    None,
                    None,
                    Some((HeapSource::Used(2), vk::ImageAspectFlags::COLOR))
                ]
            ),
        }
    }
}

/// `argument_buffer_sample_compare` of the owned `handles` fixture: a depth texture sampled with
/// comparison through the float 2D depth-comparison class. Its array holds the depth member through
/// its depth-aspect view and gives the color member, which a comparison cannot read, the class's
/// depth neutral; the comparison heap sampler is the layout's immutable sampler and takes no write.
#[test]
fn a_compared_depth_member_is_held_by_the_depth_comparison_class() {
    let dev = m3();
    let Some(bc) = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll"))) else {
        return;
    };
    let pipe = compute(&dev, &bc, "argument_buffer_sample_compare");
    let state = BindState {
        used_resources: vec![
            used(
                1,
                TextureType::Type2D,
                PixelFormat::Depth32Float,
                read(),
                Some(A),
            ),
            used(
                2,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(B),
            ),
        ],
        ..Default::default()
    };
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let heap = &pipe.layout.sets[1].bindings;
    assert_eq!(heap.len(), 1);
    let BindingSource::Heap(HandleClass::Image(class)) = heap[0].source else {
        panic!("{:?}", heap[0].source);
    };
    assert!(class.depth_compare);
    assert_eq!(class.sampled_type, SampledType::Float);
    let writes = heap_writes(&plan);
    let views: Vec<_> = writes
        .iter()
        .map(|w| match w {
            WriteSource::HeapEntry { source, view, .. } => Some((*source, view.aspect)),
            WriteSource::Neutral(n) => {
                assert!(
                    matches!(
                        n,
                        NeutralResource::Image {
                            depth_compare: true,
                            ..
                        }
                    ),
                    "{n:?}"
                );
                None
            }
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(
        views,
        vec![
            None,
            Some((HeapSource::Used(1), vk::ImageAspectFlags::DEPTH)),
            None
        ]
    );
    // The comparison heap sampler is immutable: the plan writes no sampler.
    let samplers: Vec<_> = pipe.layout.sets[0]
        .bindings
        .iter()
        .filter(|b| matches!(b.source, BindingSource::HeapSampler { .. }))
        .collect();
    assert_eq!(samplers.len(), 1);
    assert!(samplers[0].immutable_sampler.is_some());
    assert!(!plan
        .descriptors
        .iter()
        .any(|d| d.set == 0 && d.binding == samplers[0].binding));
}

/// Each member's table entry states its texel class's index (`heap_class_index`, in
/// [`FLAG_TEXEL_CLASS`]) beside its out-of-bounds alpha (a depth format has no alpha). The
/// search's flagged form carries the alpha in the handle's bit 31 and the class in place where it
/// carries them; a miss carries nothing.
#[test]
fn table_entries_state_each_members_texel_class() {
    let dev = m3();
    let Some(bc) = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll"))) else {
        return;
    };
    let pipe = compute(&dev, &bc, "argument_buffer_sample_compare");
    let state = BindState {
        used_resources: vec![
            used(
                1,
                TextureType::Type2D,
                PixelFormat::Depth16Unorm,
                read(),
                Some(A),
            ),
            used(
                2,
                TextureType::Type2D,
                PixelFormat::Depth32Float,
                read(),
                Some(B),
            ),
            used(
                3,
                TextureType::Type2D,
                PixelFormat::Depth32FloatStencil8,
                read(),
                Some(C),
            ),
            used(
                4,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(D),
            ),
        ],
        ..Default::default()
    };
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let (set, binding, _) = table_write(&plan, Stage::Kernel);
    let cmds = plan
        .instantiate_dispatch(&state, &one_group())
        .expect("dispatch");
    let alpha = FLAG_OUT_OF_BOUNDS_ALPHA_ONE;
    let class = |pf| {
        crate::metal::texel_format::heap_class_index(
            pf,
            crate::metal::TextureSwizzleChannels::DEFAULT,
        ) << FLAG_TEXEL_CLASS.trailing_zeros()
    };
    let (d16, d32, d32s8, rgba8) = (
        class(PixelFormat::Depth16Unorm),
        class(PixelFormat::Depth32Float),
        class(PixelFormat::Depth32FloatStencil8),
        class(PixelFormat::RGBA8Unorm),
    );
    assert!(d16 != 0 && d32 != 0 && rgba8 != 0 && d16 != rgba8);
    assert_eq!(
        entries(&cmds.descriptor_bytes, set, binding),
        vec![
            (0, 0, 0),
            (D, 4, rgba8),
            (A, 1, alpha | d16),
            (B, 2, alpha | d32),
            (C, 3, alpha | d32s8),
        ]
    );
    let bytes = &cmds
        .descriptor_bytes
        .iter()
        .find(|b| b.set == set && b.binding == binding)
        .expect("bytes")
        .bytes;
    let keys = [D, A, B, C, 0, E];
    let flagged = Lookup::Flagged {
        alpha: true,
        class: false,
        red_blue: false,
    };
    let one = HANDLE_ALPHA_ONE;
    assert_eq!(
        emitted_lookup_in(bytes, &keys, flagged),
        vec![4, 1 | one, 2 | one, 3 | one, 0, 0]
    );
    let classes = Lookup::Flagged {
        alpha: false,
        class: true,
        red_blue: false,
    };
    assert_eq!(
        emitted_lookup_in(bytes, &keys, classes),
        vec![4 | rgba8, 1 | d16, 2 | d32, 3 | d32s8, 0, 0]
    );
}

/// A heap texture sampled through a constexpr sampler reads a set-0 heap sampler: a `SAMPLER`
/// binding whose immutable sampler is the constexpr one, which the plan never writes. An
/// opaque-black border meets every member the sampled array holds, so an `ABGR4Unorm` member is
/// refused `SamplerBorderColor` on `moltenvk-m3`, which cannot keep that border exactly through its
/// view, as a slot sampled with that border is; without the border it is held.
#[test]
fn heap_samplers_take_no_write_and_meet_every_held_member() {
    let dev = m3();
    let Some(bc) = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll"))) else {
        return;
    };
    let sampler_bindings = |p: &ComputePipeline| -> Vec<u32> {
        p.layout.sets[0]
            .bindings
            .iter()
            .filter(|b| matches!(b.source, BindingSource::HeapSampler { .. }))
            .inspect(|b| {
                assert_eq!(b.ty, vk::DescriptorType::SAMPLER);
                assert_eq!(b.count, 1);
                assert!(b.immutable_sampler.is_some());
            })
            .map(|b| b.binding)
            .collect()
    };
    let state = |pf| BindState {
        used_resources: vec![
            used(1, TextureType::Type2D, pf, read(), Some(A)),
            used(
                2,
                TextureType::Type2D,
                PixelFormat::RGBA32Float,
                TextureUsage::SHADER_WRITE,
                Some(B),
            ),
        ],
        ..Default::default()
    };
    let copy = compute(&dev, &bc, "argument_buffer_sample");
    let samplers = sampler_bindings(&copy);
    assert_eq!(samplers.len(), 2);
    for pf in [PixelFormat::RGBA8Unorm, PixelFormat::ABGR4Unorm] {
        let plan = copy.plan(&dev, &state(pf).shape()).expect("plan");
        assert!(plan
            .descriptors
            .iter()
            .all(|d| d.set != 0 || !samplers.contains(&d.binding)));
        // The sampled member is held by the sampled class.
        let held = plan.descriptors.iter().any(|d| {
            matches!(
                d.source,
                WriteSource::HeapEntry {
                    source: HeapSource::Used(1),
                    ..
                }
            )
        });
        assert!(held, "{pf:?}");
    }
    let border = compute(&dev, &bc, "argument_buffer_sample_border");
    assert_eq!(sampler_bindings(&border).len(), 1);
    border
        .plan(&dev, &state(PixelFormat::RGBA8Unorm).shape())
        .expect("an exact border");
    match border.plan(&dev, &state(PixelFormat::ABGR4Unorm).shape()) {
        Err(BindError::Unsupported(u)) => assert_eq!(u.feature, Feature::SamplerBorderColor),
        other => panic!("{other:?}"),
    }
}

/// `argument_buffer_runtime_sampler` of the owned `handles` fixture samples a heap texture through
/// its runtime `[[sampler(0)]]`. Its set-0 heap sampler binding has no immutable sampler; the plan
/// writes it with the sampler bound at the kernel's slot 0, made for the use (no comparison, a
/// float texel), or the neutral sampler where none is bound. A bound opaque-black border meets
/// every member the sampled array holds, as a constexpr one does: an `ABGR4Unorm` member is refused
/// `SamplerBorderColor` on `moltenvk-m3`, an `RGBA8Unorm` one is held.
#[test]
fn a_runtime_heap_sampler_is_written_with_the_bound_sampler() {
    let dev = m3();
    let Some(bc) = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll"))) else {
        return;
    };
    let pipe = compute(&dev, &bc, "argument_buffer_runtime_sampler");
    let samplers: Vec<_> = pipe.layout.sets[0]
        .bindings
        .iter()
        .filter(|b| matches!(b.source, BindingSource::HeapSampler { .. }))
        .collect();
    assert_eq!(samplers.len(), 1);
    let binding = samplers[0].binding;
    assert_eq!(samplers[0].ty, vk::DescriptorType::SAMPLER);
    assert!(samplers[0].immutable_sampler.is_none());
    let state = |pf, s: Option<BoundSampler>| {
        let mut state = BindState {
            used_resources: vec![used(1, TextureType::Type2D, pf, read(), Some(A))],
            ..Default::default()
        };
        let mut t = StageTable::default();
        if let Some(s) = s {
            t.samplers.insert(0, s);
        }
        state.stages.insert(Stage::Kernel, t);
        state
    };
    let written = |plan: &BindingPlan| -> Vec<WriteSource> {
        plan.descriptors
            .iter()
            .filter(|d| d.set == 0 && d.binding == binding)
            .map(|d| d.source.clone())
            .collect()
    };
    let border = |color| {
        let mut d = SamplerDescriptor::new();
        d.s_address_mode = SamplerAddressMode::ClampToBorderColor;
        d.border_color = color;
        BoundSampler::new(d)
    };
    let s0 = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Sampler,
        index: 0,
    };
    let plan = pipe
        .plan(
            &dev,
            &state(
                PixelFormat::RGBA8Unorm,
                Some(BoundSampler::new(SamplerDescriptor::new())),
            )
            .shape(),
        )
        .expect("plan");
    assert_eq!(
        written(&plan),
        vec![WriteSource::Sampler {
            slot: s0,
            compare: false,
            integer: false,
            opaque_black: crate::metal::sampler::OpaqueBlackBorder::Stated,
        }]
    );
    let plan = pipe
        .plan(&dev, &state(PixelFormat::RGBA8Unorm, None).shape())
        .expect("plan");
    assert_eq!(
        written(&plan),
        vec![WriteSource::Neutral(NeutralResource::Sampler {
            compare: false
        })]
    );
    let black = border(SamplerBorderColor::OpaqueBlack);
    pipe.plan(&dev, &state(PixelFormat::RGBA8Unorm, Some(black)).shape())
        .expect("an exact border");
    match pipe.plan(&dev, &state(PixelFormat::ABGR4Unorm, Some(black)).shape()) {
        Err(BindError::Unsupported(u)) => assert_eq!(u.feature, Feature::SamplerBorderColor),
        other => panic!("{other:?}"),
    }
    let clear = border(SamplerBorderColor::TransparentBlack);
    pipe.plan(&dev, &state(PixelFormat::ABGR4Unorm, Some(clear)).shape())
        .expect("a transparent border");
}

/// A heap sampler that filters linearly meets every member of its stage that a sampled float array
/// holds, so each such member's view format must have the filter features it needs, as a combined
/// pair's texture must. On every committed profile with the sampled heap, an `R32Float` member is
/// refused `Unsupported{Format}` naming `SampledImageFilterLinear` of `R32_SFLOAT` exactly where
/// the profile lacks it, through the constexpr linear sampler of `argument_buffer_sample` and
/// through a bound linear runtime sampler of `argument_buffer_runtime_sampler`. A nearest or
/// unbound runtime sampler (the neutral sampler filters nearest) plans everywhere, and so does an
/// `RGBA8Unorm` member, whose linear filtering Vulkan requires.
#[test]
fn heap_samplers_need_the_filter_features_of_every_held_member() {
    let Some(bc) = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll"))) else {
        return;
    };
    let state = |pf, s: Option<BoundSampler>| {
        let mut state = BindState {
            used_resources: vec![used(1, TextureType::Type2D, pf, read(), Some(A))],
            ..Default::default()
        };
        let mut t = StageTable::default();
        if let Some(s) = s {
            t.samplers.insert(0, s);
        }
        state.stages.insert(Stage::Kernel, t);
        state
    };
    let sampler = |filter| {
        let mut d = SamplerDescriptor::new();
        d.min_filter = filter;
        d.mag_filter = filter;
        BoundSampler::new(d)
    };
    let linear = Some(sampler(SamplerMinMagFilter::Linear));
    let nearest = Some(sampler(SamplerMinMagFilter::Nearest));
    let missing = Unsupported::needs(
        Feature::Format,
        Capability::Format {
            format: vk::Format::R32_SFLOAT,
            usage: FormatUsage::SampledImageFilterLinear,
        },
    );
    let (mut refused, mut passed) = (0, 0);
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let lib = Library::from_air(&bc).expect("library");
        let filters = p.supports_format(
            vk::Format::R32_SFLOAT,
            FormatUsage::SampledImageFilterLinear,
        );
        for (entry, linear) in [
            ("argument_buffer_sample", None),
            ("argument_buffer_runtime_sampler", linear),
        ] {
            let f = lib.function(entry).expect("entry");
            let d =
                ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
            let Ok(pipe) = dev.compile_compute_pipeline(&d, &Budget::standard()) else {
                continue;
            };
            let what = format!("{entry} on {stem}");
            match pipe.plan(&dev, &state(PixelFormat::R32Float, linear).shape()) {
                Ok(_) => {
                    assert!(filters, "{what} planned without linear filtering");
                    passed += 1;
                }
                Err(BindError::Unsupported(u)) => {
                    assert!(!filters, "{what}: {u:?}");
                    assert_eq!(u, missing, "{what}");
                    refused += 1;
                }
                Err(e) => panic!("{what}: {e}"),
            }
            pipe.plan(&dev, &state(PixelFormat::RGBA8Unorm, linear).shape())
                .unwrap_or_else(|e| panic!("{what} RGBA8Unorm: {e}"));
            if entry == "argument_buffer_runtime_sampler" {
                for s in [nearest, None] {
                    pipe.plan(&dev, &state(PixelFormat::R32Float, s).shape())
                        .unwrap_or_else(|e| panic!("{what} {s:?}: {e}"));
                }
            }
        }
    }
    assert!(
        refused > 0 && passed > 0,
        "refused {refused}, passed {passed}"
    );
}

/// Two kernels whose texture slot 0 may also be a heap member: `slot_write_heap_read` writes
/// `[[texture(0)]]` and reads a texture from an argument buffer; `slot_read_heap_write` reads
/// `[[texture(0)]]` and writes a texture from an argument buffer.
const SLOT_AND_HEAP_KERNELS: &str = r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T }
define void @slot_write_heap_read(ptr addrspace(1) %o, ptr addrspace(2) %r, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %o, <2 x i32> %c, <4 x float> %x, i32 0, i32 2)
  ret void
}
define void @slot_read_heap_write(ptr addrspace(1) %i, ptr addrspace(2) %r, i32 %gid) {
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %i, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %c, <4 x float> %x, i32 0, i32 2)
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.kernel = !{!0, !10}
!air.version = !{!8}
!air.language_version = !{!9}
!0 = !{ptr @slot_write_heap_read, !1, !2}
!1 = !{}
!2 = !{!3, !4, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"o"}
!4 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!5 = !{i32 0, i32 8, i32 0, !"texture2d<float, read>", !"t", !"air.indirect_argument", !6}
!6 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
!10 = !{ptr @slot_read_heap_write, !1, !11}
!11 = !{!12, !13, !7}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"i"}
!13 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !14, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!14 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"t", !"air.indirect_argument", !15}
!15 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
"#;

/// The layout of `slot`'s own write in `plan`, with its descriptor type.
fn slot_layout(plan: &BindingPlan, s: SlotRef) -> (vk::DescriptorType, vk::ImageLayout) {
    let mut found = plan.descriptors.iter().filter_map(|d| match &d.source {
        WriteSource::Texture { slot, view } if *slot == s => Some((d.ty, view.layout)),
        _ => None,
    });
    let one = found.next().expect("the slot's write");
    assert!(found.next().is_none(), "one write of {s:?}");
    one
}

/// The uses of `s` in `plan`.
fn slot_uses(plan: &BindingPlan, s: SlotRef) -> Vec<&ResourceUse> {
    plan.uses
        .iter()
        .filter(|u| u.resource == UseTarget::Slot(s))
        .collect()
}

/// A texture bound at a slot and held by a heap array has one layout in every write of it:
/// `GENERAL` when the slot's own binding is a storage image (the heap's sampled element follows
/// it) or a storage heap array holds it (the slot's sampled view follows it). The plan uses the
/// slot once, widened to the heap's access; a texture no storage view reaches keeps its sampled
/// layout.
#[test]
fn a_bound_member_shares_its_slots_layout() {
    let dev = m3();
    let Some(bc) = assemble("slot_and_heap", SLOT_AND_HEAP_KERNELS) else {
        return;
    };
    let s0 = slot(Stage::Kernel, 0);
    let state = |usage: TextureUsage| {
        let mut facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm);
        facts.usage = usage;
        let mut k = StageTable::default();
        k.textures.insert(
            0,
            BoundTexture::Texture {
                facts,
                resource_id: Some(A),
            },
        );
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, k);
        state
    };
    let rw = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let general = vk::ImageLayout::GENERAL;
    let rgba = vk::Format::R8G8B8A8_UNORM;

    // A storage slot and a sampled heap: the heap element follows the slot.
    let pipe = compute(&dev, &bc, "slot_write_heap_read");
    let st = state(rw);
    let plan = pipe.plan(&dev, &st.shape()).expect("plan");
    assert_eq!(
        slot_layout(&plan, s0),
        (vk::DescriptorType::STORAGE_IMAGE, general)
    );
    let w = heap_writes(&plan);
    assert_eq!(
        w.iter().map(|x| held(x)).collect::<Vec<_>>(),
        vec![None, Some((HeapSource::Bound(s0), rgba, general))]
    );
    match slot_uses(&plan, s0).as_slice() {
        [u] => {
            assert!(u.access.read && u.access.write, "{u:?}");
            assert_eq!(u.class, AccessClass::StorageImage);
        }
        other => panic!("{other:?}"),
    }
    let cmds = plan
        .instantiate_dispatch(
            &st,
            &Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [1, 1, 1],
            },
        )
        .expect("dispatch");
    let (set, binding, _) = table_write(&plan, Stage::Kernel);
    assert_eq!(
        table(&cmds.descriptor_bytes, set, binding),
        vec![(0, 0), (A, 1)]
    );

    // A sampled slot and a storage heap holding the texture: the slot follows the heap.
    let pipe = compute(&dev, &bc, "slot_read_heap_write");
    let plan = pipe.plan(&dev, &state(rw).shape()).expect("plan");
    assert_eq!(
        slot_layout(&plan, s0),
        (vk::DescriptorType::SAMPLED_IMAGE, general)
    );
    assert_eq!(
        heap_writes(&plan)
            .iter()
            .map(|x| held(x))
            .collect::<Vec<_>>(),
        vec![None, Some((HeapSource::Bound(s0), rgba, general))]
    );
    match slot_uses(&plan, s0).as_slice() {
        [u] => {
            assert!(u.access.read && u.access.write, "{u:?}");
            assert_eq!(u.class, AccessClass::StorageImage);
        }
        other => panic!("{other:?}"),
    }

    // A read-only texture: the storage heap cannot hold it, so nothing makes it `GENERAL`.
    let plan = pipe
        .plan(&dev, &state(TextureUsage::SHADER_READ).shape())
        .expect("plan");
    assert_eq!(
        slot_layout(&plan, s0),
        (
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        )
    );
    assert_eq!(
        heap_writes(&plan)
            .iter()
            .map(|x| held(x))
            .collect::<Vec<_>>(),
        vec![None, None]
    );
    match slot_uses(&plan, s0).as_slice() {
        [u] => {
            assert!(u.access.read && !u.access.write, "{u:?}");
            assert_eq!(u.class, AccessClass::Sampled);
        }
        other => panic!("{other:?}"),
    }
}

const RENDER: &str = r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T }
define <{ <4 x float> }> @heap_vertex(i32 noundef %vid, ptr addrspace(2) %r) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %c = insertelement <2 x i32> zeroinitializer, i32 %vid, i64 0
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %p = extractvalue { <4 x float>, i8 } %v, 0
  %o = insertvalue <{ <4 x float> }> undef, <4 x float> %p, 0
  ret <{ <4 x float> }> %o
}
define <4 x float> @heap_fragment(<4 x float> noundef %pos, ptr addrspace(2) %r) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 0)
  %c = extractvalue { <4 x float>, i8 } %v, 0
  ret <4 x float> %c
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.vertex = !{!20}
!air.fragment = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_fragment, !1, !2}
!1 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!12, !3}
!12 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!3 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, read>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{ptr @heap_vertex, !21, !23}
!21 = !{!22}
!22 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!23 = !{!24, !25}
!24 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!25 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
"#;

fn render(dev: &Device) -> Option<RenderPipeline> {
    let bc = assemble("heap_render", RENDER)?;
    let lib = Library::from_air(&bc).expect("library");
    let v = lib.function("heap_vertex").expect("vertex");
    let f = lib.function("heap_fragment").expect("fragment");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(f)));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    Some(
        dev.compile_render_pipeline(&d, &Budget::standard())
            .expect("render pipeline"),
    )
}

/// A render heap: both stages share the members' elements in the one array, and each stage's
/// table enters the members that reach it (a used texture's render stages, a bound texture's own
/// stage); a used texture of no render stage is no member.
#[test]
fn render_stages_share_the_heap_and_keep_their_tables() {
    let dev = m3();
    let Some(pipe) = render(&dev) else {
        return;
    };
    let tex = |token, stages, id| {
        let mut u = used(
            token,
            TextureType::Type2D,
            PixelFormat::RGBA8Unorm,
            read(),
            Some(id),
        );
        u.stages = stages;
        u
    };
    let mut state = BindState {
        used_resources: vec![
            tex(1, RenderStages::VERTEX, A),
            tex(2, RenderStages::FRAGMENT, B),
            tex(3, RenderStages::VERTEX | RenderStages::FRAGMENT, C),
            tex(4, RenderStages::empty(), D),
        ],
        ..Default::default()
    };
    let mut v = StageTable::default();
    v.textures
        .insert(0, bound(PixelFormat::RGBA8Unorm, Some(E)));
    state.stages.insert(Stage::Vertex, v);
    let mut f = StageTable::default();
    f.textures
        .insert(2, bound(PixelFormat::RGBA8Unorm, Some(F)));
    state.stages.insert(Stage::Fragment, f);

    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    // Members: tokens 1, 2, 3 (elements 1..=3), vertex [[texture(0)]] (4), fragment
    // [[texture(2)]] (5).
    let w = heap_writes(&plan);
    assert_eq!(
        w.iter().map(|x| held(x).map(|h| h.0)).collect::<Vec<_>>(),
        vec![
            None,
            Some(HeapSource::Used(1)),
            Some(HeapSource::Used(2)),
            Some(HeapSource::Used(3)),
            Some(HeapSource::Bound(slot(Stage::Vertex, 0))),
            Some(HeapSource::Bound(slot(Stage::Fragment, 2))),
        ]
    );
    let vt = table_write(&plan, Stage::Vertex);
    let ft = table_write(&plan, Stage::Fragment);
    assert_ne!((vt.0, vt.1), (ft.0, ft.1));
    assert_eq!(vt.2, 4 * TABLE_ENTRY_BYTES);
    assert_eq!(ft.2, 4 * TABLE_ENTRY_BYTES);
    let cmds = plan
        .instantiate_draw(
            &state,
            &DrawCall::Primitives {
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        )
        .expect("draw");
    let vertex = table(&cmds.descriptor_bytes, vt.0, vt.1);
    let fragment = table(&cmds.descriptor_bytes, ft.0, ft.1);
    assert_eq!(vertex, vec![(0, 0), (A, 1), (C, 3), (E, 4)]);
    assert_eq!(fragment, vec![(0, 0), (B, 2), (C, 3), (F, 5)]);
    // A texture resident for the other stage only misses (Metal's residency is per stage).
    assert_eq!(lookup_host(&vertex, B), 0);
    assert_eq!(lookup_host(&fragment, A), 0);
    assert_eq!(lookup_host(&vertex, D), 0);
}

/// Two kernels over one argument-buffer member that only the MSL type name calls an integer
/// texture: `heap_query` only queries it, `heap_query_read` also reads it as `uint`.
const QUERY_KERNELS: &str = r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T }
define void @heap_query(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}
define void @heap_query_read(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %t2 = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %v = call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %t2, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %x4 = extractvalue { <4 x i32>, i8 } %v, 0
  %x = extractelement <4 x i32> %x4, i64 0
  %sum = add i32 %w, %x
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %sum, ptr addrspace(1) %o, align 4
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.kernel = !{!0, !11}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_query, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<uint, read>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"t"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
!11 = !{ptr @heap_query_read, !1, !2}
"#;

/// An integer texture in a query-only heap group: its class is the `Float` default
/// (`RuntimeFact::HeapNumericType`), so the texture's element is the neutral and the plan reports
/// it (`IncompatibleHeapEntry`), as a slot reports an incompatible binding; a `float` texture is
/// held. Where a read of the member types the group, the integer texture is held and nothing is
/// reported.
#[test]
fn an_integer_texture_in_a_query_only_group_is_reported() {
    let dev = m3();
    let Some(bc) = assemble("heap_query", QUERY_KERNELS) else {
        return;
    };
    let state = BindState {
        used_resources: vec![
            used(
                1,
                TextureType::Type2D,
                PixelFormat::R32Uint,
                read(),
                Some(A),
            ),
            used(
                2,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(B),
            ),
            // Another texture type: another class's, not reported.
            used(
                3,
                TextureType::Type3D,
                PixelFormat::R32Uint,
                read(),
                Some(C),
            ),
        ],
        ..Default::default()
    };
    let pipe = compute(&dev, &bc, "heap_query");
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let w = heap_writes(&plan);
    assert_eq!(
        w.iter().map(|x| held(x).map(|h| h.0)).collect::<Vec<_>>(),
        vec![None, None, Some(HeapSource::Used(2)), None]
    );
    let WriteSource::Neutral(neutral) = w[1] else {
        panic!("element 1 {:?}", w[1]);
    };
    match plan.diagnostics.as_slice() {
        [PlanDiagnostic::IncompatibleHeapEntry {
            class: HandleClass::Image(c),
            element: 1,
            source: HeapSource::Used(1),
            substituted,
        }] => {
            assert_eq!(c.sampled_type, SampledType::Float);
            assert_eq!(substituted, neutral);
        }
        other => panic!("{other:?}"),
    }

    let pipe = compute(&dev, &bc, "heap_query_read");
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    assert_eq!(
        heap_writes_of(&plan, SampledType::Uint)
            .iter()
            .map(|x| held(x).map(|h| h.0))
            .collect::<Vec<_>>(),
        vec![None, Some(HeapSource::Used(1)), None, None]
    );
    assert!(plan.diagnostics.is_empty(), "{:?}", plan.diagnostics);
}

/// The heap classes a function declares.
fn heap_classes_of(a: &FunctionArtifact) -> Vec<ImageHandleClass> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::Heap(HandleClass::Image(c)) => Some(c),
            _ => None,
        })
        .collect()
}

/// A heap group that is only queried has no texel class: its array is declared `Float`, and
/// `RuntimeFact::HeapNumericType` says so. A load of the same member that is read types every
/// group loaded from it: the member has one declared type.
#[test]
fn a_member_read_anywhere_types_its_queries() {
    let Some(bc) = assemble("heap_query", QUERY_KERNELS) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let a = compile(&bc, "heap_query", &p).expect("heap_query");
    link_and_validate(&a, &p).expect("L3");
    let classes = heap_classes_of(&a);
    // The float class, and its unsigned integer twin, which a member of a form a heap reads is
    // queried through.
    assert_eq!(classes.len(), 2);
    assert_eq!(classes[0].sampled_type, SampledType::Float);
    assert_eq!(
        classes[1],
        ImageHandleClass {
            sampled_type: SampledType::Uint,
            ..classes[0]
        }
    );
    assert!(a
        .facts
        .contains(&RuntimeFact::HeapNumericType { class: classes[0] }));

    let a = compile(&bc, "heap_query_read", &p).expect("heap_query_read");
    link_and_validate(&a, &p).expect("L3");
    assert_eq!(
        a.strategies
            .entries
            .get("access.handle.lookup")
            .map_or(0, |u| u.uses),
        2
    );
    // Both loads of the member share the unsigned integer class; the read and the query read a
    // member of another numeric class through its shape's class of that type.
    let mut classes = heap_classes_of(&a);
    classes.sort_by_key(|c| c.sampled_type);
    assert_eq!(
        classes,
        [SampledType::Float, SampledType::Sint, SampledType::Uint].map(|sampled_type| {
            ImageHandleClass {
                sampled_type,
                ..classes[0]
            }
        }),
        "both loads of the member share one class"
    );
    assert!(!a
        .facts
        .iter()
        .any(|f| matches!(f, RuntimeFact::HeapNumericType { .. })));
}

/// An entry of the owned `tail_argument_samplers` fixture on `dev`.
fn argument_sampler_pipeline(dev: &Device, entry: &str) -> Option<ComputePipeline> {
    let bc = assembled(
        "tail_argument_samplers",
        llvm_assemble(&engine_dir().join("tail_argument_samplers.ll")),
    )?;
    Some(compute(dev, &bc, entry))
}

/// An argument sampler of `desc` whose resource id is `id`.
fn argument_sampler(desc: SamplerDescriptor, id: u64) -> ArgumentSampler {
    ArgumentSampler::new(desc, id)
}

/// A pixel-coordinate sampler (nearest, clamp to edge).
fn pixel_sampler() -> SamplerDescriptor {
    let mut d = SamplerDescriptor::new();
    d.normalized_coordinates = false;
    d
}

/// A bind state with the used `RGBA8Unorm` texture `A` (token 1) and the argument samplers
/// `samplers`.
fn sampler_state(samplers: Vec<ArgumentSampler>) -> BindState {
    BindState {
        used_resources: vec![used(
            1,
            TextureType::Type2D,
            PixelFormat::RGBA8Unorm,
            read(),
            Some(A),
        )],
        argument_samplers: samplers,
        ..Default::default()
    }
}

/// A set-1 sampler array's binding and its use `(compare, integer)`.
type SamplerArray = (u32, (bool, bool));

/// The bindings and uses of the pipeline's set-1 sampler arrays, and the `(set, binding)` of its
/// sampler table.
fn sampler_layout(p: &ComputePipeline) -> (Vec<SamplerArray>, (u32, u32)) {
    let arrays = p.layout.sets[1]
        .bindings
        .iter()
        .filter_map(|b| match b.source {
            BindingSource::Heap(HandleClass::Sampler { compare, integer }) => {
                assert_eq!(b.ty, vk::DescriptorType::SAMPLER);
                assert_eq!(b.flags, super::engine_handles::heap_flags(&p.requirements));
                Some((b.binding, (compare, integer)))
            }
            _ => None,
        })
        .collect();
    let table = p.layout.sets[0]
        .bindings
        .iter()
        .find(|b| {
            matches!(
                b.source,
                BindingSource::Synthetic {
                    binding: SyntheticBinding::SamplerTable,
                    ..
                }
            )
        })
        .map(|b| (0, b.binding))
        .expect("a sampler table");
    (arrays, table)
}

/// The writes of the set-1 sampler array at `binding`, by element.
fn sampler_writes(plan: &BindingPlan, binding: u32) -> Vec<WriteSource> {
    let mut w: Vec<_> = plan
        .descriptors
        .iter()
        .filter(|d| d.set == 1 && d.binding == binding)
        .collect();
    w.sort_by_key(|d| d.array_element);
    for (e, d) in w.iter().enumerate() {
        assert_eq!(d.array_element as usize, e, "one write per element");
        assert_eq!(d.ty, vk::DescriptorType::SAMPLER);
    }
    w.into_iter().map(|d| d.source.clone()).collect()
}

/// The entries `(id, element, flags)` of the sampler table at `(set, binding)` in `bytes`: a
/// sampler's flags are its word, within [`SamplerWord::FIELDS`].
fn sampler_entries(bytes: &[DescriptorBytes], set: u32, binding: u32) -> Vec<(u64, u32, u32)> {
    let b = &bytes
        .iter()
        .find(|b| b.set == set && b.binding == binding)
        .expect("sampler table bytes")
        .bytes;
    b.chunks(TABLE_ENTRY_BYTES as usize)
        .map(|e| {
            let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().expect("word"));
            let id = u64::from(w(0)) | (u64::from(w(1)) << 32);
            assert_eq!(w(3) & !SamplerWord::FIELDS, 0, "other bits are reserved");
            (id, w(2), w(3))
        })
        .collect()
}

fn one_group() -> Dispatch {
    Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    }
}

/// A sampler loaded from an argument buffer reads the set-1 sampler array of its use at the element
/// its sampler table gives. The plan writes element 0 of each array with the use's neutral sampler
/// and every other element of twice the shape's capacity class (32 entries for up to 15 samplers)
/// with `ArgumentSampler` of that listed sampler, made for the use; the sampler table is a handle
/// table of as many entries. Per dispatch its bytes list each listed sampler's nonzero id with its
/// element and its state word, deduplicated and sorted after leading null entries, and the emitted
/// search's entry form finds each id's entry, the last null entry for the null id and the first (0)
/// for any other id. An integer texel or a comparison reads the array of its own use.
#[test]
fn argument_samplers_fill_their_arrays_and_tables() {
    let dev = m3();
    let Some(pipe) = argument_sampler_pipeline(&dev, "argument_sampler_sample") else {
        return;
    };
    let (arrays, (tset, tbinding)) = sampler_layout(&pipe);
    assert_eq!(arrays.len(), 1);
    let (binding, use_) = arrays[0];
    assert_eq!(use_, (false, false));
    let linear = {
        let mut d = SamplerDescriptor::new();
        d.min_filter = crate::metal::SamplerMinMagFilter::Linear;
        d.mag_filter = crate::metal::SamplerMinMagFilter::Linear;
        d
    };
    // The second sampler's id repeats the first's (one sampler listed twice), the third has no
    // id, and the fourth uses pixel coordinates.
    let state = sampler_state(vec![
        argument_sampler(linear, C),
        argument_sampler(linear, C),
        argument_sampler(SamplerDescriptor::new(), 0),
        argument_sampler(pixel_sampler(), B),
    ]);
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let writes = sampler_writes(&plan, binding);
    assert_eq!(writes.len(), 32);
    assert_eq!(
        writes[0],
        WriteSource::Neutral(NeutralResource::Sampler { compare: false })
    );
    for (e, w) in writes.iter().enumerate().skip(1) {
        assert_eq!(
            *w,
            WriteSource::ArgumentSampler {
                index: e as u32 - 1,
                compare: false,
                integer: false,
            }
        );
    }
    let sampler_table = plan
        .descriptors
        .iter()
        .find(|d| d.set == tset && d.binding == tbinding)
        .expect("the sampler table's write");
    assert_eq!(
        sampler_table.source,
        WriteSource::HandleTable {
            stage: Stage::Kernel,
            bytes_len: 32 * TABLE_ENTRY_BYTES,
        }
    );
    let cmds = plan
        .instantiate_dispatch(&state, &one_group())
        .expect("dispatch");
    let entries = sampler_entries(&cmds.descriptor_bytes, tset, tbinding);
    let mut want = vec![(0, 0, 0); 30];
    want.push((B, 4, SamplerWord::of(&pixel_sampler()).0));
    want.push((C, 1, SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR));
    assert_eq!(entries, want);
    let bytes = &cmds
        .descriptor_bytes
        .iter()
        .find(|b| b.set == tset && b.binding == tbinding)
        .expect("bytes")
        .bytes;
    assert_eq!(
        emitted_lookup_in(bytes, &[C, B, 0, A, D], Lookup::Entry),
        vec![31, 30, 29, 0, 0]
    );
    // The texture's own table is unchanged by the samplers.
    let (set, b, _) = table_write(&plan, Stage::Kernel);
    assert_eq!(table(&cmds.descriptor_bytes, set, b), vec![(0, 0), (A, 1)]);

    // A comparison and an integer texel read the arrays of their uses.
    for (entry, use_) in [
        ("argument_sampler_compare", (true, false)),
        ("argument_sampler_integer", (false, true)),
    ] {
        let Some(pipe) = argument_sampler_pipeline(&dev, entry) else {
            return;
        };
        let (arrays, _) = sampler_layout(&pipe);
        assert_eq!(arrays.len(), 1, "{entry}");
        assert_eq!(arrays[0].1, use_, "{entry}");
        let mut state = sampler_state(vec![argument_sampler(SamplerDescriptor::new(), C)]);
        state.used_resources.clear();
        match pipe.plan(&dev, &state.shape()) {
            Ok(plan) => {
                let writes = sampler_writes(&plan, arrays[0].0);
                assert_eq!(
                    writes[0],
                    WriteSource::Neutral(NeutralResource::Sampler { compare: use_.0 }),
                    "{entry}"
                );
                assert_eq!(
                    writes[1],
                    WriteSource::ArgumentSampler {
                        index: 0,
                        compare: use_.0,
                        integer: use_.1,
                    },
                    "{entry}"
                );
            }
            // The device withholds mutable comparison samplers (portability).
            Err(BindError::Unsupported(u)) if use_.0 => {
                assert_eq!(u.feature, Feature::SamplerCompare, "{entry}")
            }
            Err(e) => panic!("{entry}: {e}"),
        }
    }
}

/// A kernel sampling the cube of its constant argument buffer `M { texturecube t; sampler s; }`
/// with explicit gradients through the member sampler.
const CUBE_GRADIENT: &str = r#"%struct.M = type { ptr addrspace(1), ptr addrspace(2) }
define void @argument_sampler_cube_grad(ptr addrspace(2) %m, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %ps = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 1
  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8
  %v = call { <4 x float>, i8 } @air.sample_texture_cube_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <3 x float> <float 1.0, float 0.0, float 0.0>, <3 x float> <float 0.0, float 0x3F847AE140000000, float 0.0>, <3 x float> <float 0.0, float 0.0, float 0x3F847AE140000000>, float 0.0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}
declare { <4 x float>, i8 } @air.sample_texture_cube_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, <3 x float>, <3 x float>, float, i32)
!air.kernel = !{!0}
!air.version = !{!11}
!air.language_version = !{!12}
!0 = !{ptr @argument_sampler_cube_grad, !1, !2}
!1 = !{}
!2 = !{!3, !8, !9}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}
!4 = !{i32 0, i32 8, i32 0, !"texturecube<float, sample>", !"t", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!8 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!9 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!11 = !{i32 2, i32 8, i32 0}
!12 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A cube's explicit gradients through an argument buffer's sampler take their footprint by the
/// sampler's table entry word, so the plan takes every listed sampler for that use, an anisotropic
/// one included: an isotropic sampler plans with the array's `ArgumentSampler` writes, as an
/// anisotropic one does. A driver whose cube-gradient route is not measured (`moltenvk-m3` without
/// its measured route) takes Vulkan's rule, the device's reading, alike. Run on every committed
/// profile that compiles the use, and the unmeasured description.
#[test]
fn a_cube_gradient_use_takes_an_anisotropic_argument_sampler_by_its_word() {
    let Some(bc) = assemble("argument_sampler_cube_grad", CUBE_GRADIENT) else {
        return;
    };
    let description = |stem: &str| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        DeviceDescription::from_khronos_json(&text, None).expect("description")
    };
    let mut devices: Vec<(String, Device)> = crate::spirv::testing::committed_profiles()
        .expect("profiles")
        .into_iter()
        .map(|(stem, _)| {
            let dev = Device::new(description(&stem), DevicePolicy::default()).expect("device");
            (stem, dev)
        })
        .collect();
    let route = crate::error::DriverDeviation::CubeGradientRoute;
    let unmeasured = Device::with_deviation(
        description("moltenvk-m3"),
        DevicePolicy::default(),
        route,
        false,
    )
    .expect("device");
    devices.push(("unmeasured".to_string(), unmeasured));
    let (mut taken_on, mut unmeasured_on) = (0, 0);
    for (stem, dev) in devices {
        let lib = Library::from_air(&bc).expect("library");
        let f = lib.function("argument_sampler_cube_grad").expect("entry");
        let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
        let Ok(pipe) = dev.compile_compute_pipeline(&d, &Budget::standard()) else {
            assert_ne!(
                stem, "unmeasured",
                "the unmeasured description compiles the use"
            );
            continue;
        };
        let route = dev.profile().fields().cube_gradients.compute;
        if stem == "unmeasured" {
            assert_eq!(route, crate::device::profile::CubeGradientRoute::Vulkan);
        }
        let arrays: Vec<(u32, HandleClass)> = pipe.layout.sets[1]
            .bindings
            .iter()
            .filter_map(|b| match b.source {
                BindingSource::Heap(c @ HandleClass::Sampler { .. }) => Some((b.binding, c)),
                _ => None,
            })
            .collect();
        let cube_class = HandleClass::Sampler {
            compare: false,
            integer: false,
        };
        assert_eq!(arrays.len(), 1, "{stem}");
        assert_eq!(arrays[0].1, cube_class, "{stem}");
        let cube_state = |samplers: Vec<ArgumentSampler>| BindState {
            used_resources: vec![used(
                1,
                TextureType::Cube,
                PixelFormat::RGBA8Unorm,
                read(),
                Some(A),
            )],
            argument_samplers: samplers,
            ..Default::default()
        };
        let linear = {
            let mut d = SamplerDescriptor::new();
            d.min_filter = crate::metal::SamplerMinMagFilter::Linear;
            d.mag_filter = crate::metal::SamplerMinMagFilter::Linear;
            d
        };
        let anisotropic = {
            let mut d = linear;
            d.max_anisotropy = 4;
            d
        };
        let write = WriteSource::ArgumentSampler {
            index: 0,
            compare: false,
            integer: false,
        };
        for (what, desc) in [("isotropic", linear), ("anisotropic", anisotropic)] {
            let plan = pipe
                .plan(&dev, &cube_state(vec![argument_sampler(desc, C)]).shape())
                .unwrap_or_else(|e| panic!("{stem}: an {what} sampler plans: {e}"));
            assert_eq!(
                sampler_writes(&plan, arrays[0].0)[1],
                write,
                "{stem} {what}"
            );
        }
        taken_on += 1;
        unmeasured_on += usize::from(stem == "unmeasured");
    }
    assert!(taken_on > 0, "no profile took the anisotropic sampler");
    assert_eq!(unmeasured_on, 1, "the unmeasured description takes it");
}

/// A plan depends on the argument samplers only through their shape (`ArgumentSamplersShape`: the
/// capacity class of their count and their distinct shapes). A sampler made or released within the
/// class whose shape another listed sampler shares keeps the shape, so the memoized plan serves
/// both states unchanged and only the per-dispatch table bytes and the resolution of its
/// `ArgumentSampler` writes differ; a pixel sampler shares the default sampler's shape, its
/// coordinates being its table word's. A sampler of a new shape is a new shape key, hence a whole
/// replan, which here plans the same writes; a count leaving its class replans the sampler arrays
/// and tables at twice the next class's entries and nothing else.
#[test]
fn an_unrelated_argument_sampler_keeps_the_plan() {
    let dev = m3();
    let Some(pipe) = argument_sampler_pipeline(&dev, "argument_sampler_sample") else {
        return;
    };
    let (arrays, (tset, tbinding)) = sampler_layout(&pipe);
    let binding = arrays[0].0;
    let samplers = |n: u64, desc: SamplerDescriptor| -> Vec<ArgumentSampler> {
        (1..=n)
            .map(|i| argument_sampler(desc, 0x1000 + i))
            .collect()
    };
    let before = sampler_state(samplers(3, SamplerDescriptor::new()));
    let plan = pipe.plan(&dev, &before.shape()).expect("plan");
    // One more sampler of a listed shape: the same shape, so the same plan.
    let mut same = before.clone();
    same.argument_samplers
        .push(argument_sampler(SamplerDescriptor::new(), 0x2000));
    assert_eq!(same.shape(), before.shape());
    let cmds = plan
        .instantiate_dispatch(&same, &one_group())
        .expect("dispatch");
    let entries = sampler_entries(&cmds.descriptor_bytes, tset, tbinding);
    assert_eq!(entries.last(), Some(&(0x2000, 4, 0)));
    assert_eq!(entries.iter().filter(|e| e.0 != 0).count(), 4);
    // Released again: still the same plan.
    let fewer = sampler_state(samplers(1, SamplerDescriptor::new()));
    assert_eq!(fewer.shape(), before.shape());
    plan.instantiate_dispatch(&fewer, &one_group())
        .expect("dispatch");
    // A pixel sampler keeps the shape: its coordinates are its table word's, so only the bytes
    // differ.
    let mut pixel = before.clone();
    pixel
        .argument_samplers
        .push(argument_sampler(pixel_sampler(), 0x3000));
    assert_eq!(pixel.shape(), before.shape());
    // A sampler of a new shape (a linear one): another shape key (a replan, of the same writes);
    // a state of another shape is refused by the plan made for the first.
    let mut linear = SamplerDescriptor::new();
    linear.min_filter = crate::metal::SamplerMinMagFilter::Linear;
    let mut new_shape = before.clone();
    new_shape
        .argument_samplers
        .push(argument_sampler(linear, 0x3000));
    assert_ne!(new_shape.shape(), before.shape());
    let replanned = pipe.plan(&dev, &new_shape.shape()).expect("plan");
    assert_eq!(replanned.descriptors, plan.descriptors);
    assert!(matches!(
        plan.instantiate_dispatch(&new_shape, &one_group()),
        Err(BindError::Internal(_))
    ));
    // Sixteen samplers leave the first class (16 entries, the null one included): twice the
    // next class's 32 entries, in the sampler array and the table alone.
    let more = sampler_state(samplers(16, SamplerDescriptor::new()));
    assert_ne!(more.shape(), before.shape());
    assert_eq!(
        sampler_state(samplers(15, SamplerDescriptor::new())).shape(),
        before.shape()
    );
    let grown = pipe.plan(&dev, &more.shape()).expect("plan");
    assert_eq!(sampler_writes(&grown, binding).len(), 64);
    let other = |p: &BindingPlan| -> Vec<_> {
        p.descriptors
            .iter()
            .filter(|d| {
                !(d.set == 1 && d.binding == binding) && !(d.set == tset && d.binding == tbinding)
            })
            .cloned()
            .collect()
    };
    assert_eq!(other(&grown), other(&plan));
    let table_len = |p: &BindingPlan| {
        p.descriptors
            .iter()
            .find(|d| d.set == tset && d.binding == tbinding)
            .map(|d| d.source.clone())
    };
    assert_eq!(
        table_len(&grown),
        Some(WriteSource::HandleTable {
            stage: Stage::Kernel,
            bytes_len: 64 * TABLE_ENTRY_BYTES,
        })
    );
}

/// The argument samplers plus the null entry must fit the heap's capacity. A class whose least
/// count does not fit is `HeapTooLarge` at plan time; a class straddling the capacity plans its
/// entries clamped to it, and a list past them is `HeapTooLarge` at instantiation. Every listed
/// shape is checked for every use the arrays declare, and an opaque-black border among them meets
/// every member the sampled image arrays hold.
#[test]
fn argument_samplers_are_bounded_and_checked() {
    let dev = m3();
    let Some(mut pipe) = argument_sampler_pipeline(&dev, "argument_sampler_sample") else {
        return;
    };
    for set in &mut pipe.layout.sets {
        for b in &mut set.bindings {
            if matches!(b.source, BindingSource::Heap(_)) {
                b.count = 20;
            }
        }
    }
    let (arrays, _) = sampler_layout(&pipe);
    let samplers = |n: u64| -> Vec<ArgumentSampler> {
        (1..=n)
            .map(|i| argument_sampler(SamplerDescriptor::new(), i))
            .collect()
    };
    // 17 to 31 samplers: class 1 (32 entries), clamped to the arrays' 20.
    let fits = sampler_state(samplers(19));
    let plan = pipe.plan(&dev, &fits.shape()).expect("plan");
    assert_eq!(sampler_writes(&plan, arrays[0].0).len(), 20);
    plan.instantiate_dispatch(&fits, &one_group())
        .expect("19 samplers and the null entry fit 20");
    let past = sampler_state(samplers(20));
    assert_eq!(past.shape(), fits.shape());
    assert_eq!(
        plan.instantiate_dispatch(&past, &one_group()).map(|_| ()),
        Err(BindError::HeapTooLarge {
            count: 21,
            limit: 20
        })
    );
    // 32 samplers: class 2, whose least count (33) exceeds 20.
    assert_eq!(
        pipe.plan(&dev, &sampler_state(samplers(32)).shape())
            .map(|_| ()),
        Err(BindError::HeapTooLarge {
            count: 33,
            limit: 20
        })
    );
    // A listed sampler that cannot be made for the use refuses the plan, though no buffer may
    // pair it with the use: here invalid LOD clamps.
    let mut lod = SamplerDescriptor::new();
    lod.lod_min_clamp = 2.0;
    lod.lod_max_clamp = 1.0;
    let mut bad = sampler_state(samplers(1));
    bad.argument_samplers.push(argument_sampler(lod, 99));
    assert!(matches!(
        pipe.plan(&dev, &bad.shape()),
        Err(BindError::InvalidDescriptor(_))
    ));
    // An opaque-black border meets the members the sampled arrays hold.
    let mut black = SamplerDescriptor::new();
    black.s_address_mode = SamplerAddressMode::ClampToBorderColor;
    black.border_color = SamplerBorderColor::OpaqueBlack;
    let mut state = sampler_state(vec![argument_sampler(black, 7)]);
    pipe.plan(&dev, &state.shape()).expect("an exact border");
    state.used_resources = vec![used(
        1,
        TextureType::Type2D,
        PixelFormat::ABGR4Unorm,
        read(),
        Some(A),
    )];
    match pipe.plan(&dev, &state.shape()) {
        Err(BindError::Unsupported(u)) => assert_eq!(u.feature, Feature::SamplerBorderColor),
        other => panic!("{other:?}"),
    }
}

/// A stage sampling through argument samplers tests each one's entry for a cubic filter
/// (`RuntimeFact::CubicArgumentSampler`), so a listed cubic sampler is planned beside it, listed
/// with `SamplerWord::CUBIC` in its entry's flags, which the stage's test reads where a buffer
/// names it; the null entry's flags hold no such bit. A stage compiled without the test (its
/// dependencies without the fact) is refused `NotImplemented(SamplerBicubicFilter)` where a
/// cubic sampler is listed, and planned where none is.
#[test]
fn a_listed_cubic_sampler_is_planned_beside_a_stage_that_tests_it() {
    let dev = m3();
    let Some(pipe) = argument_sampler_pipeline(&dev, "argument_sampler_sample") else {
        return;
    };
    let tests = RuntimeFact::CubicArgumentSampler {
        stage: Stage::Kernel,
    };
    assert!(pipe.dependencies.iter().any(|d| d.fact == tests));
    assert!(pipe
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::ExecutionStatus));
    let (_, (tset, tbinding)) = sampler_layout(&pipe);
    let mut cubic = SamplerDescriptor::new();
    cubic.min_filter = SamplerMinMagFilter::Cubic;
    cubic.mag_filter = SamplerMinMagFilter::Cubic;
    let state = sampler_state(vec![
        argument_sampler(SamplerDescriptor::new(), B),
        argument_sampler(cubic, C),
    ]);
    let plan = pipe
        .plan(&dev, &state.shape())
        .expect("a stage that tests its samplers meets a cubic one");
    let cmds = plan
        .instantiate_dispatch(&state, &one_group())
        .expect("dispatch");
    let entries = sampler_entries(&cmds.descriptor_bytes, tset, tbinding);
    let flags = |id: u64| {
        entries
            .iter()
            .find(|e| e.0 == id)
            .map(|e| e.2)
            .expect("listed")
    };
    assert_eq!(flags(C) & SamplerWord::CUBIC, SamplerWord::CUBIC);
    assert_eq!(flags(B) & SamplerWord::CUBIC, 0);
    assert_eq!(flags(0), 0, "the null entry is no cubic sampler");
    // Without the test, a listed cubic sampler is refused; a list without one is planned.
    let Some(mut untested) = argument_sampler_pipeline(&dev, "argument_sampler_sample") else {
        return;
    };
    untested.dependencies.retain(|d| d.fact != tests);
    match untested.plan(&dev, &state.shape()) {
        Err(BindError::Unsupported(u)) => assert_eq!(u.feature, Feature::SamplerBicubicFilter),
        other => panic!("{:?}", other.map(|_| ())),
    }
    let plain = sampler_state(vec![argument_sampler(SamplerDescriptor::new(), B)]);
    untested
        .plan(&dev, &plain.shape())
        .expect("no cubic sampler");
}

/// The states a command buffer lists again as it commits fit the plan of a command encoded with one
/// state (class 0, 16 entries) up to twice the class's entries, 31 states and the null entry, each
/// written at its element as the plan writes them; 32 do not (`relisted_sampler_tables` is `None`,
/// where the runtime keeps the states listed before, announcing `unlisted-argument-sampler`). A
/// state of a new shape is held where the
/// plan's checks pass: a linear one, and a cubic one beside the kernel's test; a cubic one beside a
/// stage compiled without the test is not, nor one whose LOD clamps no sampler can take.
#[test]
fn late_argument_samplers_fit_twice_the_class_and_the_plan_s_checks() {
    let dev = m3();
    let Some(pipe) = argument_sampler_pipeline(&dev, "argument_sampler_sample") else {
        return;
    };
    let (_, (tset, tbinding)) = sampler_layout(&pipe);
    let samplers = |n: u64| -> Vec<ArgumentSampler> {
        (1..=n)
            .map(|i| argument_sampler(SamplerDescriptor::new(), 0x1000 + i))
            .collect()
    };
    let encoded = sampler_state(samplers(1));
    let plan = pipe.plan(&dev, &encoded.shape()).expect("plan");
    let placement = crate::bind::HeapPlacement::Planned;
    let relisted = |plan: &BindingPlan, list: &[ArgumentSampler]| {
        plan.relisted_sampler_tables(&dev, list, placement)
            .expect("relisted")
    };
    // Twice the class: 31 states and the null entry, written as an instantiation of the
    // class's plan writes them.
    let late = samplers(31);
    let tables = relisted(&plan, &late).expect("31 states fit");
    let bytes = sampler_entries(&tables, tset, tbinding);
    assert_eq!(bytes.len(), 32);
    let wide = pipe
        .plan(&dev, &sampler_state(late.clone()).shape())
        .expect("the class of 31");
    let cmds = wide
        .instantiate_dispatch(&sampler_state(late.clone()), &one_group())
        .expect("dispatch");
    let listed = sampler_entries(&cmds.descriptor_bytes, tset, tbinding);
    let ids = |entries: &[(u64, u32, u32)]| -> Vec<(u64, u32)> {
        entries
            .iter()
            .filter(|e| e.0 != 0)
            .map(|e| (e.0, e.1))
            .collect()
    };
    assert_eq!(ids(&bytes), ids(&listed));
    assert_eq!(relisted(&plan, &samplers(32)), None, "past twice the class");
    // A new shape the plan's checks pass.
    let mut linear = SamplerDescriptor::new();
    linear.min_filter = SamplerMinMagFilter::Linear;
    let mut cubic = SamplerDescriptor::new();
    cubic.min_filter = SamplerMinMagFilter::Cubic;
    cubic.mag_filter = SamplerMinMagFilter::Cubic;
    let mut lod = SamplerDescriptor::new();
    lod.lod_min_clamp = 2.0;
    lod.lod_max_clamp = 1.0;
    let with = |desc: SamplerDescriptor| {
        let mut list = samplers(1);
        list.push(argument_sampler(desc, C));
        assert!(!encoded
            .shape()
            .argument_samplers
            .shapes
            .contains(&desc.shape()));
        list
    };
    let tables = relisted(&plan, &with(linear)).expect("a linear state");
    let flags = |tables: &[DescriptorBytes]| -> u32 {
        sampler_entries(tables, tset, tbinding)
            .iter()
            .find(|e| e.0 == C)
            .map(|e| e.2)
            .expect("listed")
    };
    assert_eq!(flags(&tables), SamplerWord::of(&linear).0);
    let tables = relisted(&plan, &with(cubic)).expect("a cubic state beside the test");
    assert_eq!(flags(&tables) & SamplerWord::CUBIC, SamplerWord::CUBIC);
    assert_eq!(relisted(&plan, &with(lod)), None, "invalid LOD clamps");
    let Some(mut untested) = argument_sampler_pipeline(&dev, "argument_sampler_sample") else {
        return;
    };
    untested.dependencies.retain(|d| {
        d.fact
            != RuntimeFact::CubicArgumentSampler {
                stage: Stage::Kernel,
            }
    });
    let plan = untested.plan(&dev, &encoded.shape()).expect("plan");
    assert!(relisted(&plan, &with(linear)).is_some());
    assert_eq!(relisted(&plan, &with(cubic)), None, "beside no test");
}

/// A filtered `float` sample of a texture an argument buffer holds records its filter note on a
/// device that samples through Metal's own sampler (`moltenvk-m3`), under what the command's
/// handles and an argument buffer's sampler can name (`lower::sampler::linear::filter_note`): the
/// plan gives `SpecKey::HeapYCbCrTextures` 1 exactly where a texture the command makes resident is
/// two-plane YCbCr, and `SpecKey::FilteringArgumentSampler` 1 exactly where a listed argument
/// sampler filters (a linear minification, magnification or mip filter, or anisotropy), and the
/// note holds (`StageModule::precision_notes_bound`) exactly where both do. Through a runtime
/// sampler slot the bound sampler's word decides in the second's place. A comparison of a depth
/// texture, whose class holds no two-plane texture, and an integer texture's sample record no
/// such note; lavapipe, which samples natively, records none.
#[test]
fn a_heap_texture_s_ycbcr_filter_note_holds_where_its_sampler_can_filter() {
    use crate::metal::SamplerMipFilter;
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::sym::SpecKey;
    let note = PrecisionNote::FloatSampleFilter;
    let dev = m3();
    let (Some(argument), Some(compare), Some(integer)) = (
        argument_sampler_pipeline(&dev, "argument_sampler_sample"),
        argument_sampler_pipeline(&dev, "argument_sampler_compare"),
        argument_sampler_pipeline(&dev, "argument_sampler_integer"),
    ) else {
        return;
    };
    let Some(bc) = assembled("handles", llvm_assemble(&engine_dir().join("handles.ll"))) else {
        return;
    };
    let runtime = compute(&dev, &bc, "argument_buffer_runtime_sampler");
    let keys = |p: &ComputePipeline| -> Vec<Vec<SpecKey>> {
        (p.stage.specialized_notes.iter())
            .filter(|s| s.note == note)
            .flat_map(|s| &s.when)
            .map(|c| c.all.iter().map(|t| t.key).collect())
            .collect()
    };
    let heap = SpecKey::HeapYCbCrTextures(Stage::Kernel);
    assert_eq!(
        keys(&argument),
        vec![vec![heap, SpecKey::FilteringArgumentSampler]]
    );
    let sampler = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Sampler,
        index: 0,
    };
    let runtime_keys = keys(&runtime);
    assert!(!runtime_keys.is_empty());
    for k in &runtime_keys {
        assert!(
            k.contains(&heap) && k.contains(&SpecKey::BoundSamplerWord(sampler)),
            "{k:?}"
        );
    }
    for p in [&compare, &integer] {
        assert!(keys(p).is_empty(), "{:?}", p.stage.specialized_notes);
        assert!(!p.stage.precision_notes.contains(&note));
    }
    let lavapipe = {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/lavapipe.json");
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    for entry in ["argument_sampler_sample", "argument_sampler_compare"] {
        let native = argument_sampler_pipeline(&lavapipe, entry).expect("lavapipe");
        assert!(keys(&native).is_empty(), "{entry} on lavapipe");
        assert!(!native.stage.precision_notes.contains(&note), "{entry}");
    }
    let desc = |linear: bool, mip: SamplerMipFilter, anisotropy: u32| {
        let mut d = SamplerDescriptor::new();
        if linear {
            d.min_filter = SamplerMinMagFilter::Linear;
            d.mag_filter = SamplerMinMagFilter::Linear;
        }
        d.mip_filter = mip;
        d.max_anisotropy = anisotropy;
        d
    };
    let nearest = desc(false, SamplerMipFilter::NotMipmapped, 1);
    // Each list of live argument samplers, and whether one of them filters.
    let lists = [
        ("none", vec![], false),
        ("nearest", vec![nearest], false),
        (
            "nearest mip filter",
            vec![desc(false, SamplerMipFilter::Nearest, 1)],
            false,
        ),
        (
            "nearest and linear",
            vec![nearest, desc(true, SamplerMipFilter::NotMipmapped, 1)],
            true,
        ),
        (
            "linear mip filter",
            vec![desc(false, SamplerMipFilter::Linear, 1)],
            true,
        ),
        (
            "anisotropic",
            vec![desc(false, SamplerMipFilter::NotMipmapped, 16)],
            true,
        ),
    ];
    let formats = [
        (PixelFormat::YCBCR84442P, true),
        (PixelFormat::YCBCR104442PSRGB, true),
        (PixelFormat::RGBA16Float, false),
        (PixelFormat::RGBA8Unorm, false),
    ];
    for (pf, ycbcr) in formats {
        for (name, list, filters) in &lists {
            let what = format!("{pf:?} beside {name}");
            let state = BindState {
                used_resources: vec![used(1, TextureType::Type2D, pf, read(), Some(A))],
                argument_samplers: (list.iter().zip(1u64..))
                    .map(|(d, id)| argument_sampler(*d, id))
                    .collect(),
                ..Default::default()
            };
            let plan = argument
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                plan.specialized_value(heap),
                Some(u64::from(ycbcr)),
                "{what}"
            );
            let filtering = plan.specialized_value(SpecKey::FilteringArgumentSampler);
            assert_eq!(filtering, Some(u64::from(*filters)), "{what}");
            let held = (argument.stage)
                .precision_notes_bound(&plan.specialization, &|k| plan.specialized_value(k))
                .contains(&note);
            assert_eq!(held, ycbcr && *filters, "{what}");
        }
        // Through a runtime sampler slot: the bound sampler's word.
        for (name, d, filters) in [
            ("nearest", nearest, false),
            (
                "linear",
                desc(true, SamplerMipFilter::NotMipmapped, 1),
                true,
            ),
        ] {
            let what = format!("{pf:?} through a runtime {name} sampler");
            let mut state = BindState {
                used_resources: vec![used(1, TextureType::Type2D, pf, read(), Some(A))],
                ..Default::default()
            };
            let mut t = StageTable::default();
            t.samplers.insert(0, BoundSampler::new(d));
            state.stages.insert(Stage::Kernel, t);
            let plan = runtime
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                plan.specialized_value(heap),
                Some(u64::from(ycbcr)),
                "{what}"
            );
            let held = (runtime.stage)
                .precision_notes_bound(&plan.specialization, &|k| plan.specialized_value(k))
                .contains(&note);
            assert_eq!(held, ycbcr && filters, "{what}");
        }
    }
}
