//! Tests of the public compile API: library parsing and lookup, compile keys, compute and render
//! pipelines from `Function` and `Compiled` stage sources, the work a `Compiled` stage does not do,
//! panics surfacing as `Internal`, and concurrent compiles of one function.
//!
//! `llvm-as` assembles the owned fixtures; a missing tool skips a test visibly. SPIR-V is checked
//! with L3 (`spirv-val` under the profile, requirements ⊆ profile) when `spirv-val` is present.

use std::path::Path;

use crate::air::container::sha256;
use crate::air::error::{AbiError, ContainerError, InputErrorKind};
use crate::budget::{Budget, Meter};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{DescriptorError, Error, Feature, InternalKind, Needs};
use crate::library::{FunctionConstantValue, FunctionConstantValues, FunctionStage, Library};
use crate::metal;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor,
    RenderPipelineDescriptor, StageInputDescriptor, StageSource, TessellationDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::lower_inputs::{LoweringInputs, VertexFetchProjection};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::{check_words_l3, L3Failure};
use crate::vk;
use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};

/// A test-only panic in the lowering path, armed per thread.
pub(crate) mod inject {
    use std::cell::Cell;

    thread_local! {
        static ARMED: Cell<bool> = const { Cell::new(false) };
    }

    /// Panic when this thread armed the injection.
    pub(crate) fn panic_if_armed() {
        if ARMED.with(Cell::get) {
            panic!("injected lowering panic");
        }
    }

    /// Run `f` with the injection armed on this thread.
    pub(crate) fn armed<T>(f: impl FnOnce() -> T) -> T {
        ARMED.with(|a| a.set(true));
        let out = f();
        ARMED.with(|a| a.set(false));
        out
    }
}

/// Owned fixture `stem` as opaque bitcode, or `None` (skipped visibly) without `llvm-as`.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// Committed profile `stem` as a device with the default policy.
fn device(stem: &str) -> Device {
    Device::new(description(stem), DevicePolicy::default()).expect("device")
}

/// Committed profile `stem` as the Khronos reader describes it.
fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

/// L3 of one stage's final words; a missing `spirv-val` skips visibly.
fn l3(words: &[u32], requirements: &crate::device::requirements::Requirements, dev: &Device) {
    match check_words_l3(words, requirements, dev.profile()) {
        Ok(()) => {}
        Err(L3Failure::Tooling(why)) => eprintln!("SKIPPED spirv-val: {why}"),
        Err(e) => panic!("L3: {e}"),
    }
}

fn kernel(f: crate::library::Function) -> ComputePipelineDescriptor {
    ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)))
}

fn has_tint(v: bool) -> FunctionConstantValues {
    FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(metal::DataType::Bool, vec![u8::from(v)]),
    )
}

#[test]
fn public_types_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Library>();
    send_sync::<crate::library::Function>();
    send_sync::<crate::library::CompiledFunction>();
    send_sync::<Device>();
    send_sync::<super::api::RenderPipelineKeys>();
}

/// A wrapper or raw module is a library of **every** entry it declares, in metadata order, with its
/// constants; lookup by name and index is typed.
#[test]
fn from_air_enumerates_every_entry() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    assert_eq!(lib, Library::parse(&bc).expect("parse"));
    assert_eq!(lib.content_hash(), &sha256(&bc));
    let names: Vec<(&str, FunctionStage)> = lib.functions().map(|f| (f.name, f.stage)).collect();
    // The fixture declares five vertex and four fragment entries.
    assert_eq!(names.len(), 9, "{names:?}");
    assert!(names.contains(&("stage_io_vertex", FunctionStage::Vertex)));
    assert!(names.contains(&("stage_io_fragment", FunctionStage::Fragment)));
    // Vertex entries come before fragment entries (`!air.vertex`, then `!air.fragment`).
    let first_fragment = names.iter().position(|n| n.1 == FunctionStage::Fragment);
    assert!(names[..first_fragment.unwrap_or(0)]
        .iter()
        .all(|n| n.1 == FunctionStage::Vertex));
    for f in lib.functions() {
        assert_eq!(f.air_hash, sha256(&bc));
    }
    let v = lib.function("stage_io_vertex").expect("vertex");
    assert_eq!(v.info().name, "stage_io_vertex");
    assert_eq!(lib.function_at(v.index()).expect("by index"), v);
    assert!(matches!(
        lib.function("no_such_entry"),
        Err(Error::InvalidDescriptor(DescriptorError::EntryNotFound { available, .. }))
            if available == lib.len()
    ));
    assert!(matches!(
        lib.function_at(lib.len()),
        Err(Error::InvalidDescriptor(
            DescriptorError::EntryNotFound { .. }
        ))
    ));

    let Some(gated) = fixture("stage_io_gated") else {
        return;
    };
    let lib = Library::from_air(&gated).expect("library");
    let gated_fn = lib.function("stage_io_vertex_gated").expect("gated");
    let info = gated_fn.info();
    assert_eq!(info.constants.len(), 1);
    let c = &info.constants[0];
    assert_eq!((c.name.as_str(), c.index), ("has_tint", Some(0)));
    // AIR metadata spells the type only as an MSL name.
    assert_eq!(c.ty, None);
}

/// A metallib's table comes from its tags alone (no bitcode is decoded at parse), its function
/// `HASH` is the air hash, `from_air` refuses a metallib, and `from_metallib` refuses bare AIR.
#[test]
fn a_metallib_parses_from_its_tags_alone() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let wrapped = wrap(&bc);
    // The second function's bitcode is not bitcode: parsing never decodes it.
    let garbage = wrap(&[0x42, 0x43, 0xC0, 0xDE, 1, 2, 3, 4]);
    let bytes = metallib(&[
        MtlbFunction {
            name: "store_const",
            type_code: 2,
            bitcode: wrapped.clone(),
            constants: vec![("scale", 3, 5)],
            public: vec![],
            tags: vec![],
        },
        MtlbFunction {
            name: "broken",
            type_code: 2,
            bitcode: garbage,
            constants: vec![],
            public: vec![],
            tags: vec![],
        },
    ]);
    let lib = Library::parse(&bytes).expect("metallib");
    assert_eq!(lib.len(), 2);
    let store = lib.function("store_const").expect("function");
    let info = store.info();
    assert_eq!(info.stage, FunctionStage::Kernel);
    assert_eq!(info.air_hash, sha256(&wrapped));
    assert_eq!(info.constants[0].ty, Some(metal::DataType::Float));
    assert_eq!(info.constants[0].index, Some(5));
    assert!(matches!(
        Library::from_air(&bytes),
        Err(Error::InvalidInput(e)) if matches!(e.kind, InputErrorKind::Container(ContainerError::BadMagic))
    ));
    // `from_metallib` takes the metallib and refuses bare AIR, raw or wrapped, as Metal's
    // `newLibraryWithData:` does.
    assert_eq!(Library::from_metallib(&bytes).expect("metallib").len(), 2);
    for bare in [&bc, &wrapped] {
        assert!(matches!(
            Library::from_metallib(bare),
            Err(Error::InvalidInput(e)) if matches!(e.kind, InputErrorKind::Container(ContainerError::BadMagic))
        ));
    }
    // Compiling the broken function fails at decode, typed.
    let dev = device("lunarg-desktop-baseline-2023");
    let broken = lib.function("broken").expect("broken");
    assert!(matches!(
        dev.compile_compute_pipeline(&kernel(broken), &Budget::standard()),
        Err(Error::InvalidInput(_))
    ));
}

/// A metallib function compiles through its container cross-check: its `CNST` entries must match
/// the declared constants, so a wrong `MTLDataType` refuses the function.
#[test]
fn a_metallib_function_compiles_after_its_container_check() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let dev = device("lunarg-desktop-baseline-2023");
    let bytes = metallib(&[MtlbFunction {
        name: "store_const",
        type_code: 2,
        bitcode: wrap(&bc),
        constants: vec![],
        public: vec![],
        tags: vec![],
    }]);
    let lib = Library::parse(&bytes).expect("metallib");
    let f = lib.function("store_const").expect("function");
    let pipe = dev
        .compile_compute_pipeline(&kernel(f.clone()), &Budget::standard())
        .expect("compute pipeline");
    l3(&pipe.stage.spirv, &pipe.requirements, &dev);

    // A `TYPE` that disagrees with the AIR entry's stage.
    let vertex_tag = metallib(&[MtlbFunction {
        name: "store_const",
        type_code: 0,
        bitcode: wrap(&bc),
        constants: vec![],
        public: vec![],
        tags: vec![],
    }]);
    let lib = Library::parse(&vertex_tag).expect("metallib");
    let err = dev
        .compile_compute_pipeline(
            &kernel(lib.function("store_const").expect("f")),
            &Budget::standard(),
        )
        .expect_err("mismatch");
    assert!(
        matches!(&err, Error::InvalidInput(e) if matches!(e.kind, InputErrorKind::Abi(AbiError::ContainerMismatch))),
        "{err:?}"
    );

    let Some(gated) = fixture("stage_io_gated") else {
        return;
    };
    for (ty, ok) in [(0x35u8, true), (3u8, false)] {
        let bytes = metallib(&[MtlbFunction {
            name: "stage_io_vertex_gated",
            type_code: 0,
            bitcode: wrap(&gated),
            constants: vec![("has_tint", ty, 0)],
            public: vec![],
            tags: vec![],
        }]);
        let lib = Library::parse(&bytes).expect("metallib");
        let mut fd = FunctionDescriptor::new(lib.function("stage_io_vertex_gated").expect("f"));
        fd.constants = has_tint(true);
        let r = dev.compile_function(&fd, &LoweringInputs::new(), &Budget::standard());
        assert_eq!(r.is_ok(), ok, "MTLDataType {ty}: {r:?}");
    }
}

/// A compute pipeline end to end from a `Function` source: its words pass L3, its key is the
/// no-decode key, and its Metal info answers the compute queries.
#[test]
fn a_compute_pipeline_compiles_from_a_function() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("store_const").expect("function");
    for stem in ["moltenvk-m3", "lunarg-desktop-baseline-2023", "vk12-floor"] {
        let dev = device(stem);
        let mut d = kernel(f.clone());
        d.max_total_threads_per_threadgroup = Some(64);
        d.support_indirect_command_buffers = true;
        let pipe = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .expect(stem);
        assert_eq!(pipe.key, dev.compute_pipeline_key(&d), "{stem}");
        assert_eq!(pipe.stage.entry_point, "main");
        assert_eq!(pipe.stage.stage, crate::vk::ShaderStage::COMPUTE);
        assert_eq!(pipe.layout.sets.len(), 1);
        assert_eq!(pipe.layout.sets[0].bindings.len(), 1, "{stem}");
        assert_eq!(pipe.metal_info.max_total_threads_per_threadgroup, 64);
        assert_eq!(pipe.metal_info.thread_execution_width, 32);
        assert_eq!(pipe.metal_info.static_threadgroup_memory_length, 0);
        assert!(pipe.metal_info.support_indirect_command_buffers);
        assert!(pipe
            .dependencies
            .iter()
            .all(|dep| dep.mechanism == crate::device::mechanism(&dep.fact, dev.profile())));
        l3(&pipe.stage.spirv, &pipe.requirements, &dev);
        // Metal's 0 is the device maximum.
        d.max_total_threads_per_threadgroup = None;
        let pipe = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .expect(stem);
        assert_eq!(
            pipe.metal_info.max_total_threads_per_threadgroup,
            dev.profile()
                .fields()
                .limits
                .max_compute_work_group_invocations
        );
    }
}

/// A `Compiled` stage answers what it may store through each of its slots from what its compile
/// kept, with no decode, and its answer is the `Function` path's: the two read the same entry's
/// roles, declared accesses and slots. It is the answer a consumer needs for a state it already
/// made, and a compile keeps it whatever a reflection of the same entry could describe.
#[test]
fn a_compiled_stage_answers_its_slot_writes_without_decoding() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let f = Library::from_air(&bc)
        .expect("library")
        .function("store_const")
        .expect("function");
    let fd = FunctionDescriptor::new(f.clone());
    let inp = LoweringInputs::new();
    let compiled = dev
        .compile_function(&fd, &inp, &Budget::standard())
        .expect("compile");

    let from_function = ComputePipelineDescriptor::new(StageSource::Function(fd))
        .slot_writes(&Budget::default())
        .expect("the function path answers");
    let from_compiled = ComputePipelineDescriptor::new(StageSource::Compiled(compiled))
        .slot_writes(&Budget::default())
        .expect("the compiled path answers");
    assert_eq!(from_compiled, from_function);

    // `device uint *out [[buffer(0)]]`: read-write, as Metal answers a device buffer.
    let slots = from_compiled.kernel_slots();
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].name, "out");
    assert_eq!(slots[0].index, Some(0));
    assert_eq!(
        crate::reflection::SlotWrite::may_write(slots, metal::BindingType::Buffer, 0),
        crate::reflection::SlotWritability::Maybe
    );
    // No argument binds buffer 1, so nothing stores through it.
    assert_eq!(
        crate::reflection::SlotWrite::may_write(slots, metal::BindingType::Buffer, 1),
        crate::reflection::SlotWritability::Never
    );
}

/// A `Compiled` stage is link and relocate only. The pipeline meter does no work (no decode, no
/// lowering), and the result equals the `Function` path's.
#[test]
fn a_compiled_stage_links_without_decoding() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let f = Library::from_air(&bc)
        .expect("library")
        .function("store_const")
        .expect("function");
    let fd = FunctionDescriptor::new(f.clone());
    let inp = LoweringInputs::new();
    let compiled = dev
        .compile_function(&fd, &inp, &Budget::standard())
        .expect("compile");
    assert_eq!(compiled.key(), &dev.function_compile_key(&fd, &inp));
    // The artifact carries its provenance.
    assert_eq!(compiled.artifact().profile, dev.profile().fingerprint());

    let mut m = Meter::new(&Budget::default());
    let from_function = dev
        .compute_pipeline(&kernel(f), &inp, &mut m)
        .expect("function");
    assert!(m.work() > 0, "the function path decodes and lowers");

    let d = ComputePipelineDescriptor::new(StageSource::Compiled(compiled.clone()));
    let mut m = Meter::new(&Budget::default());
    let from_compiled = dev.compute_pipeline(&d, &inp, &mut m).expect("compiled");
    assert_eq!(
        (m.work(), m.ledger_peak()),
        (0, 0),
        "a compiled stage is not decoded"
    );
    assert_eq!(from_compiled, from_function);
    assert_eq!(from_compiled.key, dev.compute_pipeline_key(&d));

    // A function compiled for another profile is refused, not linked.
    let other = device("vk12-floor");
    assert_eq!(
        other.compile_compute_pipeline(&d, &Budget::standard()),
        Err(Error::InvalidDescriptor(
            DescriptorError::ForeignCompiledFunction
        ))
    );
}

/// A value for a constant the function does not declare, by index or by name, is ignored, as
/// Metal's `newFunctionWithName:constantValues:` ignores it: the function compiles to the artifact
/// it compiles to without the value (the key, which covers the values as supplied, aside), a
/// declared constant's value still binds, and a declared constant's value of another type is still
/// refused.
#[test]
fn values_for_undeclared_constants_are_ignored() {
    let Some(bc) = fixture("stage_io_gated") else {
        return;
    };
    let f = Library::from_air(&bc)
        .expect("library")
        .function("stage_io_vertex_gated")
        .expect("function");
    let dev = device("moltenvk-m3");
    let compile = |constants: FunctionConstantValues| {
        let mut fd = FunctionDescriptor::new(f.clone());
        fd.constants = constants;
        dev.compile_function(&fd, &LoweringInputs::new(), &Budget::standard())
    };
    let uint = || FunctionConstantValue::new(metal::DataType::UInt, 7u32.to_le_bytes());
    for tint in [true, false] {
        let declared = compile(has_tint(tint)).expect("the declared constant alone");
        let extra = compile(
            has_tint(tint)
                .with_index(9, uint())
                .with_name("undeclared", uint()),
        )
        .expect("a value of no declared constant is ignored");
        let mut ignored = extra.artifact().clone();
        assert_ne!(ignored.key, declared.artifact().key);
        ignored.key = declared.artifact().key;
        assert_eq!(&ignored, declared.artifact(), "has_tint {tint}");
    }
    // The declared constant's value binds: the two values compile differently.
    assert_ne!(
        compile(has_tint(true)).expect("true").artifact().module,
        compile(has_tint(false)).expect("false").artifact().module
    );
    // A value of another type for the declared constant is still refused.
    let wrong = FunctionConstantValues::defaults().with_index(0, uint());
    assert_eq!(
        compile(wrong).expect_err("a uint for a bool constant"),
        Error::InvalidDescriptor(DescriptorError::FunctionConstantType { index: 0 })
    );
}

/// The key covers the constants, the profile, and the lowering inputs, and needs no decode; an
/// empty vertex-fetch projection is the canonical "none".
#[test]
fn function_keys_cover_their_inputs() {
    let Some(bc) = fixture("stage_io_gated") else {
        return;
    };
    let f = Library::from_air(&bc)
        .expect("library")
        .function("stage_io_vertex_gated")
        .expect("function");
    let (a, b) = (device("moltenvk-m3"), device("vk12-floor"));
    let mut fd = FunctionDescriptor::new(f);
    let none = LoweringInputs::new();
    let base = a.function_compile_key(&fd, &none);
    assert_ne!(base, b.function_compile_key(&fd, &none));
    let mut empty = LoweringInputs::new();
    empty.vertex_fetch = Some(VertexFetchProjection::new());
    assert_eq!(base, a.function_compile_key(&fd, &empty));
    fd.constants = has_tint(true);
    let with_true = a.function_compile_key(&fd, &none);
    assert_ne!(base, with_true);
    fd.constants = has_tint(false);
    assert_ne!(with_true, a.function_compile_key(&fd, &none));
    // The compile's own key is the no-decode key.
    fd.constants = has_tint(true);
    let c = a
        .compile_function(&fd, &none, &Budget::standard())
        .expect("compile");
    assert_eq!(*c.key(), with_true);
    // The emulated-simdgroups input is part of the key.
    let mut emulated = LoweringInputs::new();
    emulated.emulated_simdgroups = true;
    assert_ne!(with_true, a.function_compile_key(&fd, &emulated));
    // So is the dropped vertex `Layer` output.
    let mut dropped = LoweringInputs::new();
    dropped.dropped_vertex_layer = true;
    let dropped = a.function_compile_key(&fd, &dropped);
    assert_ne!(with_true, dropped);
    assert_ne!(a.function_compile_key(&fd, &emulated), dropped);
    // So are the signed-integer color attachments, by attachment.
    let signed = |a_: &[u32]| {
        let mut i = LoweringInputs::new();
        i.signed_color_attachments = a_.iter().copied().collect();
        a.function_compile_key(&fd, &i)
    };
    assert_ne!(with_true, signed(&[0]));
    assert_ne!(signed(&[0]), signed(&[1]));
    assert_ne!(dropped, signed(&[0]));
    // So is a multisampled pipeline's fragment input.
    let mut multisampled = LoweringInputs::new();
    multisampled.multisampled = true;
    let multisampled = a.function_compile_key(&fd, &multisampled);
    assert_ne!(with_true, multisampled);
    assert_ne!(dropped, multisampled);
    assert_ne!(signed(&[0]), multisampled);
    // So is a device-replay variant.
    let mut replay = LoweringInputs::new();
    replay.replay_bindings = true;
    let replay = a.function_compile_key(&fd, &replay);
    assert_ne!(with_true, replay);
    assert_ne!(multisampled, replay);
    // So is OpenCL mode, which binds the hidden buffers and honors `air.convert`'s suffixes: a
    // pipeline of either mode never takes the other's translation.
    let mut cl = LoweringInputs::new();
    cl.open_cl_mode = true;
    let cl = a.function_compile_key(&fd, &cl);
    assert_ne!(with_true, cl);
    assert_ne!(replay, cl);
    // So is the interlocked framebuffer fetch's pipeline state, by its attachments' state.
    let fetch = |format| {
        let mut i = LoweringInputs::new();
        let colors = [Some(
            crate::pipeline::descriptor::ColorAttachmentDescriptor::new(format),
        )];
        i.framebuffer_fetch = Some(crate::pipeline::FramebufferFetchInputs::of(
            [0],
            &colors,
            false,
            false,
            1,
        ));
        a.function_compile_key(&fd, &i)
    };
    let r32 = fetch(metal::PixelFormat::R32Float);
    assert_ne!(with_true, r32);
    assert_ne!(replay, r32);
    assert_ne!(r32, fetch(metal::PixelFormat::RGBA8Unorm));
    // So is the raw viewport index's varying.
    let mut viewport = LoweringInputs::new();
    viewport.viewport_index_varying = true;
    let viewport = a.function_compile_key(&fd, &viewport);
    assert_ne!(with_true, viewport);
    assert_ne!(multisampled, viewport);
    assert_ne!(replay, viewport);
    // So is a flat replay variant.
    let mut flat = LoweringInputs::new();
    flat.replay_bindings = true;
    flat.flat_grid = true;
    assert_ne!(replay, a.function_compile_key(&fd, &flat));
    // So is a flat variant of the call's threadgroups.
    let mut plain_flat = LoweringInputs::new();
    plain_flat.flat_grid = true;
    let mut call_flat = plain_flat.clone();
    call_flat.call_threadgroups = true;
    assert_ne!(
        a.function_compile_key(&fd, &plain_flat),
        a.function_compile_key(&fd, &call_flat)
    );
    // So is the raw render target slice index's varying.
    let mut layer = LoweringInputs::new();
    layer.render_target_array_index_varying = true;
    let layer = a.function_compile_key(&fd, &layer);
    assert_ne!(with_true, layer);
    assert_ne!(viewport, layer);
    assert_ne!(dropped, layer);
    // So are draws of one amplification each, by what the vertex function writes.
    let draws = |layered, viewports| {
        let mut i = LoweringInputs::new();
        i.amplification_draws = Some(crate::pipeline::AmplificationDraws::new(layered, viewports));
        a.function_compile_key(&fd, &i)
    };
    let plain = draws(false, false);
    assert_ne!(with_true, plain);
    assert_ne!(layer, plain);
    assert_ne!(plain, draws(true, false));
    assert_ne!(plain, draws(false, true));
    assert_ne!(draws(true, false), draws(false, true));
    // So is how the fragment function rounds its color outputs, by attachment and rule.
    use crate::pipeline::ChannelWrite;
    let writes = |entries: &[(u32, [Option<ChannelWrite>; 4])]| {
        let mut i = LoweringInputs::new();
        i.color_writes = entries.iter().copied().collect();
        a.function_compile_key(&fd, &i)
    };
    let unorm = [Some(ChannelWrite::Unorm { bits: 8 }); 4];
    let srgb = [
        Some(ChannelWrite::Srgb),
        Some(ChannelWrite::Srgb),
        Some(ChannelWrite::Srgb),
        Some(ChannelWrite::Unorm { bits: 8 }),
    ];
    let device = [Some(ChannelWrite::Device); 4];
    assert_ne!(with_true, writes(&[(0, unorm)]));
    assert_ne!(writes(&[(0, unorm)]), writes(&[(1, unorm)]));
    assert_ne!(writes(&[(0, unorm)]), writes(&[(0, srgb)]));
    assert_ne!(writes(&[(0, unorm)]), writes(&[(0, device)]));
    assert_ne!(layer, writes(&[(0, device)]));
    // So are the address-mode maps of constexpr-sampled slots, by slot and map; an empty set of
    // maps (every ordinary texture's) keeps the key.
    use crate::metal::sampler::AddressModeMap;
    use crate::metal::SamplerAddressMode as M;
    let maps = |entries: &[(u32, AddressModeMap)]| {
        let mut i = LoweringInputs::new();
        i.address_maps = entries
            .iter()
            .map(|&(index, map)| {
                let slot = crate::pipeline::sym::SlotRef {
                    stage: crate::pipeline::sym::Stage::Vertex,
                    class: crate::pipeline::sym::SlotClass::Texture,
                    index,
                };
                (slot, map)
            })
            .collect();
        a.function_compile_key(&fd, &i)
    };
    let edges = AddressModeMap::IDENTITY.with(M::Repeat, M::ClampToEdge);
    let mirrored = AddressModeMap::IDENTITY.with(M::Repeat, M::MirrorRepeat);
    assert_eq!(with_true, maps(&[]));
    assert_ne!(with_true, maps(&[(0, edges)]));
    assert_ne!(maps(&[(0, edges)]), maps(&[(1, edges)]));
    assert_ne!(maps(&[(0, edges)]), maps(&[(0, mirrored)]));
    // So are the extended-range colors whose `half` shadow the draw's instance holds.
    let shadowed = |colors: &[u32]| {
        let mut i = LoweringInputs::new();
        i.extended_shadows = colors.iter().copied().collect();
        a.function_compile_key(&fd, &i)
    };
    assert_eq!(with_true, shadowed(&[]));
    assert_ne!(with_true, shadowed(&[0]));
    assert_ne!(shadowed(&[0]), shadowed(&[1]));
    assert_ne!(shadowed(&[0]), shadowed(&[0, 1]));
}

/// A key without the emulated-simdgroups and dropped-vertex-layer inputs hashes the inputs as it
/// did before those inputs existed, the derived `Debug` of the three earlier fields, so no cached
/// function of the common path changes key.
#[test]
fn the_inputs_encoding_is_the_earlier_derived_debug() {
    use crate::pipeline::compile::key::inputs_encoding;
    use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass, VertexFetch};
    use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
    let slot = |class, index| SlotRef {
        stage: Stage::Kernel,
        class,
        index,
    };
    let mut full = LoweringInputs::new();
    let mut projection = VertexFetchProjection::new();
    projection.attributes.insert(2, VertexFetch::FixedFunction);
    full.vertex_fetch = Some(projection);
    full.storage_images.insert(
        slot(SlotClass::Texture, 1),
        StorageImageChoice::new(
            crate::vk::Format::R8G8B8A8_UNORM,
            StorageImageClass::Identity,
        ),
    );
    full.psb_slots.insert(slot(SlotClass::Buffer, 4));
    for inputs in [LoweringInputs::new(), full] {
        let derived = format!("{inputs:?}");
        assert_eq!(
            inputs_encoding(&inputs),
            derived
                .replace(", emulated_simdgroups: false", "")
                .replace(", one_row: false", "")
                .replace(", dropped_vertex_layer: false", "")
                .replace(", stage_input: None", "")
                .replace(", storage_view_slots: {}", "")
                .replace(", signed_color_attachments: {}", "")
                .replace(", tile_attachments: None", "")
                .replace(", multisampled: false", "")
                .replace(", replay_bindings: false", "")
                .replace(", merged_draws: false", "")
                .replace(", framebuffer_fetch: None", "")
                .replace(", srgb_textures: {}", "")
                .replace(", snapshot_textures: {}", "")
                .replace(", ycbcr_in_place: {}", "")
                .replace(", ycbcr_planes_sampled: {}", "")
                .replace(", texel_rows: {}", "")
                .replace(", texel_buffer_parts: {}", "")
                .replace(", viewport_index_varying: false", "")
                .replace(", max_call_stack_depth: None", "")
                .replace(", open_cl_mode: false", "")
                .replace(", flat_grid: false", "")
                .replace(", call_threadgroups: false", "")
                .replace(", render_target_array_index_varying: false", "")
                .replace(", primitive_id_varying: false", "")
                .replace(", primitive_id_indices: None", "")
                .replace(", quad_primitive_ids: false", "")
                .replace(", private_list: false", "")
                .replace(", amplification_draws: None", "")
                .replace(", color_writes: {}", "")
                .replace(", primitive_varyings: {}", "")
                .replace(", absent_varyings: {}", "")
                .replace(", viewport_compensation: None", "")
                .replace(", folded_groups: false", "")
                .replace(", partial_threadgroups: false", "")
                .replace(", extended_range: {}", "")
                .replace(", extended_range_colors: {}", "")
                .replace(", word_colors: {}", "")
                .replace(", zeroed_colors: {}", "")
                .replace(", dropped_colors: {}", "")
                .replace(", kept_imageblock_bytes: None", "")
                .replace(", depth_clamp_range: false", "")
                .replace(", rotated_weights: false", "")
                .replace(", swizzled_storage: {}", "")
                .replace(", addressed_function_tables: false", "")
                .replace(", unattached_colors: {}", "")
                .replace(", sample_mask: None", "")
                .replace(", alpha_test: None", "")
                .replace(", logic_operation: None", "")
                .replace(", depth_compare_clamp_mask: 0", "")
                .replace(", read_only_tests: false", "")
                .replace(", shader_tests: false", "")
                .replace(", shader_stencil: false", "")
                .replace(", window_depth_varying: false", "")
                .replace(", rotated_colors: {}", "")
                .replace(", alpha_in_red_colors: {}", "")
                .replace(", rotated_factors: {}", "")
                .replace(", threadgroup_size_matches_tile_size: false", "")
                .replace(", address_maps: {}", "")
                .replace(", cubic_samplers: {}", "")
                .replace(", rotated_blends: {}", "")
                .replace(", extended_shadows: {}", "")
                .replace(", cross_class: {}", "")
                .replace(", texture_write_rounding: TowardZero", "")
                .replace(", anisotropic_samplers: {}", "")
                .replace(", open_cl_lines: {}", "")
                .replace(", uniform_varyings: {}", "")
                .replace(", instanced_attributes: {}", "")
                .replace(", color_group: None", "")
                .replace(", object_payload: None", "")
                .replace(", tessellation: None", "")
                .replace(", region_viewports: false", "")
                .replace(", region_copies: false", "")
                .replace(", unit_axes: 0", "")
                .replace(", push_base: 0", ""),
            "{derived}"
        );
    }
}

/// A panic in the lowering path is `Internal(Panic)`, never an unwind into the caller.
#[test]
fn a_lowering_panic_surfaces_as_internal() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let f = Library::from_air(&bc)
        .expect("library")
        .function("store_const")
        .expect("function");
    let fd = FunctionDescriptor::new(f.clone());
    let r = inject::armed(|| dev.compile_compute_pipeline(&kernel(f), &Budget::standard()));
    assert!(
        matches!(&r, Err(Error::Internal(e)) if e.kind == InternalKind::Panic),
        "{r:?}"
    );
    let r =
        inject::armed(|| dev.compile_function(&fd, &LoweringInputs::new(), &Budget::standard()));
    assert!(
        matches!(&r, Err(Error::Internal(e)) if e.kind == InternalKind::Panic),
        "{r:?}"
    );
    // The injection is per thread and disarmed again.
    assert!(dev
        .compile_function(&fd, &LoweringInputs::new(), &Budget::standard())
        .is_ok());
}

/// Concurrent compiles of one `Function` share no mutable state: every thread gets the same
/// artifact.
#[test]
fn concurrent_compiles_of_one_function_agree() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let dev = device("lunarg-desktop-baseline-2023");
    let f = Library::from_air(&bc)
        .expect("library")
        .function("store_const")
        .expect("function");
    let fd = FunctionDescriptor::new(f.clone());
    let compile = |dev: &Device| {
        let c = dev.compile_function(&fd, &LoweringInputs::new(), &Budget::standard())?;
        let p = dev.compile_compute_pipeline(&kernel(f.clone()), &Budget::standard())?;
        Ok::<_, Error>((c, p))
    };
    let serial = compile(&dev).expect("serial");
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4).map(|_| s.spawn(|| compile(&dev))).collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    for r in results {
        assert_eq!(r.expect("compile"), serial);
    }
}

/// The stage words of the owned `runtime_examples` fixture's `colored_vertex` and
/// `sampled_fragment` (a `half` sample: a texel format class constant) to an RGBA8 and a BGRA8
/// target on two profiles, each compiled on a fresh device, labelled by profile, format and
/// stage; `None` where `llvm-as` is missing.
fn determinism_stage_words() -> Option<Vec<(String, Vec<u32>)>> {
    let path = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return None;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected runtime_examples: {why}"),
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = lib.function("colored_vertex").expect("vertex");
    let fragment = lib.function("sampled_fragment").expect("fragment");
    let mut out = Vec::new();
    for stem in ["moltenvk-m3", "lunarg-desktop-baseline-2023"] {
        for format in [
            metal::PixelFormat::RGBA8Unorm,
            metal::PixelFormat::BGRA8Unorm,
        ] {
            let dev = device(stem);
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(vertex.clone()),
            ));
            d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
                fragment.clone(),
            )));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
            let pipe = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem} {format:?}: {e}"));
            assert_eq!(pipe.stages.len(), 2, "{stem} {format:?}: two stages");
            for s in &pipe.stages {
                out.push((format!("{stem} {format:?} {:?}", s.stage), s.spirv.to_vec()));
            }
        }
    }
    Some(out)
}

/// Each of [`determinism_stage_words`]' labels with the SHA-256 of its words, in hex.
fn determinism_hashes(words: &[(String, Vec<u32>)]) -> Vec<String> {
    words
        .iter()
        .map(|(label, w)| {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            let hex: String = sha256(&bytes).iter().map(|b| format!("{b:02x}")).collect();
            format!("{label} {hex}")
        })
        .collect()
}

/// A render pipeline compiled again, in other threads (each with its own hash seeds), on fresh
/// devices, gives byte-identical stage SPIR-V: the runtime's shared module key, so a state converts
/// to the same MSL text, which Metal's own shader cache finds.
#[test]
fn a_render_pipeline_s_stage_words_are_the_same_in_every_compile() {
    let Some(first) = determinism_stage_words() else {
        return;
    };
    let others: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..3).map(|_| s.spawn(determinism_stage_words)).collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    for other in others {
        assert!(other == Some(first.clone()), "the stage words differ");
    }
}

/// [`determinism_hashes`], printed for [`a_render_pipeline_s_stage_words_are_the_same_in_other_processes`].
#[test]
#[ignore = "run in a child process by a_render_pipeline_s_stage_words_are_the_same_in_other_processes"]
fn print_the_determinism_hashes() {
    if let Some(words) = determinism_stage_words() {
        for line in determinism_hashes(&words) {
            println!("WORDS {line}");
        }
    }
}

/// The same stage words in other processes (each with its own allocator, addresses and hash
/// seeds): this test's binary run twice more as child processes prints the words' hashes
/// ([`print_the_determinism_hashes`]), which are this process's. So no emission depends on a
/// `HashMap`'s order or an address, and a later process's state keys the same module and
/// converts the same MSL.
#[test]
fn a_render_pipeline_s_stage_words_are_the_same_in_other_processes() {
    let Some(words) = determinism_stage_words() else {
        return;
    };
    let mine = determinism_hashes(&words);
    let exe = std::env::current_exe().expect("the test binary");
    for run in 0..2 {
        let out = std::process::Command::new(&exe)
            .args([
                "--exact",
                "pipeline::compile::tests::print_the_determinism_hashes",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .output()
            .expect("the child process");
        assert!(out.status.success(), "child {run}: {out:?}");
        let theirs: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            // The harness may print the test's name on the line before its first output.
            .filter_map(|l| {
                l.find("WORDS ")
                    .map(|at| l[at + "WORDS ".len()..].to_string())
            })
            .collect();
        assert_eq!(theirs, mine, "child {run}'s stage words");
    }
}

/// A stage-input descriptor is a compute pipeline's lowering input, and a kernel in a render
/// pipeline's vertex slot is a stage mismatch.
#[test]
fn a_stage_input_descriptor_and_a_mismatched_stage_are_checked() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let f = Library::from_air(&bc)
        .expect("library")
        .function("store_const")
        .expect("function");
    // A stage-input descriptor is a compute pipeline's lowering input: a kernel without
    // `[[stage_in]]` compiles under one as without it.
    let mut d = kernel(f.clone());
    d.stage_input = Some(StageInputDescriptor::new());
    let with = dev
        .compile_compute_pipeline(&d, &Budget::standard())
        .expect("a kernel without stage inputs under a stage-input descriptor");
    let without = dev
        .compile_compute_pipeline(&kernel(f.clone()), &Budget::standard())
        .expect("the kernel alone");
    assert_eq!(with.stage.spirv, without.stage.spirv);
    let r = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    // A kernel in the vertex slot is a stage mismatch.
    assert!(matches!(
        dev.compile_render_pipeline(&r, &Budget::standard()),
        Err(Error::InvalidDescriptor(
            DescriptorError::StageMismatch { .. }
        ))
    ));
}

/// A post-tessellation vertex function compiles to an evaluation stage, which link puts behind the
/// synthesized vertex and control stages; its key is computed under the pulled patch projection,
/// and the descriptor's tessellation state is part of the pipeline key.
#[test]
fn a_render_pipeline_of_a_post_tessellation_function_is_tessellated() {
    let Some(bc) = fixture("tessellation") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let lib = Library::from_air(&bc).expect("library");
    for entry in ["tess_quad_buffer", "tess_triangle_points"] {
        let f = lib.function(entry).expect("function");
        let mut r =
            RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
        let mut d = VertexDescriptor::new();
        let mut points = VertexBufferLayoutDescriptor::new(Some(24));
        points.step_function = metal::VertexStepFunction::PerPatchControlPoint;
        let mut patches = VertexBufferLayoutDescriptor::new(Some(16));
        patches.step_function = metal::VertexStepFunction::PerPatch;
        d.layouts.insert(0, points);
        d.layouts.insert(1, patches);
        for (a, format, offset, buffer) in [
            (0, metal::VertexFormat::Float4, 0, 0),
            (1, metal::VertexFormat::Half2, 16, 0),
            (5, metal::VertexFormat::Float4, 0, 1),
        ] {
            d.attributes
                .insert(a, VertexAttributeDescriptor::new(format, offset, buffer));
        }
        r.vertex_descriptor = Some(d);
        let keys = dev
            .render_pipeline_key(&r, &Budget::standard())
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        let p = dev
            .compile_render_pipeline(&r, &Budget::standard())
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(p.key, keys.pipeline, "{entry}");
        let stages: Vec<vk::ShaderStage> = p.stages.iter().map(|m| m.stage).collect();
        assert_eq!(
            stages,
            [
                vk::ShaderStage::VERTEX,
                vk::ShaderStage::TESSELLATION_CONTROL,
                vk::ShaderStage::TESSELLATION_EVALUATION
            ],
            "{entry}"
        );
        r.tessellation = Some(TessellationDescriptor {
            partition_mode: metal::TessellationPartitionMode::Integer,
            ..TessellationDescriptor::default()
        });
        let other = dev
            .render_pipeline_key(&r, &Budget::standard())
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_ne!(other.pipeline, keys.pipeline, "{entry}");
    }
}

/// A vertex descriptor with each `(attribute, format)` of `attributes` in buffer 0, 16 bytes apart.
fn vertex_descriptor(attributes: &[(u32, metal::VertexFormat)]) -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    d.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(64)));
    for (offset, (a, format)) in (0u32..).step_by(16).zip(attributes) {
        d.attributes
            .insert(*a, VertexAttributeDescriptor::new(*format, offset, 0));
    }
    d
}

/// A render pipeline end to end: a vertex-only pipeline (rasterization disabled) compiles, its
/// words pass L3, its keys are the ones `render_pipeline_key` names, and the `Compiled` path
/// gives the same pipeline without work.
#[test]
fn a_render_pipeline_compiles_from_functions() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = lib.function("stage_io_vertex_position").expect("vertex");
    for stem in ["moltenvk-m3", "lunarg-desktop-baseline-2023"] {
        let dev = device(stem);
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            vertex.clone(),
        )));
        d.rasterization_enabled = false;
        let keys = dev
            .render_pipeline_key(&d, &Budget::standard())
            .expect("keys");
        let pipe = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .expect(stem);
        assert_eq!(pipe.key, keys.pipeline, "{stem}");
        assert_eq!(keys.fragment, None);
        assert_eq!(pipe.stages.len(), 1);
        assert_eq!(pipe.stages[0].stage, crate::vk::ShaderStage::VERTEX);
        l3(&pipe.stages[0].spirv, &pipe.requirements, &dev);
        let compiled = dev
            .compile_function(
                &FunctionDescriptor::new(vertex.clone()),
                &LoweringInputs::new(),
                &Budget::standard(),
            )
            .expect("compile");
        assert_eq!(*compiled.key(), keys.vertex);
        assert_eq!(
            compiled.artifact().kernel,
            crate::library::KernelFacts::default(),
            "a vertex function has no kernel facts"
        );
        d.vertex = StageSource::Compiled(compiled);
        let mut m = Meter::new(&Budget::default());
        assert_eq!(
            dev.render_pipeline(&d, &crate::cache::VariantKey::new(), &mut m)
                .expect("compiled"),
            pipe
        );
        assert_eq!(m.work(), 0);
    }

    // A vertex and a fragment stage: linked when link numbers their interfaces, else a typed
    // refusal (never a panic or an invalid module).
    let dev = device("moltenvk-m3");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        vertex.clone(),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        lib.function("stage_io_fragment_primitive")
            .expect("fragment"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA8Unorm,
    ));
    match dev.compile_render_pipeline(&d, &Budget::standard()) {
        Ok(pipe) => {
            assert_eq!(pipe.stages.len(), 2);
            let keys = dev
                .render_pipeline_key(&d, &Budget::standard())
                .expect("keys");
            assert_eq!(pipe.key, keys.pipeline);
            for s in &pipe.stages {
                l3(&s.spirv, &pipe.requirements, &dev);
            }
        }
        Err(Error::Unsupported(_)) => {}
        Err(e) => panic!("{e:?}"),
    }
}

/// A stage compiled from a creation's entries is the stage `compile_function` compiles, and the
/// compile makes no entry the creation's key already made: each stage is decoded and specialized
/// once per creation. A stage no check or key made is made there, once.
#[test]
fn a_compile_from_a_creations_entries_is_the_plain_compile() {
    use crate::pipeline::compile::StageEntries;
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = FunctionDescriptor::new(lib.function("stage_io_vertex_position").expect("vertex"));
    let fragment = FunctionDescriptor::new(
        lib.function("stage_io_fragment_primitive")
            .expect("fragment"),
    );
    let dev = device("moltenvk-m3");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex.clone()));
    d.fragment = Some(StageSource::Function(fragment.clone()));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA8Unorm,
    ));
    let budget = Budget::standard();
    let entries = StageEntries::new();
    let made = || entries.counts().0;
    let keys = dev
        .render_variant_key(&d, &crate::cache::VariantKey::new(), &budget, &entries)
        .expect("keys");
    let after_key = made();
    assert!(after_key >= 1, "the key made an entry");
    for (fd, inputs) in [
        (&vertex, keys.vertex_inputs.as_ref().expect("vertex inputs")),
        (
            &fragment,
            keys.fragment_inputs.as_ref().expect("fragment inputs"),
        ),
    ] {
        let plain = dev.compile_function(fd, inputs, &budget).expect("plain");
        let from = dev
            .compile_function_from_entries(fd, inputs, &budget, &entries)
            .expect("from the entries");
        assert_eq!(from, plain);
    }
    assert_eq!(
        made(),
        after_key.max(2),
        "one entry per stage, the key's kept"
    );
    // Fresh entries: the compile makes the stage's entry there, once.
    let fresh = StageEntries::new();
    let inputs = keys.vertex_inputs.as_ref().expect("vertex inputs");
    for _ in 0..2 {
        assert_eq!(
            dev.compile_function_from_entries(&vertex, inputs, &budget, &fresh),
            dev.compile_function(&vertex, inputs, &budget)
        );
    }
    assert_eq!(fresh.counts().0, 1);
}

/// One entry set given to the keys of pipeline states that differ only in their own fields
/// specializes each stage once, not once a state: an entry is the function, its constants, its
/// linked functions and its dynamic libraries alone, so a consumer that keeps one object and
/// gives it to every creation decodes and specializes a sweep's functions once.
#[test]
fn one_entry_set_specializes_a_sweeps_stages_once() {
    use crate::pipeline::compile::StageEntries;
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = FunctionDescriptor::new(lib.function("stage_io_vertex_position").expect("vertex"));
    let fragment = FunctionDescriptor::new(
        lib.function("stage_io_fragment_primitive")
            .expect("fragment"),
    );
    let dev = device("moltenvk-m3");
    let budget = Budget::standard();
    let entries = StageEntries::new();
    // Eight states of the one function pair, each writing its colors through a different write
    // mask, which is a field of the state and of no stage's entry.
    let masks = [
        metal::ColorWriteMask::ALL,
        metal::ColorWriteMask::RED,
        metal::ColorWriteMask::GREEN,
        metal::ColorWriteMask::BLUE,
        metal::ColorWriteMask::ALPHA,
        metal::ColorWriteMask::RED | metal::ColorWriteMask::GREEN,
        metal::ColorWriteMask::BLUE | metal::ColorWriteMask::ALPHA,
        metal::ColorWriteMask::RED | metal::ColorWriteMask::ALPHA,
    ];
    for mask in masks {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex.clone()));
        d.fragment = Some(StageSource::Function(fragment.clone()));
        let mut color = ColorAttachmentDescriptor::new(metal::PixelFormat::RGBA8Unorm);
        color.write_mask = mask;
        d.color_attachments[0] = Some(color);
        dev.render_variant_key(&d, &crate::cache::VariantKey::new(), &budget, &entries)
            .expect("keys");
    }
    let (specialized, shared) = entries.counts();
    assert_eq!(
        specialized, 2,
        "the vertex and the fragment stage, once each over the eight states"
    );
    assert!(shared >= 14, "every later ask read an entry: {shared}");
    // A set of its own for every state specializes both stages for every state, which is the
    // work the shared set spares.
    let apart = (0..8)
        .map(|_| {
            let own = StageEntries::new();
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex.clone()));
            d.fragment = Some(StageSource::Function(fragment.clone()));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
                metal::PixelFormat::RGBA8Unorm,
            ));
            dev.render_variant_key(&d, &crate::cache::VariantKey::new(), &budget, &own)
                .expect("keys");
            own.counts().0
        })
        .sum::<u64>();
    assert_eq!(apart, 16, "two stages specialized for each of the eight");
}

/// The old runtime's render descriptor key keeps its value: one known descriptor's key, pinned.
/// The old runtime keeps answers under it in its cache directory, so a change of value is a
/// change to the old runtime.
///
/// The key hashes the stages' AIR bytes, so the descriptor's functions come from a committed
/// `.air` fixture, the same bytes on every host. An `.ll` fixture is assembled by the host's
/// `llvm-as`, whose bitcode differs between LLVM releases (23.1.1, 23.1.4 and 23.1.5 each write
/// other bytes of `stage_io.ll`), which made the pinned value depend on the host that ran it.
#[test]
fn the_old_render_descriptor_key_keeps_its_value() {
    let path = engine_dir().join("fbl_fetch_dual.air");
    let bc = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let bytes: String = sha256(&bc).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        bytes, "c77a9c963c06df07997811c3b73ded8a1dfdb40a79ae300f7d285ac01f89cbdd",
        "the fixture's bytes, which the key hashes, changed: pin the key the new bytes give"
    );
    let lib = Library::from_air(&bc).expect("library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function("fbd_vertex").expect("vertex"),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        lib.function("fbd_dual_float").expect("fragment"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA8Unorm,
    ));
    let key = device("moltenvk-m3").render_descriptor_key(&d);
    let hex: String = key.digest().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex, "eeff468606512cc27447cb56b4dc3a5e4ee7f5e18b45873a95f48a9825480798",
        "the old runtime's key of the descriptor"
    );
}

/// The key on `Device::from_queries`' path is the same for an equal descriptor and on another
/// device of the same profile, and differs from the descriptor's where one field group of it
/// changes: the stages, the vertex descriptor, a colour attachment's every field, the depth and
/// stencil formats, the sample and rasterization state, the topology class and amplification,
/// tessellation, indirect command buffers, the call-stack depths, binary functions, buffer
/// mutability, the rounding mode, each private field group, and the profile; and the old
/// runtime's key tells each change apart too.
#[test]
fn a_queried_render_descriptor_key_covers_each_field_group() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = lib.function("stage_io_vertex_position").expect("vertex");
    let fragment = lib
        .function("stage_io_fragment_primitive")
        .expect("fragment");
    let dev = device("moltenvk-m3");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        vertex.clone(),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        fragment.clone(),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA8Unorm,
    ));
    let key = dev.queried_render_descriptor_key(&d);
    assert_eq!(dev.queried_render_descriptor_key(&d.clone()), key);
    assert_eq!(device("moltenvk-m3").queried_render_descriptor_key(&d), key);
    assert_ne!(
        device("lunarg-desktop-baseline-2023").queried_render_descriptor_key(&d),
        key,
        "the profile"
    );
    assert_ne!(
        dev.render_descriptor_key(&d),
        key,
        "the old runtime's values"
    );
    let with_layout = |e: &mut RenderPipelineDescriptor, stride| {
        let mut v = VertexDescriptor::new();
        v.layouts
            .insert(0, VertexBufferLayoutDescriptor::new(stride));
        v.attributes.insert(
            0,
            VertexAttributeDescriptor::new(metal::VertexFormat::Float4, 0, 0),
        );
        e.vertex_descriptor = Some(v);
    };
    let color = |e: &RenderPipelineDescriptor| e.color_attachments[0].expect("the attachment");
    type Change<'a> = Box<dyn Fn(&mut RenderPipelineDescriptor) + 'a>;
    let changes: Vec<(&str, Change<'_>)> = vec![
        ("no fragment", Box::new(|e| e.fragment = None)),
        (
            "the fragment's function",
            Box::new(|e| {
                e.fragment = Some(StageSource::Function(FunctionDescriptor::new(
                    vertex.clone(),
                )));
            }),
        ),
        (
            "constants",
            Box::new(|e| {
                if let StageSource::Function(f) = &mut e.vertex {
                    f.constants = has_tint(true);
                }
            }),
        ),
        (
            "a vertex descriptor",
            Box::new(|e| with_layout(e, Some(16))),
        ),
        ("another stride", Box::new(|e| with_layout(e, Some(32)))),
        ("the dynamic stride", Box::new(|e| with_layout(e, None))),
        (
            "a step function",
            Box::new(|e| {
                with_layout(e, Some(16));
                if let Some(v) = &mut e.vertex_descriptor {
                    if let Some(l) = v.layouts.get_mut(&0) {
                        l.step_function = metal::VertexStepFunction::PerInstance;
                    }
                }
            }),
        ),
        (
            "an attribute's offset",
            Box::new(|e| {
                with_layout(e, Some(16));
                if let Some(v) = &mut e.vertex_descriptor {
                    if let Some(a) = v.attributes.get_mut(&0) {
                        a.offset = 4;
                    }
                }
            }),
        ),
        (
            "a color format",
            Box::new(|e| {
                e.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
                    metal::PixelFormat::BGRA8Unorm,
                ));
            }),
        ),
        (
            "a second attachment",
            Box::new(|e| {
                e.color_attachments[1] = Some(color(e));
            }),
        ),
        (
            "blending",
            Box::new(|e| {
                let mut c = color(e);
                c.blending_enabled = true;
                e.color_attachments[0] = Some(c);
            }),
        ),
        (
            "a blend factor",
            Box::new(|e| {
                let mut c = color(e);
                c.destination_alpha_blend_factor = metal::BlendFactor::One;
                e.color_attachments[0] = Some(c);
            }),
        ),
        (
            "a blend operation",
            Box::new(|e| {
                let mut c = color(e);
                c.alpha_blend_operation = metal::BlendOperation::Max;
                e.color_attachments[0] = Some(c);
            }),
        ),
        (
            "a write mask",
            Box::new(|e| {
                let mut c = color(e);
                c.write_mask = metal::ColorWriteMask::RED;
                e.color_attachments[0] = Some(c);
            }),
        ),
        (
            "depth",
            Box::new(|e| e.depth_format = Some(metal::PixelFormat::Depth32Float)),
        ),
        (
            "stencil",
            Box::new(|e| e.stencil_format = Some(metal::PixelFormat::Stencil8)),
        ),
        ("samples", Box::new(|e| e.raster_sample_count = 4)),
        (
            "alpha to coverage",
            Box::new(|e| e.alpha_to_coverage = true),
        ),
        ("alpha to one", Box::new(|e| e.alpha_to_one = true)),
        (
            "rasterization",
            Box::new(|e| e.rasterization_enabled = false),
        ),
        (
            "the topology class",
            Box::new(|e| e.input_primitive_topology = metal::PrimitiveTopologyClass::Line),
        ),
        (
            "amplification",
            Box::new(|e| e.max_vertex_amplification_count = 2),
        ),
        (
            "tessellation",
            Box::new(|e| e.tessellation = Some(TessellationDescriptor::default())),
        ),
        (
            "indirect command buffers",
            Box::new(|e| e.support_indirect_command_buffers = true),
        ),
        (
            "a call-stack depth",
            Box::new(|e| e.max_fragment_call_stack_depth = 2),
        ),
        (
            "binary functions",
            Box::new(|e| e.support_adding_vertex_binary_functions = true),
        ),
        (
            "a buffer's mutability",
            Box::new(|e| {
                e.fragment_buffers.insert(0, metal::Mutability::Immutable);
            }),
        ),
        (
            "the rounding mode",
            Box::new(|e| {
                e.texture_write_rounding_mode = metal::TextureWriteRoundingMode::NearestEven
            }),
        ),
        ("the sample mask", Box::new(|e| e.private.sample_mask = 1)),
        (
            "coverage 0.0",
            Box::new(|e| e.private.sample_coverage = 0.0),
        ),
        (
            "coverage -0.0",
            Box::new(|e| e.private.sample_coverage = -0.0),
        ),
        (
            "the alpha test",
            Box::new(|e| e.private.alpha_test_function = 3),
        ),
        (
            "the logic operation",
            Box::new(|e| e.private.logic_operation = 5),
        ),
        (
            "a clip distance",
            Box::new(|e| e.private.clip_distance_enable_mask = 1),
        ),
        (
            "a private flag",
            Box::new(|e| e.private.two_side_enabled = true),
        ),
        (
            "a clamp mask",
            Box::new(|e| e.private.fragment_depth_compare_clamp_mask = 1),
        ),
        (
            "the amplification mode",
            Box::new(|e| e.private.vertex_amplification_mode = 1),
        ),
        (
            "a visibility group",
            Box::new(|e| e.private.explicit_visibility_group_id = 0),
        ),
        (
            "OpenGL mode",
            Box::new(|e| e.private.open_gl_mode_enabled = true),
        ),
    ];
    let mut seen = vec![key];
    for (what, change) in &changes {
        let mut e = d.clone();
        change(&mut e);
        let k = dev.queried_render_descriptor_key(&e);
        assert!(!seen.contains(&k), "{what} changes the key");
        assert_ne!(
            dev.render_descriptor_key(&e),
            dev.render_descriptor_key(&d),
            "{what}: the old key tells it apart too"
        );
        seen.push(k);
    }
}

/// A render descriptor's key is the same for an equal descriptor and on another device of the
/// same profile, and differs where a field, a stage, a private field, or the profile does: a
/// consumer keeping answers under it never takes another descriptor's.
#[test]
fn a_render_descriptor_key_covers_its_fields() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = lib.function("stage_io_vertex_position").expect("vertex");
    let fragment = lib
        .function("stage_io_fragment_primitive")
        .expect("fragment");
    let dev = device("moltenvk-m3");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        vertex.clone(),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        fragment.clone(),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA8Unorm,
    ));
    let key = dev.render_descriptor_key(&d);
    assert_eq!(dev.render_descriptor_key(&d.clone()), key);
    assert_eq!(device("moltenvk-m3").render_descriptor_key(&d), key);
    assert_ne!(
        device("lunarg-desktop-baseline-2023").render_descriptor_key(&d),
        key,
        "the profile"
    );
    type Change = fn(&mut RenderPipelineDescriptor, &crate::library::Function);
    let changes: [(&str, Change); 11] = [
        ("no fragment", |e, _| e.fragment = None),
        ("the fragment's function", |e, v| {
            e.fragment = Some(StageSource::Function(FunctionDescriptor::new(v.clone())));
        }),
        ("constants", |e, _| {
            if let StageSource::Function(f) = &mut e.vertex {
                f.constants = has_tint(true);
            }
        }),
        ("depth", |e, _| {
            e.depth_format = Some(metal::PixelFormat::Depth32Float);
        }),
        ("a color format", |e, _| {
            e.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
                metal::PixelFormat::BGRA8Unorm,
            ));
        }),
        ("rasterization", |e, _| e.rasterization_enabled = false),
        ("indirect command buffers", |e, _| {
            e.support_indirect_command_buffers = true;
        }),
        ("a buffer's mutability", |e, _| {
            e.vertex_buffers.insert(0, metal::Mutability::Immutable);
        }),
        ("coverage 0.0", |e, _| {
            e.private.sample_coverage = 0.0;
        }),
        ("a private flag", |e, _| e.private.two_side_enabled = true),
        ("OpenGL mode", |e, _| e.private.open_gl_mode_enabled = true),
    ];
    let mut seen = vec![key];
    for (what, change) in changes {
        let mut e = d.clone();
        change(&mut e, &vertex);
        let k = dev.render_descriptor_key(&e);
        assert!(!seen.contains(&k), "{what} changes the key");
        seen.push(k);
    }
    // Coverage 0.0 and -0.0 are different descriptors.
    let mut e = d.clone();
    e.private.sample_coverage = -0.0;
    assert!(
        !seen.contains(&dev.render_descriptor_key(&e)),
        "coverage -0.0"
    );
}

/// A render pipeline requires `extendedDynamicState` exactly when its graphics state makes a state
/// of `VK_EXT_extended_dynamic_state` dynamic, which it does exactly when the device lists the
/// feature. A 1.2 device that lacks it gets a typed requirement from `required_enables`; one that
/// lists it through the extension is told to enable both; from 1.3 nothing is enabled.
#[test]
fn a_render_pipeline_requires_extended_dynamic_state_exactly_when_it_makes_one_dynamic() {
    use crate::device::requirements::{VkExtension, VkFeature};
    use crate::error::Capability;
    use crate::vk::DynamicState as D;
    const CORE: [D; 7] = [
        D::VIEWPORT,
        D::SCISSOR,
        D::DEPTH_BIAS,
        D::BLEND_CONSTANTS,
        D::STENCIL_COMPARE_MASK,
        D::STENCIL_WRITE_MASK,
        D::STENCIL_REFERENCE,
    ];
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = lib.function("stage_io_vertex_position").expect("vertex");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        vertex.clone(),
    )));
    d.rasterization_enabled = false;
    let compile = |desc: &DeviceDescription| {
        let dev = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
        let pipe = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .expect("render pipeline");
        let extended = pipe
            .graphics
            .dynamic_states
            .iter()
            .any(|s| !CORE.contains(s));
        let required = pipe
            .requirements
            .features
            .contains(&VkFeature::ExtendedDynamicState);
        assert_eq!(required, extended, "{:?}", pipe.graphics.dynamic_states);
        assert_eq!(extended, desc.has(VkFeature::ExtendedDynamicState));
        assert_eq!(extended, dev.profile().fields().dynamic_vertex_stride);
        l3(&pipe.stages[0].spirv, &pipe.requirements, &dev);
        pipe
    };
    // Vulkan 1.2 without the extension: static states, no requirement.
    let floor = description("lunarg-desktop-baseline-2023");
    assert!(!floor.has(VkFeature::ExtendedDynamicState));
    let plain = compile(&floor);
    assert!(plain
        .graphics
        .dynamic_states
        .iter()
        .all(|s| CORE.contains(s)));

    // Vulkan 1.2 with the extension and its feature: the states, the requirement, and both
    // enables; the same device without the feature enabled is told what it lacks.
    let mut extended = floor.clone();
    extended
        .extensions
        .insert(VkExtension::ExtExtendedDynamicState);
    extended.features.insert(VkFeature::ExtendedDynamicState);
    let pipe = compile(&extended);
    assert!(pipe.graphics.dynamic_states.contains(&D::CULL_MODE));
    let enables = extended
        .required_enables(&pipe.requirements)
        .expect("enables");
    assert!(enables.features.contains(&VkFeature::ExtendedDynamicState));
    assert!(enables
        .extensions
        .contains(&VkExtension::ExtExtendedDynamicState));
    assert_eq!(
        floor.required_enables(&pipe.requirements),
        Err(Capability::Feature(VkFeature::ExtendedDynamicState))
    );

    // From Vulkan 1.3 the states are core without a feature member: required, nothing to enable.
    for stem in ["lavapipe", "moltenvk-m3", "roadmap-2022"] {
        let desc = description(stem);
        assert!(desc.api_version >= crate::device::requirements::ApiVersion::V1_3);
        let pipe = compile(&desc);
        assert!(
            pipe.graphics.dynamic_states.contains(&D::CULL_MODE),
            "{stem}"
        );
        let enables = desc.required_enables(&pipe.requirements).expect(stem);
        assert!(!enables.features.contains(&VkFeature::ExtendedDynamicState));
        assert!(!enables
            .extensions
            .contains(&VkExtension::ExtExtendedDynamicState));
    }

    // A state no link makes has no guessed requirement.
    let mut unknown = pipe.graphics.clone();
    unknown.dynamic_states.push(D::RASTERIZER_DISCARD_ENABLE);
    assert!(matches!(
        unknown.dynamic_state_features(),
        Err(Error::Internal(e)) if e.kind == InternalKind::Invariant
    ));
}

/// An untessellated vertex pipeline makes its topology and primitive restart dynamic exactly where
/// the profile has both extended dynamic states, by their features and never by the device's
/// version: at Vulkan 1.2 through `VK_EXT_extended_dynamic_state` and
/// `VK_EXT_extended_dynamic_state2` with their features, which it then requires and is told to
/// enable; at 1.2 with either alone, neither is dynamic; from 1.3, where both are core without a
/// feature member, both are dynamic and nothing is enabled.
#[test]
fn a_vertex_pipeline_makes_its_input_assembly_dynamic_exactly_with_both_extended_dynamic_states() {
    use crate::device::requirements::{VkExtension, VkFeature};
    use crate::error::Capability;
    use crate::vk::DynamicState as D;
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = lib.function("stage_io_vertex_position").expect("vertex");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        vertex.clone(),
    )));
    d.rasterization_enabled = false;
    let compile = |desc: &DeviceDescription| {
        let dev = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
        dev.compile_render_pipeline(&d, &Budget::standard())
            .expect("render pipeline")
    };
    let assembly = |pipe: &crate::pipeline::RenderPipeline| {
        let states = &pipe.graphics.dynamic_states;
        let topology = states.contains(&D::PRIMITIVE_TOPOLOGY);
        assert_eq!(
            topology,
            states.contains(&D::PRIMITIVE_RESTART_ENABLE),
            "{states:?}"
        );
        assert_eq!(
            topology,
            pipe.requirements
                .features
                .contains(&VkFeature::ExtendedDynamicState2),
            "{states:?}"
        );
        topology
    };
    let floor = description("lunarg-desktop-baseline-2023");
    assert!(!floor.has(VkFeature::ExtendedDynamicState));
    assert!(!floor.has(VkFeature::ExtendedDynamicState2));
    assert!(!assembly(&compile(&floor)));
    let with = |extensions: &[(VkExtension, VkFeature)]| {
        let mut desc = floor.clone();
        for &(extension, feature) in extensions {
            desc.extensions.insert(extension);
            desc.features.insert(feature);
        }
        desc
    };
    let first = (
        VkExtension::ExtExtendedDynamicState,
        VkFeature::ExtendedDynamicState,
    );
    let second = (
        VkExtension::ExtExtendedDynamicState2,
        VkFeature::ExtendedDynamicState2,
    );
    // Either alone: the input assembly stays static.
    assert!(!assembly(&compile(&with(&[first]))));
    assert!(!assembly(&compile(&with(&[second]))));
    // Both, at Vulkan 1.2: dynamic, required, and both enabled.
    let both = with(&[first, second]);
    let pipe = compile(&both);
    assert!(assembly(&pipe));
    let enables = both.required_enables(&pipe.requirements).expect("enables");
    assert!(enables.features.contains(&VkFeature::ExtendedDynamicState2));
    assert!(enables
        .extensions
        .contains(&VkExtension::ExtExtendedDynamicState2));
    assert_eq!(
        with(&[first]).required_enables(&pipe.requirements),
        Err(Capability::Feature(VkFeature::ExtendedDynamicState2))
    );
    // From Vulkan 1.3: dynamic, nothing to enable.
    for stem in ["lavapipe", "moltenvk-m3", "roadmap-2022"] {
        let desc = description(stem);
        let pipe = compile(&desc);
        assert!(assembly(&pipe), "{stem}");
        let enables = desc.required_enables(&pipe.requirements).expect(stem);
        assert!(!enables.features.contains(&VkFeature::ExtendedDynamicState2));
        assert!(!enables
            .extensions
            .contains(&VkExtension::ExtExtendedDynamicState2));
    }
}

/// A render pipeline whose fragment function reads its color attachment orders the read exactly
/// where the device can. With rasterization-order attachment access it states that access and
/// requires `rasterizationOrderColorAttachmentAccess` (enabled with its extension). With only the
/// ordered pixel interlock it reads and writes the attachment as a storage image inside the
/// interlock: no input attachment, no color attachment state for it, the interlock, the
/// fragment-stage store, and the format's storage access required; a blending attachment's blend
/// the shader computes (`R32Float`'s the generic one, noted). Where the pipeline's state blocks
/// that form (alpha to coverage), the pipeline is refused with the blocker. With neither form it
/// reads the input attachment without rasterization-order access, which the consumer's barriers
/// order between draws alone, and its fragment stage records
/// `PrecisionNote::UnorderedFetchWithinDraw`, which neither ordered form records. A raster order
/// group on the color input (`fetch_ordered`) takes the interlock where the device has it, and
/// with neither form orders that read alone: no interlock, noted.
#[test]
fn a_framebuffer_fetch_pipeline_is_ordered_exactly_where_the_device_orders_it() {
    use crate::device::requirements::{FormatUsage, VkExtension, VkFeature};
    use crate::pipeline::{PrecisionNote, RenderPipeline};
    const ORDERED: VkFeature = VkFeature::RasterizationOrderColorAttachmentAccess;
    const INTERLOCK: VkFeature = VkFeature::FragmentShaderPixelInterlock;
    let (Some(io), Some(fetch)) = (fixture("stage_io"), fixture("fragment_fetch")) else {
        return;
    };
    let vertex = Library::from_air(&io)
        .expect("library")
        .function("stage_io_vertex_position")
        .expect("vertex");
    let fragment = Library::from_air(&fetch)
        .expect("library")
        .function("fetch_scalar")
        .expect("fragment");
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(vertex)));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(fragment)));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(metal::PixelFormat::R32Float));
    // Blended, adding the destination (a blend of one and zero, added, stores the source as no
    // blending does).
    let mut blended = d.clone();
    if let Some(c) = blended.color_attachments[0].as_mut() {
        c.blending_enabled = true;
        c.destination_rgb_blend_factor = metal::BlendFactor::One;
    }
    let compile = |desc: &DeviceDescription, policy: DevicePolicy, d: &RenderPipelineDescriptor| {
        let dev = Device::new(desc.clone(), policy).expect("device");
        let pipe = dev.compile_render_pipeline(d, &Budget::standard())?;
        for s in &pipe.stages {
            l3(&s.spirv, &pipe.requirements, &dev);
        }
        desc.required_enables(&pipe.requirements)
            .expect("the requirements are the device's");
        Ok::<RenderPipeline, Error>(pipe)
    };

    let unordered = |pipe: &RenderPipeline| {
        (pipe.stages.iter())
            .filter(|s| s.stage == vk::ShaderStage::FRAGMENT)
            .any(|s| {
                s.precision_notes
                    .contains(&PrecisionNote::UnorderedFetchWithinDraw)
            })
    };
    let lavapipe = description("lavapipe");
    assert!(lavapipe.has(ORDERED));
    {
        let pipe = compile(&lavapipe, DevicePolicy::default(), &d).expect("lavapipe");
        assert!(!unordered(&pipe));
        assert_eq!(pipe.graphics.input_attachments.len(), 1);
        assert!(pipe.graphics.storage_attachments.is_empty());
        assert!(pipe.graphics.rasterization_order_attachment_access);
        assert!(pipe.requirements.features.contains(&ORDERED));
        let enables = lavapipe
            .required_enables(&pipe.requirements)
            .expect("enables");
        assert!(enables.features.contains(&ORDERED));
        assert!(enables
            .extensions
            .contains(&VkExtension::ExtRasterizationOrderAttachmentAccess));
    }

    let mut interlocked = lavapipe.clone();
    interlocked.features.remove(&ORDERED);
    assert!(interlocked.has(INTERLOCK));
    {
        let pipe = compile(&interlocked, DevicePolicy::default(), &d).expect("interlocked");
        assert!(!unordered(&pipe));
        assert!(pipe.graphics.input_attachments.is_empty());
        assert!(pipe.graphics.color_attachments.iter().all(Option::is_none));
        let [s] = pipe.graphics.storage_attachments.as_slice() else {
            panic!("{:?}", pipe.graphics.storage_attachments);
        };
        // Declared without a format where the device grants that: lavapipe does.
        assert_eq!(
            (s.color_index, s.format, s.writes, s.formatless),
            (0, vk::Format::R32_SFLOAT, true, true)
        );
        // The capability, which the device resolves (`DeviceProfile::check`), not its feature.
        assert!(pipe.requirements.spirv_capabilities.contains(
            &crate::device::requirements::SpirvCapability::of(
                crate::spv::Capability::StorageImageReadWithoutFormat
            )
        ));
        assert!(!pipe
            .requirements
            .features
            .contains(&VkFeature::ShaderStorageImageReadWithoutFormat));
        assert!(!pipe.graphics.rasterization_order_attachment_access);
        for f in [INTERLOCK, VkFeature::FragmentStoresAndAtomics] {
            assert!(pipe.requirements.features.contains(&f), "{f:?}");
        }
        assert!(!pipe.requirements.features.contains(&ORDERED));
        assert!(pipe
            .requirements
            .formats
            .contains(&(vk::Format::R32_SFLOAT, FormatUsage::StorageImage)));
    }
    // The blend, which the shader computes: the generic one of `R32Float`, noted.
    {
        let pipe = compile(&interlocked, DevicePolicy::default(), &blended).expect("blended");
        let [s] = pipe.graphics.storage_attachments.as_slice() else {
            panic!("{:?}", pipe.graphics.storage_attachments);
        };
        assert_eq!((s.color_index, s.writes), (0, true));
        let fragment = (pipe.stages.iter())
            .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
            .expect("a fragment stage");
        assert!(fragment
            .precision_notes
            .contains(&PrecisionNote::UnmeasuredFetchBlend));
    }
    // Alpha to coverage, which can drop the fragment's samples, blocks the form.
    let mut covered = d.clone();
    covered.alpha_to_coverage = true;
    match compile(&interlocked, DevicePolicy::default(), &covered) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::FramebufferFetch, Needs::NotImplemented)
        ),
        other => panic!("{:?}", other.map(|_| ())),
    }

    let mut without = interlocked.clone();
    without.features.remove(&INTERLOCK);
    let pipe = compile(&without, DevicePolicy::default(), &d).expect("neither");
    assert!(unordered(&pipe));
    assert_eq!(pipe.graphics.input_attachments.len(), 1);
    assert!(pipe.graphics.storage_attachments.is_empty());
    assert!(!pipe.graphics.rasterization_order_attachment_access);
    for f in [ORDERED, INTERLOCK] {
        assert!(!pipe.requirements.features.contains(&f), "{f:?}");
    }
    let mut grouped = d.clone();
    grouped.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        Library::from_air(&fetch)
            .expect("library")
            .function("fetch_ordered")
            .expect("fragment"),
    )));
    let pipe = compile(&interlocked, DevicePolicy::default(), &grouped).expect("grouped");
    assert!(!unordered(&pipe));
    assert!(pipe.requirements.features.contains(&INTERLOCK));
    let pipe = compile(&without, DevicePolicy::default(), &grouped).expect("grouped, neither");
    assert!(unordered(&pipe));
    assert_eq!(pipe.graphics.input_attachments.len(), 1);
    for f in [ORDERED, INTERLOCK] {
        assert!(!pipe.requirements.features.contains(&f), "{f:?}");
    }
}

/// The SPIR-V `OpTypeImage`s of `words` as `(dim, arrayed, ms)` operand triples, and whether a
/// variable is decorated `BuiltIn SampleId`.
fn images_and_sample_id(words: &[u32]) -> (Vec<(u32, u32, u32)>, bool) {
    const TYPE_IMAGE: u32 = 25;
    const DECORATE: u32 = 71;
    const BUILT_IN: u32 = 11;
    const SAMPLE_ID: u32 = 18;
    let mut images = Vec::new();
    let mut sample_id = false;
    let mut at = 5;
    while at < words.len() {
        let (count, op) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
        let inst = &words[at..at + count.max(1)];
        match op {
            TYPE_IMAGE => images.push((inst[3], inst[5], inst[6])),
            DECORATE if inst.get(2..4) == Some(&[BUILT_IN, SAMPLE_ID]) => sample_id = true,
            _ => {}
        }
        at += count.max(1);
    }
    (images, sample_id)
}

/// The guard of the one storage write in `words` where a returned sample mask's bit gates it:
/// `Some(shift)` where the write's block is the true arm of a branch on
/// `((mask >> shift) & 1) != 0`, `shift` the shift's constant (`Some(0)` where no shift is left),
/// or `None` where the shift is not a constant (the invocation's `SampleId`); `None` where no
/// such branch guards the write.
fn mask_gate(words: &[u32]) -> Option<Option<u32>> {
    const IMAGE_WRITE: u32 = 99;
    const CONSTANT: u32 = 43;
    const I_NOT_EQUAL: u32 = 171;
    const SHIFT_RIGHT_LOGICAL: u32 = 194;
    const BITWISE_AND: u32 = 199;
    const LABEL: u32 = 248;
    const BRANCH_CONDITIONAL: u32 = 250;
    let mut defs = std::collections::HashMap::new();
    let (mut branches, mut writes, mut label) = (Vec::new(), Vec::new(), 0);
    let mut at = 5;
    while at < words.len() {
        let (count, op) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
        let inst = &words[at..at + count.max(1)];
        match op {
            CONSTANT | I_NOT_EQUAL | SHIFT_RIGHT_LOGICAL | BITWISE_AND => {
                defs.insert(inst[2], (op, &inst[3..]));
            }
            LABEL => label = inst[1],
            BRANCH_CONDITIONAL => branches.push((inst[1], inst[2])),
            IMAGE_WRITE => writes.push(label),
            _ => {}
        }
        at += count.max(1);
    }
    let def = |id: &u32, want: u32| match defs.get(id) {
        Some(&(op, operands)) if op == want => Some(operands),
        _ => None,
    };
    let constant = |id: &u32| def(id, CONSTANT).and_then(|v| v.first().copied());
    let [write] = writes.as_slice() else {
        return None;
    };
    let (condition, _) = branches.iter().find(|(_, arm)| arm == write)?;
    let [bit, zero] = def(condition, I_NOT_EQUAL)? else {
        return None;
    };
    let [shifted, one] = def(bit, BITWISE_AND)? else {
        return None;
    };
    if (constant(zero), constant(one)) != (Some(0), Some(1)) {
        return None;
    }
    Some(match def(shifted, SHIFT_RIGHT_LOGICAL) {
        Some([_, shift]) => constant(shift),
        _ => Some(0),
    })
}

/// Metal runs a function that reads a color attachment of a multisampled pass once per covered
/// sample, reading its own sample. Compiled for a pipeline of 4 samples, the fragment reads each
/// color at its `SampleId` (which runs it once per covered sample under sample shading): on
/// lavapipe through a multisampled `SubpassData` image under rasterization-order access, on a
/// device that orders fetch only through the pixel interlock through a multisampled storage image
/// of the pipeline's samples (`shaderStorageImageMultisample`, `storageImageSampleCounts`), and,
/// where that device has no multisampled storage image (ANV, MoltenVK), through planes: a
/// single-sample 2D-array storage image of one layer per sample, which needs neither. Where the
/// device lacks what the form needs, the refusal names it; a returned sample mask is exact per
/// sample (the storage write is gated on the sample's bit) and at one sample (gated on bit 0, as
/// Metal stores); `[[sample_mask]]` in is the pixel's coverage, which no device gives a per-sample
/// invocation. One sample compiles as before: no `SampleId`, a single-sample image.
#[test]
fn a_multisampled_framebuffer_fetch_reads_each_samples_own_color() {
    use crate::device::requirements::{VkFeature, VkProperty};
    use crate::error::{Capability, Unsupported};
    use crate::pipeline::{RenderPipeline, StorageAttachmentView};
    const ORDERED: VkFeature = VkFeature::RasterizationOrderColorAttachmentAccess;
    const SUBPASS_DATA: u32 = 6;
    const D2: u32 = 1;
    let (Some(io), Some(blend), Some(ms)) = (
        fixture("stage_io"),
        fixture("exf_fetch_blend"),
        fixture("msf_fetch_ms"),
    ) else {
        return;
    };
    let vertex = Library::from_air(&io)
        .expect("library")
        .function("stage_io_vertex_position")
        .expect("vertex");
    let blend = Library::from_air(&blend).expect("library");
    let ms = Library::from_air(&ms).expect("library");
    let descriptor = |lib: &Library, entry: &str, samples: u32| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            vertex.clone(),
        )));
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            lib.function(entry).expect("fragment"),
        )));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
            metal::PixelFormat::RGBA32Float,
        ));
        d.raster_sample_count = samples;
        d
    };
    let compile = |desc: &DeviceDescription, d: &RenderPipelineDescriptor| {
        let dev = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
        let pipe = dev.compile_render_pipeline(d, &Budget::standard())?;
        for s in &pipe.stages {
            l3(&s.spirv, &pipe.requirements, &dev);
        }
        desc.required_enables(&pipe.requirements)
            .expect("the requirements are the device's");
        Ok::<RenderPipeline, Error>(pipe)
    };
    let fragment_words = |pipe: &RenderPipeline| pipe.stages[1].spirv.to_vec();
    let refusal = |r: Result<RenderPipeline, Error>| match r {
        Err(Error::Unsupported(u)) => (u.feature, u.needs),
        other => panic!("{:?}", other.map(|_| ())),
    };

    // Raster order: the input attachment's own sample.
    let lavapipe = description("lavapipe");
    let pipe = compile(&lavapipe, &descriptor(&blend, "exf_fetch_blend", 4)).expect("lavapipe");
    assert_eq!(pipe.graphics.input_attachments.len(), 1);
    assert!(pipe.graphics.storage_attachments.is_empty());
    assert!(pipe.graphics.rasterization_order_attachment_access);
    assert_eq!(
        pipe.graphics.rasterization_samples,
        vk::SampleCountFlags::COUNT_4
    );
    for f in [ORDERED, VkFeature::SampleRateShading] {
        assert!(pipe.requirements.features.contains(&f), "{f:?}");
    }
    assert_eq!(
        images_and_sample_id(&fragment_words(&pipe)),
        (vec![(SUBPASS_DATA, 0, 1)], true)
    );
    let one = compile(&lavapipe, &descriptor(&blend, "exf_fetch_blend", 1)).expect("one");
    assert_eq!(
        images_and_sample_id(&fragment_words(&one)),
        (vec![(SUBPASS_DATA, 0, 0)], false)
    );
    // The returned mask and alpha to coverage stay the fixed function's.
    compile(&lavapipe, &descriptor(&ms, "msf_fetch_mask_out", 4)).expect("mask out");

    // The pixel interlock: a multisampled storage image of the pipeline's samples.
    let mut interlocked = lavapipe.clone();
    interlocked.features.remove(&ORDERED);
    let pipe = compile(&interlocked, &descriptor(&blend, "exf_fetch_blend", 4)).expect("storage");
    assert!(pipe.graphics.input_attachments.is_empty());
    let [s] = pipe.graphics.storage_attachments.as_slice() else {
        panic!("{:?}", pipe.graphics.storage_attachments);
    };
    assert_eq!(
        (s.color_index, s.format, s.writes, s.view),
        (
            0,
            vk::Format::R32G32B32A32_SFLOAT,
            true,
            StorageAttachmentView::Attachment
        )
    );
    for f in [
        VkFeature::FragmentShaderPixelInterlock,
        VkFeature::ShaderStorageImageMultisample,
        VkFeature::SampleRateShading,
    ] {
        assert!(pipe.requirements.features.contains(&f), "{f:?}");
    }
    assert!(pipe
        .requirements
        .properties
        .contains(&VkProperty::StorageImageSampleCount(4)));
    assert_eq!(
        images_and_sample_id(&fragment_words(&pipe)),
        (vec![(D2, 0, 1)], true)
    );
    let one = compile(&interlocked, &descriptor(&blend, "exf_fetch_blend", 1)).expect("one");
    assert_eq!(
        images_and_sample_id(&fragment_words(&one)),
        (vec![(D2, 0, 0)], false)
    );
    assert!(!one
        .requirements
        .properties
        .contains(&VkProperty::StorageImageSampleCount(1)));
    // A returned mask (`msf_fetch_mask_out` returns 0x5) gates the write on the invocation's own
    // sample's bit; at one sample, where no `SampleId` is read, on bit 0, as Metal stores no color
    // without it (before, refused `NotImplemented(FramebufferFetch)`).
    let four = compile(&interlocked, &descriptor(&ms, "msf_fetch_mask_out", 4)).expect("mask out");
    assert_eq!(
        images_and_sample_id(&fragment_words(&four)),
        (vec![(D2, 0, 1)], true)
    );
    assert_eq!(mask_gate(&fragment_words(&four)), Some(None));
    let one = compile(&interlocked, &descriptor(&ms, "msf_fetch_mask_out", 1)).expect("one mask");
    let [s] = one.graphics.storage_attachments.as_slice() else {
        panic!("{:?}", one.graphics.storage_attachments);
    };
    assert_eq!(
        (s.color_index, s.writes, s.view),
        (0, true, StorageAttachmentView::Attachment)
    );
    assert_eq!(
        images_and_sample_id(&fragment_words(&one)),
        (vec![(D2, 0, 0)], false)
    );
    assert_eq!(mask_gate(&fragment_words(&one)), Some(Some(0)));
    // Without multisampled storage images: planes, one single-sample layer per sample, which
    // need neither the feature nor a sample count of storage images. ANV and MoltenVK.
    let mut no_storage_ms = interlocked.clone();
    no_storage_ms
        .features
        .remove(&VkFeature::ShaderStorageImageMultisample);
    no_storage_ms.limits.storage_image_sample_counts = vk::SampleCountFlags::COUNT_1;
    for desc in [
        &no_storage_ms,
        &description("moltenvk-m3"),
        &description("intel-arl-anv"),
    ] {
        let pipe = compile(desc, &descriptor(&blend, "exf_fetch_blend", 4)).expect("planes");
        let [s] = pipe.graphics.storage_attachments.as_slice() else {
            panic!("{:?}", pipe.graphics.storage_attachments);
        };
        assert_eq!(
            (s.color_index, s.format, s.writes, s.view),
            (
                0,
                vk::Format::R32G32B32A32_SFLOAT,
                true,
                StorageAttachmentView::SamplePlanes { layers: 4 }
            )
        );
        for f in [
            VkFeature::FragmentShaderPixelInterlock,
            VkFeature::SampleRateShading,
        ] {
            assert!(pipe.requirements.features.contains(&f), "{f:?}");
        }
        assert!(!pipe
            .requirements
            .features
            .contains(&VkFeature::ShaderStorageImageMultisample));
        assert!(!pipe
            .requirements
            .properties
            .iter()
            .any(|p| matches!(p, VkProperty::StorageImageSampleCount(_))));
        assert_eq!(
            images_and_sample_id(&fragment_words(&pipe)),
            (vec![(D2, 1, 0)], true)
        );
        compile(desc, &descriptor(&ms, "msf_fetch_mask_out", 4)).expect("mask out");
        let one = compile(desc, &descriptor(&blend, "exf_fetch_blend", 1)).expect("one");
        assert_eq!(
            one.graphics.storage_attachments[0].view,
            StorageAttachmentView::Attachment
        );
        assert_eq!(
            images_and_sample_id(&fragment_words(&one)),
            (vec![(D2, 0, 0)], false)
        );
    }
    let mut no_count = interlocked.clone();
    no_count.limits.storage_image_sample_counts = vk::SampleCountFlags::COUNT_1;
    assert_eq!(
        compile(&no_count, &descriptor(&blend, "exf_fetch_blend", 4)).map(|_| ()),
        Err(Error::Unsupported(Unsupported::needs(
            Feature::FramebufferFetch,
            Capability::Property(VkProperty::StorageImageSampleCount(4))
        )))
    );
    // Without sample-rate shading no form reads a sample.
    for desc in [&lavapipe, &interlocked, &no_storage_ms] {
        let mut no_rate = desc.clone();
        no_rate.features.remove(&VkFeature::SampleRateShading);
        let (feature, needs) =
            refusal(compile(&no_rate, &descriptor(&blend, "exf_fetch_blend", 4)));
        assert_eq!(feature, Feature::FramebufferFetch);
        let Needs::AnyOf(alternatives) = needs else {
            panic!("{needs:?}");
        };
        assert!(alternatives
            .iter()
            .flatten()
            .any(|c| *c == Capability::Feature(VkFeature::SampleRateShading)));
    }
    // `[[sample_mask]]` in beside a fetch reads the pixel's coverage (measured): no device.
    for desc in [&lavapipe, &interlocked, &no_storage_ms] {
        assert_eq!(
            refusal(compile(desc, &descriptor(&ms, "msf_fetch_mask_in", 4))),
            (Feature::PixelCoverageAtSampleRate, Needs::NotRepresentable)
        );
    }
}

/// A render pipeline requires `independentBlend` exactly when the elements of its `pAttachments`
/// differ, and `dualSrcBlend` exactly when a blend reads the second output. `exf2_skip_one` writes
/// color attachments 0 and 2, not 1: on `lavapipe` attachment 1 is masked and the feature required
/// (and enabled); on `vk12-floor`, without the feature, every element is identical, an absent
/// attachment 1's included, and a written attachment that blends differently is refused typed. A
/// blend of one and zero, added, of an attachment whose channels convert by a rule other than the
/// device's (`StageIo::color_writes`) takes no Vulkan blend, so it differs from no other element.
#[test]
fn a_render_pipeline_requires_the_blend_features_its_attachments_need() {
    use crate::device::requirements::VkFeature;
    use crate::error::{Capability, Unsupported};
    use crate::pipeline::RenderPipeline;
    const INDEPENDENT: VkFeature = VkFeature::IndependentBlend;
    let (Some(io), Some(skip)) = (fixture("stage_io"), fixture("exf2_skip_one")) else {
        return;
    };
    let io = Library::from_air(&io).expect("library");
    let vertex = io.function("stage_io_vertex_position").expect("vertex");
    let fragment = Library::from_air(&skip)
        .expect("library")
        .function("exf2_skip_one")
        .expect("fragment");
    let descriptor = |fragment: &crate::library::Function, middle: bool| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            vertex.clone(),
        )));
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            fragment.clone(),
        )));
        let rgba8 = ColorAttachmentDescriptor::new(metal::PixelFormat::RGBA8Unorm);
        d.color_attachments[0] = Some(rgba8);
        d.color_attachments[1] = middle.then_some(rgba8);
        d.color_attachments[2] = Some(ColorAttachmentDescriptor::new(
            metal::PixelFormat::RGBA32Uint,
        ));
        d
    };
    let compile = |stem: &str, d: &RenderPipelineDescriptor| {
        let desc = description(stem);
        let dev = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
        let pipe = dev.compile_render_pipeline(d, &Budget::standard())?;
        let enables = desc
            .required_enables(&pipe.requirements)
            .expect("the requirements are the device's");
        assert_eq!(
            enables.features.contains(&INDEPENDENT),
            pipe.requirements.features.contains(&INDEPENDENT)
        );
        for s in &pipe.stages {
            l3(&s.spirv, &pipe.requirements, &dev);
        }
        Ok::<RenderPipeline, Error>(pipe)
    };
    let masks = |pipe: &RenderPipeline| -> Vec<vk::ColorComponentFlags> {
        pipe.graphics
            .color_blend_attachments()
            .into_iter()
            .map(|(mask, _)| mask)
            .collect()
    };
    let all = vk::ColorComponentFlags::R
        | vk::ColorComponentFlags::G
        | vk::ColorComponentFlags::B
        | vk::ColorComponentFlags::A;
    let none = vk::ColorComponentFlags::empty();
    assert!(description("lavapipe").has(INDEPENDENT));
    assert!(!description("vk12-floor").has(INDEPENDENT));

    // The unwritten attachment.
    let d = descriptor(&fragment, true);
    let pipe = compile("lavapipe", &d).expect("lavapipe");
    assert_eq!(masks(&pipe), [all, none, all]);
    assert!(pipe.requirements.features.contains(&INDEPENDENT));
    let pipe = compile("vk12-floor", &d).expect("vk12-floor");
    assert_eq!(masks(&pipe), [all, all, all]);
    assert!(!pipe.requirements.features.contains(&INDEPENDENT));
    // An absent attachment 1: its element is the others'.
    let d = descriptor(&fragment, false);
    for stem in ["lavapipe", "vk12-floor"] {
        let pipe = compile(stem, &d).expect(stem);
        assert!(pipe.graphics.color_attachments[1].is_none());
        assert_eq!(masks(&pipe), [all, all, all], "{stem}");
        assert!(!pipe.requirements.features.contains(&INDEPENDENT), "{stem}");
    }
    // Attachment 0 blends with (one, zero, add), which lavapipe's `RGBA8Unorm` write rounds in
    // the function: no Vulkan blend, so every element is the same.
    let mut d = descriptor(&fragment, false);
    if let Some(a) = d.color_attachments[0].as_mut() {
        a.blending_enabled = true;
    }
    let pipe = compile("lavapipe", &d).expect("lavapipe");
    let blend = |pipe: &RenderPipeline, at: usize| {
        pipe.graphics.color_attachments[at]
            .as_ref()
            .expect("attachment")
            .blend
    };
    assert_eq!(blend(&pipe, 0), None);
    assert!(!pipe.requirements.features.contains(&INDEPENDENT));
    // Attachment 0 blends over the destination, attachment 2 does not.
    if let Some(a) = d.color_attachments[0].as_mut() {
        a.destination_rgb_blend_factor = metal::BlendFactor::OneMinusSourceAlpha;
    }
    let pipe = compile("lavapipe", &d).expect("lavapipe");
    assert!(blend(&pipe, 0).is_some());
    assert_eq!(blend(&pipe, 2), None);
    assert!(pipe.requirements.features.contains(&INDEPENDENT));
    // Refused at attachment 2, whose state differs from attachment 0's.
    let at_2 = crate::error::Location::render_pipeline()
        .with_attachment(crate::error::Attachment::Color(2));
    assert_eq!(
        compile("vk12-floor", &d).unwrap_err(),
        Error::Unsupported(
            Unsupported::needs(
                Feature::IndependentBlending,
                Capability::Feature(INDEPENDENT)
            )
            .at(at_2)
        )
    );

    // Dual-source blending: required exactly when a factor reads the second output.
    let dual = io
        .function("stage_io_fragment_dual_source")
        .expect("fragment");
    for (factor, required) in [
        (metal::BlendFactor::Source1Alpha, true),
        (metal::BlendFactor::SourceAlpha, false),
    ] {
        let mut d = descriptor(&dual, false);
        d.color_attachments[2] = None;
        if let Some(a) = d.color_attachments[0].as_mut() {
            a.blending_enabled = true;
            a.source_rgb_blend_factor = factor;
        }
        let pipe = compile("lavapipe", &d).expect("lavapipe");
        assert_eq!(
            pipe.requirements
                .features
                .contains(&VkFeature::DualSrcBlend),
            required,
            "{factor:?}"
        );
    }
}

/// A `Function` vertex stage's key includes the vertex-fetch projection over the attributes it
/// reads, and only those: an unread descriptor attribute does not change it, a missing one is an
/// invalid descriptor.
#[test]
fn a_vertex_key_projects_the_attributes_it_reads() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let vertex = Library::from_air(&bc)
        .expect("library")
        .function("stage_io_vertex")
        .expect("vertex");
    let fd = FunctionDescriptor::new(vertex);
    // The fixture declares float3 / half2 at 0 and 1, ushort2 / uchar4 at 4 and 7; the
    // descriptor gives each attribute a format of its declared class.
    use metal::VertexFormat as V;
    let read = [
        (0u32, V::Float3),
        (1, V::Half2),
        (4, V::UShort2),
        (7, V::UChar4),
    ];
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(fd.clone()));
    d.rasterization_enabled = false;
    d.vertex_descriptor = Some(vertex_descriptor(&read));
    let keys = dev
        .render_pipeline_key(&d, &Budget::standard())
        .expect("keys");
    let expected = crate::pipeline::link::vertex::vertex_fetch_projection(
        d.vertex_descriptor.as_ref(),
        &{
            use crate::pipeline::stage_io::IoNumeric::{Float, Uint};
            [(0, Float), (1, Float), (4, Uint), (7, Uint)].into()
        },
        dev.profile(),
    )
    .expect("projection");
    let mut inp = LoweringInputs::new();
    inp.vertex_fetch = Some(expected);
    assert_eq!(keys.vertex, dev.function_compile_key(&fd, &inp));
    assert_ne!(
        keys.vertex,
        dev.function_compile_key(&fd, &LoweringInputs::new())
    );

    let mut wider = d.clone();
    let mut wide = read.to_vec();
    wide.push((9, V::Float4));
    wider.vertex_descriptor = Some(vertex_descriptor(&wide));
    let wider_keys = dev
        .render_pipeline_key(&wider, &Budget::standard())
        .expect("keys");
    assert_eq!(wider_keys.vertex, keys.vertex);
    assert_ne!(wider_keys.pipeline, keys.pipeline);

    let mut missing = d.clone();
    missing.vertex_descriptor = Some(vertex_descriptor(&read[..3]));
    assert_eq!(
        dev.render_pipeline_key(&missing, &Budget::standard()),
        Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
            attribute: 7
        }))
    );

    // A float format feeding the `uint` attribute 4 is a class mismatch the projection refuses
    // with a typed StageIo refusal, not an invalid descriptor.
    let mut mismatched = d.clone();
    let mut classes = read.to_vec();
    classes[2].1 = V::Float2;
    mismatched.vertex_descriptor = Some(vertex_descriptor(&classes));
    match dev.render_pipeline_key(&mismatched, &Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(
                u.feature,
                Feature::Lowering(crate::error::LowerFamily::StageIo)
            );
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("{other:?}"),
    }
}

/// A vertex function and a fragment function that each read a word-typed constant global through
/// a byte pointer (punned), so each keeps its global in its own constant blob: bytes 1-4 for the
/// vertex stage, 5-12 for the fragment stage.
const STAGE_BLOBS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@vt = internal addrspace(2) constant [1 x i32] [i32 67305985], align 4
@ft = internal addrspace(2) constant [2 x i32] [i32 134678021, i32 202050057], align 4

define <4 x float> @blob_vertex(i32 %vid) {
entry:
  %k = and i32 %vid, 3
  %b = getelementptr inbounds i8, ptr addrspace(2) @vt, i32 %k
  %byte = load i8, ptr addrspace(2) %b, align 1
  %z = zext i8 %byte to i32
  %f = uitofp i32 %z to float
  %v = insertelement <4 x float> <float poison, float 0.0, float 0.0, float 1.0>, float %f, i64 0
  ret <4 x float> %v
}

define <4 x float> @blob_fragment(<4 x float> %pos) {
entry:
  %x = extractelement <4 x float> %pos, i64 0
  %xi = fptoui float %x to i32
  %k = and i32 %xi, 7
  %b = getelementptr inbounds i8, ptr addrspace(2) @ft, i32 %k
  %byte = load i8, ptr addrspace(2) %b, align 1
  %z = zext i8 %byte to i32
  %f = uitofp i32 %z to float
  %v = insertelement <4 x float> <float poison, float 0.0, float 0.0, float 1.0>, float %f, i64 0
  ret <4 x float> %v
}

!air.vertex = !{!0}
!air.fragment = !{!5}
!0 = !{ptr @blob_vertex, !1, !3}
!1 = !{!2}
!2 = !{!"air.position", !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!5 = !{ptr @blob_fragment, !6, !8}
!6 = !{!7}
!7 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!8 = !{!9}
!9 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
"#;

/// `text` assembled by `llvm-as`, or `None` (skipped visibly) without it. The scratch directory
/// is removed before returning.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!(
        "m2v-compile-tests-{}-{name}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join(format!("{name}.ll"));
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// A render pipeline whose vertex and fragment stages both carry constant data compiles (before,
/// one blob per pipeline refused it), each stage module carries its own blob, the layout binds one
/// `ConstantBlob` per stage, and the pipeline round-trips through its cache encoding.
#[test]
fn each_stage_carries_its_own_constant_blob() {
    use crate::cache::PipelineArtifact;
    use crate::pipeline::artifact::RenderPipeline;
    use crate::pipeline::layout::BindingSource;
    use crate::pipeline::sym::{Stage, SyntheticBinding};

    let Some(bc) = assemble_text("stage_blobs", STAGE_BLOBS) else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = lib.function("blob_vertex").expect("vertex");
    let fragment = lib.function("blob_fragment").expect("fragment");
    for stem in [
        "moltenvk-m3",
        "lunarg-desktop-baseline-2023",
        "lavapipe",
        "vk12-floor",
    ] {
        let dev = device(stem);
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            vertex.clone(),
        )));
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            fragment.clone(),
        )));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
            metal::PixelFormat::RGBA8Unorm,
        ));
        let pipe = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .expect(stem);
        assert_eq!(pipe.stages.len(), 2, "{stem}");
        assert_eq!(
            pipe.stages[0].constant_blob.as_deref(),
            Some(&[1u8, 2, 3, 4][..]),
            "{stem}: vertex blob"
        );
        assert_eq!(
            pipe.stages[1].constant_blob.as_deref(),
            Some(&[5u8, 6, 7, 8, 9, 10, 11, 12][..]),
            "{stem}: fragment blob"
        );
        let mut blob_stages: Vec<Stage> = pipe
            .layout
            .sets
            .iter()
            .flat_map(|s| &s.bindings)
            .filter_map(|b| match b.source {
                BindingSource::Synthetic {
                    stage,
                    binding: SyntheticBinding::ConstantBlob,
                } => Some(stage),
                _ => None,
            })
            .collect();
        blob_stages.sort();
        let mut expected = vec![Stage::Vertex, Stage::Fragment];
        expected.sort();
        assert_eq!(blob_stages, expected, "{stem}");
        for s in &pipe.stages {
            l3(&s.spirv, &pipe.requirements, &dev);
        }
        let back = RenderPipeline::from_bytes(&pipe.to_bytes(), &dev).expect("decode");
        assert_eq!(back, pipe, "{stem}");
    }
}

/// The linked functions a module's references reach are decoded once per module (here three
/// functions of one module and one of another are two decodes), a value for a constant the module
/// does not declare is ignored, and a module that does not decode is recorded for canonicalization,
/// which refuses it only at a surviving call.
#[test]
fn linked_functions_decode_once_per_module_and_defer_their_errors() {
    use super::{linked_modules, LinkedSource};
    let (Some(caller), Some(functions)) = (
        fixture("linked_visible_caller"),
        fixture("linked_visible_functions"),
    ) else {
        return;
    };
    let m = &mut Meter::new(&Budget::default());
    let abi = crate::air::decode(&caller, m).expect("decode").abi;
    let none = FunctionConstantValues::defaults();
    let named = FunctionConstantValues::defaults().with_name(
        "undeclared",
        FunctionConstantValue::new(metal::DataType::UInt, 1u32.to_le_bytes()),
    );
    let garbage = [0u8; 16];
    let linked = [
        LinkedSource::new("linked_scale", &functions, &none),
        LinkedSource::new("linked_blend", &functions, &named),
        LinkedSource::new("linked_sample", &functions, &none),
        LinkedSource::new("linked_offset", &caller, &none),
        // The same function listed again is one function.
        LinkedSource::new("linked_scale", &functions, &none),
        // Nothing references it: never decoded.
        LinkedSource::new("unreferenced", &garbage, &none),
    ];
    let modules = linked_modules(&abi, &linked, m).expect("linked modules");
    let names = |i: usize| {
        modules[i]
            .functions
            .iter()
            .map(|f| String::from_utf8_lossy(&f.name).into_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(modules.len(), 2);
    assert_eq!(names(0), ["linked_scale", "linked_blend", "linked_sample"]);
    assert_eq!(names(1), ["linked_offset"]);
    assert!(modules.iter().all(|l| l.code.is_ok()));
    // `undeclared` names no constant of the module: ignored, as Metal ignores it.
    assert_eq!(modules[0].functions[1].constants, Ok(Default::default()));
    // A referenced function whose module does not decode is recorded too.
    let broken = [LinkedSource::new("linked_scale", &garbage, &none)];
    let modules = linked_modules(&abi, &broken, m).expect("linked modules");
    assert_eq!(modules.len(), 1);
    assert!(matches!(
        modules[0].code,
        Err(crate::air::error::Error::InvalidInput(_))
    ));
}

/// On a portability-subset device a render pipeline's requirements name the subset's features it
/// relies on: `constantAlphaColorBlendFactors` for a constant-alpha RGB blend factor and
/// `vertexAttributeAccessBeyondStride` for a fixed-function attribute ending past its stride.
/// The Khronos reader takes them from MoltenVK's capture (which withholds `samplerMipLodBias`);
/// a device without the subset neither lists nor needs them. Where the subset withholds one, the
/// requirements never name it (the blend under an empty mask is dropped, a static stride's
/// attribute pulled, a per-bind stride left to the binder), so they stay within the profile.
#[test]
fn a_render_pipeline_requires_the_portability_features_it_uses() {
    use crate::device::requirements::{VkExtension, VkFeature};
    let Some(io) = fixture("stage_io") else {
        return;
    };
    let io = Library::from_air(&io).expect("library");
    let alpha = VkFeature::ConstantAlphaColorBlendFactors;
    let beyond = VkFeature::VertexAttributeAccessBeyondStride;
    let portability = [
        alpha,
        beyond,
        VkFeature::SamplerMipLodBias,
        VkFeature::MutableComparisonSamplers,
        VkFeature::ShaderSampleRateInterpolationFunctions,
    ];
    let mvk = description("moltenvk-m3");
    assert!(mvk.extensions.contains(&VkExtension::KhrPortabilitySubset));
    assert_eq!(
        portability.map(|f| mvk.features.contains(&f)),
        [true, true, false, true, true]
    );
    let lavapipe = description("lavapipe");
    assert!(portability.iter().all(|f| !lavapipe.features.contains(f)));
    let without = |f: VkFeature| {
        let mut d = mvk.clone();
        d.features.remove(&f);
        d
    };
    // Every pipeline that compiles requires only what its device enabled.
    let compile = |desc: &DeviceDescription, d: &RenderPipelineDescriptor| {
        let dev = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
        dev.compile_render_pipeline(d, &Budget::standard())
            .map(|p| {
                assert_eq!(dev.profile().check(&p.requirements), Ok(()));
                (p.requirements.features, p.vertex_input)
            })
    };

    let position = io.function("stage_io_vertex_position").expect("vertex");
    let mut blended = ColorAttachmentDescriptor::new(metal::PixelFormat::RGBA8Unorm);
    blended.blending_enabled = true;
    blended.source_rgb_blend_factor = metal::BlendFactor::BlendAlpha;
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(position)));
    d.color_attachments[0] = Some(blended);
    let (features, _) = compile(&mvk, &d).expect("moltenvk-m3");
    assert!(features.contains(&alpha));
    let (features, _) = compile(&lavapipe, &d).expect("lavapipe");
    assert!(!features.contains(&alpha));
    // Without a fragment function the attachment's mask is empty, so where the subset withholds
    // the factor the blend is dropped (link refuses it on a written attachment).
    let (features, _) = compile(&without(alpha), &d).expect("withheld");
    assert!(!features.contains(&alpha));

    // The fixture's float3 / half2 at 0 and 1, ushort2 / uchar4 at 4 and 7: the float3 takes
    // bytes [8, 20) of a 16-byte element.
    use metal::VertexFormat as V;
    let attributes = io.function("stage_io_vertex").expect("vertex");
    let mut vd = vertex_descriptor(&[(1, V::Half2), (4, V::UShort2), (7, V::UChar4)]);
    vd.layouts
        .insert(1, VertexBufferLayoutDescriptor::new(Some(16)));
    vd.attributes
        .insert(0, VertexAttributeDescriptor::new(V::Float3, 8, 1));
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(attributes)));
    d.rasterization_enabled = false;
    d.vertex_descriptor = Some(vd);
    let fixed = |vi: &crate::pipeline::VertexInputState| vi.pulled.iter().all(|p| p.attribute != 0);
    let (features, vi) = compile(&mvk, &d).expect("moltenvk-m3");
    assert!(features.contains(&beyond) && fixed(&vi));
    let (features, vi) = compile(&lavapipe, &d).expect("lavapipe");
    assert!(!features.contains(&beyond) && fixed(&vi));
    let (features, vi) = compile(&without(beyond), &d).expect("withheld");
    assert!(!features.contains(&beyond) && !fixed(&vi));

    // A per-bind stride is the bind's: the pipeline relies on the feature where the device has
    // it, and where the subset withholds it the binder pulls a stride below the attribute's end,
    // so the requirements do not name it.
    let mut per_bind = d.clone();
    if let Some(vd) = per_bind.vertex_descriptor.as_mut() {
        vd.layouts
            .insert(1, VertexBufferLayoutDescriptor::new(None));
    }
    let (features, vi) = compile(&mvk, &per_bind).expect("moltenvk-m3");
    assert!(features.contains(&beyond) && fixed(&vi));
    let (features, vi) = compile(&without(beyond), &per_bind).expect("withheld");
    assert!(!features.contains(&beyond) && fixed(&vi));
    let (features, _) = compile(&lavapipe, &per_bind).expect("lavapipe");
    assert!(!features.contains(&beyond));
}

/// A multisampled render pipeline's requirements name the framebuffer sample counts its
/// attachments need, which the device it compiled for provides.
#[test]
fn a_render_pipeline_requires_its_framebuffer_sample_counts() {
    use crate::device::requirements::VkProperty;
    let Some(io) = fixture("stage_io") else {
        return;
    };
    let io = Library::from_air(&io).expect("library");
    let vertex = io.function("stage_io_vertex_position").expect("vertex");
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(vertex)));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA8Unorm,
    ));
    d.depth_format = Some(metal::PixelFormat::Depth32Float);
    for (count, want) in [
        (1, vec![]),
        (
            4,
            vec![
                VkProperty::FramebufferColorSampleCount(4),
                VkProperty::FramebufferDepthSampleCount(4),
            ],
        ),
    ] {
        d.raster_sample_count = count;
        let dev = Device::new(description("lavapipe"), DevicePolicy::default()).expect("device");
        let pipe = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .expect("lavapipe");
        let stated: Vec<VkProperty> = pipe
            .requirements
            .properties
            .iter()
            .copied()
            .filter(|p| {
                matches!(
                    p,
                    VkProperty::FramebufferColorSampleCount(_)
                        | VkProperty::FramebufferDepthSampleCount(_)
                )
            })
            .collect();
        assert_eq!(stated, want, "{count} samples");
        assert_eq!(dev.profile().check(&pipe.requirements), Ok(()));
    }
}

/// The interface states the texture slots a fragment function reads only at its own pixel
/// (`spirv::requirements::own_pixel`): owned fixture `tob_renders`'s `tob_read`
/// (`t.read(uint2(in.position.xy))`) on every profile, whether the device's robustness returns
/// Metal's out-of-bounds texel or the shader clamps the read (both kinds of profile are among
/// them); not `tob_sample` (a sample at the pixel's center), `tob_read_buffer` (a texture
/// buffer's texel by arithmetic), nor `tob_solid`, which reads no texture.
#[test]
fn the_interface_states_the_textures_read_at_the_fragments_own_pixel() {
    let Some(bc) = fixture("tob_renders") else {
        return;
    };
    let mut checks = std::collections::BTreeSet::new();
    for stem in [
        "intel-irisxe-anv",
        "nvidia-rtx5080-laptop",
        "amd-780m-radv",
        "lavapipe",
        "moltenvk-m3",
        "vk12-floor",
    ] {
        let p = crate::pipeline::testing::profile(stem, DevicePolicy::default()).expect("profile");
        checks.insert(crate::memory::contain::device_reads_texels(&p, false));
        for (entry, own) in [
            ("tob_read", &[0u32][..]),
            ("tob_sample", &[]),
            ("tob_read_buffer", &[]),
            ("tob_solid", &[]),
        ] {
            let a = crate::pipeline::testing::compile(&bc, entry, &p)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert_eq!(a.interface.own_pixel_textures, own, "{stem} {entry}");
        }
    }
    assert_eq!(checks.len(), 2, "both read checks");
}

/// The interface states why it leaves an own-pixel slot out of those a draw can read through a
/// snapshot (`StageInterface::snapshot_refusals`, `spirv::requirements::snapshot`): owned fixture
/// `snapshot_discard`'s `sd_read_discarding`, which discards by the texel it read, states none on
/// the Radeon 780M's profile, where the discard demotes the invocation and the slot qualifies, and
/// its branch alone where the profile withholds `shaderDemoteToHelperInvocation` and the discard
/// ends the invocation; `tob_renders`'s `tob_read` states none on either. The stage module of a
/// render pipeline of it states the same.
#[test]
fn the_interface_states_why_a_slot_takes_no_snapshot() {
    use crate::device::requirements::VkFeature;
    use crate::pipeline::interface::SnapshotRefusal;
    let (Some(discarding), Some(reading)) = (fixture("snapshot_discard"), fixture("tob_renders"))
    else {
        return;
    };
    let demotes = device("amd-780m-radv");
    let mut d = description("amd-780m-radv");
    assert!(
        (d.features).remove(&VkFeature::ShaderDemoteToHelperInvocation),
        "the 780M's profile demotes"
    );
    let ends = Device::new(d, DevicePolicy::default()).expect("device");
    for (name, dev, refusals) in [
        ("demote", &demotes, Vec::new()),
        ("no demote", &ends, vec![(0, SnapshotRefusal::Branch)]),
    ] {
        let p = dev.profile();
        let a = crate::pipeline::testing::compile(&discarding, "sd_read_discarding", p)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(a.interface.own_pixel_textures, [0u32], "{name}");
        let slots: &[u32] = if refusals.is_empty() { &[0] } else { &[] };
        assert_eq!(a.interface.snapshot_textures, slots, "{name}");
        assert_eq!(a.interface.snapshot_refusals, refusals, "{name}");
        let read = crate::pipeline::testing::compile(&reading, "tob_read", p)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(read.interface.snapshot_textures, [0u32], "{name}: tob_read");
        assert!(
            read.interface.snapshot_refusals.is_empty(),
            "{name}: tob_read"
        );
        let rect = Library::from_air(&reading).expect("library");
        let rect = rect.function("tob_rect").expect("vertex");
        let discard = Library::from_air(&discarding).expect("library");
        let discard = discard.function("sd_read_discarding").expect("fragment");
        let mut d =
            RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(rect)));
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(discard)));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
            metal::PixelFormat::RGBA8Unorm,
        ));
        let pipe = (dev.compile_render_pipeline(&d, &Budget::standard()))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let fragment = (pipe.stages.iter())
            .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
            .expect("a fragment stage");
        assert_eq!(fragment.snapshot_textures, slots, "{name}: the stage");
        assert_eq!(fragment.snapshot_refusals, refusals, "{name}: the stage");
    }
}

/// A selection on the texel a fragment read at its own pixel keeps the slot where a helper
/// invocation may take either arm (`spirv::requirements::snapshot`'s `arms`): owned fixture
/// `snapshot_select`'s `ss_read_selecting`, whose arm halves the red, states none on the Radeon
/// 780M's profile and on lavapipe's; `ss_read_selecting_dfdx`, whose arm also takes a derivative,
/// and `wlk_readers`'s `wlk_read_branch`, whose texel decides a loop's trip count, state their
/// branch.
#[test]
fn a_selection_on_the_texel_keeps_the_slot_where_a_helper_may_take_either_arm() {
    use crate::pipeline::interface::SnapshotRefusal;
    let (Some(selecting), Some(looping)) = (fixture("snapshot_select"), fixture("wlk_readers"))
    else {
        return;
    };
    for stem in ["amd-780m-radv", "lavapipe"] {
        let p = crate::pipeline::testing::profile(stem, DevicePolicy::default()).expect("profile");
        for (bc, entry, refusals) in [
            (&selecting, "ss_read_selecting", Vec::new()),
            (
                &selecting,
                "ss_read_selecting_dfdx",
                vec![(0, SnapshotRefusal::Branch)],
            ),
            (
                &looping,
                "wlk_read_branch",
                vec![(0, SnapshotRefusal::Branch)],
            ),
        ] {
            let a = crate::pipeline::testing::compile(bc, entry, &p)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert_eq!(a.interface.own_pixel_textures, [0u32], "{stem} {entry}");
            let slots: &[u32] = if refusals.is_empty() { &[0] } else { &[] };
            assert_eq!(a.interface.snapshot_textures, slots, "{stem} {entry}");
            assert_eq!(a.interface.snapshot_refusals, refusals, "{stem} {entry}");
        }
    }
}

/// The interface states the own-pixel slots a draw can read through a snapshot
/// (`spirv::requirements::snapshot`): `tob_read`'s, whose texel reaches only its output. Its
/// snapshot variant (`LoweringInputs::snapshot_textures`) links and validates on every profile with
/// the pixel interlock, which it requires; binds the slot's snapshot; places the phase in the
/// fragment window; still states the slot, as read at the own pixel and through a snapshot; and
/// writes no memory of the function's. A profile without the interlock refuses it, naming the
/// feature. A variant asked for `tob_sample`'s slot, which the fact does not list, would sample the
/// texture beside the copy: refused, never compiled so. All of it holds under the exact answers
/// too.
#[test]
fn a_snapshot_variant_reads_its_texture_in_the_copy_alone() {
    use crate::device::requirements::VkFeature;
    use crate::error::Capability;
    use crate::pipeline::interface::PushLoc;
    use crate::pipeline::sym::{PushField, ResourceKind, SlotClass, SlotRef, SnapshotForm, Stage};
    let Some(bc) = fixture("tob_renders") else {
        return;
    };
    let inputs = |form: SnapshotForm| {
        let mut inputs = LoweringInputs::new();
        let slot = SlotRef {
            stage: Stage::Fragment,
            class: SlotClass::Texture,
            index: 0,
        };
        inputs.snapshot_textures.insert(slot, form);
        inputs
    };
    let mut interlocked = 0;
    let policies = [
        ("default", DevicePolicy::default()),
        ("exact", DevicePolicy::default().exact_answers()),
    ];
    for (stem, name, policy) in [
        "intel-irisxe-anv",
        "nvidia-rtx5080-laptop",
        "amd-780m-radv",
        "lavapipe",
        "moltenvk-m3",
        "vk12-floor",
    ]
    .into_iter()
    .flat_map(|stem| policies.map(|(name, policy)| (stem, name, policy)))
    {
        let p = crate::pipeline::testing::profile(stem, policy).expect("profile");
        let stem = format!("{stem} {name}");
        let base = crate::pipeline::testing::compile(&bc, "tob_read", &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(base.interface.snapshot_textures, [0u32], "{stem}");
        let sampled = crate::pipeline::testing::compile(&bc, "tob_sample", &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(sampled.interface.snapshot_textures.is_empty(), "{stem}");
        let variant = crate::pipeline::testing::compile_with(
            &bc,
            "tob_read",
            &p,
            &inputs(SnapshotForm::Bits),
        );
        if !p.fields().fragment.pixel_interlock {
            assert_eq!(
                variant.map(|_| ()),
                Err(Error::Unsupported(crate::error::Unsupported::needs(
                    Feature::RasterOrderGroups,
                    Capability::Feature(VkFeature::FragmentShaderPixelInterlock),
                ))),
                "{stem}"
            );
            continue;
        }
        interlocked += 1;
        // Each form of a float slot's codes compiles, its copy alone reading the texture.
        for form in [
            SnapshotForm::Unorm8,
            SnapshotForm::Snorm8,
            SnapshotForm::Unorm16,
            SnapshotForm::Snorm16,
            SnapshotForm::Half,
        ] {
            let narrow = crate::pipeline::testing::compile_with(&bc, "tob_read", &p, &inputs(form))
                .unwrap_or_else(|e| panic!("{stem} {form:?}: {e}"));
            assert_eq!(
                narrow.interface.snapshot_textures,
                [0u32],
                "{stem} {form:?}"
            );
        }
        let a = variant.unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(a.interface.snapshot_textures, [0u32], "{stem}");
        assert_eq!(a.interface.own_pixel_textures, [0u32], "{stem}");
        assert!(!a.interface.writes_memory, "{stem}");
        assert!(
            a.interface
                .descriptors
                .iter()
                .any(|d| d.slot.kind == ResourceKind::SnapshotTexels(0)),
            "{stem}"
        );
        assert!(
            a.interface.push.fields.iter().any(|f| {
                f.field == PushField::SnapshotPhase && matches!(f.loc, PushLoc::Push { .. })
            }),
            "{stem}"
        );
        assert!(
            a.requirements
                .features
                .contains(&VkFeature::FragmentShaderPixelInterlock),
            "{stem}"
        );
        crate::pipeline::testing::link_and_validate(&a, &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        match crate::pipeline::testing::compile_with(
            &bc,
            "tob_sample",
            &p,
            &inputs(SnapshotForm::Bits),
        ) {
            Err(Error::Internal(e)) if e.kind == InternalKind::Invariant => {}
            other => panic!("{stem} tob_sample: {other:?}"),
        }
    }
    assert!(
        interlocked >= 8,
        "the interlocked profiles, under both policies"
    );
}

/// The interface states whether the function writes memory another invocation or a later command
/// can read: a buffer store, an image write (through an argument buffer's handle too), in a
/// fragment function or a kernel alike; a function that only reads does not, nor does one that only
/// reads through a pointer it loads (`WIDE_READ`), whose accesses count resident misses in the
/// `ExecutionStatus` buffer: those words are the product's (`spirv::requirements::writes`'s
/// `the_status_words_are_not_the_functions_writes`).
#[test]
fn the_interface_states_whether_the_function_writes_memory() {
    let p =
        crate::pipeline::testing::profile("lavapipe", DevicePolicy::default()).expect("profile");
    for (stem, entry, writes) in [
        ("a3_rog_buffer_rmw", "a3_rog_buffer_rmw", true),
        ("fragment_rog", "rog_loop", true),
        ("handles", "select_null_write", true),
        ("handles", "argument_buffer_array_write", true),
        ("stage_io", "stage_io_fragment_point", false),
        ("sampling", "sample_fragment", false),
    ] {
        let Some(bc) = fixture(stem) else {
            return;
        };
        let a = crate::pipeline::testing::compile(&bc, entry, &p)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(a.interface.writes_memory, writes, "{entry}");
    }
    let Some(bc) = assemble_text("wide_read", WIDE_READ) else {
        return;
    };
    let a = crate::pipeline::testing::compile(&bc, "wide_read_fragment", &p)
        .unwrap_or_else(|e| panic!("wide_read_fragment: {e}"));
    // The miss count reaches the status buffer through its pushed address.
    assert!(a
        .interface
        .push
        .fields
        .iter()
        .any(|f| f.field == crate::pipeline::sym::PushField::ExecutionStatusAddress));
    assert!(!a.interface.writes_memory);
}

/// A kernel over `out` (buffer 0, read-write) and `params` (buffer 1, read), with
/// `thread_position_in_grid` as `%2`; `body` holds the entry's instructions and its blocks, `extra`
/// any function it calls.
fn effects_kernel(name: &str, body: &str, extra: &str) -> String {
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
         target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
         {extra}\
         define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, i32 noundef %2) {{\nentry:\n{body}}}\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"params\"}}\n\
         !14 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// A compute pipeline's interface states whether its kernel writes memory, which the runtime reads
/// to record nothing for a dispatch of a kernel that does not: a kernel that returns at once, one
/// that only reads its buffers, and one that writes only its own thread's array write none; one that
/// stores to its buffer behind a branch, and one whose callee (kept a call, `noinline`) stores
/// through the pointer it is passed, write memory.
#[test]
fn a_compute_pipeline_states_whether_its_kernel_writes_memory() {
    let p =
        crate::pipeline::testing::profile("lavapipe", DevicePolicy::default()).expect("profile");
    let callee = "define void @put(ptr addrspace(1) noundef %p, i32 noundef %v) #0 {\nentry:\n  store i32 %v, ptr addrspace(1) %p, align 4\n  ret void\n}\n\nattributes #0 = { noinline }\n\n";
    for (name, body, extra, writes) in [
        ("bare_return", "  ret void\n", "", false),
        (
            "only_reads",
            "  %p = load i32, ptr addrspace(1) %1, align 4\n  %q = load i32, ptr addrspace(1) %0, align 4\n  %s = add i32 %p, %q\n  ret void\n",
            "",
            false,
        ),
        (
            "local_only",
            "  %a = alloca [4 x i32], align 4\n  %i = and i32 %2, 3\n  %ie = zext i32 %i to i64\n  %e = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 %ie\n  store i32 %2, ptr %e, align 4\n  %j = xor i32 %i, 1\n  %je = zext i32 %j to i64\n  %f = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 %je\n  %v = load i32, ptr %f, align 4\n  ret void\n",
            "",
            false,
        ),
        (
            "branch_store",
            "  %p = load i32, ptr addrspace(1) %1, align 4\n  %c = icmp ne i32 %p, 0\n  br i1 %c, label %then, label %done\nthen:\n  store i32 %2, ptr addrspace(1) %0, align 4\n  br label %done\ndone:\n  ret void\n",
            "",
            true,
        ),
        (
            "callee_store",
            "  call void @put(ptr addrspace(1) noundef %0, i32 noundef %2)\n  ret void\n",
            callee,
            true,
        ),
    ] {
        let Some(bc) = assemble_text(name, &effects_kernel(name, body, extra)) else {
            return;
        };
        let a = crate::pipeline::testing::compile(&bc, name, &p)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(a.interface.writes_memory, writes, "{name}");
        let linked = crate::pipeline::link::link(
            &[&*a],
            &crate::pipeline::link::LinkInputs::standalone(),
            &p,
        )
        .unwrap_or_else(|e| panic!("{name}: link: {e}"));
        assert_eq!(linked.interface.writes_memory, writes, "{name}: pipeline");
        crate::pipeline::testing::link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// The interface states whether the function takes an indirect command buffer, from the AIR's
/// metadata: an argument buffer's `command_buffer` member the kernel never encodes through
/// (`kernel_icb_declared_handle_is_inert`), and one it encodes compute commands through
/// (`icb_replay_encode`), count; a kernel that takes none (`icb_replay_template`, which a
/// command runs) does not. A pipeline's interface states it where one of its stages does.
#[test]
fn the_interface_states_whether_the_function_takes_a_command_buffer() {
    let p =
        crate::pipeline::testing::profile("lavapipe", DevicePolicy::default()).expect("profile");
    let dirs = [crate::pipeline::testing::fixtures_dir(), engine_dir()];
    for (dir, stem, entry, reaches) in [
        (
            &dirs[0],
            "kernel_icb_declared_handle_is_inert",
            "kernel_icb_declared_handle_is_inert",
            true,
        ),
        (&dirs[1], "icb_replay_encode", "icb_replay_encode", true),
        (
            &dirs[1],
            "icb_replay_template",
            "icb_replay_template",
            false,
        ),
    ] {
        let bc = match llvm_assemble(&dir.join(format!("{stem}.ll"))) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
        };
        let a = crate::pipeline::testing::compile(&bc, entry, &p)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(a.interface.reaches_command_buffers, reaches, "{entry}");
        let linked = crate::pipeline::link::link(
            &[&*a],
            &crate::pipeline::link::LinkInputs::standalone(),
            &p,
        )
        .unwrap_or_else(|e| panic!("{entry}: link: {e}"));
        assert_eq!(linked.interface.reaches_command_buffers, reaches, "{entry}");
    }
}

/// The interface states what the function may write into indirect command buffers, from its
/// AIR: a kernel that encodes compute commands (`icb_replay_encode`) its operations' bits, no
/// primitive type and the compute kind; kernels that draw (`icb_render_replay_encode_draw_only`,
/// `icb_render_indexed_encode`) their draws' bits, the constant primitive type they pass
/// (points) and the render kind; a kernel that takes no command buffer (`icb_replay_template`)
/// nothing. A pipeline's interface holds its stages' together.
#[test]
fn the_interface_states_what_the_function_may_write_into_command_buffers() {
    use crate::pipeline::interface::CommandWrites as W;
    let p =
        crate::pipeline::testing::profile("lavapipe", DevicePolicy::default()).expect("profile");
    for (entry, operations, primitive_types, kinds) in [
        (
            "icb_replay_encode",
            W::PIPELINE | W::BUFFER | W::THREADGROUP_MEMORY | W::DISPATCH | W::BARRIER,
            0,
            W::COMPUTE_COMMANDS,
        ),
        (
            "icb_render_replay_encode_draw_only",
            W::DRAW,
            0b1,
            W::RENDER_COMMANDS,
        ),
        (
            "icb_render_indexed_encode",
            W::DRAW_INDEXED,
            0b1,
            W::RENDER_COMMANDS,
        ),
        ("icb_replay_template", 0, 0, 0),
    ] {
        let bc = match llvm_assemble(&engine_dir().join(format!("{entry}.ll"))) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {entry}.ll: {why}"),
        };
        let a = crate::pipeline::testing::compile(&bc, entry, &p)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        let want = W {
            operations,
            primitive_types,
            kinds,
        };
        assert_eq!(a.interface.command_writes, want, "{entry}");
        let linked = crate::pipeline::link::link(
            &[&*a],
            &crate::pipeline::link::LinkInputs::standalone(),
            &p,
        )
        .unwrap_or_else(|e| panic!("{entry}: link: {e}"));
        assert_eq!(linked.interface.command_writes, want, "{entry}");
    }
}

/// The interface states whether the function may write a texture, from its AIR: a fragment
/// function's `read_write` texture it writes (`rog_texture`), an argument buffer's texture array
/// (`argument_buffer_array_write`), a heap's texture (`a4_heap_textures`), and texture atomics
/// (`trh_texture_atomic`) count; a fragment function or a kernel that writes only a buffer
/// (`rog_buffer`, `a3_rog_buffer_rmw`), and ones that only read or sample textures
/// (`argument_buffer_read`, `sample_fragment`), do not. A standalone kernel's pipeline lists its
/// stage where the function writes.
#[test]
fn the_interface_states_whether_the_function_writes_textures() {
    let p =
        crate::pipeline::testing::profile("lavapipe", DevicePolicy::default()).expect("profile");
    for (stem, entry, writes) in [
        ("fragment_rog", "rog_texture", true),
        ("handles", "argument_buffer_array_write", true),
        ("a4_heap_textures", "a4_heap_textures", true),
        ("trh_texture_atomic", "trh_texture_atomic", true),
        ("fragment_rog", "rog_buffer", false),
        ("a3_rog_buffer_rmw", "a3_rog_buffer_rmw", false),
        ("handles", "argument_buffer_read", false),
        ("sampling", "sample_fragment", false),
    ] {
        let Some(bc) = fixture(stem) else {
            return;
        };
        let a = crate::pipeline::testing::compile(&bc, entry, &p)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(a.interface.writes_textures, writes, "{entry}");
        if a.interface.stage == crate::pipeline::sym::Stage::Kernel {
            let linked = crate::pipeline::link::link(
                &[&*a],
                &crate::pipeline::link::LinkInputs::standalone(),
                &p,
            )
            .unwrap_or_else(|e| panic!("{entry}: link: {e}"));
            let stages = if writes {
                vec![crate::pipeline::sym::Stage::Kernel]
            } else {
                Vec::new()
            };
            assert_eq!(linked.interface.texture_writing_stages, stages, "{entry}");
        }
    }
}

/// The interface states whether the function reads or writes a resource through anything but
/// its bound slots, from its AIR and memory plan. Through bound slots alone: textures selected
/// between two arguments or between an argument and null, a texture array argument indexed at
/// run time (`handles`), and atomics on a bound buffer (`rd_touch`, and a fragment function's
/// `a3_rog_buffer_rmw`). Through an argument buffer's texture: a read (`argument_buffer_read`)
/// and a write (`argument_buffer_array_write`), the argument buffer's own words being a bound
/// slot. Through an address the function loads from a buffer: loads and stores
/// (`ob2_loaded_pointer`) and an atomic, which reads and writes (`rd_tally_fragment`). A
/// standalone kernel's pipeline states its function's.
#[test]
fn the_interface_states_what_the_function_reaches_beyond_its_bound_slots() {
    let p =
        crate::pipeline::testing::profile("lavapipe", DevicePolicy::default()).expect("profile");
    for (stem, entry, reads, writes) in [
        ("handles", "select_read", false, false),
        ("handles", "select_null_write", false, false),
        ("handles", "array_read", false, false),
        ("resident_draws", "rd_touch", false, false),
        ("a3_rog_buffer_rmw", "a3_rog_buffer_rmw", false, false),
        ("handles", "argument_buffer_read", true, false),
        ("handles", "argument_buffer_array_write", false, true),
        ("ob2_loaded_pointer", "ob2_loaded_pointer", true, true),
        ("resident_draws", "rd_tally_fragment", true, true),
    ] {
        let Some(bc) = fixture(stem) else {
            return;
        };
        let a = crate::pipeline::testing::compile(&bc, entry, &p)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        let facts = (
            a.interface.reads_unbound_resources,
            a.interface.writes_unbound_resources,
        );
        assert_eq!(facts, (reads, writes), "{entry}");
        if a.interface.stage == crate::pipeline::sym::Stage::Kernel {
            let linked = crate::pipeline::link::link(
                &[&*a],
                &crate::pipeline::link::LinkInputs::standalone(),
                &p,
            )
            .unwrap_or_else(|e| panic!("{entry}: link: {e}"));
            let linked_facts = (
                linked.interface.reads_unbound_resources,
                linked.interface.writes_unbound_resources,
            );
            assert_eq!(linked_facts, (reads, writes), "{entry}: pipeline");
        }
    }
}

/// A render pipeline whose fragment function reads `[[sample_id]]` and `[[point_coord]]`
/// (`stage_io_fragment_point`) compiles at one sample and at four, where Vulkan's sample shading
/// may move the point coordinate to the sample while Metal keeps it at the pixel's: there its
/// fragment module notes `MultisamplePosition`, and at one sample nothing.
#[test]
fn a_sample_rate_point_coordinate_is_noted_under_multisampling() {
    use crate::pipeline::artifact::PrecisionNote;
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function("stage_io_vertex_position").expect("vertex"),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        lib.function("stage_io_fragment_point").expect("fragment"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA16Float,
    ));
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem);
        for samples in [1, 4] {
            d.raster_sample_count = samples;
            let p = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem} at {samples}: {e}"));
            let fragment = p
                .stages
                .iter()
                .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
                .expect("fragment stage");
            assert_eq!(
                fragment
                    .precision_notes
                    .contains(&PrecisionNote::MultisamplePosition),
                samples == 4,
                "{stem} at {samples}"
            );
        }
    }
}

/// A vertex function over a two-triangle strip with a center varying `uv`, and a fragment
/// function that loads `root->next` (a `device uint *` member of its `device Node *` buffer)
/// and reads through it beside `uv`, writing nothing. Every access through the loaded pointer
/// counts a resident miss in the `ExecutionStatus` buffer under the default policy.
const WIDE_READ: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float>, <2 x float> }> @wide_read_vertex(i32 %vid) {
entry:
  %lo = and i32 %vid, 1
  %hi = lshr i32 %vid, 1
  %x = uitofp i32 %lo to float
  %y = uitofp i32 %hi to float
  %p0 = insertelement <4 x float> <float poison, float poison, float 0.5, float 1.0>, float %x, i64 0
  %p = insertelement <4 x float> %p0, float %y, i64 1
  %u0 = insertelement <2 x float> poison, float %x, i64 0
  %u = insertelement <2 x float> %u0, float %y, i64 1
  %r0 = insertvalue <{ <4 x float>, <2 x float> }> poison, <4 x float> %p, 0
  %r = insertvalue <{ <4 x float>, <2 x float> }> %r0, <2 x float> %u, 1
  ret <{ <4 x float>, <2 x float> }> %r
}

define <4 x float> @wide_read_fragment(<2 x float> %uv, ptr addrspace(1) %root) {
entry:
  %next = load ptr addrspace(1), ptr addrspace(1) %root, align 8
  %word = load i32, ptr addrspace(1) %next, align 4
  %f = uitofp i32 %word to float
  %x = extractelement <2 x float> %uv, i64 0
  %y = extractelement <2 x float> %uv, i64 1
  %a = insertelement <4 x float> <float poison, float poison, float poison, float 1.0>, float %x, i64 0
  %b = insertelement <4 x float> %a, float %y, i64 1
  %c = insertelement <4 x float> %b, float %f, i64 2
  ret <4 x float> %c
}

!air.vertex = !{!0}
!air.fragment = !{!6}
!air.version = !{!15}
!air.language_version = !{!16}
!0 = !{ptr @wide_read_vertex, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(2uvDv2_f)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!4 = !{!5}
!5 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!6 = !{ptr @wide_read_fragment, !7, !9}
!7 = !{!8}
!8 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!9 = !{!10, !11}
!10 = !{i32 0, !"air.fragment_input", !"generated(2uvDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!11 = !{i32 1, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !12, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Node", !"air.arg_name", !"root"}
!12 = !{i32 0, i32 8, i32 0, !"uint", !"next", !"air.indirect_argument", !13, i32 8, i32 4, i32 0, !"uint", !"value", !"air.indirect_argument", !14}
!13 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"next"}
!14 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"value"}
!15 = !{i32 2, i32 8, i32 0}
!16 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A multisampled render pipeline whose fragment function only reads through a pointer it loads
/// from a buffer (`WIDE_READ`) runs it at pixel rate: the resident-miss count its access keeps in
/// the `ExecutionStatus` buffer is the product's write, not the function's, so the pipeline needs
/// no sample shading and compiles on moltenvk-m3 as on lavapipe.
#[test]
fn a_fragment_reading_through_a_loaded_pointer_runs_at_pixel_rate() {
    let Some(bc) = assemble_text("wide_read", WIDE_READ) else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function("wide_read_vertex").expect("vertex"),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        lib.function("wide_read_fragment").expect("fragment"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA16Float,
    ));
    d.raster_sample_count = 4;
    for stem in ["moltenvk-m3", "lavapipe"] {
        let pipe = device(stem)
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(!pipe.graphics.sample_shading, "{stem}");
    }
}

/// A vertex function that writes `[[viewport_array_index]]` depends on the draw's viewport count,
/// which its push window holds and each draw writes from `BindState::viewport_count`, and writes
/// the index as written to a flat varying (`IoRole::ViewportIndex`). In its pipeline a fragment
/// function that reads the index compiles under `viewport_index_varying` and reads that varying, as
/// `render_pipeline_key` names; one that does not read it keeps the key it has without the field. A
/// reading fragment compiled for the other kind of vertex function is `NeedsInput(ViewportIndex)`
/// either way. Every module passes L3.
#[test]
fn a_fragment_reads_the_viewport_index_its_vertex_function_wrote() {
    use crate::bind::plan::FieldLocation;
    use crate::bind::{BindState, DrawCall};
    use crate::error::RuntimeInput;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::stage_io::{IoRole, StageBuiltin};
    use crate::pipeline::sym::{PushField, Stage, VaryingKey};
    let (Some(bc), Some(other)) = (fixture("rt6_viewports"), fixture("rt6_render_state")) else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let other = Library::from_air(&other).expect("library");
    let function = |lib: &Library, name: &str| {
        FunctionDescriptor::new(lib.function(name).unwrap_or_else(|e| panic!("{name}: {e}")))
    };
    let dev = device("moltenvk-m3");
    let budget = Budget::standard();
    let pipeline = |vertex: StageSource, fragment: StageSource| {
        let mut d = RenderPipelineDescriptor::new(vertex);
        d.fragment = Some(fragment);
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
            metal::PixelFormat::RGBA8Unorm,
        ));
        d
    };
    let select = || StageSource::Function(function(&lib, "rt6_select"));
    let reader = function(&lib, "rt6_viewport_read");

    // The vertex function: the fact, the push field, the builtin, and the raw varying.
    let vertex = dev
        .compile_function(
            &function(&lib, "rt6_select"),
            &LoweringInputs::new(),
            &budget,
        )
        .expect("vertex");
    let a = vertex.artifact();
    assert!(a.facts.contains(&RuntimeFact::ViewportCount));
    assert!(a
        .interface
        .push
        .fields
        .iter()
        .any(|f| f.field == PushField::ViewportCount));
    assert!(a
        .interface
        .io
        .builtins
        .contains(&StageBuiltin::ViewportIndex));
    let raw: Vec<_> = a
        .interface
        .io
        .outputs
        .iter()
        .filter(|v| v.role == IoRole::ViewportIndex)
        .collect();
    assert_eq!(raw.len(), 1);
    assert_eq!(raw[0].key, VaryingKey::viewport_index());

    // The reading fragment in the pipeline: the field, the varying, no builtin.
    let d = pipeline(select(), StageSource::Function(reader.clone()));
    let keys = dev.render_pipeline_key(&d, &budget).expect("keys");
    let inputs = keys.fragment_inputs.clone().expect("fragment inputs");
    assert!(inputs.viewport_index_varying);
    let pipe = dev.compile_render_pipeline(&d, &budget).expect("pipeline");
    assert_eq!(pipe.key, keys.pipeline);
    for s in &pipe.stages {
        l3(&s.spirv, &pipe.requirements, &dev);
    }
    assert!(pipe
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::ViewportCount));
    let fragment = dev
        .compile_function(&reader, &inputs, &budget)
        .expect("fragment");
    assert_eq!(Some(*fragment.key()), keys.fragment);
    let io = &fragment.artifact().interface.io;
    assert!(io.inputs.iter().any(|v| v.role == IoRole::ViewportIndex));
    assert!(!io.builtins.contains(&StageBuiltin::ViewportIndex));

    // A fragment that does not read the index keeps its key.
    let white = function(&lib, "rt6_white");
    let keys = dev
        .render_pipeline_key(
            &pipeline(select(), StageSource::Function(white.clone())),
            &budget,
        )
        .expect("keys");
    assert_eq!(
        keys.fragment,
        Some(dev.function_compile_key(&white, &LoweringInputs::new()))
    );

    // Each draw writes the encoder's viewport count into the vertex push window.
    let state = BindState {
        viewport_count: 3,
        ..BindState::default()
    };
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let placed = plan
        .push
        .fields
        .iter()
        .find(|f| f.stage == Stage::Vertex && f.field == PushField::ViewportCount)
        .expect("placed");
    let FieldLocation::PushWindow { offset } = placed.location else {
        panic!("the count spilled: {placed:?}");
    };
    let draw = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 4,
    };
    let commands = plan.instantiate_draw(&state, &draw).expect("draw");
    let write = commands
        .push_constants
        .iter()
        .find(|w| w.stages.contains(crate::vk::ShaderStageFlags::VERTEX))
        .expect("vertex push");
    let at = (offset - write.offset) as usize;
    assert_eq!(write.bytes[at..at + 4], 3u32.to_le_bytes());

    // Compiled for the other kind of vertex function: refused, never a wrong read.
    let without = dev
        .compile_function(&reader, &LoweringInputs::new(), &budget)
        .expect("fragment");
    let mut under = LoweringInputs::new();
    under.viewport_index_varying = true;
    let under = dev
        .compile_function(&reader, &under, &budget)
        .expect("fragment");
    let full = || StageSource::Function(function(&other, "rt6_full"));
    for d in [
        pipeline(select(), StageSource::Compiled(without)),
        pipeline(full(), StageSource::Compiled(under)),
    ] {
        assert_eq!(
            dev.compile_render_pipeline(&d, &budget).err(),
            Some(Error::NeedsInput(RuntimeInput::ViewportIndex))
        );
    }
}

/// A merged draw replay's vertex function (`VariantKey::merged_draws`) reads its command's slot
/// table from the header at its `MergedDrawAddress`, a hole of the vertex stage, in place of a
/// `RootTableAddress`, and its plan states the variant. A fragment function that reads a buffer
/// reads its command's fragment slot table from the header too, at the command its vertex wrote
/// (`IoRole::MergedCommand`): `MergedDrawAddress` is its push field, `RootTableAddress` none. One
/// that reads `[[primitive_id]]`, which Metal counts per command, is refused, as is a merged key
/// without replay bindings.
#[test]
fn a_merged_draw_reads_its_command_s_table_from_its_header() {
    use crate::cache::VariantKey;
    use crate::pipeline::sym::PushField;
    let dev = device("moltenvk-m3");
    let mut key = VariantKey::new();
    key.replay_bindings = true;
    key.merged_draws = true;
    let compile = |vertex: (&str, &str),
                   fragment: Option<(&str, &str, metal::PixelFormat)>,
                   key: &VariantKey| {
        let function = |(file, name): (&str, &str)| {
            let bc = fixture(file)?;
            Some(FunctionDescriptor::new(
                Library::from_air(&bc)
                    .expect("library")
                    .function(name)
                    .expect("function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(vertex)?));
        match fragment {
            Some((file, name, format)) => {
                d.fragment = Some(StageSource::Function(function((file, name))?));
                d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
            }
            None => d.rasterization_enabled = false,
        }
        Some(dev.compile_render_variant(&d, key, &Budget::standard()))
    };
    let Some(merged) = compile(("primitive_ids", "pi_place"), None, &key) else {
        return;
    };
    let merged = merged.expect("merged");
    let fields: Vec<PushField> = merged
        .interface
        .push
        .iter()
        .flat_map(|p| p.fields.iter().map(|f| f.field))
        .collect();
    assert!(fields.contains(&PushField::MergedDrawAddress), "{fields:?}");
    assert!(!fields.contains(&PushField::RootTableAddress), "{fields:?}");
    assert!(
        !fields.contains(&PushField::BaseVertexInstance),
        "{fields:?}"
    );
    let plan = merged
        .plan(&dev, &crate::bind::BindShape::default())
        .expect("plan");
    assert!(plan.variant.replay_bindings && plan.variant.merged_draws);
    let placed: Vec<_> = plan
        .push
        .fields
        .iter()
        .filter(|f| f.field == PushField::MergedDrawAddress)
        .map(|f| f.stage)
        .collect();
    assert_eq!(placed, [crate::pipeline::sym::Stage::Vertex]);
    let mut unreplayed = VariantKey::new();
    unreplayed.merged_draws = true;
    let plain = compile(("primitive_ids", "pi_place"), None, &unreplayed)
        .expect("fixture")
        .expect("merged alone is no variant");
    assert!(!plain.interface.push.iter().any(|p| p
        .fields
        .iter()
        .any(|f| f.field == PushField::MergedDrawAddress)));
    let (vertex, fragment) = (
        ("primitive_ids", "pi_place"),
        ("primitive_ids", "pi_ids", metal::PixelFormat::RGBA32Uint),
    );
    compile(vertex, Some(fragment), &VariantKey::new())
        .expect("fixture")
        .expect("plain");
    let refused = compile(vertex, Some(fragment), &key)
        .expect("fixture")
        .expect_err("a fragment's primitive ID under a merged draw");
    assert!(
        format!("{refused:?}").contains("GpuEncodedIndirectCommands"),
        "{fragment:?}: {refused:?}"
    );
    let (vertex, fragment) = (
        ("a8r_targets", "a8r_vertex"),
        ("a8r_targets", "a8r_fill", metal::PixelFormat::RGBA8Unorm),
    );
    let buffered = compile(vertex, Some(fragment), &key)
        .expect("fixture")
        .expect("a fragment's buffer under a merged draw");
    let fragment_fields: Vec<PushField> = (buffered.interface.push.iter())
        .filter(|p| p.stage == crate::pipeline::sym::Stage::Fragment)
        .flat_map(|p| p.fields.iter().map(|f| f.field))
        .collect();
    assert!(
        fragment_fields.contains(&PushField::MergedDrawAddress)
            && !fragment_fields.contains(&PushField::RootTableAddress),
        "{fragment_fields:?}"
    );
}

/// A merged draw replay's functions find their command's slot tables before any entry hook reads
/// one: a fragment function that loads a pointer from its argument buffer reads that buffer's
/// root entry at entry (`access::dynamic::emit_entry`), from the fragment slot table the command
/// its vertex wrote names, and the merged variant of `icb_merged_arguments` (a vertex function
/// that fetches `[[stage_in]]` attributes, a fragment function whose argument buffer holds two
/// textures and a constant pointer) compiles on each device that has the replay, every stage
/// passing L3, its fragment stage reading the header at its `MergedDrawAddress`.
#[test]
fn a_merged_fragment_reads_its_argument_buffer_through_its_command_s_table() {
    use crate::cache::VariantKey;
    use crate::pipeline::sym::{PushField, Stage};
    let Some(bc) = fixture("icb_merged_arguments") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(lib.function(name).expect(name)))
    };
    let mut d = RenderPipelineDescriptor::new(function("ica_vertex"));
    d.fragment = Some(function("ica_fragment"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(
        metal::PixelFormat::RGBA32Uint,
    ));
    let mut input = VertexDescriptor::new();
    input
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(8)));
    input
        .layouts
        .insert(1, VertexBufferLayoutDescriptor::new(Some(4)));
    input.attributes.insert(
        0,
        VertexAttributeDescriptor::new(metal::VertexFormat::Float2, 0, 0),
    );
    input.attributes.insert(
        1,
        VertexAttributeDescriptor::new(metal::VertexFormat::UInt, 0, 1),
    );
    d.vertex_descriptor = Some(input);
    for stem in ["moltenvk-m3", "amd-rdna3-radv", "lavapipe"] {
        let dev = device(stem);
        let base = dev
            .compile_render_variant(&d, &VariantKey::new(), &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: base: {e}"));
        let layouts = crate::bind::draw_replay_vertex_layouts(&base.vertex_input).expect("layouts");
        let shape = crate::bind::BindShape {
            far_vertex_buffers: layouts,
            ..Default::default()
        };
        let mut key = base.plan(&dev, &shape).expect("base plan").variant;
        key.replay_bindings = true;
        key.merged_draws = true;
        let merged = dev
            .compile_render_variant(&d, &key, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: merged: {e}"));
        for stage in &merged.stages {
            l3(&stage.spirv, &merged.requirements, &dev);
        }
        let fragment: Vec<PushField> = (merged.interface.push.iter())
            .filter(|p| p.stage == Stage::Fragment)
            .flat_map(|p| p.fields.iter().map(|f| f.field))
            .collect();
        assert!(
            fragment.contains(&PushField::MergedDrawAddress)
                && !fragment.contains(&PushField::RootTableAddress),
            "{stem}: {fragment:?}"
        );
    }
}

/// A render pipeline compiled as a device-replay variant (`VariantKey::replay_bindings`) reads
/// every buffer its vertex function reaches through its root entry (`BufferForm::Psb`), from the
/// slot table at the stage's own `RootTableAddress`, and needs no `SlotGuestBase`
/// (`icb_render_replay_template`); its plan states the variant. A vertex function that fetches
/// a `[[stage_in]]` attribute (`icb_render_stride_template`) in fixed function is refused:
/// fixed-function input reads no slot table. A draw replay names its layout among those the draw
/// pulls (`bind::draw_replay_vertex_layouts`, `BindShape::far_vertex_buffers`) with no buffer
/// bound there: the base plan asks for the pulled-vertex-buffer variant, whose replay variant
/// compiles, one draw per command and merged, binds no layout and pulls the attribute, whose
/// read reaches the slot table; every stage passes L3, and its plan states its key.
#[test]
fn a_replayed_draw_reads_its_buffers_through_the_slot_table() {
    use crate::cache::VariantKey;
    use crate::pipeline::interface::{BufferForm, SlotInterface};
    use crate::pipeline::sym::PushField;
    let dev = device("moltenvk-m3");
    let mut key = VariantKey::new();
    key.replay_bindings = true;
    let Some(bc) = fixture("icb_render_replay_template") else {
        return;
    };
    let vertex = Library::from_air(&bc)
        .expect("library")
        .function("icb_render_replay_template")
        .expect("vertex");
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(vertex)));
    d.rasterization_enabled = false;
    let plain = dev
        .compile_render_variant(&d, &VariantKey::new(), &Budget::standard())
        .expect("plain");
    let replayed = dev
        .compile_render_variant(&d, &key, &Budget::standard())
        .expect("replayed");
    let forms = |p: &crate::pipeline::artifact::RenderPipeline| -> Vec<BufferForm> {
        p.interface
            .slots
            .values()
            .filter_map(|s| match s {
                SlotInterface::Buffer(b) => Some(b.form),
                _ => None,
            })
            .collect()
    };
    assert_eq!(forms(&replayed).len(), 2);
    assert!(forms(&replayed).iter().all(|f| *f == BufferForm::Psb));
    assert!(forms(&plain).iter().any(|f| *f != BufferForm::Psb));
    let fields: Vec<PushField> = replayed
        .interface
        .push
        .iter()
        .flat_map(|p| p.fields.iter().map(|f| f.field))
        .collect();
    assert!(fields.contains(&PushField::RootTableAddress));
    assert!(!fields
        .iter()
        .any(|f| matches!(f, PushField::SlotGuestBase(_))));
    let plan = replayed
        .plan(&dev, &crate::bind::BindShape::default())
        .expect("plan");
    assert!(plan.variant.replay_bindings);

    let Some(bc) = fixture("icb_render_stride_template") else {
        return;
    };
    let vertex = Library::from_air(&bc)
        .expect("library")
        .function("icb_render_stride_template")
        .expect("vertex");
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(vertex)));
    d.rasterization_enabled = false;
    d.vertex_descriptor = Some(vertex_descriptor(&[(0, metal::VertexFormat::UInt)]));
    let base = dev
        .compile_render_variant(&d, &VariantKey::new(), &Budget::standard())
        .expect("fetched");
    let refused = dev
        .compile_render_variant(&d, &key, &Budget::standard())
        .expect_err("a fetch under replay bindings");
    assert!(
        format!("{refused:?}").contains("GpuEncodedIndirectCommands"),
        "{refused:?}"
    );
    // The replay names the base's layouts in the shape, no buffer bound at them: the base plan
    // asks for the pulled-vertex-buffer variant, whose replay forms' plans state themselves.
    let layouts = crate::bind::draw_replay_vertex_layouts(&base.vertex_input).expect("layouts");
    assert_eq!(layouts, std::collections::BTreeSet::from([0]));
    let shape = crate::bind::BindShape {
        far_vertex_buffers: layouts.clone(),
        ..Default::default()
    };
    let plan = base
        .plan(&dev, &shape)
        .expect("the base plan, no buffer bound");
    assert_eq!(plan.variant.pulled_vertex_buffers, layouts);
    for merged in [false, true] {
        let mut pulled = plan.variant.clone();
        pulled.replay_bindings = true;
        pulled.merged_draws = merged;
        let p = dev
            .compile_render_variant(&d, &pulled, &Budget::standard())
            .unwrap_or_else(|e| panic!("merged {merged}: {e}"));
        let input = &p.vertex_input;
        assert!(input.bindings.is_empty(), "merged {merged}");
        assert_eq!(input.pulled.len(), 1, "merged {merged}");
        assert_eq!(input.variant_pulled, layouts);
        // The pulled layout's buffer is read through its root entry, as every other slot.
        let slots = forms(&p);
        assert!(
            !slots.is_empty() && slots.iter().all(|f| *f == BufferForm::Psb),
            "merged {merged}: {slots:?}"
        );
        for stage in &p.stages {
            l3(&stage.spirv, &p.requirements, &dev);
        }
        let replayed = p.plan(&dev, &shape).expect("the variant's plan");
        assert_eq!(replayed.variant, pulled, "merged {merged}");
        assert!(replayed.vertex_buffers.is_empty(), "merged {merged}");
    }
}

/// A draw replay's variant reads every buffer through the slot table its replay's preprocess
/// writes, so inline bytes the encoder bound at a slot it reads are no refusal: commands that
/// inherit the encoder's buffers read them at the consumer's upload of them, whose address the
/// preprocess's entry takes (`bind::DrawReplayPreprocess::inherited_bytes`). Its plan for a shape
/// with inline bytes at every buffer slot plans; one with a buffer bound without a device
/// address there, which no entry can name, stays `MissingDeviceAddress`.
#[test]
fn a_replayed_draw_plans_inline_bytes_at_its_slots() {
    use crate::bind::{BindShape, BufferShape, StageShape};
    use crate::cache::VariantKey;
    use crate::error::BindError;
    use crate::pipeline::interface::SlotInterface;
    let dev = device("moltenvk-m3");
    let mut key = VariantKey::new();
    key.replay_bindings = true;
    let Some(bc) = fixture("icb_render_replay_template") else {
        return;
    };
    let vertex = Library::from_air(&bc)
        .expect("library")
        .function("icb_render_replay_template")
        .expect("vertex");
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(vertex)));
    d.rasterization_enabled = false;
    let replayed = dev
        .compile_render_variant(&d, &key, &Budget::standard())
        .expect("replayed");
    let slots: Vec<_> = (replayed.interface.slots.iter())
        .filter(|(_, s)| matches!(s, SlotInterface::Buffer(_)))
        .map(|(slot, _)| *slot)
        .collect();
    assert!(!slots.is_empty());
    let shape_with = |bound: BufferShape| {
        let mut shape = BindShape::default();
        for slot in &slots {
            let stage: &mut StageShape = shape.stages.entry(slot.stage).or_default();
            stage.buffers.insert(slot.index, bound);
        }
        shape
    };
    let bytes = BufferShape::Bytes {
        attribute_stride: None,
    };
    let plan = replayed
        .plan(&dev, &shape_with(bytes))
        .expect("inline bytes");
    assert!(plan.variant.replay_bindings);
    let unaddressed = BufferShape::Buffer {
        attribute_stride: None,
        device_address: false,
        length_class: 0,
        offset_alignment: 4,
    };
    let refused = replayed.plan(&dev, &shape_with(unaddressed));
    assert!(
        matches!(refused, Err(BindError::MissingDeviceAddress { .. })),
        "{refused:?}"
    );
}
