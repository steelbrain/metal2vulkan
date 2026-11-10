//! Pipeline states of stage functions made by name or specialized, through the runtime on the
//! device the cfg selects, under the validation layer, against what the probe
//! `undefined_function_constant_values.m` (the conformance crate's `probes/`) measured on an
//! Apple M3 Max under macOS 27.0.1: plain Metal ends its process at a state of a function made by
//! `newFunctionWithName:` that declares a function constant, at every stage, and makes one of a
//! function made by name that declares none, and of every function made with constant values,
//! none among them too, or by `newFunctionWithDescriptor:`.
//!
//! The library is the owned fixture `unspecialized_stage_functions`: kernels
//! `unspecialized_guarded` (stores its optional constant where it is defined, else 7: the probe's
//! `ko`), `unspecialized_gate` (a constant gates its buffer 1, which it reads where the constant is
//! true) and `unspecialized_free` (stores 9: the probe's `kn`); vertex,
//! fragment, tile, mesh and object functions reading a constant, and vertex, fragment and mesh
//! functions reading none (`_free`).

use super::at_plan;
use super::host_memory::on_device_with;
use super::host_memory::{on_device, run};
use super::read_buffer;
use crate::air_library::AirLibrary;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, MeshPipelineDescriptor, RenderPipelineDescriptor, Stage,
    StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Device, Error, Function, FunctionDescriptor, Invalid, Library, PipelineOptions, Problem,
    ResourceOptions, Size,
};

/// How a function is made.
#[derive(Clone, Copy, Debug)]
enum Made {
    /// `newFunctionWithName:`.
    ByName,
    /// `newFunctionWithName:constantValues:` with no value.
    Empty,
    /// `newFunctionWithDescriptor:` with no constant values.
    Descriptor,
}

/// The fixture's library.
fn library(device: &Device) -> Library {
    let air = llvm_assemble(&engine_dir().join("unspecialized_stage_functions.ll")).unwrap();
    device.new_library_with_air(&air).unwrap()
}

/// Function `name` of `library`, made `how`.
fn made(library: &Library, name: &str, how: Made) -> Function {
    match how {
        Made::ByName => library.new_function_with_name(name),
        Made::Empty => library
            .new_function_with_name_constant_values(name, &FunctionConstantValues::defaults()),
        Made::Descriptor => library.new_function_with_descriptor(&FunctionDescriptor::new(name)),
    }
    .unwrap()
}

/// The refusal in `call` of the function `name` of `stage`.
fn unspecialized(call: &'static str, stage: Stage, name: &str) -> Error {
    Error::Invalid(Invalid::new(
        call,
        Problem::UnspecializedFunction {
            stage,
            name: name.to_owned(),
        },
    ))
}

/// Whether `made` is the refusal of an unspecialized function.
fn refused_unspecialized<T>(made: &Result<T, Error>) -> bool {
    matches!(made, Err(Error::Invalid(i))
        if matches!(i.problem, Problem::UnspecializedFunction { .. }))
}

/// The word kernel `f`'s compute state stores at thread 0, buffers 0 and 1 bound, or the
/// creation's refusal.
fn word(device: &Device, f: &Function) -> Result<u32, Error> {
    let state = device.new_compute_pipeline_state_with_function(f, &PipelineOptions::default())?;
    let out = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let extra = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&extra), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    let bytes = read_buffer(&out, 0, 4).unwrap();
    Ok(u32::from_le_bytes(bytes[..4].try_into().unwrap()))
}

/// The render state of `vertex` and `fragment` into one BGRA8Unorm attachment, made or refused.
fn render(device: &Device, vertex: &Function, fragment: &Function) -> Result<(), Error> {
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    d.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .map(|_| ())
}

/// The tile state of `tile` over one BGRA8Unorm color, made or refused.
fn tile(device: &Device, tile: &Function) -> Result<(), Error> {
    let mut d =
        TileRenderPipelineDescriptor::new(StageSource::Function(tile.function_descriptor()));
    d.color_attachment_formats[0] = Some(PixelFormat::BGRA8Unorm);
    d.threadgroup_size_matches_tile_size = true;
    device
        .new_tile_render_pipeline_state(&d, &PipelineOptions::default())
        .map(|_| ())
}

/// The mesh render state of `object` where given, `mesh` and `fragment`, into one BGRA8Unorm
/// attachment, made or refused.
fn mesh(
    device: &Device,
    object: Option<&Function>,
    mesh: &Function,
    fragment: &Function,
) -> Result<(), Error> {
    let mut d = MeshPipelineDescriptor::new(StageSource::Function(mesh.function_descriptor()));
    d.object = object.map(|f| StageSource::Function(f.function_descriptor()));
    d.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    device
        .new_mesh_render_pipeline_state(&d, &PipelineOptions::default())
        .map(|_| ())
}

/// A kernel made by name that declares a constant makes no compute state, whether it reads the
/// constant only where it is defined or the constant gates its argument (the probe's `by-name:ko`
/// and `by-name:kg`); one that declares none stores 9 (`by-name:kn`). Made with no constant value
/// or by descriptor, each is made: the guarded kernel stores 7, as Metal's `ko` stored, and the
/// gated one what its undefined constant reads (docs/API.md, "Function constants"; Metal's `kg`
/// stored nothing).
#[test]
fn a_kernel_made_by_name_that_declares_a_constant_makes_no_compute_state() {
    on_device_with(at_plan, |device| {
        let library = library(device);
        let call = "new_compute_pipeline_state";
        for name in ["unspecialized_guarded", "unspecialized_gate"] {
            assert_eq!(
                word(device, &made(&library, name, Made::ByName)),
                Err(unspecialized(call, Stage::Kernel, name)),
                "{name}"
            );
        }
        let free = "unspecialized_free";
        assert_eq!(word(device, &made(&library, free, Made::ByName)), Ok(9));
        for how in [Made::Empty, Made::Descriptor] {
            let guarded = made(&library, "unspecialized_guarded", how);
            assert_eq!(word(device, &guarded), Ok(7), "{how:?}");
            let gate = made(&library, "unspecialized_gate", how);
            let gated = word(device, &gate);
            assert!(gated.is_ok(), "{how:?}: {gated:?}");
            assert_eq!(word(device, &made(&library, free, how)), Ok(9), "{how:?}");
        }
    });
}

/// A vertex or fragment function made by name that declares a constant makes no render state,
/// beside a function made by name that declares none (the probe's `by-name:vertex` and
/// `by-name:fragment`); two that declare none make one (`by-name:plain`), and so does each made
/// with no constant value or by descriptor (`empty:*`, `descriptor:*`).
#[test]
fn a_render_function_made_by_name_that_declares_a_constant_makes_no_render_state() {
    on_device(|device| {
        let library = library(device);
        let call = "new_render_pipeline_state";
        let f = |name, how| made(&library, name, how);
        let vertex_free = f("unspecialized_vertex_free", Made::ByName);
        let fragment_free = f("unspecialized_fragment_free", Made::ByName);
        assert_eq!(
            render(
                device,
                &f("unspecialized_vertex", Made::ByName),
                &fragment_free
            ),
            Err(unspecialized(call, Stage::Vertex, "unspecialized_vertex"))
        );
        assert_eq!(
            render(
                device,
                &vertex_free,
                &f("unspecialized_fragment", Made::ByName)
            ),
            Err(unspecialized(
                call,
                Stage::Fragment,
                "unspecialized_fragment"
            ))
        );
        assert_eq!(render(device, &vertex_free, &fragment_free), Ok(()));
        for how in [Made::Empty, Made::Descriptor] {
            let vertex = f("unspecialized_vertex", how);
            assert_eq!(render(device, &vertex, &fragment_free), Ok(()), "{how:?}");
            let fragment = f("unspecialized_fragment", how);
            assert_eq!(render(device, &vertex_free, &fragment), Ok(()), "{how:?}");
        }
    });
}

/// A tile, mesh or object function made by name that declares a constant makes no state (the
/// probe's `by-name:tile`, `by-name:mesh` and `by-name:object`), before anything the device lacks
/// for it is asked. Made with no constant value or by descriptor, or made by name declaring none,
/// none is refused as unspecialized; the device may refuse such a state for what it lacks (tile or
/// mesh shading).
#[test]
fn a_tile_mesh_or_object_function_made_by_name_that_declares_a_constant_makes_no_state() {
    on_device(|device| {
        let library = library(device);
        let f = |name, how| made(&library, name, how);
        let mesh_free = f("unspecialized_mesh_free", Made::ByName);
        let fragment_free = f("unspecialized_fragment_free", Made::ByName);
        assert_eq!(
            tile(device, &f("unspecialized_tile", Made::ByName)),
            Err(unspecialized(
                "new_tile_render_pipeline_state",
                Stage::Kernel,
                "unspecialized_tile"
            ))
        );
        let call = "new_mesh_render_pipeline_state";
        assert_eq!(
            mesh(
                device,
                None,
                &f("unspecialized_mesh", Made::ByName),
                &fragment_free
            ),
            Err(unspecialized(call, Stage::Mesh, "unspecialized_mesh"))
        );
        let object = f("unspecialized_object", Made::ByName);
        assert_eq!(
            mesh(device, Some(&object), &mesh_free, &fragment_free),
            Err(unspecialized(call, Stage::Object, "unspecialized_object"))
        );
        assert!(!refused_unspecialized(&mesh(
            device,
            None,
            &mesh_free,
            &fragment_free
        )));
        for how in [Made::Empty, Made::Descriptor] {
            let made_tile = tile(device, &f("unspecialized_tile", how));
            assert!(!refused_unspecialized(&made_tile), "{how:?}: {made_tile:?}");
            let mesh_constant = f("unspecialized_mesh", how);
            let made_mesh = mesh(device, None, &mesh_constant, &fragment_free);
            assert!(!refused_unspecialized(&made_mesh), "{how:?}: {made_mesh:?}");
            let object = f("unspecialized_object", how);
            let made_object = mesh(device, Some(&object), &mesh_free, &fragment_free);
            assert!(
                !refused_unspecialized(&made_object),
                "{how:?}: {made_object:?}"
            );
        }
    });
}
