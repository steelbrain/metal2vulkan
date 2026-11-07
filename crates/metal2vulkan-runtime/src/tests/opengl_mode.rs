//! OpenGL mode (`PrivateRenderState::open_gl_mode_enabled`) makes the two render states Metal
//! refuses without it, each drawing as an Apple M3 Max draws it (probe
//! `conformance/probes/opengl_mode.m`, macOS 27.0.1): a fragment function writing depth without a
//! depth attachment draws its colors as with one; a rasterizing state whose vertex function returns
//! `void` runs the vertex function and rasterizes nothing. Without the mode both are refused with
//! Metal's errors (`pipeline/linked_fragment.rs`, `pipeline/unrasterized.rs`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are those of the owned
//! fixture `validation/fixtures/public/examples/icb_linked_arguments.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{library, pass, words_of};
use crate::{
    Buffer, Device, Error, Library, PipelineOptions, Problem, Region, RenderPipelineState,
    ResourceOptions, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The target's side.
const SIDE: u64 = 8;

/// The texel `tint` holds at (1, 0), which the depth-writing fragment function writes as its
/// depth and the `void` vertex function stores.
const TEXEL: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

fn stage(library: &Library, name: &str) -> StageSource {
    StageSource::Function(
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor(),
    )
}

/// A descriptor of `vertex` and `fragment` drawing into `RGBA8Unorm`, in OpenGL mode where `gl`.
fn descriptor(
    library: &Library,
    (vertex, fragment): (&str, Option<&str>),
    gl: bool,
) -> RenderPipelineDescriptor {
    let mut d = RenderPipelineDescriptor::new(stage(library, vertex));
    d.fragment = fragment.map(|name| stage(library, name));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    d.private.open_gl_mode_enabled = gl;
    d
}

/// The problem of Metal's error `made` answers, its description beside it.
fn problem(made: Result<RenderPipelineState, Error>) -> (Problem, String) {
    match made {
        Err(Error::MetalError(e)) => (e.problem, e.localized_description.to_string()),
        Err(other) => panic!("not Metal's error: {other:?}"),
        Ok(_) => panic!("made"),
    }
}

/// A 2 x 1 `RGBA32Float` texture holding [`TEXEL`] at (1, 0).
fn tint(device: &Device) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, 2, 1, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    let bytes: Vec<u8> = [0.0; 4]
        .iter()
        .chain(&TEXEL)
        .flat_map(|v: &f32| v.to_le_bytes())
        .collect();
    texture
        .replace_region(Region::new_2d(0, 0, 2, 1), 0, 0, &bytes, 32, 0)
        .unwrap();
    texture
}

/// The bytes of a [`SIDE`]-square `RGBA8Unorm` target cleared to zero after one draw of three
/// vertices by `state`, `tint` bound at vertex and fragment texture 0 and `out` at vertex buffer 0,
/// and of `depth` where given, cleared to 1.
fn drawn(
    device: &Device,
    state: &RenderPipelineState,
    (tint, out): (&Texture, &Buffer),
    depth: Option<&Texture>,
) -> Vec<u8> {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&d).unwrap();
    let mut pass = pass(&target);
    if let Some(depth) = depth {
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = crate::LoadAction::Clear;
        pass.depth_attachment.clear_depth = 1.0;
    }
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_texture(Some(tint), 0).unwrap();
    encoder.set_fragment_texture(Some(tint), 0).unwrap();
    encoder.set_vertex_buffer(Some(out), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let mut bytes = vec![0; (SIDE * SIDE * 4) as usize];
    target
        .get_bytes(
            &mut bytes,
            SIDE * 4,
            0,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    bytes
}

/// Without the mode, a fragment function writing depth without a depth attachment is refused
/// with Metal's error; with it the state is made, and its colors are those it draws with a depth
/// attachment.
#[test]
fn opengl_mode_makes_a_depth_output_without_a_depth_attachment() {
    let scratch = Scratch::new("opengl-mode-depth");
    let device = device(&scratch.0);
    let library = library(&device, "examples/icb_linked_arguments.ll");
    let functions = ("plain_vertex", Some("depth_writing_fragment"));
    let plain = PipelineOptions::default();
    let refused = device.new_render_pipeline_state(&descriptor(&library, functions, false), &plain);
    assert_eq!(
        problem(refused),
        (
            Problem::DepthWrittenWithoutAttachment,
            "depthAttachmentPixelFormat is not valid and shader writes to depth".to_owned()
        )
    );
    let state = device
        .new_render_pipeline_state(&descriptor(&library, functions, true), &plain)
        .unwrap();
    let mut attached = descriptor(&library, functions, false);
    attached.depth_format = Some(PixelFormat::Depth32Float);
    let attached = device.new_render_pipeline_state(&attached, &plain).unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET;
    let depth = device.new_texture(&d).unwrap();
    let tint = tint(&device);
    let out = device.new_buffer(48, ResourceOptions::shared()).unwrap();
    let without = drawn(&device, &state, (&tint, &out), None);
    let with = drawn(&device, &attached, (&tint, &out), Some(&depth));
    assert!(without.iter().any(|&b| b != 0), "the state drew nothing");
    assert_eq!(without, with);
}

/// Without the mode, a rasterizing state whose vertex function returns `void` is refused with
/// Metal's error, with or without a fragment function; with it the state is made, and its draw
/// runs the vertex function, whose stores land, and leaves the target as the pass cleared it.
#[test]
fn opengl_mode_makes_a_rasterizing_void_vertex_function_rasterize_nothing() {
    let scratch = Scratch::new("opengl-mode-void");
    let device = device(&scratch.0);
    let library = library(&device, "examples/icb_linked_arguments.ll");
    let plain = PipelineOptions::default();
    let tint = tint(&device);
    for fragment in [None, Some("inputless_fragment")] {
        let functions = ("void_store_vertex", fragment);
        let refused =
            device.new_render_pipeline_state(&descriptor(&library, functions, false), &plain);
        assert_eq!(
            problem(refused),
            (
                Problem::RasterizedVoidVertex,
                "RasterizationEnabled is true but the vertex shader's return type is void"
                    .to_owned()
            ),
            "{fragment:?}"
        );
        let state = device
            .new_render_pipeline_state(&descriptor(&library, functions, true), &plain)
            .unwrap();
        let out = device
            .new_buffer_with_bytes(&[0; 48], ResourceOptions::shared())
            .unwrap();
        let target = drawn(&device, &state, (&tint, &out), None);
        assert!(target.iter().all(|&b| b == 0), "{fragment:?} rasterized");
        let stored: Vec<u32> = TEXEL.iter().map(|v| v.to_bits()).collect();
        assert_eq!(words_of(&out), stored.repeat(3), "{fragment:?}");
    }
}
