//! A guest's re-made depth and stencil texture, under a device cfg, on every device of the cfg's
//! kind: the calls a browser's GL layer makes when a page that drew a `Depth32Float_Stencil8`
//! texture's level 0 gives it more levels, then samples it.
//!
//! - A 32 x 32 `Depth32Float_Stencil8` texture of one level, `ShaderRead | RenderTarget |
//!   PixelFormatView` usage, Private, and a view of it of the same format over level 0 and slice
//!   0. One pass takes the view as its depth and its stencil attachment, both cleared and stored:
//!   the stencil cleared to 0x5a, and one draw of the owned `depth_fill` pair writing each
//!   quadrant's depth, 0.2, 0.4, 0.6 and 0.8 (left to right, then top to bottom), under an
//!   always-passing depth test that writes.
//! - One blit encoder copies the view's level 0 into level 0 of a new texture of six levels and
//!   the same usage and storage, then each of five new one-level textures, 16 x 16 down to 1 x 1,
//!   never written, into levels 1 to 5, and reads level 0's two planes back through the plane
//!   options: every texel its quadrant's depth, bit for bit, and 0x5a.
//! - A view of the six-level texture over its six levels and slice 0, swizzled red, zero, zero,
//!   one, is sampled at each texel's centre of level 0 by the owned `rrp_mirror_clamp` fixture:
//!   raw through `texture2d<float>` and a nearest sampler state (`rrp_mc_sample`), each answer
//!   (the quadrant's depth, 0, 0, 1); and through `depth2d<float>` and a sampler state comparing
//!   `LessEqual` at 0.5 (`rrp_mc_compare`), each answer 1 where 0.5 is at most the depth (0.6 and
//!   0.8) and 0 elsewhere.
//! - Recorded (`tests::layouts`): each copy into the six-level texture is one `vkCmdCopyImage`
//!   naming both aspects, from the one-level image (the view's) or a small one in
//!   `TRANSFER_SRC_OPTIMAL` into it in `TRANSFER_DST_OPTIMAL`; the run prints the formats, aspects
//!   and layouts recorded.
//!
//! `llvm-as` assembles the fixtures; without it the test skips visibly.

use crate::tests::command::on_device::{each_device, finished};
use crate::tests::draw::on_device::{floats, state, Buffers};
use crate::tests::layouts::{image_copies, Recording};
use crate::texture::TextureObject;
use crate::{
    BlitOption, Buffer, BufferLayout, CompareFunction, ComputePipelineDescriptor,
    ComputePipelineState, DepthStencilDescriptor, Device, DispatchType, FunctionDescriptor,
    IndexClass, Library, LoadAction, PixelFormat, PrimitiveType, RenderPassDescriptor,
    RenderPipelineDescriptor, SamplerDescriptor, SamplerState, StageSource, StorageMode,
    StoreAction, Texture, TextureDescriptor, TexturePlace, TextureSwizzle, TextureSwizzleChannels,
    TextureType, TextureUsage,
};
use ash::vk::{self, Handle};
use metal2vulkan::vk::{Format, ImageAspectFlags, ImageLayout};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

/// The texture's side and its levels.
const SIDE: u32 = 32;
const LEVELS: u32 = 6;
/// The quadrants' depths: top left, top right, bottom left, bottom right.
const QUADRANTS: [f32; 4] = [0.2, 0.4, 0.6, 0.8];
/// The stencil the pass clears to.
const STENCIL: u8 = 0x5a;
/// The comparison's reference.
const REFERENCE: f32 = 0.5;
/// The `float4`s each `rrp_mirror_clamp` lane writes; the answer with no offset is the first.
const PER_LANE: usize = 5;
const FORMAT: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// Texel `(x, y)`'s depth: its quadrant's.
fn depth(x: u32, y: u32) -> f32 {
    QUADRANTS[usize::from(y >= SIDE / 2) * 2 + usize::from(x >= SIDE / 2)]
}

/// The library of the owned fixture `file`, or `None` where `llvm-as` is missing.
fn library(file: &str) -> Option<Library> {
    match llvm_assemble(&engine_dir().join(file)) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
        Err(AssembleError::Missing(why)) => {
            println!("SKIPPED: {why}");
            None
        }
        Err(error) => panic!("{file}: {error:?}"),
    }
}

/// A Shared buffer's bytes, as the host reads them.
fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a Shared buffer");
    // SAFETY: the buffer is alive, `length` bytes long, and no GPU work uses it.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// The 32-bit floats of `bytes`.
fn floats_of(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}

fn level(level: u32) -> TexturePlace {
    TexturePlace {
        slice: 0,
        level,
        origin: [0; 3],
    }
}

/// `kernel` of `library`, its sampler at slot 0 a sampler state's.
fn kernel(name: &str, device: &Device, library: &Library, kernel: &str) -> ComputePipelineState {
    let function = library.function(kernel).expect("the kernel");
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    (device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
        .unwrap_or_else(|error| panic!("{name}: {kernel}: {error}"))
}

/// Panics, naming the first texel and how many differ, where `got` is not `want`.
fn exact<T: PartialEq + std::fmt::Debug>(name: &str, what: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{name}: {what}: the texels read");
    let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if let Some(&first) = differ.first() {
        let side = SIDE as usize;
        panic!(
            "{name}: {what}: {} of {} texels differ, the first at ({}, {}): {:?}, not {:?}",
            differ.len(),
            got.len(),
            first % side,
            first / side,
            got[first],
            want[first]
        );
    }
}

/// The route on `device`, its textures at indices from `first` and its samplers at `samplers`.
fn route(
    name: &str,
    device: &Device,
    (fill, sampling): (&Library, &Library),
    first: u32,
    samplers: [&SamplerState; 2],
) {
    use BlitOption::{DepthFromDepthStencil as Depth, StencilFromDepthStencil as Stencil};
    let handle = device.core().device.handle();
    let texture = |side: u32, levels: u32, index: u32| {
        let mut d = TextureDescriptor::new(TextureType::Type2D, FORMAT, side, side);
        d.mipmap_level_count = levels;
        d.usage = TextureUsage::SHADER_READ
            | TextureUsage::RENDER_TARGET
            | TextureUsage::PIXEL_FORMAT_VIEW;
        d.storage_mode = StorageMode::Private;
        (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"))
    };
    let parent = texture(SIDE, 1, first);
    let source =
        (parent.new_texture_view(FORMAT, TextureType::Type2D, 0..1, 0..1, Some(first + 1)))
            .unwrap_or_else(|error| panic!("{name}: the source view: {error}"));
    let destination = texture(SIDE, LEVELS, first + 2);
    let swizzle = TextureSwizzleChannels {
        red: TextureSwizzle::Red,
        green: TextureSwizzle::Zero,
        blue: TextureSwizzle::Zero,
        alpha: TextureSwizzle::One,
    };
    let sampled = (destination.new_texture_view_with_swizzle(
        FORMAT,
        TextureType::Type2D,
        0..LEVELS,
        0..1,
        swizzle,
        Some(first + 3),
    ))
    .unwrap_or_else(|error| panic!("{name}: the sampled view: {error}"));
    let smalls: Vec<Texture> = (1..LEVELS)
        .map(|n| texture(SIDE >> n, 1, first + 3 + n))
        .collect();

    let stage = |entry: &str| {
        StageSource::Function(FunctionDescriptor::new(
            fill.function(entry).expect("the function"),
        ))
    };
    let mut pipeline = RenderPipelineDescriptor::new(stage("depth_fill_vertex"));
    pipeline.fragment = Some(stage("depth_fill_fragment"));
    pipeline.depth_format = Some(FORMAT);
    pipeline.stencil_format = Some(FORMAT);
    let drawn = state(name, device, &pipeline);
    let mut always = DepthStencilDescriptor::default();
    always.depth_compare_function = CompareFunction::Always;
    always.depth_write_enabled = true;
    let writes = device.new_depth_stencil_state(&always);
    let raw = kernel(name, device, sampling, "rrp_mc_sample");
    let compare = kernel(name, device, sampling, "rrp_mc_compare");

    let texels = (SIDE * SIDE) as usize;
    let depths: Vec<f32> = (0..SIDE * SIDE)
        .map(|i| depth(i % SIDE, i / SIDE))
        .collect();
    let mut buffers = Buffers::sized(device, 1 << 20);
    let depth_buffer = buffers.of(device, &floats(&depths));
    let width = buffers.of(device, &SIDE.to_le_bytes());
    let centre = |c: u32| (c as f32 + 0.5) / SIDE as f32;
    let coords: Vec<f32> = (0..SIDE * SIDE)
        .flat_map(|i| [centre(i % SIDE), centre(i / SIDE), REFERENCE, 0.0])
        .collect();
    let coords = buffers.of(device, &floats(&coords));
    let unwritten = |len: usize| vec![0xee; len];
    let depth_out = buffers.of(device, &unwritten(4 * texels));
    let stencil_out = buffers.of(device, &unwritten(texels));
    let outs = [0, 1].map(|_| buffers.of(device, &unwritten(16 * PER_LANE * texels)));

    let queue = device.new_command_queue(4).expect("a queue");
    let command_buffer = queue.command_buffer().expect("a command buffer");
    let mut pass = RenderPassDescriptor::default();
    let z = &mut pass.depth_attachment;
    z.texture = Some(source.clone());
    z.load_action = LoadAction::Clear;
    z.store_action = StoreAction::Store;
    z.clear_depth = 1.0;
    let s = &mut pass.stencil_attachment;
    s.texture = Some(source.clone());
    s.load_action = LoadAction::Clear;
    s.store_action = StoreAction::Store;
    s.clear_stencil = u32::from(STENCIL);
    let ok = |result: Result<(), crate::Error>, what: &str| {
        result.unwrap_or_else(|error| panic!("{name}: {what}: {error}"))
    };
    let mut encoder = (command_buffer.render_command_encoder(&pass))
        .unwrap_or_else(|error| panic!("{name}: the pass: {error}"));
    ok(encoder.set_render_pipeline_state(&drawn), "the state");
    ok(
        encoder.set_fragment_buffer(Some(&depth_buffer), 0, 0),
        "the depths",
    );
    ok(encoder.set_fragment_buffer(Some(&width), 0, 1), "the width");
    ok(
        encoder.set_depth_stencil_state(Some(&writes)),
        "the depth state",
    );
    ok(
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
        "the draw",
    );
    encoder.end_encoding();

    let mut blit =
        (command_buffer.blit_command_encoder()).unwrap_or_else(|error| panic!("{name}: {error}"));
    let before = image_copies(handle).len();
    let whole = [SIDE, SIDE, 1];
    ok(
        blit.copy_from_texture(&source, level(0), whole, &destination, level(0)),
        "the copy into level 0",
    );
    for (n, small) in (1..LEVELS).zip(&smalls) {
        let side = SIDE >> n;
        let copy = blit.copy_from_texture(small, level(0), [side, side, 1], &destination, level(n));
        ok(copy, "a copy into a level past 0");
    }
    let tight = |bytes: u64| BufferLayout {
        offset: 0,
        bytes_per_row: bytes * u64::from(SIDE),
        bytes_per_image: bytes * u64::from(SIDE * SIDE),
    };
    ok(
        blit.copy_from_texture_to_buffer_with_options(
            &destination,
            level(0),
            whole,
            &depth_out,
            tight(4),
            Depth,
        ),
        "the depths out",
    );
    ok(
        blit.copy_from_texture_to_buffer_with_options(
            &destination,
            level(0),
            whole,
            &stencil_out,
            tight(1),
            Stencil,
        ),
        "the stencils out",
    );
    blit.end_encoding();

    let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    for ((state, sampler), out) in [&raw, &compare].into_iter().zip(samplers).zip(&outs) {
        ok(encoder.set_compute_pipeline_state(state), "the kernel");
        ok(encoder.set_sampler_state(Some(sampler), 0), "the sampler");
        ok(encoder.set_texture(Some(&sampled), 0), "the sampled view");
        ok(encoder.set_buffer(Some(&coords), 0, 0), "the lanes");
        ok(encoder.set_buffer(Some(out), 0, 1), "the answers");
        let lanes = SIDE * SIDE;
        ok(
            encoder.dispatch_threads([lanes, 1, 1], [64, 1, 1]),
            "the dispatch",
        );
    }
    encoder.end_encoding();
    command_buffer.commit().expect("a commit");
    finished(name, &command_buffer);
    assert_eq!(command_buffer.error(), None, "{name}");

    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    exact(
        name,
        "level 0's depth",
        &bits(&floats_of(&read(&depth_out))),
        &bits(&depths),
    );
    exact(
        name,
        "level 0's stencil",
        &read(&stencil_out),
        &vec![STENCIL; texels],
    );
    // Each lane's answer with no offset, as the bits of its four channels.
    let answers = |out: &Buffer| {
        let values = floats_of(&read(out));
        (0..texels)
            .map(|lane| {
                let at = 4 * PER_LANE * lane;
                bits(&values[at..at + 4])
            })
            .collect::<Vec<Vec<u32>>>()
    };
    let raw_want: Vec<Vec<u32>> = depths.iter().map(|&d| bits(&[d, 0.0, 0.0, 1.0])).collect();
    exact(name, "the raw samples", &answers(&outs[0]), &raw_want);
    let passes = |d: f32| if REFERENCE <= d { 1.0 } else { 0.0 };
    let compare_want: Vec<Vec<u32>> = (depths.iter())
        .map(|&d| bits(&[passes(d), 0.0, 0.0, 0.0]))
        .collect();
    exact(
        name,
        "the LessEqual comparisons",
        &answers(&outs[1]),
        &compare_want,
    );

    // What ran: the copies' commands into the six-level texture, as recorded.
    let images = [parent.image(), source.image(), destination.image()].map(|i| i.as_raw());
    let smalls_raw: Vec<u64> = smalls.iter().map(|t| t.image().as_raw()).collect();
    let recorded: Vec<_> = (image_copies(handle).into_iter().skip(before))
        .filter(|copy| copy.destination.as_raw() == images[2])
        .collect();
    let aspects = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
    let between = [
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    ];
    assert_eq!(
        images[0], images[1],
        "{name}: the view's image is its parent's"
    );
    assert_eq!(
        recorded.len(),
        LEVELS as usize,
        "{name}: one copy command a level"
    );
    for (n, copy) in recorded.iter().enumerate() {
        let from = match n {
            0 => images[0],
            _ => smalls_raw[n - 1],
        };
        assert_eq!(copy.source.as_raw(), from, "{name}: command {n}'s source");
        assert_eq!(
            copy.layouts.map(vk::ImageLayout::as_raw),
            between.map(vk::ImageLayout::as_raw),
            "{name}: command {n}'s layouts"
        );
        let masks: Vec<[u32; 2]> = (copy.aspects.iter())
            .map(|masks| masks.map(vk::ImageAspectFlags::as_raw))
            .collect();
        assert_eq!(
            masks,
            [[aspects.as_raw(); 2]],
            "{name}: command {n}'s aspects"
        );
    }
    let format = |texture: &Texture| Format::from_raw(texture.format().as_raw());
    let layout = |layout: vk::ImageLayout| ImageLayout::from_raw(layout.as_raw());
    println!(
        "{name}: a {SIDE} x {SIDE} view of one level ({:?}) rendered, copied into level 0 of a \
         {LEVELS}-level texture ({:?}), and levels 1 to 5 from one-level textures: {} \
         vkCmdCopyImage, each region {:?} from and to, {:?} to {:?}; level 0's {texels} depths \
         and stencils exact, its depth sampled through the (R, 0, 0, 1) view raw and compared \
         LessEqual {REFERENCE}, every texel exact",
        format(&source),
        format(&destination),
        recorded.len(),
        ImageAspectFlags::from_raw(aspects.as_raw()),
        layout(between[0]),
        layout(between[1]),
    );
}

#[test]
fn every_device_samples_a_rendered_depth_and_stencil_view_copied_into_level_0() {
    let (Some(fill), Some(sampling)) = (library("depth_fill.ll"), library("rrp_mirror_clamp.ll"))
    else {
        return;
    };
    let _recording = Recording::start();
    each_device("depth_stencil_route", |name, _, device| {
        let first = IndexClass::Texture.range().start;
        let _textures = (device.grant_indices(IndexClass::Texture, first, 3 + LEVELS))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let at = IndexClass::Sampler.range().start;
        let _samplers = (device.grant_indices(IndexClass::Sampler, at, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let plain = SamplerDescriptor::new();
        let mut less_equal = SamplerDescriptor::new();
        less_equal.compare_function = CompareFunction::LessEqual;
        let sampler = |d: &SamplerDescriptor, index| {
            (device.new_sampler_state(d, index)).unwrap_or_else(|error| panic!("{name}: {error}"))
        };
        let (plain, less_equal) = (sampler(&plain, at), sampler(&less_equal, at + 1));
        route(
            name,
            &device,
            (&fill, &sampling),
            first,
            [&plain, &less_equal],
        );
    });
}
