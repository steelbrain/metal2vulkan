//! Color attachments a render pass and its draw's pipeline do not share on the device the cfg
//! selects, under the validation layer. Metal's validation stops each; plain Metal (Metal probes;
//! an M4 with macOS 27.0.1) draws them:
//!
//! - **A color of the pass the pipeline lacks** holds zero bytes at every sample a draw covers
//!   and its clear elsewhere: a 4² pass of `RGBA8Unorm` colors 0 and 1 (1 cleared green, both
//!   stored), the lower-left half drawn white into color 0 alone (`rt6_resolves`'s
//!   `rt6_resolve_half` and `rt6_resolve_white`), stores 0 in color 1 at exactly the texels it
//!   drew in color 0 and green at the rest. A later draw reads those zeros: the fullscreen
//!   triangle of `function_constant_interface`'s `fc_fragment_gated_color` (its second output
//!   on, red to color 1, blended one plus one) leaves red where the half drew and yellow where
//!   it did not. At 4x, the fullscreen triangle zeroes every sample of color 1, stored and
//!   resolved, and the half zeroes its covered samples, which the resolve averages with the
//!   green of the others; and in the consumer's pass (both colors 4x memoryless, color 0
//!   resolved, color 1 stored `DontCare`) color 0 resolves white. A fragment function that
//!   writes such a color itself leaves the same zeros there: `fc_fragment_gated_color` with its
//!   second output on (red to color 1) through a pipeline of color 0 alone, the half and, at 4x,
//!   the fullscreen triangle resolved (measured on an M3 Max under macOS 27.0.1 with
//!   `metal2vulkan-conformance/probes/written_color_the_pipeline_lacks.m`, over 8-bit, `half`,
//!   `uint` and `int` formats, written in their own numeric class and in another, cleared and
//!   loaded).
//! - **A color of the pipeline the pass lacks** is dropped: `fc_fragment_gated_color` with both
//!   outputs on, in a pass of color 0 alone, stores in color 0 what the same function with its
//!   second output off stores.
//! - **A state that rasterizes nothing** (`rasterizationEnabled` NO, so a `void` vertex function
//!   and no fragment stage) binds in a pass of any colors it does not name, at any sample count,
//!   and its draws store into none of them: it covers no sample. Metal's API validation accepts it
//!   and its stored bytes and visibility counts are those of the pass without the draw (measured
//!   on an M3 Max under macOS 27.0.1, through the consumer's Metal platform, over passes of 1 and
//!   8 colors of mixed formats, 2 and 4 samples, depth, stencil or both, a two-slice target,
//!   vertex amplification, two viewports, a memoryless attachment, imageblocks and a tile
//!   dispatch; empty draws alone). A non-empty draw of such a state is unmeasured, and draws
//!   nothing here either.
//! - **Refused** as Metal's validation refuses them: a color of the pass in another format than
//!   the pipeline's (plain Metal stores the pipeline's encoding into the pass's bytes, which no
//!   Vulkan attachment does), and a color of the pass a pipeline that rasterizes without a
//!   fragment function lacks (unmeasured).

use super::memoryless::memoryless_msaa;
use super::resolves::{lane, quiet, Lane, Shape, MSAA, ONE};
use crate::air_library::AirLibrary;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{BlendFactor, DataType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Attachment, BlitOption, Buffer, ClearColor, DrawProblem, Error, Invalid, Library, LoadAction,
    PipelineOptions, Problem, RenderPassDescriptor, RenderPipelineState, StoreAction, Texture,
};

/// The fullscreen triangle's depth.
const Z: f32 = 0.5;

fn green() -> ClearColor {
    ClearColor::new(0.0, 1.0, 0.0, 1.0)
}

fn black() -> ClearColor {
    ClearColor::new(0.0, 0.0, 0.0, 0.0)
}

/// The engine fixture `stem`'s library on `lane`'s device.
fn library(lane: &Lane, stem: &str) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    lane.device.new_library_with_air(&bitcode).unwrap()
}

/// What a pipeline of the lane draws: its vertex function (of `rt6_resolves`), its fragment
/// function (`None` for none; `fc_fragment_gated_color` with its second output on or off, or
/// `rt6_resolve_white`), its colors (blending color 1 one plus one where `blend`), and samples.
struct Pipe<'a> {
    vertex: &'a str,
    fragment: Option<Fragment>,
    colors: &'a [Option<PixelFormat>],
    blend: bool,
    samples: u32,
}

#[derive(Clone, Copy)]
enum Fragment {
    White,
    Gated { second: bool },
}

fn pipeline(lane: &Lane, pipe: &Pipe<'_>) -> Result<RenderPipelineState, Error> {
    let resolves = library(lane, "rt6_resolves");
    let vertex = resolves.new_function_with_name(pipe.vertex).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = match pipe.fragment {
        None => None,
        Some(Fragment::White) => Some(StageSource::Function(
            resolves
                .new_function_with_name("rt6_resolve_white")
                .unwrap()
                .function_descriptor(),
        )),
        Some(Fragment::Gated { second }) => {
            let constants = FunctionConstantValues::defaults().with_index(
                2,
                FunctionConstantValue::new(DataType::Bool, [u8::from(second)]),
            );
            let function = library(lane, "function_constant_interface")
                .new_function_with_name_constant_values("fc_fragment_gated_color", &constants)
                .unwrap();
            Some(StageSource::Function(function.function_descriptor()))
        }
    };
    for (index, format) in pipe.colors.iter().enumerate() {
        descriptor.color_attachments[index] = format.map(|pf| {
            let mut color = ColorAttachmentDescriptor::new(pf);
            if pipe.blend && index == 1 {
                color.blending_enabled = true;
                color.source_rgb_blend_factor = BlendFactor::One;
                color.destination_rgb_blend_factor = BlendFactor::One;
                color.source_alpha_blend_factor = BlendFactor::One;
                color.destination_alpha_blend_factor = BlendFactor::One;
            }
            color
        });
    }
    descriptor.raster_sample_count = pipe.samples;
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A pass of `targets` (the attachment of each index, its clear, its store, and its resolve).
fn pass(targets: &[(&Texture, ClearColor, StoreAction, Option<&Texture>)]) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    for (index, &(texture, clear, store, resolve)) in targets.iter().enumerate() {
        let target = &mut descriptor.color_attachments[index];
        target.texture = Some(texture.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = clear;
        target.store_action = store;
        target.resolve_texture = resolve.cloned();
    }
    descriptor
}

/// Draw `pipelines` in turn, each its triangle, in `descriptor`'s pass.
fn draw(lane: &Lane, descriptor: &RenderPassDescriptor, pipelines: &[&RenderPipelineState]) {
    lane.run(descriptor, |lane, encoder, held: &mut Vec<Buffer>| {
        let depth = super::buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
        encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        for pipeline in pipelines {
            encoder.set_render_pipeline_state(pipeline).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        held.push(depth);
    })
    .unwrap();
}

/// The texels of a 4² `RGBA8Unorm` texture.
fn texels(lane: &Lane, texture: &Texture) -> Vec<[u8; 4]> {
    lane.read(texture, (0, 0), 4, BlitOption::None)
        .chunks(4)
        .map(|t| [t[0], t[1], t[2], t[3]])
        .collect()
}

fn target(lane: &Lane, shape: Shape) -> Texture {
    lane.texture(PixelFormat::RGBA8Unorm, shape, TextureUsage::RENDER_TARGET)
}

const RGBA8: Option<PixelFormat> = Some(PixelFormat::RGBA8Unorm);

/// A color of the pass the pipeline lacks holds zero where a draw covers it, its clear elsewhere,
/// and a later draw reads the zeros (the module docs).
#[test]
fn a_pass_color_the_pipeline_lacks_is_zeroed_where_a_draw_covers_it() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let half = |samples| Pipe {
        vertex: "rt6_resolve_half",
        fragment: Some(Fragment::White),
        colors: &[RGBA8],
        blend: false,
        samples,
    };
    let full = |samples| Pipe {
        vertex: "rt6_resolve_full",
        ..half(samples)
    };
    let half1 = pipeline(&lane, &half(1)).unwrap();
    let later = pipeline(
        &lane,
        &Pipe {
            vertex: "rt6_resolve_full",
            fragment: Some(Fragment::Gated { second: true }),
            colors: &[RGBA8, RGBA8],
            blend: true,
            samples: 1,
        },
    )
    .unwrap();
    // The half alone: color 1 zero exactly where color 0 is white.
    let (color0, color1) = (target(&lane, ONE), target(&lane, ONE));
    draw(
        &lane,
        &pass(&[
            (&color0, black(), StoreAction::Store, None),
            (&color1, green(), StoreAction::Store, None),
        ]),
        &[&half1],
    );
    let drawn = texels(&lane, &color0);
    let zeroed = texels(&lane, &color1);
    let covered: Vec<bool> = drawn.iter().map(|t| *t == [0xff; 4]).collect();
    assert!(
        drawn.iter().all(|t| *t == [0xff; 4] || *t == [0; 4]),
        "color 0 on {name}: {drawn:02x?}"
    );
    let count = covered.iter().filter(|&&c| c).count();
    assert!(
        (6..=10).contains(&count),
        "the half on {name}: {drawn:02x?}"
    );
    for (i, (&c, t)) in covered.iter().zip(&zeroed).enumerate() {
        let want = if c { [0; 4] } else { [0, 0xff, 0, 0xff] };
        assert_eq!(*t, want, "color 1's texel {i} on {name}: {zeroed:02x?}");
    }
    // A later draw blends red over color 1: red over the zeros, yellow over the green.
    let (color0, color1) = (target(&lane, ONE), target(&lane, ONE));
    draw(
        &lane,
        &pass(&[
            (&color0, black(), StoreAction::Store, None),
            (&color1, green(), StoreAction::Store, None),
        ]),
        &[&half1, &later],
    );
    let blended = texels(&lane, &color1);
    for (i, (&c, t)) in covered.iter().zip(&blended).enumerate() {
        let want = if c {
            [0xff, 0, 0, 0xff]
        } else {
            [0xff, 0xff, 0, 0xff]
        };
        assert_eq!(
            *t, want,
            "the later draw's texel {i} on {name}: {blended:02x?}"
        );
    }
    // 4x: the fullscreen triangle zeroes every sample of color 1, stored and resolved.
    let full4 = pipeline(&lane, &full(4)).unwrap();
    let (msaa0, msaa1) = (target(&lane, MSAA), target(&lane, MSAA));
    let (resolved0, resolved1) = (target(&lane, ONE), target(&lane, ONE));
    draw(
        &lane,
        &pass(&[
            (
                &msaa0,
                black(),
                StoreAction::MultisampleResolve,
                Some(&resolved0),
            ),
            (
                &msaa1,
                green(),
                StoreAction::StoreAndMultisampleResolve,
                Some(&resolved1),
            ),
        ]),
        &[&full4],
    );
    assert_eq!(
        texels(&lane, &resolved0),
        vec![[0xff; 4]; 16],
        "4x on {name}"
    );
    assert_eq!(texels(&lane, &resolved1), vec![[0; 4]; 16], "4x on {name}");
    // 4x, the half: each pixel's covered samples zero, the others green, as color 0's are white
    // and black (a resolve rounds as the device does: within one unit).
    let half4 = pipeline(&lane, &half(4)).unwrap();
    let (msaa0, msaa1) = (target(&lane, MSAA), target(&lane, MSAA));
    let (resolved0, resolved1) = (target(&lane, ONE), target(&lane, ONE));
    draw(
        &lane,
        &pass(&[
            (
                &msaa0,
                black(),
                StoreAction::MultisampleResolve,
                Some(&resolved0),
            ),
            (
                &msaa1,
                green(),
                StoreAction::MultisampleResolve,
                Some(&resolved1),
            ),
        ]),
        &[&half4],
    );
    let whites = texels(&lane, &resolved0);
    let greens = texels(&lane, &resolved1);
    for (i, (w, g)) in whites.iter().zip(&greens).enumerate() {
        let sum = u32::from(w[1]) + u32::from(g[1]);
        assert!(
            (254..=256).contains(&sum) && g[0] == 0 && g[2] == 0 && g[1] == g[3],
            "the 4x half's texel {i} on {name}: {whites:02x?} {greens:02x?}"
        );
    }
    // The consumer's pass: both colors 4x memoryless, color 1 stored `DontCare`.
    let resolved0 = target(&lane, ONE);
    let memoryless0 = memoryless_msaa(&lane, PixelFormat::RGBA8Unorm);
    let memoryless1 = memoryless_msaa(&lane, PixelFormat::RGBA8Unorm);
    draw(
        &lane,
        &pass(&[
            (
                &memoryless0,
                black(),
                StoreAction::MultisampleResolve,
                Some(&resolved0),
            ),
            (&memoryless1, green(), StoreAction::DontCare, None),
        ]),
        &[&full4],
    );
    assert_eq!(
        texels(&lane, &resolved0),
        vec![[0xff; 4]; 16],
        "the consumer's pass on {name}"
    );
    eprintln!("a pass color the pipeline lacks on {name}: zeroed where covered, as Metal");
    quiet(lane, &messages);
}

/// A color of the pass the pipeline lacks holds the same zeros where the fragment function writes
/// it itself (the module docs): `fc_fragment_gated_color` with its second output on, red to
/// color 1, through a pipeline of color 0 alone.
#[test]
fn a_pass_color_the_pipeline_lacks_is_zeroed_where_the_function_writes_it() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let writing = |vertex, samples| Pipe {
        vertex,
        fragment: Some(Fragment::Gated { second: true }),
        colors: &[RGBA8],
        blend: false,
        samples,
    };
    let half = pipeline(&lane, &writing("rt6_resolve_half", 1)).unwrap();
    let (color0, color1) = (target(&lane, ONE), target(&lane, ONE));
    draw(
        &lane,
        &pass(&[
            (&color0, black(), StoreAction::Store, None),
            (&color1, green(), StoreAction::Store, None),
        ]),
        &[&half],
    );
    let drawn = texels(&lane, &color0);
    let zeroed = texels(&lane, &color1);
    let covered: Vec<bool> = drawn.iter().map(|t| t[3] == 0xff).collect();
    let count = covered.iter().filter(|&&c| c).count();
    assert!(
        (6..=10).contains(&count),
        "the half on {name}: {drawn:02x?}"
    );
    for (i, (&c, t)) in covered.iter().zip(&zeroed).enumerate() {
        let want = if c { [0; 4] } else { [0, 0xff, 0, 0xff] };
        assert_eq!(*t, want, "color 1's texel {i} on {name}: {zeroed:02x?}");
    }
    // 4x: the fullscreen triangle zeroes every sample of color 1, resolved.
    let full4 = pipeline(&lane, &writing("rt6_resolve_full", 4)).unwrap();
    let (msaa0, msaa1) = (target(&lane, MSAA), target(&lane, MSAA));
    let (resolved0, resolved1) = (target(&lane, ONE), target(&lane, ONE));
    draw(
        &lane,
        &pass(&[
            (
                &msaa0,
                black(),
                StoreAction::MultisampleResolve,
                Some(&resolved0),
            ),
            (
                &msaa1,
                green(),
                StoreAction::MultisampleResolve,
                Some(&resolved1),
            ),
        ]),
        &[&full4],
    );
    assert_eq!(texels(&lane, &resolved1), vec![[0; 4]; 16], "4x on {name}");
    eprintln!("a pass color the pipeline lacks, written on {name}: zeroed where covered, as Metal");
    quiet(lane, &messages);
}

/// A color of the pipeline the pass lacks is dropped: color 0 holds what it holds without it.
#[test]
fn a_pipeline_color_the_pass_lacks_drops_its_output() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let stored = |second: bool, colors: &[Option<PixelFormat>]| {
        let state = pipeline(
            &lane,
            &Pipe {
                vertex: "rt6_resolve_full",
                fragment: Some(Fragment::Gated { second }),
                colors,
                blend: false,
                samples: 1,
            },
        )
        .unwrap();
        let color0 = target(&lane, ONE);
        draw(
            &lane,
            &pass(&[(&color0, black(), StoreAction::Store, None)]),
            &[&state],
        );
        texels(&lane, &color0)
    };
    let alone = stored(false, &[RGBA8]);
    let dropped = stored(true, &[RGBA8, RGBA8]);
    assert_eq!(dropped, alone, "on {name}");
    assert!(alone.iter().all(|t| t[2] == 0 && t[3] == 0xff), "on {name}");
    eprintln!("a pipeline color the pass lacks on {name}: dropped, as Metal");
    quiet(lane, &messages);
}

/// A color of the pass in another format than the pipeline's, and one a pipeline without a
/// fragment function lacks, are refused at `set_render_pipeline_state`.
#[test]
fn colors_metal_stores_otherwise_are_refused() {
    let (messages, _cache, lane) = lane();
    let refused = |index| {
        Err(Error::Invalid(Invalid::new(
            "set_render_pipeline_state",
            Problem::Draw(DrawProblem::AttachmentFormat(Attachment::Color(index))),
        )))
    };
    let other = pipeline(
        &lane,
        &Pipe {
            vertex: "rt6_resolve_full",
            fragment: Some(Fragment::White),
            colors: &[Some(PixelFormat::RGBA16Float)],
            blend: false,
            samples: 1,
        },
    )
    .unwrap();
    let bare = pipeline(
        &lane,
        &Pipe {
            vertex: "rt6_resolve_full",
            fragment: None,
            colors: &[RGBA8],
            blend: false,
            samples: 1,
        },
    )
    .unwrap();
    let (color0, color1) = (target(&lane, ONE), target(&lane, ONE));
    lane.run(
        &pass(&[
            (&color0, black(), StoreAction::Store, None),
            (&color1, green(), StoreAction::Store, None),
        ]),
        |_, encoder, _| {
            assert_eq!(encoder.set_render_pipeline_state(&other), refused(0));
            assert_eq!(encoder.set_render_pipeline_state(&bare), refused(1));
        },
    )
    .unwrap();
    quiet(lane, &messages);
}

/// A state that rasterizes nothing binds in a pass of colours it does not name, at any sample
/// count, and its draws store into none of them (the module docs): a state of no colour, one
/// sample and a `void` vertex function binds in a one-colour pass of one sample and of four, and a
/// draw of no vertex leaves the pass's colour as its clear.
#[test]
fn a_state_that_rasterizes_nothing_binds_in_a_pass_of_any_color() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let vertex = library(&lane, "rvc_draw_bases")
        .new_function_with_name("draw_bases_in_a_vertex_function")
        .unwrap()
        .function_descriptor();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    descriptor.rasterization_enabled = false;
    let unrasterized = lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    // The void vertex function's buffer: a draw of no vertex never runs it, and never writes.
    let empty_draw = |lane: &Lane, descriptor: &RenderPassDescriptor| {
        lane.run(descriptor, |lane, encoder, held: &mut Vec<Buffer>| {
            let out = super::buffer_with(&lane.device, &[0; 512]).unwrap();
            encoder.set_vertex_buffer(Some(&out), 0, 0).unwrap();
            encoder.set_render_pipeline_state(&unrasterized).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0)
                .unwrap();
            held.push(out);
        })
        .unwrap();
    };
    // One sample: the pass's colour keeps its clear.
    let color = target(&lane, ONE);
    empty_draw(&lane, &pass(&[(&color, green(), StoreAction::Store, None)]));
    let cleared = vec![[0, 0xff, 0, 0xff]; 16];
    assert_eq!(texels(&lane, &color), cleared, "one sample on {name}");
    // Four samples, the state's one: it binds there too, and the resolve is the clear.
    let msaa = target(&lane, MSAA);
    let resolved = target(&lane, ONE);
    empty_draw(
        &lane,
        &pass(&[(
            &msaa,
            green(),
            StoreAction::MultisampleResolve,
            Some(&resolved),
        )]),
    );
    assert_eq!(texels(&lane, &resolved), cleared, "four samples on {name}");
    eprintln!("an unrasterized state of no color binds in both passes on {name}, storing nothing");
    quiet(lane, &messages);
}

/// Each state lists the colors its fragment function may write, from the function its constants
/// specialized:
/// - `fc_fragment_gated_color` with its second output on lists colors 0 and 1 where the pipeline
///   has a format at color 1, and color 0 alone where it has none, Metal discarding that write;
/// - with it off it lists color 0 alone, its constant gating the output off, either way;
/// - `rt6_resolve_white` lists color 0;
/// - a state without a fragment function lists none.
#[test]
fn a_state_lists_the_colors_its_fragment_function_writes() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let outputs = |fragment: Option<Fragment>, colors: &[Option<PixelFormat>]| {
        let state = pipeline(
            &lane,
            &Pipe {
                vertex: "rt6_resolve_full",
                fragment,
                colors,
                blend: false,
                samples: 1,
            },
        )
        .unwrap();
        state.fragment_color_outputs()
    };
    let gated = |second| Some(Fragment::Gated { second });
    for (colors, second) in [(&[RGBA8][..], &[0][..]), (&[RGBA8, RGBA8], &[0, 1])] {
        let n = colors.len();
        assert_eq!(outputs(gated(true), colors), second, "{name}: {n} colors");
        assert_eq!(outputs(gated(false), colors), [0], "{name}: {n} colors");
    }
    assert_eq!(outputs(Some(Fragment::White), &[RGBA8]), [0], "{name}");
    assert_eq!(outputs(None, &[RGBA8]), [0; 0], "{name}");
    quiet(lane, &messages);
}
