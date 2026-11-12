//! Device tests of texture views of another pixel format as attachments and resolve targets
//! on the device the cfg selects, under the validation layer.
//!
//! Metal renders into such a view in the view's format and encoding, and its bytes are the
//! texture's (a probe, an M3 Max on macOS 26.7 and an M4 on macOS 27.0.1 alike,
//! plain; API validation accepts them): an sRGB view of an `RGBA8Unorm` texture stores the sRGB
//! encoding of a draw's color and of a clear, and blends in linear; an `RGBA8Unorm` view of an
//! `RGBA8Unorm_sRGB` texture stores the unorm one; a `BGRA8Unorm`, `RG16Float` or `R32Uint` view
//! stores its own format's bytes; a view of one level and slice renders that level and slice; an
//! array view renders each layer of a layered pass; a multisampled view, or a multisampled
//! attachment, resolves into a view in its format; an `X32_Stencil8` view of a combined texture is
//! its stencil attachment. Each case clears a 4x4 target to (0.5, 0.25, 0.75, 1) and draws its left
//! half in (0.2, 0.4, 0.6, 0.8), as the probe did, and reads the bytes back: a texel a conversion
//! produced within one unit of Metal's (Vulkan leaves a unorm tie's rounding and the sRGB
//! encoding's to the device), every other byte exactly.
//!
//! A `BGRA8Unorm` view of an `RGBA8Unorm` texture made without `PixelFormatView` usage draws and
//! presents as plain Metal's does, over a texture and over buffers.
//!
//! The same probe's other attachments run here too: a texture buffer's `R32Uint` view; the depth
//! planes of 3D textures: a plane, a plane of another level, a view's plane, the planes of a
//! layered pass, and a resolve into a plane, every other plane kept; and 1D textures and 1D arrays,
//! which Metal renders plain and its validation stops.

use super::buffer_texture_blits::{Held, Kind, OFF};
use super::degrade_notices::counts;
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, PixelFormat, PrimitiveTopologyClass, PrimitiveType,
    StencilOperation, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Attachment, BlitCommandEncoder, BlitOption, Buffer, ClearColor, CommandBuffer,
    CommandBufferStatus, CommandQueue, CpuCacheMode, Degradation, Device, Error,
    HazardTrackingMode, HeapDescriptor, HeapType, Invalid, Library, LoadAction, Origin,
    PipelineOptions, Problem, Region, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ScissorRect, Size, StorageMode, StoreAction, Texture,
    TextureDescriptor, TextureProblem, ViewProblem,
};
use std::path::Path;

/// The probe's clear and draw colors.
const CLEAR: [f64; 4] = [0.5, 0.25, 0.75, 1.0];
const COLOR: [f32; 4] = [0.2, 0.4, 0.6, 0.8];
/// The probe's rectangle: the target's left half.
const LEFT: [f32; 4] = [-1.0, -1.0, 0.0, 1.0];
const SIDE: u64 = 4;
const PFV: TextureUsage = TextureUsage::RENDER_TARGET
    .union(TextureUsage::SHADER_READ)
    .union(TextureUsage::PIXEL_FORMAT_VIEW);
const RT: TextureUsage = TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ);
/// The consumer's usage of its `X32_Stencil8` texture (3).
const READ_WRITE: TextureUsage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);

/// The bytes Metal stored for the left half (the draw) and the right half (the clear) of each
/// texel row, by the view's format (the probe's `v_*` cases).
const SRGB: ([u8; 4], [u8; 4]) = ([0x7c, 0xaa, 0xcb, 0xcc], [0xbc, 0x89, 0xe1, 0xff]);
const UNORM: ([u8; 4], [u8; 4]) = ([0x33, 0x66, 0x99, 0xcc], [0x80, 0x40, 0xbf, 0xff]);
const BGRA_SRGB: ([u8; 4], [u8; 4]) = ([0xcb, 0xaa, 0x7c, 0xcc], [0xe1, 0x89, 0xbc, 0xff]);

/// The lane's device, queue, and the fixtures' functions (`tob_renders`: a rectangle from a
/// buffer's corners and a solid color from a buffer; `rt6_layered`: a layer-choosing triangle and
/// a fragment storing `tag << 8 | layer`).
struct Lane {
    device: Device,
    queue: CommandQueue,
    tob: Library,
    layered: Library,
}

impl Lane {
    fn new(device: &Device) -> Lane {
        let library = |name: &str| {
            let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/public/engine")
                .join(name);
            device
                .new_library_with_air(&llvm_assemble(&ll).unwrap())
                .unwrap()
        };
        Lane {
            queue: device.new_command_queue().unwrap(),
            tob: library("tob_renders.ll"),
            layered: library("rt6_layered.ll"),
            device: device.clone(),
        }
    }

    fn function(library: &Library, name: &str) -> StageSource {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    }

    /// The rectangle pipeline drawing a solid color into one attachment of `pf`, `samples`,
    /// blending source over destination by its alpha where `blend`.
    fn solid(&self, pf: PixelFormat, samples: u32, blend: bool) -> RenderPipelineState {
        let mut d = RenderPipelineDescriptor::new(Lane::function(&self.tob, "tob_rect"));
        d.fragment = Some(Lane::function(&self.tob, "tob_solid"));
        let mut color = ColorAttachmentDescriptor::new(pf);
        if blend {
            color.blending_enabled = true;
            color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            color.source_alpha_blend_factor = BlendFactor::One;
            color.destination_alpha_blend_factor = BlendFactor::Zero;
        }
        d.color_attachments[0] = Some(color);
        d.raster_sample_count = samples;
        self.device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap()
    }

    /// The layered pipeline into one `R32Uint` attachment.
    fn tagged(&self) -> RenderPipelineState {
        let mut d =
            RenderPipelineDescriptor::new(Lane::function(&self.layered, "rt6_layered_vertex"));
        d.fragment = Some(Lane::function(&self.layered, "rt6_layered_fragment"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        d.input_primitive_topology = PrimitiveTopologyClass::Triangle;
        self.device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap()
    }

    /// The rectangle pipeline with no fragment function, into an `X32_Stencil8` attachment.
    fn stencil_only(&self) -> RenderPipelineState {
        let mut d = RenderPipelineDescriptor::new(Lane::function(&self.tob, "tob_rect"));
        d.stencil_format = Some(PixelFormat::X32Stencil8);
        self.device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap()
    }

    fn floats(&self, values: &[f32]) -> Buffer {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        buffer_with(&self.device, &bytes).unwrap()
    }

    /// The left half drawn in `color` by `pipeline`.
    fn left(
        &self,
        e: &mut RenderCommandEncoder,
        held: &mut Vec<Buffer>,
        pipeline: &RenderPipelineState,
        color: [f32; 4],
    ) {
        e.set_render_pipeline_state(pipeline).unwrap();
        let corners = self.floats(&LEFT);
        let color = self.floats(&color);
        e.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&color), 0, 0).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        held.extend([corners, color]);
    }

    /// Encode `descriptor`'s pass by `body`, commit it and wait.
    fn run(
        &self,
        descriptor: &RenderPassDescriptor,
        body: impl FnOnce(&Lane, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    ) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
        let mut held = Vec::new();
        body(self, &mut encoder, &mut held);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
    }

    /// A `Shared` texture of `pf`, `usage` and `kind`.
    fn texture(&self, pf: PixelFormat, usage: TextureUsage, kind: Shape) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, kind.side, kind.side, kind.levels > 1);
        d.mipmap_level_count = kind.levels;
        if kind.slices > 1 {
            d.texture_type = TextureType::Type2DArray;
            d.array_length = kind.slices;
        }
        if kind.samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = kind.samples;
            d.resource_options = ResourceOptions::private();
        } else {
            d.resource_options = ResourceOptions::shared();
        }
        d.usage = usage;
        self.device.new_texture(&d).unwrap()
    }

    /// The bytes of `texture`'s 4x4 texels of 4 bytes at `level` and `slice`.
    fn bytes(&self, texture: &Texture, level: u64, slice: u64) -> Vec<u8> {
        let mut bytes = vec![0; (4 * SIDE * SIDE) as usize];
        texture
            .get_bytes(
                &mut bytes,
                4 * SIDE,
                0,
                Region::new_2d(0, 0, SIDE, SIDE),
                level,
                slice,
            )
            .unwrap();
        bytes
    }

    /// Encode `body`'s blits into one encoder, commit them and wait.
    fn blits(&self, body: impl FnOnce(&mut BlitCommandEncoder) -> Result<(), Error>) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        body(&mut blit).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// The stencil bytes of the combined texture `texture`, by a blit.
    fn stencil(&self, texture: &Texture) -> Vec<u8> {
        let length = SIDE * SIDE;
        let out = buffer_with(&self.device, &vec![0xee; length as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &out,
            0,
            SIDE,
            length,
            BlitOption::StencilFromDepthStencil,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        read_buffer(&out, 0, length).unwrap()
    }
}

/// `new_texture`'s refusal of a descriptor for `problem`.
fn texture_problem(problem: TextureProblem) -> Error {
    Error::Invalid(Invalid::new("new_texture", Problem::Texture(problem)))
}

/// A texture's shape.
#[derive(Clone, Copy)]
struct Shape {
    side: u64,
    levels: u64,
    slices: u64,
    samples: u64,
}

const PLAIN: Shape = Shape {
    side: SIDE,
    levels: 1,
    slices: 1,
    samples: 1,
};

/// A pass of one color attachment, `texture` at `slice` and `level`, cleared to [`CLEAR`] (or
/// loaded) and stored, resolving into `resolve` where given.
fn pass(
    texture: &Texture,
    (level, slice): (u64, u64),
    load: LoadAction,
    resolve: Option<&Texture>,
) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(texture.clone());
    target.level = level;
    target.slice = slice;
    target.load_action = load;
    let [r, g, b, a] = CLEAR;
    target.clear_color = ClearColor::new(r, g, b, a);
    target.store_action = match resolve {
        Some(_) => StoreAction::MultisampleResolve,
        None => StoreAction::Store,
    };
    target.resolve_texture = resolve.cloned();
    descriptor
}

/// Each row's left half `left` and right half `right`, of a 4x4 target.
fn halves((left, right): ([u8; 4], [u8; 4])) -> Vec<u8> {
    let row: Vec<u8> = [left, left, right, right].concat();
    row.repeat(SIDE as usize)
}

/// Check `got` against `want`, each byte within `slack` of Metal's.
fn check(got: &[u8], want: &[u8], slack: u8, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    if let Some(i) = (0..got.len()).find(|&i| got[i].abs_diff(want[i]) > slack) {
        panic!(
            "{what}: byte {i} is {:#04x}, Metal's {:#04x} (within {slack})",
            got[i], want[i]
        );
    }
}

fn on_device(test: impl FnOnce(&Lane)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&Lane::new(&device));
    }
    layer_reported_nothing_at_all(&messages);
}

/// A view of another format renders in that format's encoding into its texture's bytes: the
/// sRGB twins both ways (with and without `PixelFormatView` usage), `BGRA8Unorm_sRGB` over
/// `BGRA8Unorm`, `BGRA8Unorm` over `RGBA8Unorm`, `RG16Float` and `RGBA8Unorm` over `R32Uint`.
#[test]
fn views_render_in_their_format() {
    on_device(|lane| {
        let rgba = PixelFormat::RGBA8Unorm;
        // Blue drawn and red cleared, in `BGRA8Unorm`'s byte order.
        let blue_red = ([0xff, 0, 0, 0xff], [0, 0, 0xff, 0xff]);
        let half_floats = ([0x00, 0x34, 0xff, 0x7b], [0x00, 0x3e, 0x00, 0xc0]);
        type Case = (
            &'static str,
            PixelFormat,
            TextureUsage,
            PixelFormat,
            ([f64; 4], [f32; 4]),
            ([u8; 4], [u8; 4]),
            u8,
        );
        let ours = (CLEAR, COLOR);
        let cases: [Case; 7] = [
            (
                "v_srgb",
                rgba,
                RT,
                PixelFormat::RGBA8UnormSRGB,
                ours,
                SRGB,
                1,
            ),
            (
                "v_srgb_pfv",
                rgba,
                PFV,
                PixelFormat::RGBA8UnormSRGB,
                ours,
                SRGB,
                1,
            ),
            (
                "v_unorm_of_srgb",
                PixelFormat::RGBA8UnormSRGB,
                RT,
                rgba,
                ours,
                UNORM,
                1,
            ),
            (
                "v_bgra_srgb",
                PixelFormat::BGRA8Unorm,
                RT,
                PixelFormat::BGRA8UnormSRGB,
                ours,
                BGRA_SRGB,
                1,
            ),
            (
                "v_bgra",
                rgba,
                PFV,
                PixelFormat::BGRA8Unorm,
                ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
                blue_red,
                0,
            ),
            (
                "v_rg16f",
                PixelFormat::R32Uint,
                PFV,
                PixelFormat::RG16Float,
                ([1.5, -2.0, 0.0, 0.0], [0.25, 65504.0, 0.0, 0.0]),
                half_floats,
                0,
            ),
            (
                "v_rgba8_of_r32uint",
                PixelFormat::R32Uint,
                PFV,
                rgba,
                ours,
                UNORM,
                1,
            ),
        ];
        for (name, base, usage, view, (clear, color), want, slack) in cases {
            let texture = lane.texture(base, usage, PLAIN);
            let viewed = texture.new_texture_view(view).unwrap();
            let mut descriptor = pass(&viewed, (0, 0), LoadAction::Clear, None);
            let [r, g, b, a] = clear;
            descriptor.color_attachments[0].clear_color = ClearColor::new(r, g, b, a);
            let pipeline = lane.solid(view, 1, false);
            lane.run(&descriptor, |l, e, h| l.left(e, h, &pipeline, color));
            check(&lane.bytes(&texture, 0, 0), &halves(want), slack, name);
        }
    });
}

/// A blend through an sRGB view blends in linear: an `RGBA8Unorm` texture of (0x80, 0x40, 0xc0,
/// 0xff), loaded through its sRGB twin, takes (0.2, 0.4, 0.6, 0.5) by its alpha as Metal's did
/// (the probe's `v_srgb_load_blend`: 0x7e 0x83 0xc6 0x80).
#[test]
fn an_srgb_view_blends_in_linear() {
    on_device(|lane| {
        let texture = lane.texture(PixelFormat::RGBA8Unorm, RT, PLAIN);
        let old = [0x80, 0x40, 0xc0, 0xff];
        let filled = old.repeat((SIDE * SIDE) as usize);
        texture
            .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &filled, 4 * SIDE, 0)
            .unwrap();
        let viewed = texture
            .new_texture_view(PixelFormat::RGBA8UnormSRGB)
            .unwrap();
        let pipeline = lane.solid(PixelFormat::RGBA8UnormSRGB, 1, true);
        lane.run(&pass(&viewed, (0, 0), LoadAction::Load, None), |l, e, h| {
            l.left(e, h, &pipeline, [0.2, 0.4, 0.6, 0.5])
        });
        check(
            &lane.bytes(&texture, 0, 0),
            &halves(([0x7e, 0x83, 0xc6, 0x80], old)),
            1,
            "v_srgb_load_blend",
        );
    });
}

/// An `R32Uint` view of an `RGBA8Unorm` texture stores its words' bytes: a clear of 0x11223344
/// and a draw's word, as Metal's (the probe's `v_r32uint` and `v_no_pfv_r32uint`, whose draw
/// wrote 0xa1b2c3d4; here `rt6_layered`'s word for instance 0 in layer 0, 0x100, scissored to the
/// left half, in a pass of one layer, as its layer-writing function needs).
#[test]
fn a_uint_view_stores_its_words() {
    on_device(|lane| {
        let texture = lane.texture(PixelFormat::RGBA8Unorm, PFV, PLAIN);
        let viewed = texture.new_texture_view(PixelFormat::R32Uint).unwrap();
        let mut descriptor = pass(&viewed, (0, 0), LoadAction::Clear, None);
        descriptor.color_attachments[0].clear_color =
            ClearColor::new(f64::from(0x1122_3344u32), 0.0, 0.0, 0.0);
        descriptor.render_target_array_length = 1;
        let pipeline = lane.tagged();
        lane.run(&descriptor, |l, e, h| {
            e.set_render_pipeline_state(&pipeline).unwrap();
            let layers = buffer_with(&l.device, &0u32.to_le_bytes()).unwrap();
            e.set_vertex_buffer(Some(&layers), 0, 0).unwrap();
            e.set_scissor_rect(ScissorRect::new(0, 0, SIDE / 2, SIDE))
                .unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            h.push(layers);
        });
        check(
            &lane.bytes(&texture, 0, 0),
            &halves((0x100u32.to_le_bytes(), 0x1122_3344u32.to_le_bytes())),
            0,
            "v_r32uint",
        );
    });
}

/// A view of one level and slice renders them, and an array view a slice it is attached at,
/// leaving the rest (the probe's `v_subview_level_slice`: an 8x8 two-slice array of two
/// levels, viewed sRGB at level 1 and slice 1; `v_array_view_slice1`: a three-slice array's sRGB
/// view at slice 1).
#[test]
fn subviews_render_their_level_and_slice() {
    on_device(|lane| {
        let rgba = PixelFormat::RGBA8Unorm;
        let srgb = PixelFormat::RGBA8UnormSRGB;
        let pipeline = lane.solid(srgb, 1, false);
        let untouched = vec![0x11; (4 * SIDE * SIDE) as usize];
        let array = Shape {
            side: 2 * SIDE,
            levels: 2,
            slices: 2,
            samples: 1,
        };
        let texture = lane.texture(rgba, PFV, array);
        for slice in 0..2 {
            texture
                .replace_region(
                    Region::new_2d(0, 0, SIDE, SIDE),
                    1,
                    slice,
                    &untouched,
                    4 * SIDE,
                    0,
                )
                .unwrap();
        }
        let viewed = texture
            .new_texture_view_with_levels_slices(srgb, TextureType::Type2D, 1..2, 1..2)
            .unwrap();
        lane.run(
            &pass(&viewed, (0, 0), LoadAction::Clear, None),
            |l, e, h| l.left(e, h, &pipeline, COLOR),
        );
        check(
            &lane.bytes(&texture, 1, 0),
            &untouched,
            0,
            "level 1 slice 0",
        );
        check(
            &lane.bytes(&texture, 1, 1),
            &halves(SRGB),
            1,
            "v_subview_level_slice",
        );

        let three = Shape { slices: 3, ..PLAIN };
        let texture = lane.texture(rgba, RT, three);
        let zeros = vec![0; (4 * SIDE * SIDE) as usize];
        for slice in 0..3 {
            texture
                .replace_region(
                    Region::new_2d(0, 0, SIDE, SIDE),
                    0,
                    slice,
                    &zeros,
                    4 * SIDE,
                    0,
                )
                .unwrap();
        }
        let viewed = texture.new_texture_view(srgb).unwrap();
        lane.run(
            &pass(&viewed, (0, 1), LoadAction::Clear, None),
            |l, e, h| l.left(e, h, &pipeline, COLOR),
        );
        for slice in [0, 2] {
            check(
                &lane.bytes(&texture, 0, slice),
                &zeros,
                0,
                "slice untouched",
            );
        }
        check(
            &lane.bytes(&texture, 0, 1),
            &halves(SRGB),
            1,
            "v_array_view_slice1",
        );
    });
}

/// An array view renders each layer of a layered pass (the probe's `v_array_view_layered`,
/// two sRGB layers of an `RGBA8Unorm` array; here, with the layer-choosing fixture, an `R32Uint`
/// view of such an array, `rt6_layered`'s instance `i` into layer `i`, each slice holding its
/// instance's word as `v_r32uint` stores a word).
#[test]
fn an_array_view_renders_each_layer() {
    on_device(|lane| {
        let two = Shape { slices: 2, ..PLAIN };
        let texture = lane.texture(PixelFormat::RGBA8Unorm, PFV, two);
        let viewed = texture.new_texture_view(PixelFormat::R32Uint).unwrap();
        let mut descriptor = pass(&viewed, (0, 0), LoadAction::Clear, None);
        descriptor.render_target_array_length = 2;
        let pipeline = lane.tagged();
        lane.run(&descriptor, |l, e, h| {
            e.set_render_pipeline_state(&pipeline).unwrap();
            let layers: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
            let layers = buffer_with(&l.device, &layers).unwrap();
            e.set_vertex_buffer(Some(&layers), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 2, 0)
                .unwrap();
            h.push(layers);
        });
        for (slice, word) in [(0, 0x100u32), (1, 0x201)] {
            let want = word.to_le_bytes().repeat((SIDE * SIDE) as usize);
            check(&lane.bytes(&texture, 0, slice), &want, 0, "layer");
        }
    });
}

/// A multisampled attachment resolves into a view in its format, and a multisampled view
/// resolves into a texture of its format: the left half drawn and the right half cleared, every
/// sample alike.
#[test]
fn resolves_into_and_from_views() {
    on_device(|lane| {
        let rgba = PixelFormat::RGBA8Unorm;
        let srgb = PixelFormat::RGBA8UnormSRGB;
        let pipeline = lane.solid(srgb, 4, false);
        let msaa = Shape {
            samples: 4,
            ..PLAIN
        };

        let attachment = lane.texture(srgb, TextureUsage::RENDER_TARGET, msaa);
        let resolved = lane.texture(rgba, RT, PLAIN);
        let into = resolved.new_texture_view(srgb).unwrap();
        lane.run(
            &pass(&attachment, (0, 0), LoadAction::Clear, Some(&into)),
            |l, e, h| l.left(e, h, &pipeline, COLOR),
        );
        check(
            &lane.bytes(&resolved, 0, 0),
            &halves(SRGB),
            1,
            "v_resolve_into_srgb_view",
        );

        let attachment = lane.texture(
            rgba,
            TextureUsage::RENDER_TARGET.union(TextureUsage::PIXEL_FORMAT_VIEW),
            msaa,
        );
        let viewed = attachment.new_texture_view(srgb).unwrap();
        let resolved = lane.texture(srgb, RT, PLAIN);
        lane.run(
            &pass(&viewed, (0, 0), LoadAction::Clear, Some(&resolved)),
            |l, e, h| l.left(e, h, &pipeline, COLOR),
        );
        check(
            &lane.bytes(&resolved, 0, 0),
            &halves(SRGB),
            1,
            "v_ms_view_resolve",
        );
    });
}

/// An `X32_Stencil8` view of a `Depth32Float_Stencil8` texture is the stencil attachment of its
/// texture's stencil: cleared to 0x5a, and 0x77 replaced over the left half.
#[test]
fn a_stencil_view_attaches_its_texture_s_stencil() {
    on_device(|lane| {
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, SIDE, SIDE, false);
        d.usage = PFV;
        d.resource_options = ResourceOptions::private();
        let texture = lane.device.new_texture(&d).unwrap();
        let viewed = texture.new_texture_view(PixelFormat::X32Stencil8).unwrap();
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.stencil_attachment.texture = Some(viewed);
        descriptor.stencil_attachment.load_action = LoadAction::Clear;
        descriptor.stencil_attachment.clear_stencil = 0x5a;
        descriptor.stencil_attachment.store_action = StoreAction::Store;
        let mut replace = StencilDescriptor::default();
        replace.stencil_compare_function = CompareFunction::Always;
        replace.depth_stencil_pass_operation = StencilOperation::Replace;
        let mut state = DepthStencilDescriptor::default();
        state.front_face_stencil = replace;
        state.back_face_stencil = replace;
        let state = lane.device.new_depth_stencil_state(&state).unwrap();
        let pipeline = lane.stencil_only();
        lane.run(&descriptor, |l, e, h| {
            e.set_render_pipeline_state(&pipeline).unwrap();
            e.set_depth_stencil_state(&state).unwrap();
            e.set_stencil_reference_value(0x77);
            let corners = l.floats(&LEFT);
            e.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            h.push(corners);
        });
        let row = [0x77, 0x77, 0x5a, 0x5a];
        check(
            &lane.stencil(&texture),
            &row.repeat(SIDE as usize),
            0,
            "v_stencil_view",
        );
    });
}

/// A texture made in `X32_Stencil8` is a stencil texture, as plain Metal makes it (an M4 under
/// macOS 27.0.1; Metal's API validation stops it but for a heap's texture): made in 2D, 2D array
/// and cube types, `Private`, `Shared`, `Memoryless` and from a heap, of usage
/// `ShaderRead | ShaderWrite` (the consumer's) among others, and refused in 1D and 3D ones
/// (`x_mk_*`, `x_ty_*`); the stencil attachment cleared to 0x10 with 0x33 replaced over the left
/// half (`x_draw_replace`), and loaded and incremented (`x_draw_inc_load`); blits in and out at a
/// byte a texel (`x_blit_roundtrip`) and copies between it and `Stencil8`, itself and `R8Uint`
/// (`x_blit_*`); a `Shared` one's bytes through the CPU (`x_shared_rw`); a view in its own format,
/// and none in another (`x_view_*`, `v_x32_as_d32s8`). A texture of `X24_Stencil8` is stopped as of
/// no format, as an Apple GPU's Metal stops it.
#[test]
fn a_texture_made_in_x32_stencil8_is_a_stencil_texture() {
    on_device(|lane| {
        let x32 = PixelFormat::X32Stencil8;
        let made = |ty: TextureType, usage: TextureUsage, options: ResourceOptions| {
            let mut d = TextureDescriptor::texture_2d(x32, SIDE, SIDE, false);
            d.texture_type = ty;
            if ty == TextureType::Type3D {
                d.depth = SIDE;
            }
            if ty == TextureType::Type1D {
                d.height = 1;
            }
            if ty == TextureType::Type2DArray {
                d.array_length = 2;
            }
            d.usage = usage;
            d.resource_options = options;
            lane.device.new_texture(&d)
        };
        let private = ResourceOptions::private();
        let memoryless = ResourceOptions::new(
            StorageMode::Memoryless,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        for (ty, usage, options) in [
            (TextureType::Type2D, RT, private),
            (TextureType::Type2D, READ_WRITE, private),
            (
                TextureType::Type2D,
                TextureUsage::SHADER_READ,
                ResourceOptions::shared(),
            ),
            (TextureType::Type2D, TextureUsage::RENDER_TARGET, memoryless),
            (TextureType::Type2DArray, RT, private),
            (TextureType::Cube, RT, private),
        ] {
            let texture = made(ty, usage, options).unwrap();
            assert_eq!(texture.pixel_format(), x32, "{ty:?} {usage:?}");
            assert_eq!(texture.texture_type(), ty);
        }
        for ty in [TextureType::Type1D, TextureType::Type3D] {
            assert_eq!(
                made(ty, TextureUsage::SHADER_READ, private).err(),
                Some(texture_problem(TextureProblem::TypeForFormat)),
                "{ty:?}"
            );
        }
        let mut x24 = TextureDescriptor::texture_2d(PixelFormat::X24Stencil8, SIDE, SIDE, false);
        x24.resource_options = private;
        // Of no format on every venue: an Apple GPU's Metal stops it as it stops `Invalid`.
        assert_eq!(
            lane.device.new_texture(&x24).err(),
            Some(texture_problem(TextureProblem::NotAFormat))
        );

        // Views: its own format only (plain Metal views it as `Depth32Float_Stencil8` in its GPU's
        // own layout, `v_x32_as_d32s8`, and stops the rest).
        let viewed = made(
            TextureType::Type2D,
            RT | TextureUsage::PIXEL_FORMAT_VIEW,
            private,
        )
        .unwrap();
        assert_eq!(viewed.new_texture_view(x32).unwrap().pixel_format(), x32);
        for pf in [
            PixelFormat::Stencil8,
            PixelFormat::Depth32FloatStencil8,
            PixelFormat::R8Uint,
        ] {
            assert_eq!(
                viewed.new_texture_view(pf).err(),
                Some(Error::Invalid(Invalid::new(
                    "new_texture_view",
                    Problem::TextureView(ViewProblem::FormatIncompatible)
                ))),
                "a view as {pf:?}"
            );
        }

        // The stencil attachment: cleared, then the left half replaced.
        let texture = made(TextureType::Type2D, RT, private).unwrap();
        let state = |operation: StencilOperation| {
            let mut face = StencilDescriptor::default();
            face.stencil_compare_function = CompareFunction::Always;
            face.depth_stencil_pass_operation = operation;
            let mut state = DepthStencilDescriptor::default();
            state.front_face_stencil = face;
            state.back_face_stencil = face;
            lane.device.new_depth_stencil_state(&state).unwrap()
        };
        let pipeline = lane.stencil_only();
        let draw = |texture: &Texture, load: LoadAction, operation: StencilOperation| {
            let mut descriptor = RenderPassDescriptor::new();
            descriptor.stencil_attachment.texture = Some(texture.clone());
            descriptor.stencil_attachment.load_action = load;
            descriptor.stencil_attachment.clear_stencil = 0x10;
            descriptor.stencil_attachment.store_action = StoreAction::Store;
            let state = state(operation);
            lane.run(&descriptor, |l, e, h| {
                e.set_render_pipeline_state(&pipeline).unwrap();
                e.set_depth_stencil_state(&state).unwrap();
                e.set_stencil_reference_value(0x33);
                let corners = l.floats(&LEFT);
                e.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
                h.push(corners);
            });
        };
        draw(&texture, LoadAction::Clear, StencilOperation::Replace);
        let row = [0x33, 0x33, 0x10, 0x10];
        check(
            &lane.stencil(&texture),
            &row.repeat(SIDE as usize),
            0,
            "x_draw_replace",
        );
        // Loaded from a blit's bytes, the left half incremented.
        let counting: Vec<u8> = (0x41..0x51).collect();
        let source = buffer_with(&lane.device, &counting).unwrap();
        lane.blits(|blit| {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                SIDE,
                0,
                Size::new(SIDE, SIDE, 1),
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
        });
        assert_eq!(lane.stencil(&texture), counting, "x_blit_roundtrip packed");
        draw(&texture, LoadAction::Load, StencilOperation::IncrementClamp);
        let incremented: Vec<u8> = counting
            .iter()
            .enumerate()
            .map(|(i, &s)| if i % 4 < 2 { s + 1 } else { s })
            .collect();
        assert_eq!(lane.stencil(&texture), incremented, "x_draw_inc_load");

        // Blits at a byte a texel: rows 8 bytes apart in, a 2x2 region out at offset 3 in rows
        // of 5 bytes.
        let rows: Vec<u8> = (0x60..0x80).collect();
        let source = buffer_with(&lane.device, &rows).unwrap();
        let out = buffer_with(&lane.device, &[0xee; 32]).unwrap();
        lane.blits(|blit| {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                8,
                0,
                Size::new(SIDE, SIDE, 1),
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )?;
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(1, 1, 0),
                Size::new(2, 2, 1),
                &out,
                3,
                5,
                0,
                BlitOption::None,
            )
        });
        let mut want = vec![0xee; 32];
        want[3..5].copy_from_slice(&[0x69, 0x6a]);
        want[8..10].copy_from_slice(&[0x71, 0x72]);
        assert_eq!(read_buffer(&out, 0, 32).unwrap(), want, "x_blit_roundtrip");

        // Copies with `Stencil8`, itself and `R8Uint`, both ways where Metal copied.
        let usage = RT;
        let other = |pf: PixelFormat| {
            let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
            d.usage = usage;
            d.resource_options = private;
            lane.device.new_texture(&d).unwrap()
        };
        let counted = buffer_with(&lane.device, &counting).unwrap();
        for (from, to) in [
            (PixelFormat::Stencil8, x32),
            (x32, PixelFormat::Stencil8),
            (x32, x32),
            (x32, PixelFormat::R8Uint),
        ] {
            let (a, b) = (other(from), other(to));
            lane.blits(|blit| {
                blit.copy_from_buffer_to_texture(
                    &counted,
                    0,
                    SIDE,
                    0,
                    Size::new(SIDE, SIDE, 1),
                    &a,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )?;
                blit.copy_texture(&a, &b)
            });
            assert_eq!(lane.stencil(&b), counting, "{from:?} into {to:?}");
        }

        // A `Shared` one's bytes through the CPU.
        let shared = made(
            TextureType::Type2D,
            TextureUsage::SHADER_READ,
            ResourceOptions::shared(),
        )
        .unwrap();
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        shared
            .replace_region(region, 0, 0, &counting, SIDE, 0)
            .unwrap();
        let mut back = vec![0xee; counting.len()];
        shared.get_bytes(&mut back, SIDE, 0, region, 0, 0).unwrap();
        assert_eq!(back, counting, "x_shared_rw");

        // A heap's, which Metal's API validation takes.
        let heap = lane
            .device
            .new_heap(&HeapDescriptor::new(HeapType::Automatic, 1 << 20, private))
            .unwrap();
        let mut d = TextureDescriptor::texture_2d(x32, SIDE, SIDE, false);
        d.usage = RT;
        d.resource_options = private;
        let placed = heap.new_texture(&d).unwrap();
        lane.blits(|blit| {
            blit.copy_from_buffer_to_texture(
                &counted,
                0,
                SIDE,
                0,
                Size::new(SIDE, SIDE, 1),
                &placed,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
        });
        assert_eq!(lane.stencil(&placed), counting, "x_heap");
    });
}

/// An sRGB view of a texture over a buffer renders its encoding into the buffer's texel bytes and
/// leaves the bytes between its rows, over an ordinary and a no-copy buffer (the aliasing image or
/// the shadow; Metal probe case `r.view_srgb` on the M3 Max and an M4, and our probe's values).
/// A render pipeline whose attachment is in a pixel format Metal describes none of is stopped, as
/// plain Metal's descriptor validation stops it ("is not a valid MTLPixelFormat"; on an Apple M4
/// under macOS 27.0.1), on every venue: a color attachment of `RGB8Unorm`, blended or not; a
/// depth or a stencil attachment of `Depth24Unorm_Stencil8` or `X24_Stencil8`, which an Apple GPU
/// has neither of, whatever the Vulkan driver has; and a color attachment of each of 630 to 650
/// the release does not describe (`Device::pixel_format_info`), blended or not.
#[test]
fn an_attachment_in_a_format_metal_describes_none_of_stops_the_pipeline() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let stopped = |attachment, format| {
            Err(Error::Invalid(Invalid::new(
                "new_render_pipeline_state",
                Problem::AttachmentNotAFormat { attachment, format },
            )))
        };
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let lane = Lane::new(&device);
        let made = |lane: &Lane, d: &RenderPipelineDescriptor| {
            lane.device
                .new_render_pipeline_state(d, &PipelineOptions::default())
                .map(|_| ())
        };
        let rect =
            |lane: &Lane| RenderPipelineDescriptor::new(Lane::function(&lane.tob, "tob_rect"));
        let colored = |lane: &Lane, pf: PixelFormat, blend: bool| {
            let mut d = rect(lane);
            d.fragment = Some(Lane::function(&lane.tob, "tob_solid"));
            let mut color = ColorAttachmentDescriptor::new(pf);
            color.blending_enabled = blend;
            d.color_attachments[0] = Some(color);
            d
        };
        for blend in [false, true] {
            let rgb8 = PixelFormat::RGB8Unorm;
            assert_eq!(
                made(&lane, &colored(&lane, rgb8, blend)),
                stopped(Attachment::Color(0), rgb8),
                "blend {blend}"
            );
        }
        for pf in [PixelFormat::Depth24UnormStencil8, PixelFormat::X24Stencil8] {
            let mut depth = rect(&lane);
            depth.depth_format = Some(pf);
            assert_eq!(
                made(&lane, &depth),
                stopped(Attachment::Depth, pf),
                "{pf:?}"
            );
            let mut stencil = rect(&lane);
            stencil.stencil_format = Some(pf);
            assert_eq!(
                made(&lane, &stencil),
                stopped(Attachment::Stencil, pf),
                "{pf:?}"
            );
        }
        let from_630: Vec<PixelFormat> = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|pf| (630..=650).contains(&pf.as_raw()))
            .collect();
        assert_eq!(from_630.len(), 21);
        for &pf in &from_630 {
            let described = device
                .pixel_format_info(u64::from(pf.as_raw()))
                .describes_a_format();
            for blend in [false, true] {
                let answer = made(&lane, &colored(&lane, pf, blend));
                let stop = stopped(Attachment::Color(0), pf);
                if described {
                    assert_ne!(answer, stop, "{pf:?} blend {blend}");
                } else {
                    assert_eq!(answer, stop, "{pf:?} blend {blend}");
                }
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn an_srgb_view_of_a_texture_over_a_buffer_renders_its_bytes() {
    on_device(|lane| {
        const BPR: u64 = 32;
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            let Some(held) = Held::new(&lane.device, kind) else {
                eprintln!("{}: no {kind:?} buffer", lane.device.name());
                continue;
            };
            let texture = held.texture_of(PixelFormat::RGBA8Unorm, PFV, (OFF, BPR), [SIDE, SIDE]);
            let viewed = texture
                .new_texture_view(PixelFormat::RGBA8UnormSRGB)
                .unwrap();
            let pipeline = lane.solid(PixelFormat::RGBA8UnormSRGB, 1, false);
            lane.run(
                &pass(&viewed, (0, 0), LoadAction::Clear, None),
                |l, e, h| l.left(e, h, &pipeline, COLOR),
            );
            let mut want = held.model.clone();
            let mut slack = vec![0u8; want.len()];
            let row = halves(SRGB);
            for y in 0..SIDE {
                let at = (OFF + y * BPR) as usize;
                let n = (4 * SIDE) as usize;
                want[at..at + n].copy_from_slice(&row[..n]);
                slack[at..at + n].fill(1);
            }
            let got = held.bytes();
            if let Some(i) = (0..got.len()).find(|&i| got[i].abs_diff(want[i]) > slack[i]) {
                panic!(
                    "{kind:?}: byte {i} is {:#04x}, Metal's {:#04x}",
                    got[i], want[i]
                );
            }
        }
    });
}

/// A `BGRA8Unorm` view of an `RGBA8Unorm` texture made without `PixelFormatView` usage, which plain
/// Metal makes and presents frames through (its API validation stops it; Metal probes on the M3 Max
/// and the M4): a draw through the view, red cleared and blue drawn, stores `BGRA8Unorm`'s bytes in
/// the texture, and a present-like blit of the view into a `BGRA8Unorm` texture copies them; for an
/// ordinary texture, and for a texture over an ordinary and a no-copy buffer (the consumer's
/// frame), whose bytes between rows stay.
#[test]
fn a_bgra_view_without_pixel_format_view_draws_and_presents() {
    on_device(|lane| {
        const BPR: u64 = 32;
        let want = halves(([0xff, 0, 0, 0xff], [0, 0, 0xff, 0xff]));
        let pipeline = lane.solid(PixelFormat::BGRA8Unorm, 1, false);
        let draw = |viewed: &Texture| {
            let mut descriptor = pass(viewed, (0, 0), LoadAction::Clear, None);
            descriptor.color_attachments[0].clear_color = ClearColor::new(1.0, 0.0, 0.0, 1.0);
            lane.run(&descriptor, |l, e, h| {
                l.left(e, h, &pipeline, [0.0, 0.0, 1.0, 1.0])
            });
        };
        let present = |viewed: &Texture| {
            let drawable = lane.texture(PixelFormat::BGRA8Unorm, RT, PLAIN);
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let at = Origin::new(0, 0, 0);
            blit.copy_from_texture_to_texture(
                viewed,
                0,
                0,
                at,
                Size::new(SIDE, SIDE, 1),
                &drawable,
                0,
                0,
                at,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            lane.bytes(&drawable, 0, 0)
        };
        let texture = lane.texture(PixelFormat::RGBA8Unorm, RT, PLAIN);
        let viewed = texture.new_texture_view(PixelFormat::BGRA8Unorm).unwrap();
        draw(&viewed);
        check(&lane.bytes(&texture, 0, 0), &want, 0, "a texture");
        check(&present(&viewed), &want, 0, "a texture's view presented");
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            let Some(held) = Held::new(&lane.device, kind) else {
                eprintln!("{}: no {kind:?} buffer", lane.device.name());
                continue;
            };
            let texture = held.texture_of(PixelFormat::RGBA8Unorm, RT, (OFF, BPR), [SIDE, SIDE]);
            let viewed = texture.new_texture_view(PixelFormat::BGRA8Unorm).unwrap();
            draw(&viewed);
            let mut model = held.model.clone();
            let n = (4 * SIDE) as usize;
            for (y, row) in want.chunks(n).enumerate() {
                let at = (OFF + y as u64 * BPR) as usize;
                model[at..at + n].copy_from_slice(row);
            }
            assert!(
                held.bytes() == model,
                "{kind:?}: the buffer's bytes are not Metal's"
            );
            let what = format!("{kind:?} buffer's texture's view presented");
            check(&present(&viewed), &want, 0, &what);
        }
    });
}

/// An `R32Uint` view of a texture buffer is an attachment and stores its words in the buffer's
/// bytes, every other byte kept (the probe's `tb_view_r32uint`: a 16-texel `RGBA8Unorm`
/// texture buffer's view cleared to 0x11223344 and its left half drawn a word; here
/// `rt6_layered`'s word 0x100, scissored to the left 8 texels), over an ordinary and a no-copy
/// buffer.
#[test]
fn a_uint_view_of_a_texture_buffer_stores_its_words() {
    on_device(|lane| {
        for kind in [Kind::Ordinary, Kind::NoCopy] {
            let Some(held) = Held::new(&lane.device, kind) else {
                eprintln!("{}: no {kind:?} buffer", lane.device.name());
                continue;
            };
            let texture_buffer = held.texture_buffer_of(PFV, OFF, 16);
            let viewed = texture_buffer
                .new_texture_view(PixelFormat::R32Uint)
                .unwrap();
            let mut descriptor = pass(&viewed, (0, 0), LoadAction::Clear, None);
            descriptor.color_attachments[0].clear_color =
                ClearColor::new(f64::from(0x1122_3344u32), 0.0, 0.0, 0.0);
            descriptor.render_target_array_length = 1;
            let pipeline = lane.tagged();
            lane.run(&descriptor, |l, e, h| {
                e.set_render_pipeline_state(&pipeline).unwrap();
                let layers = buffer_with(&l.device, &0u32.to_le_bytes()).unwrap();
                e.set_vertex_buffer(Some(&layers), 0, 0).unwrap();
                e.set_scissor_rect(ScissorRect::new(0, 0, 8, 1)).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
                h.push(layers);
            });
            let mut want = held.model.clone();
            let words: Vec<u8> = [0x100u32; 8]
                .into_iter()
                .chain([0x1122_3344u32; 8])
                .flat_map(u32::to_le_bytes)
                .collect();
            let at = OFF as usize;
            want[at..at + words.len()].copy_from_slice(&words);
            check(
                &held.bytes(),
                &want,
                0,
                &format!("tb_view_r32uint over a {kind:?} buffer"),
            );
        }
    });
}

/// A `Shared` 3D texture of `pf` and `usage`, `side` wide and high, `depth` deep, of `levels`,
/// its first level's texels (or `level`'s) all `fill`.
fn volume(
    lane: &Lane,
    pf: PixelFormat,
    usage: TextureUsage,
    [side, depth, levels]: [u64; 3],
    (level, fill): (u64, u8),
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, side, side, false);
    d.texture_type = TextureType::Type3D;
    d.depth = depth;
    d.mipmap_level_count = levels;
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    let texture = lane.device.new_texture(&d).unwrap();
    let [w, h, z] = [side >> level, side >> level, (depth >> level).max(1)];
    let bytes = vec![fill; (4 * w * h * z) as usize];
    texture
        .replace_region(
            Region::new_3d(0, 0, 0, w, h, z),
            level,
            0,
            &bytes,
            4 * w,
            4 * w * h,
        )
        .unwrap();
    texture
}

/// The texels of `texture`'s `level`, 4x4 planes of 4 bytes, `planes` deep.
fn planes(texture: &Texture, level: u64, planes: u64) -> Vec<u8> {
    let mut bytes = vec![0; (64 * planes) as usize];
    texture
        .get_bytes(
            &mut bytes,
            16,
            64,
            Region::new_3d(0, 0, 0, 4, 4, planes),
            level,
            0,
        )
        .unwrap();
    bytes
}

/// What Metal left in a 4x4x4 volume filled with 0x11 whose planes `drawn` a pass cleared and
/// drew the left half of, in `encoding`'s bytes.
fn drawn_planes(drawn: &[u64], encoding: ([u8; 4], [u8; 4])) -> Vec<u8> {
    (0..4)
        .flat_map(|plane| match drawn.contains(&plane) {
            true => halves(encoding),
            false => vec![0x11; 64],
        })
        .collect()
}

/// A 3D texture's depth plane is an attachment, every other plane kept (an M3 Max and an M4, API
/// validation passing each): plane 2 of a 4x4x4 `RGBA8Unorm` volume filled with 0x11
/// (`t3d_plane2`), plane 1 of level 1 of an 8x8x8 one (`t3d_level1_plane1`), and plane 1 through
/// an sRGB view (`t3d_view_srgb_plane1`). Its resolves: [`a_resolve_writes_the_volume_plane_it_names`].
#[test]
fn a_volume_s_plane_is_an_attachment() {
    on_device(|lane| {
        let rgba = PixelFormat::RGBA8Unorm;
        let solid = lane.solid(rgba, 1, false);
        let texture = volume(lane, rgba, RT, [4, 4, 1], (0, 0x11));
        let mut descriptor = pass(&texture, (0, 0), LoadAction::Clear, None);
        descriptor.color_attachments[0].depth_plane = 2;
        lane.run(&descriptor, |l, e, h| l.left(e, h, &solid, COLOR));
        check(
            &planes(&texture, 0, 4),
            &drawn_planes(&[2], UNORM),
            1,
            "t3d_plane2",
        );

        let texture = volume(lane, rgba, RT, [8, 8, 2], (1, 0x11));
        let mut descriptor = pass(&texture, (1, 0), LoadAction::Clear, None);
        descriptor.color_attachments[0].depth_plane = 1;
        lane.run(&descriptor, |l, e, h| l.left(e, h, &solid, COLOR));
        check(
            &planes(&texture, 1, 4),
            &drawn_planes(&[1], UNORM),
            1,
            "t3d_level1_plane1",
        );

        let texture = volume(lane, rgba, RT, [4, 4, 1], (0, 0x11));
        let viewed = texture
            .new_texture_view(PixelFormat::RGBA8UnormSRGB)
            .unwrap();
        let srgb = lane.solid(PixelFormat::RGBA8UnormSRGB, 1, false);
        let mut descriptor = pass(&viewed, (0, 0), LoadAction::Clear, None);
        descriptor.color_attachments[0].depth_plane = 1;
        lane.run(&descriptor, |l, e, h| l.left(e, h, &srgb, COLOR));
        check(
            &planes(&texture, 0, 4),
            &drawn_planes(&[1], SRGB),
            1,
            "t3d_view_srgb_plane1",
        );
    });
}

/// A 3D texture's depth plane is a resolve target, every other plane kept: a 4x multisampled
/// `RGBA8Unorm` attachment cleared and its left half drawn, resolved into each plane of a 4x4x4
/// volume filled with 0x11, one volume a plane (an M3 Max and an M4 measured plane 3,
/// `t3d_resolve_plane3`, API validation passing; `resolveDepthPlane` names the one plane the
/// resolve writes). Prints the planes each resolve wrote (`VOLUME-RESOLVE` lines); the layer's
/// reports fail it before the bytes are checked.
#[test]
fn a_resolve_writes_the_volume_plane_it_names() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let resolved = {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let lane = Lane::new(&device);
        let rgba = PixelFormat::RGBA8Unorm;
        let msaa = lane.texture(
            rgba,
            TextureUsage::RENDER_TARGET,
            Shape {
                samples: 4,
                ..PLAIN
            },
        );
        let multisampled = lane.solid(rgba, 4, false);
        let resolved: Vec<(u64, Vec<u8>)> = (0..4)
            .map(|plane| {
                let texture = volume(&lane, rgba, RT, [4, 4, 1], (0, 0x11));
                let mut descriptor = pass(&msaa, (0, 0), LoadAction::Clear, Some(&texture));
                descriptor.color_attachments[0].resolve_depth_plane = plane;
                lane.run(&descriptor, |l, e, h| l.left(e, h, &multisampled, COLOR));
                let bytes = planes(&texture, 0, 4);
                let written: Vec<usize> = (0..4)
                    .filter(|&at| bytes[64 * at..64 * (at + 1)].iter().any(|&b| b != 0x11))
                    .collect();
                eprintln!(
                    "VOLUME-RESOLVE on {}: into plane {plane}, planes written {written:?}",
                    device.name()
                );
                (plane, bytes)
            })
            .collect();
        resolved
    };
    layer_reported_nothing_at_all(&messages);
    for (plane, bytes) in resolved {
        let what = format!("t3d_resolve_plane{plane}");
        check(&bytes, &drawn_planes(&[plane], UNORM), 1, &what);
    }
}

/// A layered pass renders a 3D attachment's planes from its depth plane, instance `i` into the
/// pass's layer `i` (the probe's `t3d_layered_from1`: planes 1 and 2 of a 4x4x4 volume, the
/// others kept, API validation passing; here, with the layer-choosing fixture, an `R32Uint`
/// volume of 0x11111111 whose planes take `rt6_layered`'s words 0x100 and 0x201).
#[test]
fn a_layered_pass_renders_a_volume_s_planes() {
    on_device(|lane| {
        let texture = volume(lane, PixelFormat::R32Uint, RT, [4, 4, 1], (0, 0x11));
        let mut descriptor = pass(&texture, (0, 0), LoadAction::Clear, None);
        descriptor.color_attachments[0].depth_plane = 1;
        descriptor.render_target_array_length = 2;
        let pipeline = lane.tagged();
        lane.run(&descriptor, |l, e, h| {
            e.set_render_pipeline_state(&pipeline).unwrap();
            let layers: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
            let layers = buffer_with(&l.device, &layers).unwrap();
            e.set_vertex_buffer(Some(&layers), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 2, 0)
                .unwrap();
            h.push(layers);
        });
        let want: Vec<u8> = [None, Some(0x100u32), Some(0x201), None]
            .into_iter()
            .flat_map(|word| match word {
                Some(word) => word.to_le_bytes().repeat(16),
                None => vec![0x11; 64],
            })
            .collect();
        check(&planes(&texture, 0, 4), &want, 0, "t3d_layered_from1");
    });
}

/// A `Shared` 16-texel 1D texture of `pf` and `usage` (a 1D array of `slices` past one).
fn line(lane: &Lane, pf: PixelFormat, usage: TextureUsage, slices: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, 16, 1, false);
    d.texture_type = match slices {
        1 => TextureType::Type1D,
        _ => TextureType::Type1DArray,
    };
    d.array_length = slices;
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    lane.device.new_texture(&d).unwrap()
}

/// The 16 texels of `texture`'s `slice`.
fn row(texture: &Texture, slice: u64) -> Vec<u8> {
    let mut bytes = vec![0; 64];
    texture
        .get_bytes(&mut bytes, 64, 0, Region::new_1d(0, 16), 0, slice)
        .unwrap();
    bytes
}

/// A 16-texel row whose left 8 texels are `left` and right 8 `right`.
fn row_halves((left, right): ([u8; 4], [u8; 4])) -> Vec<u8> {
    [left.repeat(8), right.repeat(8)].concat()
}

/// A 1D texture is an attachment and a resolve target, its row rendered as a 2D target of one
/// row's would be: Metal renders and resolves into one plain, and its validation stops the pass
/// ("texture must not be MTLTextureType1D", "resolveTexture must not be MTLTextureType1D")
/// (an M3 Max and an M4: `t1d`, `t1d_array_slice1`, `t1d_resolve`): a 16-texel
/// `RGBA8Unorm` texture cleared and its left half drawn, slice 1 of a two-slice 1D array (slice
/// 0 kept), and the resolve of a 16x1 4x multisampled attachment.
#[test]
fn a_line_is_an_attachment() {
    on_device(|lane| {
        let rgba = PixelFormat::RGBA8Unorm;
        let solid = lane.solid(rgba, 1, false);
        let texture = line(lane, rgba, RT, 1);
        lane.run(
            &pass(&texture, (0, 0), LoadAction::Clear, None),
            |l, e, h| l.left(e, h, &solid, COLOR),
        );
        check(&row(&texture, 0), &row_halves(UNORM), 1, "t1d");

        let texture = line(lane, rgba, RT, 2);
        let zeros = vec![0; 64];
        for slice in 0..2 {
            texture
                .replace_region(Region::new_1d(0, 16), 0, slice, &zeros, 64, 0)
                .unwrap();
        }
        lane.run(
            &pass(&texture, (0, 1), LoadAction::Clear, None),
            |l, e, h| l.left(e, h, &solid, COLOR),
        );
        check(&row(&texture, 0), &zeros, 0, "t1d_array_slice1 slice 0");
        check(&row(&texture, 1), &row_halves(UNORM), 1, "t1d_array_slice1");

        let texture = line(lane, rgba, RT, 1);
        let mut d = TextureDescriptor::texture_2d(rgba, 16, 1, false);
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = 4;
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        let msaa = lane.device.new_texture(&d).unwrap();
        let multisampled = lane.solid(rgba, 4, false);
        lane.run(
            &pass(&msaa, (0, 0), LoadAction::Clear, Some(&texture)),
            |l, e, h| l.left(e, h, &multisampled, COLOR),
        );
        check(&row(&texture, 0), &row_halves(UNORM), 1, "t1d_resolve");
    });
}

/// A layered pass renders a 1D array's slices (the probe's `t1d_array_layered`: two slices,
/// each instance its own; here, with the layer-choosing fixture, an `R32Uint` 1D array whose
/// slices take `rt6_layered`'s words 0x100 and 0x201).
#[test]
fn a_layered_pass_renders_a_line_s_slices() {
    on_device(|lane| {
        let texture = line(lane, PixelFormat::R32Uint, RT, 2);
        let mut descriptor = pass(&texture, (0, 0), LoadAction::Clear, None);
        descriptor.render_target_array_length = 2;
        let pipeline = lane.tagged();
        lane.run(&descriptor, |l, e, h| {
            e.set_render_pipeline_state(&pipeline).unwrap();
            let layers: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
            let layers = buffer_with(&l.device, &layers).unwrap();
            e.set_vertex_buffer(Some(&layers), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 2, 0)
                .unwrap();
            h.push(layers);
        });
        for (slice, word) in [(0, 0x100u32), (1, 0x201)] {
            check(
                &row(&texture, slice),
                &word.to_le_bytes().repeat(16),
                0,
                "t1d_array_layered",
            );
        }
    });
}

/// An `RGBA8Unorm` texture that renders, made without `PixelFormatView` usage, renders into its
/// `BGRA8Unorm` and `BGRA8Unorm_sRGB` views in place: its image lists their formats, so each is a
/// view of the image (no `shadowed-texture-view`), and stores its format's bytes as Metal's does
/// (the probe's `v_*` bytes, in the view's channel order). Apple documents the `bgra8Unorm_srgb`
/// view as needing no `PixelFormatView` usage; plain Metal makes the `BGRA8Unorm` one too.
#[test]
fn a_render_target_s_reordered_views_render_in_place() {
    const BGRA_UNORM: ([u8; 4], [u8; 4]) = ([0x99, 0x66, 0x33, 0xcc], [0xbf, 0x40, 0x80, 0xff]);
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let lane = Lane::new(&device);
        for (pf, want) in [
            (PixelFormat::BGRA8Unorm, BGRA_UNORM),
            (PixelFormat::BGRA8UnormSRGB, BGRA_SRGB),
        ] {
            let texture = lane.texture(PixelFormat::RGBA8Unorm, RT, PLAIN);
            let viewed = texture.new_texture_view(pf).unwrap();
            let pipeline = lane.solid(pf, 1, false);
            lane.run(
                &pass(&viewed, (0, 0), LoadAction::Clear, None),
                |l, e, h| l.left(e, h, &pipeline, COLOR),
            );
            let what = format!("{}: a {pf:?} view", device.name());
            // Within 1, as this file's other checks: the device's own rounding.
            check(&lane.bytes(&texture, 0, 0), &halves(want), 1, &what);
        }
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&Degradation::ShadowedTextureView),
            None,
            "{}: {notices:?}",
            device.name()
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// What a 2048x2048 `RGBA8Unorm` render target made without `PixelFormatView` usage costs the
/// GPU, without the validation layer: a pass that clears it and stores, a blit copying it into
/// another such texture, and a pass that clears its `BGRA8Unorm` view and stores, each its own
/// command buffer, medians and the fastest of 40 after 8 untimed; and the notices, whose
/// `shadowed-texture-view` says the view's pass rendered through a shadow. The first two show
/// what the formats the texture's image lists cost the driver's compression. Ignored by default:
/// a measurement, not a check.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn reordered_view_render_costs() {
    const BIG: u64 = 2048;
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, BIG, BIG, false);
    descriptor.usage = RT;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let copy = device.new_texture(&descriptor).unwrap();
    let view = target.new_texture_view(PixelFormat::BGRA8Unorm).unwrap();
    let queue = device.new_command_queue().unwrap();
    let time = |encode: &dyn Fn(&CommandBuffer)| {
        let mut gpu = Vec::new();
        for run in 0..48 {
            let command_buffer = queue.command_buffer().unwrap();
            encode(&command_buffer);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let start = command_buffer.gpu_start_time().unwrap();
            let end = command_buffer.gpu_end_time().unwrap();
            if run >= 8 {
                gpu.push((end - start) * 1e6);
            }
        }
        gpu.sort_by(f64::total_cmp);
        (gpu[gpu.len() / 2], gpu[0])
    };
    let cleared = |command_buffer: &CommandBuffer, texture: &Texture| {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
    };
    let clear = time(&|command_buffer| cleared(command_buffer, &target));
    let blit = time(&|command_buffer| {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_texture(&target, &copy).unwrap();
        blit.end_encoding().unwrap();
    });
    let viewed = time(&|command_buffer| cleared(command_buffer, &view));
    let notices = counts(&messages.take_notices());
    eprintln!(
        "REORDERED-VIEW on {}: clear median {:.1} us, fastest {:.1} us; copy median {:.1} us, \
         fastest {:.1} us; the view's clear median {:.1} us, fastest {:.1} us; notices {notices:?}",
        device.name(),
        clear.0,
        clear.1,
        blit.0,
        blit.1,
        viewed.0,
        viewed.1
    );
}
