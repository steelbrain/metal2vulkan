//! Device tests of the render encoder's state (its second part: the depth/stencil state and the
//! rest of the render state) on the device the cfg selects, under the validation layer.
//!
//! Each draw is one that a Metal probe measured (on an M4 with macOS 27.0.1, each case with and
//! without Metal's API validation), drawn with the same functions (the owned fixture
//! `rt6_render_state`) into the same pass: a `size`² `RGBA8Unorm` target cleared to zero, and the
//! depth and stencil textures the case names, cleared and stored. Its texels are compared with the
//! ones Metal stored. The cases cover scissor rectangles (one past the render area clipped to it,
//! as plain Metal draws it where its validation stops it), viewports (with the implicit scissor of
//! fractional edges, and a point cut to the pixels its viewport overlaps), culling and winding,
//! depth bias and clip modes, the blend color, depth/stencil states and saturated references, and
//! separate depth and stencil textures, cleared, loaded with a `DontCare` stencil store, and of
//! different sizes (which Metal's validation stops and plain Metal renders over the smaller, its
//! clear reaching the larger's first tile).
//!
//! Five draws are the device's where Vulkan leaves the answer to it: a point wider than a viewport
//! with fractional edges, a constant bias on `Depth16Unorm` whose depth is a tie, and blended
//! channels that are ties. Each prints the device's answer beside Metal's and asserts the bound the
//! runtime owns: no pixel outside Metal's, a depth within Vulkan's permitted bias units, and a
//! channel within one unit of Metal's.
//!
//! A vertex function that selects its viewport (the owned fixture `rt6_viewports`) draws as Metal
//! probes on an M4 and the M3 Max measured: each instance through the viewport of its index and
//! that viewport's scissor rectangle, an index at or past the viewports set through the first
//! viewport and its scissor rectangle, and a fragment function reads the index as written.
//!
//! What Metal's validation stops, the runtime refuses typed, before anything draws; that includes
//! 17 viewports, which hung the GPU without validation, so they are never drawn. A triangle fill
//! mode of lines is printed rather than compared: Vulkan rasterizes those edges as lines, whose
//! pixels each implementation picks within the specification's rules.

use super::degrade_notices::made_render_kinds;
use super::viewport_ranges::{depth_ranges_refused, native_depth_range};
use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device_with, LaneChecks, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Capability, Feature};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, CullMode, DepthClipMode, PixelFormat, PrimitiveTopologyClass,
    PrimitiveType, StencilOperation, TextureUsage, TriangleFillMode, Winding,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::Degradation;
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandQueue, Device, DeviceOptions, Error,
    ExecutedPipeline, Invalid, Library, LoadAction, PipelineOptions, Problem, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, RenderStateProblem,
    ResourceOptions, RuntimeGap, ScissorRect, StoreAction, Texture, TextureDescriptor, Viewport,
};
use std::cell::RefCell;
use std::collections::BTreeSet;

/// The fixture's library on `device`.
pub(super) fn library(device: &Device) -> Library {
    let ll = metal2vulkan_devtools::engine_dir().join("rt6_render_state.ll");
    let bitcode = llvm_assemble(&ll).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// What a probe pipeline has: its functions, its depth and stencil formats, whether it blends by
/// the blend color (`src * blendColor + dst * (1 - blendColor)`, alpha likewise), and its input
/// primitive topology class (Metal's default, none, unless named).
#[derive(Clone, Copy)]
pub(super) struct Pipe {
    vertex: &'static str,
    fragment: &'static str,
    depth: Option<PixelFormat>,
    stencil: Option<PixelFormat>,
    blend: bool,
    topology: PrimitiveTopologyClass,
    /// Made with `supportIndirectCommandBuffers`, for an indirect command buffer's commands.
    indirect: bool,
}

/// The probe's default pipeline: the fullscreen triangle, white.
pub(super) const WHITE: Pipe = Pipe {
    vertex: "rt6_full",
    fragment: "rt6_white",
    depth: None,
    stencil: None,
    blend: false,
    topology: PrimitiveTopologyClass::Unspecified,
    indirect: false,
};

impl Pipe {
    pub(super) const fn vertex(self, vertex: &'static str) -> Pipe {
        Pipe { vertex, ..self }
    }

    pub(super) fn depth(self, format: PixelFormat) -> Pipe {
        Pipe {
            depth: Some(format),
            ..self
        }
    }

    pub(super) fn stencil(self) -> Pipe {
        Pipe {
            stencil: Some(PixelFormat::Stencil8),
            ..self
        }
    }

    pub(super) const fn fragment(self, fragment: &'static str) -> Pipe {
        Pipe { fragment, ..self }
    }

    pub(super) fn indirect(self) -> Pipe {
        Pipe {
            indirect: true,
            ..self
        }
    }

    /// The pipeline state on `device`.
    fn state(self, device: &Device, library: &Library) -> Result<RenderPipelineState, Error> {
        let descriptor = self.descriptor(library);
        device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// The descriptor of [`Pipe::state`]'s pipeline, of `library`'s functions.
    fn descriptor(self, library: &Library) -> RenderPipelineDescriptor {
        let function = |name| library.new_function_with_name(name).unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            function(self.vertex).function_descriptor(),
        ));
        descriptor.fragment = Some(StageSource::Function(
            function(self.fragment).function_descriptor(),
        ));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        if self.blend {
            color.blending_enabled = true;
            color.source_rgb_blend_factor = BlendFactor::BlendColor;
            color.destination_rgb_blend_factor = BlendFactor::OneMinusBlendColor;
            color.source_alpha_blend_factor = BlendFactor::BlendAlpha;
            color.destination_alpha_blend_factor = BlendFactor::OneMinusBlendAlpha;
        }
        descriptor.color_attachments[0] = Some(color);
        descriptor.depth_format = self.depth;
        descriptor.stencil_format = self.stencil;
        descriptor.input_primitive_topology = self.topology;
        descriptor.support_indirect_command_buffers = self.indirect;
        descriptor
    }
}

/// A probe pass: a `size`² target, and the depth and stencil textures of these formats, cleared
/// to these values.
#[derive(Clone, Copy)]
pub(super) struct Pass {
    size: u64,
    depth: Option<PixelFormat>,
    stencil: bool,
    clear_depth: f64,
    clear_stencil: u32,
}

/// The probe's default pass: 4², color only.
pub(super) const PASS: Pass = Pass {
    size: 4,
    depth: None,
    stencil: false,
    clear_depth: 1.0,
    clear_stencil: 0,
};

impl Pass {
    pub(super) fn size(self, size: u64) -> Pass {
        Pass { size, ..self }
    }

    pub(super) fn depth(self, format: PixelFormat, clear: f64) -> Pass {
        Pass {
            depth: Some(format),
            clear_depth: clear,
            ..self
        }
    }

    pub(super) fn stencil(self, clear: u32) -> Pass {
        Pass {
            stencil: true,
            clear_stencil: clear,
            ..self
        }
    }
}

/// What a pass stored: the target's texels, and the depth and stencil textures' when it has them.
#[derive(Debug, PartialEq)]
pub(super) struct Stored {
    pub(super) color: Vec<u8>,
    pub(super) depth: Option<Vec<u8>>,
    pub(super) stencil: Option<Vec<u8>>,
}

/// A `Shared` 2D texture of `pf`, `size`², for rendering into and reading back.
pub(super) fn texture(device: &Device, pf: PixelFormat, size: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, size, size, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// Every texel of `texture`, `bytes` each.
pub(super) fn texels(texture: &Texture, bytes: u64) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let mut out = vec![0u8; (width * height * bytes) as usize];
    texture
        .get_bytes(
            &mut out,
            width * bytes,
            width * height * bytes,
            Region::new_2d(0, 0, width, height),
            0,
            0,
        )
        .unwrap();
    out
}

/// The probe's draw of `vertex_count` vertices of `primitive` at depth `z + slope * x`: the
/// depth buffer at index 0, which the caller keeps until the command buffer completed.
fn draw_at(
    encoder: &mut RenderCommandEncoder,
    depth: &Buffer,
    primitive: PrimitiveType,
    vertex_count: u32,
) -> Result<(), Error> {
    encoder.set_vertex_buffer(Some(depth), 0, 0)?;
    encoder.draw_primitives(primitive, 0, vertex_count, 1, 0)
}

/// The device's queue, library and depth buffers for one test.
pub(super) struct Probe {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    pub(super) library: Library,
}

impl Probe {
    /// The depth bytes `[z, slope]` the fixture's vertex functions read.
    pub(super) fn depth(&self, z: f32, slope: f32) -> Buffer {
        let mut bytes = z.to_le_bytes().to_vec();
        bytes.extend_from_slice(&slope.to_le_bytes());
        buffer_with(&self.device, &bytes).unwrap()
    }

    pub(super) fn pipeline(&self, pipe: Pipe) -> RenderPipelineState {
        pipe.state(&self.device, &self.library).unwrap()
    }

    /// The kinds the making of [`Probe::pipeline`]'s state of `pipe` announces
    /// (`degrade_notices::made_render_kinds`).
    pub(super) fn made(&self, pipe: Pipe) -> BTreeSet<Degradation> {
        made_render_kinds(&self.device, &pipe.descriptor(&self.library))
    }

    /// Run `pass` with `body` encoding into it; what it stored.
    pub(super) fn run(
        &self,
        pass: Pass,
        body: impl FnOnce(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    ) -> Stored {
        self.try_run(pass, body).unwrap()
    }

    /// [`Probe::run`], or the encoder's refusal of the pass (a format the device lacks).
    fn try_run(
        &self,
        pass: Pass,
        body: impl FnOnce(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    ) -> Result<Stored, Error> {
        self.try_run_with(pass, None, body)
    }

    /// [`Probe::run`], and how many geometry stages the render pipeline of each draw it executed
    /// carries, in order.
    pub(super) fn run_staged(
        &self,
        pass: Pass,
        body: impl FnOnce(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    ) -> (Stored, Vec<usize>) {
        let (stored, command_buffer) = self.encoded(pass, None, body).unwrap();
        let stages = command_buffer
            .executions()
            .iter()
            .filter_map(|execution| match execution.pipeline() {
                ExecutedPipeline::Render(pipeline) => Some(
                    pipeline
                        .stages
                        .iter()
                        .filter(|s| s.stage == vk::ShaderStage::GEOMETRY)
                        .count(),
                ),
                _ => None,
            })
            .collect();
        (stored, stages)
    }

    /// [`Probe::try_run`] with `visibility` as the pass's visibility result buffer.
    pub(super) fn try_run_with(
        &self,
        pass: Pass,
        visibility: Option<&Buffer>,
        body: impl FnOnce(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    ) -> Result<Stored, Error> {
        self.encoded(pass, visibility, body)
            .map(|(stored, _)| stored)
    }

    /// [`Probe::try_run_with`], and the command buffer it committed.
    fn encoded(
        &self,
        pass: Pass,
        visibility: Option<&Buffer>,
        body: impl FnOnce(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    ) -> Result<(Stored, CommandBuffer), Error> {
        let color = texture(&self.device, PixelFormat::RGBA8Unorm, pass.size);
        let depth = pass.depth.map(|pf| texture(&self.device, pf, pass.size));
        let stencil = pass
            .stencil
            .then(|| texture(&self.device, PixelFormat::Stencil8, pass.size));
        let mut descriptor = RenderPassDescriptor::new();
        let target = &mut descriptor.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        // The probe's clear color, (0, 0, 0, 0): Metal's default has an alpha of 1.
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        target.store_action = StoreAction::Store;
        descriptor.depth_attachment.texture = depth.clone();
        descriptor.depth_attachment.clear_depth = pass.clear_depth;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        descriptor.stencil_attachment.texture = stencil.clone();
        descriptor.stencil_attachment.clear_stencil = pass.clear_stencil;
        descriptor.stencil_attachment.store_action = StoreAction::Store;
        descriptor.visibility_result_buffer = visibility.cloned();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&descriptor)?;
        let mut held = Vec::new();
        body(self, &mut encoder, &mut held);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        drop(held);
        let stored = Stored {
            color: texels(&color, 4),
            depth: depth.map(|t| {
                let bytes = if t.pixel_format() == PixelFormat::Depth16Unorm {
                    2
                } else {
                    4
                };
                texels(&t, bytes)
            }),
            stencil: stencil.map(|t| texels(&t, 1)),
        };
        Ok((stored, command_buffer))
    }

    /// The probe's `draw`: `pipe`'s pipeline, the depth buffer of `z` and `slope`, one triangle.
    pub(super) fn draw(
        &self,
        encoder: &mut RenderCommandEncoder,
        held: &mut Vec<Buffer>,
        pipe: Pipe,
        z: f32,
        slope: f32,
    ) {
        encoder
            .set_render_pipeline_state(&self.pipeline(pipe))
            .unwrap();
        let depth = self.depth(z, slope);
        draw_at(encoder, &depth, PrimitiveType::Triangle, 3).unwrap();
        held.push(depth);
    }

    /// A depth/stencil state of `compare` and `write`, with `stencil` on both faces.
    pub(super) fn state(
        &self,
        compare: CompareFunction,
        write: bool,
        stencil: Option<StencilDescriptor>,
    ) -> metal2vulkan_runtime::DepthStencilState {
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = compare;
        descriptor.depth_write_enabled = write;
        if let Some(face) = stencil {
            descriptor.front_face_stencil = face;
            descriptor.back_face_stencil = face;
        }
        self.device.new_depth_stencil_state(&descriptor).unwrap()
    }
}

/// The probe's stencil descriptor: `compare`, `pass` on a pass, `Keep` otherwise, the masks.
pub(super) fn stencil(
    compare: CompareFunction,
    pass: StencilOperation,
    read: u32,
    write: u32,
) -> StencilDescriptor {
    let mut face = StencilDescriptor::default();
    face.stencil_compare_function = compare;
    face.depth_stencil_pass_operation = pass;
    face.read_mask = read;
    face.write_mask = write;
    face
}

/// The `RGBA8Unorm` texels of `rows`: `#` white, `.` zero.
pub(super) fn rows(rows: &[&str]) -> Vec<u8> {
    rows.iter()
        .flat_map(|row| row.chars())
        .flat_map(|c| match c {
            '#' => [0xff; 4],
            '.' => [0; 4],
            other => panic!("{other:?} is not a texel"),
        })
        .collect()
}

/// `count` copies of `texel`.
fn each(texel: &[u8], count: usize) -> Vec<u8> {
    texel.repeat(count)
}

/// `count` depths of `bits`, as `Depth32Float` stores them.
fn depths(bits: u32, count: usize) -> Vec<u8> {
    bits.to_le_bytes().repeat(count)
}

pub(super) const FULL: [&str; 4] = ["####", "####", "####", "####"];
pub(super) const EMPTY: [&str; 4] = ["....", "....", "....", "...."];

pub(super) fn validated() -> (Messages, ScratchCache, Probe) {
    validated_without(&[])
}

/// [`validated`] on the test device without `withheld`.
pub(super) fn validated_without(withheld: &[VkFeature]) -> (Messages, ScratchCache, Probe) {
    validated_adjusted(|options| options.withheld_features.extend(withheld.iter().copied()))
}

/// [`validated`] on the test device whose validation layer checks synchronization too, as the
/// lane's `sync` switch has it ([`LaneChecks`]).
pub(super) fn validated_checking_synchronization() -> (Messages, ScratchCache, Probe) {
    validated_adjusted(|options| {
        LaneChecks {
            synchronization: true,
            ..LaneChecks::default()
        }
        .apply(options)
    })
}

/// [`validated`] on the test device made with options `adjust` changed.
fn validated_adjusted(adjust: impl FnOnce(&mut DeviceOptions)) -> (Messages, ScratchCache, Probe) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    (messages, cache, probe)
}

pub(super) fn quiet(probe: Probe, messages: &Messages) {
    drop(probe);
    layer_reported_nothing_at_all(messages);
}

/// Each probe draw stores the texels Metal stored (the module docs). Every draw runs, and the
/// ones that differ are listed together at the end.
#[test]
fn render_state_draws_store_metals_texels() {
    let (messages, _cache, probe) = validated();
    let f32_depth = PixelFormat::Depth32Float;
    let color = |stored: Stored| stored.color;
    let differ = RefCell::new(Vec::new());
    let check = |name: &str, stored: Vec<u8>, metal: Vec<u8>| {
        if stored != metal {
            differ
                .borrow_mut()
                .push(format!("{name}: stored {stored:?}, Metal {metal:?}"));
        }
    };
    // A draw whose texels the device's rasterizer decides where Vulkan leaves them to it
    // (the device differences of step 6's second part): its answer is printed per device beside
    // Metal's, and the caller asserts the bound the runtime owns.
    let device = probe.device.name();
    let device_dependent = |name: &str, stored: &[u8], metal: &[u8]| {
        if stored == metal {
            eprintln!("device-dependent: {name} on {device}: Metal's texels");
        } else {
            eprintln!("device-dependent: {name} on {device}: {stored:?} (Metal {metal:?})");
        }
    };

    // Scissor rectangles.
    let scissor = |rects: Vec<ScissorRect>| {
        move |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
            e.set_scissor_rects(&rects).unwrap();
            p.draw(e, h, WHITE, 0.5, 0.0);
        }
    };
    check(
        "scissor_inside",
        color(probe.run(PASS, scissor(vec![ScissorRect::new(1, 1, 2, 2)]))),
        rows(&["....", ".##.", ".##.", "...."]),
    );
    check(
        "scissor_zero",
        color(probe.run(PASS, scissor(vec![ScissorRect::new(0, 0, 0, 0)]))),
        rows(&EMPTY),
    );
    check(
        "scissors_one",
        color(probe.run(PASS, scissor(vec![ScissorRect::new(0, 0, 2, 4)]))),
        rows(&["##..", "##..", "##..", "##.."]),
    );
    // Two rectangles with one viewport, which Metal's validation stops: plain Metal draws
    // through the first.
    check(
        "scissors_two",
        color(probe.run(
            PASS,
            scissor(vec![
                ScissorRect::new(0, 0, 2, 4),
                ScissorRect::new(2, 0, 2, 4),
            ]),
        )),
        rows(&["##..", "##..", "##..", "##.."]),
    );
    // Seventeen, which Metal's validation stops: plain Metal draws (through the first).
    check(
        "scissors_17",
        color(probe.run(PASS, scissor(vec![ScissorRect::new(0, 0, 4, 4); 17]))),
        rows(&FULL),
    );
    // Past the render area, clipped to it as plain Metal clips it (its validation stops these).
    check(
        "scissor_past_width",
        color(probe.run(PASS, scissor(vec![ScissorRect::new(2, 0, 4, 4)]))),
        rows(&["..##", "..##", "..##", "..##"]),
    );
    check(
        "scissor_past_height",
        color(probe.run(PASS, scissor(vec![ScissorRect::new(0, 3, 4, 2)]))),
        rows(&["....", "....", "....", "####"]),
    );
    check(
        "scissor_origin_outside",
        color(probe.run(PASS, scissor(vec![ScissorRect::new(5, 5, 0, 0)]))),
        rows(&EMPTY),
    );
    check(
        "scissor_then_draw_twice",
        color(probe.run(PASS, |p, e, h| {
            e.set_scissor_rect(ScissorRect::new(0, 0, 1, 1)).unwrap();
            p.draw(e, h, WHITE, 0.5, 0.0);
            e.set_scissor_rect(ScissorRect::new(3, 3, 1, 1)).unwrap();
            p.draw(e, h, WHITE, 0.5, 0.0);
        })),
        rows(&["#...", "....", "....", "...#"]),
    );

    // Viewports, with the pixels they overlap.
    let viewport = |viewports: Vec<Viewport>, pipe: Pipe, z: f32| {
        move |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
            e.set_viewports(&viewports).unwrap();
            p.draw(e, h, pipe, z, 0.0);
        }
    };
    let at = |x, y, w, h| Viewport::new(x, y, w, h, 0.0, 1.0);
    for (name, size, viewports, metal) in [
        (
            "viewport_quadrant",
            4,
            vec![at(2.0, 0.0, 2.0, 2.0)],
            vec!["..##", "..##", "....", "...."],
        ),
        (
            "viewport_outside",
            4,
            vec![at(-2.0, -2.0, 4.0, 4.0)],
            vec!["##..", "##..", "....", "...."],
        ),
        (
            "viewport_past_target",
            4,
            vec![at(2.0, 2.0, 8.0, 8.0)],
            vec!["....", "....", "..##", "..##"],
        ),
        (
            "viewport_zero_width",
            4,
            vec![at(0.0, 0.0, 0.0, 4.0)],
            EMPTY.to_vec(),
        ),
        // Metal's validation stops a NaN origin x; plain Metal draws nothing.
        (
            "viewport_nan",
            4,
            vec![at(f64::NAN, 0.0, 4.0, 4.0)],
            EMPTY.to_vec(),
        ),
        (
            "viewport_negative_height",
            4,
            vec![at(0.0, 4.0, 4.0, -4.0)],
            FULL.to_vec(),
        ),
        (
            "viewports_two",
            4,
            vec![at(0.0, 0.0, 2.0, 2.0), at(2.0, 2.0, 2.0, 2.0)],
            vec!["##..", "##..", "....", "...."],
        ),
        (
            "viewport_fraction",
            8,
            vec![at(0.5, 0.25, 3.5, 2.75)],
            vec![
                "####....", "####....", "####....", "........", "........", "........", "........",
                "........",
            ],
        ),
    ] {
        check(
            name,
            color(probe.run(PASS.size(size), viewport(viewports, WHITE, 0.5))),
            rows(&metal),
        );
    }
    let small = WHITE.vertex("rt6_small");
    check(
        "viewport_negative_height_small",
        color(probe.run(
            PASS.size(8),
            viewport(vec![at(0.0, 8.0, 8.0, -8.0)], small, 0.5),
        )),
        rows(&[
            "........", "........", "..###...", "..##....", "..#.....", "........", "........",
            "........",
        ]),
    );
    check(
        "viewport_negative_height_cull",
        color(probe.run(PASS.size(8), |p, e, h| {
            e.set_viewport(at(0.0, 8.0, 8.0, -8.0)).unwrap();
            e.set_cull_mode(CullMode::Back);
            p.draw(e, h, small, 0.5, 0.0);
        })),
        rows(&[
            "........", "........", "..###...", "..##....", "..#.....", "........", "........",
            "........",
        ]),
    );
    let depth_range = |near, far, z| {
        move |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
            e.set_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, near, far))
                .unwrap();
            e.set_depth_stencil_state(&p.state(CompareFunction::Always, true, None))
                .unwrap();
            p.draw(e, h, WHITE.depth(PixelFormat::Depth32Float), z, 0.0);
        }
    };
    let with_depth = PASS.depth(f32_depth, 1.0);
    for (name, near, far, z, bits) in [
        ("viewport_depth_range", 0.25, 0.75, 0.5, 0x3f00_0000),
        ("viewport_depth_inverted", 1.0, 0.0, 0.25, 0x3f40_0000),
    ] {
        let stored = probe.run(with_depth, depth_range(near, far, z));
        check(name, stored.color, rows(&FULL));
        check(name, stored.depth.unwrap(), depths(bits, 16));
    }

    // Points cut to the pixels their viewport overlaps.
    let point = Pipe {
        topology: PrimitiveTopologyClass::Point,
        ..WHITE.vertex("rt6_point")
    };
    let points = |viewport: Option<Viewport>, scissor: Option<ScissorRect>| {
        move |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
            if let Some(viewport) = viewport {
                e.set_viewport(viewport).unwrap();
            }
            if let Some(rect) = scissor {
                e.set_scissor_rect(rect).unwrap();
            }
            e.set_render_pipeline_state(&p.pipeline(point)).unwrap();
            let depth = p.depth(0.5, 0.0);
            draw_at(e, &depth, PrimitiveType::Point, 1).unwrap();
            h.push(depth);
        }
    };
    let blank = "........";
    // A point wider than a viewport with fractional edges keeps Metal's pixels through the
    // widened viewports of Metal's guard band (`wide_points`).
    for (name, viewport, scissor, metal) in [
        (
            "point_no_viewport",
            None,
            None,
            [
                blank, blank, "..####..", "..####..", "..####..", "..####..", blank, blank,
            ],
        ),
        (
            "viewport_points_control",
            Some(at(4.0, 0.0, 1.0, 8.0)),
            None,
            [
                blank, blank, "....#...", "....#...", "....#...", "....#...", blank, blank,
            ],
        ),
        (
            "point_viewport_fraction",
            Some(at(4.5, 0.0, 1.0, 8.0)),
            None,
            [
                blank, blank, "....##..", "....##..", "....##..", "....##..", blank, blank,
            ],
        ),
        (
            "point_viewport_fraction_2",
            Some(at(3.25, 2.75, 1.5, 2.5)),
            None,
            [
                blank, blank, "...##...", "...##...", "...##...", "...##...", blank, blank,
            ],
        ),
        (
            "viewport_zero_width_points",
            Some(at(4.0, 0.0, 0.0, 8.0)),
            None,
            [blank; 8],
        ),
        (
            "point_viewport_zero_height",
            Some(at(0.0, 4.0, 8.0, 0.0)),
            None,
            [blank; 8],
        ),
        (
            "point_scissor_and_viewport",
            Some(at(2.0, 2.0, 4.0, 4.0)),
            Some(ScissorRect::new(3, 0, 5, 8)),
            [
                blank, blank, "...###..", "...###..", "...###..", "...###..", blank, blank,
            ],
        ),
    ] {
        let stored = color(probe.run(PASS.size(8), points(viewport, scissor)));
        check(name, stored, rows(&metal));
    }

    // Culling and winding: the fullscreen triangle is counter-clockwise.
    let culled = |cull: CullMode, winding: Option<Winding>, vertex: &'static str| {
        move |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
            if let Some(winding) = winding {
                e.set_front_facing_winding(winding);
            }
            e.set_cull_mode(cull);
            p.draw(e, h, WHITE.vertex(vertex), 0.5, 0.0);
        }
    };
    for (name, cull, winding, vertex, metal) in [
        (
            "cull_back_default_winding",
            CullMode::Back,
            None,
            "rt6_full",
            EMPTY,
        ),
        (
            "cull_front_default_winding",
            CullMode::Front,
            None,
            "rt6_full",
            FULL,
        ),
        (
            "cull_back_ccw",
            CullMode::Back,
            Some(Winding::CounterClockwise),
            "rt6_full",
            FULL,
        ),
        ("cull_back_flip", CullMode::Back, None, "rt6_flip", FULL),
    ] {
        check(
            name,
            color(probe.run(PASS, culled(cull, winding, vertex))),
            rows(&metal),
        );
    }
    check(
        "fill_fill_small",
        color(probe.run(PASS.size(8), |p, e, h| p.draw(e, h, small, 0.5, 0.0))),
        rows(&[
            blank, blank, blank, "..#.....", "..##....", "..###...", blank, blank,
        ]),
    );

    // Depth bias and clip modes.
    let biased = |bias: [f32; 3], pf: PixelFormat, z: f32, slope: f32, clamp: bool| {
        move |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
            e.set_depth_bias(bias[0], bias[1], bias[2]).unwrap();
            if clamp {
                e.set_depth_clip_mode(DepthClipMode::Clamp).unwrap();
            }
            e.set_depth_stencil_state(&p.state(CompareFunction::Always, true, None))
                .unwrap();
            p.draw(e, h, WHITE.depth(pf), z, slope);
        }
    };
    let per_column = |columns: [u32; 4]| -> Vec<u8> {
        columns
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<u8>>()
            .repeat(4)
    };
    for (name, bias, z, slope, metal) in [
        (
            "bias_none",
            [0.0, 0.0, 0.0],
            0.5,
            0.0,
            depths(0x3f00_0000, 16),
        ),
        (
            "bias_const_1",
            [1.0, 0.0, 0.0],
            0.5,
            0.0,
            depths(0x3f00_0001, 16),
        ),
        (
            "bias_const_100",
            [100.0, 0.0, 0.0],
            0.5,
            0.0,
            depths(0x3f00_0064, 16),
        ),
        (
            "bias_const_neg",
            [-3.0, 0.0, 0.0],
            0.375,
            0.0,
            depths(0x3ebf_fffd, 16),
        ),
        (
            "bias_clamp",
            [1000.0, 0.0, 0.0001],
            0.5,
            0.0,
            depths(0x3f00_03e8, 16),
        ),
        (
            "bias_slope",
            [0.0, 1.0, 0.0],
            0.5,
            0.125,
            per_column([0x3ef0_0000, 0x3f08_0000, 0x3f18_0000, 0x3f28_0000]),
        ),
        (
            "bias_no_slope_sloped",
            [0.0, 0.0, 0.0],
            0.5,
            0.125,
            per_column([0x3ed0_0000, 0x3ef0_0000, 0x3f08_0000, 0x3f18_0000]),
        ),
    ] {
        let stored = probe.run(with_depth, biased(bias, f32_depth, z, slope, false));
        check(name, stored.color, rows(&FULL));
        check(name, stored.depth.unwrap(), metal);
    }
    let d16 = probe.run(
        PASS.depth(PixelFormat::Depth16Unorm, 1.0),
        biased([3.0, 0.0, 0.0], PixelFormat::Depth16Unorm, 0.5, 0.0, false),
    );
    // Vulkan leaves a unorm format's bias unit r to the device, between 2^-16 and 2^-15 for
    // 16 bits, and this biased depth (Metal's 0x8003) is a tie: the stored depth must lie
    // between the two units' nearest values (0x8002 to 0x8005), and is printed.
    let d16_depth = d16.depth.unwrap();
    let [low, high] = unorm_bias_bounds(0.5, 3.0, 16);
    let outside = d16_depth
        .chunks(2)
        .map(|t| u16::from_le_bytes([t[0], t[1]]))
        .any(|d| !(low..=high).contains(&d));
    if outside {
        check("bias_const_d16", d16_depth.clone(), each(&[0x03, 0x80], 16));
    }
    device_dependent("bias_const_d16", &d16_depth, &each(&[0x03, 0x80], 16));
    for (name, z, clamp, metal_color, bits) in [
        ("clip_far_clip", 1.5, false, EMPTY, 0x3f80_0000),
        ("clip_far_clamp", 1.5, true, FULL, 0x3f80_0000),
        ("clip_near_clamp", -0.5, true, FULL, 0),
    ] {
        let stored = probe.run(with_depth, biased([0.0; 3], f32_depth, z, 0.0, clamp));
        check(name, stored.color, rows(&metal_color));
        check(name, stored.depth.unwrap(), depths(bits, 16));
    }
    let stored = probe.run(with_depth, |p, e, h| {
        e.set_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, 0.25, 0.75))
            .unwrap();
        e.set_depth_clip_mode(DepthClipMode::Clamp).unwrap();
        e.set_depth_stencil_state(&p.state(CompareFunction::Always, true, None))
            .unwrap();
        p.draw(e, h, WHITE.depth(f32_depth), 1.5, 0.0);
    });
    check(
        "clip_clamp_viewport_range",
        stored.depth.unwrap(),
        depths(0x3f40_0000, 16),
    );
    // NaN and infinite bias factors take Metal's measured effect (Metal probes on an M3 Max and an
    // M4 alike): a NaN constant biases as +inf; an unbounded infinite or NaN bias saturates the
    // depth at 1 or 0, in depth clamp mode at the range's larger or smaller depth; a NaN or
    // infinite clamp clamps nothing; and a clamp that bounds an infinite or NaN bias is added to
    // the depth, here through a range it keeps inside [0, 1].
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    for (name, bias, range, slope, clamp, bits) in [
        (
            "bias_const_nan",
            [nan, 0.0, 0.0],
            [0.0, 1.0],
            0.0,
            false,
            0x3f80_0000,
        ),
        (
            "bias_const_ninf",
            [-inf, 0.0, 0.0],
            [0.0, 1.0],
            0.0,
            false,
            0,
        ),
        (
            "bias_slope_inf",
            [0.0, inf, 0.0],
            [0.0, 1.0],
            0.125,
            false,
            0x3f80_0000,
        ),
        (
            "bias_slope_inf_flat",
            [0.0, inf, 0.0],
            [0.0, 1.0],
            0.0,
            false,
            0x3f80_0000,
        ),
        (
            "bias_inf_minus_inf",
            [inf, -inf, 0.0],
            [0.0, 1.0],
            0.125,
            false,
            0x3f80_0000,
        ),
        (
            "bias_clamp_nan",
            [1000.0, 0.0, nan],
            [0.0, 1.0],
            0.0,
            false,
            0x3f00_03e8,
        ),
        (
            "bias_clamp_ninf",
            [-inf, 0.0, nan],
            [0.0, 1.0],
            0.0,
            false,
            0,
        ),
        (
            "bias_const_nan_range",
            [nan, 0.0, 0.0],
            [0.25, 0.75],
            0.0,
            false,
            0x3f80_0000,
        ),
        (
            "bias_const_inf_clamp",
            [inf, 0.0, 0.125],
            [0.25, 0.75],
            0.0,
            false,
            0x3f20_0000,
        ),
        (
            "bias_const_ninf_clamp",
            [-inf, 0.0, -0.125],
            [0.25, 0.75],
            0.0,
            false,
            0x3ec0_0000,
        ),
        (
            "bias_slope_nan_clamp",
            [0.0, nan, -0.125],
            [0.25, 0.75],
            0.0,
            false,
            0x3ec0_0000,
        ),
        (
            "bias_inf_clamp_mode",
            [inf, 0.0, 0.0],
            [0.75, 0.25],
            0.0,
            true,
            0x3f40_0000,
        ),
        (
            "bias_ninf_clamp_mode",
            [-inf, 0.0, 0.0],
            [0.25, 0.75],
            0.0,
            true,
            0x3e80_0000,
        ),
        (
            "bias_slope_nan_clamp_mode",
            [0.0, nan, 0.0],
            [0.25, 0.75],
            0.0,
            true,
            0x3f40_0000,
        ),
    ] {
        let stored = probe.run(with_depth, |p, e, h| {
            e.set_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, range[0], range[1]))
                .unwrap();
            biased(bias, f32_depth, 0.5, slope, clamp)(p, e, h);
        });
        check(name, stored.color, rows(&FULL));
        check(name, stored.depth.unwrap(), depths(bits, 16));
    }

    // The blend color: Metal's default is zero, and a normalized target clamps it.
    let blend = Pipe {
        blend: true,
        ..WHITE
    };
    // A channel blended to 0.5 of 255 is a tie, whose rounding Vulkan leaves to the device:
    // those channels are printed, and the others compared.
    for (name, set, metal, ties) in [
        ("blend_color_default", None, [0, 0, 0, 0], &[][..]),
        (
            "blend_color_set",
            Some([0.25, 0.5, 0.75, 1.0]),
            [0x40, 0x80, 0xbf, 0xff],
            &[1][..],
        ),
        (
            "blend_color_outside",
            Some([2.0, -1.0, 0.5, 0.5]),
            [0xff, 0x00, 0x80, 0x80],
            &[2, 3][..],
        ),
    ] {
        let stored = probe.run(PASS, move |p, e, h| {
            if let Some([r, g, b, a]) = set {
                e.set_blend_color(r, g, b, a);
            }
            p.draw(e, h, blend, 0.5, 0.0);
        });
        let channels = |texels: &[u8], tie: bool| -> Vec<u8> {
            texels
                .iter()
                .enumerate()
                .filter(|(i, _)| ties.contains(&(i % 4)) == tie)
                .map(|(_, &byte)| byte)
                .collect()
        };
        let metal = each(&metal, 16);
        check(
            name,
            channels(&stored.color, false),
            channels(&metal, false),
        );
        if !ties.is_empty() {
            // A tie's rounding is the device's: within one unit of Metal's.
            let (stored_ties, metal_ties) = (channels(&stored.color, true), channels(&metal, true));
            if stored_ties
                .iter()
                .zip(&metal_ties)
                .any(|(a, b)| a.abs_diff(*b) > 1)
            {
                check(name, stored_ties.clone(), metal_ties.clone());
            }
            device_dependent(&format!("{name}_ties"), &stored_ties, &metal_ties);
        }
    }

    // Depth/stencil states and references.
    let stored = probe.run(PASS.depth(f32_depth, 0.5), |p, e, h| {
        e.set_depth_stencil_state(&p.state(CompareFunction::Less, true, None))
            .unwrap();
        for z in [0.25, 0.375, 0.75] {
            p.draw(e, h, WHITE.depth(f32_depth), z, 0.0);
        }
    });
    check("ds_less", stored.depth.unwrap(), depths(0x3e80_0000, 16));
    // `setDepthStencilState:nil` after a state is Metal's default state again: a draw behind the
    // cleared depth passes `Always` and writes no depth (plain Metal on an M4;
    // its validation stops nil).
    let stored = probe.run(PASS.depth(f32_depth, 0.5), |p, e, h| {
        e.set_depth_stencil_state(&p.state(CompareFunction::Less, true, None))
            .unwrap();
        e.set_depth_stencil_state(None).unwrap();
        p.draw(e, h, WHITE.depth(f32_depth), 0.75, 0.0);
    });
    check("ds_nil_color", stored.color, rows(&FULL));
    check(
        "ds_nil_depth",
        stored.depth.unwrap(),
        depths(0x3f00_0000, 16),
    );
    let with_stencil = WHITE.stencil();
    let referenced = |face: StencilDescriptor, front: u32, back: u32| {
        move |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
            e.set_depth_stencil_state(&p.state(CompareFunction::Always, false, Some(face)))
                .unwrap();
            e.set_stencil_reference_values(front, back);
            p.draw(e, h, with_stencil, 0.5, 0.0);
        }
    };
    use CompareFunction::{Always, Equal, Less};
    use StencilOperation::{IncrementClamp, Keep, Replace};
    for (name, clear, face, [front, back], metal_color, metal) in [
        (
            "stencil_ref_wide_equal",
            0xff,
            stencil(Equal, Keep, 0xff, 0xff),
            [0x1ff; 2],
            FULL,
            0xff,
        ),
        (
            "stencil_ref_wide_replace",
            0,
            stencil(Always, Replace, 0xff, 0xff),
            [0x1a5; 2],
            FULL,
            0xff,
        ),
        (
            "stencil_refs_front_back",
            0,
            stencil(Always, Replace, 0xff, 0xff),
            [0x11, 0x22],
            FULL,
            0x22,
        ),
        (
            "stencil_mask_wide",
            0x0f,
            stencil(Equal, IncrementClamp, 0x10f, 0x1f0),
            [0x0f; 2],
            FULL,
            0x1f,
        ),
        (
            "ref_0x100",
            0,
            stencil(Always, Replace, 0xff, 0xff),
            [0x100; 2],
            FULL,
            0xff,
        ),
        (
            "ref_max",
            0,
            stencil(Always, Replace, 0xff, 0xff),
            [u32::MAX; 2],
            FULL,
            0xff,
        ),
        (
            "ref_0x1a5_masked_write",
            0,
            stencil(Always, Replace, 0xff, 0x0f),
            [0x1a5; 2],
            FULL,
            0x0f,
        ),
        (
            "ref_0x1a5_less",
            0xfe,
            stencil(Less, IncrementClamp, 0xff, 0xff),
            [0x1a5; 2],
            EMPTY,
            0xfe,
        ),
    ] {
        let stored = probe.run(PASS.stencil(clear), referenced(face, front, back));
        check(name, stored.color, rows(&metal_color));
        check(name, stored.stencil.unwrap(), vec![metal; 16]);
    }
    // A stencil descriptor whose only change is its masks or its failure operation needs no
    // stencil attachment.
    let mut fail_op = StencilDescriptor::default();
    fail_op.stencil_failure_operation = Replace;
    for (name, face) in [
        ("ns_masks_only", stencil(Always, Keep, 0x0f, 0xf0)),
        ("ns_fail_op_only", fail_op),
    ] {
        let stored = probe.run(PASS, move |p, e, h| {
            e.set_depth_stencil_state(&p.state(Always, false, Some(face)))
                .unwrap();
            p.draw(e, h, WHITE, 0.5, 0.0);
        });
        check(name, stored.color, rows(&FULL));
    }

    // Separate depth and stencil textures.
    let separate = |pf: PixelFormat, z: f32, face: StencilDescriptor| {
        move |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
            e.set_depth_stencil_state(&p.state(Less, true, Some(face)))
                .unwrap();
            p.draw(e, h, WHITE.depth(pf).stencil(), z, 0.0);
        }
    };
    let mut depth_fail = StencilDescriptor::default();
    depth_fail.depth_failure_operation = StencilOperation::Invert;
    let increment = stencil(Always, IncrementClamp, 0xff, 0xff);
    for (name, pf, z, face, metal_color, depth, stencil_value) in [
        (
            "separate_depth_stencil",
            f32_depth,
            0.25,
            increment,
            FULL,
            depths(0x3e80_0000, 16),
            0x41,
        ),
        (
            "separate_depth_stencil_fail",
            f32_depth,
            0.75,
            depth_fail,
            EMPTY,
            depths(0x3f00_0000, 16),
            0xbf,
        ),
        (
            "separate_d16_s8",
            PixelFormat::Depth16Unorm,
            0.25,
            increment,
            FULL,
            each(&[0x00, 0x40], 16),
            0x41,
        ),
    ] {
        let pass = PASS.depth(pf, 0.5).stencil(0x40);
        // A device without the combined format refuses the pass, typed (D16_UNORM_S8_UINT is
        // not a format every device has).
        let stored = match probe.try_run(pass, separate(pf, z, face)) {
            Err(Error::Unsupported(unsupported)) => {
                eprintln!(
                    "unsupported: {name} on {}: {unsupported}",
                    probe.device.name()
                );
                continue;
            }
            stored => stored.unwrap(),
        };
        check(name, stored.color, rows(&metal_color));
        check(name, stored.depth.unwrap(), depth);
        check(name, stored.stencil.unwrap(), vec![stencil_value; 16]);
    }
    let differ = differ.into_inner();
    for line in &differ {
        eprintln!("differs: {line}");
    }
    separate_load_dontcare(&probe);

    // A triangle fill mode of lines: printed (the module docs).
    let lines = probe.run(PASS.size(8), |p, e, h| {
        e.set_triangle_fill_mode(TriangleFillMode::Lines).unwrap();
        p.draw(e, h, small, 0.5, 0.0);
    });
    let metal = rows(&[
        blank, blank, ".##.....", ".#.#....", ".#..#...", ".#####..", blank, blank,
    ]);
    eprintln!(
        "fill lines on {}: {}",
        probe.device.name(),
        if lines.color == metal {
            "Metal's pixels".to_owned()
        } else {
            format!("{:?}", picture(&lines.color, 8))
        }
    );
    quiet(probe, &messages);
    assert!(
        differ.is_empty(),
        "{} draws differ from Metal's texels (listed above)",
        differ.len()
    );
}

/// The stored values Vulkan permits for a depth `z` biased by a positive constant factor `k` in an
/// `n`-bit unorm format: its unit r lies between 2^-n and twice that (VK_EXT_depth_bias_control
/// pins it only with `depthBiasExact`), and the conversion rounds to nearest either way at a tie.
fn unorm_bias_bounds(z: f64, k: f64, n: i32) -> [u16; 2] {
    let scale = f64::from((1u32 << n) - 1);
    let unit = 2f64.powi(-n);
    let low = ((z + k * unit) * scale - 0.5).ceil();
    let high = ((z + 2.0 * k * unit) * scale + 0.5).floor();
    [low as u16, high as u16]
}

/// The bias bound of probe 4's `Depth16Unorm` case: 0.5 biased by 3 units is 0x8002 to 0x8005
/// (Metal stored 0x8003, RADV and ANV 0x8002, NVIDIA 0x8005).
#[test]
fn unorm_bias_bounds_span_vulkans_units() {
    assert_eq!(unorm_bias_bounds(0.5, 3.0, 16), [0x8002, 0x8005]);
    assert_eq!(unorm_bias_bounds(0.25, 1.0, 16), [0x4001, 0x4002]);
}

/// `texels` (white or zero `RGBA8Unorm`) as rows of `#` and `.`, `?` for any other texel.
fn picture(texels: &[u8], size: usize) -> Vec<String> {
    texels
        .chunks(4 * size)
        .map(|row| {
            row.chunks(4)
                .map(|t| match t {
                    [0xff, 0xff, 0xff, 0xff] => '#',
                    [0, 0, 0, 0] => '.',
                    _ => '?',
                })
                .collect()
        })
        .collect()
}

/// Separate depth and stencil textures filled by a first pass (depth 0.5, stencil 0x40), then a
/// second pass that loads both, draws at 0.25 under depth `Less` and a stencil increment, and
/// stores the depth while the stencil's store is `DontCare`: Metal kept the old stencil (probe
/// 4b), and so does the runtime, which copies a `DontCare` aspect back from its scratch image
/// not at all.
fn separate_load_dontcare(probe: &Probe) {
    let device = &probe.device;
    let depth = texture(device, PixelFormat::Depth32Float, 4);
    let stencil_texture = texture(device, PixelFormat::Stencil8, 4);
    let color = texture(device, PixelFormat::RGBA8Unorm, 4);
    let run = |load: LoadAction, stencil_store: StoreAction, draw: bool| {
        let mut descriptor = RenderPassDescriptor::new();
        let target = &mut descriptor.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::Store;
        descriptor.depth_attachment.texture = Some(depth.clone());
        descriptor.depth_attachment.load_action = load;
        descriptor.depth_attachment.clear_depth = 0.5;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        descriptor.stencil_attachment.texture = Some(stencil_texture.clone());
        descriptor.stencil_attachment.load_action = load;
        descriptor.stencil_attachment.clear_stencil = 0x40;
        descriptor.stencil_attachment.store_action = stencil_store;
        let command_buffer = probe.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        let mut held = Vec::new();
        if draw {
            let face = stencil(
                CompareFunction::Always,
                StencilOperation::IncrementClamp,
                0xff,
                0xff,
            );
            encoder
                .set_depth_stencil_state(&probe.state(CompareFunction::Less, true, Some(face)))
                .unwrap();
            probe.draw(
                &mut encoder,
                &mut held,
                WHITE.depth(PixelFormat::Depth32Float).stencil(),
                0.25,
                0.0,
            );
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    };
    run(LoadAction::Clear, StoreAction::Store, false);
    assert_eq!(texels(&depth, 4), depths(0x3f00_0000, 16), "the fill");
    assert_eq!(texels(&stencil_texture, 1), vec![0x40; 16], "the fill");
    run(LoadAction::Load, StoreAction::DontCare, true);
    assert_eq!(texels(&color, 4), rows(&FULL), "separate_load_dontcare");
    assert_eq!(
        texels(&depth, 4),
        depths(0x3e80_0000, 16),
        "separate_load_dontcare"
    );
    assert_eq!(
        texels(&stencil_texture, 1),
        vec![0x40; 16],
        "separate_load_dontcare: the old stencil"
    );
}

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// What Metal's validation stops in the render state, refused typed before anything draws, and
/// the state the runtime does not build, `NotImplemented` (the module docs). Each refusal keeps
/// the state set before it, so a draw after it draws as before.
#[test]
fn render_state_is_refused_as_metal_refuses_it() {
    let (messages, _cache, probe) = validated();
    let state = |s: RenderStateProblem| Problem::RenderState(s);
    let stored = probe.run(PASS, |p, e, h| {
        assert_eq!(
            e.set_scissor_rects(&[]),
            Err(invalid(
                "set_scissor_rects",
                state(RenderStateProblem::Count { count: 0 })
            ))
        );
        let whole = Viewport::new(0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
        // A NaN width, unmeasured (a NaN origin x draws nothing, `viewport_nan`).
        assert_eq!(
            e.set_viewport(Viewport::new(0.0, 0.0, f64::NAN, 4.0, 0.0, 1.0)),
            Err(invalid(
                "set_viewport",
                state(RenderStateProblem::ViewportNotFinite)
            ))
        );
        // 17 viewports hung an M4 without validation: refused here, never drawn.
        assert_eq!(
            e.set_viewports(&[whole; 17]),
            Err(invalid(
                "set_viewports",
                state(RenderStateProblem::Count { count: 17 })
            ))
        );
        // A negative width, which the vertex stage mirrors, and a depth range past [0, 1] where
        // the device takes it neither natively nor by the vertex stage's compensation
        // (`viewport_ranges.rs`).
        e.set_viewport(Viewport::new(4.0, 0.0, -4.0, 4.0, 0.0, 1.0))
            .unwrap();
        let range = Viewport::new(0.0, 0.0, 4.0, 4.0, -1.0, 2.0);
        match depth_ranges_refused(&p.device) {
            None => {
                e.set_viewport(range).unwrap();
                e.set_viewport(whole).unwrap();
            }
            Some(error) => assert_eq!(e.set_viewport(range), Err(error)),
        }
        // An infinite slope factor biases a flat triangle to NaN and a sloped one to -inf, which
        // Metal stores as 1 and 0 (Metal probes on the M3 Max and an M4): no static state gives
        // both.
        assert_eq!(
            e.set_depth_bias(0.0, f32::NEG_INFINITY, 0.0),
            Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias))
        );
        e.set_render_pipeline_state(&p.pipeline(WHITE)).unwrap();
        let depth = p.depth(0.5, 0.0);
        // An infinite bias a clamp bounds adds the clamp, which moves the viewport's depth
        // range past [0, 1], where Metal clamps the depth: drawn where the device takes the moved
        // range natively under the clamp to [0, 1]
        // (`viewport_ranges::saturated_and_offset_biases_store_metals_depths` checks the depths),
        // refused at the draw elsewhere.
        e.set_depth_bias(f32::INFINITY, 0.0, 0.125).unwrap();
        let drawn = draw_at(e, &depth, PrimitiveType::Triangle, 3);
        match native_depth_range(&p.device) {
            true => assert_eq!(drawn, Ok(())),
            false => assert_eq!(
                drawn,
                Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias))
            ),
        }
        e.set_depth_bias(0.0, 0.0, 0.0).unwrap();
        // `Never` draws, writing nothing (a probe, `depth_never.rs`). A state that needs a depth or
        // stencil attachment the pass lacks draws against its hidden ones (`hidden_depth.rs`).
        e.set_depth_stencil_state(&p.state(CompareFunction::Never, false, None))
            .unwrap();
        draw_at(e, &depth, PrimitiveType::Triangle, 3).unwrap();
        // After every refusal, the state set before draws.
        e.set_depth_stencil_state(&p.state(CompareFunction::Always, false, None))
            .unwrap();
        draw_at(e, &depth, PrimitiveType::Triangle, 3).unwrap();
        h.push(depth);
    });
    assert_eq!(stored.color, rows(&FULL), "the draw after the refusals");

    // A vertex function that writes a point size, under the line and triangle classes (probe 3c).
    let point = WHITE.vertex("rt6_point");
    for topology in [
        PrimitiveTopologyClass::Line,
        PrimitiveTopologyClass::Triangle,
    ] {
        assert_eq!(
            Pipe { topology, ..point }
                .state(&probe.device, &probe.library)
                .err(),
            Some(invalid(
                "new_render_pipeline_state",
                Problem::PointSizeTopology
            )),
            "{topology:?}"
        );
    }
    for topology in [
        PrimitiveTopologyClass::Unspecified,
        PrimitiveTopologyClass::Point,
    ] {
        let state = Pipe { topology, ..point }
            .state(&probe.device, &probe.library)
            .unwrap();
        // The state states what its vertex function writes: a point size, and no memory.
        assert!(state.writes_point_size(), "{topology:?}");
        assert!(!state.vertex_writes_memory(), "{topology:?}");
    }
    // A vertex function without a point-size output writes none, its module's 1.0 aside.
    let full = WHITE.state(&probe.device, &probe.library).unwrap();
    assert!(!full.writes_point_size() && !full.vertex_writes_memory());
    // Nor a render target slice or a viewport index.
    assert!(!full.writes_render_target_array_index() && !full.writes_viewport_array_index());
    // Nor a texture: the fragment function's color is no texture write.
    assert!(!full.vertex_writes_textures() && !full.fragment_writes_textures());

    // Separate depth and stencil textures of different sizes (probe 4b), as plain Metal renders
    // them.
    separate_sizes(&probe);
    quiet(probe, &messages);
}

/// Separate depth and stencil textures of different sizes, which Metal's validation stops, as
/// plain Metal renders them (the runtime's `depth_stencil_scratch.rs`): probe 4b's
/// `separate_sizes` (an 8x8 `Depth32Float` beside a 4x4 `Stencil8` and color, cleared to 0.5
/// and 0x40, drawn at 0.25 under `Less` and a stencil increment) shades the 4x4 render area and
/// clears the depth whole; our probe's `sep_d64_s4` (a 64x64 depth), whose clear Metal
/// stops at its first 32x32 tile, is `NotImplemented(Render)`.
fn separate_sizes(probe: &Probe) {
    let device = &probe.device;
    let color = texture(device, PixelFormat::RGBA8Unorm, 4);
    let pass = |depth: &Texture, stencil: &Texture| {
        let mut descriptor = RenderPassDescriptor::new();
        let target = &mut descriptor.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::Store;
        descriptor.depth_attachment.texture = Some(depth.clone());
        descriptor.depth_attachment.clear_depth = 0.5;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        descriptor.stencil_attachment.texture = Some(stencil.clone());
        descriptor.stencil_attachment.clear_stencil = 0x40;
        descriptor.stencil_attachment.store_action = StoreAction::Store;
        descriptor
    };
    let depth = texture(device, PixelFormat::Depth32Float, 8);
    let stencil_texture = texture(device, PixelFormat::Stencil8, 4);
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&depth, &stencil_texture))
        .unwrap();
    let mut held = Vec::new();
    let face = stencil(
        CompareFunction::Always,
        StencilOperation::IncrementClamp,
        0xff,
        0xff,
    );
    encoder
        .set_depth_stencil_state(&probe.state(CompareFunction::Less, true, Some(face)))
        .unwrap();
    probe.draw(
        &mut encoder,
        &mut held,
        WHITE.depth(PixelFormat::Depth32Float).stencil(),
        0.25,
        0.0,
    );
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    drop(held);
    assert_eq!(texels(&color, 4), rows(&FULL), "separate_sizes");
    // Rows 0 to 3: four drawn (0.25), then four cleared (0.5); rows 4 to 7 cleared.
    let expected: Vec<u8> = (0..8u32)
        .flat_map(|y| {
            (0..8u32).flat_map(move |x| {
                let word: u32 = if x < 4 && y < 4 {
                    0x3e80_0000
                } else {
                    0x3f00_0000
                };
                word.to_le_bytes()
            })
        })
        .collect();
    assert_eq!(texels(&depth, 4), expected, "separate_sizes");
    assert_eq!(
        texels(&stencil_texture, 1),
        vec![0x41; 16],
        "separate_sizes"
    );
    let large = texture(device, PixelFormat::Depth32Float, 64);
    let command_buffer = probe.queue.command_buffer().unwrap();
    assert_eq!(
        command_buffer
            .render_command_encoder(&pass(&large, &stencil_texture))
            .err(),
        Some(Error::NotImplemented(RuntimeGap::Render)),
        "sep_d64_s4"
    );
}

/// A vertex function that selects its viewport (`[[viewport_array_index]]`, the owned fixture
/// `rt6_viewports`): each draw one that a Metal probe measured (an M4, macOS 27.0.1, with and
/// without Metal's API validation), the fullscreen triangle into the probe's 4² target, one
/// instance a viewport. Each instance draws through the viewport of its index and that viewport's
/// scissor rectangle, or the one rectangle set: through two viewports, the second alone (base
/// instance 1), two scissors, one scissor, the default viewport, and sixteen viewports of one pixel
/// each. A device without `multiViewport` draws through one viewport
/// ([`with_one_viewport_every_index_draws_through_it`]) and skips these draws, printed; one with it
/// but without a `ViewportIndex` output or a geometry stage to store one refuses the pipeline,
/// typed, printed.
///
/// Scissor counts other than 1 and the viewport count, which Metal's validation stops, draw as
/// plain Metal drew them (probes `vs_*`, an M4 with macOS 27.0.1): each viewport through the
/// rectangle of its index, a viewport past two or more rectangles through none, and a rectangle
/// past the viewports unused.
///
/// An index at or past the viewports set draws through the first viewport and its scissor
/// rectangle: index 2 and 3 of two viewports, 4 and 3 of three, 17 of sixteen (probes 7 and 7b, an
/// M4), and index 3 of two under two scissor rectangles, the first one pixel inside the first
/// viewport, then the two swapped so the first lies outside it (probe 7c, an M4 and an M3 Max,
/// macOS 27.0.1, with and without API validation). A fragment function that reads the index
/// (`rt6_viewport_read`, red the index mod 256) reads it as written there: 3 of two, 17 of
/// sixteen, 5 with the default viewport alone, and 1 of two inside the count (probe 7c).
#[test]
fn a_vertex_function_draws_through_the_viewport_it_selects() {
    viewport_selection(&[]);
}

/// [`a_vertex_function_draws_through_the_viewport_it_selects`] with `shaderOutputViewportIndex`
/// withheld: the vertex function stores the viewport it selects to a varying and the synthesized
/// geometry stages store `ViewportIndex` from it, every draw (asserted from the pipeline it
/// executed). A device without `geometryShader` refuses the pipeline, typed, printed.
#[test]
fn through_a_geometry_stage_a_vertex_function_draws_through_the_viewport_it_selects() {
    viewport_selection(&[VkFeature::ShaderOutputViewportIndex]);
}

/// The probe over `rt6_viewports` on the test device without `withheld`.
fn viewports_probe(withheld: &[VkFeature]) -> (Messages, ScratchCache, Probe) {
    let (messages, cache, probe) = validated_without(withheld);
    let Probe { device, queue, .. } = probe;
    let ll = metal2vulkan_devtools::engine_dir().join("rt6_viewports.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let probe = Probe {
        device,
        queue,
        library,
    };
    (messages, cache, probe)
}

/// [`a_vertex_function_draws_through_the_viewport_it_selects`]'s draws on the test device without
/// `withheld`.
fn viewport_selection(withheld: &[VkFeature]) {
    let (messages, _cache, probe) = viewports_probe(withheld);
    let select = WHITE.vertex("rt6_select");
    let name = probe.device.name();
    let pipeline = match select.state(&probe.device, &probe.library) {
        Ok(pipeline) => pipeline,
        Err(Error::Unsupported(u)) => {
            eprintln!("refused: viewport selection on {name} without {withheld:?}: {u:?}");
            return quiet(probe, &messages);
        }
        Err(other) => panic!("viewport selection on {name}: {other}"),
    };
    let enabled = |feature| {
        probe
            .device
            .compile_device()
            .profile()
            .enabled()
            .features
            .contains(&feature)
    };
    if !enabled(VkFeature::MultiViewport) {
        eprintln!("SKIPPED viewport selection on {name}: one viewport (no multiViewport)");
        return quiet(probe, &messages);
    }
    // The synthesized geometry stages store the index where the device has no
    // `shaderOutputViewportIndex`.
    let stages = match enabled(VkFeature::ShaderOutputViewportIndex) {
        true => 0,
        false => 3,
    };
    let reading = Pipe {
        fragment: "rt6_viewport_read",
        ..select
    }
    .state(&probe.device, &probe.library)
    .unwrap();
    // `rt6_select` writes `[[viewport_array_index]]` and no render target slice, whatever its
    // fragment function reads.
    for state in [&pipeline, &reading] {
        assert!(state.writes_viewport_array_index());
        assert!(!state.writes_render_target_array_index());
    }
    let at = |x, y, w, h| Viewport::new(x, y, w, h, 0.0, 1.0);
    let two = vec![at(0.0, 0.0, 2.0, 2.0), at(2.0, 2.0, 2.0, 2.0)];
    let three = [two.clone(), vec![at(2.0, 0.0, 2.0, 2.0)]].concat();
    let sixteen: Vec<Viewport> = (0..16u32)
        .map(|i| at(f64::from(i % 4), f64::from(i / 4), 1.0, 1.0))
        .collect();
    let first = ["##..", "##..", "....", "...."];
    let corner = ["#...", "....", "....", "...."];
    let rect = ScissorRect::new;
    let cases = [
        (
            "select_two",
            two.clone(),
            vec![],
            2,
            0,
            ["##..", "##..", "..##", "..##"],
        ),
        (
            "select_second",
            two.clone(),
            vec![],
            1,
            1,
            ["....", "....", "..##", "..##"],
        ),
        (
            "select_scissors",
            two.clone(),
            vec![rect(0, 0, 1, 1), rect(3, 3, 1, 1)],
            2,
            0,
            ["#...", "....", "....", "...#"],
        ),
        (
            "select_one_scissor",
            two.clone(),
            vec![rect(1, 1, 2, 2)],
            2,
            0,
            ["....", ".#..", "..#.", "...."],
        ),
        ("select_one_viewport", vec![], vec![], 1, 0, FULL),
        (
            "select_three_viewports_two_scissors",
            [two.clone(), vec![at(2.0, 0.0, 2.0, 2.0)]].concat(),
            vec![rect(0, 0, 3, 3), rect(3, 3, 1, 1)],
            3,
            0,
            ["##..", "##..", "....", "...#"],
        ),
        (
            "select_two_viewports_three_scissors",
            two.clone(),
            vec![rect(0, 0, 1, 1), rect(3, 3, 1, 1), rect(0, 0, 4, 4)],
            2,
            0,
            ["#...", "....", "....", "...#"],
        ),
        (
            "select_one_viewport_two_scissors",
            vec![],
            vec![rect(0, 0, 2, 4), rect(2, 0, 2, 4)],
            2,
            0,
            ["##..", "##..", "##..", "##.."],
        ),
        ("select_sixteen", sixteen.clone(), vec![], 16, 0, FULL),
        // Probes 7 and 7b: at or past the viewports set, the first.
        ("past_2_of_2", two.clone(), vec![], 1, 2, first),
        ("past_3_of_2", two.clone(), vec![], 1, 3, first),
        ("past_4_of_3", three.clone(), vec![], 1, 4, first),
        ("past_3_of_3", three, vec![], 1, 3, first),
        ("past_17_of_16", sixteen.clone(), vec![], 1, 17, corner),
        // Probe 7c: the first viewport's scissor rectangle, and the second's inside the count.
        (
            "scissor_3_of_2",
            two.clone(),
            vec![rect(0, 0, 1, 1), rect(3, 3, 1, 1)],
            1,
            3,
            corner,
        ),
        (
            "scissor_3_of_2_swapped",
            two.clone(),
            vec![rect(2, 2, 2, 2), rect(0, 0, 2, 2)],
            1,
            3,
            ["....", "....", "....", "...."],
        ),
        (
            "scissor_1_of_2",
            two.clone(),
            vec![rect(0, 0, 1, 1), rect(3, 3, 1, 1)],
            1,
            1,
            ["....", "....", "....", "...#"],
        ),
    ];
    let draw = |pipeline: &RenderPipelineState,
                viewports: &[Viewport],
                scissors: &[ScissorRect],
                instances,
                base| {
        let (stored, executed) = probe.run_staged(PASS, |p, e, h| {
            e.set_render_pipeline_state(pipeline).unwrap();
            if !viewports.is_empty() {
                e.set_viewports(viewports).unwrap();
            }
            if !scissors.is_empty() {
                e.set_scissor_rects(scissors).unwrap();
            }
            let depth = p.depth(0.5, 0.0);
            e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, instances, base)
                .unwrap();
            h.push(depth);
        });
        assert_eq!(
            executed,
            [stages],
            "the geometry stages of the draw on {name}"
        );
        stored
    };
    for (case, viewports, scissors, instances, base, metal) in cases {
        let stored = draw(&pipeline, &viewports, &scissors, instances, base);
        assert_eq!(
            picture(&stored.color, 4),
            metal,
            "{case} on {name}: Metal's texels"
        );
        eprintln!("viewport selection {case} on {name}: Metal's texels");
    }
    // Probe 7c: the index the fragment function reads, in red, where the triangle draws.
    let reads = [
        (
            "read_1_of_2",
            two.clone(),
            1,
            ["....", "....", "..##", "..##"],
        ),
        ("read_3_of_2", two, 3, first),
        ("read_17_of_16", sixteen, 17, corner),
        ("read_5_of_default", vec![], 5, FULL),
    ];
    for (case, viewports, index, drawn) in reads {
        let stored = draw(&reading, &viewports, &[], 1, index);
        let metal: Vec<u8> = drawn
            .iter()
            .flat_map(|row| row.chars())
            .flat_map(|c| match c {
                '#' => [index as u8, 0xff, 0, 0xff],
                _ => [0; 4],
            })
            .collect();
        assert_eq!(stored.color, metal, "{case} on {name}: Metal's texels");
        eprintln!("viewport read {case} on {name}: Metal's texels");
    }
    // Mirrored and unmirrored viewports in one draw (a Metal probe, an M3 Max and an M4 alike,
    // plain and validated): each viewport mirrors the draw by its own width and flips its facing,
    // so culling drops the triangle in one and keeps it in the other (the fullscreen triangle is
    // back-facing under Metal's clockwise front face in a viewport of positive width).
    let mirrored = |x: f64, y, w: f64, h| Viewport::new(x + w, y, -w, h, 0.0, 1.0);
    let both = ["##..", "##..", "..##", "..##"];
    let (left, right) = (first, ["....", "....", "..##", "..##"]);
    let mixed = [
        (
            "plain_then_mirrored",
            vec![at(0.0, 0.0, 2.0, 2.0), mirrored(2.0, 2.0, 2.0, 2.0)],
            [both, right, left],
        ),
        (
            "mirrored_then_plain",
            vec![mirrored(0.0, 0.0, 2.0, 2.0), at(2.0, 2.0, 2.0, 2.0)],
            [both, left, right],
        ),
    ];
    for (case, viewports, metal) in mixed {
        for (cull, metal) in [CullMode::None, CullMode::Back, CullMode::Front]
            .into_iter()
            .zip(metal)
        {
            let (stored, executed) = probe.run_staged(PASS, |p, e, h| {
                e.set_render_pipeline_state(&pipeline).unwrap();
                e.set_viewports(&viewports).unwrap();
                e.set_cull_mode(cull);
                let depth = p.depth(0.5, 0.0);
                e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 0, 3, 2, 0)
                    .unwrap();
                h.push(depth);
            });
            assert_eq!(executed, [stages], "{case} {cull:?} on {name}");
            assert_eq!(
                picture(&stored.color, 4),
                metal,
                "{case} {cull:?} on {name}: Metal's texels"
            );
            eprintln!("mirrored viewports {case} {cull:?} on {name}: Metal's texels");
        }
    }
    quiet(probe, &messages);
}

/// On a device without `multiViewport` a draw has one viewport, so `rt6_select` stores no
/// `ViewportIndex` (no geometry stage either) and every index draws through viewport 0, as Metal
/// draws an index at or past the viewports set (probe 7, an M4; a Metal probe on an M3 Max with
/// macOS 27.0.1 drew index 2 of one viewport through it and read 2), the fragment function reading
/// the index as written. The pipeline still states that its vertex function writes the index, and a
/// draw of it while the encoder holds two viewports, where Metal would select the second, is
/// refused, `Unsupported(LayeredRendering)` needing `multiViewport`.
#[test]
fn with_one_viewport_every_index_draws_through_it() {
    let withheld = [
        VkFeature::ShaderOutputViewportIndex,
        VkFeature::MultiViewport,
    ];
    let (messages, _cache, probe) = viewports_probe(&withheld);
    let name = probe.device.name();
    let select = WHITE.vertex("rt6_select");
    let pipeline = select.state(&probe.device, &probe.library).unwrap();
    let reading = Pipe {
        fragment: "rt6_viewport_read",
        ..select
    }
    .state(&probe.device, &probe.library)
    .unwrap();
    for state in [&pipeline, &reading] {
        assert!(state.writes_viewport_array_index());
    }
    let one = vec![Viewport::new(0.0, 0.0, 2.0, 2.0, 0.0, 1.0)];
    let first = ["##..", "##..", "....", "...."];
    let draw = |pipeline: &RenderPipelineState, viewports: &[Viewport], base| {
        let (stored, executed) = probe.run_staged(PASS, |p, e, h| {
            e.set_render_pipeline_state(pipeline).unwrap();
            if !viewports.is_empty() {
                e.set_viewports(viewports).unwrap();
            }
            let depth = p.depth(0.5, 0.0);
            e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, base)
                .unwrap();
            h.push(depth);
        });
        assert_eq!(executed, [0], "no geometry stage on {name}");
        stored
    };
    for (case, viewports, base, metal) in [
        ("select_one_viewport", vec![], 0, FULL),
        ("past_2_of_1", one.clone(), 2, first),
    ] {
        let stored = draw(&pipeline, &viewports, base);
        assert_eq!(
            picture(&stored.color, 4),
            metal,
            "{case} on {name}: Metal's texels"
        );
    }
    for (case, viewports, index, drawn) in [
        ("read_5_of_default", vec![], 5u32, FULL),
        ("read_2_of_1", one, 2, first),
    ] {
        let stored = draw(&reading, &viewports, index);
        let metal: Vec<u8> = drawn
            .iter()
            .flat_map(|row| row.chars())
            .flat_map(|c| match c {
                '#' => [index as u8, 0xff, 0, 0xff],
                _ => [0; 4],
            })
            .collect();
        assert_eq!(stored.color, metal, "{case} on {name}: Metal's texels");
    }
    let two = [
        Viewport::new(0.0, 0.0, 2.0, 2.0, 0.0, 1.0),
        Viewport::new(2.0, 2.0, 2.0, 2.0, 0.0, 1.0),
    ];
    probe.run(PASS, |p, e, h| {
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_viewports(&two).unwrap();
        let depth = p.depth(0.5, 0.0);
        e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        assert_eq!(
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 2, 0),
            Err(Error::Unsupported(metal2vulkan::error::Unsupported::needs(
                Feature::LayeredRendering,
                Capability::Feature(VkFeature::MultiViewport),
            ))),
            "two viewports on {name}"
        );
        h.push(depth);
    });
    eprintln!("one viewport on {name}: Metal's texels, two refused");
    quiet(probe, &messages);
}
