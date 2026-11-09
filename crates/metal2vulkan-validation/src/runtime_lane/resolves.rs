//! Device tests of multisample resolves (its fourth part) on the device the cfg selects, under the
//! validation layer.
//!
//! Each pass is one that a Metal probe measured (an M4 with macOS 27.0.1, each case with and
//! without Metal's API validation), drawn with the same per-sample values (the owned fixture
//! `rt6_resolves`) into a 4x 4² target cleared to zero, and resolved; the resolve texture is
//! compared with the one Metal resolved. A depth/stencil pass gives its samples the probe's depths
//! 0.5, 0.1, 0.9 and 0.3 and stencils 20 to 23, one sample a draw through a returned sample mask,
//! the depth from the vertex and the stencil from the reference.
//!
//! A color resolve averages: Vulkan leaves the average's rounding (and an sRGB average's space)
//! to the device, and Metal rounds a unorm tie up (127.5 to 128). The runtime resolves a color
//! attachment by the device's average, so each resolve asserts Metal's texels on a driver measured
//! to resolve as Metal does (`DriverFact::ColorResolveAverageIsMetals`), and elsewhere within the
//! bound `device-rounded-resolve` states: a unorm channel within one code (an 8-bit one within
//! two, an sRGB channel's bound), a half within one unit (one half ulp), a 32-bit float's
//! difference printed and not bounded. A resolve that differs is printed as a near-equivalent.
//! Depth and stencil resolves pick one sample, which Vulkan states exactly. What Metal's validation
//! stops, the runtime refuses typed, apart from a color attachment's resolve that names no texture
//! or a texture under a store action that does not resolve, which resolves nothing and leaves that
//! texture untouched, and a multisampled color attachment's resolve into its format's sRGB twin,
//! which resolves in the resolve texture's encoding, as plain Metal's.

use super::degrade_notices::{assert_once, counts};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::ResolveModeFlags as Mode;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Attachment, BlitOption, Buffer, ClearColor, CommandQueue, Degradation, DepthStencilState,
    Device, DeviceOptions, Error, Invalid, Library, LoadAction, MultisampleDepthResolveFilter,
    MultisampleStencilResolveFilter, Origin, PipelineOptions, Problem, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPassProblem, RenderPipelineState,
    ResourceOptions, RuntimeGap, Size, StoreAction, Texture, TextureDescriptor,
};
use std::cell::RefCell;
use std::path::Path;

/// The lane's device, queue and the fixture's library (`memoryless.rs` draws with them too).
pub(super) struct Lane {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    pub(super) library: Library,
}

pub(super) fn lane() -> (Messages, ScratchCache, Lane) {
    lane_with(|_| {})
}

/// [`lane`], its device made with `adjust`'s options (a measurement's, without the validation
/// layer).
pub(super) fn lane_with(adjust: impl FnOnce(&mut DeviceOptions)) -> (Messages, ScratchCache, Lane) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/rt6_resolves.ll");
    let bitcode = llvm_assemble(&ll).unwrap();
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        library: device.new_library_with_air(&bitcode).unwrap(),
        device,
    };
    (messages, cache, lane)
}

pub(super) fn quiet(lane: Lane, messages: &Messages) {
    drop(lane);
    layer_reported_nothing_at_all(messages);
}

/// The per-sample color fragment of the probe's `f_samples`: red s/3, green 1 on sample 3, blue
/// 0.5, alpha 1.
pub(super) const SAMPLES: &str = "rt6_resolve_samples";
/// The probe's `f_samples_float`: s, 0.1 s, -s, 0.25 s.
const SAMPLE_VALUES: &str = "rt6_resolve_sample_values";
pub(super) const WHITE: &str = "rt6_resolve_white";
pub(super) const FULL: &str = "rt6_resolve_full";
const HALF: &str = "rt6_resolve_half";
/// White on the samples of a mask (`buffer(0)`) alone.
pub(super) const MASKED: &str = "rt6_resolve_masked";

/// The probe's 4x targets are 4².
pub(super) const SIZE: u64 = 4;
const SAMPLES_PER_PIXEL: u64 = 4;

impl Lane {
    /// A pipeline of `vertex` and `fragment` drawing `color` (and `color1` at index 1), the
    /// depth and stencil formats, and `samples`.
    pub(super) fn pipeline(
        &self,
        functions: (&str, &str),
        colors: &[PixelFormat],
        depth_stencil: (Option<PixelFormat>, Option<PixelFormat>),
        samples: u64,
    ) -> RenderPipelineState {
        self.pipeline_of(functions, colors, depth_stencil, (samples, false))
    }

    /// [`Lane::pipeline`] of `samples`, made with `supportIndirectCommandBuffers` where
    /// `indirect`.
    pub(super) fn pipeline_of(
        &self,
        (vertex, fragment): (&str, &str),
        colors: &[PixelFormat],
        depth_stencil: (Option<PixelFormat>, Option<PixelFormat>),
        (samples, indirect): (u64, bool),
    ) -> RenderPipelineState {
        let function = |name| self.library.new_function_with_name(name).unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            function(vertex).function_descriptor(),
        ));
        descriptor.fragment = Some(StageSource::Function(
            function(fragment).function_descriptor(),
        ));
        for (index, &pf) in colors.iter().enumerate() {
            descriptor.color_attachments[index] = Some(ColorAttachmentDescriptor::new(pf));
        }
        descriptor.depth_format = depth_stencil.0;
        descriptor.stencil_format = depth_stencil.1;
        descriptor.raster_sample_count = samples as u32;
        descriptor.support_indirect_command_buffers = indirect;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// A texture of `pf`: 4x multisampled when `samples` is 4, else 2D (an array of `slices`
    /// with `levels` levels) of `size`², of `usage`, shared or private.
    pub(super) fn texture(&self, pf: PixelFormat, shape: Shape, usage: TextureUsage) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(pf, shape.size, shape.size, false);
        descriptor.texture_type = match (shape.samples, shape.slices) {
            (1, 1) => TextureType::Type2D,
            (1, _) => TextureType::Type2DArray,
            _ => TextureType::Type2DMultisample,
        };
        descriptor.sample_count = shape.samples;
        descriptor.array_length = shape.slices;
        descriptor.mipmap_level_count = shape.levels;
        descriptor.usage = usage;
        descriptor.resource_options = if shape.shared {
            ResourceOptions::shared()
        } else {
            ResourceOptions::private()
        };
        self.device.new_texture(&descriptor).unwrap()
    }

    /// A 4x 4² render target of `pf`.
    pub(super) fn msaa(&self, pf: PixelFormat) -> Texture {
        self.texture(pf, MSAA, TextureUsage::RENDER_TARGET)
    }

    /// A private 4² render target of `pf`, a resolve texture.
    pub(super) fn single(&self, pf: PixelFormat) -> Texture {
        self.texture(pf, ONE, TextureUsage::RENDER_TARGET)
    }

    /// The texels of `texture`'s `level` and `slice`, `bytes` each (one aspect of a combined
    /// depth/stencil texture by `option`), through a blit into a shared buffer.
    pub(super) fn read(
        &self,
        texture: &Texture,
        (level, slice): (u64, u64),
        bytes: u64,
        option: BlitOption,
    ) -> Vec<u8> {
        let width = (texture.width() >> level).max(1);
        let height = (texture.height() >> level).max(1);
        let length = bytes * width * height;
        let buffer = buffer_with(&self.device, &vec![0; length as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            slice,
            level,
            Origin::new(0, 0, 0),
            Size::new(width, height, 1),
            &buffer,
            0,
            bytes * width,
            length,
            option,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        read_buffer(&buffer, 0, length).unwrap()
    }

    /// Encode `descriptor`'s pass with `body`, commit and wait; the encoder's refusal of the
    /// pass, if any.
    pub(super) fn run(
        &self,
        descriptor: &RenderPassDescriptor,
        body: impl FnOnce(&Lane, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    ) -> Result<(), Error> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(descriptor)?;
        let mut held = Vec::new();
        body(self, &mut encoder, &mut held);
        encoder.end_encoding()?;
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        drop(held);
        Ok(())
    }

    /// One draw of the fullscreen triangle at depth `z` with `pipeline`.
    pub(super) fn draw(
        &self,
        encoder: &mut RenderCommandEncoder,
        held: &mut Vec<Buffer>,
        pipeline: &RenderPipelineState,
        z: f32,
    ) {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let depth = buffer_with(&self.device, &z.to_le_bytes()).unwrap();
        encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        held.push(depth);
    }
}

/// A texture's shape: its side, samples, slices and levels, and whether it is shared.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) size: u64,
    pub(super) samples: u64,
    pub(super) slices: u64,
    pub(super) levels: u64,
    pub(super) shared: bool,
}

pub(super) const MSAA: Shape = Shape {
    size: SIZE,
    samples: SAMPLES_PER_PIXEL,
    slices: 1,
    levels: 1,
    shared: false,
};

pub(super) const ONE: Shape = Shape {
    size: SIZE,
    samples: 1,
    slices: 1,
    levels: 1,
    shared: false,
};

/// A pass of `msaa` cleared to zero (alpha too) and stored by `store`, resolving into `resolve`
/// at `level` and `slice`.
pub(super) fn color_pass(
    msaa: &Texture,
    resolve: Option<&Texture>,
    store: StoreAction,
    (level, slice): (u64, u64),
) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(msaa.clone());
    target.load_action = LoadAction::Clear;
    target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    target.store_action = store;
    target.resolve_texture = resolve.cloned();
    target.resolve_level = level;
    target.resolve_slice = slice;
    descriptor
}

/// `count` texels of `texel`.
pub(super) fn repeat(texel: &[u8], count: usize) -> Vec<u8> {
    texel.repeat(count)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// How a format's channels are encoded, for "within one unit".
#[derive(Clone, Copy)]
pub(super) enum Channels {
    /// 8-bit unorm channels.
    Unorm8,
    /// 16-bit unorm channels.
    Unorm16,
    /// Half floats.
    Half,
    /// 32-bit floats.
    Float,
}

impl Channels {
    /// The channels of `bytes` as integers one unit apart: unorm values, or float bit patterns
    /// (adjacent values of one sign differ by one).
    fn values(self, bytes: &[u8]) -> Vec<i64> {
        match self {
            Channels::Unorm8 => bytes.iter().map(|&b| i64::from(b)).collect(),
            Channels::Unorm16 | Channels::Half => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i64::from(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Channels::Float => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect(),
        }
    }
}

/// Print the device's resolved texels beside Metal's, and assert them Metal's on a driver whose
/// average is measured to be Metal's, else within the bound `device-rounded-resolve` states for
/// `channels` (the module docs), for half floats each channel within one unit on every driver;
/// the differing cases are named in `differ`. A resolve that is not Metal's is printed as a
/// near-equivalent, never as Metal's answer.
pub(super) fn near(
    lane: &Lane,
    name: &str,
    resolved: &[u8],
    metal: &[u8],
    channels: Channels,
    differ: &RefCell<Vec<String>>,
) {
    let device = lane.device.name();
    let metals = lane
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::ColorResolveAverageIsMetals);
    let exact = resolved == metal;
    eprintln!(
        "resolve {name} on {device}: {} ({})",
        hex(resolved),
        if exact {
            "Metal's".to_owned()
        } else {
            format!("near-equivalent, not Metal's {}", hex(metal))
        }
    );
    let (got, want) = (channels.values(resolved), channels.values(metal));
    let bound = match (channels, metals) {
        (Channels::Half, _) => Some(1),
        (_, true) => Some(0),
        (Channels::Unorm8, false) => Some(2),
        (Channels::Unorm16, false) => Some(1),
        (Channels::Float, false) => None,
    };
    let within = got.len() == want.len()
        && bound.is_none_or(|bound| got.iter().zip(&want).all(|(a, b)| (a - b).abs() <= bound));
    if !(exact || within) {
        differ.borrow_mut().push(format!(
            "{name}: {} where Metal resolved {}",
            hex(resolved),
            hex(metal)
        ));
    }
}

/// Probe 6's per-sample resolves of each color format: Metal's, or within the device's announced
/// bound of Metal's (the module docs).
#[test]
fn color_resolves_are_metals() {
    let (messages, _cache, lane) = lane();
    let differ = RefCell::new(Vec::new());
    let texels = (SIZE * SIZE) as usize;
    // Each case: its name in the probe, the format, the fragment, and Metal's texel.
    let cases = [
        (
            "res_unorm_samples",
            PixelFormat::RGBA8Unorm,
            SAMPLES,
            vec![0x80, 0x40, 0x80, 0xff],
            Channels::Unorm8,
        ),
        (
            "res_bgra_samples",
            PixelFormat::BGRA8Unorm,
            SAMPLES,
            vec![0x80, 0x40, 0x80, 0xff],
            Channels::Unorm8,
        ),
        (
            "res_srgb_samples",
            PixelFormat::RGBA8UnormSRGB,
            SAMPLES,
            vec![0xbb, 0x89, 0xbc, 0xff],
            Channels::Unorm8,
        ),
        (
            "res_unorm16_samples",
            PixelFormat::RGBA16Unorm,
            SAMPLES,
            vec![0x00, 0x80, 0x00, 0x40, 0x00, 0x80, 0xff, 0xff],
            Channels::Unorm16,
        ),
        (
            "res_half_samples",
            PixelFormat::RGBA16Float,
            SAMPLE_VALUES,
            vec![0x00, 0x3e, 0xcc, 0x30, 0x00, 0xbe, 0x00, 0x36],
            Channels::Half,
        ),
        (
            "res_float_samples",
            PixelFormat::RGBA32Float,
            SAMPLE_VALUES,
            [1.5f32, 0.15, -1.5, 0.375]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            Channels::Float,
        ),
        (
            "res_r32float_samples",
            PixelFormat::R32Float,
            SAMPLE_VALUES,
            1.5f32.to_le_bytes().to_vec(),
            Channels::Float,
        ),
    ];
    for (name, pf, fragment, texel, channels) in cases {
        let bytes = texel.len() as u64;
        let msaa = lane.msaa(pf);
        let resolve = lane.single(pf);
        let pipeline = lane.pipeline((FULL, fragment), &[pf], (None, None), 4);
        lane.run(
            &color_pass(
                &msaa,
                Some(&resolve),
                StoreAction::MultisampleResolve,
                (0, 0),
            ),
            |l, e, h| l.draw(e, h, &pipeline, 0.5),
        )
        .unwrap();
        let resolved = lane.read(&resolve, (0, 0), bytes, BlitOption::None);
        near(
            &lane,
            name,
            &resolved,
            &repeat(&texel, texels),
            channels,
            &differ,
        );
    }
    // Probe 6's `res_unorm_edge`: half the target white, its diagonal pixels half covered (two
    // samples of four), 127.5 resolved to 128.
    let msaa = lane.msaa(PixelFormat::RGBA8Unorm);
    let resolve = lane.single(PixelFormat::RGBA8Unorm);
    let pipeline = lane.pipeline((HALF, WHITE), &[PixelFormat::RGBA8Unorm], (None, None), 4);
    lane.run(
        &color_pass(
            &msaa,
            Some(&resolve),
            StoreAction::MultisampleResolve,
            (0, 0),
        ),
        |_, e, _| {
            e.set_render_pipeline_state(&pipeline).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        },
    )
    .unwrap();
    let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
    near(
        &lane,
        "res_unorm_edge",
        &resolved,
        &half_covered([0x80; 4], [0xff; 4]),
        Channels::Unorm8,
        &differ,
    );
    let differ = differ.into_inner();
    assert!(
        differ.is_empty(),
        "resolves not Metal's (or a half one past one unit): {differ:#?}"
    );
    quiet(lane, &messages);
}

/// The texels of probe 6's half-covered target: `diagonal` on the diagonal, `below` under it
/// (rows from the top), zero above it.
fn half_covered(diagonal: [u8; 4], below: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::new();
    for row in 0..SIZE {
        for column in 0..SIZE {
            out.extend(match column.cmp(&row) {
                std::cmp::Ordering::Equal => diagonal,
                std::cmp::Ordering::Less => below,
                std::cmp::Ordering::Greater => [0; 4],
            });
        }
    }
    out
}

/// Probe 6's resolves that pick no tie: the samples a pass stores and resolves are loaded and
/// resolved again by a later pass; a resolve writes only its level and slice; a pass without
/// draws resolves its clear color; a late resolving store action; one of two targets resolved;
/// and a resolve texture without `RenderTarget` usage, which Metal resolves into.
#[test]
fn resolves_write_where_metal_writes() {
    let (messages, _cache, lane) = lane();
    let differ = RefCell::new(Vec::new());
    let rgba8 = PixelFormat::RGBA8Unorm;
    let texels = (SIZE * SIZE) as usize;
    let samples = lane.pipeline((FULL, SAMPLES), &[rgba8], (None, None), 4);

    // `res_store_and_resolve`: half the target per sample, then a pass loading the samples.
    let msaa = lane.msaa(rgba8);
    let (first, second) = (lane.single(rgba8), lane.single(rgba8));
    let half = lane.pipeline((HALF, SAMPLES), &[rgba8], (None, None), 4);
    lane.run(
        &color_pass(
            &msaa,
            Some(&first),
            StoreAction::StoreAndMultisampleResolve,
            (0, 0),
        ),
        |_, e, _| {
            e.set_render_pipeline_state(&half).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        },
    )
    .unwrap();
    let mut again = color_pass(
        &msaa,
        Some(&second),
        StoreAction::MultisampleResolve,
        (0, 0),
    );
    again.color_attachments[0].load_action = LoadAction::Load;
    lane.run(&again, |_, _, _| {}).unwrap();
    // Samples 2 and 3 cover a diagonal pixel: red (170 + 255) / 4, green 255 / 4, blue
    // (128 + 128) / 4 and alpha 510 / 4, each rounded to nearest.
    let metal = half_covered([0x6a, 0x40, 0x40, 0x80], [0x80, 0x40, 0x80, 0xff]);
    for (name, texture) in [
        ("res_store_and_resolve", &first),
        ("res_store_and_resolve again", &second),
    ] {
        let resolved = lane.read(texture, (0, 0), 4, BlitOption::None);
        near(&lane, name, &resolved, &metal, Channels::Unorm8, &differ);
    }

    // `res_level_slice`: level 1 of slice 1 of an 8² array of 2 slices and 2 levels, every
    // other subresource keeping its 0x11 bytes.
    let array = lane.texture(
        rgba8,
        Shape {
            size: 8,
            samples: 1,
            slices: 2,
            levels: 2,
            shared: true,
        },
        TextureUsage::RENDER_TARGET,
    );
    for slice in 0..2 {
        for level in 0..2u64 {
            let side = 8 >> level;
            array
                .replace_region(
                    Region::new_2d(0, 0, side, side),
                    level,
                    slice,
                    &vec![0x11; (4 * side * side) as usize],
                    4 * side,
                    4 * side * side,
                )
                .unwrap();
        }
    }
    lane.run(
        &color_pass(&msaa, Some(&array), StoreAction::MultisampleResolve, (1, 1)),
        |l, e, h| l.draw(e, h, &samples, 0.5),
    )
    .unwrap();
    for slice in 0..2 {
        for level in 0..2u64 {
            let side = (8 >> level) as usize;
            let resolved = lane.read(&array, (level, slice), 4, BlitOption::None);
            let name = format!("res_level_slice slice {slice} level {level}");
            if (slice, level) == (1, 1) {
                let metal = repeat(&[0x80, 0x40, 0x80, 0xff], texels);
                near(&lane, &name, &resolved, &metal, Channels::Unorm8, &differ);
            } else {
                assert_eq!(resolved, vec![0x11; 4 * side * side], "{name}");
            }
        }
    }

    // `res_no_draw`: the clear color (0.25, 0.5, 0.75, 1) resolved.
    let resolve = lane.single(rgba8);
    let mut cleared = color_pass(
        &msaa,
        Some(&resolve),
        StoreAction::MultisampleResolve,
        (0, 0),
    );
    cleared.color_attachments[0].clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
    lane.run(&cleared, |_, _, _| {}).unwrap();
    let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
    near(
        &lane,
        "res_no_draw",
        &resolved,
        &repeat(&[0x40, 0x80, 0xbf, 0xff], texels),
        Channels::Unorm8,
        &differ,
    );

    // `res_late_resolve`: the store action `Unknown`, then `MultisampleResolve`.
    let resolve = lane.single(rgba8);
    lane.run(
        &color_pass(&msaa, Some(&resolve), StoreAction::Unknown, (0, 0)),
        |l, e, h| {
            e.set_color_store_action(StoreAction::MultisampleResolve, 0)
                .unwrap();
            l.draw(e, h, &samples, 0.5);
        },
    )
    .unwrap();
    let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
    let metal = repeat(&[0x80, 0x40, 0x80, 0xff], texels);
    near(
        &lane,
        "res_late_resolve",
        &resolved,
        &metal,
        Channels::Unorm8,
        &differ,
    );

    // `res_two_targets`: color 0 resolved, color 1 stored.
    let resolve = lane.single(rgba8);
    let mut two = color_pass(
        &msaa,
        Some(&resolve),
        StoreAction::MultisampleResolve,
        (0, 0),
    );
    let second_target = lane.msaa(rgba8);
    two.color_attachments[1].texture = Some(second_target.clone());
    two.color_attachments[1].load_action = LoadAction::Clear;
    two.color_attachments[1].store_action = StoreAction::Store;
    let both = lane.pipeline((FULL, SAMPLES), &[rgba8, rgba8], (None, None), 4);
    lane.run(&two, |l, e, h| l.draw(e, h, &both, 0.5)).unwrap();
    let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
    near(
        &lane,
        "res_two_targets",
        &resolved,
        &metal,
        Channels::Unorm8,
        &differ,
    );

    // `ref_usage`: a shared resolve texture of `ShaderRead` usage only, which Metal resolves
    // into (through a scratch image here).
    let read_only = lane.texture(
        rgba8,
        Shape {
            shared: true,
            ..ONE
        },
        TextureUsage::SHADER_READ,
    );
    lane.run(
        &color_pass(
            &msaa,
            Some(&read_only),
            StoreAction::MultisampleResolve,
            (0, 0),
        ),
        |l, e, h| l.draw(e, h, &samples, 0.5),
    )
    .unwrap();
    let resolved = lane.read(&read_only, (0, 0), 4, BlitOption::None);
    near(
        &lane,
        "ref_usage",
        &resolved,
        &metal,
        Channels::Unorm8,
        &differ,
    );

    let differ = differ.into_inner();
    assert!(
        differ.is_empty(),
        "resolves not Metal's (or a half one past one unit): {differ:#?}"
    );
    quiet(lane, &messages);
}

/// The probe's per-sample depths and stencils, sample by sample.
pub(super) const DEPTHS: [f32; 4] = [0.5, 0.1, 0.9, 0.3];
pub(super) const STENCILS: [u32; 4] = [20, 21, 22, 23];

/// Probe 6's depth and stencil resolves: each filter takes the sample Metal took. `Min` and
/// `Max` with the stencil's `DepthResolvedSample` take the stencil of the sample of the smallest
/// or largest depth (probe 7), which no Vulkan mode states: the runtime draws it after the pass
/// from the attachment's samples, sampled, so those cases' attachment is made for shader reads
/// too (a device that cannot make it is printed as a "refused:" line). A `Min` or `Max` depth
/// the compile device's resolve modes lack is drawn likewise where the driver's fragment depth
/// stores values unchanged (`DriverFact::FragmentDepthStoresUnchanged`; every case's depth is a
/// 32-bit float), printed as a "drawn:" line. Probe 6e's: a combined texture resolving one aspect
/// while its other attachment is not resolved keeps the resolve texture's other aspect (an
/// earlier pass cleared it to depth 0.75 and stencil 7). Those, and both aspects in different
/// modes, also run on a device that resolves the pair only in one mode (one subpass per aspect,
/// `resolve.rs`). A filter the device's resolve modes lack where no draw takes it is refused
/// (`NotImplemented`, printed as a "refused:" line).
#[test]
fn depth_and_stencil_resolve_the_samples_metal_picks() {
    use MultisampleDepthResolveFilter as D;
    use MultisampleStencilResolveFilter as S;
    let (messages, _cache, lane) = lane();
    let device_resolves = device_resolves(&lane.device);
    let drawn = drawn_depth(&lane.device);
    let combined = PixelFormat::Depth32FloatStencil8;
    let rgba8 = PixelFormat::RGBA8Unorm;
    let texels = (SIZE * SIZE) as usize;
    let state = |stencil: bool| {
        let mut always = StencilDescriptor::default();
        always.stencil_compare_function = CompareFunction::Always;
        always.depth_stencil_pass_operation = StencilOperation::Replace;
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Always;
        state.depth_write_enabled = true;
        if stencil {
            state.front_face_stencil = always;
            state.back_face_stencil = always;
        }
        lane.device.new_depth_stencil_state(&state).unwrap()
    };
    // Each case: the depth filter (`None`: the depth attachment is not resolved), the stencil
    // attachment (`None`: none; `Some(None)`: not resolved) and the resolve texture's depth and
    // stencil Metal left (`None`: no stencil), or `Err` for a pair the runtime does not build.
    type Metal = Result<(f32, Option<u32>), ()>;
    type Case = (&'static str, Option<D>, Option<Option<S>>, Metal);
    let cases: [Case; 9] = [
        (
            "ds_sample0_sample0",
            Some(D::Sample0),
            Some(Some(S::Sample0)),
            Ok((0.5, Some(20))),
        ),
        (
            "ds_min_sample0",
            Some(D::Min),
            Some(Some(S::Sample0)),
            Ok((0.1, Some(20))),
        ),
        (
            "ds_max_sample0",
            Some(D::Max),
            Some(Some(S::Sample0)),
            Ok((0.9, Some(20))),
        ),
        (
            "ds_sample0_depth",
            Some(D::Sample0),
            Some(Some(S::DepthResolvedSample)),
            Ok((0.5, Some(20))),
        ),
        // Probe 7: the stencil of sample 1 (depth 0.1) and of sample 2 (depth 0.9).
        (
            "ds_min_depth",
            Some(D::Min),
            Some(Some(S::DepthResolvedSample)),
            Ok((0.1, Some(21))),
        ),
        (
            "ds_max_depth",
            Some(D::Max),
            Some(Some(S::DepthResolvedSample)),
            Ok((0.9, Some(22))),
        ),
        ("d_min", Some(D::Min), None, Ok((0.1, None))),
        // Probe 6e: the other aspect is not resolved, and the resolve texture keeps its own.
        (
            "ds_min_unresolved",
            Some(D::Min),
            Some(None),
            Ok((0.1, Some(7))),
        ),
        (
            "ds_unresolved_sample0",
            None,
            Some(Some(S::Sample0)),
            Ok((0.75, Some(20))),
        ),
    ];
    for (name, depth_filter, stencil, metal) in cases {
        let pf = if stencil.is_some() {
            combined
        } else {
            PixelFormat::Depth32Float
        };
        let depth_mode = depth_filter.map(|filter| match filter {
            D::Sample0 => Mode::SAMPLE_ZERO,
            D::Min => Mode::MIN,
            D::Max => Mode::MAX,
            other => panic!("{name}: a filter the cases do not name: {other:?}"),
        });
        // A depth picked where the device has no mode for it, and a stencil picked by depth, are
        // drawn from the samples, read through sampled views.
        let depth_drawn = depth_mode.is_some_and(&drawn);
        let picked = depth_filter.is_some_and(|f| f != D::Sample0)
            && stencil == Some(Some(S::DepthResolvedSample));
        if depth_drawn {
            eprintln!("drawn: {name}: the depth {depth_mode:?}");
        }
        let msaa = if picked || depth_drawn {
            let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            let mut descriptor = TextureDescriptor::texture_2d(pf, MSAA.size, MSAA.size, false);
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = MSAA.samples;
            descriptor.usage = usage;
            descriptor.resource_options = ResourceOptions::private();
            match lane.device.new_texture(&descriptor) {
                Ok(texture) => texture,
                Err(error) => {
                    eprintln!("refused: {name}: a sampled {pf:?} attachment: {error:?}");
                    continue;
                }
            }
        } else {
            lane.msaa(pf)
        };
        let resolve = lane.single(pf);
        // Probe 6e's earlier pass: the resolve texture cleared to depth 0.75 and stencil 7.
        let mut clear = RenderPassDescriptor::new();
        clear.depth_attachment.texture = Some(resolve.clone());
        clear.depth_attachment.clear_depth = 0.75;
        clear.depth_attachment.store_action = StoreAction::Store;
        if stencil.is_some() {
            clear.stencil_attachment.texture = Some(resolve.clone());
            clear.stencil_attachment.clear_stencil = 7;
            clear.stencil_attachment.store_action = StoreAction::Store;
        }
        lane.run(&clear, |_, _, _| {}).unwrap();
        let mut descriptor = RenderPassDescriptor::new();
        let color = lane.msaa(rgba8);
        descriptor.color_attachments[0].texture = Some(color);
        descriptor.color_attachments[0].load_action = LoadAction::Clear;
        descriptor.color_attachments[0].store_action = StoreAction::DontCare;
        let depth = &mut descriptor.depth_attachment;
        depth.texture = Some(msaa.clone());
        depth.store_action = StoreAction::DontCare;
        if let Some(filter) = depth_filter {
            depth.store_action = StoreAction::MultisampleResolve;
            depth.resolve_texture = Some(resolve.clone());
            depth.depth_resolve_filter = filter;
        }
        if let Some(filter) = stencil {
            let stencil = &mut descriptor.stencil_attachment;
            stencil.texture = Some(msaa.clone());
            stencil.store_action = StoreAction::DontCare;
            if let Some(filter) = filter {
                stencil.store_action = StoreAction::MultisampleResolve;
                stencil.resolve_texture = Some(resolve.clone());
                stencil.stencil_resolve_filter = filter;
            }
        }
        let stencil_format = stencil.map(|_| pf);
        let pipeline = lane.pipeline((FULL, MASKED), &[rgba8], (Some(pf), stencil_format), 4);
        let state = state(stencil.is_some());
        let ran = lane.run(&descriptor, |l, e, h| {
            e.set_depth_stencil_state(&state).unwrap();
            for sample in 0..4 {
                e.set_stencil_reference_value(STENCILS[sample]);
                let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &pipeline, DEPTHS[sample]);
                h.push(mask);
            }
        });
        // A drawn depth is no subpass's. `DepthResolvedSample` takes sample zero's stencil after
        // `Sample0`; after `Min` or `Max` no subpass resolves the stencil (it is drawn after the
        // pass).
        let depth_mode = depth_mode.filter(|_| !depth_drawn);
        let stencil_mode = stencil
            .flatten()
            .filter(|_| !picked)
            .map(|_| Mode::SAMPLE_ZERO);
        let both = stencil.is_some();
        let metal = match metal {
            Ok(_) if !device_resolves(depth_mode, stencil_mode, both) => {
                eprintln!(
                    "refused: {name}: the device's resolve modes lack depth {depth_mode:?} with \
                     stencil {stencil_mode:?}"
                );
                Err(())
            }
            metal => metal,
        };
        match metal {
            Ok((depth, stencil_value)) => {
                ran.unwrap_or_else(|error| panic!("{name}: {error:?}"));
                let option = if stencil.is_some() {
                    BlitOption::DepthFromDepthStencil
                } else {
                    BlitOption::None
                };
                let resolved = lane.read(&resolve, (0, 0), 4, option);
                assert_eq!(
                    resolved,
                    repeat(&f32::to_le_bytes(depth), texels),
                    "{name}: depth"
                );
                if let Some(value) = stencil_value {
                    let resolved =
                        lane.read(&resolve, (0, 0), 1, BlitOption::StencilFromDepthStencil);
                    assert_eq!(resolved, vec![value as u8; texels], "{name}: stencil");
                }
            }
            Err(()) => assert_eq!(
                ran,
                Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve)),
                "{name}"
            ),
        }
    }
    quiet(lane, &messages);
}

/// Whether the runtime draws a depth resolve of a mode after the pass on `device` (`resolve.rs`):
/// a `MIN` or `MAX` its compile device's description lacks, where the driver's fragment depth
/// stores values unchanged (`DriverFact::FragmentDepthStoresUnchanged`), for a 32-bit float depth
/// (every case's).
pub(super) fn drawn_depth(device: &Device) -> impl Fn(Mode) -> bool {
    let depth_modes = device
        .compile_device()
        .profile()
        .limits()
        .supported_depth_resolve_modes;
    let exact = device
        .properties()
        .driver_facts
        .contains(&DriverFact::FragmentDepthStoresUnchanged);
    move |mode| (mode == Mode::MIN || mode == Mode::MAX) && !depth_modes.contains(mode) && exact
}

/// Each sample's code in [`a_16_bit_depth_resolves_to_the_code_of_the_sample_metal_picks`]:
/// probe 6's depths 0.5, 0.1, 0.9 and 0.3, each rounded to its nearest 16-bit code.
const CODES: [u16; 4] = [32768, 6554, 58981, 19661];

/// A 4x `Depth16Unorm` attachment's resolves keep the code of the sample each filter picks:
/// `Sample0` sample 0's, `Min` the smallest and `Max` the largest, each sample drawn at its
/// code's depth (the code over 65535, which every conversion measured stores as the code). Where
/// the device has no `MIN` or `MAX` depth resolve mode (lavapipe), the runtime draws the picked
/// depth after the pass where its driver keeps a 16-bit depth's codes
/// (`DriverFact::FragmentDepthKeepsDepth16Codes`), printed as a "drawn:" line, and refuses it
/// `NotImplemented(DepthStencilResolve)` elsewhere, printed as a "refused:" line. On MoltenVK the
/// resolve is Metal's own. The attachment is made for shader reads, which the draws need.
#[test]
fn a_16_bit_depth_resolves_to_the_code_of_the_sample_metal_picks() {
    use MultisampleDepthResolveFilter as D;
    let (messages, _cache, lane) = lane();
    let pf = PixelFormat::Depth16Unorm;
    let rgba8 = PixelFormat::RGBA8Unorm;
    let device_resolves = device_resolves(&lane.device);
    let depth_modes = lane
        .device
        .compile_device()
        .profile()
        .limits()
        .supported_depth_resolve_modes;
    let keeps = lane
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::FragmentDepthKeepsDepth16Codes);
    let texels = (SIZE * SIZE) as usize;
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = true;
    let state = lane.device.new_depth_stencil_state(&state).unwrap();
    let pipeline = lane.pipeline((FULL, MASKED), &[rgba8], (Some(pf), None), 4);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    for (filter, mode, code) in [
        (D::Sample0, Mode::SAMPLE_ZERO, CODES[0]),
        (D::Min, Mode::MIN, CODES[1]),
        (D::Max, Mode::MAX, CODES[2]),
    ] {
        let drawn = mode != Mode::SAMPLE_ZERO && !depth_modes.contains(mode) && keeps;
        let resolve = lane.single(pf);
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.color_attachments[0].texture = Some(lane.msaa(rgba8));
        descriptor.color_attachments[0].load_action = LoadAction::Clear;
        descriptor.color_attachments[0].store_action = StoreAction::DontCare;
        let depth = &mut descriptor.depth_attachment;
        depth.texture = Some(lane.texture(pf, MSAA, usage));
        depth.store_action = StoreAction::MultisampleResolve;
        depth.resolve_texture = Some(resolve.clone());
        depth.depth_resolve_filter = filter;
        let ran = lane.run(&descriptor, |l, e, h| {
            e.set_depth_stencil_state(&state).unwrap();
            for (sample, &code) in CODES.iter().enumerate() {
                let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &pipeline, f32::from(code) / 65535.0);
                h.push(mask);
            }
        });
        if drawn {
            eprintln!("drawn: depth16 {filter:?}");
        }
        if drawn || device_resolves(Some(mode), None, false) {
            ran.unwrap_or_else(|error| panic!("depth16 {filter:?}: {error:?}"));
            let resolved = lane.read(&resolve, (0, 0), 2, BlitOption::None);
            assert_eq!(
                resolved,
                repeat(&code.to_le_bytes(), texels),
                "depth16 {filter:?}"
            );
        } else {
            eprintln!("refused: depth16 {filter:?}: the device's resolve modes lack {mode:?}");
            assert_eq!(
                ran,
                Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve)),
                "depth16 {filter:?}"
            );
        }
    }
    quiet(lane, &messages);
}

/// Whether `device` resolves a depth/stencil attachment by `depth` and `stencil` mode (`None`:
/// the aspect is not resolved) of a format with `both` aspects, as its compile device's
/// description states the modes: each mode in its aspect's `supported*ResolveModes`, and the two
/// paired as `independentResolve` and `independentResolveNone` allow
/// (`VkSubpassDescriptionDepthStencilResolve`'s rules); or, as `resolve.rs` falls back to, each
/// resolved aspect in a subpass of its own whose other aspect is not resolved or takes the
/// same mode.
pub(super) fn device_resolves(
    device: &Device,
) -> impl Fn(Option<Mode>, Option<Mode>, bool) -> bool {
    let limits = device.compile_device().profile().limits();
    let (depth_modes, stencil_modes) = (
        limits.supported_depth_resolve_modes,
        limits.supported_stencil_resolve_modes,
    );
    let (independent_none, independent) =
        (limits.independent_resolve_none, limits.independent_resolve);
    let allows = move |depth: Option<Mode>, stencil: Option<Mode>, both: bool| {
        depth.is_none_or(|m| depth_modes.contains(m))
            && stencil.is_none_or(|m| stencil_modes.contains(m))
            && (!both
                || depth == stencil
                || independent
                || (independent_none && (depth.is_none() || stencil.is_none())))
    };
    move |depth, stencil, both| {
        allows(depth, stencil, both)
            || (both
                && depth
                    .is_none_or(|m| allows(Some(m), None, true) || allows(Some(m), Some(m), true))
                && stencil
                    .is_none_or(|m| allows(None, Some(m), true) || allows(Some(m), Some(m), true)))
    }
}

/// What Metal's validation stops of a resolve (probe 6), refused typed when the encoder opens or
/// the late store action is set, before anything records.
#[test]
fn resolves_are_refused_as_metal_refuses_them() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let c0 = Attachment::Color(0);
    let refused = |call, problem| {
        Err(Error::Invalid(Invalid::new(
            call,
            Problem::RenderPass(problem),
        )))
    };
    let open = |descriptor: &RenderPassDescriptor| {
        lane.queue
            .command_buffer()
            .unwrap()
            .render_command_encoder(descriptor)
            .map(|_| ())
    };
    let msaa = lane.msaa(rgba8);
    let resolve = lane.single(rgba8);
    let call = "render_command_encoder";
    // `ref_format`, `ref_size`, `ref_multisampled`, and the integer formats `res_uint_samples`
    // and `res_sint_samples` (`ref_single_sampled_source` is copied:
    // `a_single_sampled_colour_resolve_copies_its_texels`).
    let other = |resolve: &Texture| {
        color_pass(
            &msaa,
            Some(resolve),
            StoreAction::MultisampleResolve,
            (0, 0),
        )
    };
    // Plain Metal's answer depends on the pair: into `BGRA8Unorm` the values in its channel order
    // (not built), into `RGBA8Snorm` and `R8Unorm` its codes (not built); into `R32Uint` or
    // `RGBA16Float` nothing (`a_resolve_metal_does_not_store_leaves_its_texture_untouched`); the
    // sRGB twin resolves in the resolve texture's encoding
    // (`a_resolve_into_the_srgb_twin_resolves_in_the_resolve_texture_s_encoding`).
    assert_eq!(
        open(&other(&lane.single(PixelFormat::BGRA8Unorm))),
        Err(Error::NotImplemented(RuntimeGap::Render))
    );
    for pf in [PixelFormat::RGBA8Snorm, PixelFormat::R8Unorm] {
        assert_eq!(
            open(&other(&lane.single(pf))),
            refused(call, RenderPassProblem::ResolveFormat(c0)),
            "{pf:?}"
        );
    }
    // A color level reaching past an attachment that loads (after a `Clear` load plain Metal's
    // answer is the clear color there, `resolve_sizes_and_counts_are_plain_metals`).
    let large = lane.texture(rgba8, Shape { size: 8, ..ONE }, TextureUsage::RENDER_TARGET);
    for load in [LoadAction::Load, LoadAction::DontCare] {
        let mut loading = other(&large);
        loading.color_attachments[0].load_action = load;
        assert_eq!(
            open(&loading),
            refused(call, RenderPassProblem::ResolveSize(c0)),
            "{load:?}"
        );
    }
    assert_eq!(
        open(&other(&lane.msaa(rgba8))),
        refused(call, RenderPassProblem::ResolveSamples(c0))
    );
    for pf in [PixelFormat::RGBA8Uint, PixelFormat::RGBA16Sint] {
        let integer = color_pass(
            &lane.msaa(pf),
            Some(&lane.single(pf)),
            StoreAction::MultisampleResolve,
            (0, 0),
        );
        assert_eq!(
            open(&integer),
            refused(call, RenderPassProblem::ResolveFormat(c0)),
            "{pf:?}"
        );
    }
    // `ref_level_past`, and a slice past an array's; `ref_level_size` (level 1 of 8² is 4²) is
    // taken. A 2D texture's slice is ignored (`an_index_the_texture_type_lacks_is_ignored`).
    let pair = lane.texture(
        rgba8,
        Shape { slices: 2, ..ONE },
        TextureUsage::RENDER_TARGET,
    );
    for (texture, at) in [(&resolve, (1, 0)), (&pair, (1, 0)), (&pair, (0, 2))] {
        assert_eq!(
            open(&color_pass(
                &msaa,
                Some(texture),
                StoreAction::MultisampleResolve,
                at
            )),
            refused(call, RenderPassProblem::ResolveSubresource(c0))
        );
    }
    let mipmapped = lane.texture(
        rgba8,
        Shape {
            size: 8,
            levels: 2,
            ..ONE
        },
        TextureUsage::RENDER_TARGET,
    );
    assert_eq!(
        open(&color_pass(
            &msaa,
            Some(&mipmapped),
            StoreAction::MultisampleResolve,
            (1, 0)
        )),
        Ok(())
    );
    // `ref_depth_resolved_sample_without_depth_resolve`: the depth stored, the stencil resolved
    // by `DepthResolvedSample`, one texture: the mixed actions refused. Without a depth
    // attachment, `DepthResolvedSample` is refused; with one that does not resolve, its filter
    // picks (`depth_resolved_samples_and_stored_stencils_are_plain_metals`).
    let combined = PixelFormat::Depth32FloatStencil8;
    let ds = lane.msaa(combined);
    let mut descriptor = RenderPassDescriptor::new();
    descriptor.depth_attachment.texture = Some(ds.clone());
    descriptor.depth_attachment.store_action = StoreAction::Store;
    descriptor.stencil_attachment.texture = Some(ds.clone());
    descriptor.stencil_attachment.store_action = StoreAction::MultisampleResolve;
    descriptor.stencil_attachment.resolve_texture = Some(lane.single(combined));
    descriptor.stencil_attachment.stencil_resolve_filter =
        MultisampleStencilResolveFilter::DepthResolvedSample;
    assert_eq!(
        open(&descriptor),
        refused(call, RenderPassProblem::DepthStencilResolveMix)
    );
    descriptor.depth_attachment.texture = None;
    assert_eq!(
        open(&descriptor),
        refused(
            call,
            RenderPassProblem::DepthResolvedSampleWithoutDepthResolve
        )
    );
    // The stencil resolved alone beside the stored depth (above), refused at the stencil's late
    // store action too, and at the depth's late `Store` beside a stencil resolved alone; stored
    // and resolved there instead, it runs
    // (`stored_and_resolved_aspects_of_one_texture_are_metals`).
    let mut descriptor = RenderPassDescriptor::new();
    descriptor.depth_attachment.texture = Some(ds.clone());
    descriptor.depth_attachment.store_action = StoreAction::Store;
    descriptor.stencil_attachment.texture = Some(ds.clone());
    descriptor.stencil_attachment.store_action = StoreAction::Unknown;
    descriptor.stencil_attachment.resolve_texture = Some(lane.single(combined));
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    assert_eq!(
        encoder.set_stencil_store_action(StoreAction::MultisampleResolve),
        refused(
            "set_stencil_store_action",
            RenderPassProblem::DepthStencilResolveMix
        )
    );
    assert_eq!(
        encoder.set_stencil_store_action(StoreAction::StoreAndMultisampleResolve),
        Ok(())
    );
    drop(encoder);
    drop(command_buffer);
    let mut descriptor = RenderPassDescriptor::new();
    descriptor.depth_attachment.texture = Some(ds.clone());
    descriptor.depth_attachment.store_action = StoreAction::Unknown;
    descriptor.stencil_attachment.texture = Some(ds.clone());
    descriptor.stencil_attachment.store_action = StoreAction::MultisampleResolve;
    descriptor.stencil_attachment.resolve_texture = Some(lane.single(combined));
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    assert_eq!(
        encoder.set_depth_store_action(StoreAction::Store),
        refused(
            "set_depth_store_action",
            RenderPassProblem::DepthStencilResolveMix
        )
    );
    assert_eq!(
        encoder.set_depth_store_action(StoreAction::DontCare),
        Ok(())
    );
    drop(encoder);
    drop(command_buffer);
    quiet(lane, &messages);
}

/// A color attachment's resolve that Metal's validation stops, as plain Metal runs it (probe 6m):
/// a resolving store action without a resolve texture, the descriptor's
/// (`ref_resolve_without_texture`) or the encoder's (`res_late_resolve_without_texture`, and
/// `StoreAndMultisampleResolve`, which keeps the samples), completes and resolves nothing; a
/// resolve texture under `Store`, the descriptor's (`ref_texture_with_store`) or the encoder's
/// (`res_late_store_with_resolve_texture`), is left untouched (each byte 0x11 before the pass).
#[test]
fn colour_resolves_metal_runs_past_its_validation_resolve_nothing() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let pipeline = lane.pipeline((FULL, WHITE), &[rgba8], (None, None), 4);
    let msaa = lane.msaa(rgba8);
    let kept = vec![0x11; (4 * SIZE * SIZE) as usize];
    let run = |resolve: Option<&Texture>, store, late: Option<StoreAction>| {
        lane.run(&color_pass(&msaa, resolve, store, (0, 0)), |l, e, h| {
            l.draw(e, h, &pipeline, 0.5);
            if let Some(late) = late {
                e.set_color_store_action(late, 0).unwrap();
            }
        })
    };
    assert_eq!(
        run(None, StoreAction::MultisampleResolve, None),
        Ok(()),
        "ref_resolve_without_texture"
    );
    for late in [
        StoreAction::MultisampleResolve,
        StoreAction::StoreAndMultisampleResolve,
    ] {
        assert_eq!(
            run(None, StoreAction::Unknown, Some(late)),
            Ok(()),
            "res_late_resolve_without_texture, {late:?}"
        );
    }
    let resolve = untouched(&lane);
    run(Some(&resolve), StoreAction::Store, None).unwrap();
    assert_eq!(
        lane.read(&resolve, (0, 0), 4, BlitOption::None),
        kept,
        "ref_texture_with_store"
    );
    let resolve = untouched(&lane);
    run(
        Some(&resolve),
        StoreAction::Unknown,
        Some(StoreAction::Store),
    )
    .unwrap();
    assert_eq!(
        lane.read(&resolve, (0, 0), 4, BlitOption::None),
        kept,
        "res_late_store_with_resolve_texture"
    );
    // `Unknown` still at `endEncoding` stores nothing and leaves the resolve texture untouched.
    let resolve = untouched(&lane);
    run(Some(&resolve), StoreAction::Unknown, None).unwrap();
    assert_eq!(
        lane.read(&resolve, (0, 0), 4, BlitOption::None),
        kept,
        "resolve_color_unknown_left"
    );
    quiet(lane, &messages);
}

/// A depth or stencil attachment's resolve texture under a store action that does not resolve,
/// the descriptor's or a late one, or under `Unknown` still at `endEncoding`, which Metal's
/// validation stops, as plain Metal runs it (an M4 with macOS 27.0.1, the
/// `resolve_depth_*` and `resolve_stencil_*` cases): the pass runs and the resolve texture is
/// left untouched (it holds depth 0.75 and stencil 7 before the pass, which clears to 0.25 and
/// 3).
#[test]
fn depth_and_stencil_resolve_textures_metal_runs_past_its_validation_are_untouched() {
    let (messages, _cache, lane) = lane();
    let combined = PixelFormat::Depth32FloatStencil8;
    let texels = (SIZE * SIZE) as usize;
    let held = (0.75f32.to_le_bytes().repeat(texels), vec![7u8; texels]);
    let msaa = lane.msaa(combined);
    // (the aspect, the descriptor's store action, the late one).
    let cases = [
        (Attachment::Depth, StoreAction::Store, None),
        (Attachment::Depth, StoreAction::DontCare, None),
        (
            Attachment::Depth,
            StoreAction::Unknown,
            Some(StoreAction::Store),
        ),
        (
            Attachment::Depth,
            StoreAction::Unknown,
            Some(StoreAction::DontCare),
        ),
        (Attachment::Depth, StoreAction::Unknown, None),
        (Attachment::Stencil, StoreAction::Store, None),
        (Attachment::Stencil, StoreAction::DontCare, None),
        (
            Attachment::Stencil,
            StoreAction::Unknown,
            Some(StoreAction::Store),
        ),
        (
            Attachment::Stencil,
            StoreAction::Unknown,
            Some(StoreAction::DontCare),
        ),
        (Attachment::Stencil, StoreAction::Unknown, None),
    ];
    for (which, store, late) in cases {
        let resolve = held_resolve(&lane);
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.depth_attachment.texture = Some(msaa.clone());
        descriptor.depth_attachment.clear_depth = 0.25;
        descriptor.depth_attachment.store_action = StoreAction::DontCare;
        descriptor.stencil_attachment.texture = Some(msaa.clone());
        descriptor.stencil_attachment.clear_stencil = 3;
        descriptor.stencil_attachment.store_action = StoreAction::DontCare;
        if which == Attachment::Depth {
            descriptor.depth_attachment.store_action = store;
            descriptor.depth_attachment.resolve_texture = Some(resolve.clone());
        } else {
            descriptor.stencil_attachment.store_action = store;
            descriptor.stencil_attachment.resolve_texture = Some(resolve.clone());
        }
        let ran = lane.run(&descriptor, |_, encoder, _| {
            if let Some(late) = late {
                let set = match which {
                    Attachment::Depth => encoder.set_depth_store_action(late),
                    _ => encoder.set_stencil_store_action(late),
                };
                set.unwrap();
            }
        });
        let name = format!("{which:?} {store:?} then {late:?}");
        assert_eq!(ran, Ok(()), "{name}");
        assert_eq!(depth_and_stencil(&lane, &resolve), held, "{name}");
    }
    quiet(lane, &messages);
}

/// A shared single-sampled `RGBA8Unorm` resolve texture of the lane's size, each byte 0x11.
fn untouched(lane: &Lane) -> Texture {
    filled(lane, TextureUsage::RENDER_TARGET)
}

/// A shared single-sampled `RGBA8Unorm` texture of the lane's size and `usage`, each byte 0x11.
fn filled(lane: &Lane, usage: TextureUsage) -> Texture {
    let shape = Shape {
        shared: true,
        ..ONE
    };
    let texture = lane.texture(PixelFormat::RGBA8Unorm, shape, usage);
    texture
        .replace_region(
            Region::new_2d(0, 0, SIZE, SIZE),
            0,
            0,
            &[0x11; (4 * SIZE * SIZE) as usize],
            4 * SIZE,
            4 * SIZE * SIZE,
        )
        .unwrap();
    texture
}

/// A single-sampled color attachment's resolve, which Metal's validation stops, as plain Metal
/// runs it (probe 6m's `ref_single_sampled_source`: the resolve texture holds the draw): its
/// texels copied into the resolve texture, under `MultisampleResolve` and
/// `StoreAndMultisampleResolve`, the descriptor's or a late one (each byte 0x11 before the
/// pass, the draw white).
#[test]
fn a_single_sampled_colour_resolve_copies_its_texels() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let pipeline = lane.pipeline((FULL, WHITE), &[rgba8], (None, None), 1);
    let source = lane.single(rgba8);
    let white = vec![0xff; (4 * SIZE * SIZE) as usize];
    for (store, late) in [
        (StoreAction::MultisampleResolve, None),
        (StoreAction::StoreAndMultisampleResolve, None),
        (StoreAction::Unknown, Some(StoreAction::MultisampleResolve)),
    ] {
        let resolve = untouched(&lane);
        lane.run(
            &color_pass(&source, Some(&resolve), store, (0, 0)),
            |l, e, h| {
                l.draw(e, h, &pipeline, 0.5);
                if let Some(late) = late {
                    e.set_color_store_action(late, 0).unwrap();
                }
            },
        )
        .unwrap();
        assert_eq!(
            lane.read(&resolve, (0, 0), 4, BlitOption::None),
            white,
            "{store:?}, late {late:?}"
        );
    }
    quiet(lane, &messages);
}

/// A multisampled color attachment resolving into a texture or a view of its format's sRGB twin,
/// which Metal's validation stops, resolves in the resolve texture's encoding as plain Metal's
/// (our probe `srgbres` on an M3 Max under macOS 27.0.1, 2 and 4 samples, the `RGBA8` and `BGRA8`
/// pairs both ways, four to six processes each and every one alike).
///
/// A pixel whose samples agree reads neither encoding: 0x00, 0x80 and 0xff are fixed points of
/// both averages, and identical codes average to themselves whatever space the average is in. So
/// each pair is read twice: cleared to (0.5, 0.25, 0.75, 1) with no draw, where an `RGBA8Unorm`
/// attachment stores `8040bfff` and an `RGBA8Unorm_sRGB` one `bc89e1ff` under either rule; and
/// cleared to zero with white on samples 0 and 1 alone, where the resolve texture's encoding
/// decides it. There an `RGBA8Unorm_sRGB` resolve texture takes `bcbcbc80` (each color channel's
/// codes 0xff and 0x00 decoded, averaged in linear space and encoded again, and alpha's averaged
/// as the linear channel it is) and an `RGBA8Unorm` one `80808080` (every channel's codes
/// averaged), whichever format the attachment is. Into an sRGB texture the resolve decodes and
/// encodes by the device's own sRGB conversions, so each color channel there stands within the
/// two codes `device-rounded-resolve` states of Metal's ([`twin_resolved`]). The clear's codes
/// are the device's conversion of the clear color, which Vulkan leaves to it (either code nearest
/// a channel's value, and the sRGB encode), announced as `device-color-conversion`: on a device
/// whose clear is not Metal's (no `DriverFact::ColorClearIsMetals`) each color channel of the
/// agreeing samples' texel stands within a code of Metal's (ANV on the Iris Xe of RPL-P clears an
/// `RGBA8Unorm_sRGB` channel's 0.5, 187.52 codes, to `0xbb` where Metal stores `0xbc`).
#[test]
fn a_resolve_into_the_srgb_twin_resolves_in_the_resolve_texture_s_encoding() {
    let (messages, _cache, lane) = lane();
    let metals_clear =
        (lane.device.properties().driver_facts).contains(&DriverFact::ColorClearIsMetals);
    let mut cleared_apart = false;
    let unorm = PixelFormat::RGBA8Unorm;
    let srgb = PixelFormat::RGBA8UnormSRGB;
    for (attachment, into, view) in [
        (unorm, srgb, None),
        (srgb, unorm, None),
        (unorm, unorm, Some(srgb)),
        (srgb, srgb, Some(unorm)),
        (PixelFormat::BGRA8Unorm, PixelFormat::BGRA8UnormSRGB, None),
        (PixelFormat::BGRA8UnormSRGB, PixelFormat::BGRA8Unorm, None),
    ] {
        // The format the resolve texture, or the view the pass names, encodes into.
        let written = view.unwrap_or(into);
        let srgb_written = matches!(
            written,
            PixelFormat::RGBA8UnormSRGB | PixelFormat::BGRA8UnormSRGB
        );
        // Samples that agree: the clear's codes in the attachment's format and channel order,
        // which both rules leave alone (the pair shares the order).
        let [r, g, b] = match attachment {
            PixelFormat::RGBA8Unorm | PixelFormat::BGRA8Unorm => [0x80, 0x40, 0xbf],
            _ => [0xbc, 0x89, 0xe1],
        };
        let uniform = match attachment {
            PixelFormat::BGRA8Unorm | PixelFormat::BGRA8UnormSRGB => [b, g, r, 0xff],
            _ => [r, g, b, 0xff],
        };
        // Samples that disagree: white on samples 0 and 1 over a clear to zero, read by the
        // encoding the resolve texture is in.
        let masked = match srgb_written {
            true => [0xbc, 0xbc, 0xbc, 0x80],
            false => [0x80, 0x80, 0x80, 0x80],
        };
        let pipeline = lane.pipeline((FULL, MASKED), &[attachment], (None, None), 4);
        for (clear, draw, texel) in [
            (None, true, masked),
            (Some(ClearColor::new(0.5, 0.25, 0.75, 1.0)), false, uniform),
        ] {
            let msaa = lane.msaa(attachment);
            let resolve = lane.single(into);
            let target = match view {
                Some(pf) => resolve.new_texture_view(pf).unwrap(),
                None => resolve.clone(),
            };
            let mut descriptor = color_pass(
                &msaa,
                Some(&target),
                StoreAction::MultisampleResolve,
                (0, 0),
            );
            if let Some(clear) = clear {
                descriptor.color_attachments[0].clear_color = clear;
            }
            let mask = buffer_with(&lane.device, &3u32.to_le_bytes()).unwrap();
            lane.run(&descriptor, |l, e, h| {
                if draw {
                    e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                    l.draw(e, h, &pipeline, 0.5);
                }
            })
            .unwrap();
            let got = lane.read(&resolve, (0, 0), 4, BlitOption::None);
            let what = format!(
                "{attachment:?} into {into:?} through a view of {view:?}, samples {}",
                match draw {
                    true => "that disagree",
                    false => "that agree",
                }
            );
            let either = clear.is_some() && !metals_clear;
            twin_resolved_within(&got, texel, twin_codes(srgb_written, either), &what);
            cleared_apart |= either && got.chunks(4).any(|t| t != texel);
        }
    }
    // A clear whose codes stand apart from Metal's was announced.
    let notices = counts(&messages.take_notices());
    assert!(
        !cleared_apart || notices.contains_key(&Degradation::DeviceColorConversion),
        "a clear apart from Metal's codes, unannounced: {notices:?}"
    );
    quiet(lane, &messages);
}

/// The codes each channel of a twin resolve's texel may stand from Metal's: where the resolve
/// writes an sRGB texture (`srgb`), a color channel within the two `device-rounded-resolve` states;
/// where the texel is a clear's codes the device converts its own way (`cleared`), a color channel
/// within one; the alpha exactly.
fn twin_codes(srgb: bool, cleared: bool) -> [u8; 4] {
    let color = match (srgb, cleared) {
        (true, _) => 2,
        (false, true) => 1,
        (false, false) => 0,
    };
    [color, color, color, 0]
}

/// Hold each texel of a twin resolve's `got` to Metal's `texel` (the lane's 4x4 texture): every
/// byte where the resolve writes a UNORM texture, and where it writes an sRGB one (`srgb`), whose
/// decode and encode are the device's own (`bind::ColorResolveFormat::device_srgb`), each color
/// channel within the two codes `device-rounded-resolve` states and the alpha exactly. A texel
/// apart from Metal's is printed.
pub(super) fn twin_resolved(got: &[u8], texel: [u8; 4], srgb: bool, what: &str) {
    twin_resolved_within(got, texel, twin_codes(srgb, false), what);
}

/// [`twin_resolved`], each channel `c` within `codes[c]` codes of Metal's `texel`.
fn twin_resolved_within(got: &[u8], texel: [u8; 4], codes: [u8; 4], what: &str) {
    let within = got.chunks(4).all(|t| {
        t.iter()
            .zip(texel)
            .zip(codes)
            .all(|((&g, m), codes)| g.abs_diff(m) <= codes)
    });
    assert!(
        within && got.len() == 4 * (SIZE * SIZE) as usize,
        "{what}: {} where Metal's is {}",
        hex(got),
        hex(&texel)
    );
    if got.chunks(4).any(|t| t != texel) {
        eprintln!(
            "{what}: {} where Metal's is {} (the device's sRGB conversions)",
            hex(&got[..4]),
            hex(&texel)
        );
    }
}

/// A multisampled color attachment resolved into its format's sRGB twin keeps the samples it held
/// before the pass, as plain Metal's does (the DontCare-store probe's rule for
/// `MultisampleResolve`, which stores nothing of the attachment, an M3 Max under macOS 27.0.1; the
/// consumer's measurement of the pair at 16x16, both load actions and both resolve-texture shapes).
/// The twin's resolve is the crate-owned one on every device (no subpass resolve writes the twin's
/// encoding), which reads the samples the pass stores, so the pass renders the attachment through
/// an image its texture keeps and copies nothing back (`resolve::keep_samples`). Both directions of
/// the `BGRA8` and `RGBA8` pairs, a resolve texture and a named view of the twin alike, over a
/// `Clear` load (whose resolve texels are the measured ones of samples that disagree) and a `Load`
/// one. Without the kept image the attachment holds the resolving pass's own clear and its draw:
/// the four runtime files at the branch point leave each `RGBA8Unorm` texel reading `80808080`,
/// the average of white on samples 0 and 1 over the pass's clear to zero, where Metal keeps
/// `bf4080ff`, the clear the pass before it stored.
#[test]
fn samples_a_twin_resolve_reads_are_kept_as_metal_keeps_them() {
    let (messages, _cache, lane) = lane();
    for (attachment, twin) in [
        (PixelFormat::RGBA8Unorm, PixelFormat::RGBA8UnormSRGB),
        (PixelFormat::RGBA8UnormSRGB, PixelFormat::RGBA8Unorm),
        (PixelFormat::BGRA8Unorm, PixelFormat::BGRA8UnormSRGB),
        (PixelFormat::BGRA8UnormSRGB, PixelFormat::BGRA8Unorm),
    ] {
        let pipeline = lane.pipeline((FULL, MASKED), &[attachment], (None, None), 4);
        // The twin the resolve writes: the resolve texture's own format, or the view the pass
        // names of a texture of the attachment's format.
        for (into, view) in [(twin, None), (attachment, Some(twin))] {
            for load in [LoadAction::Clear, LoadAction::Load] {
                let label = format!("{attachment:?} into {into:?} through {view:?}, {load:?}");
                // The samples before the pass: a clear of its own, stored.
                let msaa = lane.msaa(attachment);
                let earlier = ClearColor::new(0.75, 0.25, 0.5, 1.0);
                let mut before = color_pass(&msaa, None, StoreAction::Store, (0, 0));
                before.color_attachments[0].clear_color = earlier;
                lane.run(&before, |_, _, _| {}).unwrap();
                let kept = samples_of(&lane, &msaa, attachment, 4);
                // The resolving pass: cleared to zero or loaded, white on samples 0 and 1, and
                // resolved alone into the twin.
                let resolve = lane.single(into);
                let target = match view {
                    Some(pf) => resolve.new_texture_view(pf).unwrap(),
                    None => resolve.clone(),
                };
                let mut pass = color_pass(
                    &msaa,
                    Some(&target),
                    StoreAction::MultisampleResolve,
                    (0, 0),
                );
                pass.color_attachments[0].load_action = load;
                let mask = buffer_with(&lane.device, &3u32.to_le_bytes()).unwrap();
                lane.run(&pass, |l, e, h| {
                    e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                    l.draw(e, h, &pipeline, 0.5);
                })
                .unwrap_or_else(|e| panic!("{label}: {e}"));
                // The resolve texels of the measured case: white on half the samples over a clear
                // to zero, read by the encoding the resolve texture is in (the twin resolve's own
                // test asserts every shape of it).
                if load == LoadAction::Clear {
                    let srgb = matches!(
                        twin,
                        PixelFormat::RGBA8UnormSRGB | PixelFormat::BGRA8UnormSRGB
                    );
                    let texel: [u8; 4] = match srgb {
                        true => [0xbc, 0xbc, 0xbc, 0x80],
                        false => [0x80, 0x80, 0x80, 0x80],
                    };
                    twin_resolved(
                        &lane.read(&resolve, (0, 0), 4, BlitOption::None),
                        texel,
                        srgb,
                        &format!("{label}: the resolve texture"),
                    );
                }
                assert_eq!(
                    hex(&samples_of(&lane, &msaa, attachment, 4)),
                    hex(&kept),
                    "{label}: the attachment's samples"
                );
            }
        }
    }
    quiet(lane, &messages);
}

/// An extended-range attachment's resolve into its own format is the crate-owned one on every
/// device too (no device's average resolves the codes it holds, `color_resolve.rs`), and it reads
/// the samples the pass stores alike: the attachment keeps the samples it held before the pass, as
/// plain Metal's does under `MultisampleResolve`, which stores nothing of it (the DontCare-store
/// probe's rule, an M3 Max under macOS 27.0.1). `BGRA10_XR` and `BGR10_XR` at 4 samples, over a
/// `Clear` load and a `Load` one. This resolve took the crate-owned path before the sRGB twin's
/// did, so its samples were overwritten the same way: without the kept image each `BGRA10_XR` texel
/// reads `0x80003000` (white on samples 0 and 1 over the pass's clear to zero) where Metal keeps
/// `0x7bdeef7b`, the clear the pass before it stored.
#[test]
fn samples_an_extended_range_resolve_reads_are_kept_as_metal_keeps_them() {
    let (messages, _cache, lane) = lane();
    // `BGRA10_XR` holds four 16-bit fields, `BGR10_XR` three 10-bit ones in a word.
    for (pf, bytes) in [(PixelFormat::BGRA10XR, 8), (PixelFormat::BGR10XR, 4)] {
        let pipeline = lane.pipeline((FULL, MASKED), &[pf], (None, None), 4);
        for load in [LoadAction::Clear, LoadAction::Load] {
            let label = format!("{pf:?}, {load:?}");
            let msaa = lane.msaa(pf);
            let mut before = color_pass(&msaa, None, StoreAction::Store, (0, 0));
            before.color_attachments[0].clear_color = ClearColor::new(0.75, 0.25, 0.5, 1.0);
            lane.run(&before, |_, _, _| {}).unwrap();
            let kept = samples_of(&lane, &msaa, pf, bytes);
            let resolve = lane.single(pf);
            let mut pass = color_pass(
                &msaa,
                Some(&resolve),
                StoreAction::MultisampleResolve,
                (0, 0),
            );
            pass.color_attachments[0].load_action = load;
            let mask = buffer_with(&lane.device, &3u32.to_le_bytes()).unwrap();
            lane.run(&pass, |l, e, h| {
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &pipeline, 0.5);
            })
            .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                hex(&samples_of(&lane, &msaa, pf, bytes)),
                hex(&kept),
                "{label}: the attachment's samples"
            );
        }
    }
    quiet(lane, &messages);
}

/// What `msaa`, a 4x attachment of `pf` of `bytes` a texel, holds: its samples resolved into a
/// texture of its own format by a pass that loads it and stores nothing of it, which leaves them
/// as they are (the
/// device's resolve of the format, as `a_resolve_metal_does_not_store_leaves_its_texture_untouched`
/// reads them).
fn samples_of(lane: &Lane, msaa: &Texture, pf: PixelFormat, bytes: u64) -> Vec<u8> {
    let held = lane.single(pf);
    let mut read = color_pass(msaa, Some(&held), StoreAction::MultisampleResolve, (0, 0));
    read.color_attachments[0].load_action = LoadAction::Load;
    lane.run(&read, |_, _, _| {}).unwrap();
    lane.read(&held, (0, 0), bytes, BlitOption::None)
}

/// A multisampled color attachment's resolve into a texture of a pixel format Metal does not store
/// its tile into, which Metal's validation stops, as plain Metal runs it (our probe `xres`, the M3
/// Max under macOS 27.0.1, 2 and 4 samples): the resolve texture keeps its texels (each byte 0x5a
/// before the pass), and the attachment holds what its store action leaves without the resolve:
/// after `StoreAndMultisampleResolve` the pass's clear and draw, as `Store` leaves them; after
/// `MultisampleResolve` its texels from before the pass, cleared and stored by a pass before it to
/// (0.75, 0.25, 0.5, 1), as Metal's tile store leaves them in every pass measured (on a device
/// that stores a `DontCare` attachment, `DontCare::Stored`, the pass's texels, Metal's answer
/// where its tiler overflows). Read through a resolve of the attachment into its own format by a
/// pass that loads it. Each pass clears to (0.25, 0.5, 0.75, 1) and draws white or nothing. The
/// pairs: `BGRA8Unorm_sRGB` and `BGRA8Unorm` into `RGBA16Float` (the formats macOS 26.6.2's Maps
/// resolves across), `RGBA8Unorm` into `R32Float` and `RGBA8Uint`, `RG16Unorm` into
/// `RGBA8Unorm`, and `RGBA16Snorm` into `RG32Float`.
#[test]
fn a_resolve_metal_does_not_store_leaves_its_texture_untouched() {
    let (messages, _cache, lane) = lane();
    let facts = &lane.device.properties().driver_facts;
    // A device that stores `DONT_CARE` as Metal's own keeps Metal's answer; one that keeps one of
    // Metal's two either; every other one is handed `STORE` (`render_target::DontCare`).
    let metals = facts.contains(&DriverFact::DontCareStoreIsMetals);
    let one_of = facts.contains(&DriverFact::DontCareStoreIsOneOfMetals);
    for (from, into, bytes) in [
        (PixelFormat::BGRA8UnormSRGB, PixelFormat::RGBA16Float, 8u64),
        (PixelFormat::BGRA8Unorm, PixelFormat::RGBA16Float, 8),
        (PixelFormat::RGBA8Unorm, PixelFormat::R32Float, 4),
        (PixelFormat::RGBA8Unorm, PixelFormat::RGBA8Uint, 4),
        (PixelFormat::RG16Unorm, PixelFormat::RGBA8Unorm, 4),
        (PixelFormat::RGBA16Snorm, PixelFormat::RG32Float, 8),
    ] {
        let own_bytes = match from {
            PixelFormat::RGBA16Snorm => 8,
            _ => 4,
        };
        let pipeline = lane.pipeline((FULL, WHITE), &[from], (None, None), 4);
        for store in [
            StoreAction::MultisampleResolve,
            StoreAction::StoreAndMultisampleResolve,
        ] {
            for drawn in [false, true] {
                let label = format!("{from:?} into {into:?}, {store:?}, drawn {drawn}");
                let msaa = lane.msaa(from);
                let resolve = lane.texture(
                    into,
                    Shape {
                        shared: true,
                        ..ONE
                    },
                    TextureUsage::SHADER_READ,
                );
                let filled = vec![0x5a; (bytes * SIZE * SIZE) as usize];
                resolve
                    .replace_region(
                        Region::new_2d(0, 0, SIZE, SIZE),
                        0,
                        0,
                        &filled,
                        bytes * SIZE,
                        bytes * SIZE * SIZE,
                    )
                    .unwrap();
                // The texels before the pass.
                let mut before = color_pass(&msaa, None, StoreAction::Store, (0, 0));
                before.color_attachments[0].clear_color = ClearColor::new(0.75, 0.25, 0.5, 1.0);
                lane.run(&before, |_, _, _| {}).unwrap();
                let mut pass = color_pass(&msaa, Some(&resolve), store, (0, 0));
                pass.color_attachments[0].clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
                lane.run(&pass, |l, e, h| {
                    if drawn {
                        l.draw(e, h, &pipeline, 0.5);
                    }
                })
                .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(
                    lane.read(&resolve, (0, 0), bytes, BlitOption::None),
                    filled,
                    "{label}: the resolve texture"
                );
                // What the attachment holds, and what each pass would leave there.
                let held = lane.single(from);
                let mut read =
                    color_pass(&msaa, Some(&held), StoreAction::MultisampleResolve, (0, 0));
                read.color_attachments[0].load_action = LoadAction::Load;
                lane.run(&read, |_, _, _| {}).unwrap();
                let held = lane.read(&held, (0, 0), own_bytes, BlitOption::None);
                let reference = |clear: ClearColor, draw: bool| {
                    let texture = lane.msaa(from);
                    let own = lane.single(from);
                    let mut pass = color_pass(
                        &texture,
                        Some(&own),
                        StoreAction::StoreAndMultisampleResolve,
                        (0, 0),
                    );
                    pass.color_attachments[0].clear_color = clear;
                    lane.run(&pass, |l, e, h| {
                        if draw {
                            l.draw(e, h, &pipeline, 0.5);
                        }
                    })
                    .unwrap();
                    lane.read(&own, (0, 0), own_bytes, BlitOption::None)
                };
                let rendered = reference(ClearColor::new(0.25, 0.5, 0.75, 1.0), drawn);
                let earlier = reference(ClearColor::new(0.75, 0.25, 0.5, 1.0), false);
                // A pass without a draw call renders the attachment through a backing its
                // texture keeps, on every device (`render.rs`).
                let kept = store == StoreAction::MultisampleResolve && (!drawn || metals);
                let either = store == StoreAction::MultisampleResolve && drawn && one_of;
                let texels = |v: &[u8]| {
                    v.chunks(own_bytes as usize)
                        .map(<[u8]>::to_vec)
                        .collect::<Vec<_>>()
                };
                if kept {
                    assert_eq!(held, earlier, "{label}: the attachment's samples");
                } else if either {
                    let all = texels(&held)
                        .iter()
                        .zip(texels(&earlier))
                        .zip(texels(&rendered))
                        .all(|((h, e), r)| *h == e || *h == r);
                    assert!(all, "{label}: the attachment's samples {held:02x?}");
                } else {
                    assert_eq!(held, rendered, "{label}: the attachment's samples");
                }
            }
        }
    }
    quiet(lane, &messages);
}

/// A color texture without `RenderTarget` usage, which Metal's validation stops, renders as plain
/// Metal renders it (the probe's `usage_*`, an M4; Metal probe case `usage_missing` on an
/// M4), through a stand-in: cleared and drawn white, then stored, it holds the draw; loaded and
/// stored, its texels; cleared and drawn, then stored `DontCare`, its texels (each byte 0x11 before
/// the pass); cleared and stored, the clear. A depth or stencil texture without it is
/// [`a_depth_or_stencil_texture_without_render_target_usage_renders_as_metal`]'s.
#[test]
fn a_colour_texture_without_render_target_usage_renders_as_metal() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let pipeline = lane.pipeline((FULL, WHITE), &[rgba8], (None, None), 1);
    let bytes = (4 * SIZE * SIZE) as usize;
    for (name, load, store, draw, texel) in [
        (
            "usage_shared_clear_draw",
            LoadAction::Clear,
            StoreAction::Store,
            true,
            0xff,
        ),
        (
            "usage_load",
            LoadAction::Load,
            StoreAction::Store,
            false,
            0x11,
        ),
        (
            "usage_shared_clear_dontcare",
            LoadAction::Clear,
            StoreAction::DontCare,
            true,
            0x11,
        ),
        (
            "usage_shared_clear",
            LoadAction::Clear,
            StoreAction::Store,
            false,
            0x00,
        ),
    ] {
        let texture = filled(&lane, TextureUsage::SHADER_READ);
        let mut descriptor = color_pass(&texture, None, store, (0, 0));
        descriptor.color_attachments[0].load_action = load;
        lane.run(&descriptor, |l, e, h| {
            if draw {
                l.draw(e, h, &pipeline, 0.5);
            }
        })
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            lane.read(&texture, (0, 0), 4, BlitOption::None),
            vec![texel; bytes],
            "{name}"
        );
    }
    quiet(lane, &messages);
}

/// The depth texels of [`a_depth_or_stencil_texture_without_render_target_usage_renders_as_metal`]
/// in a texture of `pf`, as our probe printed them: the fill 0.75 (written as
/// `0xc000` in 16 bits), the clear 0.5 (`0x7fff`) and the draw's 0.25 (`0x4000`).
fn depth_texel(pf: PixelFormat, which: DepthTexel) -> Vec<u8> {
    let (float, unorm): (f32, u16) = match which {
        DepthTexel::Filled => (0.75, 0xc000),
        DepthTexel::Cleared => (0.5, 0x7fff),
        DepthTexel::Drawn => (0.25, 0x4000),
    };
    match pf {
        PixelFormat::Depth16Unorm => unorm.to_le_bytes().to_vec(),
        _ => float.to_le_bytes().to_vec(),
    }
}

/// Which of [`depth_texel`]'s texels.
#[derive(Clone, Copy)]
enum DepthTexel {
    Filled,
    Cleared,
    Drawn,
}

/// A depth or stencil texture without `RenderTarget` usage (shader-read usage alone, private),
/// which Metal's validation stops, renders as plain Metal renders it (an M4
/// with macOS 27.0.1: `Depth32Float`, `Depth16Unorm`, `Stencil8` and `Depth32Float_Stencil8`,
/// each aspect of the combined format attached), through a stand-in: filled first by a blit
/// (depth 0.75, stencil 0x22), each aspect cleared (0.5, 0x11) or loaded, drawn or not, stored or
/// `DontCare`. Stored, it holds the clear (on `Depth16Unorm`, the device's code for 0.5: Metal's
/// truncated 0x7fff or the next, announced as `UnormDepthRounding`) or its texels where nothing is
/// drawn, and the draw's
/// depth 0.25 (`Less`, written) and stencil 0x5a (`Always`, `Replace`) where the draw passes, as
/// the probe's drawn pixel holds them (here every pixel: the fullscreen triangle); under
/// `DontCare`, its texels before the pass.
#[test]
fn a_depth_or_stencil_texture_without_render_target_usage_renders_as_metal() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let (load, clear) = (LoadAction::Load, LoadAction::Clear);
    let (store, dontcare) = (StoreAction::Store, StoreAction::DontCare);
    let shared = Shape {
        shared: true,
        ..ONE
    };
    for pf in [
        PixelFormat::Depth32Float,
        PixelFormat::Depth16Unorm,
        PixelFormat::Stencil8,
        PixelFormat::Depth32FloatStencil8,
    ] {
        let has_depth = pf != PixelFormat::Stencil8;
        let has_stencil = matches!(
            pf,
            PixelFormat::Stencil8 | PixelFormat::Depth32FloatStencil8
        );
        let combined = has_depth && has_stencil;
        let depth_bytes = depth_texel(pf, DepthTexel::Filled).len() as u64;
        let options = |depth: bool| match (combined, depth) {
            (false, _) => BlitOption::None,
            (true, true) => BlitOption::DepthFromDepthStencil,
            (true, false) => BlitOption::StencilFromDepthStencil,
        };
        let pipeline = lane.pipeline(
            (FULL, WHITE),
            &[rgba8],
            (has_depth.then_some(pf), has_stencil.then_some(pf)),
            1,
        );
        let mut state = DepthStencilDescriptor::default();
        if has_depth {
            state.depth_compare_function = CompareFunction::Less;
            state.depth_write_enabled = true;
        }
        if has_stencil {
            let mut face = StencilDescriptor::default();
            face.depth_stencil_pass_operation = StencilOperation::Replace;
            state.front_face_stencil = face;
            state.back_face_stencil = face;
        }
        let state = lane.device.new_depth_stencil_state(&state).unwrap();
        for (load_name, load_action) in [("clear", clear), ("load", load)] {
            for (store_name, store_action, draw) in [
                ("store_draw", store, true),
                ("store", store, false),
                ("dontcare_draw", dontcare, true),
            ] {
                let case = format!("{pf:?}_{load_name}_{store_name}");
                let texture = lane.texture(pf, ONE, TextureUsage::SHADER_READ);
                // Filled by a blit, as the probe fills it.
                let fill = |depth: bool, texel: Vec<u8>| {
                    let bytes = texel.len() as u64;
                    let source = buffer_with(&lane.device, &texel.repeat(16)).unwrap();
                    let command_buffer = lane.queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.copy_from_buffer_to_texture(
                        &source,
                        0,
                        bytes * SIZE,
                        bytes * SIZE * SIZE,
                        Size::new(SIZE, SIZE, 1),
                        &texture,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        options(depth),
                    )
                    .unwrap();
                    blit.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.error(), None, "{case}: fill");
                };
                if has_depth {
                    fill(true, depth_texel(pf, DepthTexel::Filled));
                }
                if has_stencil {
                    fill(false, vec![0x22]);
                }
                let color = lane.texture(rgba8, shared, TextureUsage::RENDER_TARGET);
                let mut descriptor = color_pass(&color, None, StoreAction::Store, (0, 0));
                if has_depth {
                    let d = &mut descriptor.depth_attachment;
                    d.texture = Some(texture.clone());
                    d.load_action = load_action;
                    d.clear_depth = 0.5;
                    d.store_action = store_action;
                }
                if has_stencil {
                    let s = &mut descriptor.stencil_attachment;
                    s.texture = Some(texture.clone());
                    s.load_action = load_action;
                    s.clear_stencil = 0x11;
                    s.store_action = store_action;
                }
                let ran = lane.run(&descriptor, |l, e, h| {
                    if draw {
                        e.set_depth_stencil_state(&state).unwrap();
                        e.set_stencil_reference_value(0x5a);
                        l.draw(e, h, &pipeline, 0.25);
                    }
                });
                ran.unwrap_or_else(|e| panic!("{case} on {name}: {e}"));
                // What each aspect holds: the draw's where it ran and the pass stored, else the
                // clear where it cleared and stored, else the texels before the pass.
                let stored = store_action == StoreAction::Store;
                let (depth, stencil) = match (stored, draw, load_action) {
                    (true, true, _) => (DepthTexel::Drawn, 0x5a),
                    (true, false, LoadAction::Clear) => (DepthTexel::Cleared, 0x11),
                    _ => (DepthTexel::Filled, 0x22),
                };
                if has_depth {
                    let read = lane.read(&texture, (0, 0), depth_bytes, options(true));
                    // The clear 0.5 is 32767.5 `Depth16Unorm` codes: Metal's clear truncates it
                    // to 0x7fff, the device converts it its own way, announced as
                    // `UnormDepthRounding` (asserted below): the code next to Metal's too.
                    let next = matches!(depth, DepthTexel::Cleared)
                        && pf == PixelFormat::Depth16Unorm
                        && read == 0x8000_u16.to_le_bytes().repeat(16);
                    if !next {
                        assert_eq!(
                            read,
                            depth_texel(pf, depth).repeat(16),
                            "{case} depth on {name}"
                        );
                    }
                }
                if has_stencil {
                    assert_eq!(
                        lane.read(&texture, (0, 0), 1, options(false)),
                        vec![stencil; 16],
                        "{case} stencil on {name}"
                    );
                }
                let white = if draw { 0xff } else { 0x00 };
                assert_eq!(
                    lane.read(&color, (0, 0), 4, BlitOption::None),
                    vec![white; 64],
                    "{case} color on {name}"
                );
            }
        }
        let rounding =
            counts(&messages.take_notices()).contains_key(&Degradation::UnormDepthRounding);
        assert_eq!(
            rounding,
            pf == PixelFormat::Depth16Unorm,
            "{pf:?} on {name}: depth rounding announced"
        );
    }
    quiet(lane, &messages);
}

/// The depth and stencil textures of a split resolve case: separate `Depth32Float` and
/// `Stencil8` textures, or one `Depth32Float_Stencil8` attachment resolving into two textures, or
/// into slices 0 and 1 of one array.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Split {
    Separate,
    TwoTextures,
    TwoSlices,
}

/// A pass whose depth and stencil resolve into different textures resolves each as Metal does (a
/// probe on an M3 Max and an M4: separate textures resolve each aspect by its filter; one
/// combined attachment's two resolve textures each take their aspect alone, the other aspect
/// keeping its texels, which Metal's debug layer stops and plain Metal runs; two slices of one
/// array likewise, a Metal probe on the M3 Max and an M4). The samples are those of a Metal probe
/// on an M4 (depths 0.5, 0.1, 0.9 and 0.3, stencils 20 to 23); each resolve texture holds depth
/// 0.75 and stencil 7 before the pass. An aspect drawn after the pass (a stencil by
/// `DepthResolvedSample` after `Min` or `Max`, or a depth the device has no mode for) is drawn
/// into its own resolve texture, printed as a "drawn:" line; both aspects drawn is
/// `NotImplemented(DepthStencilResolve)` in a split resolve, as is a pair of modes the device
/// lacks; a device that refuses the separate textures' sample copies prints its refusal.
#[test]
fn split_depth_and_stencil_resolves_are_metals() {
    use MultisampleDepthResolveFilter as D;
    use MultisampleStencilResolveFilter as S;
    let (messages, _cache, lane) = lane();
    let device_resolves = device_resolves(&lane.device);
    let drawn = drawn_depth(&lane.device);
    let texels = (SIZE * SIZE) as usize;
    let rgba8 = PixelFormat::RGBA8Unorm;
    let combined = PixelFormat::Depth32FloatStencil8;
    let state = {
        let mut always = StencilDescriptor::default();
        always.stencil_compare_function = CompareFunction::Always;
        always.depth_stencil_pass_operation = StencilOperation::Replace;
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Always;
        state.depth_write_enabled = true;
        state.front_face_stencil = always;
        state.back_face_stencil = always;
        lane.device.new_depth_stencil_state(&state).unwrap()
    };
    let resolving = StoreAction::MultisampleResolve;
    // (name, textures, the depth's filter and store action, the stencil's, and Metal's resolved
    // depth and stencil: `None` for an aspect not resolved).
    type Case = (
        &'static str,
        Split,
        (D, StoreAction),
        (S, StoreAction),
        Option<f32>,
        Option<u8>,
    );
    let cases: [Case; 12] = [
        (
            "sep_sample0",
            Split::Separate,
            (D::Sample0, resolving),
            (S::Sample0, resolving),
            Some(0.5),
            Some(20),
        ),
        (
            "sep_min",
            Split::Separate,
            (D::Min, resolving),
            (S::Sample0, resolving),
            Some(0.1),
            Some(20),
        ),
        (
            "sep_max",
            Split::Separate,
            (D::Max, resolving),
            (S::Sample0, resolving),
            Some(0.9),
            Some(20),
        ),
        (
            "sep_sample0_depth",
            Split::Separate,
            (D::Sample0, resolving),
            (S::DepthResolvedSample, resolving),
            Some(0.5),
            Some(20),
        ),
        (
            "sep_min_depth",
            Split::Separate,
            (D::Min, resolving),
            (S::DepthResolvedSample, resolving),
            Some(0.1),
            Some(21),
        ),
        (
            "sep_depth_resolves_stencil_stores",
            Split::Separate,
            (D::Min, resolving),
            (S::Sample0, StoreAction::Store),
            Some(0.1),
            None,
        ),
        (
            "sep_stencil_resolves_depth_stores",
            Split::Separate,
            (D::Sample0, StoreAction::Store),
            (S::Sample0, resolving),
            None,
            Some(20),
        ),
        (
            "two_sample0",
            Split::TwoTextures,
            (D::Sample0, resolving),
            (S::Sample0, resolving),
            Some(0.5),
            Some(20),
        ),
        (
            "two_min",
            Split::TwoTextures,
            (D::Min, resolving),
            (S::Sample0, resolving),
            Some(0.1),
            Some(20),
        ),
        (
            "two_max",
            Split::TwoTextures,
            (D::Max, resolving),
            (S::Sample0, resolving),
            Some(0.9),
            Some(20),
        ),
        (
            "slices_sample0",
            Split::TwoSlices,
            (D::Sample0, resolving),
            (S::Sample0, resolving),
            Some(0.5),
            Some(20),
        ),
        (
            "slices_max",
            Split::TwoSlices,
            (D::Max, resolving),
            (S::Sample0, resolving),
            Some(0.9),
            Some(20),
        ),
    ];
    for (
        name,
        split,
        (depth_filter, depth_store),
        (stencil_filter, stencil_store),
        depth,
        stencil,
    ) in cases
    {
        let (depth_pf, stencil_pf) = match split {
            Split::Separate => (PixelFormat::Depth32Float, PixelFormat::Stencil8),
            Split::TwoTextures | Split::TwoSlices => (combined, combined),
        };
        // A drawn aspect: a depth picked where the device has no mode for it, or a stencil
        // picked by depth after `Min` or `Max`, each drawn from the samples, read through
        // sampled views (separate textures' through the image they render through, which the
        // runtime makes sampled).
        let depth_mode = depth.map(|_| match depth_filter {
            D::Sample0 => Mode::SAMPLE_ZERO,
            D::Min => Mode::MIN,
            D::Max => Mode::MAX,
            other => panic!("{name}: a filter the cases do not name: {other:?}"),
        });
        let depth_drawn = depth_mode.is_some_and(&drawn);
        let stencil_drawn = stencil.is_some()
            && stencil_filter == S::DepthResolvedSample
            && depth_filter != D::Sample0;
        let depth_msaa = match split {
            Split::Separate => lane.msaa(depth_pf),
            _ if depth_drawn || stencil_drawn => lane.texture(
                depth_pf,
                MSAA,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            ),
            _ => lane.msaa(depth_pf),
        };
        let stencil_msaa = match split {
            Split::Separate => lane.msaa(stencil_pf),
            Split::TwoTextures | Split::TwoSlices => depth_msaa.clone(),
        };
        // Each aspect's resolve texture and slice.
        let (depth_resolve, stencil_resolve) = match split {
            Split::TwoSlices => {
                let shape = Shape { slices: 2, ..ONE };
                let array = lane.texture(combined, shape, TextureUsage::RENDER_TARGET);
                ((array.clone(), 0), (array, 1))
            }
            _ => ((lane.single(depth_pf), 0), (lane.single(stencil_pf), 0)),
        };
        for (texture, slice) in [&depth_resolve, &stencil_resolve] {
            let mut clear = RenderPassDescriptor::new();
            let pf = texture.pixel_format();
            if pf != PixelFormat::Stencil8 {
                clear.depth_attachment.texture = Some(texture.clone());
                clear.depth_attachment.slice = *slice;
                clear.depth_attachment.clear_depth = 0.75;
                clear.depth_attachment.store_action = StoreAction::Store;
            }
            if pf != PixelFormat::Depth32Float {
                clear.stencil_attachment.texture = Some(texture.clone());
                clear.stencil_attachment.slice = *slice;
                clear.stencil_attachment.clear_stencil = 7;
                clear.stencil_attachment.store_action = StoreAction::Store;
            }
            lane.run(&clear, |_, _, _| {}).unwrap();
        }
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.color_attachments[0].texture = Some(lane.msaa(rgba8));
        descriptor.color_attachments[0].load_action = LoadAction::Clear;
        descriptor.color_attachments[0].store_action = StoreAction::DontCare;
        let d = &mut descriptor.depth_attachment;
        d.texture = Some(depth_msaa.clone());
        d.store_action = depth_store;
        d.depth_resolve_filter = depth_filter;
        if depth_store.resolves() {
            d.resolve_texture = Some(depth_resolve.0.clone());
            d.resolve_slice = depth_resolve.1;
        }
        let s = &mut descriptor.stencil_attachment;
        s.texture = Some(stencil_msaa.clone());
        s.store_action = stencil_store;
        s.stencil_resolve_filter = stencil_filter;
        if stencil_store.resolves() {
            s.resolve_texture = Some(stencil_resolve.0.clone());
            s.resolve_slice = stencil_resolve.1;
        }
        let pipeline = lane.pipeline(
            (FULL, MASKED),
            &[rgba8],
            (Some(depth_pf), Some(stencil_pf)),
            4,
        );
        let ran = lane.run(&descriptor, |l, e, h| {
            e.set_depth_stencil_state(&state).unwrap();
            for sample in 0..4 {
                e.set_stencil_reference_value(STENCILS[sample]);
                let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &pipeline, DEPTHS[sample]);
                h.push(mask);
            }
        });
        // Each aspect a subpass resolves in a subpass of its own, and one drawn into its own
        // texture after the pass; both drawn is not built here.
        let subpass_depth = depth_mode.filter(|_| !depth_drawn);
        let subpass_stencil = stencil
            .map(|_| Mode::SAMPLE_ZERO)
            .filter(|_| !stencil_drawn);
        let built = !(depth_drawn && stencil_drawn)
            && device_resolves(subpass_depth, None, true)
            && device_resolves(None, subpass_stencil, true);
        if built && (depth_drawn || stencil_drawn) {
            eprintln!("drawn: {name}: the depth {depth_drawn}, the stencil {stencil_drawn}");
        }
        // The separate textures' scratch image, and its copies, are made before the resolve.
        match (built, ran) {
            (_, Err(Error::Unsupported(u))) if split == Split::Separate => {
                eprintln!("refused: {name}: the separate textures' sample copies: {u:?}");
                continue;
            }
            (false, ran) => {
                assert_eq!(
                    ran,
                    Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve)),
                    "{name}"
                );
                eprintln!("refused: {name}: a split resolve the runtime does not build here");
                continue;
            }
            (true, ran) => ran.unwrap_or_else(|error| panic!("{name}: {error:?}")),
        }
        let combined_option = |aspect| match split {
            Split::Separate => BlitOption::None,
            Split::TwoTextures | Split::TwoSlices => aspect,
        };
        let (depth_texture, depth_slice) = &depth_resolve;
        let (stencil_texture, stencil_slice) = &stencil_resolve;
        if let Some(depth) = depth {
            let read = lane.read(
                depth_texture,
                (0, *depth_slice),
                4,
                combined_option(BlitOption::DepthFromDepthStencil),
            );
            assert_eq!(read, repeat(&depth.to_le_bytes(), texels), "{name}: depth");
            if split != Split::Separate {
                let kept = lane.read(
                    depth_texture,
                    (0, *depth_slice),
                    1,
                    BlitOption::StencilFromDepthStencil,
                );
                assert_eq!(kept, vec![7; texels], "{name}: the depth texture's stencil");
            }
        }
        if let Some(stencil) = stencil {
            let read = lane.read(
                stencil_texture,
                (0, *stencil_slice),
                1,
                combined_option(BlitOption::StencilFromDepthStencil),
            );
            assert_eq!(read, vec![stencil; texels], "{name}: stencil");
            if split != Split::Separate {
                let kept = lane.read(
                    stencil_texture,
                    (0, *stencil_slice),
                    4,
                    BlitOption::DepthFromDepthStencil,
                );
                assert_eq!(
                    kept,
                    repeat(&0.75f32.to_le_bytes(), texels),
                    "{name}: the stencil texture's depth"
                );
            }
        }
        eprintln!("{name}: Metal's resolves");
    }
    // `StoreAndMultisampleResolve` on separate textures resolves each and keeps their samples: a
    // later pass loads them and resolves the depth's maximum.
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let depth_msaa = lane.texture(PixelFormat::Depth32Float, MSAA, usage);
    let stencil_msaa = lane.texture(PixelFormat::Stencil8, MSAA, usage);
    let pass = |load, store, depth_filter, into: [&Texture; 2]| {
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.color_attachments[0].texture = Some(lane.msaa(rgba8));
        descriptor.color_attachments[0].load_action = LoadAction::Clear;
        descriptor.color_attachments[0].store_action = StoreAction::DontCare;
        let d = &mut descriptor.depth_attachment;
        d.texture = Some(depth_msaa.clone());
        d.load_action = load;
        d.store_action = store;
        d.depth_resolve_filter = depth_filter;
        d.resolve_texture = Some(into[0].clone());
        let s = &mut descriptor.stencil_attachment;
        s.texture = Some(stencil_msaa.clone());
        s.load_action = load;
        s.store_action = store;
        s.stencil_resolve_filter = S::Sample0;
        s.resolve_texture = Some(into[1].clone());
        descriptor
    };
    let first = [
        lane.single(PixelFormat::Depth32Float),
        lane.single(PixelFormat::Stencil8),
    ];
    let then = [
        lane.single(PixelFormat::Depth32Float),
        lane.single(PixelFormat::Stencil8),
    ];
    let pipeline = lane.pipeline(
        (FULL, MASKED),
        &[rgba8],
        (Some(PixelFormat::Depth32Float), Some(PixelFormat::Stencil8)),
        4,
    );
    let stored = pass(
        LoadAction::Clear,
        StoreAction::StoreAndMultisampleResolve,
        D::Sample0,
        [&first[0], &first[1]],
    );
    let ran = lane.run(&stored, |l, e, h| {
        e.set_depth_stencil_state(&state).unwrap();
        for sample in 0..4 {
            e.set_stencil_reference_value(STENCILS[sample]);
            let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
            e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
            l.draw(e, h, &pipeline, DEPTHS[sample]);
            h.push(mask);
        }
    });
    let loaded = pass(
        LoadAction::Load,
        StoreAction::MultisampleResolve,
        D::Max,
        [&then[0], &then[1]],
    );
    // A drawn depth is no subpass's: it is drawn into its own texture after the pass.
    let max_built = device_resolves(Some(Mode::MAX).filter(|&m| !drawn(m)), None, true);
    let ran = ran.and_then(|()| lane.run(&loaded, |_, _, _| {}));
    match (max_built, ran) {
        (_, Err(Error::Unsupported(u))) => {
            eprintln!("refused: keep: the separate textures' sample copies: {u:?}");
        }
        (false, ran) => {
            assert_eq!(
                ran,
                Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve)),
                "keep"
            );
            eprintln!("refused: keep: a split resolve the runtime does not build here");
        }
        (true, ran) => {
            ran.unwrap_or_else(|error| panic!("keep: {error:?}"));
            for (texture, depth) in [(&first[0], 0.5f32), (&then[0], 0.9)] {
                let read = lane.read(texture, (0, 0), 4, BlitOption::None);
                assert_eq!(read, repeat(&depth.to_le_bytes(), texels), "keep: depth");
            }
            for texture in [&first[1], &then[1]] {
                let read = lane.read(texture, (0, 0), 1, BlitOption::None);
                assert_eq!(read, vec![20; texels], "keep: stencil");
            }
            eprintln!("keep: Metal's resolves");
        }
    }
    quiet(lane, &messages);
}

/// An index the texture's type does not have is ignored, as plain Metal ignores it (probes on an
/// M4, four fills alike; its validation stops each): a 2D resolve texture's `resolveSlice` (any)
/// and `resolveDepthPlane` resolve into its one slice, an array's `resolveDepthPlane` changes
/// nothing, and a 2D attachment's slice and depth plane render its slice, each exactly as the index
/// 0 does.
#[test]
fn an_index_the_texture_type_lacks_is_ignored() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let samples = lane.pipeline((FULL, SAMPLES), &[rgba8], (None, None), 4);
    let msaa = lane.msaa(rgba8);
    let pair = || {
        lane.texture(
            rgba8,
            Shape { slices: 2, ..ONE },
            TextureUsage::RENDER_TARGET,
        )
    };
    // The resolve of the probe's samples into `resolve` named at `(slice, plane)`, read back at
    // `read`.
    let resolved = |resolve: &Texture, (slice, plane): (u64, u64), read: u64| {
        let mut descriptor = color_pass(
            &msaa,
            Some(resolve),
            StoreAction::MultisampleResolve,
            (0, slice),
        );
        descriptor.color_attachments[0].resolve_depth_plane = plane;
        lane.run(&descriptor, |l, e, h| l.draw(e, h, &samples, 0.5))
            .unwrap_or_else(|error| panic!("a resolve at {slice}, {plane}: {error:?}"));
        lane.read(resolve, (0, read), 4, BlitOption::None)
    };
    let reference = resolved(&lane.single(rgba8), (0, 0), 0);
    for named in [(1, 0), (5, 0), (0, 1), (3, 2)] {
        assert_eq!(
            resolved(&lane.single(rgba8), named, 0),
            reference,
            "a 2D resolve texture at slice and plane {named:?}"
        );
    }
    assert_eq!(
        resolved(&pair(), (1, 1), 1),
        reference,
        "an array's slice 1 at plane 1"
    );
    // A pass clearing `target` named at `(slice, plane)`, read back at `read`.
    let cleared = |target: &Texture, (slice, plane): (u64, u64), read: u64| {
        let mut descriptor = RenderPassDescriptor::new();
        let attachment = &mut descriptor.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.slice = slice;
        attachment.depth_plane = plane;
        attachment.load_action = LoadAction::Clear;
        attachment.clear_color = ClearColor::new(0.25, 0.75, 0.0, 1.0);
        attachment.store_action = StoreAction::Store;
        lane.run(&descriptor, |_, _, _| {})
            .unwrap_or_else(|error| panic!("a pass at {slice}, {plane}: {error:?}"));
        lane.read(target, (0, read), 4, BlitOption::None)
    };
    let clear = cleared(&lane.single(rgba8), (0, 0), 0);
    assert_eq!(
        clear,
        repeat(&[0x40, 0xbf, 0x00, 0xff], (SIZE * SIZE) as usize)
    );
    for named in [(1, 0), (5, 0), (0, 1), (3, 2)] {
        assert_eq!(
            cleared(&lane.single(rgba8), named, 0),
            clear,
            "a 2D attachment at slice and plane {named:?}"
        );
    }
    assert_eq!(
        cleared(&pair(), (1, 1), 1),
        clear,
        "an array's slice 1 at plane 1"
    );
    quiet(lane, &messages);
}

/// A depth/stencil state whose tests always pass, writing the depth and replacing the stencil.
fn writing_state(lane: &Lane) -> DepthStencilState {
    let mut always = StencilDescriptor::default();
    always.stencil_compare_function = CompareFunction::Always;
    always.depth_stencil_pass_operation = StencilOperation::Replace;
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = true;
    state.front_face_stencil = always;
    state.back_face_stencil = always;
    lane.device.new_depth_stencil_state(&state).unwrap()
}

/// A single-sampled `Depth32Float_Stencil8` resolve texture holding depth 0.75 and stencil 7.
fn held_resolve(lane: &Lane) -> Texture {
    let resolve = lane.single(PixelFormat::Depth32FloatStencil8);
    let mut clear = RenderPassDescriptor::new();
    clear.depth_attachment.texture = Some(resolve.clone());
    clear.depth_attachment.clear_depth = 0.75;
    clear.depth_attachment.store_action = StoreAction::Store;
    clear.stencil_attachment.texture = Some(resolve.clone());
    clear.stencil_attachment.clear_stencil = 7;
    clear.stencil_attachment.store_action = StoreAction::Store;
    lane.run(&clear, |_, _, _| {}).unwrap();
    resolve
}

/// The depth and the stencil bytes of a `Depth32Float_Stencil8` texture's level 0, slice 0.
fn depth_and_stencil(lane: &Lane, texture: &Texture) -> (Vec<u8>, Vec<u8>) {
    let depth = lane.read(texture, (0, 0), 4, BlitOption::DepthFromDepthStencil);
    let stencil = lane.read(texture, (0, 0), 1, BlitOption::StencilFromDepthStencil);
    (depth, stencil)
}

/// `DepthResolvedSample` with a depth attachment that does not resolve takes the stencil of the
/// sample its filter picks, as for one that does (of equal depths the last after `Min`, the first
/// after `Max`); and one texture's depth resolved alone with its stencil stored resolves the
/// depth by its filter and keeps the stencil's samples, the resolve texture's stencil untouched
/// (probes on an M4, four fills alike; Metal's validation stops both). The samples are probe 6's
/// (depths 0.5, 0.1, 0.9 and 0.3, stencils 20 to 23); the resolve texture holds depth 0.75 and
/// stencil 7 before the pass. A stored stencil is resolved by a later pass that loads it, by
/// `Sample0`.
#[test]
fn depth_resolved_samples_and_stored_stencils_are_plain_metals() {
    use MultisampleDepthResolveFilter as D;
    let (messages, _cache, lane) = lane();
    let device_resolves = device_resolves(&lane.device);
    let drawn = drawn_depth(&lane.device);
    let combined = PixelFormat::Depth32FloatStencil8;
    let rgba8 = PixelFormat::RGBA8Unorm;
    let texels = (SIZE * SIZE) as usize;
    let state = writing_state(&lane);
    let resolve_texture = || held_resolve(&lane);
    let read = |resolve: &Texture| depth_and_stencil(&lane, resolve);
    let pipeline = lane.pipeline(
        (FULL, MASKED),
        &[rgba8],
        (Some(combined), Some(combined)),
        4,
    );
    // (name, the depth's filter, whether the depth resolves (else it is not stored) and the
    // stencil is stored (else it resolves by `DepthResolvedSample`), and Metal's resolve
    // texture's depth and stencil).
    type Case = (&'static str, D, bool, f32, u8);
    let cases: [Case; 6] = [
        ("drs_d_dontcare_sample0", D::Sample0, false, 0.75, 20),
        ("drs_d_dontcare_min", D::Min, false, 0.75, 21),
        ("drs_d_dontcare_max", D::Max, false, 0.75, 22),
        ("mix_d_resolve_s_store", D::Sample0, true, 0.5, 7),
        ("mix_d_resolve_s_store_min", D::Min, true, 0.1, 7),
        ("mix_d_resolve_s_store_max", D::Max, true, 0.9, 7),
    ];
    for (name, filter, mixed, depth, stencil) in cases {
        let mode = match filter {
            D::Sample0 => Mode::SAMPLE_ZERO,
            D::Min => Mode::MIN,
            D::Max => Mode::MAX,
            other => panic!("{name}: a filter the cases do not name: {other:?}"),
        };
        // The stencil picked by depth after `Min` or `Max`, or a depth the device has no mode
        // for, is drawn from the samples, read through sampled views.
        let depth_drawn = mixed && drawn(mode);
        let picked = !mixed && filter != D::Sample0;
        let msaa = {
            let mut descriptor =
                TextureDescriptor::texture_2d(combined, MSAA.size, MSAA.size, false);
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = MSAA.samples;
            descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::private();
            match lane.device.new_texture(&descriptor) {
                Ok(texture) => texture,
                Err(error) => {
                    eprintln!("refused: {name}: a sampled {combined:?} attachment: {error:?}");
                    continue;
                }
            }
        };
        let resolve = resolve_texture();
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.color_attachments[0].texture = Some(lane.msaa(rgba8));
        descriptor.color_attachments[0].load_action = LoadAction::Clear;
        descriptor.color_attachments[0].store_action = StoreAction::DontCare;
        let d = &mut descriptor.depth_attachment;
        d.texture = Some(msaa.clone());
        d.load_action = LoadAction::Clear;
        d.store_action = StoreAction::DontCare;
        d.depth_resolve_filter = filter;
        let s = &mut descriptor.stencil_attachment;
        s.texture = Some(msaa.clone());
        s.load_action = LoadAction::Clear;
        s.clear_stencil = 0x42;
        if mixed {
            let d = &mut descriptor.depth_attachment;
            d.store_action = StoreAction::MultisampleResolve;
            d.resolve_texture = Some(resolve.clone());
            descriptor.stencil_attachment.store_action = StoreAction::Store;
        } else {
            let s = &mut descriptor.stencil_attachment;
            s.store_action = StoreAction::MultisampleResolve;
            s.resolve_texture = Some(resolve.clone());
            s.stencil_resolve_filter = MultisampleStencilResolveFilter::DepthResolvedSample;
        }
        let ran = lane.run(&descriptor, |l, e, h| {
            e.set_depth_stencil_state(&state).unwrap();
            for sample in 0..4 {
                e.set_stencil_reference_value(STENCILS[sample]);
                let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &pipeline, DEPTHS[sample]);
                h.push(mask);
            }
        });
        // The device's subpass modes: the resolved depth's (none where drawn), the stencil's
        // `SAMPLE_ZERO` after `Sample0` (none where picked, drawn after the pass).
        let depth_mode = (mixed && !depth_drawn).then_some(mode);
        let stencil_mode = (!mixed && !picked).then_some(Mode::SAMPLE_ZERO);
        if !device_resolves(depth_mode, stencil_mode, true) {
            assert_eq!(
                ran,
                Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve)),
                "{name}"
            );
            eprintln!(
                "refused: {name}: the device's resolve modes lack depth {depth_mode:?} with \
                 stencil {stencil_mode:?}"
            );
            continue;
        }
        ran.unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(
            read(&resolve),
            (repeat(&depth.to_le_bytes(), texels), vec![stencil; texels]),
            "{name}: the resolve texture's depth and stencil"
        );
        if !mixed {
            continue;
        }
        // The stored stencil's samples: sample 0's, resolved by a pass loading them.
        let again = resolve_texture();
        let mut load = RenderPassDescriptor::new();
        let s = &mut load.stencil_attachment;
        s.texture = Some(msaa.clone());
        s.load_action = LoadAction::Load;
        s.store_action = StoreAction::MultisampleResolve;
        s.resolve_texture = Some(again.clone());
        let ran = lane.run(&load, |_, _, _| {});
        if !device_resolves(None, Some(Mode::SAMPLE_ZERO), true) {
            eprintln!("refused: {name}: the device resolves no stencil alone: {ran:?}");
            continue;
        }
        ran.unwrap_or_else(|error| panic!("{name}: the stored stencil's resolve: {error:?}"));
        assert_eq!(
            read(&again),
            (repeat(&0.75f32.to_le_bytes(), texels), vec![20; texels]),
            "{name}: the stored stencil's sample 0"
        );
    }
    quiet(lane, &messages);
}

/// The kind [`stored_and_resolved_aspects_of_one_texture_are_metals`] takes: one texture's depth
/// stored and resolved beside its stencil resolved alone, which Metal's validation stops and plain
/// Metal answers with a layout pattern.
pub(super) const MIXED_RESOLVES: [Degradation; 1] = [Degradation::MixedDepthStencilResolve];

/// One texture's depth or stencil stored and resolved (`StoreAndMultisampleResolve`) beside the
/// other aspect stored, the descriptor's action or a late one, as Metal runs it (a probe,
/// hazard-tracked textures, an M3 Max and an M4 alike; valid usage): that aspect resolved alone,
/// the resolve texture's other aspect untouched, and both aspects' samples kept, which a later
/// pass loads and resolves by sample 0. Beside them, the depth stored and resolved beside the
/// stencil resolved alone, which Metal's validation stops, as the runtime runs it: a recorded
/// near-equivalent, announced once.
#[test]
fn stored_and_resolved_aspects_of_one_texture_are_metals() {
    let (messages, _cache, lane) = lane();
    let device_resolves = device_resolves(&lane.device);
    let drawn = drawn_depth(&lane.device);
    let combined = PixelFormat::Depth32FloatStencil8;
    let rgba8 = PixelFormat::RGBA8Unorm;
    let texels = (SIZE * SIZE) as usize;
    let state = writing_state(&lane);
    let resolve_texture = || held_resolve(&lane);
    let read = |resolve: &Texture| depth_and_stencil(&lane, resolve);
    let pipeline = lane.pipeline(
        (FULL, MASKED),
        &[rgba8],
        (Some(combined), Some(combined)),
        4,
    );
    let (stored, both) = (StoreAction::Store, StoreAction::StoreAndMultisampleResolve);
    // (name, the depth's and the stencil's store actions, whether they are set late, and
    // Metal's resolve texture's depth and stencil). The depth resolves by `Min`, the stencil by
    // `Sample0`.
    type Case = (&'static str, StoreAction, StoreAction, bool, f32, u8);
    let cases: [Case; 4] = [
        ("v_d_samr_s_store", both, stored, false, 0.1, 7),
        ("v_d_store_s_samr", stored, both, false, 0.75, 20),
        ("late_d_samr_s_store", both, stored, true, 0.1, 7),
        ("late_d_store_s_samr", stored, both, true, 0.75, 20),
    ];
    for (name, depth_store, stencil_store, late, depth, stencil) in cases {
        let depth_resolves = depth_store == both;
        let msaa = lane.msaa(combined);
        let resolve = resolve_texture();
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.color_attachments[0].texture = Some(lane.msaa(rgba8));
        descriptor.color_attachments[0].load_action = LoadAction::Clear;
        descriptor.color_attachments[0].store_action = StoreAction::DontCare;
        let d = &mut descriptor.depth_attachment;
        d.texture = Some(msaa.clone());
        d.load_action = LoadAction::Clear;
        d.store_action = depth_store;
        d.depth_resolve_filter = MultisampleDepthResolveFilter::Min;
        let s = &mut descriptor.stencil_attachment;
        s.texture = Some(msaa.clone());
        s.load_action = LoadAction::Clear;
        s.clear_stencil = 0x42;
        s.store_action = stencil_store;
        if depth_resolves {
            descriptor.depth_attachment.resolve_texture = Some(resolve.clone());
        } else {
            descriptor.stencil_attachment.resolve_texture = Some(resolve.clone());
        }
        if late {
            if depth_resolves {
                descriptor.depth_attachment.store_action = StoreAction::Unknown;
            } else {
                descriptor.stencil_attachment.store_action = StoreAction::Unknown;
            }
        }
        let ran = lane.run(&descriptor, |l, e, h| {
            e.set_depth_stencil_state(&state).unwrap();
            for sample in 0..4 {
                e.set_stencil_reference_value(STENCILS[sample]);
                let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &pipeline, DEPTHS[sample]);
                h.push(mask);
            }
            if late && depth_resolves {
                e.set_depth_store_action(both).unwrap();
            } else if late {
                e.set_stencil_store_action(both).unwrap();
            }
        });
        // The device's subpass modes: the resolved depth's (none where drawn) or the stencil's.
        let depth_mode = (depth_resolves && !drawn(Mode::MIN)).then_some(Mode::MIN);
        let stencil_mode = (!depth_resolves).then_some(Mode::SAMPLE_ZERO);
        if !device_resolves(depth_mode, stencil_mode, true) {
            assert_eq!(
                ran,
                Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve)),
                "{name}"
            );
            eprintln!(
                "refused: {name}: the device's resolve modes lack depth {depth_mode:?} with \
                 stencil {stencil_mode:?}"
            );
            continue;
        }
        ran.unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(
            read(&resolve),
            (repeat(&depth.to_le_bytes(), texels), vec![stencil; texels]),
            "{name}: the resolve texture's depth and stencil"
        );
        // Both aspects' samples kept: sample 0's, resolved by a pass loading them.
        let again = resolve_texture();
        let mut load = RenderPassDescriptor::new();
        let d = &mut load.depth_attachment;
        d.texture = Some(msaa.clone());
        d.load_action = LoadAction::Load;
        d.store_action = StoreAction::MultisampleResolve;
        d.resolve_texture = Some(again.clone());
        let s = &mut load.stencil_attachment;
        s.texture = Some(msaa.clone());
        s.load_action = LoadAction::Load;
        s.store_action = StoreAction::MultisampleResolve;
        s.resolve_texture = Some(again.clone());
        let zero = Some(Mode::SAMPLE_ZERO);
        let ran = lane.run(&load, |_, _, _| {});
        if !device_resolves(zero, zero, true) {
            eprintln!("refused: {name}: the device resolves no depth and stencil: {ran:?}");
            continue;
        }
        ran.unwrap_or_else(|error| panic!("{name}: the stored samples' resolve: {error:?}"));
        assert_eq!(
            read(&again),
            (
                repeat(&DEPTHS[0].to_le_bytes(), texels),
                vec![STENCILS[0] as u8; texels]
            ),
            "{name}: the stored samples' sample 0"
        );
    }
    // The depth stored and resolved beside the stencil resolved alone, which Metal's validation
    // stops (plain Metal: a layout pattern of the resolved depth), runs as it once ran: both
    // aspects resolved and both aspects' samples kept, announced once per device (a recorded
    // near-equivalent).
    messages.take_notices();
    let msaa = lane.msaa(combined);
    let resolve = resolve_texture();
    let mut descriptor = RenderPassDescriptor::new();
    descriptor.color_attachments[0].texture = Some(lane.msaa(rgba8));
    descriptor.color_attachments[0].load_action = LoadAction::Clear;
    descriptor.color_attachments[0].store_action = StoreAction::DontCare;
    let d = &mut descriptor.depth_attachment;
    d.texture = Some(msaa.clone());
    d.load_action = LoadAction::Clear;
    d.store_action = both;
    d.depth_resolve_filter = MultisampleDepthResolveFilter::Min;
    d.resolve_texture = Some(resolve.clone());
    let s = &mut descriptor.stencil_attachment;
    s.texture = Some(msaa.clone());
    s.load_action = LoadAction::Clear;
    s.clear_stencil = 0x42;
    s.store_action = StoreAction::MultisampleResolve;
    s.resolve_texture = Some(resolve.clone());
    let run = || {
        lane.run(&descriptor, |l, e, h| {
            e.set_depth_stencil_state(&state).unwrap();
            for sample in 0..4 {
                e.set_stencil_reference_value(STENCILS[sample]);
                let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &pipeline, DEPTHS[sample]);
                h.push(mask);
            }
        })
    };
    let ran = run();
    let depth_mode = (!drawn(Mode::MIN)).then_some(Mode::MIN);
    if device_resolves(depth_mode, Some(Mode::SAMPLE_ZERO), true) {
        ran.unwrap_or_else(|error| panic!("near: {error:?}"));
        assert_eq!(
            read(&resolve),
            (repeat(&0.1f32.to_le_bytes(), texels), vec![20; texels]),
            "near: the resolve texture's depth and stencil"
        );
        run().unwrap_or_else(|error| panic!("near, again: {error:?}"));
        assert_once(&messages.take_notices(), &MIXED_RESOLVES);
        // Both aspects' samples kept (the stencil's too, so a driver over Metal runs Metal's
        // valid store and resolve of both): sample 0's, resolved by a pass loading them.
        let again = resolve_texture();
        let mut load = RenderPassDescriptor::new();
        let d = &mut load.depth_attachment;
        d.texture = Some(msaa.clone());
        d.load_action = LoadAction::Load;
        d.store_action = StoreAction::MultisampleResolve;
        d.resolve_texture = Some(again.clone());
        let s = &mut load.stencil_attachment;
        s.texture = Some(msaa.clone());
        s.load_action = LoadAction::Load;
        s.store_action = StoreAction::MultisampleResolve;
        s.resolve_texture = Some(again.clone());
        let ran = lane.run(&load, |_, _, _| {});
        let zero = Some(Mode::SAMPLE_ZERO);
        if device_resolves(zero, zero, true) {
            ran.unwrap_or_else(|error| panic!("near: the kept samples' resolve: {error:?}"));
            assert_eq!(
                read(&again),
                (
                    repeat(&DEPTHS[0].to_le_bytes(), texels),
                    vec![STENCILS[0] as u8; texels]
                ),
                "near: the kept samples' sample 0"
            );
        } else {
            eprintln!("refused: near: the device resolves no depth and stencil: {ran:?}");
        }
    } else {
        assert_eq!(
            ran,
            Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve)),
            "near"
        );
        eprintln!("refused: near: the device's resolve modes lack depth {depth_mode:?} with stencil SAMPLE_ZERO");
    }
    quiet(lane, &messages);
}

/// A resolve level of another extent than the attachment's resolves the overlap, as plain Metal
/// does (probes on an M4, four fills alike; its validation stops each): a color level reaching past
/// an attachment cleared by its load takes the clear color past it, and a depth or stencil level
/// keeps its texels there. A multisampled color resolve texture of another sample count is left
/// untouched. The samples are probe 6's.
#[test]
fn resolve_sizes_and_counts_are_plain_metals() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let samples = lane.pipeline((FULL, SAMPLES), &[rgba8], (None, None), 4);
    let msaa = lane.msaa(rgba8);
    let clear = ClearColor::new(0.25, 0.75, 0.0, 1.0);
    let shaped = |pf, size, samples| {
        lane.texture(
            pf,
            Shape {
                size,
                samples,
                ..ONE
            },
            TextureUsage::RENDER_TARGET,
        )
    };
    // `texture` cleared to `value` by a pass of its own.
    let fill = |texture: &Texture, value: ClearColor| {
        let mut descriptor = RenderPassDescriptor::new();
        let attachment = &mut descriptor.color_attachments[0];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.clear_color = value;
        attachment.store_action = StoreAction::Store;
        lane.run(&descriptor, |_, _, _| {}).unwrap();
    };
    let before = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    // The probe's samples resolved into `resolve`, filled with `before` first, read back whole.
    let resolved = |resolve: &Texture| {
        fill(resolve, before);
        let mut descriptor = color_pass(
            &msaa,
            Some(resolve),
            StoreAction::MultisampleResolve,
            (0, 0),
        );
        descriptor.color_attachments[0].clear_color = clear;
        lane.run(&descriptor, |l, e, h| l.draw(e, h, &samples, 0.5))
            .unwrap_or_else(|error| panic!("a {}² resolve: {error:?}", resolve.width()));
        lane.read(resolve, (0, 0), 4, BlitOption::None)
    };
    let reference = resolved(&shaped(rgba8, SIZE, 1));
    let cleared = [0x40, 0xbf, 0x00, 0xff];
    // The texel at `(x, y)` of a `width`-wide level: the reference's within the attachment, the
    // clear color past it.
    let expected = |width: u64, height: u64| {
        let mut texels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                texels.extend_from_slice(match x < SIZE && y < SIZE {
                    true => {
                        let at = 4 * (y * SIZE + x) as usize;
                        &reference[at..at + 4]
                    }
                    false => &cleared,
                });
            }
        }
        texels
    };
    for (width, height) in [(8, 8), (2, 2), (8, 2), (2, 8)] {
        let mut descriptor = TextureDescriptor::texture_2d(rgba8, width, height, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        let resolve = lane.device.new_texture(&descriptor).unwrap();
        assert_eq!(
            resolved(&resolve),
            expected(width, height),
            "a {width}x{height} color level"
        );
    }

    // Depth: a 4x Depth32Float attachment of probe 6's depths, resolved by `Sample0` (0.5) into
    // levels holding 0.75.
    let depth_pf = PixelFormat::Depth32Float;
    let depth_msaa = lane.msaa(depth_pf);
    let depth_pipeline = lane.pipeline((FULL, MASKED), &[rgba8], (Some(depth_pf), None), 4);
    let state = {
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Always;
        state.depth_write_enabled = true;
        lane.device.new_depth_stencil_state(&state).unwrap()
    };
    for side in [8, 2] {
        let resolve = shaped(depth_pf, side, 1);
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.depth_attachment.texture = Some(resolve.clone());
        descriptor.depth_attachment.clear_depth = 0.75;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        lane.run(&descriptor, |_, _, _| {}).unwrap();
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.color_attachments[0].texture = Some(lane.msaa(rgba8));
        descriptor.color_attachments[0].load_action = LoadAction::Clear;
        descriptor.color_attachments[0].store_action = StoreAction::DontCare;
        let depth = &mut descriptor.depth_attachment;
        depth.texture = Some(depth_msaa.clone());
        depth.load_action = LoadAction::Clear;
        depth.store_action = StoreAction::MultisampleResolve;
        depth.resolve_texture = Some(resolve.clone());
        lane.run(&descriptor, |l, e, h| {
            e.set_depth_stencil_state(&state).unwrap();
            for (sample, &z) in DEPTHS.iter().enumerate() {
                let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &depth_pipeline, z);
                h.push(mask);
            }
        })
        .unwrap_or_else(|error| panic!("a {side}² depth resolve: {error:?}"));
        let mut metal = Vec::new();
        for y in 0..side {
            for x in 0..side {
                let depth: f32 = if x < SIZE && y < SIZE { 0.5 } else { 0.75 };
                metal.extend_from_slice(&depth.to_le_bytes());
            }
        }
        assert_eq!(
            lane.read(&resolve, (0, 0), 4, BlitOption::None),
            metal,
            "a {side}² depth level"
        );
    }

    // Another sample count: a 2x texture holding `before` keeps it.
    match lane.device.new_texture(&{
        let mut descriptor = TextureDescriptor::texture_2d(rgba8, SIZE, SIZE, false);
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = 2;
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        descriptor
    }) {
        Err(error) => eprintln!("refused: a 2x {rgba8:?} texture: {error:?}"),
        Ok(two) => {
            fill(&two, before);
            lane.run(
                &color_pass(&msaa, Some(&two), StoreAction::MultisampleResolve, (0, 0)),
                |l, e, h| l.draw(e, h, &samples, 0.5),
            )
            .unwrap();
            // Its samples, resolved by a pass loading them.
            let single = shaped(rgba8, SIZE, 1);
            let mut load = color_pass(&two, Some(&single), StoreAction::MultisampleResolve, (0, 0));
            load.color_attachments[0].load_action = LoadAction::Load;
            lane.run(&load, |_, _, _| {}).unwrap();
            assert_eq!(
                lane.read(&single, (0, 0), 4, BlitOption::None),
                repeat(&[0x00, 0x00, 0xff, 0xff], (SIZE * SIZE) as usize),
                "the 2x texture keeps its texels"
            );
        }
    }
    quiet(lane, &messages);
}
