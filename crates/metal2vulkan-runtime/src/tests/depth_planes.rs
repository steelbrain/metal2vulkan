//! A 3D texture's depth planes as colour attachments, under a device cfg, on every device of the
//! cfg's kind: the passes a browser's GL layer makes on a 3D texture (`texImage3D` of an RGB
//! format, whose alpha it clears through a pass on each plane; `framebufferTextureLayer` of a 3D
//! texture, then a clear or a draw), on the texture its guest makes: `ShaderRead | ShaderWrite |
//! RenderTarget`, Managed.
//!
//! For `RGBA8Unorm` and `RGBA16Float` (the layer's `RGB16F`): an 8 x 8 x 4 texture of two levels
//! (the second 4 x 4 x 2, a level shallower than the first) and an 8 x 8 x 1 one of one level, every
//! plane of every level written first through `replace_region`, each texel naming its level, `x`,
//! `y` and plane. One command buffer then encodes:
//!
//! - level 0's plane 2: `Clear`, and one draw of the top half (the owned `colored_vertex` pair);
//! - level 1's plane 1: the same, on the shallower level;
//! - level 0's plane 0: `Load`, and the same draw;
//! - level 0's plane 1: `Clear` alone, with no draw;
//! - the one-deep texture's plane 0: `Clear` and the draw;
//!
//! then copies every plane of both levels and the one-deep texture into a Shared buffer, and samples
//! each texel's centre through a nearest sampler state (the owned `msf_slot3d_linear_zero`): level
//! 0 through the texture, level 1 through a 3D view of it. Every texel is the texture's own answer,
//! the planes no pass named included. Two passes are refused as they begin: a plane past level 1's
//! depth ([`Refusal::AttachmentPlace`]) and a pass of two layers on a 3D texture
//! ([`RenderPassKind::TextureType`]).
//!
//! Recorded (`tests::layouts`): each rendering's colour attachment on a 3D image is a 2D view of one
//! layer, the plane, at its level, of an image made `VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT`; the
//! run prints them.
//!
//! `llvm-as` assembles the fixtures; without it the test skips visibly.

use crate::tests::command::on_device::{each_device, finished};
use crate::tests::draw::on_device::{
    descriptor, examples, floats, library, state, vertices, Buffers, GREEN,
};
use crate::tests::layouts::{rendered, Recording};
use crate::texture::TextureObject;
use crate::{
    Buffer, BufferLayout, ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType,
    Error, FunctionDescriptor, HostBytes, IndexClass, LoadAction, NumberedSampler, PixelFormat,
    PrimitiveType, Region, RenderPassDescriptor, RenderPassKind, RenderPipelineState,
    SamplerDescriptor, SamplerState, StageSource, StorageMode, StoreAction, Texture,
    TextureDescriptor, TexturePlace, TextureType, TextureUsage,
};
use crate::{Gap, Refusal};
use ash::vk::{self, Handle};
use metal2vulkan_devtools::engine_dir;

/// The deep texture's extent and levels.
const SIDE: u32 = 8;
const DEPTH: u32 = 4;
const LEVELS: u32 = 2;
/// The guest's usage: `ShaderRead | ShaderWrite | RenderTarget`.
const USAGE: TextureUsage = TextureUsage::SHADER_READ
    .union(TextureUsage::SHADER_WRITE)
    .union(TextureUsage::RENDER_TARGET);
/// The two clear colours, as numerators of the format's scale.
const CLEAR_A: [u32; 4] = [30, 180, 60, 255];
const CLEAR_B: [u32; 4] = [200, 20, 120, 255];
/// The textures each format's cell makes an index for.
const TEXTURES: u32 = 3;

/// How a format holds a channel.
#[derive(Clone, Copy)]
enum Kind {
    Unorm8,
    Half,
}

impl Kind {
    /// A texel's bytes.
    fn unit(self) -> usize {
        match self {
            Kind::Unorm8 => 4,
            Kind::Half => 8,
        }
    }

    /// The denominator of the values the test writes: each is a numerator below it over it, which
    /// the format holds exactly.
    fn scale(self) -> f32 {
        match self {
            Kind::Unorm8 => 255.0,
            Kind::Half => 256.0,
        }
    }

    /// The bytes of a texel of `color`, a value the format holds exactly.
    fn bytes(self, color: [f32; 4]) -> Vec<u8> {
        match self {
            Kind::Unorm8 => color
                .iter()
                .map(|&c| {
                    let byte = (c * 255.0).round();
                    assert!((c * 255.0 - byte).abs() < 1e-3, "{c} is no unorm8");
                    byte as u8
                })
                .collect(),
            Kind::Half => color.iter().flat_map(|&c| half(c).to_le_bytes()).collect(),
        }
    }
}

/// `x`'s `half` bits, for a value a `half` holds exactly: zero, or a normal one.
fn half(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent) && bits & 0x1fff == 0, "{x}");
    ((bits >> 16) & 0x8000) as u16 | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// The colour of numerators `n` over `kind`'s scale.
fn color(kind: Kind, n: [u32; 4]) -> [f32; 4] {
    n.map(|n| n as f32 / kind.scale())
}

/// The texel written first at `(x, y)` of plane `z` of level `level`: each channel names one of
/// them.
fn written(kind: Kind, level: u32, [x, y, z]: [u32; 3]) -> [f32; 4] {
    color(kind, [10 * x + level, 10 * y + 3, 40 * z + 5, 200 + level])
}

/// What a pass leaves at row `y` of the plane it renders, of a level `side` texels high: the draw's
/// green over the top half where it draws, else its clear colour, or what was there for a `Load`.
fn passed(drawn: bool, clear: Option<[f32; 4]>, side: u32, y: u32, before: [f32; 4]) -> [f32; 4] {
    match (drawn && y < side / 2, clear) {
        (true, _) => GREEN,
        (false, Some(clear)) => clear,
        (false, None) => before,
    }
}

/// One pass on a texture: its level and plane, its load action's clear colour (`None` for `Load`)
/// and whether it draws.
struct Planed<'a> {
    texture: &'a Texture,
    level: u32,
    plane: u32,
    clear: Option<[f32; 4]>,
    drawn: bool,
}

/// The extent of level `level` of `texture`.
fn extent(texture: &Texture, level: u32) -> [u32; 3] {
    let d = texture.descriptor();
    [d.width, d.height, d.depth].map(|e| (e >> level).max(1))
}

/// Every texel of level `level` of `texture` once `passes` ran, plane by plane, row by row.
fn want(kind: Kind, texture: &Texture, level: u32, passes: &[Planed<'_>]) -> Vec<[f32; 4]> {
    let [w, h, d] = extent(texture, level);
    let mut texels = Vec::new();
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let mut texel = written(kind, level, [x, y, z]);
                let named = (passes.iter()).filter(|p| {
                    p.texture.image() == texture.image() && p.level == level && p.plane == z
                });
                for p in named {
                    texel = passed(p.drawn, p.clear, h, y, texel);
                }
                texels.push(texel);
            }
        }
    }
    texels
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

/// Panics, naming the first texel and how many differ, where `got` is not `want`.
fn exact<T: PartialEq + std::fmt::Debug>(name: &str, what: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{name}: {what}: the texels read");
    let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if let Some(&first) = differ.first() {
        panic!(
            "{name}: {what}: {} of {} texels differ, the first (texel {first}): {:?}, not {:?}",
            differ.len(),
            got.len(),
            got[first],
            want[first]
        );
    }
}

/// `msf_slot3d_linear_zero`'s state, its constexpr sampler numbered `constexpr`.
fn sampling(name: &str, device: &Device, constexpr: u32) -> Option<ComputePipelineState> {
    let library = library(&engine_dir().join("msf_runtime_samplers.ll"))?;
    let function = (library.function("msf_slot3d_linear_zero")).expect("the kernel");
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    let state = device.new_compute_pipeline_state(&descriptor, |samplers| {
        assert!(
            matches!(samplers, [NumberedSampler::Constexpr(_)]),
            "{name}: {samplers:?}"
        );
        Ok(vec![constexpr])
    });
    Some(state.unwrap_or_else(|error| panic!("{name}: msf_slot3d_linear_zero: {error}")))
}

/// What one device needs for every format: the draw's vertices, the sampling kernel and its
/// sampler state.
struct Shared<'a> {
    device: &'a Device,
    sampler: &'a SamplerState,
    kernel: &'a ComputePipelineState,
    vertices: &'a (Buffer, Buffer),
}

/// The cell for `pf` held as `kind` with the draw `drawn`, its textures at indices from `first`.
fn cell(
    name: &str,
    shared: &Shared<'_>,
    (drawn, first): (&RenderPipelineState, u32),
    pf: PixelFormat,
    kind: Kind,
) {
    let Shared {
        device,
        sampler,
        kernel,
        vertices: (positions, colors),
    } = shared;
    let name = &format!("{name}: {pf:?}");
    let texture = |depth: u32, levels: u32, index: u32| {
        let mut d = TextureDescriptor::new(TextureType::Type3D, pf, SIDE, SIDE);
        d.depth = depth;
        d.mipmap_level_count = levels;
        d.usage = USAGE;
        d.storage_mode = StorageMode::Managed;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        for level in 0..levels {
            let size = extent(&texture, level);
            let [w, h, d] = size;
            let bytes: Vec<u8> = (0..d)
                .flat_map(|z| (0..h).flat_map(move |y| (0..w).map(move |x| [x, y, z])))
                .flat_map(|at| kind.bytes(written(kind, level, at)))
                .collect();
            let row = u64::from(w) * kind.unit() as u64;
            let region = Region {
                origin: [0; 3],
                size,
            };
            let image = row * u64::from(h);
            let bytes = HostBytes::from(&bytes[..]);
            (texture.replace_region(region, level, 0, bytes, row, image))
                .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
        }
        texture
    };
    let deep = texture(DEPTH, LEVELS, first);
    let flat = texture(1, 1, first + 1);
    let second = (deep.new_texture_view(pf, TextureType::Type3D, 1..2, 0..1, Some(first + 2)))
        .unwrap_or_else(|error| panic!("{name}: the view of level 1: {error}"));

    let [a, b] = [CLEAR_A, CLEAR_B].map(|n| color(kind, n));
    let passes = [
        Planed {
            texture: &deep,
            level: 0,
            plane: 2,
            clear: Some(a),
            drawn: true,
        },
        Planed {
            texture: &deep,
            level: 1,
            plane: 1,
            clear: Some(a),
            drawn: true,
        },
        Planed {
            texture: &deep,
            level: 0,
            plane: 0,
            clear: None,
            drawn: true,
        },
        Planed {
            texture: &deep,
            level: 0,
            plane: 1,
            clear: Some(b),
            drawn: false,
        },
        Planed {
            texture: &flat,
            level: 0,
            plane: 0,
            clear: Some(a),
            drawn: true,
        },
    ];
    let attachment = |p: &Planed<'_>| {
        let mut d = RenderPassDescriptor::default();
        let c = &mut d.color_attachments[0];
        c.texture = Some(p.texture.clone());
        c.level = p.level;
        c.depth_plane = p.plane;
        c.store_action = StoreAction::Store;
        match p.clear {
            Some([red, green, blue, alpha]) => {
                c.load_action = LoadAction::Clear;
                c.clear_color = crate::ClearColor::new(
                    f64::from(red),
                    f64::from(green),
                    f64::from(blue),
                    f64::from(alpha),
                );
            }
            None => c.load_action = LoadAction::Load,
        }
        d
    };

    let queue = device.new_command_queue(4).expect("a queue");
    // Refused as they begin: a plane past level 1's two, and a pass of two layers.
    let refused = queue.command_buffer().expect("a command buffer");
    let mut past = attachment(&passes[1]);
    past.color_attachments[0].depth_plane = 2;
    match refused.render_command_encoder(&past) {
        Err(Error::Refused(Refusal::AttachmentPlace { level: 1, slice: 2 })) => {}
        Err(error) => panic!("{name}: a plane past the level's depth: {error}"),
        Ok(_) => panic!("{name}: a plane past the level's depth began"),
    }
    let mut layered = attachment(&passes[0]);
    layered.render_target_array_length = 2;
    match refused.render_command_encoder(&layered) {
        Err(Error::Unsupported(Gap::RenderPass {
            kind: RenderPassKind::TextureType,
        })) => {}
        Err(error) => panic!("{name}: a layered pass on a 3D texture: {error}"),
        Ok(_) => panic!("{name}: a layered pass on a 3D texture began"),
    }
    drop(refused);

    let handle = device.core().device.handle();
    let before = rendered(handle).len();
    let command_buffer = queue.command_buffer().expect("a command buffer");
    let ok = |result: Result<(), Error>, what: &str| {
        result.unwrap_or_else(|error| panic!("{name}: {what}: {error}"))
    };
    for p in &passes {
        let what = format!("level {}'s plane {}", p.level, p.plane);
        let mut encoder = (command_buffer.render_command_encoder(&attachment(p)))
            .unwrap_or_else(|error| panic!("{name}: {what}: {error}"));
        if p.drawn {
            ok(encoder.set_render_pipeline_state(drawn), "the state");
            ok(
                encoder.set_vertex_buffer(Some(positions), 0, 0),
                "positions",
            );
            ok(encoder.set_vertex_buffer(Some(colors), 0, 1), "colours");
            ok(
                encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                &what,
            );
        }
        encoder.end_encoding();
    }

    // Every plane of both levels and of the one-deep texture, out; then each texel sampled.
    let levels = [(&deep, 0, &deep), (&deep, 1, &second), (&flat, 0, &flat)];
    let mut buffers = Buffers::sized(device, 1 << 20);
    let unit = kind.unit() as u64;
    let outs: Vec<Buffer> = (levels.iter())
        .map(|&(texture, level, _)| {
            let [w, h, d] = extent(texture, level);
            buffers.of(device, &vec![0xee; (unit * u64::from(w * h * d)) as usize])
        })
        .collect();
    let mut blit =
        (command_buffer.blit_command_encoder()).unwrap_or_else(|error| panic!("{name}: {error}"));
    for (&(texture, level, _), out) in levels.iter().zip(&outs) {
        let size = extent(texture, level);
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: unit * u64::from(size[0]),
            bytes_per_image: unit * u64::from(size[0] * size[1]),
        };
        let from = TexturePlace {
            slice: 0,
            level,
            origin: [0; 3],
        };
        ok(
            blit.copy_from_texture_to_buffer(texture, from, size, out, layout),
            "a level out",
        );
    }
    blit.end_encoding();
    let lanes_of = |&(texture, level, _): &(&Texture, u32, &Texture)| {
        let [w, h, d] = extent(texture, level);
        w * h * d
    };
    let samples: Vec<(Buffer, Buffer)> = (levels.iter())
        .map(|level| {
            let [w, h, d] = extent(level.0, level.1);
            let centre = |c: u32, n: u32| (c as f32 + 0.5) / n as f32;
            let coords: Vec<f32> = (0..d)
                .flat_map(|z| (0..h).flat_map(move |y| (0..w).map(move |x| [x, y, z])))
                .flat_map(|[x, y, z]| [centre(x, w), centre(y, h), centre(z, d), 0.0])
                .collect();
            let lanes = lanes_of(level) as usize;
            (
                buffers.of(device, &floats(&coords)),
                buffers.of(device, &vec![0xee; 32 * lanes]),
            )
        })
        .collect();
    let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    ok(encoder.set_compute_pipeline_state(kernel), "the kernel");
    ok(encoder.set_sampler_state(Some(*sampler), 0), "the sampler");
    for (level, (coords, out)) in levels.iter().zip(&samples) {
        ok(encoder.set_texture(Some(level.2), 0), "the sampled texture");
        ok(encoder.set_buffer(Some(coords), 0, 0), "the lanes");
        ok(encoder.set_buffer(Some(out), 0, 1), "the answers");
        ok(
            encoder.dispatch_threads([lanes_of(level), 1, 1], [32, 1, 1]),
            "the dispatch",
        );
    }
    encoder.end_encoding();
    command_buffer.commit().expect("a commit");
    finished(name, &command_buffer);
    assert_eq!(command_buffer.error(), None, "{name}");

    for (((&(texture, level, _), out), (_, sampled)), what) in levels
        .iter()
        .zip(&outs)
        .zip(&samples)
        .zip(["level 0", "level 1", "the one-deep texture"])
    {
        let want = want(kind, texture, level, &passes);
        let bytes: Vec<u8> = want.iter().flat_map(|&t| kind.bytes(t)).collect();
        exact(name, &format!("{what}, copied out"), &read(out), &bytes);
        // Each lane's answer through the nearest sampler state, the second of its two, as the
        // format's bytes.
        let answers = floats_of(&read(sampled));
        let got: Vec<Vec<u8>> = (0..want.len())
            .map(|lane| {
                let at = 8 * lane + 4;
                let texel = [0, 1, 2, 3].map(|c| answers[at + c]);
                kind.bytes(texel.map(|c| (c * kind.scale()).round() / kind.scale()))
            })
            .collect();
        let samples_want: Vec<Vec<u8>> = want.iter().map(|&t| kind.bytes(t)).collect();
        exact(name, &format!("{what}, sampled"), &got, &samples_want);
        // The raw answers, bit for bit, where the format's values are a half's.
        if let Kind::Half = kind {
            let raw: Vec<[u32; 4]> = (0..want.len())
                .map(|lane| [0, 1, 2, 3].map(|c| answers[8 * lane + 4 + c].to_bits()))
                .collect();
            let raw_want: Vec<[u32; 4]> = want.iter().map(|t| t.map(f32::to_bits)).collect();
            exact(name, &format!("{what}, sampled raw"), &raw, &raw_want);
        }
    }

    // What ran: each pass's colour attachment, as recorded.
    let attachments: Vec<_> = rendered(handle).into_iter().skip(before).collect();
    assert_eq!(
        attachments.len(),
        passes.len(),
        "{name}: one rendering a pass"
    );
    let mut shown = Vec::new();
    for (p, a) in passes.iter().zip(&attachments) {
        assert_eq!(
            a.image.as_raw(),
            p.texture.image().as_raw(),
            "{name}: the attachment's image"
        );
        assert_eq!(
            a.image_type.as_raw(),
            vk::ImageType::TYPE_3D.as_raw(),
            "{name}: the image type"
        );
        assert!(
            a.flags
                .contains(vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE),
            "{name}: the image's flags {:#x}",
            a.flags.as_raw()
        );
        let r = a.range;
        assert_eq!(
            (
                a.view_type.as_raw(),
                r.base_mip_level,
                r.level_count,
                r.base_array_layer,
                r.layer_count
            ),
            (vk::ImageViewType::TYPE_2D.as_raw(), p.level, 1, p.plane, 1),
            "{name}: level {}'s plane {}: the view",
            p.level,
            p.plane
        );
        shown.push(format!(
            "level {} plane {} ({}): a 2D view (VkImageViewType {}), base level {}, base layer \
             {}, {} layer, of a 3D image (VkImageType {}) made with flags {:#x} \
             (2D_ARRAY_COMPATIBLE among them)",
            p.level,
            p.plane,
            match (p.clear, p.drawn) {
                (Some(_), true) => "clear and draw",
                (Some(_), false) => "clear",
                (None, _) => "load and draw",
            },
            a.view_type.as_raw(),
            r.base_mip_level,
            r.base_array_layer,
            r.layer_count,
            a.image_type.as_raw(),
            a.flags.as_raw()
        ));
    }
    println!(
        "{name}: {SIDE} x {SIDE} x {DEPTH} of {LEVELS} levels and {SIDE} x {SIDE} x 1, {USAGE:?}, \
         Managed: {}; every plane of both levels and of the one-deep texture exact, copied out \
         and sampled at each texel's centre through a nearest sampler state",
        shown.join("; ")
    );
}

#[test]
fn every_device_renders_into_one_depth_plane_of_a_3d_texture_and_keeps_the_others() {
    let Some(examples) = examples() else {
        return;
    };
    let _recording = Recording::start();
    each_device("depth_planes", |name, _, device| {
        let at = IndexClass::Sampler.range().start;
        let _samplers = (device.grant_indices(IndexClass::Sampler, at, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let Some(kernel) = sampling(name, &device, at) else {
            return;
        };
        let first = IndexClass::Texture.range().start;
        let _textures = (device.grant_indices(IndexClass::Texture, first, 2 * TEXTURES))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let sampler = (device.new_sampler_state(&SamplerDescriptor::new(), at + 1))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut buffers = Buffers::new(&device);
        let vertices = vertices(&device, &mut buffers);
        let shared = Shared {
            device: &device,
            sampler: &sampler,
            kernel: &kernel,
            vertices: &vertices,
        };
        let formats = [
            (PixelFormat::RGBA8Unorm, Kind::Unorm8),
            (PixelFormat::RGBA16Float, Kind::Half),
        ];
        for (n, (pf, kind)) in (0..).zip(formats) {
            let drawn = state(
                name,
                &device,
                &descriptor(&examples, "colored_fragment", pf),
            );
            cell(name, &shared, (&drawn, first + n * TEXTURES), pf, kind);
        }
    });
}
