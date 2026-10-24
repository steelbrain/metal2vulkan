//! Render-pipeline link: vertex fetch projection, `VertexInputState`, and `GraphicsState`.

use super::*;
use crate::device::requirements::VkFeature;
use crate::error::{Attachment, Capability, Location};
use crate::metal::{PixelFormat, VertexFormat, VertexStepFunction};
use crate::pipeline::descriptor::{VertexAttributeDescriptor, VertexBufferLayoutDescriptor};
use crate::pipeline::lower_inputs::VertexFetch;
use crate::pipeline::stage_io::PulledInput;
use crate::pipeline::sym::SampledType;
use crate::pipeline::vertex::VertexStride;

/// [`crate::pipeline::link::link`] of `stages` whose fragment function (the second stage) was
/// compiled for the pipeline: under the color writes `inputs` give it, as
/// `Device::compile_render_pipeline` compiles it. `color_writes_are_the_pipelines` links stages as
/// given.
fn link(
    stages: &[&FunctionArtifact],
    inputs: &LinkInputs,
    p: &DeviceProfile,
) -> Result<Linked, Error> {
    let fragment = stages.get(1).map(|f| compiled_for(f, inputs, p));
    let mut all = stages.to_vec();
    if let (Some(slot), Some(f)) = (all.get_mut(1), fragment.as_ref()) {
        *slot = f;
    }
    crate::pipeline::link::link(&all, inputs, p)
}

/// `fragment` with the color writes a render pipeline of `inputs` on `p` compiles it under, an
/// attachment whose rules it selects (`ChannelWrite::Selected`) recorded as a four-lane output's
/// (`StageIo::selected_colors`), which rounds an 8-bit unorm or a half-float channel where
/// `fragment` lists the attachment as rounded and nothing otherwise.
fn compiled_for(
    fragment: &FunctionArtifact,
    inputs: &LinkInputs,
    p: &DeviceProfile,
) -> FunctionArtifact {
    use crate::pipeline::color_writes::{HALF_CODE, UNORM8_CODE};
    use crate::pipeline::lower_inputs::ChannelWrite;
    let mut out = fragment.clone();
    if let (LinkInputs::Render(r), Stage::Fragment) = (inputs, fragment.interface.stage) {
        let io = &mut out.interface.io;
        io.color_writes = crate::pipeline::color_writes::wanted(
            fragment,
            &r.color_attachments,
            (r.alpha_to_coverage, r.texture_write_rounding_mode),
            p,
        );
        io.selected_colors = io
            .color_writes
            .iter()
            .filter(|(_, rules)| rules.contains(&Some(ChannelWrite::Selected)))
            .map(|(&a, _)| {
                let rounds = fragment.interface.io.rounded_colors.contains(&a);
                let write = crate::pipeline::stage_io::SelectedWrite {
                    lanes: 4,
                    rounds: match rounds {
                        true => 1 << UNORM8_CODE | 1 << HALF_CODE,
                        false => 0,
                    },
                    nans: 0,
                };
                (a, write)
            })
            .collect();
    }
    out
}

/// A blend of one and zero, added, of an attachment the fragment function rounds takes no Vulkan
/// blend, since the function writes what Metal's blend stores, nor the format's blend support
/// (lavapipe blends no snorm format); another blend keeps the device's, and so does a (one, zero)
/// blend of an attachment the function does not round.
#[test]
fn a_rounded_one_zero_blend_takes_no_vulkan_blend() {
    use crate::device::requirements::FormatUsage;
    use crate::metal::BlendFactor;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let f4 = io_ty(IoNumeric::Float, 4);
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fa = io_stage(Stage::Fragment, stage_io(vec![], vec![color(0, 0, f4)]));
    let mut rounding = fa.clone();
    rounding.interface.io.rounded_colors = vec![0];
    let mut replace = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    replace.blending_enabled = true;
    let mut over = replace;
    over.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let p = profile("lavapipe");
    let blend_of = |fragment: &FunctionArtifact, a: ColorAttachmentDescriptor| {
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(a);
        let inputs = LinkInputs::Render(Box::new(r));
        let fragment = compiled_for(fragment, &inputs, &p);
        let g = crate::pipeline::link::link(&[&va, &fragment], &inputs, &p)
            .expect("links")
            .graphics
            .expect("graphics");
        g.color_attachments[0].expect("attachment 0").blend
    };
    assert_eq!(blend_of(&rounding, replace), None);
    assert!(blend_of(&rounding, over).is_some());
    assert!(blend_of(&fa, replace).is_some());
    let snorm = vk::Format::R8G8B8A8_SNORM;
    assert!(!p.supports_format(snorm, FormatUsage::ColorAttachmentBlend));
    let mut replace_snorm = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Snorm);
    replace_snorm.blending_enabled = true;
    assert_eq!(blend_of(&rounding, replace_snorm), None);
    let mut r = RenderLinkInputs::new();
    replace_snorm.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    r.color_attachments[0] = Some(replace_snorm);
    let inputs = LinkInputs::Render(Box::new(r));
    let over_snorm = compiled_for(&rounding, &inputs, &p);
    assert!(matches!(
        crate::pipeline::link::link(&[&va, &over_snorm], &inputs, &p),
        Err(Error::Unsupported(_))
    ));
}

/// A blend of one and zero, added, of a 16-bit float attachment the fragment function rounds, from
/// a module without fast math (`StageIo::half_blend_colors` for a `half` output,
/// `StageIo::float_blend_colors` for a `float` one), takes the Vulkan blend: Metal keeps the
/// destination's product by zero there, which no write of the output stores, and
/// the pipeline records `HalfOutputPrecision`. Of a module that lists the attachment in neither
/// set, or into `RGBA8Unorm`, it takes none and records nothing.
#[test]
fn a_safe_one_zero_blend_of_a_sixteen_bit_float_attachment_blends() {
    use crate::metal::BlendFactor;
    use crate::pipeline::color_writes::pipeline_writes;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let f4 = io_ty(IoNumeric::Float, 4);
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let mut rounding = io_stage(Stage::Fragment, stage_io(vec![], vec![color(0, 0, f4)]));
    rounding.interface.io.rounded_colors = vec![0];
    let mut safe_float = rounding.clone();
    safe_float.interface.io.float_blend_colors = vec![0];
    let mut safe_half = rounding.clone();
    safe_half.interface.io.half_blend_colors = vec![0];
    let p = profile("lavapipe");
    let linked = |fragment: &FunctionArtifact, pf: PixelFormat, factor: BlendFactor| {
        let mut a = ColorAttachmentDescriptor::new(pf);
        a.blending_enabled = true;
        a.destination_rgb_blend_factor = factor;
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(a);
        let inputs = LinkInputs::Render(Box::new(r));
        let fragment = compiled_for(fragment, &inputs, &p);
        let g = crate::pipeline::link::link(&[&va, &fragment], &inputs, &p)
            .expect("links")
            .graphics
            .expect("graphics");
        let (notes, _) = pipeline_writes(
            &fragment,
            &g.color_write_rules,
            &g.color_attachments,
            g.alpha_to_coverage,
        )
        .expect("the rule words");
        let noted = notes.contains(&crate::pipeline::PrecisionNote::HalfOutputPrecision);
        (
            g.color_attachments[0]
                .expect("attachment 0")
                .blend
                .is_some(),
            noted,
        )
    };
    let (half, unorm) = (PixelFormat::RGBA16Float, PixelFormat::RGBA8Unorm);
    assert_eq!(linked(&rounding, half, BlendFactor::Zero), (false, false));
    assert_eq!(linked(&safe_float, half, BlendFactor::Zero), (true, true));
    assert_eq!(linked(&safe_half, half, BlendFactor::Zero), (true, true));
    assert_eq!(
        linked(&safe_float, unorm, BlendFactor::Zero),
        (false, false)
    );
    // Another blend is the device's already, the float output's noted by its conversion.
    let over = BlendFactor::OneMinusSourceAlpha;
    assert_eq!(linked(&safe_float, half, over), (true, false));
}

/// A fragment function links only under the color writes its pipeline gives the attachments it
/// writes. On lavapipe and RADV a `float4` output to an `RGBA8Unorm` attachment rounds by the unorm
/// rule, blended with (one, zero) too; under another blend, and for alpha-to-coverage's alpha, the
/// channel is the device's; an `RGBA32Float` attachment takes no entry. Where a channel so rounds,
/// the attachment selects its rules at the draw (`ChannelWrite::Selected`), by the rule word link
/// gives it: `0x2222` for the four unorm channels, `0x1222` with alpha-to-coverage's alpha,
/// `0x3333` for an `RGBA16Float` one's half-float channels, so a fragment compiled for one such
/// attachment links into a pipeline of another; one all of whose channels are the device's selects
/// nothing. A fragment compiled under the channels' own rules (`color_writes::exact_writes`) links
/// too, into a pipeline of those rules. A fragment compiled without the map, or for a format whose
/// rules it does not select (`RGBA16Unorm`), is `NeedsInput(ColorWrites)`. On MoltenVK, which
/// converts as Metal does, the map names only the half-float channels, blended or not, and selects
/// nothing.
#[test]
fn color_writes_are_the_pipelines() {
    use crate::error::RuntimeInput;
    use crate::metal::BlendFactor;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::lower_inputs::ChannelWrite as C;
    let f4 = io_ty(IoNumeric::Float, 4);
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fa = io_stage(Stage::Fragment, stage_io(vec![], vec![color(0, 0, f4)]));
    let render = |c: ColorAttachmentDescriptor, coverage: bool| {
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(c);
        r.alpha_to_coverage = coverage;
        LinkInputs::Render(Box::new(r))
    };
    let unorm8 = Some(C::Unorm { bits: 8 });
    let plain = render(
        ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm),
        false,
    );
    let mut replace = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    replace.blending_enabled = true;
    let mut blend = replace;
    blend.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let wide = render(
        ColorAttachmentDescriptor::new(PixelFormat::RGBA16Unorm),
        false,
    );
    let float = render(
        ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float),
        false,
    );
    let half_float = render(
        ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float),
        false,
    );
    for stem in ["lavapipe", "amd-rdna3-radv"] {
        let p = profile(stem);
        let writes = |inputs: &LinkInputs, exact: bool| match inputs {
            LinkInputs::Render(r) => {
                let colors = &r.color_attachments;
                let rounding = (r.alpha_to_coverage, r.texture_write_rounding_mode);
                match exact {
                    true => crate::pipeline::color_writes::exact_writes([0], colors, rounding, &p),
                    false => crate::pipeline::color_writes::wanted(&fa, colors, rounding, &p),
                }
            }
            _ => unreachable!(),
        };
        let device = Some(C::Device);
        let half = Some(crate::pipeline::color_writes::HALF);
        let selected = Some([Some(C::Selected); 4]);
        let coverage = render(
            ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm),
            true,
        );
        for (inputs, exact, want, word) in [
            (&plain, Some([unorm8; 4]), selected, Some(0x2222)),
            (
                &render(replace, false),
                Some([unorm8; 4]),
                selected,
                Some(0x2222),
            ),
            (&render(blend, false), Some([device; 4]), None, None),
            (
                &coverage,
                Some([unorm8, unorm8, unorm8, device]),
                selected,
                Some(0x1222),
            ),
            (&half_float, Some([half; 4]), selected, Some(0x3333)),
            (&float, None, None, None),
        ] {
            assert_eq!(writes(inputs, true).get(&0).copied(), exact, "{stem}");
            let want = want.or(exact);
            assert_eq!(writes(inputs, false).get(&0).copied(), want, "{stem}");
            let l = link(&[&va, &fa], inputs, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            let words = &l.graphics.as_ref().expect("graphics").color_write_rules;
            assert_eq!(
                words.clone(),
                Vec::from_iter(word.map(|w| (0, w))),
                "{stem}"
            );
        }
        // Compiled for one attachment whose rules it selects, linked into another's pipeline.
        let for_plain = compiled_for(&fa, &plain, &p);
        for inputs in [&half_float, &coverage] {
            crate::pipeline::link::link(&[&va, &for_plain], inputs, &p)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
        }
        // Compiled under the channels' own rules, it links into the pipeline of those rules alone.
        let mut own = fa.clone();
        own.interface.io.color_writes = writes(&plain, true);
        crate::pipeline::link::link(&[&va, &own], &plain, &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(
            crate::pipeline::link::link(&[&va, &own], &half_float, &p).map(|_| ()),
            Err(Error::NeedsInput(RuntimeInput::ColorWrites)),
            "{stem}"
        );
        let needs = Err(Error::NeedsInput(RuntimeInput::ColorWrites));
        // Compiled without the map, and for another format.
        assert_eq!(
            crate::pipeline::link::link(&[&va, &fa], &plain, &p).map(|_| ()),
            needs,
            "{stem}"
        );
        let for_wide = compiled_for(&fa, &wide, &p);
        assert_eq!(
            crate::pipeline::link::link(&[&va, &for_wide], &plain, &p).map(|_| ()),
            needs,
            "{stem}"
        );
        assert_eq!(
            crate::pipeline::link::link(&[&va, &for_wide], &float, &p).map(|_| ()),
            needs,
            "{stem}"
        );
    }
    let p = profile("moltenvk-m3");
    let default = crate::metal::TextureWriteRoundingMode::TowardZero;
    assert!(
        crate::pipeline::color_writes::wanted(&fa, &[None; 8], (false, default), &p).is_empty()
    );
    crate::pipeline::link::link(&[&va, &fa], &plain, &p).expect("moltenvk-m3");
    // There the map keeps the half-float channels alone, whatever the blend.
    let half = Some(crate::pipeline::color_writes::HALF);
    let mut blended_half = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    blended_half.blending_enabled = true;
    blended_half.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let half_target = render(
        ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float),
        false,
    );
    for (inputs, want) in [
        (&plain, None),
        (&float, None),
        (
            &render(
                ColorAttachmentDescriptor::new(PixelFormat::RG11B10Float),
                false,
            ),
            None,
        ),
        (&half_target, Some([half; 4])),
        (&render(blended_half, true), Some([half; 4])),
    ] {
        let LinkInputs::Render(r) = inputs else {
            unreachable!()
        };
        let got = crate::pipeline::color_writes::wanted(
            &fa,
            &r.color_attachments,
            (r.alpha_to_coverage, r.texture_write_rounding_mode),
            &p,
        );
        assert_eq!(got.get(&0).copied(), want, "{:?}", r.color_attachments[0]);
        link(&[&va, &fa], inputs, &p).expect("moltenvk-m3");
    }
    assert_eq!(
        crate::pipeline::link::link(&[&va, &fa], &half_target, &p).map(|_| ()),
        Err(Error::NeedsInput(RuntimeInput::ColorWrites))
    );
}

/// Under `TextureWriteRoundingMode::NearestEven` a half-float attachment's channels round to
/// nearest even in the function on every profile, MoltenVK's included, where they store the output
/// as written (no blend, or factors one and zero), alpha-to-coverage's alpha included where the
/// attachment stores it, which a function rounding it announces as `CoverageReadsRoundedAlpha` in
/// a pipeline covering by alpha (an alpha the write mask does not store keeps the default mode's
/// rule, read by the coverage as computed); after another blend, one and one included, the device
/// rounds the blend's result (`HALF_BLENDED`, announced as `BlendedWriteRounding`). Another
/// format's writes are the default mode's. A fragment compiled under the default mode does not
/// link into such a pipeline (`NeedsInput(ColorWrites)`), and one compiled under the pipeline's
/// mode does.
#[test]
fn half_colors_round_to_nearest_even_under_mode_two() {
    use crate::error::RuntimeInput;
    use crate::metal::{BlendFactor, TextureWriteRoundingMode as Mode};
    use crate::pipeline::color_writes::{pipeline_writes, HALF_BLENDED, HALF_NEAREST};
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::PrecisionNote;
    let f4 = io_ty(IoNumeric::Float, 4);
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fa = io_stage(Stage::Fragment, stage_io(vec![], vec![color(0, 0, f4)]));
    // A `float` output, which rounds where its channels take a rule.
    let mut rounding = fa.clone();
    rounding.interface.io.rounded_colors = vec![0];
    let render = |c: ColorAttachmentDescriptor, coverage: bool, mode: Mode| {
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(c);
        r.alpha_to_coverage = coverage;
        r.texture_write_rounding_mode = mode;
        LinkInputs::Render(Box::new(r))
    };
    let half = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    let mut replace = half;
    replace.blending_enabled = true;
    let mut blend = replace;
    blend.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let mut add = replace;
    add.destination_rgb_blend_factor = BlendFactor::One;
    add.destination_alpha_blend_factor = BlendFactor::One;
    // Stores no alpha: alpha to coverage reads the alpha as computed, as Metal's does.
    let mut rgb = half;
    rgb.write_mask = crate::metal::ColorWriteMask::RED
        .union(crate::metal::ColorWriteMask::GREEN)
        .union(crate::metal::ColorWriteMask::BLUE);
    let nearest = Some(HALF_NEAREST);
    let blended = Some(HALF_BLENDED);
    for stem in ["moltenvk-m3", "lavapipe", "amd-780m-radv"] {
        let p = profile(stem);
        let writes = |inputs: &LinkInputs| match inputs {
            LinkInputs::Render(r) => crate::pipeline::color_writes::wanted(
                &fa,
                &r.color_attachments,
                (r.alpha_to_coverage, r.texture_write_rounding_mode),
                &p,
            )
            .get(&0)
            .copied(),
            _ => unreachable!(),
        };
        // The default mode's own rule of the alpha the coverage reads: the output as written.
        let unstored = crate::pipeline::color_writes::exact_writes(
            [0],
            &[Some(rgb)],
            (true, Mode::TowardZero),
            &p,
        )[&0][3];
        assert_ne!(unstored, nearest, "{stem}");
        for (c, coverage, want) in [
            (half, false, [nearest; 4]),
            (replace, false, [nearest; 4]),
            (blend, false, [blended; 4]),
            (add, false, [blended; 4]),
            (blend, true, [blended; 4]),
            (half, true, [nearest; 4]),
            (replace, true, [nearest; 4]),
            (rgb, false, [nearest; 4]),
            (rgb, true, [nearest, nearest, nearest, unstored]),
        ] {
            let inputs = render(c, coverage, Mode::NearestEven);
            assert_eq!(writes(&inputs), Some(want), "{stem} {c:?} {coverage}");
            let fragment = compiled_for(&rounding, &inputs, &p);
            let g = crate::pipeline::link::link(&[&va, &fragment], &inputs, &p)
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .graphics
                .expect("graphics");
            assert_eq!(g.alpha_to_coverage, coverage, "{stem} {c:?}");
            let (notes, _) = pipeline_writes(
                &fragment,
                &g.color_write_rules,
                &g.color_attachments,
                g.alpha_to_coverage,
            )
            .expect("the rule words");
            assert_eq!(
                notes.contains(&PrecisionNote::CoverageReadsRoundedAlpha),
                coverage && want[3] == nearest,
                "{stem} {c:?} {coverage}: {notes:?}"
            );
        }
        for format in [PixelFormat::RGBA8Unorm, PixelFormat::RG11B10Float] {
            let c = ColorAttachmentDescriptor::new(format);
            let default = writes(&render(c, false, Mode::TowardZero));
            assert_eq!(
                writes(&render(c, false, Mode::NearestEven)),
                default,
                "{stem} {format:?}"
            );
        }
        let inputs = render(half, false, Mode::NearestEven);
        let default = compiled_for(&fa, &render(half, false, Mode::TowardZero), &p);
        assert_eq!(
            crate::pipeline::link::link(&[&va, &default], &inputs, &p).map(|_| ()),
            Err(Error::NeedsInput(RuntimeInput::ColorWrites)),
            "{stem}"
        );
    }
}

/// Layout 0 per vertex (stride 32), layout 3 per instance with a per-bind stride; attributes
/// 0 and 1 in layout 0, attribute 4 in layout 3.
fn descriptor() -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    d.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(32)));
    let mut inst = VertexBufferLayoutDescriptor::new(None);
    inst.step_function = VertexStepFunction::PerInstance;
    d.layouts.insert(3, inst);
    d.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 0),
    );
    d.attributes.insert(
        1,
        VertexAttributeDescriptor::new(VertexFormat::Float2, 16, 0),
    );
    d.attributes
        .insert(4, VertexAttributeDescriptor::new(VertexFormat::Float, 0, 3));
    d
}

/// Every attribute of `attrs` read as `float`.
fn floats(attrs: &[u32]) -> BTreeMap<u32, IoNumeric> {
    attrs.iter().map(|a| (*a, IoNumeric::Float)).collect()
}

/// A vertex stage reading `attrs` as `float4`, compiled under their projection on `p`: an
/// `Attribute` input per fixed-function attribute, a pulled entry per pulled one.
fn vertex_stage(p: &DeviceProfile, d: &VertexDescriptor, attrs: &[u32]) -> StageInterface {
    let proj = vertex::vertex_fetch_projection(Some(d), &floats(attrs), p).expect("projection");
    let mut vs = iface(Stage::Vertex, vec![], vec![]);
    let f4 = io_ty(IoNumeric::Float, 4);
    for (a, f) in &proj.attributes {
        match f {
            VertexFetch::FixedFunction => vs.io.inputs.push(attr(*a, f4)),
            _ => vs.io.pulled.push(PulledInput::new(*a, f4, *f)),
        }
    }
    vs
}

fn render_with(d: VertexDescriptor) -> LinkInputs {
    let mut r = RenderLinkInputs::new();
    r.vertex_descriptor = Some(d);
    LinkInputs::Render(Box::new(r))
}

/// Fixed-function attributes sit at dense locations in attribute order, bindings are dense in
/// buffer order with their rate and stride, and every committed profile agrees with the
/// projection.
#[test]
fn vertex_input_follows_the_projection_on_every_committed_profile() {
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    assert!(!profiles.is_empty());
    for (stem, p) in &profiles {
        let d = descriptor();
        let vs = vertex_stage(p, &d, &[0, 1, 4]);
        let va = artifact(vs, None);
        let l = link(&[&va], &render_with(d), p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let vi = &l.vertex_input;
        // Every attribute is fixed function exactly where the profile's format table lists its
        // format for vertex buffers, and pulled elsewhere.
        let fixed =
            |f| p.supports_format(f, crate::device::requirements::FormatUsage::VertexBuffer);
        let want = [
            (0, 0, vk::Format::R32G32B32A32_SFLOAT, 0),
            (1, 0, vk::Format::R32G32_SFLOAT, 16),
            (4, 3, vk::Format::R32_SFLOAT, 0),
        ];
        let mut binds_want = Vec::new();
        let mut locs_want = Vec::new();
        let mut pulled_want = Vec::new();
        for (attr, buffer, format, offset) in want {
            if !fixed(format) {
                pulled_want.push(attr);
                continue;
            }
            if !binds_want.contains(&buffer) {
                binds_want.push(buffer);
            }
            let binding = binds_want.iter().position(|b| *b == buffer).unwrap_or(0) as u32;
            locs_want.push((locs_want.len() as u32, binding, format, offset));
        }
        let locs: Vec<(u32, u32, vk::Format, u32)> = vi
            .attributes
            .iter()
            .map(|a| (a.location, a.binding, a.format, a.offset))
            .collect();
        assert_eq!(locs, locs_want, "{stem}");
        let binds: Vec<(u32, u32, vk::VertexInputRate, VertexStride)> = vi
            .bindings
            .iter()
            .map(|b| (b.binding, b.buffer_index, b.input_rate, b.stride))
            .collect();
        let all_binds = [
            (0, vk::VertexInputRate::VERTEX, VertexStride::Static(32)),
            (3, vk::VertexInputRate::INSTANCE, VertexStride::PerBind),
        ];
        let binds_want: Vec<(u32, u32, vk::VertexInputRate, VertexStride)> = binds_want
            .iter()
            .enumerate()
            .map(|(n, buffer)| {
                let (_, rate, stride) = all_binds
                    .iter()
                    .find(|b| b.0 == *buffer)
                    .copied()
                    .expect("layout");
                (n as u32, *buffer, rate, stride)
            })
            .collect();
        assert_eq!(binds, binds_want, "{stem}");
        let pulled: Vec<u32> = vi.pulled.iter().map(|a| a.attribute).collect();
        assert_eq!(pulled, pulled_want, "{stem}");
        let g = l.graphics.as_ref().expect("graphics");
        assert_eq!(
            g.dynamic_states
                .contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE),
            p.fields().dynamic_vertex_stride && fixed(vk::Format::R32_SFLOAT),
            "{stem}"
        );
    }
}

/// Constant step functions, attributes beyond the device's attribute or binding limits, and
/// attributes whose format class differs from the declared type are pulled; a stage compiled
/// under another projection is refused.
#[test]
fn limits_pull_attributes_and_a_foreign_projection_is_refused() {
    let mut p = profile("moltenvk-m3");
    let mut d = descriptor();
    d.layouts.get_mut(&3).expect("layout 3").step_function = VertexStepFunction::Constant;
    let used = floats(&[0, 1, 4]);
    let proj = vertex::vertex_fetch_projection(Some(&d), &used, &p).expect("projection");
    assert_eq!(
        proj.attributes[&4],
        VertexFetch::Pulled {
            format: VertexFormat::Float,
            buffer_index: 3,
            offset: 0,
            step_function: VertexStepFunction::Constant,
            step_rate: 0,
        }
    );
    let d = descriptor();
    p.fields.limits.max_vertex_input_attributes = 2;
    let proj = vertex::vertex_fetch_projection(Some(&d), &used, &p).expect("projection");
    assert!(matches!(proj.attributes[&4], VertexFetch::Pulled { .. }));
    assert_eq!(proj.attributes[&1], VertexFetch::FixedFunction);
    let mut p = profile("moltenvk-m3");
    p.fields.limits.max_vertex_input_bindings = 1;
    let proj = vertex::vertex_fetch_projection(Some(&d), &used, &p).expect("projection");
    assert!(matches!(proj.attributes[&4], VertexFetch::Pulled { .. }));
    assert_eq!(proj.attributes[&0], VertexFetch::FixedFunction);
    // Linking the pulled attribute lists it as pulled with its layout's step.
    let vs = vertex_stage(&p, &d, &[0, 4]);
    let va = artifact(vs, None);
    let l = link(&[&va], &render_with(d.clone()), &p).expect("link");
    assert_eq!(l.vertex_input.pulled.len(), 1);
    assert_eq!(l.vertex_input.pulled[0].attribute, 4);
    assert_eq!(l.vertex_input.pulled[0].buffer_index, 3);
    assert_eq!(l.vertex_input.bindings.len(), 1);
    // The same stage against a device where attribute 4 is fixed function: refused.
    let full = profile("moltenvk-m3");
    assert_eq!(
        link(&[&va], &render_with(d.clone()), &full).unwrap_err(),
        Error::InvalidDescriptor(DescriptorError::VertexAttribute { attribute: 4 })
    );
    // An attribute or layout the descriptor lacks.
    let mut vs = iface(Stage::Vertex, vec![], vec![]);
    vs.io.inputs.push(attr(9, io_ty(IoNumeric::Float, 4)));
    let va = artifact(vs, None);
    assert_eq!(
        link(&[&va], &render_with(d.clone()), &full).unwrap_err(),
        Error::InvalidDescriptor(DescriptorError::VertexAttribute { attribute: 9 })
    );
    let mut no_layout = d;
    no_layout.layouts.remove(&0);
    let vs = vertex_stage(&full, &descriptor(), &[0]);
    let va = artifact(vs, None);
    assert_eq!(
        link(&[&va], &render_with(no_layout), &full).unwrap_err(),
        Error::InvalidDescriptor(DescriptorError::VertexLayout { layout: 0 })
    );
}

/// An attribute is fixed function when fixed-function input delivers the class the shader
/// declares (`float` for float, normalized, and packed formats, `uint` for unsigned-integer
/// ones), and so is an unnormalized 8- or 16-bit integer format read into a `float` attribute,
/// in its scaled form; signed-integer formats into `uint` are pulled and sign-extended. A 32-bit
/// integer format into `float` is an invalid descriptor, as on Metal, and a float-class format into
/// `uint` is refused by the one projection.
#[test]
fn the_format_class_must_match_the_declared_type() {
    use crate::error::{Feature, LowerFamily, Needs};
    #[derive(Debug)]
    enum Want {
        Fixed,
        Pulled,
        Refused,
        Invalid,
    }
    let p = profile("moltenvk-m3");
    let mut d = VertexDescriptor::new();
    d.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(64)));
    let cases = [
        (VertexFormat::UChar4, IoNumeric::Uint, Want::Fixed),
        (VertexFormat::UShort2, IoNumeric::Uint, Want::Fixed),
        (
            VertexFormat::UChar4Normalized,
            IoNumeric::Float,
            Want::Fixed,
        ),
        (VertexFormat::Half2, IoNumeric::Float, Want::Fixed),
        (VertexFormat::Float3, IoNumeric::Float, Want::Fixed),
        (VertexFormat::Char2, IoNumeric::Float, Want::Fixed),
        (VertexFormat::UShort2, IoNumeric::Float, Want::Fixed),
        (VertexFormat::Short4, IoNumeric::Float, Want::Fixed),
        (VertexFormat::Char4, IoNumeric::Uint, Want::Pulled),
        (VertexFormat::Int, IoNumeric::Uint, Want::Pulled),
        (VertexFormat::UInt, IoNumeric::Float, Want::Invalid),
        (VertexFormat::Int3, IoNumeric::Float, Want::Invalid),
        (VertexFormat::Float2, IoNumeric::Uint, Want::Refused),
        (
            VertexFormat::Short2Normalized,
            IoNumeric::Uint,
            Want::Refused,
        ),
    ];
    for (i, (format, numeric, want)) in cases.into_iter().enumerate() {
        let i = i as u32;
        d.attributes
            .insert(i, VertexAttributeDescriptor::new(format, 0, 0));
        let got = vertex::vertex_fetch_projection(Some(&d), &BTreeMap::from([(i, numeric)]), &p);
        match (want, got) {
            (Want::Fixed, Ok(proj)) => {
                assert_eq!(
                    proj.attributes[&i],
                    VertexFetch::FixedFunction,
                    "{format:?}"
                )
            }
            (Want::Pulled, Ok(proj)) => assert!(
                matches!(proj.attributes[&i], VertexFetch::Pulled { format: f, .. } if f == format),
                "{format:?} as {numeric:?}: {:?}",
                proj.attributes[&i]
            ),
            (Want::Refused, Err(Error::Unsupported(u))) => {
                assert_eq!(u.feature, Feature::Lowering(LowerFamily::StageIo));
                assert_eq!(u.needs, Needs::NotImplemented);
            }
            (
                Want::Invalid,
                Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute { attribute })),
            ) => assert_eq!(attribute, i, "{format:?}"),
            (want, got) => panic!("{format:?} as {numeric:?}: want {want:?}, got {got:?}"),
        }
    }
}

/// Metal allows any byte offset and stride on Apple5 and later; Vulkan's fixed-function
/// attribute address must be a multiple of the component size (the whole size of a packed
/// format). An attribute whose offset or static stride is not is pulled; a per-bind stride is the
/// binder's runtime fact and does not decide the projection.
#[test]
fn unaligned_attributes_are_pulled() {
    let p = profile("moltenvk-m3");
    let cases = [
        // (format, offset, stride, aligned)
        (VertexFormat::Float3, 0, Some(12), true),
        (VertexFormat::Float3, 2, Some(16), false),
        (VertexFormat::Float3, 0, Some(14), false),
        (VertexFormat::Half2, 2, Some(6), true),
        (VertexFormat::Half2, 1, Some(8), false),
        (VertexFormat::Half2, 0, Some(5), false),
        (VertexFormat::UChar4, 3, Some(7), true),
        (VertexFormat::UChar4NormalizedBGRA, 1, Some(5), true),
        (VertexFormat::UInt1010102Normalized, 2, Some(8), false),
        (VertexFormat::FloatRG11B10, 4, Some(8), true),
        (VertexFormat::Float, 2, None, false),
        (VertexFormat::Float, 4, None, true),
    ];
    for (format, offset, stride, aligned) in cases {
        let mut d = VertexDescriptor::new();
        d.layouts
            .insert(0, VertexBufferLayoutDescriptor::new(stride));
        d.attributes
            .insert(0, VertexAttributeDescriptor::new(format, offset, 0));
        let numeric = if crate::lower::stage_io::fetch::encoding(format)
            .expect("encoding")
            .is_float()
        {
            IoNumeric::Float
        } else {
            IoNumeric::Uint
        };
        // Aligned, the format's own fetch decides (some formats are not vertex-buffer formats
        // on the profile); the plain float formats are.
        let base = matches!(
            crate::metal::vertex::attribute_fetch(format, VertexStepFunction::PerVertex, 1, &p),
            Ok(crate::metal::vertex::AttributeFetch::FixedFunction { .. })
        );
        if matches!(
            format,
            VertexFormat::Float3 | VertexFormat::Half2 | VertexFormat::Float
        ) {
            assert!(base, "{format:?}");
        }
        let proj = vertex::vertex_fetch_projection(Some(&d), &BTreeMap::from([(0, numeric)]), &p)
            .expect("projection");
        assert_eq!(
            proj.attributes[&0] == VertexFetch::FixedFunction,
            aligned && base,
            "{format:?} at {offset} stride {stride:?}: {:?}",
            proj.attributes[&0]
        );
    }
}

/// On a portability-subset device a static stride must be a multiple of
/// `minVertexInputBindingStrideAlignment` and at least as large
/// (`VUID-VkVertexInputBindingDescription-stride-04456`); a layout that is not is pulled.
/// MoltenVK's committed capture reports 1, so there only a zero stride is pulled; a device
/// without the rule keeps a zero stride fixed function.
#[test]
fn portability_stride_alignment_pulls_static_strides() {
    let fetch = |p: &DeviceProfile, stride: u32| {
        let mut d = VertexDescriptor::new();
        d.layouts
            .insert(0, VertexBufferLayoutDescriptor::new(Some(stride)));
        d.attributes.insert(
            0,
            VertexAttributeDescriptor::new(VertexFormat::UChar4, 0, 0),
        );
        vertex::vertex_fetch_projection(Some(&d), &BTreeMap::from([(0, IoNumeric::Uint)]), p)
            .expect("projection")
            .attributes[&0]
    };
    let fixed = VertexFetch::FixedFunction;
    let mvk = profile("moltenvk-m3");
    assert_eq!(
        mvk.fields()
            .limits
            .min_vertex_input_binding_stride_alignment,
        Some(1)
    );
    assert_eq!(fetch(&mvk, 4), fixed);
    assert_eq!(fetch(&mvk, 5), fixed);
    assert_ne!(fetch(&mvk, 0), fixed);
    let mut four = mvk.clone();
    four.fields.limits.min_vertex_input_binding_stride_alignment = Some(4);
    assert_eq!(fetch(&four, 8), fixed);
    assert_ne!(fetch(&four, 6), fixed);
    assert_ne!(fetch(&four, 2), fixed);
    let lavapipe = profile("lavapipe");
    assert_eq!(fetch(&lavapipe, 0), fixed);
    assert_eq!(fetch(&lavapipe, 6), fixed);
}

/// On a portability-subset device without `vertexAttributeAccessBeyondStride` an attribute must
/// end within its binding's stride
/// (`VUID-VkVertexInputAttributeDescription-vertexAttributeAccessBeyondStride-04457`): one whose
/// static stride it passes is pulled. MoltenVK's capture has the feature, which keeps it fixed
/// function, as does a device without the subset.
#[test]
fn the_portability_subset_can_pull_attributes_past_their_stride() {
    let fetch = |p: &DeviceProfile, stride: u32| {
        let mut d = VertexDescriptor::new();
        d.layouts
            .insert(0, VertexBufferLayoutDescriptor::new(Some(stride)));
        // Bytes [8, 24) of each element.
        d.attributes.insert(
            0,
            VertexAttributeDescriptor::new(VertexFormat::Float4, 8, 0),
        );
        vertex::vertex_fetch_projection(Some(&d), &floats(&[0]), p)
            .expect("projection")
            .attributes[&0]
    };
    let fixed = VertexFetch::FixedFunction;
    let beyond = VkFeature::VertexAttributeAccessBeyondStride;
    let mvk = profile("moltenvk-m3");
    assert!(mvk.fields().enabled.features.contains(&beyond));
    assert_eq!(fetch(&mvk, 16), fixed);
    let mut withheld = mvk.clone();
    withheld.fields.enabled.features.remove(&beyond);
    assert_ne!(fetch(&withheld, 16), fixed);
    assert_eq!(fetch(&withheld, 24), fixed);
    assert_eq!(fetch(&profile("lavapipe"), 16), fixed);
}

/// A constant-alpha RGB blend factor (Metal's `BlendAlpha`, `OneMinusBlendAlpha`) is refused at
/// link on a portability-subset device without `constantAlphaColorBlendFactors`; as an alpha
/// factor, or on a device without the subset, it links. On an attachment the draw cannot write (an
/// empty mask) the blend writes nothing, so it is dropped there instead.
#[test]
fn the_portability_subset_can_withhold_constant_alpha_color_factors() {
    use crate::metal::BlendFactor;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let f4 = io_ty(IoNumeric::Float, 4);
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fa = io_stage(Stage::Fragment, stage_io(vec![], vec![color(0, 0, f4)]));
    let inputs = |rgb: bool, factor| {
        let mut a = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        a.blending_enabled = true;
        if rgb {
            a.source_rgb_blend_factor = factor;
        } else {
            a.destination_alpha_blend_factor = factor;
        }
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(a);
        LinkInputs::Render(Box::new(r))
    };
    let constant_alpha = VkFeature::ConstantAlphaColorBlendFactors;
    let mvk = profile("moltenvk-m3");
    assert!(mvk.fields().enabled.features.contains(&constant_alpha));
    let mut withheld = mvk.clone();
    withheld.fields.enabled.features.remove(&constant_alpha);
    for factor in [BlendFactor::BlendAlpha, BlendFactor::OneMinusBlendAlpha] {
        for p in [&mvk, &profile("lavapipe")] {
            assert!(
                link(&[&va, &fa], &inputs(true, factor), p).is_ok(),
                "{factor:?}"
            );
        }
        assert_eq!(
            link(&[&va, &fa], &inputs(true, factor), &withheld).unwrap_err(),
            tied(
                Unsupported::needs(
                    Feature::ConstantAlphaBlendFactor,
                    Capability::Feature(constant_alpha)
                ),
                Attachment::Color(0)
            ),
            "{factor:?}"
        );
        assert!(link(&[&va, &fa], &inputs(false, factor), &withheld).is_ok());
    }
    assert!(link(
        &[&va, &fa],
        &inputs(true, BlendFactor::BlendColor),
        &withheld
    )
    .is_ok());
    // Attachment 1, which no output names, blends with `BlendAlpha` under the empty mask link
    // gives it: kept where the device has the feature, dropped where the subset withholds it.
    let unwritten = || {
        let mut a = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        a.blending_enabled = true;
        a.source_rgb_blend_factor = BlendFactor::BlendAlpha;
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        r.color_attachments[1] = Some(a);
        LinkInputs::Render(Box::new(r))
    };
    let blend_of_1 = |p: &DeviceProfile| {
        let g = link(&[&va, &fa], &unwritten(), p)
            .expect("links")
            .graphics
            .expect("graphics");
        let a = g.color_attachments[1].expect("attachment 1");
        assert!(a.write_mask.is_empty());
        a.blend
    };
    assert!(blend_of_1(&mvk).is_some_and(|b| b.constant_alpha_color()));
    assert_eq!(blend_of_1(&withheld), None);
}

/// The descriptor checks read the blend Metal was given, before the constant-alpha drop: a
/// `Source1Color` factor with no index-0 output (written by the second output alone, or by nothing
/// under an empty mask) is refused with `BlendAlpha` beside it whether or not the portability
/// subset withholds `constantAlphaColorBlendFactors`.
#[test]
fn the_constant_alpha_drop_follows_the_descriptor_checks() {
    use crate::metal::{BlendFactor, ColorWriteMask};
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let f4 = io_ty(IoNumeric::Float, 4);
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let inputs = |mask| {
        let mut a = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        a.blending_enabled = true;
        a.source_rgb_blend_factor = BlendFactor::BlendAlpha;
        a.destination_rgb_blend_factor = BlendFactor::Source1Color;
        a.write_mask = mask;
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(a);
        LinkInputs::Render(Box::new(r))
    };
    let constant_alpha = VkFeature::ConstantAlphaColorBlendFactors;
    let mvk = profile("moltenvk-m3");
    assert!(mvk.fields().fragment.dual_src_blend);
    let mut withheld = mvk.clone();
    withheld.fields.enabled.features.remove(&constant_alpha);
    let invalid = Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 0 });
    for (output, mask) in [
        (color(0, 1, f4), ColorWriteMask::ALL),
        (color(0, 0, f4), ColorWriteMask::NONE),
    ] {
        let fa = io_stage(Stage::Fragment, stage_io(vec![], vec![output]));
        for p in [&mvk, &withheld] {
            assert_eq!(
                link(&[&va, &fa], &inputs(mask), p).map(|_| ()),
                Err(invalid.clone()),
                "{mask:?}"
            );
        }
    }
}

/// Under dynamic strides (a fixed-function per-bind layout on a profile with extended dynamic
/// state) every binding's stride is passed per bind, and a nonzero static stride must cover the
/// attributes its binding fetches (`VUID-vkCmdBindVertexBuffers2-pStrides-06209`): an attribute
/// ending past it is pulled, one within it and a zero stride stay fixed function. Without dynamic
/// strides the static stride is pipeline state and nothing is pulled.
#[test]
fn static_strides_under_dynamic_strides_cover_their_attributes() {
    let project = |p: &DeviceProfile, stride: u32| {
        let mut d = VertexDescriptor::new();
        d.layouts
            .insert(0, VertexBufferLayoutDescriptor::new(Some(stride)));
        d.layouts.insert(1, VertexBufferLayoutDescriptor::new(None));
        // Bytes [0, 8) and [8, 12) of layout 0; layout 1's stride is per bind.
        d.attributes.insert(
            0,
            VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
        );
        d.attributes
            .insert(1, VertexAttributeDescriptor::new(VertexFormat::Float, 8, 0));
        d.attributes
            .insert(2, VertexAttributeDescriptor::new(VertexFormat::Float, 0, 1));
        vertex::vertex_fetch_projection(Some(&d), &floats(&[0, 1, 2]), p).expect("projection")
    };
    let fixed = |proj: &crate::pipeline::lower_inputs::VertexFetchProjection| {
        proj.attributes
            .iter()
            .filter(|(_, f)| **f == VertexFetch::FixedFunction)
            .map(|(a, _)| *a)
            .collect::<Vec<u32>>()
    };
    let lavapipe = profile("lavapipe");
    assert!(lavapipe.fields().dynamic_vertex_stride);
    assert_eq!(fixed(&project(&lavapipe, 12)), [0, 1, 2]);
    assert_eq!(fixed(&project(&lavapipe, 0)), [0, 1, 2]);
    assert_eq!(
        fixed(&project(&lavapipe, 8)),
        [0, 2],
        "the float at 8 ends past 8"
    );
    assert_eq!(fixed(&project(&lavapipe, 4)), [2], "both end past 4");
    let mut static_only = lavapipe.clone();
    static_only.fields.dynamic_vertex_stride = false;
    assert_eq!(fixed(&project(&static_only, 4)), [0, 1, 2]);
}

/// Color attachments, depth, samples, input attachments, and dynamic states map through the
/// B3 tables.
#[test]
fn graphics_state_maps_the_descriptor() {
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    // Profiles whose format tables list the attachment formats (the floor profiles carry none).
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna3-radv"] {
        let p = profile(stem);
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        // Blended, adding the destination (one and zero, added, would store the source as no
        // blending does).
        let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float);
        blended.blending_enabled = true;
        blended.destination_rgb_blend_factor = crate::metal::BlendFactor::One;
        r.color_attachments[2] = Some(blended);
        r.depth_format = Some(PixelFormat::Depth32Float);
        r.input_primitive_topology = crate::metal::PrimitiveTopologyClass::Line;
        let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
        let fr = Stage::Fragment;
        let fetch = StageDescriptor::new(
            res(fr, ResourceKind::InputAttachment(2), 0),
            vk::DescriptorType::INPUT_ATTACHMENT,
            1,
        );
        let fa = artifact(iface(fr, vec![fetch], vec![]), None);
        let l = link(&[&va, &fa], &LinkInputs::Render(Box::new(r.clone())), &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let g = l.graphics.expect("graphics");
        assert_eq!(g.color_attachments.len(), 3, "{stem}");
        let c0 = g.color_attachments[0].expect("attachment 0");
        assert_eq!(c0.format, vk::Format::R8G8B8A8_UNORM, "{stem}");
        assert!(c0.blend.is_none());
        assert!(g.color_attachments[1].is_none());
        let c2 = g.color_attachments[2].expect("attachment 2");
        assert_eq!(c2.format, vk::Format::R32G32B32A32_SFLOAT, "{stem}");
        assert!(c2.blend.is_some());
        assert_eq!(g.depth_format, Some(vk::Format::D32_SFLOAT), "{stem}");
        assert_eq!(g.rasterization_samples, vk::SampleCountFlags::COUNT_1);
        assert_eq!(g.topology, vk::PrimitiveTopology::LINE_LIST);
        assert!(!g.rasterizer_discard);
        assert_eq!(g.input_attachments.len(), 1);
        assert_eq!(g.input_attachments[0].color_index, 2);
        assert_eq!(g.input_attachments[0].format_class, SampledType::Float);
        // The fetch is ordered wherever the device can order it.
        assert_eq!(
            g.rasterization_order_attachment_access,
            p.fields().fragment.rasterization_order_color_access,
            "{stem}"
        );
        assert_eq!(
            g.rasterization_order_attachment_access,
            stem == "lavapipe",
            "{stem}: lavapipe alone of these orders attachment reads"
        );
        // Extended dynamic state makes the cull mode dynamic; the topology and its restart
        // follow with `extendedDynamicState2` too, together.
        assert_eq!(
            g.dynamic_states.contains(&vk::DynamicState::CULL_MODE),
            p.fields().dynamic_vertex_stride,
            "{stem}"
        );
        let input_assembly = p.fields().dynamic_vertex_stride
            && p.fields()
                .enabled
                .features
                .contains(&crate::device::requirements::VkFeature::ExtendedDynamicState2);
        for state in [
            vk::DynamicState::PRIMITIVE_TOPOLOGY,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
        ] {
            assert_eq!(
                g.dynamic_states.contains(&state),
                input_assembly,
                "{stem}: {state:?}"
            );
        }
        assert!(g.dynamic_states.contains(&vk::DynamicState::VIEWPORT));
        assert_eq!(
            l.layout.sets[0].bindings[0].source,
            BindingSource::InputAttachment(2)
        );
        // Framebuffer fetch of a color index without an attachment.
        let mut missing = r.clone();
        missing.color_attachments[2] = None;
        assert_eq!(
            link(&[&va, &fa], &LinkInputs::Render(Box::new(missing)), &p).unwrap_err(),
            Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 2 })
        );
        // The sample count maps (the fetching stage was compiled for one sample).
        let mut msaa = r.clone();
        msaa.raster_sample_count = 4;
        let g = link(&[&va], &LinkInputs::Render(Box::new(msaa)), &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"))
            .graphics
            .expect("graphics");
        assert_eq!(g.rasterization_samples, vk::SampleCountFlags::COUNT_4);
        let mut odd = r.clone();
        odd.raster_sample_count = 3;
        assert!(matches!(
            link(&[&va, &fa], &LinkInputs::Render(Box::new(odd)), &p),
            Err(Error::InvalidDescriptor(_))
        ));
        // Alpha to one is carried into the state on a device with the feature, and the
        // pipeline's multisample requirements name it. Each of these profiles has `alphaToOne`.
        let mut a2one = r.clone();
        a2one.alpha_to_one = true;
        let g = link(&[&va, &fa], &LinkInputs::Render(Box::new(a2one)), &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"))
            .graphics
            .expect("graphics");
        assert!(g.alpha_to_one, "{stem}");
        assert!(
            g.multisample_features().contains(&VkFeature::AlphaToOne),
            "{stem}"
        );
    }
}

/// `alphaToOneEnabled` is refused only where the device lacks Vulkan's `alphaToOne`, naming the
/// feature; a device that has it draws the pipeline (`graphics_state_maps_the_descriptor`).
#[test]
fn alpha_to_one_is_refused_only_without_the_feature() {
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    r.alpha_to_one = true;
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fa = artifact(iface(Stage::Fragment, vec![], vec![]), None);
    let inputs = LinkInputs::Render(Box::new(r));
    // `swiftshader` lists `alphaToOne` as unsupported, and `vk12-floor` states no such member.
    for stem in ["swiftshader", "vk12-floor"] {
        let p = profile(stem);
        assert!(
            !p.fields().enabled.features.contains(&VkFeature::AlphaToOne),
            "{stem} has the feature"
        );
        assert_eq!(
            link(&[&va, &fa], &inputs, &p).unwrap_err(),
            Error::Unsupported(Unsupported::needs(
                Feature::AlphaToOne,
                Capability::Feature(VkFeature::AlphaToOne)
            )),
            "{stem}"
        );
    }
    // The same inputs on a device with the feature link, and carry it.
    let p = profile("moltenvk-m3");
    let g = link(&[&va, &fa], &inputs, &p)
        .expect("moltenvk-m3 links")
        .graphics
        .expect("graphics");
    assert!(g.alpha_to_one);
}

/// The graphics state states whether the vertex function writes memory, and a rate-mapped pass
/// drawn in more than one region refuses such a pipeline, where one region, or a vertex function
/// without side effects, is drawn as Metal draws it.
#[test]
fn a_vertex_function_that_writes_memory_is_refused_in_a_pass_of_several_regions() {
    use crate::bind::{RasterizationRateMap, RateMapAxis, RegionError, RegionLimits};
    let p = profile("lavapipe");
    let r = RenderLinkInputs::new();
    let state = |writes: bool| {
        let mut vs = iface(Stage::Vertex, vec![], vec![]);
        vs.writes_memory = writes;
        let va = artifact(vs, None);
        link(&[&va], &LinkInputs::Render(Box::new(r.clone())), &p)
            .expect("link")
            .graphics
            .expect("graphics")
    };
    let (writes, pure) = (state(true), state(false));
    assert!(writes.vertex_writes_memory);
    assert!(!pure.vertex_writes_memory);
    let axis = |tiles: &[(u32, u32)]| RateMapAxis::from_tiles(32, tiles).expect("axis");
    let limits = RegionLimits::new([4096, 4096], [-8192.0, 8191.0]);
    // Two horizontal lines (63 to 32, then 4 to 4) make two regions; the identity makes one.
    let coarse = RasterizationRateMap::from_layers(
        [256, 32],
        vec![[
            axis(&[(63, 32), (63, 32), (63, 32), (63, 32), (4, 4)]),
            axis(&[(32, 32)]),
        ]],
    )
    .expect("map")
    .regions(0, [0.0, 0.0, 256.0, 32.0], None, &limits)
    .expect("plan");
    let identity = RasterizationRateMap::from_layers(
        [256, 32],
        vec![[axis(&[(32, 32); 8]), axis(&[(32, 32)])]],
    )
    .expect("map")
    .regions(0, [0.0, 0.0, 256.0, 32.0], None, &limits)
    .expect("plan");
    assert_eq!((coarse.regions().len(), identity.regions().len()), (2, 1));
    assert_eq!(
        coarse.admits(&writes),
        Err(RegionError::VertexSideEffects { regions: 2 })
    );
    assert_eq!(coarse.admits(&pure), Ok(()));
    assert_eq!(identity.admits(&writes), Ok(()));
}

/// The pipeline's interface lists each stage whose function may write a texture, in stage order,
/// and only those.
#[test]
fn the_interface_lists_the_stages_that_write_textures() {
    let p = profile("lavapipe");
    let r = RenderLinkInputs::new();
    let stages = |vertex: bool, fragment: bool| {
        let mut vs = iface(Stage::Vertex, vec![], vec![]);
        vs.writes_textures = vertex;
        let mut fs = iface(Stage::Fragment, vec![], vec![]);
        fs.writes_textures = fragment;
        let (va, fa) = (artifact(vs, None), artifact(fs, None));
        link(&[&va, &fa], &LinkInputs::Render(Box::new(r.clone())), &p)
            .expect("link")
            .interface
            .texture_writing_stages
    };
    assert_eq!(stages(false, false), Vec::<Stage>::new());
    assert_eq!(stages(false, true), vec![Stage::Fragment]);
    assert_eq!(stages(true, false), vec![Stage::Vertex]);
    assert_eq!(stages(true, true), vec![Stage::Vertex, Stage::Fragment]);
}

/// The graphics state states whether the vertex function selects its viewport, and a rate-mapped
/// pass refuses a pipeline that does, or that is layered (a stage declares `Layer`, or it amplifies
/// vertices), even in one region: the plan sets one viewport per region for one map layer. A
/// layered pipeline whose slices the consumer bounds to the plan's layer is admitted
/// (`admits_planned_layers`).
#[test]
fn a_layered_or_viewport_selecting_pipeline_is_refused_in_a_rate_mapped_pass() {
    use crate::bind::{RasterizationRateMap, RateMapAxis, RegionError, RegionLimits};
    use crate::pipeline::stage_io::StageBuiltin as B;
    let p = profile("lavapipe");
    let r = RenderLinkInputs::new();
    let state = |builtins: Vec<B>| {
        let mut vs = iface(Stage::Vertex, vec![], vec![]);
        vs.io.builtins.extend(builtins);
        let va = artifact(vs, None);
        link(&[&va], &LinkInputs::Render(Box::new(r.clone())), &p)
            .expect("link")
            .graphics
            .expect("graphics")
    };
    let (plain, viewport, layer) = (
        state(vec![]),
        state(vec![B::ViewportIndex]),
        state(vec![B::Layer]),
    );
    assert!(!plain.writes_viewport_index && !plain.declares_layer);
    assert!(viewport.writes_viewport_index);
    assert!(layer.declares_layer);
    let mut amplified = plain.clone();
    amplified.max_vertex_amplification_count = 2;
    let axis = |tiles: &[(u32, u32)]| RateMapAxis::from_tiles(32, tiles).expect("axis");
    let one =
        RasterizationRateMap::from_layers([64, 32], vec![[axis(&[(64, 32)]), axis(&[(32, 32)])]])
            .expect("map")
            .regions(
                0,
                [0.0, 0.0, 64.0, 32.0],
                None,
                &RegionLimits::new([4096, 4096], [-8192.0, 8191.0]),
            )
            .expect("plan");
    assert_eq!(one.regions().len(), 1);
    assert_eq!(one.admits(&plain), Ok(()));
    assert_eq!(one.admits(&viewport), Err(RegionError::ViewportArray));
    assert_eq!(one.admits(&layer), Err(RegionError::LayeredPipeline));
    assert_eq!(one.admits(&amplified), Err(RegionError::LayeredPipeline));
    // A draw whose slices the consumer bounds to the plan's layer: the layered pipelines are
    // admitted, the viewport-selecting one is not.
    assert_eq!(one.admits_planned_layers(&plain), Ok(()));
    assert_eq!(one.admits_planned_layers(&layer), Ok(()));
    assert_eq!(one.admits_planned_layers(&amplified), Ok(()));
    assert_eq!(
        one.admits_planned_layers(&viewport),
        Err(RegionError::ViewportArray)
    );
}

/// A fragment stage that reads its color attachments was compiled for its pipeline's sample
/// count: compiled for a multisampled pipeline it reads a multisampled `SubpassData` image at its
/// `SampleId`, compiled for one sample a single-sample image. Link takes each only into the kind of
/// pipeline it was compiled for, on every profile, whether or not the device orders attachment
/// reads, and answers the other `NeedsInput(Multisampled)` before any pipeline state exists. The
/// per-sample stage links wherever the device's color attachments, or its `RGBA8Unorm` ones, take
/// the count, as a stage without the fetch does.
#[test]
fn a_framebuffer_fetch_links_into_the_sample_count_it_was_compiled_for() {
    use crate::error::RuntimeInput;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::stage_io::StageBuiltin;
    let fr = Stage::Fragment;
    let fetch = || {
        StageDescriptor::new(
            res(fr, ResourceKind::InputAttachment(0), 0),
            vk::DescriptorType::INPUT_ATTACHMENT,
            1,
        )
    };
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fa = artifact(iface(fr, vec![fetch()], vec![]), None);
    let mut per_sample = iface(fr, vec![fetch()], vec![]);
    per_sample.io.multisampled = true;
    per_sample.io.builtins.push(StageBuiltin::SampleId);
    let sa = artifact(per_sample, None);
    let plain = artifact(iface(fr, vec![], vec![]), None);
    let recompile = Error::NeedsInput(RuntimeInput::Multisampled);
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna3-radv"] {
        let p = profile(stem);
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        for samples in [2, 4, 8] {
            let mut msaa = r.clone();
            msaa.raster_sample_count = samples;
            let inputs = LinkInputs::Render(Box::new(msaa));
            assert_eq!(
                link(&[&va, &fa], &inputs, &p).unwrap_err(),
                recompile,
                "{stem} at {samples} samples"
            );
            let enabled = &p.fields().enabled.properties;
            let supported = crate::pipeline::graphics::color_sample_count_alternatives(
                vk::Format::R8G8B8A8_UNORM,
                samples,
            )
            .iter()
            .any(|x| enabled.contains(x));
            for (fragment, fetches) in [(&sa, 1), (&plain, 0)] {
                match link(&[&va, fragment], &inputs, &p) {
                    Ok(l) => {
                        assert!(supported, "{stem} at {samples} samples");
                        let g = l.graphics.expect("graphics");
                        assert_eq!(g.input_attachments.len(), fetches, "{stem}");
                        assert_eq!(g.rasterization_samples.as_raw(), samples, "{stem}");
                    }
                    Err(Error::Unsupported(u)) if !supported => {
                        assert_eq!(u.feature, Feature::RasterSampleCount)
                    }
                    Err(e) => panic!("{stem} at {samples} samples: {e}"),
                }
            }
        }
        let one = LinkInputs::Render(Box::new(r));
        assert_eq!(
            link(&[&va, &sa], &one, &p).unwrap_err(),
            recompile,
            "{stem}"
        );
        let g = link(&[&va, &fa], &one, &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"))
            .graphics
            .expect("graphics");
        assert_eq!(g.input_attachments.len(), 1, "{stem}");
        assert_eq!(g.rasterization_samples, vk::SampleCountFlags::COUNT_1);
    }
}

/// A fragment stage that reads color attachment 0 in the interlocked storage form links into a
/// pipeline whose state chooses that form with the format the stage declares: the attachment is a
/// storage attachment, with no color attachment state and no input attachment. A pipeline whose
/// state chooses otherwise (another format, a blending attachment) did not compile the stage:
/// `NeedsInput(FramebufferFetch)`, and one of more samples did not compile it for its sample count:
/// `NeedsInput(Multisampled)`, and one with a depth attachment did not compile it with early tests.
/// A device that orders fetch otherwise never compiles such a stage. Compiled for more than one
/// sample, the stage reads planes where the device has no multisampled storage image (MoltenVK),
/// which need no sample count of storage images, and the pipeline's multisampled attachment
/// elsewhere (RADV).
#[test]
fn an_interlocked_fetch_links_only_into_the_state_it_was_compiled_for() {
    use crate::error::{InternalKind, RuntimeInput};
    use crate::metal::BlendFactor;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::graphics::{StorageAttachmentUse, StorageAttachmentView};
    let p = profile("moltenvk-m3");
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fr = Stage::Fragment;
    let storage = StageDescriptor::new(
        res(fr, ResourceKind::TileAttachment(0), 0),
        vk::DescriptorType::STORAGE_IMAGE,
        1,
    );
    let mut fa = artifact(iface(fr, vec![storage], vec![]), None);
    // Compiled without a depth or stencil attachment: no early tests.
    fa.facts.push(RuntimeFact::InterlockedFramebufferFetch {
        color: 0,
        format: vk::Format::R32_SFLOAT,
        writes: true,
        early_tests: false,
        kept: vk::ColorComponentFlags::empty(),
        blend: None,
        rounded: false,
    });
    // Blending adds the destination: a blend of one and zero, added, stores the source as no
    // blending does, and links into the storage form.
    let state_with = |format, blending: bool, samples: u32, depth: bool| {
        let mut r = RenderLinkInputs::new();
        let mut c = ColorAttachmentDescriptor::new(format);
        if blending {
            c.blending_enabled = true;
            c.destination_rgb_blend_factor = BlendFactor::One;
        }
        r.color_attachments[0] = Some(c);
        r.raster_sample_count = samples;
        r.depth_format = depth.then_some(PixelFormat::Depth32Float);
        LinkInputs::Render(Box::new(r))
    };
    let state = |format, blending: bool, samples: u32| state_with(format, blending, samples, false);
    let g = link(&[&va, &fa], &state(PixelFormat::R32Float, false, 1), &p)
        .expect("link")
        .graphics
        .expect("graphics");
    assert_eq!(
        g.storage_attachments,
        [StorageAttachmentUse {
            color_index: 0,
            format: vk::Format::R32_SFLOAT,
            writes: true,
            // Declared without a format where the device grants that.
            formatless: crate::metal::format::accessed_without_format(
                vk::Format::R32_SFLOAT,
                crate::metal::format::StorageUse::texture(true, true),
                &p
            ),
            view: StorageAttachmentView::Attachment,
            view_alias: None,
            kept: vk::ColorComponentFlags::empty(),
        }]
    );
    assert_eq!(g.color_attachments, [None]);
    assert!(g.input_attachments.is_empty());
    assert!(!g.rasterization_order_attachment_access);
    for other in [
        state(PixelFormat::RGBA8Unorm, false, 1),
        state(PixelFormat::R32Float, true, 1),
        // Its writes would precede a failed depth test.
        state_with(PixelFormat::R32Float, false, 1, true),
    ] {
        assert_eq!(
            link(&[&va, &fa], &other, &p).unwrap_err(),
            Error::NeedsInput(RuntimeInput::FramebufferFetch),
            "{other:?}"
        );
    }
    assert_eq!(
        link(&[&va, &fa], &state(PixelFormat::R32Float, false, 4), &p).unwrap_err(),
        Error::NeedsInput(RuntimeInput::Multisampled)
    );
    // Compiled for more than one sample: planes without multisampled storage images.
    let mut per_sample = fa.clone();
    per_sample.interface.io.multisampled = true;
    per_sample
        .interface
        .io
        .builtins
        .push(crate::pipeline::stage_io::StageBuiltin::SampleId);
    let radv = profile("amd-rdna3-radv");
    for (p, view) in [
        (&p, StorageAttachmentView::SamplePlanes { layers: 4 }),
        (&radv, StorageAttachmentView::Attachment),
    ] {
        let g = link(
            &[&va, &per_sample],
            &state(PixelFormat::R32Float, false, 4),
            p,
        )
        .expect("link")
        .graphics
        .expect("graphics");
        assert_eq!(
            g.storage_attachments
                .iter()
                .map(|a| a.view)
                .collect::<Vec<_>>(),
            [view]
        );
        assert_eq!(g.rasterization_samples, vk::SampleCountFlags::COUNT_4);
    }

    match link(
        &[&va, &fa],
        &state(PixelFormat::R32Float, false, 1),
        &profile("lavapipe"),
    ) {
        Err(Error::Internal(e)) => assert_eq!(e.kind, InternalKind::Invariant),
        other => panic!("{other:?}"),
    }
}

/// A color attachment that no output of the fragment function names keeps its contents, as Metal
/// kept one where measured: its write mask is empty, while every attachment the function writes
/// keeps the descriptor's mask and every attachment its blend state. A pipeline without a fragment
/// function writes no attachment. Under the descriptor's mask the older text of Vulkan's fragment
/// output interface leaves an attachment the fragment shader has no output for undefined (the
/// current text keeps it when no output is declared at its location), and with no fragment shader
/// undefined values may be written to every color attachment.
#[test]
fn a_color_attachment_the_fragment_does_not_write_keeps_its_contents() {
    use crate::metal::state::color_write_mask;
    use crate::metal::ColorWriteMask;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let f4 = io_ty(IoNumeric::Float, 4);
    let u4 = io_ty(IoNumeric::Uint, 4);
    let partial = ColorWriteMask::RED | ColorWriteMask::ALPHA;
    let none = vk::ColorComponentFlags::empty();
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna3-radv"] {
        let p = profile(stem);
        let mut r = RenderLinkInputs::new();
        let mut masked = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        masked.write_mask = partial;
        r.color_attachments[0] = Some(masked);
        r.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        r.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        // Blended, adding the destination (one and zero, added, would store the source as no
        // blending does).
        let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        blended.blending_enabled = true;
        blended.destination_rgb_blend_factor = crate::metal::BlendFactor::One;
        r.color_attachments[3] = Some(blended);
        let inputs = LinkInputs::Render(Box::new(r));
        let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
        // Writes attachments 0 and 2; 1 (between them) and 3 have no output.
        let fa = io_stage(
            Stage::Fragment,
            stage_io(vec![], vec![color(0, 0, f4), color(2, 0, u4)]),
        );
        let masks = |stages: &[&FunctionArtifact]| -> Vec<vk::ColorComponentFlags> {
            let g = link(stages, &inputs, &p)
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .graphics
                .expect("graphics");
            assert!(g.color_attachments[3]
                .expect("attachment 3")
                .blend
                .is_some());
            g.color_attachments
                .iter()
                .map(|c| c.expect("every attachment is set").write_mask)
                .collect()
        };
        let all = color_write_mask(ColorWriteMask::ALL).expect("mask");
        assert_eq!(
            masks(&[&va, &fa]),
            [color_write_mask(partial).expect("mask"), none, all, none],
            "{stem}"
        );
        assert_eq!(masks(&[&va]), [none; 4], "{stem}: no fragment function");
    }
}

/// Without `independentBlend` every element of `pAttachments` must be identical
/// (`VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605`). On `vk12-floor`, which lacks
/// the feature, link gives every attachment one state where that is exact and refuses
/// `Unsupported{IndependentBlending, Feature(IndependentBlend)}` otherwise; on `lavapipe`, which
/// has it, the states stay the descriptor's (unwritten attachments masked) and the color blend
/// features name `independentBlend` exactly when the elements differ, an absent attachment's
/// element being the first present attachment's.
#[test]
fn attachment_states_are_identical_or_need_independent_blend() {
    use crate::device::requirements::FormatUsage;
    use crate::metal::state::color_write_mask;
    use crate::metal::{BlendFactor, ColorWriteMask};
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::graphics::GraphicsState;
    let f4 = io_ty(IoNumeric::Float, 4);
    let u4 = io_ty(IoNumeric::Uint, 4);
    let all = color_write_mask(ColorWriteMask::ALL).expect("mask");
    let red = color_write_mask(ColorWriteMask::RED).expect("mask");
    let none = vk::ColorComponentFlags::empty();
    let plain = ColorAttachmentDescriptor::new;
    let blended = |format| {
        let mut a = ColorAttachmentDescriptor::new(format);
        a.blending_enabled = true;
        a.source_rgb_blend_factor = BlendFactor::SourceAlpha;
        a.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        a
    };
    let masked = |mut a: ColorAttachmentDescriptor, mask| {
        a.write_mask = mask;
        a
    };
    let floor = profile("vk12-floor");
    let lavapipe = profile("lavapipe");
    assert!(!floor.fields().fragment.independent_blend);
    assert!(lavapipe.fields().fragment.independent_blend);
    // Refused at the attachment whose state differs from the others'.
    let refused = |index: u32| {
        tied(
            Unsupported::needs(
                Feature::IndependentBlending,
                Capability::Feature(VkFeature::IndependentBlend),
            ),
            Attachment::Color(index),
        )
    };
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fragment = |outputs| io_stage(Stage::Fragment, stage_io(vec![], outputs));
    let link_on = |p: &DeviceProfile,
                   attachments: &[(usize, ColorAttachmentDescriptor)],
                   fs: Option<&FunctionArtifact>|
     -> Result<GraphicsState, Error> {
        let mut r = RenderLinkInputs::new();
        for (index, a) in attachments {
            r.color_attachments[*index] = Some(*a);
        }
        let stages: Vec<&FunctionArtifact> = std::iter::once(&va).chain(fs).collect();
        let g = link(&stages, &LinkInputs::Render(Box::new(r)), p)?
            .graphics
            .expect("graphics");
        // Without the feature, every element is identical, so none is required.
        if !p.fields().fragment.independent_blend {
            let elements = g.color_blend_attachments();
            assert!(elements.windows(2).all(|e| e[0] == e[1]), "{elements:?}");
            assert!(!g
                .color_blend_features()
                .contains(&VkFeature::IndependentBlend));
        }
        Ok(g)
    };
    let masks = |g: &GraphicsState| -> Vec<Option<vk::ColorComponentFlags>> {
        g.color_attachments
            .iter()
            .map(|a| a.map(|a| a.write_mask))
            .collect()
    };
    let independent = |g: &GraphicsState| {
        g.color_blend_features()
            .contains(&VkFeature::IndependentBlend)
    };

    // (i) An attachment the fragment function has no output for, between written ones, and an
    // absent attachment: masked where the device blends independently, written under the
    // others' mask where it does not (Vulkan leaves it unmodified).
    let attachments = [
        (0, plain(PixelFormat::RGBA8Unorm)),
        (1, plain(PixelFormat::RGBA32Float)),
        (3, plain(PixelFormat::RGBA32Uint)),
    ];
    let writes_0_3 = fragment(vec![color(0, 0, f4), color(3, 0, u4)]);
    let g = link_on(&lavapipe, &attachments, Some(&writes_0_3)).expect("lavapipe");
    assert_eq!(masks(&g), [Some(all), Some(none), None, Some(all)]);
    assert!(independent(&g));
    let g = link_on(&floor, &attachments, Some(&writes_0_3)).expect("floor");
    assert_eq!(masks(&g), [Some(all), Some(all), None, Some(all)]);
    assert_eq!(g.color_blend_attachments(), [(all, None); 4]);
    // Every attachment written: identical everywhere, nothing required.
    let writes_0_1_3 = fragment(vec![color(0, 0, f4), color(1, 0, f4), color(3, 0, u4)]);
    for p in [&floor, &lavapipe] {
        let g = link_on(p, &attachments, Some(&writes_0_1_3)).expect("every attachment written");
        assert_eq!(masks(&g), [Some(all), Some(all), None, Some(all)]);
        assert!(!independent(&g));
    }

    // No fragment function: every mask is empty, so no blend writes anything; without the
    // feature none blends.
    let attachments = [
        (0, blended(PixelFormat::RGBA8Unorm)),
        (1, plain(PixelFormat::RGBA32Float)),
    ];
    let g = link_on(&lavapipe, &attachments, None).expect("lavapipe");
    assert_eq!(masks(&g), [Some(none), Some(none)]);
    assert!(g.color_attachments[0]
        .expect("attachment 0")
        .blend
        .is_some());
    assert!(independent(&g));
    let g = link_on(&floor, &attachments, None).expect("floor");
    assert_eq!(g.color_blend_attachments(), [(none, None); 2]);

    // (ii) Written attachments whose blend states or masks differ: no identical state is exact.
    let writes_0_1 = fragment(vec![color(0, 0, f4), color(1, 0, f4)]);
    for attachments in [
        [
            (0, blended(PixelFormat::RGBA8Unorm)),
            (1, plain(PixelFormat::RGBA8Unorm)),
        ],
        [
            (0, plain(PixelFormat::RGBA8Unorm)),
            (
                1,
                masked(plain(PixelFormat::RGBA8Unorm), ColorWriteMask::RED),
            ),
        ],
        [
            (0, plain(PixelFormat::RGBA8Unorm)),
            (
                1,
                masked(plain(PixelFormat::RGBA8Unorm), ColorWriteMask::NONE),
            ),
        ],
    ] {
        assert_eq!(
            link_on(&floor, &attachments, Some(&writes_0_1)).unwrap_err(),
            refused(1),
            "{attachments:?}"
        );
        let g = link_on(&lavapipe, &attachments, Some(&writes_0_1)).expect("lavapipe");
        assert!(independent(&g), "{attachments:?}");
    }
    // The same blend and mask on every written attachment: kept as they are.
    let attachments = [
        (
            0,
            masked(blended(PixelFormat::RGBA8Unorm), ColorWriteMask::RED),
        ),
        (
            1,
            masked(blended(PixelFormat::RGBA16Float), ColorWriteMask::RED),
        ),
    ];
    for p in [&floor, &lavapipe] {
        let g = link_on(p, &attachments, Some(&writes_0_1)).expect("identical");
        assert_eq!(masks(&g), [Some(red), Some(red)]);
        assert!(g
            .color_attachments
            .iter()
            .all(|a| a.unwrap().blend.is_some()));
        assert!(!independent(&g));
    }

    // An unwritten attachment takes the written one's blend only where its format blends.
    let writes_0 = fragment(vec![color(0, 0, f4)]);
    let blend_into = |format| [(0, blended(PixelFormat::RGBA8Unorm)), (1, plain(format))];
    assert!(!floor.supports_format(
        vk::Format::R32G32B32A32_SFLOAT,
        FormatUsage::ColorAttachmentBlend
    ));
    assert_eq!(
        link_on(
            &floor,
            &blend_into(PixelFormat::RGBA32Float),
            Some(&writes_0)
        )
        .unwrap_err(),
        refused(1)
    );
    let g = link_on(
        &floor,
        &blend_into(PixelFormat::RGBA16Float),
        Some(&writes_0),
    )
    .expect("a blendable unwritten attachment");
    let kept = g.color_attachments[0].expect("attachment 0");
    assert_eq!(
        g.color_attachments[1].expect("attachment 1").blend,
        kept.blend
    );
    assert_eq!(masks(&g), [Some(all), Some(all)]);
    for format in [PixelFormat::RGBA32Float, PixelFormat::RGBA16Float] {
        let g = link_on(&lavapipe, &blend_into(format), Some(&writes_0)).expect("lavapipe");
        assert_eq!(masks(&g), [Some(all), Some(none)]);
        assert!(g.color_attachments[1].unwrap().blend.is_none());
    }

    // A written attachment under an empty mask and an unwritten one: nothing is written, so
    // neither blends without the feature.
    let attachments = [
        (
            0,
            masked(blended(PixelFormat::RGBA8Unorm), ColorWriteMask::NONE),
        ),
        (1, plain(PixelFormat::RGBA32Float)),
    ];
    let g = link_on(&floor, &attachments, Some(&writes_0)).expect("floor");
    assert_eq!(g.color_blend_attachments(), [(none, None); 2]);
    let g = link_on(&lavapipe, &attachments, Some(&writes_0)).expect("lavapipe");
    assert!(independent(&g), "the masks agree, the blends differ");
}

/// A blend that reads the fragment's second output needs `dualSrcBlend`: the color blend features
/// name it exactly then, and a device without it refuses the factor at link.
#[test]
fn a_dual_source_blend_needs_dual_src_blend() {
    use crate::metal::BlendFactor;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let f4 = io_ty(IoNumeric::Float, 4);
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fa = io_stage(
        Stage::Fragment,
        stage_io(vec![], vec![color(0, 0, f4), color(0, 1, f4)]),
    );
    let inputs = |factor| {
        let mut a = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        a.blending_enabled = true;
        a.destination_rgb_blend_factor = factor;
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(a);
        LinkInputs::Render(Box::new(r))
    };
    let lavapipe = profile("lavapipe");
    assert!(lavapipe.fields().fragment.dual_src_blend);
    for (factor, dual) in [
        (BlendFactor::OneMinusSource1Color, true),
        (BlendFactor::OneMinusSourceAlpha, false),
    ] {
        let g = link(&[&va, &fa], &inputs(factor), &lavapipe)
            .expect("lavapipe")
            .graphics
            .expect("graphics");
        assert_eq!(
            g.color_blend_features(),
            if dual {
                BTreeSet::from([VkFeature::DualSrcBlend])
            } else {
                BTreeSet::new()
            },
            "{factor:?}"
        );
    }
    let floor = profile("vk12-floor");
    assert!(!floor.fields().fragment.dual_src_blend);
    assert_eq!(
        link(
            &[&va, &fa],
            &inputs(BlendFactor::OneMinusSource1Color),
            &floor
        )
        .unwrap_err(),
        tied(
            Unsupported::needs(
                Feature::DualSourceBlending,
                Capability::Feature(VkFeature::DualSrcBlend)
            ),
            Attachment::Color(0)
        )
    );
}

/// More color attachments than `maxColorAttachments` is refused naming the limit.
#[test]
fn color_attachments_beyond_the_limit_are_refused() {
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let mut p = profile("moltenvk-m3");
    p.fields.limits.max_color_attachments = 4;
    let mut r = RenderLinkInputs::new();
    r.color_attachments[5] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    match link(&[&va], &LinkInputs::Render(Box::new(r)), &p) {
        Err(Error::Unsupported(u)) => assert_eq!(
            u.needs,
            crate::error::Needs::AnyOf(vec![vec![Capability::Limit {
                limit: VkLimit::MaxColorAttachments,
                required: 6,
                available: 4
            }]])
        ),
        other => panic!("{other:?}"),
    }
}

/// Metal's fragment output rules at link, each measured on an Apple M3 Max: an output with fewer
/// components than its attachment's channels is Metal's pipeline error (either dual-source index,
/// `A8Unorm` needing four); an attachment named only by a dual-source second output is left
/// unmodified (unblended, or blended without a `Source1` factor), and a blend that reads the second
/// output needs the first; without independent blending such an attachment keeps its empty mask.
#[test]
fn color_outputs_follow_metals_measured_rules() {
    use crate::metal::state::color_write_mask;
    use crate::metal::{BlendFactor, ColorWriteMask};
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::graphics::GraphicsState;
    let f = |n| io_ty(IoNumeric::Float, n);
    let all = color_write_mask(ColorWriteMask::ALL).expect("mask");
    let none = vk::ColorComponentFlags::empty();
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fragment = |outputs| io_stage(Stage::Fragment, stage_io(vec![], outputs));
    let link_on = |p: &DeviceProfile,
                   attachments: &[(usize, ColorAttachmentDescriptor)],
                   fs: &FunctionArtifact|
     -> Result<GraphicsState, Error> {
        let mut r = RenderLinkInputs::new();
        for (index, a) in attachments {
            r.color_attachments[*index] = Some(*a);
        }
        Ok(link(&[&va, fs], &LinkInputs::Render(Box::new(r)), p)?
            .graphics
            .expect("graphics"))
    };
    let invalid = |index| Error::InvalidDescriptor(DescriptorError::ColorAttachment { index });
    let lavapipe = profile("lavapipe");
    let floor = profile("vk12-floor");
    for p in [&lavapipe, &profile("moltenvk-m3")] {
        // Components against channels.
        for (format, components, ok) in [
            (PixelFormat::RGBA8Unorm, 2, false),
            (PixelFormat::RGBA8Unorm, 3, false),
            (PixelFormat::RGBA8Unorm, 4, true),
            (PixelFormat::BGRA8Unorm, 3, false),
            (PixelFormat::RGB10A2Unorm, 3, false),
            (PixelFormat::RGBA16Float, 1, false),
            (PixelFormat::RG11B10Float, 2, false),
            (PixelFormat::RG11B10Float, 3, true),
            (PixelFormat::RG16Float, 1, false),
            (PixelFormat::RG16Float, 2, true),
            (PixelFormat::R32Float, 1, true),
            (PixelFormat::R32Float, 4, true),
        ] {
            let attachments = [(0, ColorAttachmentDescriptor::new(format))];
            for index in [0, 1] {
                let fs = fragment(if index == 0 {
                    vec![color(0, 0, f(components))]
                } else {
                    vec![color(0, 1, f(components))]
                });
                let got = link_on(p, &attachments, &fs).map(|_| ());
                assert_eq!(
                    got,
                    if ok { Ok(()) } else { Err(invalid(0)) },
                    "{format:?} from {components} components at index {index}"
                );
            }
        }
        // An output to an absent attachment is discarded whatever its width.
        let attachments = [(0, ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm))];
        let narrow_absent = fragment(vec![color(0, 0, f(4)), color(1, 0, f(2))]);
        link_on(p, &attachments, &narrow_absent).expect("an absent attachment");
    }

    // An attachment named only by a second output keeps its contents.
    let second_only = fragment(vec![color(0, 1, f(4))]);
    // Blended, adding the destination (one and zero, added, would store the source as no
    // blending does).
    let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    blended.blending_enabled = true;
    blended.destination_rgb_blend_factor = crate::metal::BlendFactor::One;
    for a in [
        ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float),
        blended,
    ] {
        let g = link_on(&lavapipe, &[(0, a)], &second_only).expect("second output only");
        assert_eq!(
            g.color_attachments[0].expect("attachment 0").write_mask,
            none
        );
    }
    // A blend that reads the second output needs the first.
    let mut dual = blended;
    dual.destination_rgb_blend_factor = BlendFactor::Source1Color;
    assert_eq!(
        link_on(&lavapipe, &[(0, dual)], &second_only).unwrap_err(),
        invalid(0)
    );
    let both = fragment(vec![color(0, 0, f(4)), color(0, 1, f(4))]);
    let g = link_on(&lavapipe, &[(0, dual)], &both).expect("both outputs");
    assert_eq!(
        g.color_attachments[0].expect("attachment 0").write_mask,
        all
    );
    // Without independent blending, the second-output-only attachment's empty mask stays: beside
    // a written attachment no identical state is exact, alone it is masked.
    let plain = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    let mixed = fragment(vec![color(0, 1, f(4)), color(1, 0, f(4))]);
    assert_eq!(
        link_on(&floor, &[(0, plain), (1, plain)], &mixed).unwrap_err(),
        tied(
            Unsupported::needs(
                Feature::IndependentBlending,
                Capability::Feature(VkFeature::IndependentBlend),
            ),
            Attachment::Color(0)
        )
    );
    let g = link_on(&lavapipe, &[(0, plain), (1, plain)], &mixed).expect("lavapipe");
    let masks: Vec<_> = g
        .color_attachments
        .iter()
        .map(|a| a.expect("present").write_mask)
        .collect();
    assert_eq!(masks, [none, all]);
    let g = link_on(&floor, &[(0, plain)], &second_only).expect("floor");
    assert_eq!(g.color_blend_attachments(), [(none, None)]);
}

/// Alpha to coverage reads the alpha of the first output at attachment 0. Metal leaves the coverage
/// as rasterized without a four-component float output there (none, a narrower or an integer one;
/// measured at 1 and 4 samples), and Vulkan requires one
/// (`VUID-VkGraphicsPipelineCreateInfo-alphaToCoverageEnable-08891`), so link turns it off; over
/// an integer attachment 0 Metal refuses the descriptor.
#[test]
fn alpha_to_coverage_reads_a_four_component_float_output_at_attachment_0() {
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let p = profile("lavapipe");
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let fragment = |outputs| io_stage(Stage::Fragment, stage_io(vec![], outputs));
    let coverage = |attachments: &[(usize, PixelFormat)],
                    fs: Option<&FunctionArtifact>|
     -> Result<bool, Error> {
        let mut r = RenderLinkInputs::new();
        r.alpha_to_coverage = true;
        for &(index, format) in attachments {
            r.color_attachments[index] = Some(ColorAttachmentDescriptor::new(format));
        }
        let stages: Vec<&FunctionArtifact> = std::iter::once(&va).chain(fs).collect();
        Ok(link(&stages, &LinkInputs::Render(Box::new(r)), &p)?
            .graphics
            .expect("graphics")
            .alpha_to_coverage)
    };
    let f = |n| io_ty(IoNumeric::Float, n);
    let u4 = io_ty(IoNumeric::Uint, 4);
    let rgba = [(0, PixelFormat::RGBA16Float), (1, PixelFormat::RGBA16Float)];
    assert_eq!(
        coverage(&rgba, Some(&fragment(vec![color(0, 0, f(4))]))),
        Ok(true)
    );
    // Alpha at attachment 0 counts when the attachment is absent, as Metal reads it there.
    assert_eq!(
        coverage(
            &[(1, PixelFormat::RGBA16Float)],
            Some(&fragment(vec![color(0, 0, f(4)), color(1, 0, f(4))]))
        ),
        Ok(true)
    );
    for (attachments, fs) in [
        (&rgba[..], None),
        (&rgba[..], Some(fragment(vec![color(1, 0, f(4))]))),
        (&rgba[..], Some(fragment(vec![color(0, 1, f(4))]))),
        (
            &[(0, PixelFormat::RG11B10Float)][..],
            Some(fragment(vec![color(0, 0, f(3))])),
        ),
        (
            &[(0, PixelFormat::R32Float)][..],
            Some(fragment(vec![color(0, 0, f(1))])),
        ),
        (
            &[(1, PixelFormat::RGBA16Float)][..],
            Some(fragment(vec![color(0, 0, u4), color(1, 0, f(4))])),
        ),
    ] {
        assert_eq!(
            coverage(attachments, fs.as_ref()),
            Ok(false),
            "{attachments:?}"
        );
    }
    assert_eq!(
        coverage(
            &[(0, PixelFormat::RGBA32Uint)],
            Some(&fragment(vec![color(0, 0, u4)]))
        ),
        Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
            index: 0
        }))
    );
}

/// A raster sample count above 1 must be one the device's framebuffer sample counts list for
/// every kind of attachment the pipeline has: float and integer color, depth, and stencil by their
/// formats' components, or none at all; a color attachment may instead take its own format's count.
/// The state names the properties, link refuses a count the profile lacks naming every alternative,
/// and one sample or a pipeline without rasterization needs nothing.
#[test]
fn the_raster_sample_count_needs_the_framebuffer_sample_counts() {
    use crate::device::requirements::VkProperty as P;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::graphics::GraphicsState;
    let base = profile("moltenvk-m3");
    for count in [1, 2, 4] {
        for property in [
            P::FramebufferColorSampleCount,
            P::FramebufferIntegerColorSampleCount,
            P::FramebufferDepthSampleCount,
            P::FramebufferStencilSampleCount,
            P::FramebufferNoAttachmentsSampleCount,
        ] {
            assert!(base.fields().enabled.properties.contains(&property(count)));
        }
    }
    assert!(!base
        .fields()
        .enabled
        .properties
        .contains(&P::FramebufferColorSampleCount(8)));
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let link_on = |p: &DeviceProfile,
                   colors: &[PixelFormat],
                   depth: Option<PixelFormat>,
                   stencil: Option<PixelFormat>,
                   count: u32,
                   rasterize: bool|
     -> Result<GraphicsState, Error> {
        let mut r = RenderLinkInputs::new();
        for (i, &format) in colors.iter().enumerate() {
            r.color_attachments[i] = Some(ColorAttachmentDescriptor::new(format));
        }
        r.depth_format = depth;
        r.stencil_format = stencil;
        r.raster_sample_count = count;
        r.rasterization_enabled = rasterize;
        Ok(link(&[&va], &LinkInputs::Render(Box::new(r)), p)?
            .graphics
            .expect("graphics"))
    };
    // Refused at the attachment the property is the need of: the first color attachment of the
    // kind, the depth or the stencil attachment, and none without attachments.
    let refused = |alternatives: &[P], colors: &[PixelFormat]| {
        let u = Unsupported::needs_any_of(
            Feature::RasterSampleCount,
            alternatives
                .iter()
                .map(|&p| vec![Capability::Property(p)])
                .collect(),
        );
        let integer =
            |f: PixelFormat| crate::metal::format::sampled_type(f) != Some(SampledType::Float);
        let attachment = match alternatives[0] {
            P::FramebufferColorSampleCount(_) => colors.iter().position(|&f| !integer(f)),
            P::FramebufferIntegerColorSampleCount(_) => colors.iter().position(|&f| integer(f)),
            P::FramebufferDepthSampleCount(_) => return tied(u, Attachment::Depth),
            P::FramebufferStencilSampleCount(_) => return tied(u, Attachment::Stencil),
            _ => None,
        };
        match attachment {
            Some(index) => tied(u, Attachment::Color(index as u32)),
            None => Error::Unsupported(u),
        }
    };
    // A need's alternatives: a color attachment's own format's count after its kind's.
    let alternatives = |property: P| match property {
        P::FramebufferColorSampleCount(n) => vec![
            property,
            P::ColorAttachmentSampleCount(vk::Format::R8G8B8A8_UNORM, n),
        ],
        P::FramebufferIntegerColorSampleCount(n) => vec![
            property,
            P::ColorAttachmentSampleCount(vk::Format::R32G32B32A32_UINT, n),
        ],
        _ => vec![property],
    };
    let set = |props: &[P]| props.iter().copied().collect::<BTreeSet<P>>();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let uint = PixelFormat::RGBA32Uint;
    let d32s8 = PixelFormat::Depth32FloatStencil8;
    // Each kind of attachment names its own property.
    for (colors, depth, stencil, want) in [
        (
            vec![rgba8],
            None,
            None,
            vec![P::FramebufferColorSampleCount(4)],
        ),
        (
            vec![rgba8, uint],
            None,
            None,
            vec![
                P::FramebufferColorSampleCount(4),
                P::FramebufferIntegerColorSampleCount(4),
            ],
        ),
        (
            vec![],
            Some(PixelFormat::Depth32Float),
            None,
            vec![P::FramebufferDepthSampleCount(4)],
        ),
        (
            vec![],
            None,
            Some(PixelFormat::Stencil8),
            vec![P::FramebufferStencilSampleCount(4)],
        ),
        (
            vec![],
            Some(d32s8),
            Some(d32s8),
            vec![
                P::FramebufferDepthSampleCount(4),
                P::FramebufferStencilSampleCount(4),
            ],
        ),
        (
            vec![],
            None,
            None,
            vec![P::FramebufferNoAttachmentsSampleCount(4)],
        ),
    ] {
        let g = link_on(&base, &colors, depth, stencil, 4, true).expect("4 samples");
        assert_eq!(
            g.sample_count_properties(&base.fields().enabled.properties),
            Ok(set(&want)),
            "{colors:?} {depth:?}"
        );
        // The same pipeline on a profile lacking any one of them, and its format's own count, is
        // refused naming both.
        for &missing in &want {
            let mut p = base.clone();
            for alternative in alternatives(missing) {
                p.fields.enabled.properties.remove(&alternative);
            }
            assert_eq!(
                link_on(&p, &colors, depth, stencil, 4, true).unwrap_err(),
                refused(&alternatives(missing), &colors),
                "{missing:?}"
            );
        }
        // Eight samples are beyond this device for every kind.
        assert!(matches!(
            link_on(&base, &colors, depth, stencil, 8, true),
            Err(Error::Unsupported(u)) if u.feature == Feature::RasterSampleCount
        ));
        // One sample, or no rasterization, needs nothing.
        let mut bare = base.clone();
        bare.fields.enabled.properties.clear();
        let none = &bare.fields().enabled.properties;
        let g = link_on(&bare, &colors, depth, stencil, 1, true).expect("one sample");
        assert_eq!(g.sample_count_properties(none), Ok(BTreeSet::new()));
        let g = link_on(&bare, &colors, depth, stencil, 8, false).expect("no rasterization");
        assert_eq!(g.sample_count_properties(none), Ok(BTreeSet::new()));
    }
    // A color attachment's own format may take a count its kind's limit lacks: the pipeline then
    // needs that format's count; the kind's limit, where the device has it, is named first.
    let integer = P::FramebufferIntegerColorSampleCount(4);
    let own = P::ColorAttachmentSampleCount(vk::Format::R32G32B32A32_UINT, 4);
    let mut p = base.clone();
    p.fields.enabled.properties.remove(&integer);
    p.fields.enabled.properties.insert(own);
    let g = link_on(&p, &[uint], None, None, 4, true).expect("the format's own count");
    assert_eq!(
        g.sample_count_properties(&p.fields().enabled.properties),
        Ok(set(&[own]))
    );
    let mut both = base.clone();
    both.fields.enabled.properties.insert(own);
    let g = link_on(&both, &[uint], None, None, 4, true).expect("the kind's count");
    assert_eq!(
        g.sample_count_properties(&both.fields().enabled.properties),
        Ok(set(&[integer]))
    );
}

/// Under multisampling Metal (Apple M3 Max, 4x) runs a fragment function that writes memory once
/// per covered sample, whatever its inputs, and every invocation reads its position, its center
/// and centroid varyings, its point and barycentric coordinates, and its sample mask as the
/// pixel's. Vulkan runs per sample only under sample shading, which `SampleId` turns on, a `Sample`
/// input may, and the pipeline enables for a function that writes memory; it moves those reads to
/// the sample there. The same pipelines at one sample, a pixel-rate function that reads them (but
/// its `[[sample_mask]]` on lavapipe, whose split of a clipped triangle's fragments changes it), a
/// per-sample one that does not, one compiled for the pipeline that reads `[[position]]`'s x and y
/// or its center varyings at the pixel, and a per-sample one that reads `[[position]]`'s z or w, a
/// centroid varying, or its point or barycentric coordinates where Vulkan evaluates them (which
/// the compile layer notes), link, with sample shading exactly where the function writes memory
/// at four samples. A per-sample function whose pixel reads all have forms on the
/// device, compiled without `multisampled`, needs that input. A center varying has no pixel form on
/// `moltenvk-m3` at a release not measured to read it at the pixel center itself (1.4.0), whose
/// `InterpolateAtOffset` reads off the center, which the refusal names; at 1.4.1, measured so, its
/// read as it is is the pixel form, as on a device whose offset reads from the center. A device
/// without `sampleRateShading` is named for the side effects and for a read that recompiling would
/// not fix.
#[test]
fn multisampled_fragment_invocations_follow_metals_measured_rules() {
    use crate::device::requirements::VkProperty;
    use crate::device::DriverWorkaround;
    use crate::error::RuntimeInput;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::stage_io::StageBuiltin as B;
    let f4 = io_ty(IoNumeric::Float, 4);
    let smooth = |sampling| {
        Some(Interpolation::Smooth {
            sampling,
            perspective: true,
        })
    };
    let mut vs = iface(Stage::Vertex, vec![], vec![]);
    vs.io = stage_io(
        vec![],
        ["c", "k", "s", "f"]
            .iter()
            .map(|n| varying(n, f4, None))
            .collect(),
    );
    let va = artifact(vs, None);
    let fragment = |inputs: Vec<IoVariable>, builtins: Vec<B>, writes: bool| {
        let mut fs = iface(Stage::Fragment, vec![], vec![]);
        fs.io = stage_io(inputs, vec![color(0, 0, f4)]);
        fs.io.builtins = builtins;
        fs.writes_memory = writes;
        artifact(fs, None)
    };
    // Compiled for a multisampled pipeline: `[[position]]`'s x and y at the pixel, center
    // varyings at the pixel when `varyings`, and `[[position]]`'s z or w read when `depth`.
    let compiled = |inputs: Vec<IoVariable>, builtins: Vec<B>, varyings: bool, depth: bool| {
        let mut fs = iface(Stage::Fragment, vec![], vec![]);
        fs.io = stage_io(inputs, vec![color(0, 0, f4)]);
        fs.io.builtins = builtins;
        fs.io.multisampled = true;
        fs.io.varyings_at_pixel = varyings;
        fs.io.position_depth = depth;
        artifact(fs, None)
    };
    let center = varying("c", f4, smooth(Sampling::Center));
    let centroid = varying("k", f4, smooth(Sampling::Centroid));
    let sample = varying("s", f4, smooth(Sampling::Sample));
    let flat = varying("f", f4, Some(Interpolation::Flat));
    let bary = B::BaryCoord { perspective: false };
    let pixel_reads = vec![B::FragCoord, B::PointCoord, B::SampleMaskIn, bary];
    // At 4 samples, what link does with each function on a device that has its pixel forms.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum At4 {
        Links,
        // Every pixel read has a form the device has, which a function compiled without
        // `multisampled` does not use.
        Recompile,
        // A center varying, which on a device whose `InterpolateAtOffset` reads off the center
        // has no pixel form.
        Center,
        // The pixel's coverage, which no device gives a per-sample invocation.
        Coverage,
    }
    use At4::{Center, Coverage, Links, Recompile};
    let cases = [
        // Pixel rate, no side effect: every pixel read is the pixel's in both APIs.
        (
            fragment(vec![center.clone(), centroid.clone()], pixel_reads, false),
            Links,
        ),
        // Side effects: the pipeline's sample shading runs them once per covered sample.
        (fragment(vec![], vec![], true), Links),
        (
            fragment(vec![flat.clone()], vec![B::FrontFacing], true),
            Links,
        ),
        (fragment(vec![sample.clone()], vec![], true), Links),
        // ... and moves the pixel reads to the sample, where they have no pixel form.
        (fragment(vec![center.clone()], vec![], true), Center),
        (fragment(vec![], vec![B::FragCoord], true), Recompile),
        (fragment(vec![], vec![B::SampleMaskIn], true), Coverage),
        // Side effects with `SampleId`: once per covered sample in both.
        (
            fragment(
                vec![sample.clone(), flat.clone()],
                vec![B::FrontFacing, B::SampleId, B::SampleMaskOut],
                true,
            ),
            Links,
        ),
        // Sample rate beside a pixel read: a center varying needs its pixel form; the others read
        // where Vulkan evaluates them, which the compile layer notes.
        (
            fragment(vec![center.clone()], vec![B::SampleId], false),
            Center,
        ),
        (
            fragment(vec![centroid.clone()], vec![B::SampleId], false),
            Links,
        ),
        (
            fragment(vec![sample.clone(), center.clone()], vec![], false),
            Center,
        ),
        (
            fragment(vec![sample.clone(), centroid.clone()], vec![], false),
            Links,
        ),
        (
            fragment(vec![], vec![B::FragCoord, B::SampleId], false),
            Recompile,
        ),
        (
            fragment(vec![], vec![B::PointCoord, B::SampleId], false),
            Links,
        ),
        (
            fragment(vec![], vec![B::SampleId, B::SampleMaskIn], false),
            Coverage,
        ),
        (fragment(vec![], vec![B::SampleId, bary], false), Links),
        (
            fragment(vec![sample.clone()], vec![B::FragCoord], false),
            Recompile,
        ),
        // Sample rate without one.
        (
            fragment(vec![sample.clone(), flat], vec![B::SampleId], false),
            Links,
        ),
        (
            fragment(vec![sample.clone()], vec![B::SampleMaskOut], false),
            Links,
        ),
        // Compiled for the pipeline: the pixel forms link, and so do the reads without one.
        (
            compiled(vec![], vec![B::FragCoord, B::SampleId], false, false),
            Links,
        ),
        (
            compiled(vec![], vec![B::FragCoord, B::SampleId], false, true),
            Links,
        ),
        (
            compiled(
                vec![center.clone(), sample.clone()],
                vec![B::FragCoord],
                true,
                false,
            ),
            Links,
        ),
        (
            compiled(vec![center.clone()], vec![B::SampleId], false, false),
            Center,
        ),
        (
            compiled(vec![centroid.clone()], vec![B::SampleId], true, false),
            Links,
        ),
        (
            compiled(vec![], vec![B::PointCoord, B::SampleId], true, false),
            Links,
        ),
        (
            compiled(vec![sample], vec![B::SampleMaskIn], true, false),
            Coverage,
        ),
        // Side effects beside the pixel forms, and beside a center varying without its form.
        (
            {
                let mut a = compiled(vec![center.clone()], vec![B::FragCoord], true, false);
                a.interface.writes_memory = true;
                a
            },
            Links,
        ),
        (
            {
                let mut a = compiled(vec![center.clone()], vec![B::FragCoord], false, false);
                a.interface.writes_memory = true;
                a
            },
            Center,
        ),
    ];
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let at = |samples: u32| {
        let mut r = r.clone();
        r.raster_sample_count = samples;
        LinkInputs::Render(Box::new(r))
    };
    let refused = Error::Unsupported(Unsupported::not_implemented(Feature::SampleRateShading));
    let coverage = Error::Unsupported(Unsupported::not_representable(
        Feature::PixelCoverageAtSampleRate,
    ));
    let recompile = Error::NeedsInput(RuntimeInput::Multisampled);
    // The portability profile as another device, from its committed description changed by
    // `edit`.
    let mvk_as = |edit: &dyn Fn(&mut crate::device::DeviceDescription)| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
        let mut d = crate::device::DeviceDescription::from_khronos_json(
            &std::fs::read_to_string(path).expect("profile"),
            None,
        )
        .expect("description");
        edit(&mut d);
        crate::device::Device::new(d, DevicePolicy::default())
            .expect("device")
            .profile()
            .clone()
    };
    // A device whose `InterpolateAtOffset` reads from the center: a driver other than
    // MoltenVK's.
    let centered_mvk = mvk_as(&|d| d.driver_id = None);
    // MoltenVK 1.4.0, whose `InterpolateAtOffset` reads off the center and which is not measured to
    // read center varyings at the pixel center itself: it refuses a per-sample function's center
    // varying naming the property it lacks; every other case links or is refused as on the same
    // device without the deviation.
    let unmeasured_mvk =
        mvk_as(&|d| d.driver_version = crate::device::driver_facts::moltenvk(1, 4, 0));
    let needs_centered = Error::Unsupported(Unsupported::needs(
        Feature::SampleRateShading,
        Capability::Property(VkProperty::InterpolateAtOffsetCentered),
    ));
    let committed = profile("moltenvk-m3");
    assert!(committed
        .fields()
        .enabled
        .properties
        .contains(&VkProperty::MetalCenterVaryings));
    for (stem, p, biased) in [
        ("moltenvk-m3", committed, false),
        ("moltenvk-m3 at 1.4.0", unmeasured_mvk, true),
        ("moltenvk-m3 centered", centered_mvk, false),
        ("lavapipe", profile("lavapipe"), false),
    ] {
        for (i, (fa, expect)) in cases.iter().enumerate() {
            let one =
                link(&[&va, fa], &at(1), &p).unwrap_or_else(|e| panic!("{stem} case {i}: {e}"));
            assert!(
                !one.graphics.expect("graphics").sample_shading,
                "{stem} case {i}"
            );
            let four = link(&[&va, fa], &at(4), &p).map(|l| {
                let g = l.graphics.expect("graphics");
                assert_eq!(
                    g.sample_shading, fa.interface.writes_memory,
                    "{stem} case {i}"
                );
                // Sample shading needs `sampleRateShading`.
                assert_eq!(
                    g.multisample_features(),
                    if g.sample_shading {
                        BTreeSet::from([VkFeature::SampleRateShading])
                    } else {
                        BTreeSet::new()
                    },
                    "{stem} case {i}"
                );
            });
            // lavapipe splits a clipped triangle's pixel-rate fragments, which a read of the
            // coverage sees (`split_pixel_fragments_are_refused_where_measured`).
            let split = DriverWorkaround::ClippedTrianglesSplitPixelFragments;
            let want = match expect {
                Links
                    if p.fields().workarounds.contains(&split)
                        && fa.interface.io.builtins.contains(&B::SampleMaskIn) =>
                {
                    Err(Error::Unsupported(Unsupported::driver_defect(
                        Feature::PixelCoverage,
                        split,
                    )))
                }
                // ... and a centroid varying of a function that sample shading may leave at pixel
                // rate (only a `Sample` input makes it per sample); one that surely runs per sample
                // reads it at its sample, which no split moves.
                Links
                    if p.fields().workarounds.contains(&split)
                        && fa.interface.io.inputs.contains(&centroid)
                        && !fa.interface.io.builtins.contains(&B::SampleId)
                        && !fa.interface.writes_memory =>
                {
                    Err(Error::Unsupported(Unsupported::driver_defect(
                        Feature::CentroidVarying,
                        split,
                    )))
                }
                Links => Ok(()),
                Coverage => Err(coverage.clone()),
                Recompile => Err(recompile.clone()),
                Center if biased => Err(needs_centered.clone()),
                // Compiled for the pipeline, the function would read them at the pixel.
                Center if !fa.interface.io.multisampled => Err(recompile.clone()),
                Center => Err(refused.clone()),
            };
            assert_eq!(four, want, "{stem} case {i} at 4 samples");
        }
    }
    let p = profile("vk12-floor");
    assert!(!p.fields().fragment.sample_rate_shading);
    assert_eq!(
        link(&[&va, &cases[1].0], &at(4), &p).unwrap_err(),
        Error::Unsupported(Unsupported::needs(
            Feature::SampleRateShading,
            Capability::Feature(VkFeature::SampleRateShading)
        ))
    );
    link(&[&va, &cases[1].0], &at(1), &p).expect("one sample");
    assert_eq!(cases[12].1, Recompile);
    assert_eq!(
        link(&[&va, &cases[12].0], &at(4), &p).unwrap_err(),
        Error::Unsupported(Unsupported::needs(
            Feature::SampleRateShading,
            Capability::Feature(VkFeature::SampleRateShading)
        ))
    );
    // The pixel's coverage at sample rate is no device's to gain: a device without
    // `sampleRateShading` refuses it as not representable too, not as the feature it lacks.
    let covering: Vec<_> = cases.iter().filter(|(_, at4)| *at4 == Coverage).collect();
    assert_eq!(covering.len(), 3);
    for (fa, _) in covering {
        assert_eq!(link(&[&va, fa], &at(4), &p).unwrap_err(), coverage);
    }
}

/// Lavapipe rasterizes a triangle it clips as several, so at more than one sample a pixel-rate
/// fragment runs once per piece in a pixel on their shared edge, with that piece's coverage.
/// Measured with a bare Vulkan program: `SampleMask` holds part of the pixel's samples and a
/// `Centroid` input moves to the first covered sample; a center input
/// stays at the pixel center. So every committed profile whose driver is lavapipe, and only those,
/// carries the defect, and link refuses there exactly a multisampled pixel-rate function that reads
/// `[[sample_mask]]` or a centroid varying, naming the defect; at one sample, and with a center or
/// flat varying alone, it links.
#[test]
fn split_pixel_fragments_are_refused_where_measured() {
    use crate::device::DriverWorkaround;
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::stage_io::StageBuiltin as B;
    let defect = DriverWorkaround::ClippedTrianglesSplitPixelFragments;
    let f4 = io_ty(IoNumeric::Float, 4);
    let smooth = |sampling| {
        Some(Interpolation::Smooth {
            sampling,
            perspective: true,
        })
    };
    let mut vs = iface(Stage::Vertex, vec![], vec![]);
    vs.io = stage_io(
        vec![],
        ["c", "k", "f"]
            .iter()
            .map(|n| varying(n, f4, None))
            .collect(),
    );
    let va = artifact(vs, None);
    let fragment = |inputs: Vec<IoVariable>, builtins: Vec<B>| {
        let mut fs = iface(Stage::Fragment, vec![], vec![]);
        fs.io = stage_io(inputs, vec![color(0, 0, f4)]);
        fs.io.builtins = builtins;
        artifact(fs, None)
    };
    let center = varying("c", f4, smooth(Sampling::Center));
    let centroid = varying("k", f4, smooth(Sampling::Centroid));
    let flat = varying("f", f4, Some(Interpolation::Flat));
    // Each function, and the feature a split device refuses at 4 samples.
    let cases = [
        (
            fragment(vec![], vec![B::SampleMaskIn]),
            Some(Feature::PixelCoverage),
        ),
        (
            fragment(vec![centroid.clone()], vec![]),
            Some(Feature::CentroidVarying),
        ),
        (
            fragment(vec![centroid, center.clone()], vec![B::SampleMaskIn]),
            Some(Feature::PixelCoverage),
        ),
        (
            fragment(vec![center, flat], vec![B::FragCoord, B::SampleMaskOut]),
            None,
        ),
    ];
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let at = |samples: u32| {
        let mut r = r.clone();
        r.raster_sample_count = samples;
        LinkInputs::Render(Box::new(r))
    };
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    let mut split = Vec::new();
    for (stem, p) in &profiles {
        let has = p.fields().workarounds.contains(&defect);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let description = crate::device::DeviceDescription::from_khronos_json(
            &std::fs::read_to_string(path).expect("profile"),
            None,
        )
        .expect("description");
        assert_eq!(
            has,
            description.driver_id == Some(vk::DriverId::MESA_LLVMPIPE),
            "{stem}"
        );
        if has {
            split.push(stem.as_str());
        }
        for (i, (fa, refused)) in cases.iter().enumerate() {
            link(&[&va, fa], &at(1), p).unwrap_or_else(|e| panic!("{stem} case {i}: {e}"));
            let four = link(&[&va, fa], &at(4), p).map(|_| ());
            let want = match refused {
                Some(feature) if has => Err(Error::Unsupported(Unsupported::driver_defect(
                    feature.clone(),
                    defect,
                ))),
                _ => Ok(()),
            };
            assert_eq!(four, want, "{stem} case {i} at 4 samples");
        }
    }
    split.sort_unstable();
    assert_eq!(split, ["gate-lavapipe", "gate-lavapipe-x86_64", "lavapipe"]);
}

/// The graphics state states whether the fragment function reads `[[barycentric_coord]]`, from the
/// fragment stage's builtins: the perspective-correct weights do, the no-perspective ones do, and
/// both together; a function that reads neither (a flat varying alone, its position, or nothing)
/// does not, nor a pipeline without a fragment function.
#[test]
fn the_graphics_state_states_whether_the_fragment_function_reads_barycentric_coordinates() {
    use crate::pipeline::stage_io::StageBuiltin;
    let p = profile("lavapipe");
    let inputs = LinkInputs::Render(Box::default());
    let f4 = io_ty(IoNumeric::Float, 4);
    let stage = |stage: Stage, io: StageIo| {
        let mut i = iface(stage, vec![], vec![]);
        i.io = io;
        artifact(i, None)
    };
    let reads = |va: &FunctionArtifact, fa: Option<&FunctionArtifact>| {
        let stages: Vec<&FunctionArtifact> = std::iter::once(va).chain(fa).collect();
        link(&stages, &inputs, &p)
            .unwrap()
            .graphics
            .expect("graphics")
            .reads_barycentric_coord
    };
    let va = stage(
        Stage::Vertex,
        stage_io(vec![], vec![varying("v", f4, None)]),
    );
    let with = |builtins: Vec<StageBuiltin>| StageIo {
        builtins,
        ..stage_io(vec![varying("v", f4, Some(Interpolation::Flat))], vec![])
    };
    let perspective = StageBuiltin::BaryCoord { perspective: true };
    let linear = StageBuiltin::BaryCoord { perspective: false };
    let cases = [
        ("perspective", vec![perspective], true),
        ("no perspective", vec![linear], true),
        ("both", vec![linear, perspective], true),
        ("a flat varying alone", vec![], false),
        ("its position", vec![StageBuiltin::FragCoord], false),
    ];
    for (name, builtins, want) in cases {
        let fa = stage(Stage::Fragment, with(builtins));
        assert_eq!(reads(&va, Some(&fa)), want, "{name}");
    }
    let none = stage(Stage::Fragment, StageIo::default());
    assert!(!reads(&va, Some(&none)), "no inputs");
    assert!(!reads(&va, None), "no fragment function");
}

/// The graphics state states whether the fragment function may read its position's z or w lanes,
/// from the fragment stage's `FragCoord` builtin and `position_depth`: the position read for its
/// depth does; read for its x and y alone, it does not, nor a function that reads no position, nor
/// a pipeline without a fragment function.
#[test]
fn the_graphics_state_states_whether_the_fragment_function_reads_its_position_s_depth() {
    use crate::pipeline::stage_io::StageBuiltin;
    let p = profile("lavapipe");
    let inputs = LinkInputs::Render(Box::default());
    let f4 = io_ty(IoNumeric::Float, 4);
    let stage = |stage: Stage, io: StageIo| {
        let mut i = iface(stage, vec![], vec![]);
        i.io = io;
        artifact(i, None)
    };
    let reads = |va: &FunctionArtifact, fa: Option<&FunctionArtifact>| {
        let stages: Vec<&FunctionArtifact> = std::iter::once(va).chain(fa).collect();
        link(&stages, &inputs, &p)
            .unwrap()
            .graphics
            .expect("graphics")
            .reads_position_depth
    };
    let va = stage(
        Stage::Vertex,
        stage_io(vec![], vec![varying("v", f4, None)]),
    );
    let with = |builtins: Vec<StageBuiltin>, position_depth: bool| StageIo {
        builtins,
        position_depth,
        ..stage_io(vec![varying("v", f4, Some(Interpolation::Flat))], vec![])
    };
    let cases = [
        (
            "its position's depth",
            vec![StageBuiltin::FragCoord],
            true,
            true,
        ),
        (
            "its position's x and y",
            vec![StageBuiltin::FragCoord],
            false,
            false,
        ),
        ("no position", vec![], false, false),
    ];
    for (name, builtins, position_depth, want) in cases {
        let fa = stage(Stage::Fragment, with(builtins, position_depth));
        assert_eq!(reads(&va, Some(&fa)), want, "{name}");
    }
    assert!(!reads(&va, None), "no fragment function");
}

/// The graphics state states whether the fragment function writes memory, from the fragment stage's
/// interface: a function that writes does, one that does not, or a pipeline without a fragment
/// function, does not; the vertex function's writes are its own fact.
#[test]
fn the_graphics_state_states_whether_the_fragment_function_writes_memory() {
    let p = profile("lavapipe");
    let inputs = LinkInputs::Render(Box::default());
    let writes = |vertex: bool, fragment: Option<bool>| {
        let mut vs = iface(Stage::Vertex, vec![], vec![]);
        vs.writes_memory = vertex;
        let va = artifact(vs, None);
        let fa = fragment.map(|writes| {
            let mut fs = iface(Stage::Fragment, vec![], vec![]);
            fs.writes_memory = writes;
            artifact(fs, None)
        });
        let stages: Vec<&FunctionArtifact> = std::iter::once(&va).chain(fa.as_ref()).collect();
        let graphics = link(&stages, &inputs, &p)
            .unwrap()
            .graphics
            .expect("graphics");
        (
            graphics.fragment_writes_memory,
            graphics.vertex_writes_memory,
        )
    };
    assert_eq!(
        writes(false, Some(true)),
        (true, false),
        "a fragment function that writes"
    );
    assert_eq!(
        writes(true, Some(false)),
        (false, true),
        "one that does not"
    );
    assert_eq!(writes(true, None), (false, true), "no fragment function");
}

/// The graphics state masks the color attachments the fragment function writes, from its stage's
/// outputs: each that a first (index 0) output names, whatever the order of the outputs. It
/// leaves out an attachment that only a dual-source second output names, and one the function
/// has no output for, though the pipeline has a format there. A pipeline without a fragment
/// function masks none.
#[test]
fn the_graphics_state_masks_the_colors_the_fragment_function_writes() {
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    let p = profile("lavapipe");
    let f4 = io_ty(IoNumeric::Float, 4);
    let mut r = RenderLinkInputs::new();
    for index in 0..4 {
        r.color_attachments[index] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    }
    let inputs = LinkInputs::Render(Box::new(r));
    let va = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let outputs = |fragment: Option<Vec<IoVariable>>| {
        let fa = fragment.map(|outputs| io_stage(Stage::Fragment, stage_io(vec![], outputs)));
        let stages: Vec<&FunctionArtifact> = std::iter::once(&va).chain(fa.as_ref()).collect();
        link(&stages, &inputs, &p)
            .unwrap()
            .graphics
            .expect("graphics")
            .color_outputs
    };
    assert_eq!(
        outputs(Some(vec![
            color(3, 0, f4),
            color(0, 0, f4),
            color(1, 1, f4)
        ])),
        0b1001,
        "first outputs at 3 and 0, a second output alone at 1"
    );
    assert_eq!(outputs(Some(vec![])), 0, "no output");
    assert_eq!(outputs(None), 0, "no fragment function");
}

/// The graphics state states whether the fragment function reads a flat input, from the fragment
/// stage's inputs: a flat varying does; a smooth one, perspective-correct or not, does not, nor a
/// function of no inputs, nor a pipeline without a fragment function; the render target array index
/// the vertex function writes, read as the `Layer` builtin, does.
#[test]
fn the_graphics_state_states_whether_the_fragment_function_reads_a_flat_input() {
    use crate::pipeline::stage_io::StageBuiltin;
    let p = profile("lavapipe");
    let inputs = LinkInputs::Render(Box::default());
    let f4 = io_ty(IoNumeric::Float, 4);
    let stage = |stage: Stage, io: StageIo| {
        let mut i = iface(stage, vec![], vec![]);
        i.io = io;
        artifact(i, None)
    };
    let flat = |va: &FunctionArtifact, fa: Option<&FunctionArtifact>| {
        let stages: Vec<&FunctionArtifact> = std::iter::once(va).chain(fa).collect();
        link(&stages, &inputs, &p)
            .unwrap()
            .graphics
            .expect("graphics")
            .flat_fragment_inputs
    };
    let va = stage(
        Stage::Vertex,
        stage_io(vec![], vec![varying("v", f4, None)]),
    );
    let smooth = |perspective| {
        Some(Interpolation::Smooth {
            sampling: Sampling::Center,
            perspective,
        })
    };
    let cases = [
        ("flat", Some(Interpolation::Flat), true),
        ("perspective", smooth(true), false),
        ("no perspective", smooth(false), false),
    ];
    for (name, interpolation, want) in cases {
        let io = stage_io(vec![varying("v", f4, interpolation)], vec![]);
        assert_eq!(flat(&va, Some(&stage(Stage::Fragment, io))), want, "{name}");
    }
    let none = stage(Stage::Fragment, StageIo::default());
    assert!(!flat(&va, Some(&none)), "no inputs");
    assert!(!flat(&va, None), "no fragment function");
    let layered = |io: StageIo| StageIo {
        builtins: vec![StageBuiltin::Layer],
        ..io
    };
    let vl = stage(
        Stage::Vertex,
        layered(stage_io(vec![], vec![varying("v", f4, None)])),
    );
    let fl = stage(Stage::Fragment, layered(StageIo::default()));
    assert!(flat(&vl, Some(&fl)), "the layer builtin");
    assert!(
        !flat(&vl, Some(&none)),
        "a layer the fragment function does not read"
    );
}

/// Refusal `u` tied to render pipeline attachment `attachment`, as link refuses that
/// attachment's state.
fn tied(u: Unsupported, attachment: Attachment) -> Error {
    Error::Unsupported(u.at(Location::render_pipeline().with_attachment(attachment)))
}
