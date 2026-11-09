//! Metal's depth and stencil resolves no Vulkan resolve mode makes, drawn after the pass.
//!
//! Metal resolves a `DepthResolvedSample` stencil to the stencil of the sample whose depth the
//! depth filter took: of equal depths, the last sample after `Min` and the first after `Max`
//! (a probe, an M3 Max and an M4 alike). No Vulkan resolve mode states it. And a device may have no
//! `MIN` or `MAX` depth resolve mode (lavapipe). `resolve.rs` marks each such aspect
//! `ResolveMode::Picked`: the pass's subpass resolves the other aspects (the depth by `MIN` or
//! `MAX` where the device has the mode, the picked ones `NONE`), and after the pass these draws
//! write the picked aspects into the same resolve attachment (the resolve texture's level and
//! slice, or the scratch image copied there after them), or into one of their own where the
//! subpass resolves nothing:
//!
//! - **The draws**: a render pass of the resolve attachment alone, one invocation per pixel, each
//!   reading every sample's depth (and stencil) of the multisampled attachment through sampled
//!   views and taking Metal's sample by depth ([`DepthPick`]), of the draws of
//!   `bind::depth_stencil_resolve_modules`. A picked depth is [`ResolveWrite::Depth`]'s draw, its
//!   `FragDepth` written under compare `ALWAYS` with the stencil test off: only where
//!   `depth_copy::copies_exactly` holds (a 32-bit float depth, on a driver whose fragment depth
//!   stores every value unchanged, `DriverFact::FragmentDepthStoresUnchanged`, or a 16-bit one
//!   whose codes it keeps, `DriverFact::FragmentDepthKeepsDepth16Codes`), which
//!   `resolve.rs` checks before it picks; elsewhere the device's missing mode stays refused. A
//!   picked stencil is the eight bit-plane draws ([`ResolveWrite::StencilPlane`]) over the
//!   stencil cleared to 0 over the render area: each survives where its sample's stencil has its
//!   bit, which the draw sets under write mask [`StencilBit::mask`], reference 0xFF, compare
//!   `ALWAYS` and pass op `REPLACE`, the depth test off; exact on every device, stencil export or
//!   not. The pass loads and stores the depth (drawn first where it is picked), and loads and
//!   stores the stencil where only the depth is drawn.
//! - **The samples**: the pass's last instance stores the aspects the draws read
//!   (`render_target::final_pass`), whatever Metal's store action, and the attachment's image
//!   must be sampled (`SAMPLED` usage, which a texture made for shader reads has, and a
//!   multisampled depth/stencil render target the runtime makes takes where it costs nothing);
//!   elsewhere the pass is refused `NotImplemented(DepthStencilResolve)` when the encoder opens.
//!   A layered pass's draws read its first layer alone, which Metal resolves alone (a probe;
//!   `resolve.rs`).
//!
//! The shaders and each pick, sample count, resolve format and set of drawn aspects' render pass
//! and pipelines are made once per device, at first use (when a pass that needs them opens), and
//! destroyed with the device; a resolve's descriptor set and framebuffer are its own, held by its
//! command buffer until it completes. The draws run between barriers from everything before and to
//! everything after.

use super::device::{device_error, lock};
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::release;
use super::render_pass::{PassLayout, Target};
use super::resource::ResourceKey;
use super::sample_planes::{pipeline_layout, set_layout, shader_module};
use super::stencil_copy::{
    aspect_view, attachment_use, draw_framebuffer, record_draws, sampled, sampled_use, PassObjects,
    DRAW_STAGES,
};
use super::texture::image_aspects;
use super::tracker::{Barrier, Use};
use super::Device;
use crate::shader_cache::driver::Making;
use ash::vk;
use metal2vulkan::bind::{
    depth_stencil_resolve_modules, DepthPick, DepthStencilResolveDraw, DepthStencilResolveModules,
    ResolveWrite, StencilBit,
};
use metal2vulkan::cache::Hash256;
use metal2vulkan::error::Feature;
use metal2vulkan::vk::{Format, ImageAspectFlags};

/// The aspects the draws write (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Drawn {
    Depth,
    Stencil,
    Both,
}

impl Drawn {
    /// The draws writing `aspects`: `None` for neither aspect.
    pub(super) fn of(aspects: ImageAspectFlags) -> Option<Drawn> {
        match (
            aspects.contains(ImageAspectFlags::DEPTH),
            aspects.contains(ImageAspectFlags::STENCIL),
        ) {
            (true, false) => Some(Drawn::Depth),
            (false, true) => Some(Drawn::Stencil),
            (true, true) => Some(Drawn::Both),
            (false, false) => None,
        }
    }

    fn depth(self) -> bool {
        self != Drawn::Stencil
    }

    fn stencil(self) -> bool {
        self != Drawn::Depth
    }

    /// The draws, in recording order: the depth's, then each [`StencilBit`]'s plane.
    pub(super) fn writes(self) -> Vec<ResolveWrite> {
        let depth = self.depth().then_some(ResolveWrite::Depth);
        let planes = StencilBit::ALL
            .into_iter()
            .filter(|_| self.stencil())
            .map(ResolveWrite::StencilPlane);
        depth.into_iter().chain(planes).collect()
    }

    /// The descriptor types of the draws' set, in binding order (`SET`): the depth aspect's
    /// alone where the stencil is not drawn.
    pub(super) fn set(self) -> &'static [vk::DescriptorType] {
        match self.stencil() {
            true => &SET,
            false => &SET[..1],
        }
    }
}

/// The draws' shaders for one pick, sample count and set of drawn aspects, the device's.
#[derive(Clone, Debug)]
struct Shaders {
    vertex: vk::ShaderModule,
    /// One per draw, in `Drawn::writes`' order.
    fragments: Vec<vk::ShaderModule>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    /// The digest of its modules' SPIR-V words, which its pipelines' records are named by
    /// (`shader_cache/driver.rs`).
    words: Hash256,
}

/// The draws' render pass and pipelines for one pick, sample count, resolve format and set of
/// drawn aspects.
#[derive(Clone, Debug)]
struct Pipelines {
    render_pass: vk::RenderPass,
    /// One per draw, in `Drawn::writes`' order.
    pipelines: Vec<vk::Pipeline>,
}

/// The resolves' objects one device has made (the module docs): bounded by the two picks, the
/// sample counts, the depth/stencil formats and the three sets of drawn aspects.
#[derive(Default)]
pub(super) struct DepthStencilResolveObjects {
    shaders: Vec<((DepthPick, u32, Drawn), Shaders)>,
    pipelines: Vec<((DepthPick, u32, Format, Drawn), Pipelines)>,
}

impl DepthStencilResolveObjects {
    /// Destroy every object; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        // SAFETY: the device is idle, so no pending command uses the objects; destroying a null
        // handle is a no-op.
        unsafe {
            for (_, made) in &self.pipelines {
                destroy_pipelines(raw, made);
            }
            for (_, shaders) in &self.shaders {
                destroy_shaders(raw, shaders);
            }
        }
    }
}

/// Destroy `shaders`' objects.
///
/// # Safety
///
/// No pending command uses them; a null handle is a no-op.
unsafe fn destroy_shaders(raw: &ash::Device, shaders: &Shaders) {
    // SAFETY: the caller's contract.
    unsafe {
        release::object(raw, shaders.layout);
        release::object(raw, shaders.set_layout);
        for &fragment in &shaders.fragments {
            release::object(raw, fragment);
        }
        release::object(raw, shaders.vertex);
    }
}

/// Destroy `made`'s objects.
///
/// # Safety
///
/// As for `destroy_shaders`.
unsafe fn destroy_pipelines(raw: &ash::Device, made: &Pipelines) {
    // SAFETY: the caller's contract.
    unsafe {
        for &pipeline in &made.pipelines {
            release::object(raw, pipeline);
        }
        release::object(raw, made.render_pass);
    }
}

/// The descriptor types of the draws' set, in binding order: the depth aspect, then the stencil
/// aspect, each sampled.
const SET: [vk::DescriptorType; 2] = [vk::DescriptorType::SAMPLED_IMAGE; 2];

impl Device {
    /// The `drawn` draws' shaders taking `pick` of `samples` samples, and their pipelines for a
    /// resolve attachment of `format`, made on first use; refused (`Unsupported`, naming
    /// `Feature::Format`) where the device lacks what they need.
    fn resolve_draws(
        &self,
        pick: DepthPick,
        drawn: Drawn,
        samples: u32,
        format: Format,
    ) -> Result<(Shaders, Pipelines), Error> {
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn = crate::pipeline::budget::helper_turn(
            &self.shared().depth_stencil_resolves,
            |objects| {
                objects
                    .pipelines
                    .iter()
                    .any(|(key, _)| *key == (pick, samples, format, drawn))
            },
        );
        let mut objects = lock(&self.shared().depth_stencil_resolves);
        let shared = self.shared();
        let raw = shared.vk();
        let made_shaders = objects
            .shaders
            .iter()
            .find(|(key, _)| *key == (pick, samples, drawn))
            .map(|(_, shaders)| shaders.clone());
        let shaders = match made_shaders {
            Some(shaders) => shaders,
            None => {
                if DepthStencilResolveModules::SET != 0
                    || DepthStencilResolveModules::DEPTH_BINDING != 0
                    || DepthStencilResolveModules::STENCIL_BINDING != 1
                {
                    return Err(internal(
                        "the depth/stencil resolve's bindings out of their set layout's order",
                    ));
                }
                let writes = drawn.writes();
                let mut modules = Vec::with_capacity(writes.len());
                for &write in &writes {
                    let draw = DepthStencilResolveDraw::new(pick, samples, write);
                    modules.push(
                        depth_stencil_resolve_modules(self.compile_device(), draw, Feature::Format)
                            .map_err(device_error)?,
                    );
                }
                let vertex_words = modules
                    .first()
                    .map(|m| m.vertex.clone())
                    .ok_or_else(|| internal("a resolve of no draw"))?;
                let words = modules
                    .iter()
                    .fold(
                        Making::new("depth/stencil resolve modules").words(&vertex_words),
                        |making, module| making.words(&module.fragment),
                    )
                    .finish();
                let mut shaders = Shaders {
                    vertex: vk::ShaderModule::null(),
                    fragments: Vec::with_capacity(modules.len()),
                    set_layout: vk::DescriptorSetLayout::null(),
                    layout: vk::PipelineLayout::null(),
                    words,
                };
                let made = (|| {
                    shaders.vertex = shader_module(raw, &vertex_words)?;
                    for module in &modules {
                        shaders
                            .fragments
                            .push(shader_module(raw, &module.fragment)?);
                    }
                    shaders.set_layout =
                        set_layout(raw, drawn.set(), vk::ShaderStageFlags::FRAGMENT)?;
                    shaders.layout = pipeline_layout(raw, shaders.set_layout)?;
                    Ok(())
                })();
                if let Err(error) = made {
                    // SAFETY: nothing used the objects.
                    unsafe { destroy_shaders(raw, &shaders) };
                    return Err(error);
                }
                objects
                    .shaders
                    .push(((pick, samples, drawn), shaders.clone()));
                shaders
            }
        };
        if let Some((_, made)) = objects
            .pipelines
            .iter()
            .find(|(key, _)| *key == (pick, samples, format, drawn))
        {
            return Ok((shaders, made.clone()));
        }
        let mut made = Pipelines {
            render_pass: vk::RenderPass::null(),
            pipelines: Vec::with_capacity(shaders.fragments.len()),
        };
        let result = (|| {
            // The stencil drawn is cleared first; a depth drawn alone keeps the stencil.
            made.render_pass = match drawn.stencil() {
                true => super::stencil_copy::planes_pass(raw, format, 1)?,
                false => super::depth_copy::depth_pass(raw, format, 1)?,
            };
            for (&fragment, write) in shaders.fragments.iter().zip(drawn.writes()) {
                let making = Making::new("depth/stencil resolve")
                    .digest(&shaders.words)
                    .text(&(pick, samples, format, drawn, write));
                let pipeline =
                    draw_pipeline(self, &shaders, fragment, made.render_pass, (write, making))?;
                made.pipelines.push(pipeline);
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                objects
                    .pipelines
                    .push(((pick, samples, format, drawn), made.clone()));
                Ok((shaders, made))
            }
            Err(error) => {
                // SAFETY: nothing used the objects.
                unsafe { destroy_pipelines(raw, &made) };
                Err(error)
            }
        }
    }
}

/// The draw of `write`, whose fragment stage is `fragment`, in `render_pass`: the triangle over
/// the viewport at one sample; for the depth, the depth written under compare `ALWAYS` with the
/// stencil test off; for a stencil plane, the depth test off and the stencil set to the reference
/// where the fragment lives, under the bit's write mask.
/// Its record is the one `making` names.
fn draw_pipeline(
    device: &Device,
    shaders: &Shaders,
    fragment: vk::ShaderModule,
    render_pass: vk::RenderPass,
    (write, making): (ResolveWrite, Making),
) -> Result<vk::Pipeline, Error> {
    let shared = device.shared();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shaders.vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = match write {
        ResolveWrite::Depth => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::ALWAYS)
            .stencil_test_enable(false),
        ResolveWrite::StencilPlane(bit) => {
            let face = vk::StencilOpState::default()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::REPLACE)
                .depth_fail_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .compare_mask(0xff)
                .write_mask(bit.mask())
                .reference(0xff);
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(false)
                .depth_write_enable(false)
                .stencil_test_enable(true)
                .front(face)
                .back(face)
        }
        _ => return Err(internal("a resolve draw the runtime does not record")),
    };
    let blend = vk::PipelineColorBlendStateCreateInfo::default();
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(shaders.layout)
        .render_pass(render_pass)
        .subpass(0);
    shared.make_graphics_pipeline(making, info)
}

/// Whether the draws can resolve `aspects` of `source` by `pick` into `into`, and their objects
/// made (the module docs): `source` a multisampled attachment of one layer whose image is
/// sampled, with a depth and each drawn aspect, and `into` a single-sampled one with each drawn
/// aspect. Refused as the module docs say, or `Unsupported` where the device lacks what the draws
/// need.
pub(super) fn draws_run(
    device: &Device,
    source: &Target,
    into: &Target,
    pick: DepthPick,
    aspects: ImageAspectFlags,
) -> Result<(), Error> {
    let drawn = Drawn::of(aspects).ok_or_else(|| internal("a drawn resolve of no aspect"))?;
    if !image_aspects(source.format).contains(aspects | ImageAspectFlags::DEPTH)
        || !image_aspects(into.format).contains(aspects)
        || into.samples != 1
    {
        return Err(internal("a drawn resolve of other aspects or samples"));
    }
    if !sampled(source)? {
        return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
    }
    device
        .resolve_draws(pick, drawn, source.samples, into.format)
        .map(|_| ())
}

/// A pass's drawn depth/stencil resolve, planned before the pass records and recorded after it.
pub(super) struct DrawnResolve {
    objects: PassObjects,
    shaders: Shaders,
    made: Pipelines,
    set: vk::DescriptorSet,
    /// The render area, from the origin.
    area: [u32; 2],
    uses: Vec<Use<ResourceKey>>,
}

impl DrawnResolve {
    /// The draws of `layout`'s (its final layout) depth/stencil resolve whose aspects are picked
    /// by depth (`ResolveMode::Picked`), into its first step's resolve attachment or its own;
    /// `None` where it has none.
    pub(super) fn plan(
        device: &Device,
        layout: &PassLayout,
    ) -> Result<Option<DrawnResolve>, Error> {
        let Some(ds) = layout.depth_stencil.as_ref() else {
            return Ok(None);
        };
        let Some(resolve) = ds.resolve.as_ref() else {
            return Ok(None);
        };
        let Some((pick, aspects)) = resolve.picked() else {
            return Ok(None);
        };
        let into = &resolve
            .drawn_target()
            .ok_or_else(|| internal("a drawn resolve without its attachment"))?
            .attachment;
        let source = &ds.target;
        draws_run(device, source, into, pick, aspects)?;
        let drawn = Drawn::of(aspects).ok_or_else(|| internal("a drawn resolve of no aspect"))?;
        let (shaders, made) = device.resolve_draws(pick, drawn, source.samples, into.format)?;
        let mut views = vec![(SET[0], aspect_view(source, ImageAspectFlags::DEPTH)?)];
        if drawn.stencil() {
            views.push((SET[1], aspect_view(source, ImageAspectFlags::STENCIL)?));
        }
        let (mut objects, set) = PassObjects::with_set(device, shaders.set_layout, &views)?;
        draw_framebuffer(&mut objects, made.render_pass, into, layout.extent)?;
        Ok(Some(DrawnResolve {
            objects,
            shaders,
            made,
            set,
            area: layout.extent,
            uses: vec![sampled_use(source)?, attachment_use(into)?],
        }))
    }

    /// Record the draws in `core`'s command buffer after the pass, between barriers from
    /// everything before and to everything after, and hold their objects until the command
    /// buffer completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let DrawnResolve {
            objects,
            shaders,
            made,
            set,
            area,
            uses,
        } = self;
        let framebuffer = objects.framebuffer;
        core.barrier(Barrier::everything());
        core.command(DRAW_STAGES, &uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording outside a render pass; the
            // pipelines, framebuffer, set and the images they view are live until it completes
            // (the device keeps the pipelines, these objects the framebuffer and set, the encoder
            // the textures); every image rests in `GENERAL`.
            unsafe {
                record_draws(
                    device,
                    raw,
                    (made.render_pass, framebuffer),
                    (shaders.layout, set),
                    &made.pipelines,
                    area,
                );
            }
        });
        core.barrier(Barrier::everything());
        core.retain(objects);
    }
}
