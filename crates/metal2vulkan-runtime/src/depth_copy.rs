//! A multisampled depth copied without colour copies: where maintenance8's depth-stencil/colour
//! `vkCmdCopyImage` is missing or runs wrong
//! (`RuntimeDriverDefect::MultisampledDepthStencilColourCopies`), the separate textures' depth
//! moves between its texture and the scratch image (`depth_stencil_scratch.rs`) by a draw.
//!
//! - **The draw** ([`DepthDraw`]): a render pass of the destination alone, its depth and stencil
//!   loaded and stored, of one sample-rate draw of `bind::depth_stencil_copy_modules`'
//!   [`DepthStencilCopy::Depth`], which reads each sample of the source's depth aspect through a
//!   sampled view and writes it as the sample's `FragDepth`, under depth compare `ALWAYS` with
//!   depth writes on and the stencil test off.
//! - **Where it is exact** ([`copies_exactly`]): a 32-bit float depth, on a driver whose fragment
//!   depth stores every depth value unchanged: each value in `[0, 1]` it is given, and, for the
//!   rest, the value it stored when given it again.
//!   `runtime_lane::separate_samples::fragment_depth_stores_depth_values_unchanged` probes that
//!   with signed zero, denormals, 1, values outside `[0, 1]`, infinities and NaN payloads, and the
//!   driver fact [`DriverFact::FragmentDepthStoresUnchanged`] lists each driver where it held,
//!   from the release measured (`device/driver_facts.rs`). And a 16-bit unsigned normalized depth,
//!   on a driver whose fragment depth stores each code's fetched `float` as that code
//!   (`fragment_depth_keeps_every_16_bit_code` probes every code; the fact
//!   [`DriverFact::FragmentDepthKeepsDepth16Codes`]). Elsewhere a 32-bit float depth has no exact
//!   form on a device whose copies go by draws (Vulkan writes a multisampled depth image's
//!   samples by no other means there), so the draw runs as an announced near-equivalent
//!   (`drawn-depth-copy`, [`drawn`]): a blit copy into such a depth, and the separate textures'
//!   depth copies. Its reason names the values the driver is measured to change
//!   ([`fragment_depth_changes`]: MoltenVK, NVIDIA's), or, on a driver whose fragment depth is not
//!   measured (another driver, or a release of lavapipe not measured), those Vulkan leaves to the
//!   driver ([`UNMEASURED_FLOAT_DEPTH`]). A 16-bit depth is drawn only where its codes are
//!   measured kept; elsewhere its consumer refuses, typed.
//!
//! - **The bits draws** ([`DepthDraw::bits`]): the other way, a depth or stencil aspect's samples
//!   into a multisampled unsigned colour image of their bits (`R32_UINT`, `R16_UINT`, `R8_UINT`),
//!   for the multisampled blit copies between depth/stencil and colour textures
//!   (`sample_copy.rs`). A render pass of the colour image alone, of one sample-rate draw of
//!   [`DepthStencilCopy::DepthBits`], [`DepthStencilCopy::Depth16Bits`] or
//!   [`DepthStencilCopy::StencilBits`], which fetches each sample through a sampled view of the
//!   aspect and writes its bits (a 32-bit float depth's `f32` bit-cast, a 16-bit normalized
//!   depth's code, a stencil's value) as the sample's colour. Exact wherever the fetch returns
//!   the texel: the same fetch reads the samples back in every lane and in the
//!   `FragmentDepthStoresUnchanged` and `FragmentDepthKeepsDepth16Codes` probes above. The draw
//!   back reads its depth from such a colour image ([`DepthDraw::from_colour`],
//!   [`depth_carrier`]): a 32-bit float depth's floats from an `R32_SFLOAT` one, a 16-bit
//!   normalized depth's codes from the `R16_UINT` one.
//!
//!   A 16-bit depth crosses as its **codes** and never as floats, which is what makes both
//!   halves exact by construction rather than by a measured tolerance: the draw back's
//!   `code / 65535` is bit for bit the depth the `FragmentDepthKeepsDepth16Codes` probe stores,
//!   and the bits draw's `round(depth * 65535)` recovers every one of the 65536 codes
//!   (`bind::depth_stencil_copy`'s module docs carry the margin and the exhaustive check).
//!
//! - **The depth stand-in's copies** (`depth_stand_in.rs`): a pass's depth samples into an
//!   `R32_SFLOAT` stand-in, multisampled or sample planes, by a dispatch of
//!   `bind::depth_split_module` ([`DepthSplit`]), and back by the `FragDepth` draw of
//!   [`DepthStencilCopy::DepthFromStandIn`] ([`DepthDraw::from_stand_in`]), which reads the
//!   stand-in as a storage image; exact or announced as the draw above is.
//!
//! The draws read and write a render area anywhere in their attachment
//! (`stencil_copy::DrawArea`). The shaders of each draw and each draw, format and sample count's
//! render pass and pipeline are made once per device, at first use, and destroyed with the
//! device; a copy's descriptor set and framebuffer are its own, held by its command buffer until
//! it completes.

use super::binding_view::VkView;
use super::degradation::Degradation;
use super::device::{device_error, lock};
use super::error::{internal, Error};
use super::release;
use super::render_pass::Target;
use super::sample_planes::{pipeline_layout, set_layout, shader_module};
use super::stencil_copy::{aspect_view, draw_framebuffer, record_draws_in, DrawArea, PassObjects};
use super::texture::image_aspects;
use super::Device;
use crate::shader_cache::driver::Making;
use ash::vk;
use metal2vulkan::bind::{
    depth_split_module, depth_stencil_copy_modules, DepthSplitModule, DepthStandInForm,
    DepthStencilCopy, DepthStencilCopyModules,
};
use metal2vulkan::cache::Hash256;
use metal2vulkan::device::DriverFact;
use metal2vulkan::error::Feature;
use metal2vulkan::vk::{
    ComponentMapping, DriverId, Format, ImageAspectFlags, ImageUsageFlags, ImageViewType,
};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Whether `format`'s depth is a 32-bit float's, the one depth the `FragDepth` draw writes.
pub(super) fn float_depth(format: Format) -> bool {
    matches!(format, Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT)
}

/// Whether `format`'s depth is a 16-bit unsigned normalized one's.
pub(super) fn unorm16_depth(format: Format) -> bool {
    matches!(format, Format::D16_UNORM | Format::D16_UNORM_S8_UINT)
}

/// The colour format a multisampled colour image carries `format`'s depth in, between the bits
/// draw that reads the aspect and the draw that writes it back (the module docs): a 32-bit float
/// depth's own floats in `R32_SFLOAT`, and a 16-bit unsigned normalized depth's **codes** in
/// `R16_UINT`, which is what makes that pair exact by construction
/// (`bind::DepthStencilCopy::Depth16Code`). `None` for a depth no draw writes.
pub(super) fn depth_carrier(format: Format) -> Option<Format> {
    if float_depth(format) {
        return Some(Format::R32_SFLOAT);
    }
    unorm16_depth(format).then_some(Format::R16_UINT)
}

/// Whether the draw copies a depth aspect of `format` exactly on `device` (the module docs): a
/// 32-bit float depth, where the device's driver has
/// [`DriverFact::FragmentDepthStoresUnchanged`], and a 16-bit unsigned normalized one, where it
/// has [`DriverFact::FragmentDepthKeepsDepth16Codes`].
pub(super) fn copies_exactly(device: &Device, format: Format) -> bool {
    copies_exactly_by(&device.properties().driver_facts, format)
}

/// [`copies_exactly`] on a driver with `facts`.
fn copies_exactly_by(facts: &BTreeSet<DriverFact>, format: Format) -> bool {
    (float_depth(format) && facts.contains(&DriverFact::FragmentDepthStoresUnchanged))
        || (unorm16_depth(format) && facts.contains(&DriverFact::FragmentDepthKeepsDepth16Codes))
}

/// What the draw does to a depth aspect it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Drawn {
    /// Every value stored unchanged ([`copies_exactly`]).
    Exact,
    /// The values the draw may change, named: those the driver's fragment depth is measured to
    /// change ([`fragment_depth_changes`]), or those Vulkan leaves to a driver not measured.
    /// Metal's copy keeps them, so the copy is the announced near-equivalent `drawn-depth-copy`.
    Changes(&'static str),
}

/// What the draw does to a depth aspect of `format` on `device` ([`Drawn`]): exact where
/// [`copies_exactly`]; a 32-bit float depth on a driver measured to change some values changes
/// them, and on any other driver may change the values Vulkan leaves to it. `None` for any other
/// depth (a 16-bit one whose codes are not measured kept), where the copy is refused.
pub(super) fn drawn(device: &Device, format: Format) -> Option<Drawn> {
    let description = &device.properties().description;
    drawn_by(
        &device.properties().driver_facts,
        description.driver_id,
        format,
    )
}

/// [`drawn`] on a driver of `driver` with `facts`.
pub(super) fn drawn_by(
    facts: &BTreeSet<DriverFact>,
    driver: Option<DriverId>,
    format: Format,
) -> Option<Drawn> {
    if copies_exactly_by(facts, format) {
        return Some(Drawn::Exact);
    }
    float_depth(format)
        .then(|| Drawn::Changes(fragment_depth_changes(driver).unwrap_or(UNMEASURED_FLOAT_DEPTH)))
}

/// What the `FragDepth` draw may change of a 32-bit float depth on a driver whose fragment depth
/// is not measured: what Vulkan leaves to the driver. A fragment depth outside the depth range
/// `[0, 1]` is undefined (the fragment operations chapter: "If zf is not in the range
/// [zmin, zmax], then zf is undefined"), a denormal may be flushed to zero, and the probe found
/// -0 stored as +0 on a driver (MoltenVK); every other value is stored as the shader wrote it.
pub(super) const UNMEASURED_FLOAT_DEPTH: &str = "its driver's fragment depth is not measured: \
     Vulkan leaves undefined a value outside [0, 1] (past 1, below 0, an infinity or a NaN), \
     which it may store clamped, as a value of its own or not at all, and it may flush a \
     denormal to zero or store -0 as +0; every other value lands as Metal's copy leaves it";

/// What the `FragDepth` draw changes on drivers measured to store some depth values otherwise
/// (`runtime_lane::separate_samples::fragment_depth_stores_depth_values_unchanged`, signed zero,
/// denormals, 1, values outside `[0, 1]`, infinities and NaN payloads, each stored once and
/// again; `DriverFact::FragmentDepthStoresUnchanged`'s record): the driver, and the values it
/// changes. Where the multisampled copies go by draws, a copy into a 32-bit float depth runs
/// on these as the announced `drawn-depth-copy`, naming them; on a driver neither here nor exact
/// it runs announced too, naming what Vulkan leaves to the driver ([`UNMEASURED_FLOAT_DEPTH`]).
const FRAGMENT_DEPTH_CHANGES: &[(DriverId, &str)] = &[
    // MoltenVK 1.4.1 on an M3 Max and an M4 (the NaNs by our probe on an M4, which drew them
    // over a depth of 0.25: changed).
    (
        DriverId::MOLTENVK,
        "its driver's fragment depth changes values: a value with the sign bit set (-0, a \
         negative value, -inf, a negative NaN) is stored as +0, a value past 1 and +inf as 1, \
         and a positive NaN writes nothing, the sample keeping the depth it held",
    ),
    // NVIDIA's 615.71.09 on the RTX 3090, with `VK_EXT_depth_range_unrestricted`: values outside
    // `[0, 1]` are kept.
    (
        DriverId::NVIDIA_PROPRIETARY,
        "its driver's fragment depth changes values: denormals are flushed to zero of their \
         sign, and NaN is stored as -inf",
    ),
];

/// The values the `FragDepth` draw changes on a driver of `driver`, where they are measured
/// ([`FRAGMENT_DEPTH_CHANGES`]); `None` where they are not.
fn fragment_depth_changes(driver: Option<DriverId>) -> Option<&'static str> {
    let driver = driver?;
    FRAGMENT_DEPTH_CHANGES
        .iter()
        .find(|(of, _)| *of == driver)
        .map(|(_, changes)| *changes)
}

/// Announce `drawn-depth-copy` once for `what`, a copy whose `FragDepth` draw may change the
/// values `changes` names ([`Drawn::Changes`]): the one notice of every copy the draw makes
/// inexactly, a blit copy into a multisampled depth and a pass's separate depth texture's copies
/// to and from its scratch image.
pub(super) fn announce_changes(
    device: &Device,
    changes: &'static str,
    what: impl FnOnce() -> String,
) {
    device.degrade(Degradation::DrawnDepthCopy, || {
        (
            what(),
            format!(
                "the device copies a multisampled depth by a fragment-depth draw, and {changes}"
            ),
        )
    });
}

/// A draw's shaders, the device's.
#[derive(Clone, Copy, Debug)]
struct DepthShaders {
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    /// The digest of its modules' SPIR-V words, which its pipelines' records are named by
    /// (`shader_cache/driver.rs`).
    words: Hash256,
}

/// A draw's render pass and pipeline for one destination format and sample count.
#[derive(Clone, Copy, Debug)]
struct DepthPipeline {
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
}

/// The depth copies' objects one device has made (the module docs): each draw's shaders, and
/// its pipelines for each destination format and sample count.
#[derive(Default)]
pub(super) struct DepthCopyObjects {
    shaders: Vec<(DepthStencilCopy, DepthShaders)>,
    pipelines: Vec<((DepthStencilCopy, Format, u32), DepthPipeline)>,
    /// The depth split of each stand-in form.
    splits: Vec<(DepthStandInForm, SplitKernel)>,
}

/// The depth split's kernel for one stand-in form, the device's.
#[derive(Clone, Copy, Debug)]
struct SplitKernel {
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// The descriptor types of the depth split's set, in binding order.
const SPLIT_SET: [vk::DescriptorType; 2] = [
    vk::DescriptorType::SAMPLED_IMAGE,
    vk::DescriptorType::STORAGE_IMAGE,
];

/// The descriptor type of the source of the draw `copy`: the stand-in a
/// [`DepthStencilCopy::DepthFromStandIn`] reads is a storage image, every other source sampled.
fn draw_source(copy: DepthStencilCopy) -> vk::DescriptorType {
    match copy {
        DepthStencilCopy::DepthFromStandIn { .. } => vk::DescriptorType::STORAGE_IMAGE,
        _ => vk::DescriptorType::SAMPLED_IMAGE,
    }
}

impl DepthCopyObjects {
    /// Destroy every object; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        // SAFETY: the device is idle, so no pending command uses the objects; destroying a null
        // handle is a no-op.
        unsafe {
            for (_, made) in &self.pipelines {
                release::object(raw, made.pipeline);
                release::object(raw, made.render_pass);
            }
            for (_, shaders) in &self.shaders {
                release::object(raw, shaders.layout);
                release::object(raw, shaders.set_layout);
                release::object(raw, shaders.fragment);
                release::object(raw, shaders.vertex);
            }
            for (_, kernel) in &self.splits {
                release::object(raw, kernel.pipeline);
                release::object(raw, kernel.layout);
                release::object(raw, kernel.set_layout);
                release::object(raw, kernel.module);
            }
        }
    }
}

impl Device {
    /// The shaders of `copy` (the `FragDepth` draw or a bits draw) and its pipeline for a
    /// destination of `format` at `samples`, made on first use; refused naming `feature` where
    /// the device lacks what they need.
    fn depth_draw(
        &self,
        copy: DepthStencilCopy,
        format: Format,
        samples: u32,
        feature: &Feature,
    ) -> Result<(DepthShaders, DepthPipeline), Error> {
        self.depth_draw_by(copy, (format, samples), feature, true)
    }

    /// The depth stand-in's copies of `form` for a depth target of `format` at `samples`: the
    /// draw back from the stand-in and the depth split, made by a render pipeline state's
    /// creation (`depth_stand_in::make_copies`) so that its first tested draw makes neither.
    /// Each refusal is left to that draw, which meets it again.
    pub(super) fn make_stand_in_copies(
        &self,
        form: DepthStandInForm,
        (format, samples): (Format, u32),
        feature: &Feature,
    ) {
        let back = DepthStencilCopy::DepthFromStandIn { form };
        let _ = self.depth_draw_by(back, (format, samples), feature, false);
        let _ = self.depth_split(form, feature, false);
    }

    /// [`depth_draw`](Self::depth_draw), counting a stand-in's copy back made by a draw's
    /// encoding where `at_draw`, else by a state's creation (`WorkCounts::stand_in_copies_made`).
    fn depth_draw_by(
        &self,
        copy: DepthStencilCopy,
        (format, samples): (Format, u32),
        feature: &Feature,
        at_draw: bool,
    ) -> Result<(DepthShaders, DepthPipeline), Error> {
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn = crate::pipeline::budget::helper_turn(&self.shared().depth_copies, |objects| {
            objects
                .pipelines
                .iter()
                .any(|(key, _)| *key == (copy, format, samples))
        });
        let mut objects = lock(&self.shared().depth_copies);
        let shared = self.shared();
        let raw = shared.vk();
        let made_shaders = objects
            .shaders
            .iter()
            .find(|(of, _)| *of == copy)
            .map(|(_, shaders)| *shaders);
        let shaders = match made_shaders {
            Some(shaders) => shaders,
            None => {
                let modules =
                    depth_stencil_copy_modules(self.compile_device(), copy, feature.clone())
                        .map_err(device_error)?;
                if DepthStencilCopyModules::SET != 0 || DepthStencilCopyModules::SOURCE_BINDING != 0
                {
                    return Err(internal("the depth copy's binding out of its set layout"));
                }
                let mut shaders = DepthShaders {
                    vertex: vk::ShaderModule::null(),
                    fragment: vk::ShaderModule::null(),
                    set_layout: vk::DescriptorSetLayout::null(),
                    layout: vk::PipelineLayout::null(),
                    words: Making::new("depth copy modules")
                        .words(&modules.vertex)
                        .words(&modules.fragment)
                        .finish(),
                };
                let made = (|| {
                    shaders.vertex = shader_module(raw, &modules.vertex)?;
                    shaders.fragment = shader_module(raw, &modules.fragment)?;
                    shaders.set_layout =
                        set_layout(raw, &[draw_source(copy)], vk::ShaderStageFlags::FRAGMENT)?;
                    shaders.layout = pipeline_layout(raw, shaders.set_layout)?;
                    Ok(())
                })();
                if let Err(error) = made {
                    // SAFETY: nothing used the objects; destroying a null handle is a no-op.
                    unsafe {
                        release::object(raw, shaders.layout);
                        release::object(raw, shaders.set_layout);
                        release::object(raw, shaders.fragment);
                        release::object(raw, shaders.vertex);
                    }
                    return Err(error);
                }
                objects.shaders.push((copy, shaders));
                shaders
            }
        };
        if let Some((_, made)) = objects
            .pipelines
            .iter()
            .find(|(key, _)| *key == (copy, format, samples))
        {
            return Ok((shaders, *made));
        }
        let mut made = DepthPipeline {
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
        };
        // Which draws write a colour attachment rather than a depth one. Stated as a match over
        // every variant, not a `matches!` that falls through to false: a new bits draw left out
        // of a fallthrough list builds a depth render pass over its colour image, which is how
        // `Depth16Bits` first shipped broken.
        let colour =
            match copy {
                DepthStencilCopy::DepthBits
                | DepthStencilCopy::StencilBits
                | DepthStencilCopy::Depth16Bits => true,
                DepthStencilCopy::Depth
                | DepthStencilCopy::Depth16Code
                | DepthStencilCopy::StencilPlane { .. }
                | DepthStencilCopy::StencilExport { .. }
                | DepthStencilCopy::DepthFromStandIn { .. } => false,
                // `DepthStencilCopy` is `#[non_exhaustive]`, so a variant added in the translator
                // crate reaches here. A bits draw must say so above; defaulting to a depth pass is
                // the failure this match exists to stop, so refuse rather than guess.
                _ => return Err(internal(
                    "a depth/stencil copy draw whose attachment kind this runtime does not state",
                )),
            };
        let result = (|| {
            made.render_pass = match colour {
                true => bits_pass(raw, format, samples)?,
                false => depth_pass(raw, format, samples)?,
            };
            let making = Making::new("depth copy")
                .digest(&shaders.words)
                .text(&(copy, format, samples));
            made.pipeline =
                depth_pipeline(self, &shaders, made.render_pass, samples, (colour, making))?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                objects.pipelines.push(((copy, format, samples), made));
                if let DepthStencilCopy::DepthFromStandIn { .. } = copy {
                    shared.counts.stand_in_copy_made(at_draw);
                }
                Ok((shaders, made))
            }
            Err(error) => {
                // SAFETY: nothing used the objects; destroying a null handle is a no-op.
                unsafe {
                    release::object(raw, made.pipeline);
                    release::object(raw, made.render_pass);
                }
                Err(error)
            }
        }
    }

    /// The depth split into a stand-in of `form`, made on first use, by a draw's encoding where
    /// `at_draw` (`WorkCounts::stand_in_copies_made`); refused naming `feature` where the device
    /// lacks what it needs.
    fn depth_split(
        &self,
        form: DepthStandInForm,
        feature: &Feature,
        at_draw: bool,
    ) -> Result<SplitKernel, Error> {
        let _turn = crate::pipeline::budget::helper_turn(&self.shared().depth_copies, |objects| {
            objects.splits.iter().any(|(of, _)| *of == form)
        });
        let mut objects = lock(&self.shared().depth_copies);
        if let Some((_, kernel)) = objects.splits.iter().find(|(of, _)| *of == form) {
            return Ok(*kernel);
        }
        let module = depth_split_module(self.compile_device(), form, feature.clone())
            .map_err(device_error)?;
        if DepthSplitModule::SOURCE_BINDING != 0 || DepthSplitModule::DESTINATION_BINDING != 1 {
            return Err(internal(
                "the depth split's bindings out of their set layout's order",
            ));
        }
        let shared = self.shared();
        let raw = shared.vk();
        let mut kernel = SplitKernel {
            module: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        };
        let made = (|| {
            kernel.module = shader_module(raw, &module.spirv)?;
            kernel.set_layout = set_layout(raw, &SPLIT_SET, vk::ShaderStageFlags::COMPUTE)?;
            kernel.layout = pipeline_layout(raw, kernel.set_layout)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(kernel.module)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(kernel.layout);
            kernel.pipeline = shared
                .make_compute_pipeline(Making::new("depth split").words(&module.spirv), info)?;
            Ok(())
        })();
        match made {
            Ok(()) => {
                objects.splits.push((form, kernel));
                shared.counts.stand_in_copy_made(at_draw);
                Ok(kernel)
            }
            Err(error) => {
                // SAFETY: nothing used the objects; destroying a null handle is a no-op.
                unsafe {
                    release::object(raw, kernel.pipeline);
                    release::object(raw, kernel.layout);
                    release::object(raw, kernel.set_layout);
                    release::object(raw, kernel.module);
                }
                Err(error)
            }
        }
    }
}

/// The storage view of a depth stand-in `stand_in` of `form`: 2D, or 2D-array of its planes (its
/// `layers` from its `layer`).
pub(super) fn stand_in_view(
    stand_in: &Target,
    form: DepthStandInForm,
) -> Result<vk::ImageView, Error> {
    let (view_type, layers) = match form {
        DepthStandInForm::Multisampled => (ImageViewType::TYPE_2D, 1),
        DepthStandInForm::SamplePlanes => (ImageViewType::TYPE_2D_ARRAY, stand_in.layers),
    };
    stand_in.texture.image_view(&VkView {
        view_type,
        format: Format::R32_SFLOAT,
        components: ComponentMapping::IDENTITY,
        aspect: ImageAspectFlags::COLOR,
        base_level: stand_in.level,
        level_count: 1,
        base_layer: stand_in.layer,
        layer_count: layers,
        usage: ImageUsageFlags::STORAGE,
    })
}

/// One copy of a depth aspect's samples into a depth stand-in by the depth split (the module
/// docs).
#[derive(Clone)]
pub(super) struct DepthSplit {
    /// The descriptor set's pool, held until the command buffer completes (dropping the last
    /// handle frees it).
    _objects: Arc<PassObjects>,
    kernel: SplitKernel,
    set: vk::DescriptorSet,
    /// The workgroups the dispatch runs.
    groups: [u32; 3],
}

impl DepthSplit {
    /// The split of `source`'s depth aspect, sampled, into `stand_in` of `form` over the area
    /// `area` from the origin. Refused naming `feature` where the device lacks what the kernel
    /// needs.
    pub(super) fn new(
        device: &Device,
        source: &Target,
        stand_in: &Target,
        form: DepthStandInForm,
        area: [u32; 2],
        feature: &Feature,
    ) -> Result<DepthSplit, Error> {
        if !image_aspects(source.format).contains(ImageAspectFlags::DEPTH)
            || stand_in.format != Format::R32_SFLOAT
        {
            return Err(internal("a depth split from no depth or into no stand-in"));
        }
        let kernel = device.depth_split(form, feature, true)?;
        let source_view = aspect_view(source, ImageAspectFlags::DEPTH)?;
        let destination = stand_in_view(stand_in, form)?;
        let (objects, set) = PassObjects::with_set(
            device,
            kernel.set_layout,
            &[
                (vk::DescriptorType::SAMPLED_IMAGE, source_view),
                (vk::DescriptorType::STORAGE_IMAGE, destination),
            ],
        )?;
        let size = DepthSplitModule::LOCAL_SIZE;
        let [width, height] = area;
        Ok(DepthSplit {
            _objects: Arc::new(objects),
            kernel,
            set,
            groups: [width.div_ceil(size), height.div_ceil(size), source.samples],
        })
    }

    /// Record, outside a render pass, the dispatch.
    ///
    /// # Safety
    ///
    /// `raw` is the device's command buffer, recording outside a render pass, which only the
    /// caller records; the source, the stand-in and this split's objects are live until it
    /// completes and rest in `GENERAL`.
    pub(super) unsafe fn record(&self, device: &ash::Device, raw: vk::CommandBuffer) {
        let compute = vk::PipelineBindPoint::COMPUTE;
        device.cmd_bind_pipeline(raw, compute, self.kernel.pipeline);
        device.cmd_bind_descriptor_sets(raw, compute, self.kernel.layout, 0, &[self.set], &[]);
        let [x, y, z] = self.groups;
        device.cmd_dispatch(raw, x, y, z);
    }
}

/// The draw's render pass for a destination of `format` at `samples`: the destination alone,
/// `GENERAL` throughout, its depth and stencil loaded and stored.
pub(super) fn depth_pass(
    raw: &ash::Device,
    format: Format,
    samples: u32,
) -> Result<vk::RenderPass, Error> {
    let general = vk::ImageLayout::GENERAL;
    let attachments = [vk::AttachmentDescription::default()
        .format(format.into())
        .samples(vk::SampleCountFlags::from_raw(samples))
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::LOAD)
        .stencil_store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(general)
        .final_layout(general)];
    let depth_stencil = vk::AttachmentReference::default()
        .attachment(0)
        .layout(general);
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_stencil)];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    // SAFETY: `info` and its arrays are live for the call.
    unsafe { crate::creation::render_pass(raw, &info) }
}

/// A bits draw's render pass for a colour image of `format` at `samples`: the image alone,
/// `GENERAL` throughout, not loaded (the draw writes every sample of the render area) and stored.
fn bits_pass(raw: &ash::Device, format: Format, samples: u32) -> Result<vk::RenderPass, Error> {
    let general = vk::ImageLayout::GENERAL;
    let attachments = [vk::AttachmentDescription::default()
        .format(format.into())
        .samples(vk::SampleCountFlags::from_raw(samples))
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(general)
        .final_layout(general)];
    let colour = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(general)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colour)];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    // SAFETY: `info` and its arrays are live for the call.
    unsafe { crate::creation::render_pass(raw, &info) }
}

/// The draw in `render_pass` at `samples`: the triangle over the viewport at sample rate, the
/// depth written under compare `ALWAYS` with the stencil test off, or, for a bits draw
/// (`colour`), the colour written whole, unblended, with no depth/stencil attachment.
/// Its record is the one `making` names.
fn depth_pipeline(
    device: &Device,
    shaders: &DepthShaders,
    render_pass: vk::RenderPass,
    samples: u32,
    (colour, making): (bool, Making),
) -> Result<vk::Pipeline, Error> {
    let shared = device.shared();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shaders.vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shaders.fragment)
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
    let mask = [u32::MAX];
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::from_raw(samples))
        .sample_shading_enable(true)
        .min_sample_shading(1.0)
        .sample_mask(&mask);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::ALWAYS)
        .stencil_test_enable(false);
    let written = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = match colour {
        true => vk::PipelineColorBlendStateCreateInfo::default().attachments(&written),
        false => vk::PipelineColorBlendStateCreateInfo::default(),
    };
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(shaders.layout)
        .render_pass(render_pass)
        .subpass(0);
    let info = match colour {
        true => info,
        false => info.depth_stencil_state(&depth_stencil),
    };
    shared.make_graphics_pipeline(making, info)
}

/// Make, where they are not yet made, the draw's objects for a destination of `format` at
/// `samples`: refused naming `feature` where the device lacks what the draw needs.
pub(super) fn depth_draw_runs(
    device: &Device,
    format: Format,
    samples: u32,
    feature: &Feature,
) -> Result<(), Error> {
    device
        .depth_draw(DepthStencilCopy::Depth, format, samples, feature)
        .map(|_| ())
}

/// One copy of a depth aspect into a depth/stencil target's by the draw, or of a depth or
/// stencil aspect into a colour target by a bits draw (the module docs).
#[derive(Clone)]
pub(super) struct DepthDraw {
    objects: Arc<PassObjects>,
    shaders: DepthShaders,
    made: DepthPipeline,
    set: vk::DescriptorSet,
    /// The render area.
    area: DrawArea,
}

impl DepthDraw {
    /// The draw that writes `source`'s depth aspect, sampled, into `destination`'s over the
    /// render area `area`, which both cover. Refused naming `feature` where the device lacks
    /// what the draw needs.
    pub(super) fn new(
        device: &Device,
        source: &Target,
        destination: &Target,
        area: [u32; 2],
        feature: &Feature,
    ) -> Result<DepthDraw, Error> {
        let depth =
            |target: &Target| image_aspects(target.format).contains(ImageAspectFlags::DEPTH);
        if source.samples != destination.samples || !depth(source) || !depth(destination) {
            return Err(internal(
                "a depth copy between targets of other samples or without depth",
            ));
        }
        let view = aspect_view(source, ImageAspectFlags::DEPTH)?;
        let area = DrawArea::whole(area);
        DepthDraw::of_view(
            device,
            DepthStencilCopy::Depth,
            view,
            destination,
            area,
            feature,
        )
    }

    /// The draw that writes the depths the stand-in `stand_in` of `form` holds at each pixel and
    /// sample of the area `area` from the origin into `destination`'s depth there.
    pub(super) fn from_stand_in(
        device: &Device,
        stand_in: &Target,
        form: DepthStandInForm,
        destination: &Target,
        area: [u32; 2],
        feature: &Feature,
    ) -> Result<DepthDraw, Error> {
        if stand_in.format != Format::R32_SFLOAT
            || !image_aspects(destination.format).contains(ImageAspectFlags::DEPTH)
        {
            return Err(internal(
                "a depth copy from no stand-in or into a target without depth",
            ));
        }
        let view = stand_in_view(stand_in, form)?;
        DepthDraw::of_view(
            device,
            DepthStencilCopy::DepthFromStandIn { form },
            view,
            destination,
            DrawArea::whole(area),
            feature,
        )
    }

    /// The draw that writes the depths `source`, a multisampled colour target, holds at each
    /// pixel of `area` into `destination`'s depth there: an `R32_SFLOAT` source's floats into a
    /// 32-bit float depth, or an `R16_UINT` source's codes into a 16-bit unsigned normalized one
    /// ([`DepthStencilCopy::Depth16Code`], the module docs).
    pub(super) fn from_colour(
        device: &Device,
        source: &Target,
        destination: &Target,
        area: DrawArea,
        feature: &Feature,
    ) -> Result<DepthDraw, Error> {
        let copy = match unorm16_depth(destination.format) {
            true => DepthStencilCopy::Depth16Code,
            false => DepthStencilCopy::Depth,
        };
        let carrier = depth_carrier(destination.format)
            .ok_or_else(|| internal("a depth copy into a target of no drawn depth"))?;
        if source.samples != destination.samples
            || source.format != carrier
            || !image_aspects(destination.format).contains(ImageAspectFlags::DEPTH)
        {
            return Err(internal(
                "a depth copy from a colour target of other samples or another format",
            ));
        }
        let view = aspect_view(source, ImageAspectFlags::COLOR)?;
        DepthDraw::of_view(device, copy, view, destination, area, feature)
    }

    /// The bits draw that writes the bits of `source`'s `aspect` (a 32-bit float depth's `f32`
    /// bit-cast, a 16-bit unsigned normalized depth's code, a stencil's value) at each pixel of
    /// `area` into `destination` there: a multisampled `R32_UINT` colour target for a 32-bit
    /// float depth, `R16_UINT` for a 16-bit one, `R8_UINT` for a stencil.
    pub(super) fn bits(
        device: &Device,
        source: &Target,
        aspect: ImageAspectFlags,
        destination: &Target,
        area: DrawArea,
        feature: &Feature,
    ) -> Result<DepthDraw, Error> {
        let (copy, format) = match aspect {
            ImageAspectFlags::DEPTH if unorm16_depth(source.format) => {
                (DepthStencilCopy::Depth16Bits, Format::R16_UINT)
            }
            ImageAspectFlags::DEPTH => (DepthStencilCopy::DepthBits, Format::R32_UINT),
            _ => (DepthStencilCopy::StencilBits, Format::R8_UINT),
        };
        if source.samples != destination.samples
            || !image_aspects(source.format).contains(aspect)
            || destination.format != format
        {
            return Err(internal(
                "a bits copy between targets of other samples or formats",
            ));
        }
        let view = aspect_view(source, aspect)?;
        DepthDraw::of_view(device, copy, view, destination, area, feature)
    }

    /// The draw of `copy` reading `view` into `destination` over `area`.
    fn of_view(
        device: &Device,
        copy: DepthStencilCopy,
        view: vk::ImageView,
        destination: &Target,
        area: DrawArea,
        feature: &Feature,
    ) -> Result<DepthDraw, Error> {
        let (shaders, made) =
            device.depth_draw(copy, destination.format, destination.samples, feature)?;
        let (mut objects, set) =
            PassObjects::with_set(device, shaders.set_layout, &[(draw_source(copy), view)])?;
        draw_framebuffer(&mut objects, made.render_pass, destination, area.end())?;
        Ok(DepthDraw {
            objects: Arc::new(objects),
            shaders,
            made,
            set,
            area,
        })
    }

    /// Record, outside a render pass, the draw's render pass.
    ///
    /// # Safety
    ///
    /// `raw` is the device's command buffer, recording outside a render pass, which only the
    /// caller records; the source, the destination and this draw's objects are live until it
    /// completes and rest in `GENERAL`.
    pub(super) unsafe fn record(&self, device: &ash::Device, raw: vk::CommandBuffer) {
        record_draws_in(
            device,
            raw,
            (self.made.render_pass, self.objects.framebuffer),
            (self.shaders.layout, self.set),
            &[self.made.pipeline],
            self.area,
        );
    }
}
