//! Fixed-function state mapping: blend, color write masks, depth/stencil, primitive topology,
//! index types, and the render encoder's raster state. A render pipeline's
//! [`GraphicsState`](crate::pipeline::GraphicsState) holds the results as plain `vk::` values;
//! Metal's depth/stencil state is a separate encoder object and maps here to [`DepthStencilState`].
//!
//! The render encoder's triangle fill mode, depth clip mode, and depth bias are not inputs of a
//! render pipeline: Metal sets them per encoder, so the consumer owns them and maps them here,
//! each with the device feature it needs ([`polygon_mode`], [`depth_clamp_enable`],
//! [`depth_bias`]). The first two are static Vulkan pipeline state (pipeline variants); the depth
//! bias factors are dynamic state on every render pipeline (`GraphicsState::dynamic_states`).

use crate::device::requirements::VkFeature;
use crate::device::DeviceProfile;
use crate::error::{Capability, DescriptorError, Error, Feature, Unsupported};
use crate::metal::sampler::compare_op;
use crate::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, CompareFunction, DepthClipMode, IndexType,
    PixelFormat, PrimitiveType, StencilOperation, TriangleFillMode,
};
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::graphics::BlendState;
use crate::vk;

fn placeholder(ty: &'static str, value: u32) -> Error {
    Error::InvalidDescriptor(DescriptorError::PlaceholderEnumValue { ty, value })
}

/// `MTLBlendFactor` → `VkBlendFactor`. The `Source1*` factors read the second fragment output
/// and need `dualSrcBlend`; `Unspecialized` is a placeholder, not a factor.
pub fn blend_factor(f: BlendFactor, profile: &DeviceProfile) -> Result<vk::BlendFactor, Error> {
    use vk::BlendFactor as V;
    use BlendFactor as M;
    let dual = |v: V| {
        if profile.fields().fragment.dual_src_blend {
            Ok(v)
        } else {
            Err(Error::from(Unsupported::needs(
                Feature::DualSourceBlending,
                Capability::Feature(VkFeature::DualSrcBlend),
            )))
        }
    };
    Ok(match f {
        M::Zero => V::ZERO,
        M::One => V::ONE,
        M::SourceColor => V::SRC_COLOR,
        M::OneMinusSourceColor => V::ONE_MINUS_SRC_COLOR,
        M::SourceAlpha => V::SRC_ALPHA,
        M::OneMinusSourceAlpha => V::ONE_MINUS_SRC_ALPHA,
        M::DestinationColor => V::DST_COLOR,
        M::OneMinusDestinationColor => V::ONE_MINUS_DST_COLOR,
        M::DestinationAlpha => V::DST_ALPHA,
        M::OneMinusDestinationAlpha => V::ONE_MINUS_DST_ALPHA,
        M::SourceAlphaSaturated => V::SRC_ALPHA_SATURATE,
        M::BlendColor => V::CONSTANT_COLOR,
        M::OneMinusBlendColor => V::ONE_MINUS_CONSTANT_COLOR,
        M::BlendAlpha => V::CONSTANT_ALPHA,
        M::OneMinusBlendAlpha => V::ONE_MINUS_CONSTANT_ALPHA,
        M::Source1Color => dual(V::SRC1_COLOR)?,
        M::OneMinusSource1Color => dual(V::ONE_MINUS_SRC1_COLOR)?,
        M::Source1Alpha => dual(V::SRC1_ALPHA)?,
        M::OneMinusSource1Alpha => dual(V::ONE_MINUS_SRC1_ALPHA)?,
        M::Unspecialized => return Err(placeholder(BlendFactor::METAL_NAME, f.as_raw())),
    })
}

/// `MTLBlendOperation` → `VkBlendOp`.
pub fn blend_op(op: BlendOperation) -> Result<vk::BlendOp, Error> {
    Ok(match op {
        BlendOperation::Add => vk::BlendOp::ADD,
        BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOperation::Min => vk::BlendOp::MIN,
        BlendOperation::Max => vk::BlendOp::MAX,
        BlendOperation::Unspecialized => {
            return Err(placeholder(BlendOperation::METAL_NAME, op.as_raw()))
        }
    })
}

/// `MTLColorWriteMask` → `VkColorComponentFlags`. Metal numbers the channels from alpha
/// (`Alpha = 1`, `Red = 8`), Vulkan from red (`R = 1`, `A = 8`).
///
/// `MTLColorWriteMaskUnspecialized` (`0x10`) is MTL4's placeholder for a mask a render command
/// specializes later, but a **classic** `MTLRenderPipelineDescriptor` takes it and draws: measured
/// on an Apple M3 Max, macOS 27.0.1, a pipeline whose `writeMask` is `0x10` stores **every**
/// channel of its fragment, and so does one whose mask is `0x18` or `0x1f`, so the bit dominates
/// the four channel bits rather than being ignored (ignoring it would leave `0x10` as a mask of no
/// channel, which stores nothing). Metal's setter keeps five bits: `0x20` reads back as `0x0`,
/// `0x30` as `0x10` and `0xffffffff` as `0x1f`. So the bit is every channel here, which is what
/// `cases/write_mask_unspecialized.rs` of the conformance suite compares against Metal.
pub fn color_write_mask(mask: ColorWriteMask) -> Result<vk::ColorComponentFlags, Error> {
    if mask.contains(ColorWriteMask::UNSPECIALIZED) {
        return Ok(vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A);
    }
    let mut out = vk::ColorComponentFlags::empty();
    for (m, v) in [
        (ColorWriteMask::RED, vk::ColorComponentFlags::R),
        (ColorWriteMask::GREEN, vk::ColorComponentFlags::G),
        (ColorWriteMask::BLUE, vk::ColorComponentFlags::B),
        (ColorWriteMask::ALPHA, vk::ColorComponentFlags::A),
    ] {
        if mask.contains(m) {
            out |= v;
        }
    }
    Ok(out)
}

/// The Vulkan blend state of a Metal color attachment: `None` when blending is disabled (the
/// factors and operations are then ignored, placeholders included).
///
/// A portability-subset device without `constantAlphaColorBlendFactors` cannot take `BlendAlpha`
/// or `OneMinusBlendAlpha` as an RGB factor: refused as `Unsupported{ConstantAlphaBlendFactor,
/// Feature(ConstantAlphaColorBlendFactors)}`. Link applies the check only to an attachment the draw
/// can write.
pub fn blend_state(
    attachment: &ColorAttachmentDescriptor,
    profile: &DeviceProfile,
) -> Result<Option<BlendState>, Error> {
    let state = blend_factors(attachment, profile)?;
    if let Some(s) = &state {
        check_constant_alpha(s, profile)?;
    }
    Ok(state)
}

/// [`blend_state`] without the portability subset's constant-alpha check, which link applies only
/// to an attachment the draw can write.
///
/// An `A8Unorm` attachment has no color channel, so its color equation blends nothing: it takes
/// the alpha operation there too, which a driver that blends the format's one channel by the
/// color operation (lavapipe, Mesa 26.2: with the alpha factors) then applies as Metal does.
pub(crate) fn blend_factors(
    attachment: &ColorAttachmentDescriptor,
    profile: &DeviceProfile,
) -> Result<Option<BlendState>, Error> {
    if !attachment.blending_enabled {
        return Ok(None);
    }
    let color_op = match attachment.pixel_format {
        PixelFormat::A8Unorm => attachment.alpha_blend_operation,
        _ => attachment.rgb_blend_operation,
    };
    Ok(Some(BlendState {
        src_color: blend_factor(attachment.source_rgb_blend_factor, profile)?,
        dst_color: blend_factor(attachment.destination_rgb_blend_factor, profile)?,
        color_op: blend_op(color_op)?,
        src_alpha: blend_factor(attachment.source_alpha_blend_factor, profile)?,
        dst_alpha: blend_factor(attachment.destination_alpha_blend_factor, profile)?,
        alpha_op: blend_op(attachment.alpha_blend_operation)?,
    }))
}

/// Whether `profile` withholds a constant-alpha RGB factor `state` uses: a portability-subset
/// device without `constantAlphaColorBlendFactors`
/// (`VUID-VkPipelineColorBlendAttachmentState-constantAlphaColorBlendFactors-04454`, `-04455`).
pub(crate) fn constant_alpha_withheld(state: &BlendState, profile: &DeviceProfile) -> bool {
    state.constant_alpha_color()
        && profile
            .fields()
            .enabled
            .portability_withholds(VkFeature::ConstantAlphaColorBlendFactors)
}

/// `Unsupported{ConstantAlphaBlendFactor, Feature(ConstantAlphaColorBlendFactors)}` where
/// [`constant_alpha_withheld`].
pub(crate) fn check_constant_alpha(
    state: &BlendState,
    profile: &DeviceProfile,
) -> Result<(), Error> {
    if constant_alpha_withheld(state, profile) {
        return Err(Unsupported::needs(
            Feature::ConstantAlphaBlendFactor,
            Capability::Feature(VkFeature::ConstantAlphaColorBlendFactors),
        )
        .into());
    }
    Ok(())
}

/// `MTLStencilOperation` → `VkStencilOp` (the same eight operations in the same order).
pub fn stencil_op(op: StencilOperation) -> vk::StencilOp {
    use vk::StencilOp as V;
    use StencilOperation as M;
    match op {
        M::Keep => V::KEEP,
        M::Zero => V::ZERO,
        M::Replace => V::REPLACE,
        M::IncrementClamp => V::INCREMENT_AND_CLAMP,
        M::DecrementClamp => V::DECREMENT_AND_CLAMP,
        M::Invert => V::INVERT,
        M::IncrementWrap => V::INCREMENT_AND_WRAP,
        M::DecrementWrap => V::DECREMENT_AND_WRAP,
    }
}

/// `MTLStencilDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StencilDescriptor {
    pub stencil_compare_function: CompareFunction,
    pub stencil_failure_operation: StencilOperation,
    pub depth_failure_operation: StencilOperation,
    pub depth_stencil_pass_operation: StencilOperation,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl Default for StencilDescriptor {
    /// Metal's defaults: `Always`, `Keep` everywhere, all mask bits set.
    fn default() -> StencilDescriptor {
        StencilDescriptor {
            stencil_compare_function: CompareFunction::Always,
            stencil_failure_operation: StencilOperation::Keep,
            depth_failure_operation: StencilOperation::Keep,
            depth_stencil_pass_operation: StencilOperation::Keep,
            read_mask: u32::MAX,
            write_mask: u32::MAX,
        }
    }
}

impl StencilDescriptor {
    /// Whether the stencil test on this face can neither fail nor change the stencil buffer.
    fn is_noop(&self) -> bool {
        self.stencil_compare_function == CompareFunction::Always
            && self.depth_failure_operation == StencilOperation::Keep
            && self.depth_stencil_pass_operation == StencilOperation::Keep
    }
}

/// `MTLDepthStencilDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DepthStencilDescriptor {
    pub depth_compare_function: CompareFunction,
    pub depth_write_enabled: bool,
    pub front_face_stencil: StencilDescriptor,
    pub back_face_stencil: StencilDescriptor,
}

impl Default for DepthStencilDescriptor {
    /// Metal's defaults: depth `Always` without writes, default stencil on both faces.
    fn default() -> DepthStencilDescriptor {
        DepthStencilDescriptor {
            depth_compare_function: CompareFunction::Always,
            depth_write_enabled: false,
            front_face_stencil: StencilDescriptor::default(),
            back_face_stencil: StencilDescriptor::default(),
        }
    }
}

/// `VkStencilOpState` without `reference` (Metal sets it per encoder: dynamic state).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StencilOpState {
    pub fail_op: vk::StencilOp,
    pub pass_op: vk::StencilOp,
    pub depth_fail_op: vk::StencilOp,
    pub compare_op: vk::CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
}

/// The depth/stencil part of `VkPipelineDepthStencilStateCreateInfo` (no depth bounds; Metal
/// has none).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DepthStencilState {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub stencil_test_enable: bool,
    pub front: StencilOpState,
    pub back: StencilOpState,
}

/// The Vulkan depth/stencil state of a Metal depth/stencil descriptor.
///
/// Vulkan's depth test also gates depth writes, so the test is enabled whenever Metal compares or
/// writes (an `Always` test is the identity). The stencil test is enabled unless both faces are
/// no-ops (`Always`, and `Keep` on pass and depth failure), which is exact.
pub fn depth_stencil_state(desc: &DepthStencilDescriptor) -> DepthStencilState {
    let face = |s: &StencilDescriptor| StencilOpState {
        fail_op: stencil_op(s.stencil_failure_operation),
        pass_op: stencil_op(s.depth_stencil_pass_operation),
        depth_fail_op: stencil_op(s.depth_failure_operation),
        compare_op: compare_op(s.stencil_compare_function),
        compare_mask: s.read_mask,
        write_mask: s.write_mask,
    };
    DepthStencilState {
        depth_test_enable: desc.depth_compare_function != CompareFunction::Always
            || desc.depth_write_enabled,
        depth_write_enable: desc.depth_write_enabled,
        depth_compare_op: compare_op(desc.depth_compare_function),
        stencil_test_enable: !(desc.front_face_stencil.is_noop()
            && desc.back_face_stencil.is_noop()),
        front: face(&desc.front_face_stencil),
        back: face(&desc.back_face_stencil),
    }
}

/// The Vulkan input assembly of a Metal primitive type: the topology and whether primitive
/// restart is enabled. Metal always restarts strips at the all-ones index; Vulkan 1.2 allows
/// restart only for strip topologies, and restart is inert for non-indexed draws.
pub fn input_assembly(ty: PrimitiveType) -> (vk::PrimitiveTopology, bool) {
    use vk::PrimitiveTopology as T;
    match ty {
        PrimitiveType::Point => (T::POINT_LIST, false),
        PrimitiveType::Line => (T::LINE_LIST, false),
        PrimitiveType::LineStrip => (T::LINE_STRIP, true),
        PrimitiveType::Triangle => (T::TRIANGLE_LIST, false),
        PrimitiveType::TriangleStrip => (T::TRIANGLE_STRIP, true),
    }
}

/// `MTLIndexType` → `VkIndexType`.
pub fn index_type(ty: IndexType) -> vk::IndexType {
    match ty {
        IndexType::UInt16 => vk::IndexType::UINT16,
        IndexType::UInt32 => vk::IndexType::UINT32,
    }
}

/// `Ok` when the consumer enabled `feature`, else a refusal of `what` naming it.
fn require(profile: &DeviceProfile, feature: VkFeature, what: Feature) -> Result<(), Error> {
    if profile.fields().enabled.features.contains(&feature) {
        Ok(())
    } else {
        Err(Unsupported::needs(what, Capability::Feature(feature)).into())
    }
}

/// The render encoder's `MTLTriangleFillMode` (`setTriangleFillMode:`) → the pipeline's
/// `VkPipelineRasterizationStateCreateInfo::polygonMode`. Both APIs apply the mode to triangles
/// only. `Lines` rasterizes each triangle's edges as lines (`LINE`, with `lineWidth` 1) and needs
/// `fillModeNonSolid` (`VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507`).
pub fn polygon_mode(
    mode: TriangleFillMode,
    profile: &DeviceProfile,
) -> Result<vk::PolygonMode, Error> {
    Ok(match mode {
        TriangleFillMode::Fill => vk::PolygonMode::FILL,
        TriangleFillMode::Lines => {
            require(
                profile,
                VkFeature::FillModeNonSolid,
                Feature::TriangleFillLines,
            )?;
            vk::PolygonMode::LINE
        }
    })
}

/// The render encoder's `MTLDepthClipMode` (`setDepthClipMode:`) → the pipeline's
/// `VkPipelineRasterizationStateCreateInfo::depthClampEnable`. `Clip` discards what lies beyond
/// the near or far plane, as Vulkan's clipping does; `Clamp` keeps it and clamps its depth to the
/// viewport's depth range, as Vulkan's depth clamp does, and needs `depthClamp`
/// (`VUID-VkPipelineRasterizationStateCreateInfo-depthClampEnable-00782`).
pub fn depth_clamp_enable(mode: DepthClipMode, profile: &DeviceProfile) -> Result<bool, Error> {
    Ok(match mode {
        DepthClipMode::Clip => false,
        DepthClipMode::Clamp => {
            require(profile, VkFeature::DepthClamp, Feature::DepthClipClamp)?;
            true
        }
    })
}

/// The render encoder's depth bias (`setDepthBias:slopeScale:clamp:`) as Vulkan's: see
/// [`depth_bias`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DepthBias {
    /// `VkPipelineRasterizationStateCreateInfo::depthBiasEnable`: whether the pipeline's
    /// primitives take a bias at all. It follows the topology only, never the bias values, so one
    /// pipeline serves every bias the encoder sets.
    pub enable: bool,
    /// `depthBiasConstantFactor`: Metal's `depthBias`, in the unit `unit` names.
    pub constant_factor: f32,
    /// `depthBiasClamp`: Metal's `clamp` (0 does not clamp in either API).
    pub clamp: f32,
    /// `depthBiasSlopeFactor`: Metal's `slopeScale`.
    pub slope_factor: f32,
    /// The unit `constant_factor` counts, and so the command that records the three factors.
    pub unit: DepthBiasUnit,
}

/// The unit a [`DepthBias`]'s constant factor counts (Vulkan's r), which names the command that
/// records the factors.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DepthBiasUnit {
    /// The device's own for the depth attachment's format: `vkCmdSetDepthBias`.
    Device,
    /// Depth itself, so the constant factor is the bias: `vkCmdSetDepthBias2EXT` with a
    /// `VkDepthBiasRepresentationInfoEXT` of `VK_DEPTH_BIAS_REPRESENTATION_FLOAT_EXT` and
    /// `depthBiasExact` (`VK_EXT_depth_bias_control`'s `floatRepresentation` and
    /// `depthBiasExact`).
    Depth,
    /// 2^-n for a depth aspect of n fixed-point bits: `vkCmdSetDepthBias2EXT` with a
    /// `VkDepthBiasRepresentationInfoEXT` of
    /// `VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORMAT_EXT` and `depthBiasExact`
    /// (`VK_EXT_depth_bias_control`'s `depthBiasExact`).
    ExactFormatUnit,
}

/// The Vulkan depth bias of the render encoder's `setDepthBias:slopeScale:clamp:` for a pipeline
/// of `topology` (`GraphicsState::topology`, or the draw's from [`input_assembly`]), in a pass
/// whose depth attachment's texture is of `depth_format` (`None` without one). `DEPTH_BIAS` is
/// dynamic state on every render pipeline: create the pipeline with `enable` and record the three
/// factors by the command `unit` names (Metal's default is no bias, all three 0).
///
/// Metal biases triangles only, never points or lines; Vulkan biases triangles (a tessellation
/// pipeline's included) and may bias points and lines, so `enable` is set for triangle and patch
/// topologies alone, and the factors of another topology are 0. A nonzero clamp needs
/// `depthBiasClamp` (`VUID-vkCmdSetDepthBias-depthBiasClamp-00790`) unless both bias factors are
/// 0, when no bias exists to clamp and the clamp is dropped.
///
/// A float depth format's constant factor counts Vulkan's r, which is Metal's. A `Depth16Unorm`
/// depth's is Metal's own rule (measured on an Apple M3 Max and an M4, plain and validated, and on
/// the M3 Max under macOS 27.0.1 with slopes too): wherever the constant `c` (finite) or the slope
/// factor is nonzero, the constant biases by `c + 1` units of 1/65535 for `c` of 0 or more and
/// `c - 1` below, as `float` (so a slope alone brings one unit), which is added to the depth
/// rounded toward zero, and the sum stored rounded to nearest. Vulkan leaves a unorm
/// format's unit to the device, up to twice 2^-16, unless `depthBiasExact` pins it: with
/// `floatRepresentation` too the factor is that bias in depth ([`DepthBiasUnit::Depth`]), and with
/// `depthBiasExact` alone it is the same bias counted in 2^-16
/// ([`DepthBiasUnit::ExactFormatUnit`]), a scale by a power of two, so the device adds Metal's
/// `float` bias either way. The device's own addition rounds the sum, to nearest where measured, so
/// a biased depth within one `float` ulp above a code's midpoint can store one code above Metal's.
/// A device without `depthBiasExact` takes Metal's `c` in its own unit ([`DepthBiasUnit::Device`]),
/// as a Metal device does; elsewhere that unit is the device's.
pub fn depth_bias(
    topology: vk::PrimitiveTopology,
    depth_bias: f32,
    slope_scale: f32,
    clamp: f32,
    depth_format: Option<PixelFormat>,
    profile: &DeviceProfile,
) -> Result<DepthBias, Error> {
    use vk::PrimitiveTopology as T;
    let enable = matches!(
        topology,
        T::TRIANGLE_LIST
            | T::TRIANGLE_STRIP
            | T::TRIANGLE_FAN
            | T::TRIANGLE_LIST_WITH_ADJACENCY
            | T::TRIANGLE_STRIP_WITH_ADJACENCY
            | T::PATCH_LIST
    );
    if !enable || (depth_bias == 0.0 && slope_scale == 0.0) {
        return Ok(DepthBias {
            enable,
            constant_factor: 0.0,
            clamp: 0.0,
            slope_factor: 0.0,
            unit: DepthBiasUnit::Device,
        });
    }
    if clamp != 0.0 {
        require(profile, VkFeature::DepthBiasClamp, Feature::DepthBiasClamp)?;
    }
    let enabled = &profile.fields().enabled.features;
    // A bias is set (the return above): a constant, a slope or both.
    let unorm16 = depth_format == Some(PixelFormat::Depth16Unorm)
        && depth_bias.is_finite()
        && enabled.contains(&VkFeature::DepthBiasExact);
    let (constant_factor, unit) = if !unorm16 {
        (depth_bias, DepthBiasUnit::Device)
    } else {
        let units = depth16_units(depth_bias);
        if enabled.contains(&VkFeature::FloatRepresentation) {
            ((units / 65535.0) as f32, DepthBiasUnit::Depth)
        } else {
            (
                (units * 65536.0 / 65535.0) as f32,
                DepthBiasUnit::ExactFormatUnit,
            )
        }
    };
    Ok(DepthBias {
        enable,
        constant_factor,
        clamp,
        slope_factor: slope_scale,
        unit,
    })
}

/// Metal's `c ± 1` units of 1/65535 for a `Depth16Unorm` depth's constant `c` (finite), exact in
/// `f64`: `c + 1` for `c` of 0 or more, `c - 1` below ([`depth_bias`]).
fn depth16_units(c: f32) -> f64 {
    f64::from(c) + if c >= 0.0 { 1.0 } else { -1.0 }
}

/// The depth bias a fragment stage that stores Metal's `Depth16Unorm` codes itself adds
/// (`BindState::depth_bias`, `RuntimeFact::Depth16Codes`), for the render encoder's
/// `setDepthBias:slopeScale:clamp:` factors `depth_bias` and `slope_scale` (both finite) and
/// `clamp` (finite; Metal's NaN or infinite clamp clamps nothing, as 0 does): `[u, s, k]`, `u`
/// Metal's `c ± 1` units of 1/65535 rounded once to `float` ([`depth_bias`]'s rule), `s` and `k`
/// the factors. All three are 0 where neither factor is nonzero, which biases nothing; the caller
/// passes 0 for a draw of points or lines, which Metal never biases.
pub fn depth16_bias(depth_bias: f32, slope_scale: f32, clamp: f32) -> [f32; 3] {
    if depth_bias == 0.0 && slope_scale == 0.0 {
        return [0.0; 3];
    }
    let units = (depth16_units(depth_bias) / 65535.0) as f32;
    [units, slope_scale, clamp]
}

/// The side of the depth bias a fragment stage that stores Metal's `Depth16Unorm` codes itself
/// adds for the factors `depth_bias`, `slope_scale` and `clamp` ([`depth16_bias`]'s), where every
/// triangle's biased depth lies on it: `Some(true)` where Metal's units are above 0, the slope
/// factor finite and not below 0, and a clamp above 0 not below the units, so the stored depth is
/// never below the fragment's own; `Some(false)` for the mirror, never above it; `None` for any
/// other bias, whose side a triangle's slope decides, and for none. The caller also needs every
/// Vulkan depth range of the draw inside `[0, 1]` (`BindState::depth16_bias_raises`); the proof is
/// `lower::stage_io::depth16::side_bounded`'s.
pub fn depth16_bias_side(depth_bias: f32, slope_scale: f32, clamp: f32) -> Option<bool> {
    let [units, slope, clamp] = depth16_bias(depth_bias, slope_scale, clamp);
    if !units.is_finite() || !slope.is_finite() {
        return None;
    }
    // A NaN or infinite clamp bounds nothing (`depth16_bias`), as the stage's comparisons find.
    let raises = units > 0.0 && slope >= 0.0 && !(clamp > 0.0 && clamp < units);
    let lowers = units < 0.0 && slope <= 0.0 && !(clamp < 0.0 && clamp > units);
    match (raises, lowers) {
        (true, _) => Some(true),
        (_, true) => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DevicePolicy};
    use crate::metal::format::tests::committed_descriptions;

    fn device(dual: bool) -> Device {
        let (_, mut desc) = committed_descriptions()
            .into_iter()
            .find(|(s, _)| s == "vk12-floor")
            .unwrap();
        if dual {
            desc.features.insert(VkFeature::DualSrcBlend);
        } else {
            desc.features.remove(&VkFeature::DualSrcBlend);
        }
        Device::new(desc, DevicePolicy::default()).unwrap()
    }

    /// Every concrete factor maps, the names agree, and only `Source1*` needs `dualSrcBlend`.
    #[test]
    fn blend_factors_and_operations() {
        let (with, without) = (device(true), device(false));
        for &f in BlendFactor::ALL {
            let name = f.name();
            if f == BlendFactor::Unspecialized {
                assert!(blend_factor(f, with.profile()).is_err());
                continue;
            }
            let v = blend_factor(f, with.profile()).unwrap();
            assert_eq!(
                name.contains("Source1"),
                blend_factor(f, without.profile()).is_err(),
                "{name}"
            );
            let vn = v.name().unwrap();
            assert_eq!(
                name.contains("OneMinus"),
                vn.contains("ONE_MINUS"),
                "{name} -> {vn}"
            );
        }
        for &op in BlendOperation::ALL {
            match blend_op(op) {
                Ok(v) => assert_eq!(v.as_raw() as u32, op.as_raw()),
                Err(_) => assert_eq!(op, BlendOperation::Unspecialized),
            }
        }
        let mut att = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        assert_eq!(blend_state(&att, without.profile()), Ok(None));
        att.blending_enabled = true;
        att.source_rgb_blend_factor = BlendFactor::SourceAlpha;
        att.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        let b = blend_state(&att, without.profile()).unwrap().unwrap();
        assert_eq!(
            (b.src_color, b.dst_color, b.color_op),
            (
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD
            )
        );
        // An `A8Unorm` attachment, which has no color channel, takes its alpha operation in
        // both equations.
        att.pixel_format = PixelFormat::A8Unorm;
        att.alpha_blend_operation = BlendOperation::ReverseSubtract;
        let b = blend_state(&att, without.profile()).unwrap().unwrap();
        assert_eq!(
            (b.color_op, b.alpha_op),
            (vk::BlendOp::REVERSE_SUBTRACT, vk::BlendOp::REVERSE_SUBTRACT)
        );
    }

    #[test]
    fn write_masks_reverse_channel_order() {
        assert_eq!(
            color_write_mask(ColorWriteMask::ALL),
            Ok(vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A)
        );
        assert_eq!(
            color_write_mask(ColorWriteMask::RED | ColorWriteMask::ALPHA),
            Ok(vk::ColorComponentFlags::R | vk::ColorComponentFlags::A)
        );
        assert_eq!(
            color_write_mask(ColorWriteMask::NONE),
            Ok(vk::ColorComponentFlags::empty())
        );
        // Metal draws every channel under the placeholder bit, measured, and the bit dominates the
        // channel bits rather than being masked out of them: `0x10` alone and `0x10 | RED` are both
        // every channel, where dropping the bit would make the first a mask of none.
        let all = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        assert_eq!(color_write_mask(ColorWriteMask::UNSPECIALIZED), Ok(all));
        assert_eq!(
            color_write_mask(ColorWriteMask::UNSPECIALIZED | ColorWriteMask::RED),
            Ok(all)
        );
        assert_eq!(
            color_write_mask(ColorWriteMask::ALL.union(ColorWriteMask::UNSPECIALIZED)),
            Ok(all)
        );
    }

    #[test]
    fn depth_stencil_defaults_are_inert_and_writes_enable_the_test() {
        let s = depth_stencil_state(&DepthStencilDescriptor::default());
        assert!(!s.depth_test_enable && !s.depth_write_enable && !s.stencil_test_enable);
        let mut d = DepthStencilDescriptor {
            depth_write_enabled: true,
            ..DepthStencilDescriptor::default()
        };
        let s = depth_stencil_state(&d);
        assert!(s.depth_test_enable && s.depth_write_enable);
        assert_eq!(s.depth_compare_op, vk::CompareOp::ALWAYS);
        d.back_face_stencil.depth_stencil_pass_operation = StencilOperation::IncrementWrap;
        d.back_face_stencil.read_mask = 0x0f;
        let s = depth_stencil_state(&d);
        assert!(s.stencil_test_enable);
        assert_eq!(s.back.pass_op, vk::StencilOp::INCREMENT_AND_WRAP);
        assert_eq!(s.back.compare_mask, 0x0f);
        assert_eq!(s.front.pass_op, vk::StencilOp::KEEP);
        for &op in StencilOperation::ALL {
            assert_eq!(stencil_op(op).as_raw() as u32, op.as_raw());
        }
    }

    fn committed(name: &str) -> Device {
        let (_, desc) = committed_descriptions()
            .into_iter()
            .find(|(s, _)| s == name)
            .unwrap();
        Device::new(desc, DevicePolicy::default()).unwrap()
    }

    /// The encoder's fill mode, depth clip mode, and depth bias clamp map where the device has
    /// their features and are refused, naming the feature, where it does not.
    #[test]
    fn encoder_raster_state_needs_its_features() {
        let (floor, lavapipe) = (committed("vk12-floor"), committed("lavapipe"));
        let needs =
            |what, feature| Error::from(Unsupported::needs(what, Capability::Feature(feature)));
        assert_eq!(
            polygon_mode(TriangleFillMode::Fill, floor.profile()),
            Ok(vk::PolygonMode::FILL)
        );
        assert_eq!(
            polygon_mode(TriangleFillMode::Lines, floor.profile()),
            Err(needs(
                Feature::TriangleFillLines,
                VkFeature::FillModeNonSolid
            ))
        );
        assert_eq!(
            polygon_mode(TriangleFillMode::Lines, lavapipe.profile()),
            Ok(vk::PolygonMode::LINE)
        );
        assert_eq!(
            depth_clamp_enable(DepthClipMode::Clip, floor.profile()),
            Ok(false)
        );
        assert_eq!(
            depth_clamp_enable(DepthClipMode::Clamp, floor.profile()),
            Err(needs(Feature::DepthClipClamp, VkFeature::DepthClamp))
        );
        assert_eq!(
            depth_clamp_enable(DepthClipMode::Clamp, lavapipe.profile()),
            Ok(true)
        );

        let triangles = vk::PrimitiveTopology::TRIANGLE_LIST;
        let bias = |enable, constant_factor, clamp, slope_factor| {
            Ok(DepthBias {
                enable,
                constant_factor,
                clamp,
                slope_factor,
                unit: DepthBiasUnit::Device,
            })
        };
        assert_eq!(
            depth_bias(triangles, 1.0, 2.0, 0.0, None, floor.profile()),
            bias(true, 1.0, 0.0, 2.0)
        );
        assert_eq!(
            depth_bias(triangles, 1.0, 2.0, 0.5, None, floor.profile()),
            Err(needs(Feature::DepthBiasClamp, VkFeature::DepthBiasClamp))
        );
        assert_eq!(
            depth_bias(triangles, 1.0, 2.0, -0.5, None, lavapipe.profile()),
            bias(true, 1.0, -0.5, 2.0)
        );
        // No bias to clamp: the clamp is dropped on any device.
        assert_eq!(
            depth_bias(triangles, 0.0, 0.0, 0.5, None, floor.profile()),
            bias(true, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            depth_bias(
                vk::PrimitiveTopology::PATCH_LIST,
                1.0,
                0.0,
                0.0,
                None,
                floor.profile()
            ),
            bias(true, 1.0, 0.0, 0.0)
        );
        // Metal never biases points or lines; Vulkan may, so the pipeline does not enable it.
        for &ty in PrimitiveType::ALL {
            let (topology, _) = input_assembly(ty);
            let triangle = matches!(ty, PrimitiveType::Triangle | PrimitiveType::TriangleStrip);
            let mapped = depth_bias(topology, 1.0, 2.0, 0.5, None, lavapipe.profile()).unwrap();
            assert_eq!(mapped.enable, triangle, "{ty:?}");
            if !triangle {
                assert_eq!(
                    mapped,
                    DepthBias {
                        enable: false,
                        constant_factor: 0.0,
                        clamp: 0.0,
                        slope_factor: 0.0,
                        unit: DepthBiasUnit::Device,
                    }
                );
                assert!(depth_bias(topology, 1.0, 2.0, 0.5, None, floor.profile()).is_ok());
            }
        }
    }

    /// A `Depth16Unorm` depth's constant bias is Metal's `c ± 1` units of 1/65535 as `float`
    /// (one unit for a constant of 0 beside a slope): in depth where the device has
    /// `floatRepresentation` and `depthBiasExact`, in 2^-16 where it has `depthBiasExact` alone,
    /// and Metal's `c` in the device's own unit elsewhere, as on a float depth format.
    #[test]
    fn a_unorm_depth_bias_is_metals_where_the_device_pins_its_unit() {
        let triangles = vk::PrimitiveTopology::TRIANGLE_LIST;
        let d16 = Some(PixelFormat::Depth16Unorm);
        let (lavapipe, nvidia, moltenvk) = (
            committed("lavapipe"),
            committed("nvidia-rtx5080-laptop"),
            committed("moltenvk-m3"),
        );
        let mapped = |device: &Device, constant: f32, format| {
            let bias = depth_bias(triangles, constant, 0.5, 0.0, format, device.profile()).unwrap();
            assert_eq!(bias.slope_factor, 0.5);
            (bias.constant_factor, bias.unit)
        };
        for (constant, units) in [
            (3.0f32, 4.0f64),
            (-3.0, -4.0),
            (0.5, 1.5),
            (-1000.0, -1001.0),
        ] {
            assert_eq!(
                mapped(&lavapipe, constant, d16),
                ((units / 65535.0) as f32, DepthBiasUnit::Depth)
            );
            let (scaled, unit) = mapped(&nvidia, constant, d16);
            assert_eq!(unit, DepthBiasUnit::ExactFormatUnit);
            // A power of two apart: the device's 2^-16 times the factor is the same `float`.
            assert_eq!(scaled / 65536.0, (units / 65535.0) as f32);
            assert_eq!(
                mapped(&moltenvk, constant, d16),
                (constant, DepthBiasUnit::Device)
            );
            for format in [None, Some(PixelFormat::Depth32Float)] {
                assert_eq!(
                    mapped(&lavapipe, constant, format),
                    (constant, DepthBiasUnit::Device)
                );
            }
        }
        // A slope alone (`mapped`'s 0.5) brings one unit; no bias at all brings none.
        assert_eq!(
            mapped(&lavapipe, 0.0, d16),
            ((1.0f64 / 65535.0) as f32, DepthBiasUnit::Depth)
        );
        let none = depth_bias(triangles, 0.0, 0.0, 0.0, d16, lavapipe.profile()).unwrap();
        assert_eq!(
            (none.constant_factor, none.unit),
            (0.0, DepthBiasUnit::Device)
        );
        assert_eq!(mapped(&moltenvk, 0.0, d16), (0.0, DepthBiasUnit::Device));
        assert_eq!(
            mapped(&lavapipe, f32::INFINITY, d16),
            (f32::INFINITY, DepthBiasUnit::Device)
        );
    }

    #[test]
    fn topologies_restart_only_strips() {
        for &ty in PrimitiveType::ALL {
            let (topology, restart) = input_assembly(ty);
            assert_eq!(restart, topology.name().unwrap().ends_with("_STRIP"));
        }
        assert_eq!(index_type(IndexType::UInt32), vk::IndexType::UINT32);
    }
}
