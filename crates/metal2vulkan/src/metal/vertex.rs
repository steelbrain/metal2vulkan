//! Vertex-format and step-function mapping.
//!
//! A Metal vertex attribute is fetched by fixed-function vertex input when its format and step
//! function have an exact Vulkan equivalent on the profile and the profile bounds fixed-function
//! input (buffer robustness, or containment off), and **pulled** (fetched and decoded in the
//! shader from its buffer) otherwise. There is no third representation: widening a 3-component
//! 8/16-bit format to 4 components reads past the attribute, which at the end of a buffer is
//! out of bounds and may replace the whole attribute.

use crate::device::requirements::FormatUsage;
use crate::device::DeviceProfile;
use crate::error::{DescriptorError, Error, Feature, Unsupported};
use crate::memory::contain::{self, Check};
use crate::metal::{VertexFormat, VertexStepFunction};
use crate::vk;

/// The Vulkan format with exactly the layout and numeric conversion of a Metal vertex format.
///
/// Metal's `…Normalized` signed conversion is `max(c / (2^(b-1) - 1), -1)`, as Vulkan's `SNORM`;
/// missing components read as `(0, 0, 0, 1)` in both.
pub fn vk_vertex_format(format: VertexFormat) -> Result<vk::Format, Error> {
    use vk::Format as F;
    use VertexFormat as V;
    Ok(match format {
        V::Invalid => {
            return Err(Error::InvalidDescriptor(
                DescriptorError::PlaceholderEnumValue {
                    ty: VertexFormat::METAL_NAME,
                    value: format.as_raw(),
                },
            ))
        }
        V::UChar => F::R8_UINT,
        V::UChar2 => F::R8G8_UINT,
        V::UChar3 => F::R8G8B8_UINT,
        V::UChar4 => F::R8G8B8A8_UINT,
        V::Char => F::R8_SINT,
        V::Char2 => F::R8G8_SINT,
        V::Char3 => F::R8G8B8_SINT,
        V::Char4 => F::R8G8B8A8_SINT,
        V::UCharNormalized => F::R8_UNORM,
        V::UChar2Normalized => F::R8G8_UNORM,
        V::UChar3Normalized => F::R8G8B8_UNORM,
        V::UChar4Normalized => F::R8G8B8A8_UNORM,
        V::CharNormalized => F::R8_SNORM,
        V::Char2Normalized => F::R8G8_SNORM,
        V::Char3Normalized => F::R8G8B8_SNORM,
        V::Char4Normalized => F::R8G8B8A8_SNORM,
        V::UChar4NormalizedBGRA => F::B8G8R8A8_UNORM,
        V::UShort => F::R16_UINT,
        V::UShort2 => F::R16G16_UINT,
        V::UShort3 => F::R16G16B16_UINT,
        V::UShort4 => F::R16G16B16A16_UINT,
        V::Short => F::R16_SINT,
        V::Short2 => F::R16G16_SINT,
        V::Short3 => F::R16G16B16_SINT,
        V::Short4 => F::R16G16B16A16_SINT,
        V::UShortNormalized => F::R16_UNORM,
        V::UShort2Normalized => F::R16G16_UNORM,
        V::UShort3Normalized => F::R16G16B16_UNORM,
        V::UShort4Normalized => F::R16G16B16A16_UNORM,
        V::ShortNormalized => F::R16_SNORM,
        V::Short2Normalized => F::R16G16_SNORM,
        V::Short3Normalized => F::R16G16B16_SNORM,
        V::Short4Normalized => F::R16G16B16A16_SNORM,
        V::Half => F::R16_SFLOAT,
        V::Half2 => F::R16G16_SFLOAT,
        V::Half3 => F::R16G16B16_SFLOAT,
        V::Half4 => F::R16G16B16A16_SFLOAT,
        V::Float => F::R32_SFLOAT,
        V::Float2 => F::R32G32_SFLOAT,
        V::Float3 => F::R32G32B32_SFLOAT,
        V::Float4 => F::R32G32B32A32_SFLOAT,
        V::Int => F::R32_SINT,
        V::Int2 => F::R32G32_SINT,
        V::Int3 => F::R32G32B32_SINT,
        V::Int4 => F::R32G32B32A32_SINT,
        V::UInt => F::R32_UINT,
        V::UInt2 => F::R32G32_UINT,
        V::UInt3 => F::R32G32B32_UINT,
        V::UInt4 => F::R32G32B32A32_UINT,
        // 10:10:10:2 with x in the low bits (Metal names these from the least significant bit).
        V::Int1010102Normalized => F::A2B10G10R10_SNORM_PACK32,
        V::UInt1010102Normalized => F::A2B10G10R10_UNORM_PACK32,
        V::FloatRG11B10 => F::B10G11R11_UFLOAT_PACK32,
        V::FloatRGB9E5 => F::E5B9G9R9_UFLOAT_PACK32,
    })
}

/// The Vulkan `USCALED`/`SSCALED` format with the layout of `format`, an unnormalized 8- or
/// 16-bit integer Metal vertex format, and `None` for every other format. Fixed-function input
/// delivers each component of such a format to a float attribute as its integer value, exactly
/// (16 bits fit an `f32` significand): Metal's conversion of that format read into a float
/// attribute (`lower::stage_io::fetch::ClassMatch::IntToFloat`).
pub(crate) fn vk_scaled_vertex_format(format: VertexFormat) -> Option<vk::Format> {
    use vk::Format as F;
    use VertexFormat as V;
    Some(match format {
        V::UChar => F::R8_USCALED,
        V::UChar2 => F::R8G8_USCALED,
        V::UChar3 => F::R8G8B8_USCALED,
        V::UChar4 => F::R8G8B8A8_USCALED,
        V::Char => F::R8_SSCALED,
        V::Char2 => F::R8G8_SSCALED,
        V::Char3 => F::R8G8B8_SSCALED,
        V::Char4 => F::R8G8B8A8_SSCALED,
        V::UShort => F::R16_USCALED,
        V::UShort2 => F::R16G16_USCALED,
        V::UShort3 => F::R16G16B16_USCALED,
        V::UShort4 => F::R16G16B16A16_USCALED,
        V::Short => F::R16_SSCALED,
        V::Short2 => F::R16G16_SSCALED,
        V::Short3 => F::R16G16B16_SSCALED,
        V::Short4 => F::R16G16B16A16_SSCALED,
        _ => return None,
    })
}

/// The alignment in bytes Vulkan's fixed-function vertex input requires of an attribute's address
/// in `format` ("Vertex Input Address Calculation", `VUID-vkCmdDraw-format-10389`/`-10390`): the
/// [`vk::FormatBlock::vertex_alignment`] of its Vulkan format, the component size or the whole
/// word of a packed format. It is the one alignment table: link pulls a static layout it does not
/// divide, and the binder checks bind offsets and per-bind strides against the same blocks.
pub fn vertex_alignment(format: VertexFormat) -> Result<u32, Error> {
    let f = vk_vertex_format(format)?;
    f.block().and_then(|b| b.vertex_alignment()).ok_or_else(|| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::Invariant,
            format!("vertex format {format:?} maps to {f:?}, which has no vertex alignment"),
        ))
    })
}

/// The largest [`vertex_alignment`] of any Metal vertex format: a bind offset this divides meets
/// every attribute's alignment.
pub const MAX_VERTEX_ALIGNMENT: u32 = 4;

/// How an attribute's data reaches the shader.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AttributeFetch {
    /// Fixed-function vertex input of this format, with this binding input rate and divisor
    /// (divisor 1 when not divided, and 0 -- the same element for every instance -- for Metal's
    /// constant step, whose binding link emits with a stride of 0).
    FixedFunction {
        format: vk::Format,
        input_rate: vk::VertexInputRate,
        divisor: u32,
    },
    /// Fetched in the shader (`VertexFetch::Pulled`).
    Pulled,
}

/// How an attribute of `format`, in a buffer layout stepped by `step`/`step_rate`, is fetched on
/// `profile`, read into an attribute of the format's own numeric class (an integer format read into
/// a float attribute is fetched in its `USCALED` or `SSCALED` form).
///
/// - `PerVertex`: vertex rate. `PerInstance` with rate 1: instance rate. `PerInstance` with rate
///   `n > 1`: instance rate with divisor `n` when the profile has
///   `vertexAttributeInstanceRateDivisor`, `n ≤ maxVertexAttribDivisor`, and
///   `supportsNonZeroFirstInstance` (a Metal draw may have a nonzero base instance, and without
///   it Vulkan requires `firstInstance == 0` for any divisor other than 1), else pulled. Vulkan
///   fetches at `firstInstance + (instance - firstInstance) / n`, the same element MoltenVK maps
///   Metal's step rate to (expected, not verified on Metal).
/// - `Constant`: instance rate with divisor 0, on a binding `pipeline::link::vertex` emits with a
///   **stride of 0**; pulled where that binding is not available. Metal fetches the layout's first
///   element whatever the base instance, and Vulkan's zero divisor fetches at
///   `firstInstance * stride` ("Vertex Input Extraction":
///   `effectiveVertexOffset = firstInstance * stride` when `divisor` is 0), so the two name the
///   same element for every draw exactly when that binding's stride is 0 -- which it can be,
///   because a fetch that reads one element has no use for the layout's own stride. The binding
///   needs `vertexAttributeInstanceRateZeroDivisor` and `vertexAttributeInstanceRateDivisor`
///   (`VUID-VkVertexInputBindingDivisorDescription-vertexAttributeInstanceRateZeroDivisor-02228`
///   and `-vertexAttributeInstanceRateDivisor-02229`), and, being a divisor other than 1 under a
///   Metal draw that may carry a nonzero base instance, `supportsNonZeroFirstInstance`
///   (VUID-vkCmdDraw-pNext-09461); and a stride of 0 must be legal, which the portability
///   subset's `minVertexInputBindingStrideAlignment` forbids
///   (`VUID-VkVertexInputBindingDescription-stride-04456`: at least as large as the alignment).
///   Under `robustBufferAccess` a stride of 0 bounds the fetch at the bound range itself, which is
///   the attribute extent the pulled fetch reads.
/// - A format the profile cannot fetch as `VERTEX_BUFFER`: pulled.
/// - `FloatRGB9E5` (`E5B9G9R9_UFLOAT_PACK32`): pulled on every profile
///   (`fixed_function_format`).
/// - Every attribute on a profile whose buffer containment is checked in the shader
///   (`contain::encoding` is `Check::InShader`: containment on, neither `robustBufferAccess`
///   nor `robustBufferAccess2` enabled): pulled. Vulkan bounds fixed-function vertex input only
///   under robust buffer access ("Shader Out-of-Bounds Memory Access": the device feature, or
///   `VK_EXT_pipeline_robustness`'s `vertexInputs`, which no profile models); without it a draw
///   must not fetch past a vertex buffer binding (`VUID-vkCmdDraw-None-02721`), and a Metal draw
///   may, so containment moves the fetch into the shader, where it reads through the normative
///   check. Under `Containment::UnsafeOff` the consumer opted out of containment and input stays
///   fixed function.
/// - `PerPatch`/`PerPatchControlPoint`: tessellation, refused. A `PerInstance` rate of 0 is an
///   invalid descriptor.
pub fn attribute_fetch(
    format: VertexFormat,
    step: VertexStepFunction,
    step_rate: u32,
    profile: &DeviceProfile,
) -> Result<AttributeFetch, Error> {
    fetch_as(vk_vertex_format(format)?, step, step_rate, profile)
}

/// [`attribute_fetch`] of an attribute fixed-function input would deliver in `vk_format`:
/// [`vk_vertex_format`]'s, or [`vk_scaled_vertex_format`]'s for an integer format read into a
/// float attribute (`pipeline::link::vertex`).
pub(crate) fn fetch_as(
    vk_format: vk::Format,
    step: VertexStepFunction,
    step_rate: u32,
    profile: &DeviceProfile,
) -> Result<AttributeFetch, Error> {
    let f = profile.fields();
    let (input_rate, divisor) = match step {
        VertexStepFunction::PerVertex => (vk::VertexInputRate::VERTEX, 1),
        VertexStepFunction::PerInstance => match step_rate {
            0 => {
                return Err(Error::InvalidDescriptor(DescriptorError::VertexStepRate {
                    step_rate,
                }))
            }
            1 => (vk::VertexInputRate::INSTANCE, 1),
            // A Metal draw may have a nonzero base instance, so a divisor is usable only where
            // `firstInstance` may then be nonzero (VUID-vkCmdDraw-pNext-09461).
            n if f.fragment.vertex_attribute_divisor
                && f.limits.supports_non_zero_first_instance
                && n <= f.limits.max_vertex_attrib_divisor =>
            {
                (vk::VertexInputRate::INSTANCE, n)
            }
            _ => return Ok(AttributeFetch::Pulled),
        },
        // Metal's constant step reads the layout's first element whatever the base instance, and a
        // zero divisor reads element `firstInstance`: the same element for every draw exactly on a
        // binding of stride 0, which link emits for this fetch and the portability subset forbids.
        // A divisor other than 1 also needs a nonzero `firstInstance` to be legal at all, as a
        // Metal draw may carry a base instance (the `PerInstance` arm above).
        VertexStepFunction::Constant
            if f.fragment.vertex_attribute_divisor
                && f.fragment.vertex_attribute_zero_divisor
                && f.limits.supports_non_zero_first_instance
                && f.limits.portable_vertex_stride(0) =>
        {
            (vk::VertexInputRate::INSTANCE, 0)
        }
        VertexStepFunction::Constant => return Ok(AttributeFetch::Pulled),
        VertexStepFunction::PerPatch | VertexStepFunction::PerPatchControlPoint => {
            return Err(Unsupported::not_implemented(Feature::Tessellation).into())
        }
    };
    // Without buffer robustness nothing bounds fixed-function input (VUID-vkCmdDraw-None-02721).
    if contain::encoding(profile) == Check::InShader {
        return Ok(AttributeFetch::Pulled);
    }
    if !fixed_function_format(vk_format)
        || !profile.supports_format(vk_format, FormatUsage::VertexBuffer)
    {
        return Ok(AttributeFetch::Pulled);
    }
    Ok(AttributeFetch::FixedFunction {
        format: vk_format,
        input_rate,
        divisor,
    })
}

/// Whether fixed-function input delivers an attribute of `vk_format` where the profile reports
/// the format's `VERTEX_BUFFER`: every format but the shared-exponent `E5B9G9R9_UFLOAT_PACK32`
/// (`FloatRGB9E5`), which is pulled on every profile. Vulkan requires no device to fetch that
/// format as vertex input (its mandatory features are a sampled image's, "Mandatory Format
/// Support: 64-bit/uneven Components"), and its decode has no rounding to leave to the device:
/// each component `m · 2^(e - 24)` is an `f32`, and the missing fourth one is 1, Metal's and
/// Vulkan's alike. So the module's decode (`lower::stage_io::fetch`) answers what a conformant
/// fixed-function fetch answers, on a device whose fetch does not (lavapipe's reports the bit
/// and answers 0 in the fourth component and other values in the first three) too.
fn fixed_function_format(vk_format: vk::Format) -> bool {
    vk_format != vk::Format::E5B9G9R9_UFLOAT_PACK32
}

/// Every Vulkan vertex format a Metal vertex format maps to, its own ([`vk_vertex_format`]) or
/// read into a float attribute ([`vk_scaled_vertex_format`]), sorted and without duplicates.
/// Consumers call `FormatSupport::queries`; crate-private.
pub(crate) fn vertex_formats_queried() -> Vec<vk::Format> {
    let mut out: Vec<vk::Format> = VertexFormat::ALL
        .iter()
        .filter_map(|&v| vk_vertex_format(v).ok())
        .chain(
            VertexFormat::ALL
                .iter()
                .filter_map(|&v| vk_scaled_vertex_format(v)),
        )
        .collect();
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::description::FormatProperties;
    use crate::device::requirements::{VkExtension, VkFeature};
    use crate::device::{Device, DevicePolicy};
    use crate::metal::format::tests::committed_descriptions;

    /// Every Metal vertex format has one Vulkan format; the map is one-to-one; components,
    /// widths, and numeric types agree by name.
    #[test]
    fn vertex_formats_round_trip() {
        let mut seen = std::collections::BTreeMap::new();
        for &v in VertexFormat::ALL {
            if v == VertexFormat::Invalid {
                assert!(vk_vertex_format(v).is_err());
                continue;
            }
            let f = vk_vertex_format(v).unwrap();
            assert!(f.is_known());
            assert!(seen.insert(f, v).is_none(), "{v:?} and {:?}", seen[&f]);
            let (mn, vn) = (v.name(), f.name().unwrap());
            let numeric = if mn.contains("Normalized") {
                if mn.contains("UChar") || mn.contains("UShort") || mn.contains("UInt") {
                    "_UNORM"
                } else {
                    "_SNORM"
                }
            } else if mn.contains("Half") || mn.contains("Float") {
                "FLOAT"
            } else if mn.contains("UChar") || mn.contains("UShort") || mn.contains("UInt") {
                "_UINT"
            } else {
                "_SINT"
            };
            assert!(vn.contains(numeric), "{mn} -> {vn}");
            // Component count: a trailing digit in the Metal name, else 1 (packed formats: 4/3).
            let count = if mn.contains("1010102") || mn.contains("BGRA") {
                4
            } else if mn.contains("RG11B10") || mn.contains("RGB9E5") {
                3
            } else {
                mn.chars()
                    .find_map(|c| c.to_digit(10))
                    .map_or(1, |n| n as usize)
            };
            let letters = vn
                .trim_start_matches("VK_FORMAT_")
                .split('_')
                .next()
                .unwrap()
                .chars()
                .filter(|c| matches!(c, 'R' | 'G' | 'B' | 'A'))
                .count();
            assert_eq!(letters, count, "{mn} -> {vn}");
        }
        let scaled = VertexFormat::ALL
            .iter()
            .filter(|&&v| vk_scaled_vertex_format(v).is_some())
            .count();
        assert_eq!(vertex_formats_queried().len(), seen.len() + scaled);
    }

    /// The scaled formats are exactly the unnormalized 8- and 16-bit integer formats', each with
    /// its own format's layout and signedness, and every one is queried.
    #[test]
    fn scaled_formats_keep_the_integer_layout() {
        use crate::lower::stage_io::fetch::{encoding, Encoding};
        let queried = vertex_formats_queried();
        let mut n = 0;
        for &v in VertexFormat::ALL {
            let scaled = vk_scaled_vertex_format(v);
            let Ok(e) = encoding(v) else {
                assert_eq!(scaled, None);
                continue;
            };
            let Encoding::Int {
                bits: 8 | 16,
                signed,
                normalized: false,
                ..
            } = e
            else {
                assert_eq!(scaled, None, "{v:?}");
                continue;
            };
            let scaled = scaled.unwrap_or_else(|| panic!("{v:?} has no scaled format"));
            let own = vk_vertex_format(v).unwrap();
            let name = scaled.name().unwrap();
            let want = own
                .name()
                .unwrap()
                .replace(if signed { "_SINT" } else { "_UINT" }, "");
            let suffix = if signed { "_SSCALED" } else { "_USCALED" };
            assert_eq!(name, format!("{want}{suffix}"), "{v:?}");
            let (sb, ob) = (scaled.block().unwrap(), own.block().unwrap());
            assert_eq!(
                (sb.size, sb.class, sb.component_bits),
                (ob.size, ob.class, ob.component_bits),
                "{v:?}"
            );
            assert!(queried.contains(&scaled), "{v:?}");
            n += 1;
        }
        assert_eq!(n, 16);
    }

    /// The one alignment table: each Metal vertex format's alignment is its component size read
    /// from the byte encoding the pulled fetch decodes (the whole word of a packed format), and
    /// `MAX_VERTEX_ALIGNMENT` is the largest.
    #[test]
    fn vertex_alignment_is_the_component_or_packed_word() {
        use crate::lower::stage_io::fetch::{encoding, Encoding};
        let mut max = 0;
        for &v in VertexFormat::ALL {
            if v == VertexFormat::Invalid {
                assert!(vertex_alignment(v).is_err());
                continue;
            }
            let expected = match encoding(v).unwrap() {
                Encoding::Int { bits, .. } => u32::from(bits / 8),
                Encoding::Half { .. } => 2,
                Encoding::Bgra8Unorm => 1,
                Encoding::Float { .. }
                | Encoding::Rgb10A2 { .. }
                | Encoding::Rg11B10Float
                | Encoding::Rgb9E5Float => 4,
            };
            assert_eq!(vertex_alignment(v).unwrap(), expected, "{v:?}");
            max = max.max(expected);
        }
        assert_eq!(max, MAX_VERTEX_ALIGNMENT);
    }

    fn floor() -> crate::device::DeviceDescription {
        committed_descriptions()
            .into_iter()
            .find(|(s, _)| s == "vk12-floor")
            .unwrap()
            .1
    }

    #[test]
    fn fetch_is_fixed_function_only_when_exact() {
        let mut desc = floor();
        desc.formats.formats.clear();
        desc.formats.formats.insert(
            vk::Format::R32G32B32_SFLOAT,
            FormatProperties {
                buffer: vk::FormatFeatureFlags2::VERTEX_BUFFER,
                ..FormatProperties::default()
            },
        );
        desc.features
            .remove(&VkFeature::VertexAttributeInstanceRateDivisor);
        let plain = Device::new(desc.clone(), DevicePolicy::default()).unwrap();
        desc.features
            .insert(VkFeature::VertexAttributeInstanceRateDivisor);
        // Listed with the extension that exposes it below Vulkan 1.4.
        desc.extensions
            .insert(VkExtension::KhrVertexAttributeDivisor);
        desc.limits.max_vertex_attrib_divisor = 4;
        // No nonzero first instance with a divisor (VUID-vkCmdDraw-pNext-09461): pulled.
        desc.limits.supports_non_zero_first_instance = false;
        let zero_first = Device::new(desc.clone(), DevicePolicy::default()).unwrap();
        desc.limits.supports_non_zero_first_instance = true;
        let divisor = Device::new(desc, DevicePolicy::default()).unwrap();
        use VertexStepFunction as S;
        let fetch = |dev: &Device, f, s, r| attribute_fetch(f, s, r, dev.profile());
        let fixed = |rate, divisor| {
            Ok(AttributeFetch::FixedFunction {
                format: vk::Format::R32G32B32_SFLOAT,
                input_rate: rate,
                divisor,
            })
        };
        let float3 = VertexFormat::Float3;
        assert_eq!(
            fetch(&plain, float3, S::PerVertex, 1),
            fixed(vk::VertexInputRate::VERTEX, 1)
        );
        assert_eq!(
            fetch(&plain, float3, S::PerInstance, 1),
            fixed(vk::VertexInputRate::INSTANCE, 1)
        );
        assert_eq!(
            fetch(&plain, float3, S::PerInstance, 3),
            Ok(AttributeFetch::Pulled)
        );
        assert_eq!(
            fetch(&divisor, float3, S::PerInstance, 3),
            fixed(vk::VertexInputRate::INSTANCE, 3)
        );
        assert_eq!(
            fetch(&zero_first, float3, S::PerInstance, 3),
            Ok(AttributeFetch::Pulled)
        );
        assert_eq!(
            fetch(&zero_first, float3, S::PerInstance, 1),
            fixed(vk::VertexInputRate::INSTANCE, 1)
        );
        assert_eq!(
            fetch(&divisor, float3, S::PerInstance, 5),
            Ok(AttributeFetch::Pulled)
        );
        assert_eq!(
            fetch(&divisor, float3, S::Constant, 0),
            Ok(AttributeFetch::Pulled)
        );
        // Not a vertex-buffer format on this device.
        assert_eq!(
            fetch(&plain, VertexFormat::UChar3, S::PerVertex, 1),
            Ok(AttributeFetch::Pulled)
        );
        assert!(matches!(
            fetch(&plain, float3, S::PerInstance, 0),
            Err(Error::InvalidDescriptor(DescriptorError::VertexStepRate {
                step_rate: 0
            }))
        ));
        assert_eq!(
            fetch(&plain, float3, S::PerPatch, 1),
            Err(Unsupported::not_implemented(Feature::Tessellation).into())
        );
    }

    /// `FloatRGB9E5` is pulled at every step on a device that reports `E5B9G9R9_UFLOAT_PACK32`'s
    /// `VERTEX_BUFFER` (`fixed_function_format`), while `FloatRG11B10`, the other packed float
    /// format, stays fixed function there.
    #[test]
    fn shared_exponent_attributes_are_pulled_where_the_device_fetches_them() {
        let mut desc = floor();
        for f in [
            vk::Format::E5B9G9R9_UFLOAT_PACK32,
            vk::Format::B10G11R11_UFLOAT_PACK32,
        ] {
            desc.formats.formats.insert(
                f,
                FormatProperties {
                    buffer: vk::FormatFeatureFlags2::VERTEX_BUFFER,
                    ..FormatProperties::default()
                },
            );
        }
        let dev = Device::new(desc, DevicePolicy::default()).unwrap();
        assert!(dev.profile().supports_format(
            vk::Format::E5B9G9R9_UFLOAT_PACK32,
            FormatUsage::VertexBuffer
        ));
        use VertexStepFunction as S;
        for (step, rate) in [(S::PerVertex, 1), (S::PerInstance, 1)] {
            assert_eq!(
                attribute_fetch(VertexFormat::FloatRGB9E5, step, rate, dev.profile()),
                Ok(AttributeFetch::Pulled),
                "{step:?}"
            );
            assert!(matches!(
                attribute_fetch(VertexFormat::FloatRG11B10, step, rate, dev.profile()),
                Ok(AttributeFetch::FixedFunction {
                    format: vk::Format::B10G11R11_UFLOAT_PACK32,
                    ..
                })
            ));
        }
    }

    /// Without buffer robustness Vulkan does not bound fixed-function vertex input
    /// (`VUID-vkCmdDraw-None-02721`), so every attribute is pulled and read through the in-shader
    /// check, whatever its format and step; with `robustBufferAccess` or `robustBufferAccess2` it
    /// stays fixed function, and so it does under `Containment::UnsafeOff`, where the consumer
    /// opted out. The descriptor errors and the tessellation refusal come first in every case.
    #[test]
    fn fixed_function_input_needs_buffer_robustness() {
        use crate::device::policy::ContainmentPolicy;
        use crate::device::profile::GuestIsolation;
        use VertexStepFunction as S;
        let robust = floor();
        assert!(robust.features.contains(&VkFeature::RobustBufferAccess));
        let mut bare = robust.clone();
        bare.features.remove(&VkFeature::RobustBufferAccess);
        bare.features.remove(&VkFeature::RobustBufferAccess2);
        let unsafe_off = DevicePolicy {
            containment: ContainmentPolicy::UnsafeOff,
            isolation: GuestIsolation::Off,
            ..DevicePolicy::UNTRUSTED
        };
        let checked = Device::new(bare.clone(), DevicePolicy::default()).unwrap();
        let off = Device::new(bare, unsafe_off).unwrap();
        let robust = Device::new(robust, DevicePolicy::default()).unwrap();
        assert_eq!(contain::encoding(checked.profile()), Check::InShader);
        assert_eq!(contain::encoding(robust.profile()), Check::None);
        assert_eq!(contain::encoding(off.profile()), Check::None);
        let float4 = VertexFormat::Float4;
        for (step, rate) in [(S::PerVertex, 1), (S::PerInstance, 1)] {
            let fetch = |dev: &Device| attribute_fetch(float4, step, rate, dev.profile());
            assert_eq!(fetch(&checked), Ok(AttributeFetch::Pulled), "{step:?}");
            for dev in [&robust, &off] {
                assert!(
                    matches!(fetch(dev), Ok(AttributeFetch::FixedFunction { .. })),
                    "{step:?}: {:?}",
                    fetch(dev)
                );
            }
        }
        assert!(matches!(
            attribute_fetch(float4, S::PerInstance, 0, checked.profile()),
            Err(Error::InvalidDescriptor(DescriptorError::VertexStepRate {
                step_rate: 0
            }))
        ));
        assert_eq!(
            attribute_fetch(float4, S::PerPatchControlPoint, 1, checked.profile()),
            Err(Unsupported::not_implemented(Feature::Tessellation).into())
        );
    }

    /// Vulkan's mandatory vertex-buffer formats are fixed function on every committed profile
    /// that reports them.
    #[test]
    fn mandatory_vertex_formats_are_fixed_function() {
        for (stem, desc) in committed_descriptions() {
            if !desc
                .formats
                .formats
                .contains_key(&vk::Format::R32G32B32A32_SFLOAT)
            {
                continue;
            }
            let dev = Device::new(desc, DevicePolicy::default()).unwrap();
            for v in [
                VertexFormat::Float,
                VertexFormat::Float2,
                VertexFormat::Float3,
                VertexFormat::Float4,
                VertexFormat::UChar4Normalized,
                VertexFormat::Half2,
            ] {
                let got = attribute_fetch(v, VertexStepFunction::PerVertex, 1, dev.profile());
                assert!(
                    matches!(got, Ok(AttributeFetch::FixedFunction { .. })),
                    "{stem} {v:?}: {got:?}"
                );
            }
        }
    }
}
