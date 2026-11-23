//! The 32-bit float depths a case gives a depth image, and whether the engine's candidate can put
//! them there as Metal holds them: Metal's `Depth32Float` texel holds any `float`, the runtime's
//! upload leaves a depth outside `[0, 1]` the device's
//! (`metal2vulkan_runtime::Texture::FLOAT_DEPTH_UPLOAD_NOTE`), and the candidate fills one by a
//! draw, which writes depths clamped to the viewport's `[0, 1]`. A
//! case whose depth texels or initial depth hold a value outside it stops the candidate on every
//! device ([`ToolingRequirement::DepthOutsideUnitRange`], the runtime's limit): it neither uploads
//! the value nor clamps it silently (an answer that is not the case's). The Metal oracle runs every
//! such case.

use crate::case::{AuthoredCase, TextureFormat};
use crate::requirement::ToolingRequirement;
use base64::Engine as _;

/// Whether the 32-bit float depth `bits` lies in `[0, 1]`: `-0.0` and the positive denormals do,
/// a NaN of either sign, a negative value (a negative denormal included) and a value above 1.0
/// (`+inf` included) do not, as `metal2vulkan::bind::clamped_depth` leaves exactly the values in
/// the range as they are.
pub fn in_unit_range(bits: u32) -> bool {
    metal2vulkan::bind::clamped_depth(bits) == bits
}

/// Whether `depths`, the little-endian 32-bit float words a case gives a depth image (a
/// `Depth32Float` texture's texels, every level and sample, or a depth attachment's initial
/// depth), holds a depth outside `[0, 1]`. A trailing partial word holds no depth.
pub fn holds_depth_outside_unit_range(depths: &[u8]) -> bool {
    depths
        .as_chunks::<4>()
        .0
        .iter()
        .any(|word| !in_unit_range(u32::from_le_bytes(*word)))
}

/// The requirement `case` raises by the depths it gives depth images, on every device: each
/// `Depth32Float` texture's texels (a texture-array element's and an argument buffer's texture's
/// among them) and its depth attachment's initial depth. None where every depth lies in `[0, 1]`.
pub fn depth_range_requirement(case: &AuthoredCase) -> Option<ToolingRequirement> {
    let depth = TextureFormat::Depth32Float;
    let textures = (case.textures.iter())
        .filter(|texture| texture.format == depth)
        .map(|texture| decoded(&texture.bytes_b64, &texture.initial_bytes_b64));
    let elements = (case.texture_arrays.iter())
        .filter(|array| array.format == depth)
        .flat_map(|array| &array.elements)
        .map(|element| decoded(&element.bytes_b64, &element.initial_bytes_b64));
    let members = (case.argument_buffer_textures.iter())
        .filter(|texture| texture.format == depth)
        .map(|texture| decoded(&texture.bytes_b64, &texture.initial_bytes_b64));
    let attachment =
        (case.depth_stencil.iter()).map(|attachment| decoded(&attachment.initial_depth_b64, &None));
    textures
        .chain(elements)
        .chain(members)
        .chain(attachment)
        .any(|depths| holds_depth_outside_unit_range(&depths))
        .then_some(ToolingRequirement::DepthOutsideUnitRange)
}

/// The bytes a literal states (`bytes`, else `initial`), none where neither decodes: the case's
/// own check refuses those.
fn decoded(bytes: &Option<String>, initial: &Option<String>) -> Vec<u8> {
    (bytes.as_ref().or(initial.as_ref()))
        .and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requirement::{Executor, Limit};

    fn words(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// The range is Metal's texel's `[0, 1]` read as Vulkan's copy check reads it: `-0.0`, zero,
    /// the denormals of the positive sign, one half and one lie in it; a NaN of either sign, the
    /// infinities, a negative value of any size and a value past one do not.
    #[test]
    fn a_depth_lies_in_the_unit_range_exactly_as_the_clamp_leaves_it() {
        for bits in [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x3f00_0000,
            0x3f80_0000,
            0x3f7f_ffff,
        ] {
            assert!(in_unit_range(bits), "{bits:#010x}");
        }
        for bits in [
            0x7fc0_0000,
            0xffc0_0000,
            0x7f80_0000,
            0xff80_0000,
            0x8000_0001,
            0xbf00_0000,
            0x3f80_0001,
            0x447a_0000,
        ] {
            assert!(!in_unit_range(bits), "{bits:#010x}");
        }
    }

    /// A case whose depths all lie in `[0, 1]` raises nothing; one depth outside it, in a
    /// `Depth32Float` texture, a texture-array element, an argument buffer's texture or the depth
    /// attachment's initial depth, raises the requirement. A texture of another format holds no
    /// depth, and a trailing partial word is none.
    #[test]
    fn a_depth_outside_the_unit_range_anywhere_a_case_gives_one_raises_the_requirement() {
        use crate::case::{
            ArgumentBufferTextureResource, DepthStencilResource, ResourceRole, TextureArrayElement,
            TextureArrayResource, TextureResource, TextureType,
        };
        let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
        let inside = words(&[0, 0x8000_0000, 0x3f80_0000, 0x0000_0001]);
        let outside = words(&[0x3f00_0000, 0x4000_0000, 0, 0]);
        let mut partial = inside.clone();
        partial.extend([0xff, 0xff, 0xff]);
        let texture = |format, bytes: &[u8]| TextureResource {
            binding: 0,
            role: ResourceRole::Input,
            texture_type: TextureType::D2,
            format,
            dimensions: [2, 2, 1],
            sample_count: 1,
            bytes_b64: Some(b64(bytes)),
            initial_bytes_b64: None,
            sparse: None,
            mip_levels: 1,
        };
        let case = |textures| AuthoredCase {
            textures,
            ..crate::case::vector_function_constant_test_case("11".repeat(32), "main".into())
        };
        let depth = TextureFormat::Depth32Float;
        for clean in [
            case(vec![texture(depth, &inside)]),
            case(vec![texture(depth, &partial)]),
            case(vec![texture(TextureFormat::R32Float, &outside)]),
        ] {
            assert_eq!(
                depth_range_requirement(&clean),
                None,
                "{:?}",
                clean.textures
            );
        }
        let required = Some(ToolingRequirement::DepthOutsideUnitRange);
        assert_eq!(
            depth_range_requirement(&case(vec![texture(depth, &outside)])),
            required
        );
        let mut array = case(Vec::new());
        array.texture_arrays = vec![TextureArrayResource {
            binding: 0,
            role: ResourceRole::Input,
            texture_type: TextureType::D2,
            format: depth,
            sample_count: 1,
            overrides_texture_at_base: false,
            elements: vec![TextureArrayElement {
                dimensions: [2, 2, 1],
                bytes_b64: None,
                initial_bytes_b64: Some(b64(&outside)),
            }],
        }];
        assert_eq!(depth_range_requirement(&array), required);
        let mut member = case(Vec::new());
        member.argument_buffer_textures = vec![ArgumentBufferTextureResource {
            buffer_binding: 0,
            field_offset: 0,
            role: ResourceRole::Input,
            texture_type: TextureType::D2,
            format: depth,
            dimensions: [2, 2, 1],
            sample_count: 1,
            bytes_b64: Some(b64(&outside)),
            initial_bytes_b64: None,
        }];
        assert_eq!(depth_range_requirement(&member), required);
        let mut attachment = case(Vec::new());
        attachment.depth_stencil = Some(DepthStencilResource {
            dimensions: [2, 2],
            initial_depth_b64: Some(b64(&outside)),
            initial_stencil_b64: None,
            state: None,
        });
        assert_eq!(depth_range_requirement(&attachment), required);
    }

    /// The requirement stops the engine's candidate alone: the Metal oracle holds every `float`
    /// in a `Depth32Float` texel, so it qualifies the case, and the runtime puts no such depth in
    /// an image, the runtime's limit, which closes its rows.
    #[test]
    fn a_depth_outside_the_unit_range_stops_the_candidate_and_never_the_oracle() {
        let requirement = ToolingRequirement::DepthOutsideUnitRange;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert!(!requirement.stops_executor(Executor::MetalOracle));
        assert_eq!(requirement.limit(), Limit::Runtime);
        assert!(requirement.qualifies_on_metal());
        assert!(!requirement.blocks_closure());
        assert_eq!(requirement.as_str(), "depth_outside_unit_range");
        assert_eq!(super::super::evidence_gap(&[requirement].into()), None);
    }
}
