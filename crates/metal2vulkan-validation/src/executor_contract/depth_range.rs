//! The 32-bit float depths a case gives a depth image, and whether the engine's candidate can put
//! them there as Metal holds them: Metal's `Depth32Float` texel holds any `float`, and Vulkan takes
//! no depth outside `[0, 1]` into a depth image without `VK_EXT_depth_range_unrestricted`
//! (`VUID-vkCmdCopyBufferToImage-pRegions-07931` for a copy; a render write is clamped to the
//! viewport's depth range). On a device without the extension, a case whose depth texels or
//! initial depth hold such a value stops the candidate
//! ([`ToolingRequirement::DepthOutsideUnitRange`]): its executor neither copies the value as it is
//! (an invalid copy) nor clamps it silently (an answer that is not the case's). The Metal oracle
//! runs every such case, and a device with the extension takes every value.

use crate::requirement::ToolingRequirement;

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
        .chunks_exact(4)
        .any(|word| !in_unit_range(u32::from_le_bytes([word[0], word[1], word[2], word[3]])))
}

/// The requirement that stops the engine's candidate from giving a depth image `depths` on a
/// device that enabled `VK_EXT_depth_range_unrestricted` (`unrestricted`) or not: none where it
/// did, or where every depth lies in `[0, 1]`.
pub fn depth_range_requirement(depths: &[u8], unrestricted: bool) -> Option<ToolingRequirement> {
    (!unrestricted && holds_depth_outside_unit_range(depths))
        .then_some(ToolingRequirement::DepthOutsideUnitRange)
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

    /// A case whose depths all lie in `[0, 1]` runs on every device; one with a single depth
    /// outside it stops the candidate where the device lacks the extension, and only there. A
    /// trailing partial word is no depth.
    #[test]
    fn a_depth_outside_the_unit_range_stops_the_candidate_only_without_the_extension() {
        let inside = words(&[0, 0x8000_0000, 0x3f80_0000, 0x0000_0001]);
        assert_eq!(depth_range_requirement(&inside, false), None);
        assert_eq!(depth_range_requirement(&inside, true), None);
        let mut partial = inside.clone();
        partial.extend([0xff, 0xff, 0xff]);
        assert_eq!(depth_range_requirement(&partial, false), None);
        for outside in [0x7fc0_0000, 0xbf00_0000, 0x4000_0000, 0x8000_0001] {
            let depths = words(&[0x3f00_0000, outside, 0]);
            assert_eq!(
                depth_range_requirement(&depths, false),
                Some(ToolingRequirement::DepthOutsideUnitRange),
                "{outside:#010x}"
            );
            assert_eq!(
                depth_range_requirement(&depths, true),
                None,
                "{outside:#010x}"
            );
        }
    }

    /// The requirement stops the engine's candidate alone: the Metal oracle holds every `float`
    /// in a `Depth32Float` texel, so it qualifies the case, and the candidate's run on a device
    /// with the extension is compared with it. It is a gap of this crate's executor, never Metal's.
    #[test]
    fn a_depth_outside_the_unit_range_stops_the_candidate_and_never_the_oracle() {
        let requirement = ToolingRequirement::DepthOutsideUnitRange;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert!(!requirement.stops_executor(Executor::MetalOracle));
        assert_eq!(requirement.limit(), Limit::Tooling);
        assert_eq!(requirement.as_str(), "depth_outside_unit_range");
        let gap = super::super::evidence_gap(&[requirement].into()).unwrap();
        assert!(
            gap.contains("engine_candidate: depth_outside_unit_range"),
            "{gap}"
        );
    }
}
