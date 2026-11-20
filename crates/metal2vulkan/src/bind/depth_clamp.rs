//! A 32-bit float depth clamped into `[0, 1]` on its bits, so no denormal is flushed
//! ([`clamped_depth`]): a NaN of either sign is 1.0 (`0x3f800000`), `-0.0` stays, every other
//! value with its sign set is 0, a value above 1.0 (`+inf` among them) is 1.0, and every other
//! value stays. A copy into a depth image of a device without `VK_EXT_depth_range_unrestricted`
//! takes no value outside `[0, 1]` (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), and such a
//! device stores one clamped.

/// The bits of 32-bit float depth `bits` clamped into `[0, 1]`: a NaN 1.0,
/// `-0.0` as it is, another negative value 0, a value above 1.0 1.0, and the rest as they are.
pub const fn clamped_depth(bits: u32) -> u32 {
    const ONE: u32 = 0x3f80_0000;
    const SIGN: u32 = 0x8000_0000;
    if bits & !SIGN > 0x7f80_0000 {
        ONE
    } else if bits == SIGN {
        bits
    } else if bits > SIGN {
        0
    } else if bits > ONE {
        ONE
    } else {
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::clamped_depth;

    /// The clamp of each kind of value: inside `[0, 1]` (a denormal and `-0.0` among them) as it
    /// is, a NaN of either sign 1.0, an infinity or a value above 1.0 1.0, and every other
    /// negative value 0.
    #[test]
    fn the_clamp_keeps_unit_range_bits_and_clamps_the_rest() {
        let one = 1.0f32.to_bits();
        for (bits, want) in [
            (0, 0),
            ((-0.0f32).to_bits(), (-0.0f32).to_bits()),
            (1, 1),
            (0.5f32.to_bits(), 0.5f32.to_bits()),
            (one, one),
            (1.5f32.to_bits(), one),
            (f32::INFINITY.to_bits(), one),
            (0x7fc0_0000, one),
            (0x7f80_0001, one),
            (0xffc0_0000, one),
            ((-1.0f32).to_bits(), 0),
            (f32::NEG_INFINITY.to_bits(), 0),
            (0x8000_0001, 0),
            (0xeeee_eeee, 0),
        ] {
            assert_eq!(clamped_depth(bits), want, "{bits:#010x}");
        }
    }
}
