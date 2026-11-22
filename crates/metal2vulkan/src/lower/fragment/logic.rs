//! The formats of a render pipeline's private logic operation (`pipeline::private_state`): those
//! whose channels Metal's operation takes per channel ([`channels`]), and where Vulkan's `logicOp`
//! may store another code than Metal's ([`vulkan_differs`]).
//!
//! Metal applies the operation to the code of each channel of a normalized or integer attachment
//! the function writes, the fragment's code against the stored one, blending ignored and the write
//! mask kept after it (`pipeline::private_state`'s module docs). A signed normalized channel reads
//! its lowest code, `-2^(n-1)`, as -1, as it reads the code above it, and stores a result of the
//! lowest code as -1, the code above (probe `logic_operation`, an Apple M3 Max under macOS
//! 27.0.1); Vulkan's `logicOp` takes and stores the lowest code itself. A 10-bit unorm format
//! (`RGB10A2Unorm`, `BGR10A2Unorm`) operates on 16-bit codes: Metal widens each stored code to 16
//! bits (`round(d * 65535 / max)`), operates on it and the output's 16-bit code, and stores the
//! result narrowed back to the channel's code, rounded to nearest (the same probe), where Vulkan's
//! operation takes the channel's own code.
//!
//! The formats Metal's operation takes per channel ([`channels`]): every channel an 8-, 10- or
//! 16-bit unorm one, an 8- or 16-bit snorm one, or an 8-, 16- or 32-bit integer one. A device
//! without `logicOp` would have the function compute the operation over an ordered read of the
//! attachment, which is not built: `pipeline::private_state` refuses the pipeline there naming
//! `logicOp`.

use crate::metal::format::{write_conversion, WriteConversion};
use crate::metal::PixelFormat;
use crate::pipeline::lower_inputs::ChannelWrite;
use crate::pipeline::sym::SampledType;

/// How an attachment's channels hold their codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicNumeric {
    Unorm,
    Snorm,
    Uint,
    Sint,
}

/// The channels of an attachment Metal's logic operation takes per channel ([`channels`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicChannels {
    pub numeric: LogicNumeric,
    /// The bits of the codes the operation takes in Metal's red, green, blue and alpha: each
    /// channel's own, but 16 for a 10-bit format's (see the module docs); `None` where the format
    /// lacks one.
    pub bits: [Option<u32>; 4],
}

/// The channels of a color attachment of `pf` Metal's logic operation takes per channel (see the
/// module docs): every channel an 8-, 10- or 16-bit unorm one (a 10-bit format's alpha 2-bit),
/// an 8- or 16-bit snorm one, or an 8-, 16- or 32-bit integer one; `None` for another format.
pub(crate) fn channels(pf: PixelFormat) -> Option<LogicChannels> {
    match write_conversion(pf)? {
        WriteConversion::Rules(rules) => {
            let numeric = match rules.iter().flatten().next()? {
                ChannelWrite::Unorm { .. } => LogicNumeric::Unorm,
                ChannelWrite::Snorm { .. } => LogicNumeric::Snorm,
                _ => return None,
            };
            let mut bits = [None; 4];
            for (b, rule) in bits.iter_mut().zip(rules) {
                *b = match (numeric, rule) {
                    (_, None) => None,
                    // A 10-bit format's channels, its 2-bit alpha too, operate on 16-bit codes.
                    (LogicNumeric::Unorm, Some(ChannelWrite::Unorm { bits: 10 | 2 })) => Some(16),
                    (LogicNumeric::Unorm, Some(ChannelWrite::Unorm { bits: n @ (8 | 16) }))
                    | (
                        LogicNumeric::Snorm,
                        Some(ChannelWrite::Snorm {
                            bits: n @ (8 | 16), ..
                        }),
                    ) => Some(u32::from(n)),
                    _ => return None,
                };
            }
            Some(LogicChannels { numeric, bits })
        }
        WriteConversion::Exact => {
            let numeric = match crate::metal::format::sampled_type(pf)? {
                SampledType::Uint => LogicNumeric::Uint,
                SampledType::Sint => LogicNumeric::Sint,
                _ => return None,
            };
            let block = crate::metal::format::direct_format(pf)?.block()?;
            let width = block.component_bits.filter(|b| matches!(b, 8 | 16 | 32))?;
            let n = crate::metal::format::color_output_components(pf)?;
            let mut bits = [None; 4];
            for b in bits.iter_mut().take(usize::from(n)) {
                *b = Some(width);
            }
            Some(LogicChannels { numeric, bits })
        }
        _ => None,
    }
}

/// Whether Vulkan's `logicOp` into an attachment of `pf` may store another code than Metal's
/// operation (see the module docs), as it operates on the channels' stored codes themselves: a
/// 10-bit format's, whose codes Metal widens to 16 bits ([`widens`]), and a signed normalized
/// one's, whose lowest code Metal reads as the code above it and whose result of the lowest code
/// it stores as the code above, where Vulkan's operation takes and stores the lowest code itself
/// (the Vulkan specification's logical operations take the attachment's component value).
pub(crate) fn vulkan_differs(pf: PixelFormat) -> bool {
    channels(pf).is_some_and(|c| c.numeric == LogicNumeric::Snorm) || widens(pf)
}

/// Whether Metal's operation into an attachment of `pf` takes codes wider than its channels' own
/// (a 10-bit format's, widened to 16 bits: see the module docs).
fn widens(pf: PixelFormat) -> bool {
    let (Some(c), Some(WriteConversion::Rules(rules))) = (channels(pf), write_conversion(pf))
    else {
        return false;
    };
    c.bits
        .iter()
        .zip(rules)
        .any(|(bits, rule)| match (bits, rule) {
            (Some(bits), Some(ChannelWrite::Unorm { bits: own })) => *bits != u32::from(own),
            _ => false,
        })
}

#[cfg(test)]
mod tests;
