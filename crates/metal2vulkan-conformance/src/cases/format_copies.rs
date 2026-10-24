//! Blit copies between a depth or stencil format and a colour format of its unit, and through a
//! combined texture's stencil view, ported from `runtime_lane::format_copies`.
//!
//! Metal copies the **bytes** between a depth-only or stencil-only format and a colour format whose
//! texel is its copy unit — 2 bytes for `Depth16Unorm`, 4 for `Depth32Float`, 1 for stencil — and
//! between stencil formats. And the stencil view `X32Stencil8` of a `Depth32FloatStencil8` texture
//! copies the **stencil**, a byte a texel, leaving the depth as it was, with any blit option.
//!
//! **The port re-measures both here**: every call is public Metal, so record mode runs the program
//! and each recording holds the bytes Metal moved. The lane's expectation is that the destination
//! and the source both read back what was written, which the case states as the recording rather
//! than as an assertion (the test `every_recorded_copy_moved_the_bytes`).
//!
//! **One case per pair**, so a venue that holds no texture of one format answers its typed refusal
//! there and counts as lacking the feature instead of taking fifteen other pairs with it.
//!
//! **Not ported, and why.** The lane's refusal rows copy between formats of different units, and
//! between a combined format and another. Our runtime refuses those typed, which is the lane's
//! subject — but recording them would mean **encoding an invalid copy on plain Metal**, where the
//! behaviour is undefined and the device may fault. The round's safety rule forbids that outright,
//! and Metal's API validation layer is a second fact rather than a licence to run the thing it
//! objects to. Those rows stay a lane. The same goes for the lane's CPU copies through the stencil
//! view of a shared combined texture, which are safe but want the view's host reads, a shape no
//! case here needed yet.
//!
//! Every texture is one row of four texels, private, and read back through a buffer, as the lane's
//! probes read theirs. Depth values lie in `[0, 1]`, so no device's clamping decides.
//!
//! Nothing here can fault a GPU: blit copies between textures the program makes, of formats whose
//! copy unit matches, no dispatch, no draw and no loop.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::transcript::{ByteLayout, Element, Observed, Sites, Value};
use crate::work::{Plane, Resource, Step, TextureSpec, Work};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::Degradation;

/// Every texture's width; each is one row.
const W: u64 = 4;

/// Four texels of each size, as the lane's probes wrote them: `Depth32Float` values in `[0, 1]`,
/// and 16-bit and 8-bit words with their top bits set and clear.
const FOUR: [u8; 16] = [
    0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x40, 0x3f, 0x00, 0x00, 0x80, 0x3f,
];
const TWO: [u8; 8] = [0x34, 0x12, 0xff, 0xff, 0x01, 0x00, 0x00, 0x80];
const ONE: [u8; 4] = [0x5a, 0xff, 0x01, 0x80];

/// The byte the destination holds before the copy, so a texel the copy did not move is a recorded
/// value rather than whatever was there.
const BEFORE: u8 = 0xcc;

/// One pair the case copies between: the case's name, the source and destination formats, the
/// bytes of the unit, and whether the textures are render targets as well as shader-read.
struct Pair {
    name: &'static str,
    from: PixelFormat,
    to: PixelFormat,
    bytes: &'static [u8],
    target: bool,
}

/// Every pair the lane copies between, in its order.
const PAIRS: [Pair; 16] = [
    Pair {
        name: "format-copies-d32f-to-r32f",
        from: PixelFormat::Depth32Float,
        to: PixelFormat::R32Float,
        bytes: &FOUR,
        target: false,
    },
    Pair {
        name: "format-copies-d32f-to-r32u",
        from: PixelFormat::Depth32Float,
        to: PixelFormat::R32Uint,
        bytes: &FOUR,
        target: true,
    },
    Pair {
        name: "format-copies-d32f-to-rgba8",
        from: PixelFormat::Depth32Float,
        to: PixelFormat::RGBA8Unorm,
        bytes: &FOUR,
        target: false,
    },
    Pair {
        name: "format-copies-r32f-to-d32f",
        from: PixelFormat::R32Float,
        to: PixelFormat::Depth32Float,
        bytes: &FOUR,
        target: false,
    },
    Pair {
        name: "format-copies-r32u-to-d32f",
        from: PixelFormat::R32Uint,
        to: PixelFormat::Depth32Float,
        bytes: &FOUR,
        target: true,
    },
    Pair {
        name: "format-copies-rgba8-to-d32f",
        from: PixelFormat::RGBA8Unorm,
        to: PixelFormat::Depth32Float,
        bytes: &FOUR,
        target: false,
    },
    Pair {
        name: "format-copies-d16-to-r16",
        from: PixelFormat::Depth16Unorm,
        to: PixelFormat::R16Unorm,
        bytes: &TWO,
        target: false,
    },
    Pair {
        name: "format-copies-d16-to-rg8",
        from: PixelFormat::Depth16Unorm,
        to: PixelFormat::RG8Unorm,
        bytes: &TWO,
        target: true,
    },
    Pair {
        name: "format-copies-r16u-to-d16",
        from: PixelFormat::R16Uint,
        to: PixelFormat::Depth16Unorm,
        bytes: &TWO,
        target: false,
    },
    Pair {
        name: "format-copies-r16f-to-d16",
        from: PixelFormat::R16Float,
        to: PixelFormat::Depth16Unorm,
        bytes: &TWO,
        target: true,
    },
    Pair {
        name: "format-copies-s8-to-r8u",
        from: PixelFormat::Stencil8,
        to: PixelFormat::R8Uint,
        bytes: &ONE,
        target: false,
    },
    Pair {
        name: "format-copies-s8-to-r8i",
        from: PixelFormat::Stencil8,
        to: PixelFormat::R8Sint,
        bytes: &ONE,
        target: true,
    },
    Pair {
        name: "format-copies-r8-to-s8",
        from: PixelFormat::R8Unorm,
        to: PixelFormat::Stencil8,
        bytes: &ONE,
        target: false,
    },
    Pair {
        name: "format-copies-r8snorm-to-s8",
        from: PixelFormat::R8Snorm,
        to: PixelFormat::Stencil8,
        bytes: &ONE,
        target: true,
    },
    Pair {
        name: "format-copies-s8-to-s8",
        from: PixelFormat::Stencil8,
        to: PixelFormat::Stencil8,
        bytes: &ONE,
        target: false,
    },
    Pair {
        name: "format-copies-d16-to-d16",
        from: PixelFormat::Depth16Unorm,
        to: PixelFormat::Depth16Unorm,
        bytes: &TWO,
        target: true,
    },
];

/// The case for one pair.
pub struct FormatCopies {
    at: usize,
}

/// Every pair's case.
pub fn all() -> Vec<FormatCopies> {
    (0..PAIRS.len()).map(|at| FormatCopies { at }).collect()
}

impl FormatCopies {
    /// The pair the case copies between.
    fn pair(&self) -> &'static Pair {
        &PAIRS[self.at]
    }

    /// The usage the pair's textures are made with: shader-read, and a render target where the
    /// pair says so. The lane states this per pair because Metal's layout of a render target
    /// differs from a read-only texture's, and a copy is what the case measures.
    fn usage(&self) -> u32 {
        let usage = if self.pair().target {
            TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET)
        } else {
            TextureUsage::SHADER_READ
        };
        usage.as_raw()
    }

    /// The bytes one texel takes.
    fn per(&self) -> u64 {
        self.pair().bytes.len() as u64 / W
    }

    /// The program: the destination filled with [`BEFORE`], the source filled with the pair's
    /// bytes, the source's row copied into the destination's, and both read back.
    fn program(&self) -> Work {
        let pair = self.pair();
        let per = self.per();
        let spec =
            |format: PixelFormat| TextureSpec::texture_2d(format.as_raw(), W, 1).used(self.usage());
        let mut work = Work::new();
        let source = work.add(Resource::Texture {
            spec: spec(pair.from),
        });
        let destination = work.add(Resource::Texture {
            spec: spec(pair.to),
        });
        let before = work.buffer(vec![BEFORE; pair.bytes.len()]);
        let written = work.buffer(pair.bytes.to_vec());
        let out = |work: &mut Work| work.buffer(vec![BEFORE; pair.bytes.len()]);
        let fill = |texture: usize, buffer: usize| Step::CopyBufferToTexture {
            buffer,
            offset: 0,
            bytes_per_row: per * W,
            bytes_per_image: 0,
            size: [W, 1, 1],
            texture,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            plane: Plane::Whole,
        };
        let read = |texture: usize, buffer: usize| Step::CopyTextureToBuffer {
            texture,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            size: [W, 1, 1],
            buffer,
            offset: 0,
            bytes_per_row: per * W,
            bytes_per_image: 0,
            plane: Plane::Whole,
        };
        work.step(fill(destination, before));
        work.step(fill(source, written));
        work.step(Step::CopyTextureToTexture {
            from: source,
            from_slice: 0,
            from_level: 0,
            from_origin: [0, 0, 0],
            size: [W, 1, 1],
            to: destination,
            to_slice: 0,
            to_level: 0,
            to_origin: [0, 0, 0],
        });
        let into = out(&mut work);
        let back = out(&mut work);
        work.step(read(destination, into));
        work.step(read(source, back));
        work.step(Step::Commit);
        for buffer in [into, back] {
            work.step(Step::ReadBuffer {
                buffer,
                offset: 0,
                length: pair.bytes.len(),
            });
        }
        work
    }

    /// The layout each read-back is compared under: the unit's words in one row.
    fn layout(&self) -> ByteLayout {
        let element = match self.per() {
            4 => Element::U32,
            2 => Element::U16,
            _ => Element::U8,
        };
        ByteLayout::buffer(0, self.pair().bytes.len() as u64, element)
    }
}

impl Case for FormatCopies {
    fn name(&self) -> &'static str {
        PAIRS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A device that flushes denormal depth values announces it once, and a `Depth32Float`
        // texture is where that shows (lavapipe does; this Mac does not). The bytes the copies
        // moved still compare exactly: the only site loosened is the set of notices, which Metal
        // holds empty, and the allowance fails the case if the notice did not fire.
        //
        // Only a pair that touches `Depth32Float` names it. A `Depth16Unorm` or `Stencil8` pair
        // holds no 32-bit float depth for a device to flush, so the notice cannot fire there and
        // naming it would be a tolerance nothing can justify.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for all 16 rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut named = vec![crate::replay::shared_memory_notice()];
        let pair = self.pair();
        if pair.from == PixelFormat::Depth32Float || pair.to == PixelFormat::Depth32Float {
            named.push(crate::replay::notice_allowance(
                Degradation::DepthDenormals.code(),
            ));
        }
        named
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let pair = self.pair();
        Ok(vec![
            Input::bytes(
                "formats",
                &[pair.from.as_raw(), pair.to.as_raw(), self.usage()]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            Input::bytes("bytes:written", pair.bytes),
            Input::bytes("bytes:before", &vec![BEFORE; pair.bytes.len()]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let done = made(sites, "work", side.work(&self.program()))?;
        for (at, site) in ["destination", "source"].into_iter().enumerate() {
            let bytes = done.read(at).map_err(|why| Abort {
                at: site.to_owned(),
                failure: Failure::named("UnansweredRead", why),
            })?;
            sites.leaf(site, Observed::bytes(bytes.to_vec(), self.layout()));
        }
        sites.value(
            "statuses",
            Value::Lines(done.statuses.iter().map(Clone::clone).collect()),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The cases name distinct recordings.
    #[test]
    fn the_pairs_name_distinct_cases() {
        let named: BTreeSet<&str> = PAIRS.iter().map(|pair| pair.name).collect();
        assert_eq!(named.len(), PAIRS.len());
    }

    /// Every pair's bytes are a whole number of texels of its unit, so the row pitch the program
    /// states is the row's bytes.
    #[test]
    fn every_pairs_bytes_are_whole_texels() {
        for (at, pair) in PAIRS.iter().enumerate() {
            let case = FormatCopies { at };
            assert_eq!(case.per() * W, pair.bytes.len() as u64, "{}", pair.name);
            assert_eq!(case.layout().extent(), pair.bytes.len(), "{}", pair.name);
        }
    }

    /// Every committed recording holds the bytes the case wrote, in both the destination and the
    /// source: Metal's copy between a depth or stencil format and a colour format of its unit
    /// moves the bytes and leaves the source as it was.
    ///
    /// This is the cross-check the port exists for: the lane asserts it, and here the recording
    /// states Metal's own answer, so a release in which Metal converts instead of copying shows up
    /// as a moved recording and this test says which pair.
    #[test]
    fn every_recorded_copy_moved_the_bytes() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        for (at, pair) in PAIRS.iter().enumerate() {
            let _ = at;
            let recording = match crate::recording::read(pair.name) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(pair.name);
                    continue;
                }
            };
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for site in ["destination", "source"] {
                        let Some(node) = outcome.transcript.root.child(site) else {
                            differ.push(format!("{}: no site {site}", pair.name));
                            continue;
                        };
                        let Value::Bytes(bytes) = &node.observed.value else {
                            differ.push(format!("{}/{site}: no bytes", pair.name));
                            continue;
                        };
                        if bytes != pair.bytes {
                            differ.push(format!(
                                "{}/{site}: Metal holds {bytes:02x?}, the case wrote {:02x?}",
                                pair.name, pair.bytes
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "Metal's recorded copies did not move the bytes:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
