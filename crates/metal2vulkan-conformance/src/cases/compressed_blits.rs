//! Blits through a losslessly compressed texture, ported from `runtime_lane::compressed_blits`.
//!
//! Metal keeps a texture losslessly compressed where `allowGPUOptimizedContents` holds, the shorter
//! side of its level 0 is at least 16 texels, and its usage has neither `ShaderWrite` nor
//! `PixelFormatView`. A GPU blit that reads or writes such a texture moves its texels **through**
//! that compression, so for three families the bytes the blit writes are not the bytes it read:
//! the same value in its canonical encoding. The runtime's answer is the device's copy, which keeps
//! the bytes, announced by the note of the encoding it keeps (`compressed_blit.rs`): where the
//! device's own copy of the compressed image is not Metal's blit, those sites are `known-gaps.tsv`
//! rows that name the note. `BGRA10_XR` and its sRGB twin have an exact form: under the suite's
//! `Policy::Exact` (`ExtendedRangeBlitPolicy::Canonical`) the runtime writes Metal's encoding, so
//! their sites hold Metal's bytes and announce nothing; their note's declaration covers a
//! `--default-policy` run, where the device's copy keeps the bytes.
//!
//! Record mode runs the whole program on this M3 Max, because every call it makes is public Metal:
//! a texture of any descriptor, a texture over a buffer, the blit copies, `replaceRegion:` and
//! `getBytes:`. So each recording holds the bytes Metal's own blits wrote.
//!
//! One case per format, so a venue that holds no texture of one answers its typed refusal and
//! counts as lacking the feature rather than taking the other three down with it.
//!
//! Each case is one program of the lane's rows, over the lane's own splitmix64 bytes:
//!
//! - a buffer blitted into a private texture and back out, three ways: compressed, without
//!   `allowGPUOptimizedContents`, and 8 texels high (the last two keep the bytes);
//! - level 3 of a mipmapped compressed texture, blitted out at an offset of 4 with rows 12 bytes
//!   past the texels', so the bytes outside the texels are compared as the layout's gaps;
//! - a shared compressed texture written and read by the CPU (which keeps the bytes), then blitted
//!   out (which does not), and the other way round;
//! - the shared texture copied whole into a plain texture and back, texture to texture;
//! - the shared texture copied into a texture over a buffer whose rows are 256 bytes past the
//!   texels', the bytes between the rows compared.
//!
//! **Not ported, and why:** the lane's heap row asks a device for
//! `heapTextureSizeAndAlignWithDescriptor:`'s answer and compares a shared texture's with a private
//! one's. That is a CPU-side *query*, not a copy, and the transcript has no record kind for it —
//! the copy-and-fill vocabulary makes resources and reads bytes. It wants a query record kind of
//! its own; until there is one, the heap row stays a lane.
//!
//! Nothing here can fault a GPU: blit copies and host copies inside the textures the program makes,
//! no dispatch, no draw and no loop.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Observed, Region, Sites, Value};
use crate::work::{Rect, Resource, Step, TextureSpec, Work};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The textures' width, and the height of all but the short one: the lane's, and at least the 16
/// texels Metal needs on the shorter side to keep a texture compressed.
const SIDE: u64 = 64;

/// The short texture's height: below 16, so Metal keeps it uncompressed and its blits keep the
/// bytes.
const SHORT: u64 = 8;

/// The byte every output buffer and every gap holds before a copy, so a byte no copy wrote is a
/// recorded value rather than whatever the allocator left.
const BLANK: u8 = 0xee;

/// The bytes a copy out at an offset starts past the buffer's first byte.
const OFFSET: u64 = 4;

/// The bytes the level-3 copy's rows run past its texels'.
const PAST: u64 = 12;

/// The bytes the over-a-buffer texture's rows run past its texels'.
const OVER_PAST: u64 = 256;

/// One format whose blits through the compression change bytes: the case's name, the raw
/// `MTLPixelFormat`, and the note a blit through it announces where the device's copy keeps the
/// bytes.
struct Format {
    name: &'static str,
    format: PixelFormat,
    note: Degradation,
}

/// Every format whose blits through the compression change bytes.
const FORMATS: [Format; 4] = [
    Format {
        name: "compressed-blits-bgra10xr",
        format: PixelFormat::BGRA10XR,
        note: Degradation::ExtendedRangeBitsKept,
    },
    Format {
        name: "compressed-blits-bgra10xr-srgb",
        format: PixelFormat::BGRA10XRSRGB,
        note: Degradation::ExtendedRangeBitsKept,
    },
    Format {
        name: "compressed-blits-rg11b10f",
        format: PixelFormat::RG11B10Float,
        note: Degradation::DeviceNanEncoding,
    },
    Format {
        name: "compressed-blits-rgb9e5f",
        format: PixelFormat::RGB9E5Float,
        note: Degradation::SharedExponentKept,
    },
];

/// The declaration of the note a blit through a texture of the raw `MTLPixelFormat` `format` that
/// Metal keeps losslessly compressed announces (the runtime's `compressed_blit.rs`: the device's
/// copy keeps the bytes Metal's compression rewrites), where that note is not one of the device's
/// native notes the suite declares for every case (`crate::replay::NATIVE_NOTES`):
/// `extended-range-bits-kept` for `BGRA10_XR` and its sRGB twin, `shared-exponent-kept` for
/// `RGB9E5Float`. Every case whose program blits through such a texture declares it, at the
/// notices site alone: each value still compares against Metal's. Under `Policy::Exact` the
/// `BGRA10_XR` formats' note does not fire (`ExtendedRangeBlitPolicy::Canonical` writes Metal's
/// encoding), and the report counts its declaration as unfired.
pub(crate) fn kept_encoding_notice(format: u32) -> Option<Allowance> {
    FORMATS
        .iter()
        .find(|f| f.format.as_raw() == format)
        .map(|f| f.note)
        .filter(|note| !crate::replay::NATIVE_NOTES.contains(note))
        .map(|note| crate::replay::notice_allowance(note.code()))
}

/// A splitmix64 stream of bytes: the lane's own, seeded as the lane seeds it, so a case feeds the
/// bytes the lane measured.
struct Rng(u64);

impl Rng {
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u8
            })
            .collect()
    }
}

/// One read-back the program answers: its site's name, and how its bytes are laid out.
struct Read {
    at: &'static str,
    layout: ByteLayout,
}

/// The case for one format.
pub struct CompressedBlits {
    at: usize,
}

/// Every format's case.
pub fn all() -> Vec<CompressedBlits> {
    (0..FORMATS.len())
        .map(|at| CompressedBlits { at })
        .collect()
}

/// A buffer's bytes, as a read of one row of `element`s with nothing between the rows.
fn flat(length: u64) -> ByteLayout {
    ByteLayout::buffer(0, length, Element::U32)
}

/// A texture level's bytes read row by row, with the bytes past each row's texels compared where
/// `gaps` says: the layout a padded copy out needs, so "the bytes between the rows stayed as they
/// were" is a fact the comparison checks.
fn rows(
    level: u32,
    size: [u32; 2],
    row_offset: u32,
    row_pitch: u32,
    texel_bytes: u32,
    gaps: Gaps,
) -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
        },
        element: Element::U32,
        row_offset,
        row_elements: size[0] * texel_bytes / 4,
        row_pitch,
        rows: size[1],
        image_pitch: row_offset + row_pitch * size[1],
        images: 1,
        gaps,
    }
}

impl CompressedBlits {
    /// The format the case blits.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The bytes one texel takes: `BGRA10_XR`'s four 16-bit fields, the packed floats' word.
    fn texel(&self) -> u64 {
        match self.format().format {
            PixelFormat::BGRA10XR | PixelFormat::BGRA10XRSRGB => 8,
            _ => 4,
        }
    }

    /// Every byte the case feeds, in the order the program uses them: one stream, as the lane
    /// draws one stream per format.
    fn fed(&self) -> Fed {
        let per = self.texel();
        let whole = (SIDE * SIDE * per) as usize;
        let mut rng = Rng(0xfc5_b117 + self.at as u64);
        Fed {
            tall: rng.bytes(whole),
            plain: rng.bytes(whole),
            short: rng.bytes((SIDE * SHORT * per) as usize),
            small: rng.bytes(((SIDE >> 3) * (SIDE >> 3) * per) as usize),
            cpu_in: rng.bytes(whole),
            blit_in: rng.bytes(whole),
            out_of_compressed: rng.bytes(whole),
            into_compressed: rng.bytes(whole),
            over_a_buffer: rng.bytes(whole),
        }
    }

    /// The program the case runs, and what each of its reads is.
    fn program(&self) -> (Work, Vec<Read>) {
        let per = self.texel();
        let fed = self.fed();
        let mut work = Work::new();
        let mut reads = Vec::new();
        let texel_bytes = per as u32;
        let compressed = TextureSpec::texture_2d(self.format().format.as_raw(), SIDE, SIDE);

        // A buffer into a private texture and back out: compressed, plain, and 8 texels high.
        for (at, spec, height, bytes) in [
            ("tall", compressed, SIDE, &fed.tall),
            ("plain", compressed.plain(), SIDE, &fed.plain),
            (
                "short",
                TextureSpec::texture_2d(self.format().format.as_raw(), SIDE, SHORT),
                SHORT,
                &fed.short,
            ),
        ] {
            let source = work.buffer(bytes.clone());
            let texture = work.add(Resource::Texture { spec });
            let out = work.buffer(vec![BLANK; (SIDE * height * per) as usize]);
            work.step(Step::CopyBufferToTexture {
                buffer: source,
                offset: 0,
                bytes_per_row: SIDE * per,
                bytes_per_image: 0,
                size: [SIDE, height, 1],
                texture,
                slice: 0,
                level: 0,
                origin: [0, 0, 0],
                plane: crate::work::Plane::Whole,
            });
            work.step(Step::CopyTextureToBuffer {
                texture,
                slice: 0,
                level: 0,
                origin: [0, 0, 0],
                size: [SIDE, height, 1],
                buffer: out,
                offset: 0,
                bytes_per_row: SIDE * per,
                bytes_per_image: 0,
                plane: crate::work::Plane::Whole,
            });
            work.step(Step::Commit);
            work.step(Step::ReadBuffer {
                buffer: out,
                offset: 0,
                length: (SIDE * height * per) as usize,
            });
            reads.push(Read {
                at,
                layout: flat(SIDE * height * per),
            });
        }

        // Level 3 of a mipmapped compressed texture, out at an offset with rows past the texels'.
        let small = SIDE >> 3;
        let level_3 = work.add(Resource::Texture {
            spec: compressed.levels(4),
        });
        let source = work.buffer(fed.small.clone());
        let row = small * per + PAST;
        let out = work.buffer(vec![BLANK; (OFFSET + row * small) as usize]);
        work.step(Step::CopyBufferToTexture {
            buffer: source,
            offset: 0,
            bytes_per_row: small * per,
            bytes_per_image: 0,
            size: [small, small, 1],
            texture: level_3,
            slice: 0,
            level: 3,
            origin: [0, 0, 0],
            plane: crate::work::Plane::Whole,
        });
        work.step(Step::CopyTextureToBuffer {
            texture: level_3,
            slice: 0,
            level: 3,
            origin: [0, 0, 0],
            size: [small, small, 1],
            buffer: out,
            offset: OFFSET,
            bytes_per_row: row,
            bytes_per_image: 0,
            plane: crate::work::Plane::Whole,
        });
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: out,
            offset: 0,
            length: (OFFSET + row * small) as usize,
        });
        reads.push(Read {
            at: "level-3-padded",
            layout: rows(
                3,
                [small as u32, small as u32],
                OFFSET as u32,
                row as u32,
                texel_bytes,
                Gaps::Compared,
            ),
        });

        // A shared compressed texture: the CPU's copies keep the bytes, a blit out does not.
        let whole = (SIDE * SIDE * per) as usize;
        let shared = work.add(Resource::Texture {
            spec: compressed.shared(),
        });
        let region = Rect::new(0, 0, SIDE, SIDE);
        work.step(Step::ReplaceRegion {
            texture: shared,
            level: 0,
            slice: 0,
            region,
            // Every format here is uncompressed on the host side, so the rows are texel rows.
            rows: region.size[1],
            bytes: fed.cpu_in.clone(),
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
        });
        work.step(Step::GetBytes {
            texture: shared,
            level: 0,
            slice: 0,
            region,
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
            length: whole,
            // The read starts from zero, as this case's reads always did: it asks
            // nothing about the padding between its rows.
            before: 0,
        });
        reads.push(Read {
            at: "cpu-in-cpu-out",
            layout: rows(
                0,
                [SIDE as u32, SIDE as u32],
                0,
                (SIDE * per) as u32,
                texel_bytes,
                Gaps::Ignored,
            ),
        });
        let out = work.buffer(vec![BLANK; whole]);
        work.step(Step::CopyTextureToBuffer {
            texture: shared,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
            buffer: out,
            offset: 0,
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
            plane: crate::work::Plane::Whole,
        });
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: out,
            offset: 0,
            length: whole,
        });
        reads.push(Read {
            at: "cpu-in-blit-out",
            layout: flat(SIDE * SIDE * per),
        });

        // A blit in and the CPU out, over the same texture.
        let source = work.buffer(fed.blit_in.clone());
        work.step(Step::CopyBufferToTexture {
            buffer: source,
            offset: 0,
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
            size: [SIDE, SIDE, 1],
            texture: shared,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            plane: crate::work::Plane::Whole,
        });
        work.step(Step::Commit);
        work.step(Step::GetBytes {
            texture: shared,
            level: 0,
            slice: 0,
            region,
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
            length: whole,
            // The read starts from zero, as this case's reads always did: it asks
            // nothing about the padding between its rows.
            before: 0,
        });
        reads.push(Read {
            at: "blit-in-cpu-out",
            layout: rows(
                0,
                [SIDE as u32, SIDE as u32],
                0,
                (SIDE * per) as u32,
                texel_bytes,
                Gaps::Ignored,
            ),
        });

        // Texture to texture, out of a compressed texture's CPU bytes into a plain one, and from a
        // plain one's into the compressed one.
        let plain = work.add(Resource::Texture {
            spec: compressed.plain().shared(),
        });
        work.step(Step::ReplaceRegion {
            texture: shared,
            level: 0,
            slice: 0,
            region,
            // Every format here is uncompressed on the host side, so the rows are texel rows.
            rows: region.size[1],
            bytes: fed.out_of_compressed.clone(),
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
        });
        let whole_copy = |from: usize, to: usize| Step::CopyTextureToTexture {
            from,
            from_slice: 0,
            from_level: 0,
            from_origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
            to,
            to_slice: 0,
            to_level: 0,
            to_origin: [0, 0, 0],
        };
        work.step(whole_copy(shared, plain));
        work.step(Step::Commit);
        work.step(Step::GetBytes {
            texture: plain,
            level: 0,
            slice: 0,
            region,
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
            length: whole,
            // The read starts from zero, as this case's reads always did: it asks
            // nothing about the padding between its rows.
            before: 0,
        });
        reads.push(Read {
            at: "out-of-compressed",
            layout: rows(
                0,
                [SIDE as u32, SIDE as u32],
                0,
                (SIDE * per) as u32,
                texel_bytes,
                Gaps::Ignored,
            ),
        });
        work.step(Step::ReplaceRegion {
            texture: plain,
            level: 0,
            slice: 0,
            region,
            // Every format here is uncompressed on the host side, so the rows are texel rows.
            rows: region.size[1],
            bytes: fed.into_compressed.clone(),
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
        });
        work.step(whole_copy(plain, shared));
        work.step(Step::Commit);
        work.step(Step::GetBytes {
            texture: shared,
            level: 0,
            slice: 0,
            region,
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
            length: whole,
            // The read starts from zero, as this case's reads always did: it asks
            // nothing about the padding between its rows.
            before: 0,
        });
        reads.push(Read {
            at: "into-compressed",
            layout: rows(
                0,
                [SIDE as u32, SIDE as u32],
                0,
                (SIDE * per) as u32,
                texel_bytes,
                Gaps::Ignored,
            ),
        });

        // Into a texture over a buffer, its rows past the texels': the rule at its rows, and the
        // bytes between them compared as the layout's gaps.
        let over_row = SIDE * per + OVER_PAST;
        let backing = work.buffer(vec![BLANK; (over_row * SIDE) as usize]);
        let over = work.add(Resource::TextureOverBuffer {
            buffer: backing,
            spec: compressed.shared(),
            offset: 0,
            bytes_per_row: over_row,
        });
        work.step(Step::ReplaceRegion {
            texture: shared,
            level: 0,
            slice: 0,
            region,
            // Every format here is uncompressed on the host side, so the rows are texel rows.
            rows: region.size[1],
            bytes: fed.over_a_buffer.clone(),
            bytes_per_row: SIDE * per,
            bytes_per_image: 0,
        });
        work.step(whole_copy(shared, over));
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: backing,
            offset: 0,
            length: (over_row * SIDE) as usize,
        });
        reads.push(Read {
            at: "over-a-buffer",
            layout: rows(
                0,
                [SIDE as u32, SIDE as u32],
                0,
                over_row as u32,
                texel_bytes,
                Gaps::Compared,
            ),
        });
        (work, reads)
    }
}

/// Every byte the case feeds, one field a row of the program.
struct Fed {
    tall: Vec<u8>,
    plain: Vec<u8>,
    short: Vec<u8>,
    small: Vec<u8>,
    cpu_in: Vec<u8>,
    blit_in: Vec<u8>,
    out_of_compressed: Vec<u8>,
    into_compressed: Vec<u8>,
    over_a_buffer: Vec<u8>,
}

impl Fed {
    /// Each field in the program's order, by the name its input takes.
    fn named(&self) -> [(&'static str, &Vec<u8>); 9] {
        [
            ("tall", &self.tall),
            ("plain", &self.plain),
            ("short", &self.short),
            ("small", &self.small),
            ("cpu-in", &self.cpu_in),
            ("blit-in", &self.blit_in),
            ("out-of-compressed", &self.out_of_compressed),
            ("into-compressed", &self.into_compressed),
            ("over-a-buffer", &self.over_a_buffer),
        ]
    }
}

impl Case for CompressedBlits {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Our side's padded copies go through a tightly packed scratch buffer, a performance
        // notice the runtime announces once per device, and one that leaves the bytes exactly
        // Metal's: the only site loosened is the set of notices, which Metal holds empty, and the
        // allowance fails the case if the notice did not fire.
        //
        // `shadowed-texture`, which every row of this family announces: its texture over a buffer
        // reaches the buffer's bytes through a shadow image rather than an image aliasing them.
        //
        // This comment used to leave the code undeclared and the rows divergent on purpose, on the
        // grounds that naming an unexamined mechanism makes a permanent excuse of it, and it asked
        // whoever took it to decide whether a shadow image is the right answer for these formats
        // and then declare or change the runtime. The decision, and what it rests on:
        //
        // It is the right answer, because there is no other exact one. A texture over a buffer
        // aliases the buffer's bytes with a linear image where the device can make one
        // (`texture.rs`, `alias_image`); where it cannot, the only remaining way to reach those
        // bytes exactly is to copy them into an image and back, which is this notice. These four
        // are the formats the device makes no linear image of -- `BGRA10XR`, its sRGB twin,
        // `RG11B10Float` and `RGB9E5Float`, whose linear tiling the device does not list
        // (`linear_lack`) -- and Metal makes a texture over a buffer in all four, so refusing them
        // is not open to us either. A cheaper exact form would have to come from the device
        // gaining linear tiling for the format, not from a choice of ours.
        //
        // It is also the answer the suite has already made for this mechanism, not a new one: five
        // cases across four other families declare `shadowed-texture` for a texture whose bytes no
        // image can alias -- `two-plane-row-blits`, `texture-fills-heap-run-550-b5` and `-551-b5`,
        // `surface-fills-over-planes-and-buffers`, and `buffer-texture-blits-to-and-from`. Leaving
        // these four undeclared made them divergent at `case/notices` on every run while the same
        // code was justified elsewhere, which is noise rather than an open question.
        //
        // It loosens no value. Every byte of all four rows compares exactly with Metal's; the whole
        // difference the declaration covers is the announcement, which Metal holds empty because
        // the copy is not a thing Metal has to do. Measured announced by all four rows on Apple
        // M3 Max in the whole-set run this declaration was taken from; the earlier reading recorded
        // the same code on lavapipe.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 4 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        //
        // The note of the encoding the device's copy keeps ([`kept_encoding_notice`]).
        let mut allowances = vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::StagedTextureCopy.code()),
            crate::replay::notice_allowance(Degradation::ShadowedTexture.code()),
        ];
        allowances.extend(kept_encoding_notice(self.format().format.as_raw()));
        allowances
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fed = self.fed();
        let mut inputs = vec![Input::bytes(
            "format",
            &self.format().format.as_raw().to_le_bytes(),
        )];
        for (name, bytes) in fed.named() {
            inputs.push(Input::bytes(format!("bytes:{name}"), bytes));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let (work, reads) = self.program();
        let done = made(sites, "work", side.work(&work))?;
        for (at, read) in (0..).zip(&reads) {
            let bytes = done.read(at).map_err(|why| Abort {
                at: read.at.to_owned(),
                failure: Failure::named("UnansweredRead", why),
            })?;
            sites.leaf(
                read.at,
                Observed::bytes(bytes.to_vec(), read.layout.clone()),
            );
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

    /// The cases name distinct recordings, since a case's name is its recording's file name.
    #[test]
    fn the_formats_name_distinct_cases() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
    }

    /// Every case of the family declares exactly the notice codes it announces: the same three
    /// for every format, because every row reaches its bytes the same way, and the note of the
    /// encoding its blits keep where that note is not a native one the suite declares itself.
    ///
    /// Stated as the whole set rather than as "the codes are present", because a declaration fires
    /// on its own code: a code announced and declared nowhere is a divergence at the notices site,
    /// and a code declared that no row announces fires on no venue and names nothing the run
    /// justifies. The set is the claim, in both directions.
    #[test]
    fn every_case_declares_exactly_the_notices_it_announces() {
        use crate::allowance::Justification;
        for case in all() {
            let mut wanted = vec![
                // The venue's own: a card without unified memory holds the `Shared` buffer each
                // row reads its answer back through in system memory.
                Degradation::SharedInSystemMemory.code().to_owned(),
                Degradation::StagedTextureCopy.code().to_owned(),
                Degradation::ShadowedTexture.code().to_owned(),
            ];
            let note = case.format().note;
            if !crate::replay::NATIVE_NOTES.contains(&note) {
                wanted.push(note.code().to_owned());
            }
            let codes: Vec<String> = case
                .allowances()
                .iter()
                .map(|allowance| {
                    assert_eq!(
                        allowance.path,
                        format!("case/{}/*", crate::replay::NOTICES_SITE),
                        "{}: an allowance outside the notices site",
                        case.name()
                    );
                    match &allowance.justified_by {
                        Justification::Degradation(code) => code.clone(),
                        other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                    }
                })
                .collect();
            assert_eq!(codes, wanted, "{}", case.name());
        }
    }

    /// The program answers one read per site the case emits, so no read is left unnamed and no
    /// site reads bytes nobody asked for.
    #[test]
    fn the_program_answers_one_read_per_site() {
        for (at, format) in FORMATS.iter().enumerate() {
            let (work, reads) = CompressedBlits { at }.program();
            assert_eq!(work.reads(), reads.len(), "{}", format.name);
            for read in &reads {
                assert!(read.layout.extent() > 0, "{}", read.at);
            }
        }
    }
}
