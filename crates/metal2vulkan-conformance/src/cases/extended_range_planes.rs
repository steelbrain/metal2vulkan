//! A two-plane extended-range texture joined into one and split back again.
//!
//! A consumer holds Apple's `RGB10A8_2P_XR10` as two single-plane textures: plane 0 viewed in
//! `BGR10_XR` (or its sRGB twin) and the alpha plane in `ARGB8Unorm`, whose first component is the
//! texel's second byte. The owned fixture `xr_planes` joins the two into a `BGRA10_XR` texture (or
//! its sRGB twin) with `xr_planes_join`, and splits one back into both planes with
//! `xr_planes_split`, one thread per texel, reading and writing `half`.
//!
//! One case per pairing of plane 0's format with the joined texture's — four of them — each
//! holding the bytes of all three output textures. Every input code of every channel occurs once,
//! with the bits Metal does not read set, so a channel the conversion drops or keeps shows up as a
//! texel rather than as a hash.
//!
//! Why this family: it is the one lane of the batch whose committed expectation is Metal's own
//! bytes whole, and it is the first case to read a texture at all. The transcript's byte layout has
//! carried a texture region — level, slice, aspect, sample, origin, size and the two pitches —
//! since the core landed, and nothing exercised it. Each output site here names its texture region,
//! so a diff reads `element [row 11, 27]` of a `BGRA10_XR` level 0 rather than a byte offset.
//!
//! The `BGR10_XR` family is also the first thing a venue can honestly lack: lavapipe has no such
//! format, so the texture call returns the typed refusal and the case's verdict is the venue's gap
//! rather than a pass or a failure.
//!
//! Nothing here can fault a GPU: one compute pass a kernel, three textures each, one thread a
//! texel inside the texture, no loop and no sampling.

use crate::case::{call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::PixelFormat;

/// The fixture both kernels live in.
const FIXTURE: &str = "xr_planes";

/// The kernel that joins the two planes into one texture.
const JOIN: &str = "xr_planes_join";

/// The kernel that splits one texture back into the two planes.
const SPLIT: &str = "xr_planes_split";

/// Each texture's side: 32 by 32 texels, one per code.
const SIDE: u32 = 32;

/// The texels of a texture.
const COUNT: usize = (SIDE * SIDE) as usize;

/// The alpha plane's format: four bytes a texel, whose first component is the texel's second byte.
const ALPHA: PixelFormat = PixelFormat::ARGB8Unorm;

/// The threads in each dimension of a threadgroup.
const GROUP: u32 = 8;

/// The byte every texel of a texture a kernel writes holds beforehand, so a texel the kernel did
/// not write is a recorded value rather than whatever was there.
const BLANK: u8 = 0xab;

/// The bytes of one texel of `format`.
fn texel_bytes(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::BGRA10XR | PixelFormat::BGRA10XRSRGB => 8,
        _ => 4,
    }
}

/// Texel `n`'s codes: red, green, blue, alpha.
fn codes(n: usize) -> [u32; 4] {
    let n = n as u32;
    [(n * 7 + 3) % 1024, 1023 - n, n, (n * 13 + 5) % 1024]
}

/// Plane 0's bytes: `BGR10_XR` words, blue in bits 0 to 9, green 10 to 19, red 20 to 29, and bits
/// 30 and 31 set to `n % 4` — bits the format does not read, so a conversion that reads them shows.
fn color_bytes() -> Vec<u8> {
    (0..COUNT)
        .flat_map(|n| {
            let [red, green, blue, _] = codes(n);
            ((n as u32 % 4) << 30 | red << 20 | green << 10 | blue).to_le_bytes()
        })
        .collect()
}

/// The alpha plane's bytes: four bytes a texel, the second (its first component) taking every
/// value, the other three set to something else so a conversion that reads the wrong one shows.
fn alpha_bytes() -> Vec<u8> {
    (0..COUNT)
        .flat_map(|n| {
            [
                (n * 11 + 7) as u8,
                (n * 37 + 11) as u8,
                (n * 13) as u8,
                (n * 17 + 3) as u8,
            ]
        })
        .collect()
}

/// The joined texture's bytes: `BGRA10_XR`'s four 16-bit fields (blue, green, red, alpha), each
/// code in bits 6 to 15 and the low six bits set to something the format does not read.
fn joined_bytes() -> Vec<u8> {
    (0..COUNT)
        .flat_map(|n| {
            let [red, green, blue, alpha] = codes(n);
            let junk = n as u32 % 64;
            [
                blue << 6 | junk,
                green << 6 | (63 - junk),
                red << 6 | junk,
                alpha << 6 | 42,
            ]
            .into_iter()
            .flat_map(|field| (field as u16).to_le_bytes())
        })
        .collect()
}

/// A texture of `format` holding [`BLANK`] in every byte, for a kernel to write.
fn blank_bytes(format: PixelFormat) -> Vec<u8> {
    vec![BLANK; COUNT * texel_bytes(format) as usize]
}

/// The layout a texture of `format` is read under: one row of the layout per row of texels, so a
/// diff names the texel by its position, and the whole level with no padding between the rows.
fn layout(format: PixelFormat) -> ByteLayout {
    let pitch = SIDE * texel_bytes(format);
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        // A texel of either family is read as its fields: the 10:10:10:2 plane and the alpha view
        // as one 32-bit word, the joined texture as its four 16-bit fields, so a diff names the
        // field that moved rather than the whole texel.
        element: match texel_bytes(format) {
            8 => Element::U16,
            _ => Element::U32,
        },
        row_offset: 0,
        row_elements: match texel_bytes(format) {
            8 => SIDE * 4,
            _ => SIDE,
        },
        row_pitch: pitch,
        rows: SIDE,
        image_pitch: pitch * SIDE,
        images: 1,
        // The rows are packed, so there is no padding; the sites cover every byte.
        gaps: Gaps::Ignored,
    }
}

/// A case: one pairing of plane 0's format with the joined texture's.
pub struct ExtendedRangePlanes {
    name: &'static str,
    /// Plane 0's format, and the format the split kernel writes plane 0 back in.
    color: PixelFormat,
    /// The joined texture's format.
    joined: PixelFormat,
}

/// Every pairing of the two formats, in a fixed order.
pub fn all() -> Vec<ExtendedRangePlanes> {
    vec![
        ExtendedRangePlanes {
            name: "xr-planes-bgr10xr-bgra10xr",
            color: PixelFormat::BGR10XR,
            joined: PixelFormat::BGRA10XR,
        },
        ExtendedRangePlanes {
            name: "xr-planes-bgr10xr-bgra10xr-srgb",
            color: PixelFormat::BGR10XR,
            joined: PixelFormat::BGRA10XRSRGB,
        },
        ExtendedRangePlanes {
            name: "xr-planes-bgr10xr-srgb-bgra10xr",
            color: PixelFormat::BGR10XRSRGB,
            joined: PixelFormat::BGRA10XR,
        },
        ExtendedRangePlanes {
            name: "xr-planes-bgr10xr-srgb-bgra10xr-srgb",
            color: PixelFormat::BGR10XRSRGB,
            joined: PixelFormat::BGRA10XRSRGB,
        },
    ]
}

impl ExtendedRangePlanes {
    /// The fixture under the kernel `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }

    /// A texture of `format` holding `bytes`, at the format's own row pitch.
    fn texture(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        format: PixelFormat,
        bytes: &[u8],
    ) -> Result<Handle, Abort> {
        made(
            sites,
            at,
            side.texture_2d(format, [SIDE, SIDE], SIDE * texel_bytes(format), bytes),
        )
    }

    /// The bytes of the texture `texture` holds, under `format`'s row pitch, as a site named `at`.
    fn read(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        texture: Handle,
        format: PixelFormat,
    ) -> Result<Vec<u8>, Abort> {
        call(
            sites,
            at,
            side.read_texture(texture, [SIDE, SIDE], SIDE * texel_bytes(format)),
            |_| Value::Absent,
        )
    }

    /// Run one kernel over three textures and wait for it.
    fn dispatch(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        pipeline: Handle,
        textures: [Handle; 3],
    ) -> Result<(), Abort> {
        let command_buffer = made(
            sites,
            &format!("{at}/dispatch"),
            side.dispatch_threads(
                pipeline,
                &Bindings::textures(&textures),
                &[],
                [SIDE, SIDE, 1],
                [GROUP, GROUP, 1],
            ),
        )?;
        call(
            sites,
            &format!("{at}/status"),
            side.wait(command_buffer),
            Clone::clone,
        )?;
        Ok(())
    }
}

impl Case for ExtendedRangePlanes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in [JOIN, SPLIT] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Self::fixture(entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture(JOIN);
        let formats: Vec<u8> = [self.color, self.joined, ALPHA]
            .iter()
            .flat_map(|format| format.as_raw().to_le_bytes())
            .collect();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The three formats the case pairs: plane 0's, the joined texture's and the alpha
            // view's. A pairing that moves is a different question, so it is part of the inputs.
            Input::bytes("formats", &formats),
            Input::bytes("texture:plane0", &color_bytes()),
            Input::bytes("texture:alpha", &alpha_bytes()),
            Input::bytes("texture:joined", &joined_bytes()),
            // What a texture a kernel writes holds before it runs, in each of the three formats
            // the case writes.
            Input::bytes("blank:joined", &blank_bytes(self.joined)),
            Input::bytes("blank:plane0", &blank_bytes(self.color)),
            Input::bytes("blank:alpha", &blank_bytes(ALPHA)),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = Self::fixture(JOIN)
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let join = made(sites, "pipeline/join", side.compute_pipeline(library, JOIN))?;
        let split = made(
            sites,
            "pipeline/split",
            side.compute_pipeline(library, SPLIT),
        )?;

        // Join: plane 0 and the alpha view into one texture.
        let plane = self.texture(side, sites, "texture/plane0", self.color, &color_bytes())?;
        let alpha = self.texture(side, sites, "texture/alpha", ALPHA, &alpha_bytes())?;
        let target = self.texture(
            side,
            sites,
            "texture/joined-out",
            self.joined,
            &blank_bytes(self.joined),
        )?;
        self.dispatch(side, sites, "join", join, [plane, alpha, target])?;
        let joined = self.read(side, sites, "read-back/joined", target, self.joined)?;

        // Split: one joined texture back into both planes.
        let source = self.texture(side, sites, "texture/joined", self.joined, &joined_bytes())?;
        let plane_out = self.texture(
            side,
            sites,
            "texture/plane0-out",
            self.color,
            &blank_bytes(self.color),
        )?;
        let alpha_out =
            self.texture(side, sites, "texture/alpha-out", ALPHA, &blank_bytes(ALPHA))?;
        self.dispatch(side, sites, "split", split, [source, plane_out, alpha_out])?;
        let color = self.read(side, sites, "read-back/color", plane_out, self.color)?;
        let split_alpha = self.read(side, sites, "read-back/alpha", alpha_out, ALPHA)?;

        sites.push(
            Node::branch("output", Vec::new())
                .with(Node::leaf(
                    "joined",
                    Observed::bytes(joined, layout(self.joined)),
                ))
                .with(Node::leaf(
                    "color",
                    Observed::bytes(color, layout(self.color)),
                ))
                .with(Node::leaf(
                    "alpha",
                    Observed::bytes(split_alpha, layout(ALPHA)),
                )),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The generators write one texel per code, of the length the format wants, and each layout
    /// covers exactly those bytes: a layout that reached further would read past the texture.
    #[test]
    fn the_generators_and_the_layouts_agree_on_every_format() {
        for format in [
            PixelFormat::BGR10XR,
            PixelFormat::BGR10XRSRGB,
            PixelFormat::BGRA10XR,
            PixelFormat::BGRA10XRSRGB,
            ALPHA,
        ] {
            let bytes = COUNT * texel_bytes(format) as usize;
            assert_eq!(blank_bytes(format).len(), bytes, "{format:?}");
            assert_eq!(layout(format).extent(), bytes, "{format:?}");
        }
        assert_eq!(color_bytes().len(), COUNT * 4);
        assert_eq!(alpha_bytes().len(), COUNT * 4);
        assert_eq!(joined_bytes().len(), COUNT * 8);
    }

    /// Every texel of the two 10-bit inputs holds a different word, so each is its own question
    /// and a conversion that mixed two texels up could not pass.
    #[test]
    fn every_texel_of_the_inputs_is_its_own_code() {
        let words = |bytes: Vec<u8>, size: usize| -> usize {
            bytes
                .chunks(size)
                .map(<[u8]>::to_vec)
                .collect::<BTreeSet<Vec<u8>>>()
                .len()
        };
        assert_eq!(words(color_bytes(), 4), COUNT);
        assert_eq!(words(joined_bytes(), 8), COUNT);
    }

    /// The four cases name distinct pairings, so no two share a recording.
    #[test]
    fn the_cases_pair_the_formats_distinctly() {
        let pairs: BTreeSet<(u32, u32)> = all()
            .iter()
            .map(|case| (case.color.as_raw(), case.joined.as_raw()))
            .collect();
        assert_eq!(pairs.len(), 4);
        let named: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        assert_eq!(named.len(), 4);
    }
}
