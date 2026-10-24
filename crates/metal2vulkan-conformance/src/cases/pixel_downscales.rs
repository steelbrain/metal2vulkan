//! A linear downscale by eight through a pixel-coordinate sampler, at seven texture sizes.
//!
//! The consumer's ask of 2026-10-09: a texture drawn at an eighth of its size through a runtime
//! sampler of linear filters, no mip filter, clamp to edge and pixel coordinates. One case a
//! texture size (`SIZES`, the sizes the consumer's passes were found at, none of them a power of
//! two), each sampling the downscale's rectangle of texel centres three ways
//! (`coordinates`): `grid`, the centres `8i + 4`; `ulp`, one ulp off each; and `frac`, a random
//! fraction of the eight-texel cell past each.
//!
//! Each set runs twice. `slot` is the texture at texture slot 0 through the pixel state at sampler
//! slot 0 (the owned fixture `xr_samples`' `xr_sample_runtime`); `norm` is the lane's control — an
//! argument buffer's **normalized** state of the same filters and modes, sampled at the coordinate
//! snapped to its `1/256` cell, half up, over the size (the owned fixture `px_samplers`'
//! `px_sample_argument`). Metal answered the two alike at every size, which is the claim the
//! control exists to make.
//!
//! **What the port changes.** `runtime_lane::pixel_downscales` holds each size's three answers as
//! SHA-256 constants taken on an Apple M4 and compares a digest, and its AIR is an inline `&str`
//! in the opaque-pointer dialect, which `xcrun metal-as` does not read, so it could have no Metal
//! leg. Record mode now runs every set at every size on native Metal.
//!
//! **Why these sites are not whole values.** A set is 24,300 `float4`s at the largest size, so a
//! case's six sets would be 2.3 MiB and its recording far past the 512-KiB whole-value rule. Each
//! set therefore records as the rule's second form: the digest of all its results, with a window of
//! `WINDOW` results at each end — the rectangle's first rows and its last, which is where a
//! downscale's edge clamping shows. The digest is the whole claim and the windows name a sample
//! when one moves.
//!
//! **What is not ported.** The lane's fourth configuration at each size draws the rectangle into a
//! 256x256 target through a fragment function and hashes the target. The case vocabulary has no
//! vertex-and-fragment render pipeline and no draw — only the mesh pipelines the mesh cases needed
//! — so those seven digests stay in the lane. They need `Side` calls for a render pipeline of a
//! vertex and a fragment function, a vertex buffer, `drawPrimitives:`, and a target read-back;
//! nothing about them is blocked by a fixture any more, since `px_samplers` could carry the two
//! functions as readily as the kernels it holds.
//!
//! Nothing here can fault a GPU: one texture a case, six compute passes, no loop.

use crate::allowance::Allowance;
use crate::case::{
    made, Abort, ArgumentSlot, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec,
    Side, TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::notice_allowance;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter};
use metal2vulkan_runtime::Degradation;

/// The fixture whose kernel samples a texture slot through the runtime sampler at slot 0.
const SLOT_FIXTURE: &str = "xr_samples";
const SLOT_ENTRY: &str = "xr_sample_runtime";

/// The fixture whose kernel samples through the sampler an argument buffer holds.
const PIXEL_FIXTURE: &str = "px_samplers";
const ARGUMENT_ENTRY: &str = "px_sample_argument";

/// The results a window holds at each end of a set (the module docs).
const WINDOW: usize = 256;

/// The threads in a threadgroup. Every size's rectangle is a multiple of five, and 135 rows of
/// 180 or more columns divide by five, so the dispatch is whole groups at every size.
const GROUP: u32 = 5;

/// The byte every result buffer holds before the pass.
const BLANK: u8 = 0xab;

/// The bytes one result takes: a `float4`.
const STRIDE: u32 = 16;

/// One size of the sweep: the texture's width and height, and the downscaled rectangle's texels,
/// which sample its first `8 gw` by `8 gh` texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    width: u32,
    height: u32,
    rectangle: [u32; 2],
}

impl Size {
    const fn new(width: u32, height: u32, rectangle: [u32; 2]) -> Self {
        Self {
            width,
            height,
            rectangle,
        }
    }

    /// The samples a set takes at this size.
    fn samples(&self) -> usize {
        (self.rectangle[0] * self.rectangle[1]) as usize
    }
}

/// The sizes, in the lane's order: the consumer's 1536x1152 and 1440x1080 passes, 1080 rows under
/// three other widths that are not powers of two, and each of 1440 and 1080 alone.
const SIZES: [Size; 7] = [
    Size::new(1536, 1152, [180, 135]),
    Size::new(1440, 1080, [180, 135]),
    Size::new(1920, 1080, [240, 135]),
    Size::new(1366, 1080, [170, 135]),
    Size::new(1000, 1080, [125, 135]),
    Size::new(1440, 1152, [180, 144]),
    Size::new(1536, 1080, [192, 135]),
];

/// The case names, in [`SIZES`] order. Written out so every case's name is a `&'static str` the
/// suite can hold and `conformance replay <name>` can select.
const NAMES: [&str; 7] = [
    "pixel-downscales-1536x1152",
    "pixel-downscales-1440x1080",
    "pixel-downscales-1920x1080",
    "pixel-downscales-1366x1080",
    "pixel-downscales-1000x1080",
    "pixel-downscales-1440x1152",
    "pixel-downscales-1536x1080",
];

/// The three coordinate sets, in the lane's order.
const SETS: [&str; 3] = ["grid", "ulp", "frac"];

/// One xorshift32 step.
fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// The bytes of a texture of `size`: each the top byte of the next xorshift32 step from
/// `0x9e3779b9`, in rows of texels of four channels.
fn texels(size: Size) -> Vec<u8> {
    let mut state = 0x9e37_79b9;
    (0..size.width * size.height * 4)
        .map(|_| (xorshift(&mut state) >> 24) as u8)
        .collect()
}

/// The probe's coordinate set `set` over the rectangle of `size`, `x` fastest: `grid`, the texel
/// centres `8i + 4`; `ulp`, up where `i + j` (and for `y` where `i`) is even and down otherwise;
/// `frac`, `(r >> 8) · 2^-21` of xorshift32 from `0x2545f491` past each, `x` then `y`.
fn coordinates(set: &str, size: Size) -> Vec<[f32; 2]> {
    let mut state = 0x2545_f491;
    let mut cells = Vec::with_capacity(size.samples());
    for j in 0..size.rectangle[1] {
        for i in 0..size.rectangle[0] {
            let (x, y) = (8.0 * i as f32 + 4.0, 8.0 * j as f32 + 4.0);
            cells.push(match set {
                "grid" => [x, y],
                "ulp" => [
                    match (i + j) % 2 {
                        0 => x.next_up(),
                        _ => x.next_down(),
                    },
                    match i % 2 {
                        0 => y.next_up(),
                        _ => y.next_down(),
                    },
                ],
                _ => {
                    let dx = (xorshift(&mut state) >> 8) as f32 * 2f32.powi(-21);
                    let dy = (xorshift(&mut state) >> 8) as f32 * 2f32.powi(-21);
                    [x + dx, y + dy]
                }
            });
        }
    }
    cells
}

/// Metal's `1/256` cell of the pixel coordinate `x`, half up: what the normalized control samples
/// at, over the size.
fn snapped(x: f32) -> f32 {
    let scaled = x * 256.0;
    let floor = scaled.floor();
    (floor + if scaled - floor >= 0.5 { 1.0 } else { 0.0 }) / 256.0
}

/// The coordinate buffer a source reads: the set as it stands for the pixel sampler, or snapped
/// to its cell and divided by the size for the normalized control.
fn coordinate_bytes(set: &str, size: Size, normalized: bool) -> Vec<u8> {
    coordinates(set, size)
        .iter()
        .flat_map(|&[x, y]| match normalized {
            true => [
                snapped(x) / size.width as f32,
                snapped(y) / size.height as f32,
            ],
            false => [x, y],
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The sampler state both sources use: linear filters, no mip filter, clamp to edge on both axes,
/// argument-buffer support so the control's argument buffer can name it. Pixel coordinates unless
/// `normalized`.
fn sampler_spec(normalized: bool) -> SamplerSpec {
    let mut spec = SamplerSpec::filtered(SamplerMinMagFilter::Linear)
        .addressed(SamplerAddressMode::ClampToEdge)
        .for_argument_buffer();
    spec.normalized_coordinates = normalized;
    spec
}

/// The layout a window is read under: one row a sample of four lanes, so a diff names the sample
/// and the lane rather than a byte offset.
fn window_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: (STRIDE as usize * WINDOW) as u64,
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: STRIDE,
        rows: WINDOW as u32,
        image_pitch: STRIDE * WINDOW as u32,
        images: 1,
        // Each sample's four lanes fill its row, so there is no padding between them.
        gaps: Gaps::Ignored,
    }
}

/// A case: one texture size.
pub struct PixelDownscales {
    name: &'static str,
    size: Size,
}

/// One case a size.
pub fn all() -> Vec<PixelDownscales> {
    NAMES
        .iter()
        .zip(SIZES)
        .map(|(name, size)| PixelDownscales { name, size })
        .collect()
}

/// What one source binds.
struct Source {
    pipeline: Handle,
    texture: Option<Handle>,
    sampler: Option<Handle>,
    argument: Option<Handle>,
    coordinates: Handle,
}

impl PixelDownscales {
    /// Run one source over one coordinate set and emit the set's digest and its two windows.
    ///
    /// The whole values would be 2.3 MiB a case at the largest size, so this is the recording
    /// rule's second form: a digest of every result, with [`WINDOW`] results at each end. The
    /// digest carries the claim; the windows name a sample when one moves.
    fn samples(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        (at, named): (&str, &str),
        source: Source,
    ) -> Result<Node, Abort> {
        let samples = self.size.samples();
        let length = samples * STRIDE as usize;
        let output = made(
            sites,
            &format!("{at}/buffer"),
            side.buffer(&vec![BLANK; length]),
        )?;
        let textures: Vec<Handle> = source.texture.into_iter().collect();
        let mut buffers = vec![source.coordinates, output];
        buffers.extend(source.argument);
        let samplers: Vec<Handle> = source.sampler.into_iter().collect();
        let command_buffer = made(
            sites,
            &format!("{at}/dispatch"),
            side.dispatch_threads(
                source.pipeline,
                &Bindings::new(&textures, &buffers).sampled(&samplers),
                &[],
                [samples as u32, 1, 1],
                [GROUP, 1, 1],
            ),
        )?;
        let status = made(sites, &format!("{at}/status"), side.wait(command_buffer))?;
        if status != Value::unit_variant("Completed") {
            sites.push(Node::leaf(
                format!("{at}/status"),
                Observed::value(status.clone()),
            ));
            return Err(Abort {
                at: format!("{at}/status"),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        let bytes = made(
            sites,
            &format!("{at}/read-back"),
            side.read_buffer(output, 0, length),
        )?;
        if bytes.len() != length {
            return Err(Abort {
                at: format!("{at}/read-back"),
                failure: Failure::named(
                    "ShortReadBack",
                    format!("{} bytes of {length}", bytes.len()),
                ),
            });
        }
        let window = WINDOW * STRIDE as usize;
        Ok(Node::branch(named.to_owned(), Vec::new())
            .with(Node::leaf(
                "digest",
                Observed::absent().digested(Digest::sha256(&bytes)),
            ))
            .with(Node::leaf(
                "head",
                Observed::bytes(bytes[..window].to_vec(), window_layout()),
            ))
            .with(Node::leaf(
                "tail",
                Observed::bytes(bytes[length - window..].to_vec(), window_layout()),
            )))
    }
}

impl Case for PixelDownscales {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (fixture, entry) in [(SLOT_FIXTURE, SLOT_ENTRY), (PIXEL_FIXTURE, ARGUMENT_ENTRY)] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{fixture}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Fixture {
                            name: fixture,
                            entry,
                        }
                        .text(extension)?
                        .as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let slot = Fixture {
            name: SLOT_FIXTURE,
            entry: SLOT_ENTRY,
        };
        let pixel = Fixture {
            name: PIXEL_FIXTURE,
            entry: ARGUMENT_ENTRY,
        };
        let mut inputs = vec![
            Input {
                name: "air:slot".to_owned(),
                digest: Digest::sha256(slot.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed:slot".to_owned(),
                digest: Digest::sha256(slot.text(METAL_TEXT)?.as_bytes()),
            },
            Input {
                name: "air:pixel".to_owned(),
                digest: Digest::sha256(pixel.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed:pixel".to_owned(),
                digest: Digest::sha256(pixel.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes(
                "size",
                &[
                    self.size.width,
                    self.size.height,
                    self.size.rectangle[0],
                    self.size.rectangle[1],
                ]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<u8>>(),
            ),
            Input::bytes("sampler:pixel", &sampler_spec(false).words()),
            Input::bytes("sampler:normalized", &sampler_spec(true).words()),
            Input::bytes("texture:texels", &texels(self.size)),
            Input::bytes("buffer:blank", &[BLANK]),
        ];
        for set in SETS {
            inputs.push(Input::bytes(
                format!("coordinates:{set}"),
                &coordinate_bytes(set, self.size, false),
            ));
            inputs.push(Input::bytes(
                format!("coordinates:{set}:snapped"),
                &coordinate_bytes(set, self.size, true),
            ));
        }
        Ok(inputs)
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Metal announces nothing, so a notice of ours is a difference at the notices site.
        // These are the two kinds a run of this round provoked (see `pixel_samplers`); a venue
        // that announces another reads as divergent there until someone names it with a run.
        //
        // Two more are named from the run that shows them, which is what the paragraph above asks
        // for: on lavapipe the device's texel decode keeps a denormal where Metal flushes it
        // (`device-texel-denormals`) and decodes an sRGB code its own way (`srgb-texel-decode`).
        //
        // **Every one of these is a declaration and loosens nothing.** Each sits at the notices
        // site alone, so a texel that differs still fails the case wherever it differs.
        [
            Degradation::ArgumentPixelSamplerScaling,
            Degradation::FloatSampleFilter,
            Degradation::DeviceTexelDenormals,
            Degradation::SrgbTexelDecode,
            // The buffer each pass writes its sampled texels into is `Shared`, and a device
            // with no unified memory keeps a `Shared` allocation in system memory because that is
            // the only memory both sides can reach. The notice belongs to the venue's memory
            // model rather than to any sampler of these cases, and a device whose driver is Metal
            // announces nothing for it.
            Degradation::SharedInSystemMemory,
        ]
        .iter()
        .map(|kind| notice_allowance(kind.code()))
        .collect()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let slot_source = Fixture {
            name: SLOT_FIXTURE,
            entry: SLOT_ENTRY,
        }
        .loaded()
        .map_err(|why| stop("library/slot", why))?;
        let pixel_source = Fixture {
            name: PIXEL_FIXTURE,
            entry: ARGUMENT_ENTRY,
        }
        .loaded()
        .map_err(|why| stop("library/pixel", why))?;
        let slot_library = made(sites, "library/slot", side.library(&slot_source.source()))?;
        let slot_pipeline = made(
            sites,
            "pipeline/slot",
            side.compute_pipeline(slot_library, SLOT_ENTRY),
        )?;
        let pixel_library = made(sites, "library/pixel", side.library(&pixel_source.source()))?;
        let argument_pipeline = made(
            sites,
            "pipeline/argument",
            side.compute_pipeline(pixel_library, ARGUMENT_ENTRY),
        )?;

        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(
                &TextureSpec::new(
                    PixelFormat::RGBA8Unorm,
                    [self.size.width, self.size.height],
                    4 * self.size.width,
                    TextureRole::Read,
                ),
                &texels(self.size),
            ),
        )?;
        let pixel_state = made(sites, "sampler/pixel", side.sampler(&sampler_spec(false)))?;
        let normalized_state = made(
            sites,
            "sampler/normalized",
            side.sampler(&sampler_spec(true)),
        )?;
        let arguments = made(
            sites,
            "argument-buffer/normalized",
            side.argument_buffer(&[
                ArgumentSlot::Texture(texture),
                ArgumentSlot::Sampler(normalized_state),
            ]),
        )?;

        let mut output = Node::branch("output", Vec::new());
        for set in SETS {
            let pixel_coordinates = made(
                sites,
                &format!("coordinates/{set}"),
                side.buffer(&coordinate_bytes(set, self.size, false)),
            )?;
            let snapped_coordinates = made(
                sites,
                &format!("coordinates/{set}/snapped"),
                side.buffer(&coordinate_bytes(set, self.size, true)),
            )?;
            let mut node = Node::branch(set, Vec::new());
            node = node.with(self.samples(
                side,
                sites,
                (&format!("{set}/slot"), "slot"),
                Source {
                    pipeline: slot_pipeline,
                    texture: Some(texture),
                    sampler: Some(pixel_state),
                    argument: None,
                    coordinates: pixel_coordinates,
                },
            )?);
            node = node.with(self.samples(
                side,
                sites,
                (&format!("{set}/norm"), "norm"),
                Source {
                    pipeline: argument_pipeline,
                    texture: None,
                    sampler: None,
                    argument: Some(arguments),
                    coordinates: snapped_coordinates,
                },
            )?);
            output = output.with(node);
        }
        sites.push(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Node as TranscriptNode;
    use std::collections::BTreeSet;

    /// What an Apple M4 answered for each size's three sets through the pixel sampler, as
    /// `runtime_lane::pixel_downscales`' own probe measured it (macOS 27.0.1, two processes,
    /// identical): the case, the set, and the SHA-256 of its results. Our own measurement over our
    /// own inputs, carried here so the recorded M3 answers can be held against it.
    ///
    /// The lane's fourth configuration at each size — the draw — is not here: the case vocabulary
    /// has no vertex-and-fragment pipeline, so those seven digests stay in the lane.
    const M4: [(&str, &str, &str); 21] = [
        (
            "pixel-downscales-1536x1152",
            "grid",
            "a5a6cb0f7fd6c18020b95b55ecdbb589fef8b15512e6c87c63a47f4fe64f0b92",
        ),
        (
            "pixel-downscales-1536x1152",
            "ulp",
            "a5a6cb0f7fd6c18020b95b55ecdbb589fef8b15512e6c87c63a47f4fe64f0b92",
        ),
        (
            "pixel-downscales-1536x1152",
            "frac",
            "810ec665b6fef8901c943b832632d1ae5e7dc6e05b54e7536429d9420a9d5a14",
        ),
        (
            "pixel-downscales-1440x1080",
            "grid",
            "e15c34bb8f1ca197ae85f64c43ad4d18c866d8b9640cf478c8ce02e7e4cf7236",
        ),
        (
            "pixel-downscales-1440x1080",
            "ulp",
            "e15c34bb8f1ca197ae85f64c43ad4d18c866d8b9640cf478c8ce02e7e4cf7236",
        ),
        (
            "pixel-downscales-1440x1080",
            "frac",
            "fd36975bce4f3ede7a2c59afbd86c0691166679fc63791ba1cb086c99f3e2573",
        ),
        (
            "pixel-downscales-1920x1080",
            "grid",
            "48d428cb1852e5e54ac020269ad34f361f3816bfa1975384e1c1e3bef1b03387",
        ),
        (
            "pixel-downscales-1920x1080",
            "ulp",
            "48d428cb1852e5e54ac020269ad34f361f3816bfa1975384e1c1e3bef1b03387",
        ),
        (
            "pixel-downscales-1920x1080",
            "frac",
            "71d2afefba11442056055f137daa6eb529172022ea43bf43431f258eeb7181ed",
        ),
        (
            "pixel-downscales-1366x1080",
            "grid",
            "9857ff78b453810e4bcec887963d8251ef38d34500d278515a13ec047b4fc2cc",
        ),
        (
            "pixel-downscales-1366x1080",
            "ulp",
            "9857ff78b453810e4bcec887963d8251ef38d34500d278515a13ec047b4fc2cc",
        ),
        (
            "pixel-downscales-1366x1080",
            "frac",
            "caec2fd9f40845412cc4b24c5e57d6eeabcbefdb6699695dc4a600b506c2244b",
        ),
        (
            "pixel-downscales-1000x1080",
            "grid",
            "753599b580fcc1231f00e0642effc44f37955f75a7f16bb7a9d6647809548337",
        ),
        (
            "pixel-downscales-1000x1080",
            "ulp",
            "753599b580fcc1231f00e0642effc44f37955f75a7f16bb7a9d6647809548337",
        ),
        (
            "pixel-downscales-1000x1080",
            "frac",
            "264ceee98d6179ad0db33d7a10d196da1c8db63673b939b97ad07d9f0dcb7716",
        ),
        (
            "pixel-downscales-1440x1152",
            "grid",
            "b99e5cd231a628f1fcf76fe1668db429c2827eb7733a363ef9a3a406bd83a88b",
        ),
        (
            "pixel-downscales-1440x1152",
            "ulp",
            "b99e5cd231a628f1fcf76fe1668db429c2827eb7733a363ef9a3a406bd83a88b",
        ),
        (
            "pixel-downscales-1440x1152",
            "frac",
            "aa690279b927f7b00810a4f688eba29d013e134b258b7730c08e07ef72343c0d",
        ),
        (
            "pixel-downscales-1536x1080",
            "grid",
            "67e124800a21073ed1ed42534aa41ad81d37a759b62cd5fc57134fbcaf7bc32b",
        ),
        (
            "pixel-downscales-1536x1080",
            "ulp",
            "67e124800a21073ed1ed42534aa41ad81d37a759b62cd5fc57134fbcaf7bc32b",
        ),
        (
            "pixel-downscales-1536x1080",
            "frac",
            "6c9baf4f13010e70a06f1a1c5a82f9328fd681892d2cabdee00e2f5feb6b9d8f",
        ),
    ];

    /// The SHA-256 of every size's texels and of its three coordinate sets' bytes, the sizes in
    /// [`SIZES`] order and each size's sets in `grid`, `ulp`, `frac` order: the constant
    /// `pixel_downscales::INPUTS_SHA` pins, so the cases ask Metal the lane's question.
    const INPUTS_SHA: &str = "2e8d095059bdcbb66f74ac0c5f3c98d2dfa6f72c065c5fdfd56debbab9a397dd";

    #[test]
    fn the_inputs_are_the_lanes() {
        use sha2::Digest as _;
        let mut sha = sha2::Sha256::new();
        for size in SIZES {
            sha.update(texels(size));
            for set in SETS {
                for cell in coordinates(set, size) {
                    sha.update(cell[0].to_le_bytes());
                    sha.update(cell[1].to_le_bytes());
                }
            }
        }
        let mut hex = String::new();
        for byte in sha.finalize() {
            hex.push_str(&format!("{byte:02x}"));
        }
        assert_eq!(hex, INPUTS_SHA);
    }

    /// Seven cases, one a size, each named for its texture, and every rectangle an eighth of its
    /// texture in whole texels and a whole number of threadgroups.
    #[test]
    fn the_cases_are_one_a_size() {
        let cases = all();
        assert_eq!(cases.len(), 7);
        let names: BTreeSet<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(names.len(), 7);
        for case in &cases {
            assert_eq!(
                case.name,
                format!("pixel-downscales-{}x{}", case.size.width, case.size.height)
            );
            // The rectangle samples the texture's first 8w by 8h texels, which must fit.
            assert!(case.size.rectangle[0] * 8 <= case.size.width);
            assert!(case.size.rectangle[1] * 8 <= case.size.height);
            assert_eq!(case.size.samples() as u32 % GROUP, 0);
            // Every size is past one window at each end, which is what makes the windows two
            // different parts of the rectangle.
            assert!(case.size.samples() > 2 * WINDOW);
        }
    }

    /// The control samples the snapped cell over the size, and the pixel source the coordinate
    /// itself: two different buffers, which is the claim the control makes.
    #[test]
    fn the_control_samples_the_snapped_cell() {
        assert_eq!(snapped(4.0), 4.0);
        assert_eq!(snapped(4.0 + 1.0 / 512.0), 4.0 + 1.0 / 256.0);
        assert_eq!(snapped(4.0 + 1.0 / 1024.0), 4.0);
        for size in SIZES {
            for set in SETS {
                assert_ne!(
                    coordinate_bytes(set, size, false),
                    coordinate_bytes(set, size, true)
                );
            }
        }
    }

    /// The recorded M3 answers are the lane's M4 ones, digest for digest, at every size and every
    /// set. A difference here is a finding: the chips disagree, or the fixture no longer asks the
    /// lane's question.
    #[test]
    fn the_recorded_digests_are_the_lanes_m4_answers() {
        let mut checked = 0;
        for case in all() {
            let recording = crate::recording::read(case.name())
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            let transcript = recording
                .answers
                .first()
                .and_then(|answer| answer.deterministic())
                .unwrap_or_else(|| panic!("{}: no one recorded outcome", case.name()));
            for (named, set, m4) in M4 {
                if named != case.name() {
                    continue;
                }
                let node = node_at(transcript, &format!("case/output/{set}/slot/digest"))
                    .unwrap_or_else(|| panic!("{}: no {set} digest", case.name()));
                let digest =
                    node.observed.digest.as_ref().unwrap_or_else(|| {
                        panic!("{}: the {set} site carries no digest", case.name())
                    });
                assert_eq!(
                    digest.hex,
                    m4,
                    "{} {set}: this M3 Max and the lane's M4 disagree",
                    case.name()
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 21);
    }

    /// The node a `/`-separated path of site names names, the first being the root's.
    fn node_at<'a>(
        transcript: &'a crate::transcript::Transcript,
        path: &str,
    ) -> Option<&'a TranscriptNode> {
        let mut names = path.split('/');
        let mut node = &transcript.root;
        if names.next()? != node.name {
            return None;
        }
        for name in names {
            node = node.child(name)?;
        }
        Some(node)
    }
}
