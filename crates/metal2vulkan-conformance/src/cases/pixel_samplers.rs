//! Samples through pixel-coordinate samplers, every way a kernel can reach one.
//!
//! A 5 by 3 texture — neither side a power of two — of `RGBA8Unorm` or `RGBA16Float` is sampled at
//! 800 coordinates in and past it (`grid`: per integer from -1 to 6 the texel edge and one ulp
//! either side, `i + 1/512`, the quarter, the centre and one ulp either side, three quarters and
//! `255/256`, across ten rows from -0.5 to 3.25). Each case is one format, one filter and one
//! address mode, and holds a site per **source** — the four ways the lane sweeps:
//!
//! - `slot`: the texture at texture slot 0, through a runtime sampler state of pixel coordinates
//!   at sampler slot 0 (the owned fixture `xr_samples`' `xr_sample_runtime`);
//! - `arg`: the texture and that same pixel state inside an argument buffer (the owned fixture
//!   `px_samplers`' `px_sample_argument`);
//! - `norm`: the same argument buffer holding a **normalized** state of the same filter and mode,
//!   sampled at the coordinate divided by the size — the control the lane compares against;
//! - `const`: the texture at slot 0 through the fixture's constexpr `coord::pixel` sampler of that
//!   filter and mode, where MSL can state it (not mirror clamp to edge, which it cannot).
//!
//! **What the port changes.** `runtime_lane::pixel_samplers` holds these sixty answers as sixty
//! SHA-256 constants taken on an Apple M4, and compares a digest; on a difference it prints the
//! cells that differ from the device's own `norm` control, which is a device-against-device check,
//! not a check against Metal. Worse, the lane's AIR is an inline `&str` in the opaque-pointer
//! dialect, which `xcrun metal-as` does not read, so the lane could have no Metal leg at all. The
//! owned fixture `px_samplers` is the two shapes it needed, and record mode now runs all sixty
//! configurations on native Metal and keeps every one of the 800 results of each, so a comparison
//! names the coordinate and the lane rather than saying a hash moved.
//!
//! The digest stays beside each value as a projection of it, which is what lets the module's own
//! `the_recorded_digests_are_the_lanes_m4_answers` hold the recorded M3 answers against the lane's
//! M4 constants: same question, two chips.
//!
//! **What may differ, and what justifies it.** On a device with Metal's own sampler
//! (`VkProperty::MetalSampler`, MoltenVK here) every source is Metal's and nothing is announced.
//! Elsewhere the emulation reaches a slot's and a constexpr sampler's pixel coordinates exactly,
//! while an argument buffer's sampler is handed to the device's own, which announces
//! `argument-pixel-sampler-scaling`, and a device that converts an 8-bit texel to a float its own
//! way announces `float-sample-filter`. Those two sites — `arg` and `norm` — are the only ones
//! loosened, each naming the notice that must have fired.
//!
//! Nothing here can fault a GPU: one small texture, three or four compute passes a case, no loop.
//! The coordinates do reach outside the texture, which is what the address modes are for.

use crate::allowance::Allowance;
use crate::case::{
    made, Abort, ArgumentSlot, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec,
    Side, TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::notice_allowance;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
};
use metal2vulkan_runtime::Degradation;

/// The fixture whose kernel samples a texture slot through the runtime sampler at slot 0.
const SLOT_FIXTURE: &str = "xr_samples";

/// That kernel.
const SLOT_ENTRY: &str = "xr_sample_runtime";

/// The fixture written for the pixel-sampler lanes: the argument-buffer kernel and the constexpr
/// `coord::pixel` samplers.
const PIXEL_FIXTURE: &str = "px_samplers";

/// Its argument-buffer kernel.
const ARGUMENT_ENTRY: &str = "px_sample_argument";

/// The texture's size: neither side a power of two, as the lane's is.
const WIDTH: u32 = 5;
const HEIGHT: u32 = 3;

/// The coordinates each case samples at.
const SAMPLES: usize = 800;

/// The threads in a threadgroup, chosen so 800 threads are ten whole groups: the group size
/// changes nothing a sampler does, and a whole-group dispatch keeps the question to the sampler.
const GROUP: u32 = 80;

/// The byte every result buffer holds before the pass.
const BLANK: u8 = 0xab;

/// The lanes of one result: a `float4` a sample.
const LANES: u32 = 4;

/// The bytes one result takes.
const STRIDE: u32 = LANES * 4;

/// The probe's coordinates, `x` fastest: per integer `i` from -1 to `WIDTH + 1` the texel edge `i`
/// and one ulp either side (`2^-24` at 0), `i + 1/512`, `i + 0.25`, the centre `i + 0.5` and one
/// ulp either side, `i + 0.75` and `i + 255/256`; `y` from ten values across and past the 3 rows.
fn grid() -> Vec<[f32; 2]> {
    let mut xs = Vec::new();
    for i in -1..=(WIDTH as i32 + 1) {
        let (b, h) = (i as f32, i as f32 + 0.5);
        let (lo, hi) = match i {
            0 => (-(2f32.powi(-24)), 2f32.powi(-24)),
            _ => (b.next_down(), b.next_up()),
        };
        xs.extend([
            lo,
            b,
            hi,
            b + 2f32.powi(-9),
            b + 0.25,
            h.next_down(),
            h,
            h.next_up(),
            b + 0.75,
            b + 255.0 / 256.0,
        ]);
    }
    let ys = [
        -0.5,
        0.0,
        0.5,
        1f32.next_down(),
        1.0,
        1.25,
        1.5f32.next_up(),
        2.5,
        3.0,
        3.25,
    ];
    ys.iter()
        .flat_map(|&y| xs.iter().map(move |&x| [x, y]))
        .collect()
}

/// The coordinate buffer a source reads: the grid as it stands for a pixel sampler, or divided by
/// the size for the normalized control.
fn coordinates(normalized: bool) -> Vec<u8> {
    grid()
        .iter()
        .flat_map(|&[x, y]| match normalized {
            true => [x / WIDTH as f32, y / HEIGHT as f32],
            false => [x, y],
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The half of `value`, where a half holds `value` exactly: the single's exponent rebiased and the
/// top ten bits of its mantissa. The texels are quarters of magnitude at most 3.5, which is what
/// the module's own `the_texels_are_the_probes_quarters` holds of every one of them, so nothing
/// here rounds.
fn half_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = ((bits >> 13) & 0x3ff) as u16;
    sign | ((exponent as u16) << 10) | mantissa
}

/// The texels of a texture of `format`, as the probe wrote them: an `RGBA8Unorm` channel the code
/// `(53x + 97y + 29c + 11) mod 256`, and a `RGBA16Float` one the half `k / 4` for
/// `k = ((7x + 13y + 5c + 3) mod 29) - 14`.
fn texels(format: PixelFormat) -> Vec<u8> {
    let mut bytes = Vec::new();
    for y in 0..u64::from(HEIGHT) {
        for x in 0..u64::from(WIDTH) {
            for c in 0..4 {
                match format {
                    PixelFormat::RGBA8Unorm => {
                        bytes.push(((x * 53 + y * 97 + c * 29 + 11) % 256) as u8);
                    }
                    _ => {
                        let k = ((x * 7 + y * 13 + c * 5 + 3) % 29) as i32 - 14;
                        bytes.extend(half_bits(k as f32 / 4.0).to_le_bytes());
                    }
                }
            }
        }
    }
    bytes
}

/// The bytes a row of a texture of `format` takes.
fn row_pitch(format: PixelFormat) -> u32 {
    WIDTH
        * if format == PixelFormat::RGBA8Unorm {
            4
        } else {
            8
        }
}

/// The layout the results are read under: one row a sample of four lanes, so a diff names the
/// coordinate and the lane rather than a byte offset.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(STRIDE) * SAMPLES as u64,
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: LANES,
        row_pitch: STRIDE,
        rows: SAMPLES as u32,
        image_pitch: STRIDE * SAMPLES as u32,
        images: 1,
        // Each sample's four lanes fill its row, so there is no padding between them.
        gaps: Gaps::Ignored,
    }
}

/// An address mode the cases sweep: the lane's name for it, the mode, and the constexpr kernel of
/// that mode, where MSL can state one.
///
/// A constexpr sampler states clamp to zero and clamp to the border as one addressing mode under
/// two border colours, and has no spelling at all for mirror clamp to edge.
struct Address {
    mode: SamplerAddressMode,
    /// The constexpr kernels of this mode, nearest filter first.
    constexpr_entries: Option<[&'static str; 2]>,
}

/// The four address modes, in the lane's order.
const ADDRESSES: [Address; 4] = [
    Address {
        mode: SamplerAddressMode::ClampToEdge,
        constexpr_entries: Some(["px_const_nearest_edge", "px_const_linear_edge"]),
    },
    Address {
        mode: SamplerAddressMode::ClampToZero,
        constexpr_entries: Some(["px_const_nearest_zero", "px_const_linear_zero"]),
    },
    Address {
        mode: SamplerAddressMode::ClampToBorderColor,
        constexpr_entries: Some(["px_const_nearest_border", "px_const_linear_border"]),
    },
    Address {
        mode: SamplerAddressMode::MirrorClampToEdge,
        constexpr_entries: None,
    },
];

/// The sixteen case names, in the order [`all`] builds them. Written out rather than formatted, so
/// every case's name is a `&'static str` the suite can hold and `conformance replay <name>` can
/// select.
const NAMES: [&str; 16] = [
    "pixel-samplers-rgba8-nearest-edge",
    "pixel-samplers-rgba8-nearest-zero",
    "pixel-samplers-rgba8-nearest-border",
    "pixel-samplers-rgba8-nearest-mirror",
    "pixel-samplers-rgba8-linear-edge",
    "pixel-samplers-rgba8-linear-zero",
    "pixel-samplers-rgba8-linear-border",
    "pixel-samplers-rgba8-linear-mirror",
    "pixel-samplers-rgba16f-nearest-edge",
    "pixel-samplers-rgba16f-nearest-zero",
    "pixel-samplers-rgba16f-nearest-border",
    "pixel-samplers-rgba16f-nearest-mirror",
    "pixel-samplers-rgba16f-linear-edge",
    "pixel-samplers-rgba16f-linear-zero",
    "pixel-samplers-rgba16f-linear-border",
    "pixel-samplers-rgba16f-linear-mirror",
];

/// The site an argument buffer's pixel sampler answers at, and the site its normalized control
/// answers at: the two a device without Metal's own sampler may differ at.
const ARGUMENT_SITE: &str = "arg";
const CONTROL_SITE: &str = "norm";

/// A case: one format, one filter, one address mode, and the sources it can run.
pub struct PixelSamplers {
    name: &'static str,
    format: PixelFormat,
    filter: SamplerMinMagFilter,
    address: &'static Address,
}

/// Every pairing of format, filter and address mode, as the lane sweeps them.
pub fn all() -> Vec<PixelSamplers> {
    let mut cases = Vec::new();
    let mut names = NAMES.iter();
    for format in [PixelFormat::RGBA8Unorm, PixelFormat::RGBA16Float] {
        for filter in [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear] {
            for address in &ADDRESSES {
                cases.push(PixelSamplers {
                    name: names.next().expect("a name per pairing"),
                    format,
                    filter,
                    address,
                });
            }
        }
    }
    cases
}

impl PixelSamplers {
    /// The sampler state this case's sources use: this filter on both axes, this mode on both,
    /// an opaque white border as the probe's states had, and argument-buffer support so an
    /// argument buffer can name it. Pixel coordinates unless `normalized`.
    fn sampler_spec(&self, normalized: bool) -> SamplerSpec {
        let mut spec = SamplerSpec::new().for_argument_buffer();
        spec.min_filter = self.filter;
        spec.mag_filter = self.filter;
        spec.s_address_mode = self.address.mode;
        spec.t_address_mode = self.address.mode;
        spec.border_color = SamplerBorderColor::OpaqueWhite;
        spec.normalized_coordinates = normalized;
        spec
    }

    /// The constexpr kernel of this case's filter and mode, where the fixture declares one.
    fn constexpr_entry(&self) -> Option<&'static str> {
        self.address
            .constexpr_entries
            .map(|entries| match self.filter {
                SamplerMinMagFilter::Linear => entries[1],
                _ => entries[0],
            })
    }

    /// Run one source's 800 samples and emit its results.
    fn samples(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        pipeline: Handle,
        bindings: Sources,
    ) -> Result<Node, Abort> {
        let length = SAMPLES * STRIDE as usize;
        let output = made(
            sites,
            &format!("{at}/buffer"),
            side.buffer(&vec![BLANK; length]),
        )?;
        let textures: Vec<Handle> = bindings.texture.into_iter().collect();
        let mut buffers = vec![bindings.coordinates, output];
        buffers.extend(bindings.argument);
        let samplers: Vec<Handle> = bindings.sampler.into_iter().collect();
        let command_buffer = made(
            sites,
            &format!("{at}/dispatch"),
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&textures, &buffers).sampled(&samplers),
                &[],
                [SAMPLES as u32, 1, 1],
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
        let digest = Digest::sha256(&bytes);
        Ok(Node::leaf(
            at.to_owned(),
            Observed::bytes(bytes, layout()).digested(digest),
        ))
    }
}

/// What one source binds: the texture at a slot, the sampler at a slot, the argument buffer, and
/// the coordinates. One value rather than four arguments, so the four sources of a case cannot
/// pick up different binding orders.
struct Sources {
    texture: Option<Handle>,
    sampler: Option<Handle>,
    argument: Option<Handle>,
    coordinates: Handle,
}

impl Case for PixelSamplers {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        let mut entries = vec![(SLOT_FIXTURE, SLOT_ENTRY), (PIXEL_FIXTURE, ARGUMENT_ENTRY)];
        if let Some(entry) = self.constexpr_entry() {
            entries.push((PIXEL_FIXTURE, entry));
        }
        for (fixture, entry) in entries {
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
        Ok(vec![
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
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("sampler:pixel", &self.sampler_spec(false).words()),
            Input::bytes("sampler:normalized", &self.sampler_spec(true).words()),
            Input::bytes("texture:texels", &texels(self.format)),
            Input::bytes("coordinates:pixel", &coordinates(false)),
            Input::bytes("coordinates:normalized", &coordinates(true)),
            Input::bytes("buffer:blank", &[BLANK]),
        ])
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Metal announces nothing, so a notice of ours is a difference at the notices site.
        // These are the two kinds a run of this round provoked: on lavapipe an argument buffer's
        // pixel sampler is handed to the device's own, and the device converts an 8-bit texel to a
        // float its own way. A venue that announces another kind — `pixel-sampler-scaling`, which
        // neither venue here provoked — reads as divergent at that site until whoever adds the
        // venue names it with a run behind it.
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
        ]
        .iter()
        .map(|kind| notice_allowance(kind.code()))
        .chain([
            // The buffer each pass writes its sampled texels into is `Shared`
            // ([`crate::replay::shared_memory_notice`]).
            crate::replay::shared_memory_notice(),
        ])
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
                    self.format,
                    [WIDTH, HEIGHT],
                    row_pitch(self.format),
                    TextureRole::Read,
                ),
                &texels(self.format),
            ),
        )?;
        let pixel_coordinates = made(sites, "coordinates/pixel", side.buffer(&coordinates(false)))?;
        let normalized_coordinates = made(
            sites,
            "coordinates/normalized",
            side.buffer(&coordinates(true)),
        )?;
        let pixel_state = made(
            sites,
            "sampler/pixel",
            side.sampler(&self.sampler_spec(false)),
        )?;
        let normalized_state = made(
            sites,
            "sampler/normalized",
            side.sampler(&self.sampler_spec(true)),
        )?;
        let pixel_arguments = made(
            sites,
            "argument-buffer/pixel",
            side.argument_buffer(&[
                ArgumentSlot::Texture(texture),
                ArgumentSlot::Sampler(pixel_state),
            ]),
        )?;
        let normalized_arguments = made(
            sites,
            "argument-buffer/normalized",
            side.argument_buffer(&[
                ArgumentSlot::Texture(texture),
                ArgumentSlot::Sampler(normalized_state),
            ]),
        )?;

        let mut output = Node::branch("output", Vec::new());
        output = output.with(self.samples(
            side,
            sites,
            "slot",
            slot_pipeline,
            Sources {
                texture: Some(texture),
                sampler: Some(pixel_state),
                argument: None,
                coordinates: pixel_coordinates,
            },
        )?);
        output = output.with(self.samples(
            side,
            sites,
            ARGUMENT_SITE,
            argument_pipeline,
            Sources {
                texture: None,
                sampler: None,
                argument: Some(pixel_arguments),
                coordinates: pixel_coordinates,
            },
        )?);
        output = output.with(self.samples(
            side,
            sites,
            CONTROL_SITE,
            argument_pipeline,
            Sources {
                texture: None,
                sampler: None,
                argument: Some(normalized_arguments),
                coordinates: normalized_coordinates,
            },
        )?);
        if let Some(entry) = self.constexpr_entry() {
            let pipeline = made(
                sites,
                "pipeline/constexpr",
                side.compute_pipeline(pixel_library, entry),
            )?;
            output = output.with(self.samples(
                side,
                sites,
                "const",
                pipeline,
                Sources {
                    texture: Some(texture),
                    sampler: None,
                    argument: None,
                    coordinates: pixel_coordinates,
                },
            )?);
        }
        sites.push(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Observed;
    use std::collections::BTreeSet;

    /// What an Apple M4 answered for each configuration, as `runtime_lane::pixel_samplers`' own
    /// probe measured it (macOS 27.0.1, two processes, identical): the case, the source, and the
    /// SHA-256 of its 800 `float4` results. Our own measurement over our own inputs, carried here
    /// so the recorded M3 answers can be held against it.
    const M4: [(&str, &str, &str); 60] = [
        (
            "pixel-samplers-rgba8-nearest-edge",
            "slot",
            "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
        ),
        (
            "pixel-samplers-rgba8-nearest-edge",
            "arg",
            "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
        ),
        (
            "pixel-samplers-rgba8-nearest-edge",
            "norm",
            "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
        ),
        (
            "pixel-samplers-rgba8-nearest-edge",
            "const",
            "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
        ),
        (
            "pixel-samplers-rgba8-nearest-zero",
            "slot",
            "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
        ),
        (
            "pixel-samplers-rgba8-nearest-zero",
            "arg",
            "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
        ),
        (
            "pixel-samplers-rgba8-nearest-zero",
            "norm",
            "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
        ),
        (
            "pixel-samplers-rgba8-nearest-zero",
            "const",
            "4ce73456509dc569e5a67d724ecd5f2bea5dda4d8b7279b5f8e4a7351998c5be",
        ),
        (
            "pixel-samplers-rgba8-nearest-border",
            "slot",
            "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
        ),
        (
            "pixel-samplers-rgba8-nearest-border",
            "arg",
            "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
        ),
        (
            "pixel-samplers-rgba8-nearest-border",
            "norm",
            "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
        ),
        (
            "pixel-samplers-rgba8-nearest-border",
            "const",
            "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
        ),
        (
            "pixel-samplers-rgba8-nearest-mirror",
            "slot",
            "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
        ),
        (
            "pixel-samplers-rgba8-nearest-mirror",
            "arg",
            "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
        ),
        (
            "pixel-samplers-rgba8-nearest-mirror",
            "norm",
            "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
        ),
        (
            "pixel-samplers-rgba8-linear-edge",
            "slot",
            "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
        ),
        (
            "pixel-samplers-rgba8-linear-edge",
            "arg",
            "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
        ),
        (
            "pixel-samplers-rgba8-linear-edge",
            "norm",
            "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
        ),
        (
            "pixel-samplers-rgba8-linear-edge",
            "const",
            "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
        ),
        (
            "pixel-samplers-rgba8-linear-zero",
            "slot",
            "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
        ),
        (
            "pixel-samplers-rgba8-linear-zero",
            "arg",
            "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
        ),
        (
            "pixel-samplers-rgba8-linear-zero",
            "norm",
            "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
        ),
        (
            "pixel-samplers-rgba8-linear-zero",
            "const",
            "8c8ebb90b930331d3790933171428b8edff1c8296f8bb344a6c06362dc382c83",
        ),
        (
            "pixel-samplers-rgba8-linear-border",
            "slot",
            "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
        ),
        (
            "pixel-samplers-rgba8-linear-border",
            "arg",
            "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
        ),
        (
            "pixel-samplers-rgba8-linear-border",
            "norm",
            "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
        ),
        (
            "pixel-samplers-rgba8-linear-border",
            "const",
            "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
        ),
        (
            "pixel-samplers-rgba8-linear-mirror",
            "slot",
            "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
        ),
        (
            "pixel-samplers-rgba8-linear-mirror",
            "arg",
            "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
        ),
        (
            "pixel-samplers-rgba8-linear-mirror",
            "norm",
            "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
        ),
        (
            "pixel-samplers-rgba16f-nearest-edge",
            "slot",
            "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
        ),
        (
            "pixel-samplers-rgba16f-nearest-edge",
            "arg",
            "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
        ),
        (
            "pixel-samplers-rgba16f-nearest-edge",
            "norm",
            "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
        ),
        (
            "pixel-samplers-rgba16f-nearest-edge",
            "const",
            "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
        ),
        (
            "pixel-samplers-rgba16f-nearest-zero",
            "slot",
            "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
        ),
        (
            "pixel-samplers-rgba16f-nearest-zero",
            "arg",
            "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
        ),
        (
            "pixel-samplers-rgba16f-nearest-zero",
            "norm",
            "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
        ),
        (
            "pixel-samplers-rgba16f-nearest-zero",
            "const",
            "7b239607fcf4299166fc92e5300ea888e7224f8a7ed32d7e5265cb78cc62657e",
        ),
        (
            "pixel-samplers-rgba16f-nearest-border",
            "slot",
            "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
        ),
        (
            "pixel-samplers-rgba16f-nearest-border",
            "arg",
            "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
        ),
        (
            "pixel-samplers-rgba16f-nearest-border",
            "norm",
            "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
        ),
        (
            "pixel-samplers-rgba16f-nearest-border",
            "const",
            "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
        ),
        (
            "pixel-samplers-rgba16f-nearest-mirror",
            "slot",
            "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
        ),
        (
            "pixel-samplers-rgba16f-nearest-mirror",
            "arg",
            "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
        ),
        (
            "pixel-samplers-rgba16f-nearest-mirror",
            "norm",
            "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
        ),
        (
            "pixel-samplers-rgba16f-linear-edge",
            "slot",
            "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
        ),
        (
            "pixel-samplers-rgba16f-linear-edge",
            "arg",
            "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
        ),
        (
            "pixel-samplers-rgba16f-linear-edge",
            "norm",
            "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
        ),
        (
            "pixel-samplers-rgba16f-linear-edge",
            "const",
            "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
        ),
        (
            "pixel-samplers-rgba16f-linear-zero",
            "slot",
            "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
        ),
        (
            "pixel-samplers-rgba16f-linear-zero",
            "arg",
            "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
        ),
        (
            "pixel-samplers-rgba16f-linear-zero",
            "norm",
            "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
        ),
        (
            "pixel-samplers-rgba16f-linear-zero",
            "const",
            "dbb87cfd8c4ab3c54dede294e5ebc42ad0a2fd84166cc1b46a33e3a3b61e1777",
        ),
        (
            "pixel-samplers-rgba16f-linear-border",
            "slot",
            "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
        ),
        (
            "pixel-samplers-rgba16f-linear-border",
            "arg",
            "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
        ),
        (
            "pixel-samplers-rgba16f-linear-border",
            "norm",
            "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
        ),
        (
            "pixel-samplers-rgba16f-linear-border",
            "const",
            "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
        ),
        (
            "pixel-samplers-rgba16f-linear-mirror",
            "slot",
            "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
        ),
        (
            "pixel-samplers-rgba16f-linear-mirror",
            "arg",
            "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
        ),
        (
            "pixel-samplers-rgba16f-linear-mirror",
            "norm",
            "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
        ),
    ];

    /// The SHA-256 of the lane's coordinates, `x` then `y` little-endian: the constant
    /// `pixel_samplers::GRID_SHA` pins, so the case asks Metal the lane's question.
    const GRID_SHA: &str = "10064f01ccb480a13ee9b0853ba41bbc5e2dfa3d490ffaef9cbcc987f75249b9";

    #[test]
    fn the_coordinates_are_the_lanes() {
        let grid = grid();
        assert_eq!(grid.len(), SAMPLES);
        let bytes: Vec<u8> = grid
            .iter()
            .flat_map(|cell| cell.iter().flat_map(|axis| axis.to_le_bytes()))
            .collect();
        assert_eq!(crate::sha256_hex(&bytes), GRID_SHA);
        // The pixel coordinate buffer is that grid; the normalized one is it over the size.
        assert_eq!(coordinates(false), bytes);
        let normalized = coordinates(true);
        for (index, cell) in grid.iter().enumerate() {
            let at = index * 8;
            let axis = |offset: usize| {
                f32::from_le_bytes(normalized[at + offset..at + offset + 4].try_into().unwrap())
            };
            assert_eq!(axis(0), cell[0] / WIDTH as f32);
            assert_eq!(axis(4), cell[1] / HEIGHT as f32);
        }
    }

    /// Every `RGBA16Float` texel is the quarter the probe wrote, exactly: the half reads back as
    /// `k / 4` for the `k` the generator chose, so nothing in [`half_bits`] rounded.
    #[test]
    fn the_texels_are_the_probes_quarters() {
        let bytes = texels(PixelFormat::RGBA16Float);
        assert_eq!(bytes.len(), (WIDTH * HEIGHT * 4 * 2) as usize);
        let mut at = 0;
        for y in 0..u64::from(HEIGHT) {
            for x in 0..u64::from(WIDTH) {
                for c in 0..4 {
                    let k = ((x * 7 + y * 13 + c * 5 + 3) % 29) as i32 - 14;
                    let half = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                    assert_eq!(
                        half_to_f32(half),
                        k as f32 / 4.0,
                        "texel ({x}, {y}) channel {c}"
                    );
                    at += 2;
                }
            }
        }
        // The 8-bit texels are one byte a channel of the probe's codes.
        let codes = texels(PixelFormat::RGBA8Unorm);
        assert_eq!(codes.len(), (WIDTH * HEIGHT * 4) as usize);
        assert_eq!(codes[0], 11);
    }

    /// Where a source's results sit in a case's transcript.
    fn site(source: &str) -> String {
        format!("case/output/{source}")
    }

    /// The lane's name for a format and for a filter, as the case names spell them.
    fn format_name(format: PixelFormat) -> &'static str {
        match format {
            PixelFormat::RGBA8Unorm => "rgba8",
            _ => "rgba16f",
        }
    }

    fn filter_name(filter: SamplerMinMagFilter) -> &'static str {
        match filter {
            SamplerMinMagFilter::Linear => "linear",
            _ => "nearest",
        }
    }

    /// The lane's name for an address mode.
    fn address_name(mode: SamplerAddressMode) -> &'static str {
        match mode {
            SamplerAddressMode::ClampToEdge => "edge",
            SamplerAddressMode::ClampToZero => "zero",
            SamplerAddressMode::ClampToBorderColor => "border",
            _ => "mirror",
        }
    }

    /// The node a `/`-separated path of site names names, the first being the root's.
    fn node_at<'a>(
        transcript: &'a crate::transcript::Transcript,
        path: &str,
    ) -> Option<&'a crate::transcript::Node> {
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

    /// A half's value, for the texel check alone.
    fn half_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let mantissa = f32::from(bits & 0x3ff);
        match exponent {
            0 => sign * mantissa * 2f32.powi(-24),
            _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
        }
    }

    /// Sixteen cases, each a different question, and each name says which format, filter and mode
    /// the case is of, so the names and the pairings cannot drift apart. Mirror clamp to edge is
    /// the one mode with no constexpr sampler, which MSL cannot spell.
    #[test]
    fn the_cases_sweep_every_pairing_once() {
        let cases = all();
        assert_eq!(cases.len(), 16);
        let names: BTreeSet<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(names.len(), 16);
        let mut with_constexpr = 0;
        for case in &cases {
            let (format, filter, address) = (
                format_name(case.format),
                filter_name(case.filter),
                address_name(case.address.mode),
            );
            assert_eq!(
                case.name,
                format!("pixel-samplers-{format}-{filter}-{address}")
            );
            match case.constexpr_entry() {
                Some(entry) => {
                    with_constexpr += 1;
                    assert!(entry.contains(filter), "{entry} of {}", case.name);
                    assert!(entry.ends_with(address), "{entry}");
                }
                None => assert_eq!(address, "mirror"),
            }
        }
        assert_eq!(with_constexpr, 12);
        // Sixty configurations in all, which is the lane's sweep: four sources where a constexpr
        // sampler exists, three where it does not.
        let configurations: usize = cases
            .iter()
            .map(|case| 3 + usize::from(case.constexpr_entry().is_some()))
            .sum();
        assert_eq!(configurations, 60);
    }

    /// Both sampler states a case makes are its filter and mode, one of pixel coordinates and one
    /// normalized, and both support argument buffers, which is what lets the argument buffer name
    /// them.
    #[test]
    fn the_states_differ_only_in_their_coordinates() {
        for case in all() {
            let pixel = case.sampler_spec(false);
            let normalized = case.sampler_spec(true);
            assert!(!pixel.normalized_coordinates);
            assert!(normalized.normalized_coordinates);
            assert!(pixel.support_argument_buffers);
            assert!(normalized.support_argument_buffers);
            assert_eq!(
                SamplerSpec {
                    normalized_coordinates: true,
                    ..pixel
                },
                normalized
            );
            assert_eq!(pixel.min_filter, case.filter);
            assert_eq!(pixel.s_address_mode, case.address.mode);
            assert_ne!(pixel.words(), normalized.words());
        }
    }

    /// The recorded M3 answers are the lane's M4 ones, digest for digest: the same sixty
    /// configurations, the same 800 coordinates, two chips. A difference here is a finding — the
    /// chips disagree, or the fixture no longer asks the lane's question — and the comparison that
    /// matters is the suite's own, against the recording, not against this table.
    #[test]
    fn the_recorded_digests_are_the_lanes_m4_answers() {
        let mut checked = 0;
        for case in all() {
            let recording = crate::recording::read(case.name())
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            let answer = recording
                .answers
                .first()
                .unwrap_or_else(|| panic!("{}: no answer", case.name()));
            let transcript = answer
                .deterministic()
                .unwrap_or_else(|| panic!("{}: more than one outcome", case.name()));
            for (named, source, m4) in M4 {
                if named != case.name() {
                    continue;
                }
                let node = node_at(transcript, &site(source))
                    .unwrap_or_else(|| panic!("{}: no {source} site", case.name()));
                let Observed {
                    digest: Some(digest),
                    ..
                } = &node.observed
                else {
                    panic!("{}: the {source} site carries no digest", case.name());
                };
                assert_eq!(
                    digest.hex,
                    m4,
                    "{} {source}: this M3 Max and the lane's M4 disagree",
                    case.name()
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 60);
    }

    /// The layout covers every result and names the sample as its row, so a diff over 800 samples
    /// says which coordinate moved.
    #[test]
    fn the_layout_names_the_sample() {
        let layout = layout();
        assert_eq!(layout.extent(), SAMPLES * STRIDE as usize);
        assert_eq!(layout.at(0, 799, 3), 799 * STRIDE as usize + 3 * 4);
    }
}
