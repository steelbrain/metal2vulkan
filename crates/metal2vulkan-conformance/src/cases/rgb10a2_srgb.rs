//! What a kernel reads from, writes to and samples of, and what a draw stores into, an
//! `RGB10A2Unorm_sRGB` (586) texture.
//!
//! Eight cases over Apple's padded 10-bit sRGB format: the three of
//! `runtime_lane::rgb10a2_srgb`'s nine questions a compute pass can ask, and the one a render pass
//! asks whose answer was truncated:
//!
//! - `rgb10a2-srgb-reads`: a 1024 by 1 texture of the words
//!   `c | (1023 - c) << 10 | (7 c mod 1024) << 20 | (c mod 4) << 30`, every texel read by the owned
//!   fixture `fpv_bfloat_textures`' `fpv_bfloat_fetch`, so every colour code occurs once in each
//!   channel and the two bits the format pads with are set as well.
//! - `rgb10a2-srgb-writes`: `fpv_bfloat_write` of `k / 16384` in red, green and blue and
//!   `(k mod 4096) / 4095` in alpha for `k` up to 16,384, then sixteen specials in every channel,
//!   into a 4096 by 5 texture whose words are then read back.
//! - `rgb10a2-srgb-samples-<nearest|linear>-<f|h>`: a 16 by 16 texture of the probe's xorshift
//!   words, sampled at 4,096 of the probe's positions through a runtime sampler (clamp to edge),
//!   with a `float` and a `half` result.
//! - `rgb10a2-srgb-stores-<f|h>`: one full-screen draw of the owned fixture `rt6_samples`'
//!   `rt6_fill_vertex` and its `float4` or `half4` fragment fill of the same sweep (with the
//!   probe `r.m`'s twenty render specials, which are not the compute write's sixteen) into a new
//!   4096 by 5 private target, whose 20,480 words are read back whole through a blit.
//!
//! **What the port changes, and the defect it found.** The lane states Metal's answer for each of
//! these as the SHA-256's **first sixteen hexadecimal digits of a printed string** — `read <c> <r>
//! <g> <b> <a>` a line, and so on. Two things are wrong with that as a conformance claim, and the
//! second is a real hole:
//!
//! - A 64-bit projection of a formatted string is weaker than the bytes. These cases record the
//!   words and the results themselves, so the comparison names the code or the sample.
//! - **The writes question compared 4,079 fewer words than the kernel wrote.** The lane dispatches
//!   a 4096 by 5 grid, so the kernel writes all 20,480 texels, and the lane formatted only
//!   `got.iter().take(n)` of them, `n` being 16,401 — the 16,385 ramp values plus the sixteen
//!   specials. The remaining 4,079 texels are written from the input buffer's zero padding and
//!   appeared in no digest, inside a passing test. This case records all 20,480 words, so what the
//!   padding stores is Metal's measured answer rather than a gap; the lane now digests the whole
//!   grid too, against the digest of these recorded words. (It is the same shape batch 1a found in
//!   `surface_fills`, whose formatter truncated each run at 24 bytes.)
//!
//! **The second hole, and what it cost to close.** The stores question had the same shape as the
//! writes one and was worse off: the lane digested `got.iter().take(16405)` of the 20,480 words the
//! draw stores, so **4,075 words of each of the two stores were in no comparison**, and unlike the
//! writes nobody had ever recorded Metal's words there — the writes recording covers the compute
//! write, not the render store. Closing it needed the thing this module's docs used to say was
//! missing, a non-mesh render pass in the case vocabulary ([`crate::case::FragmentFill`]): a
//! classic vertex-and-fragment pipeline, a private render target, one `drawPrimitives:` of three
//! vertices, and the target read back through a blit. Both store cases record all 20,480 words, so
//! the tail is Metal's measured answer and the lane's digest now covers the whole target.
//!
//! The target is **private** and read back by a blit, as the probe measured it, not shared and read
//! by `getBytes`: storage mode is part of what a padded format's answer means, and a case that
//! quietly moved it would be measuring something else.
//!
//! Left to the lane, and why: the clears and the loads, whose answers the lane already digests
//! whole, and the `isRGB10A2GammaSupported` check, which is a device property rather than an
//! observation.
//!
//! Every word and every sample compares exactly and the cases carry **no allowance**: on both
//! venues this round covers, the runtime announces nothing here. A venue that does will read as
//! divergent until whoever adds it names the notice with a run behind it.
//!
//! Nothing here can fault a GPU: one compute or render pass a case, one texture and two or three
//! buffers, no loop. The sampled positions do reach outside the texture, which is what clamp to
//! edge is for.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, FragmentFill, Handle, Input, MetalSource,
    SamplerSpec, Side, TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::notice_allowance;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, SamplerMinMagFilter};

/// The format every case is of.
const FORMAT: PixelFormat = PixelFormat::RGB10A2UnormSRGB;

/// Four bytes a texel, which is what the format's word is.
const WORD_BYTES: u32 = 4;

/// The fixture the reads and the writes run.
const TEXTURE_FIXTURE: &str = "fpv_bfloat_textures";

/// The kernel that reads texel `gid`.
const FETCH: &str = "fpv_bfloat_fetch";

/// The kernel that writes `values[gid]` to texel `gid`.
const WRITE: &str = "fpv_bfloat_write";

/// The fixture the samples run.
const SAMPLE_FIXTURE: &str = "xr_samples";

/// The read case's texture: 1024 texels, one a colour code.
const READ_WIDTH: u32 = 1024;

/// The write case's texture width, as the probe dispatched it.
const WRITE_WIDTH: u32 = 4096;

/// The values the write case sweeps: the ramp, then the specials.
const WRITE_VALUES: usize = 16_385 + WRITE_SPECIALS.len();

/// The write case's texture height: enough rows for the sweep, so the grid covers 20,480 texels of
/// which 4,079 take the input buffer's zero padding.
const WRITE_HEIGHT: u32 = WRITE_VALUES.div_ceil(WRITE_WIDTH as usize) as u32;

/// The sampled texture's side.
const SAMPLE_SIDE: u32 = 16;

/// The positions each sample case samples at.
const SAMPLES: usize = 4096;

/// The threads in a threadgroup, as the probe dispatched them.
const GROUP: u32 = 64;

/// The byte every output word holds before a pass.
const BLANK: u8 = 0xab;

/// The seed the probe drew the sampled texels and the positions from, in that order.
const SAMPLE_SEED: u64 = 0x243f_6a88_85a3_08d3;

/// The write sweep's specials, in every channel: the signed zero and one, both NaNs, both
/// infinities, values past the range, denormals, the ties either side of a code and two values
/// inside the sRGB encode's steps.
const WRITE_SPECIALS: [f32; 16] = [
    -0.0,
    -1.0,
    f32::NAN,
    -f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1.5,
    2.0,
    1e-30,
    1e-45,
    0.999_999_94,
    1.000_000_1,
    0.5 / 8192.0,
    1.5 / 8192.0,
    0.003_130_8,
    0.040_45,
];

/// The fixture the store cases draw, whose full-screen triangle and one-value-a-texel fragment
/// functions the probe `r.m` used.
const RENDER_FIXTURE: &str = "rt6_samples";

/// The vertex function every store case draws: the full-screen triangle computed from
/// `[[vertex_id]]`, so the pass binds no vertex buffer.
const RENDER_VERTEX: &str = "rt6_fill_vertex";

/// The vertices the one draw of the full-screen triangle takes.
const RENDER_VERTICES: u32 = 3;

/// The store case's target width, as the probe drew it.
const STORE_WIDTH: u32 = 4096;

/// The values the store cases sweep: the ramp, then the render specials.
const STORE_VALUES: usize = 16_385 + STORE_SPECIALS.len();

/// The store case's target height: enough rows for the sweep, so the draw covers 20,480 texels of
/// which 4,075 take the input buffer's zero padding.
const STORE_HEIGHT: u32 = STORE_VALUES.div_ceil(STORE_WIDTH as usize) as u32;

/// The store sweep's specials, in every channel (the probe `r.m`'s, which are the clears' too and
/// differ from the compute write's): the signed zero and one, a NaN, both infinities, values past
/// the range, a denormal, the ties either side of a code, two values inside the sRGB encode's
/// steps, `half`'s largest finite value and the first value that rounds past it, and four
/// repeating fractions.
const STORE_SPECIALS: [f32; 20] = [
    -0.0,
    -1.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1.5,
    2.0,
    1e-30,
    0.999_999_94,
    1.000_000_1,
    0.5 / 8192.0,
    1.5 / 8192.0,
    0.003_130_8,
    0.040_45,
    65504.0,
    65520.0,
    0.333_333_34,
    0.166_666_67,
    0.5,
    0.833_333_3,
];

/// `value` as `half`'s bits, rounding the significand to nearest with ties to even and anything
/// from 65520 up to an infinity, which is what a `float` the fragment function returns into a
/// `half4` output goes through.
///
/// This is `runtime_lane::rgb10a2_srgb`'s own function, and `the_half_bits_are_the_lane_s` pins it
/// against the values that lane's test states, because the suite may not depend on the validation
/// crate and a second spelling that drifted would make the `half` case's inputs quietly another
/// case's.
fn half_bits(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7e00;
    }
    let sign = ((value.to_bits() >> 16) & 0x8000) as u16;
    let size = f64::from(value.abs());
    if size >= 65520.0 {
        return sign | 0x7c00;
    }
    // The nearest multiple of the `half` ulp at `size`, ties to even, exactly in `f64`.
    let step = if size < 2f64.powi(-14) {
        -24
    } else {
        size.log2().floor() as i32 - 10
    };
    let quantized = (size / 2f64.powi(step)).round_ties_even() * 2f64.powi(step);
    let bits = if quantized < 2f64.powi(-14) {
        (quantized / 2f64.powi(-24)) as u16
    } else {
        let exponent = quantized.log2().floor() as i32;
        let significand = (quantized / 2f64.powi(exponent) - 1.0) * 1024.0;
        (((exponent + 15) as u16) << 10) | significand as u16
    };
    sign | bits
}

/// The probe's xorshift step.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// The read case's texture bytes: code `c`'s word, so every colour code occurs once in each
/// channel and the two padding bits take every value.
fn read_words() -> Vec<u8> {
    (0..READ_WIDTH)
        .map(|code| code | (1023 - code) << 10 | (7 * code % 1024) << 20 | (code % 4) << 30)
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// The write case's `float4`s: the ramp up to 16,384, then the specials in every channel, then
/// zeros for the rest of the grid.
fn write_values() -> Vec<u8> {
    let texels = (WRITE_WIDTH * WRITE_HEIGHT) as usize;
    (0..texels)
        .flat_map(|k| {
            if k < 16_385 {
                let value = k as f32 / 16384.0;
                [value, value, value, (k % 4096) as f32 / 4095.0]
            } else if k < WRITE_VALUES {
                [WRITE_SPECIALS[k - 16_385]; 4]
            } else {
                [0.0; 4]
            }
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The store case's values at texel `k`: the ramp up to 16,384, then the render specials in every
/// channel, then zeros for the rest of the target.
fn store_value(k: usize) -> [f32; 4] {
    if k < 16_385 {
        let value = k as f32 / 16384.0;
        [value, value, value, (k % 4096) as f32 / 4095.0]
    } else if k < STORE_VALUES {
        [STORE_SPECIALS[k - 16_385]; 4]
    } else {
        [0.0; 4]
    }
}

/// The store case's values buffer: one four-lane value a texel of the whole target, in the lanes
/// the fragment function reads them as.
fn store_values(lanes: Lanes) -> Vec<u8> {
    let texels = (STORE_WIDTH * STORE_HEIGHT) as usize;
    (0..texels)
        .flat_map(store_value)
        .flat_map(|value| match lanes {
            Lanes::Half => half_bits(value).to_le_bytes().to_vec(),
            Lanes::Float => value.to_le_bytes().to_vec(),
        })
        .collect()
}

/// The fixture's `Rt6Shape` the fragment function reads: the target's width and its samples, one.
fn store_shape() -> Vec<u8> {
    let mut bytes = STORE_WIDTH.to_le_bytes().to_vec();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes
}

/// The sampled texture's bytes and then the probe's 4,096 positions, drawn from one stream in that
/// order, as the probe drew them.
fn sampled_inputs() -> (Vec<u8>, Vec<u8>) {
    let mut state = SAMPLE_SEED;
    let texels: Vec<u8> = (0..(SAMPLE_SIDE * SAMPLE_SIDE))
        .flat_map(|_| ((xorshift(&mut state) >> 17) as u32).to_le_bytes())
        .collect();
    let coordinates: Vec<u8> = (0..SAMPLES)
        .flat_map(|_| {
            let drawn = xorshift(&mut state);
            let x = (drawn % 4608) as i64 - 256;
            let y = ((drawn >> 20) % 4608) as i64 - 256;
            [x as f32 / 4096.0, y as f32 / 4096.0]
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    (texels, coordinates)
}

/// What a result's lanes are, for the sample cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lanes {
    Half,
    Float,
}

impl Lanes {
    fn entry(self) -> &'static str {
        match self {
            Lanes::Half => "xr_sample_runtime_half",
            Lanes::Float => "xr_sample_runtime",
        }
    }

    /// The fixture's fragment function whose return value these lanes are: the one Metal's own
    /// output conversion then packs into the 586 attachment.
    fn fragment(self) -> &'static str {
        match self {
            Lanes::Half => "rt6_fill_rgba16float",
            Lanes::Float => "rt6_fill_rgba32float",
        }
    }

    fn element(self) -> Element {
        match self {
            Lanes::Half => Element::F16,
            Lanes::Float => Element::F32,
        }
    }

    /// Bytes one result's four lanes take.
    fn stride(self) -> u32 {
        4 * self.element().size()
    }
}

/// The layout a buffer of `rows` four-lane results is read under: one row a thread, so a diff names
/// the code or the sample and the lane.
fn lanes_layout(lanes: Lanes, rows: u32) -> ByteLayout {
    let stride = lanes.stride();
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(stride) * u64::from(rows),
        },
        element: lanes.element(),
        row_offset: 0,
        row_elements: 4,
        row_pitch: stride,
        rows,
        image_pitch: stride * rows,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The layout the written texture's words are read under: one row of the texture per row of the
/// layout, so a diff names the texel by its position.
fn texture_layout(width: u32, height: u32) -> ByteLayout {
    let pitch = width * WORD_BYTES;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [width, height, 1],
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: width,
        row_pitch: pitch,
        rows: height,
        image_pitch: pitch * height,
        images: 1,
        // The rows are packed at the format's own pitch, so the sites cover every byte.
        gaps: Gaps::Ignored,
    }
}

/// Which question a case asks.
enum Question {
    /// Every colour code read back by a kernel.
    Reads,
    /// The sweep written by a kernel, and the texture's words after it.
    Writes,
    /// The probe's positions sampled through a runtime sampler of `filter`, `lanes`-wise.
    Samples {
        filter: SamplerMinMagFilter,
        lanes: Lanes,
    },
    /// The sweep stored by a full-screen draw whose fragment function returns `lanes`.
    Stores { lanes: Lanes },
}

/// A case: one question over the format.
pub struct Rgb10A2Srgb {
    name: &'static str,
    question: Question,
}

/// Every case of the family.
pub fn all() -> Vec<Rgb10A2Srgb> {
    vec![
        Rgb10A2Srgb {
            name: "rgb10a2-srgb-reads",
            question: Question::Reads,
        },
        Rgb10A2Srgb {
            name: "rgb10a2-srgb-writes",
            question: Question::Writes,
        },
        Rgb10A2Srgb {
            name: "rgb10a2-srgb-samples-nearest-f",
            question: Question::Samples {
                filter: SamplerMinMagFilter::Nearest,
                lanes: Lanes::Float,
            },
        },
        Rgb10A2Srgb {
            name: "rgb10a2-srgb-samples-nearest-h",
            question: Question::Samples {
                filter: SamplerMinMagFilter::Nearest,
                lanes: Lanes::Half,
            },
        },
        Rgb10A2Srgb {
            name: "rgb10a2-srgb-samples-linear-f",
            question: Question::Samples {
                filter: SamplerMinMagFilter::Linear,
                lanes: Lanes::Float,
            },
        },
        Rgb10A2Srgb {
            name: "rgb10a2-srgb-samples-linear-h",
            question: Question::Samples {
                filter: SamplerMinMagFilter::Linear,
                lanes: Lanes::Half,
            },
        },
        Rgb10A2Srgb {
            name: "rgb10a2-srgb-stores-f",
            question: Question::Stores {
                lanes: Lanes::Float,
            },
        },
        Rgb10A2Srgb {
            name: "rgb10a2-srgb-stores-h",
            question: Question::Stores { lanes: Lanes::Half },
        },
    ]
}

impl Rgb10A2Srgb {
    /// The fixture and entry point the case's shader is.
    fn shader(&self) -> Fixture {
        match &self.question {
            Question::Reads => Fixture {
                name: TEXTURE_FIXTURE,
                entry: FETCH,
            },
            Question::Writes => Fixture {
                name: TEXTURE_FIXTURE,
                entry: WRITE,
            },
            Question::Samples { lanes, .. } => Fixture {
                name: SAMPLE_FIXTURE,
                entry: lanes.entry(),
            },
            Question::Stores { lanes } => Fixture {
                name: RENDER_FIXTURE,
                entry: lanes.fragment(),
            },
        }
    }

    /// Run one kernel and wait for it, answering the command buffer's handle.
    fn dispatch(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<(), Abort> {
        let command_buffer = made(
            sites,
            &format!("{at}/dispatch"),
            side.dispatch_threads(pipeline, bindings, &[], grid, [GROUP, 1, 1]),
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
        Ok(())
    }

    /// The read case: every colour code read back as a `float4`.
    fn run_reads(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        pipeline: Handle,
    ) -> Result<Node, Abort> {
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(
                &TextureSpec::new(
                    FORMAT,
                    [READ_WIDTH, 1],
                    READ_WIDTH * WORD_BYTES,
                    TextureRole::Read,
                ),
                &read_words(),
            ),
        )?;
        let length = READ_WIDTH as usize * Lanes::Float.stride() as usize;
        let output = made(sites, "buffer/out", side.buffer(&vec![BLANK; length]))?;
        let width = made(
            sites,
            "buffer/width",
            side.buffer(&READ_WIDTH.to_le_bytes()),
        )?;
        let textures = [texture];
        let buffers = [output, width];
        self.dispatch(
            side,
            sites,
            "read",
            pipeline,
            &Bindings::new(&textures, &buffers),
            [READ_WIDTH, 1, 1],
        )?;
        let bytes = made(sites, "read/read-back", side.read_buffer(output, 0, length))?;
        let digest = Digest::sha256(&bytes);
        Ok(Node::branch("output", Vec::new()).with(Node::leaf(
            "codes",
            Observed::bytes(bytes, lanes_layout(Lanes::Float, READ_WIDTH)).digested(digest),
        )))
    }

    /// The write case: the sweep written into a 4096 by 5 texture, whose words are read back whole.
    fn run_writes(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        pipeline: Handle,
    ) -> Result<Node, Abort> {
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(
                &TextureSpec::new(
                    FORMAT,
                    [WRITE_WIDTH, WRITE_HEIGHT],
                    WRITE_WIDTH * WORD_BYTES,
                    TextureRole::Write,
                ),
                &[],
            ),
        )?;
        let values = made(sites, "buffer/values", side.buffer(&write_values()))?;
        let width = made(
            sites,
            "buffer/width",
            side.buffer(&WRITE_WIDTH.to_le_bytes()),
        )?;
        let textures = [texture];
        let buffers = [values, width];
        self.dispatch(
            side,
            sites,
            "write",
            pipeline,
            &Bindings::new(&textures, &buffers),
            [WRITE_WIDTH, WRITE_HEIGHT, 1],
        )?;
        let bytes = made(
            sites,
            "write/read-back",
            side.read_texture(
                texture,
                [WRITE_WIDTH, WRITE_HEIGHT],
                WRITE_WIDTH * WORD_BYTES,
            ),
        )?;
        let digest = Digest::sha256(&bytes);
        Ok(Node::branch("output", Vec::new()).with(Node::leaf(
            "words",
            Observed::bytes(bytes, texture_layout(WRITE_WIDTH, WRITE_HEIGHT)).digested(digest),
        )))
    }

    /// A sample case: the probe's positions through a runtime sampler of `filter`.
    fn run_samples(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        pipeline: Handle,
        filter: SamplerMinMagFilter,
        lanes: Lanes,
    ) -> Result<Node, Abort> {
        let (texels, coordinates) = sampled_inputs();
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(
                &TextureSpec::new(
                    FORMAT,
                    [SAMPLE_SIDE, SAMPLE_SIDE],
                    SAMPLE_SIDE * WORD_BYTES,
                    TextureRole::Read,
                ),
                &texels,
            ),
        )?;
        let coords = made(sites, "buffer/coordinates", side.buffer(&coordinates))?;
        let length = SAMPLES * lanes.stride() as usize;
        let output = made(sites, "buffer/out", side.buffer(&vec![BLANK; length]))?;
        let sampler = made(
            sites,
            "sampler",
            side.sampler(&SamplerSpec::filtered(filter)),
        )?;
        let textures = [texture];
        let buffers = [coords, output];
        let samplers = [sampler];
        self.dispatch(
            side,
            sites,
            "sample",
            pipeline,
            &Bindings::new(&textures, &buffers).sampled(&samplers),
            [SAMPLES as u32, 1, 1],
        )?;
        let bytes = made(
            sites,
            "sample/read-back",
            side.read_buffer(output, 0, length),
        )?;
        let digest = Digest::sha256(&bytes);
        Ok(Node::branch("output", Vec::new()).with(Node::leaf(
            "samples",
            Observed::bytes(bytes, lanes_layout(lanes, SAMPLES as u32)).digested(digest),
        )))
    }

    /// A store case: the sweep drawn into the whole 4096 by 5 target, whose words are read back
    /// whole through a blit.
    fn run_stores(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        library: Handle,
        lanes: Lanes,
    ) -> Result<Node, Abort> {
        let values = made(sites, "buffer/values", side.buffer(&store_values(lanes)))?;
        let shape = made(sites, "buffer/shape", side.buffer(&store_shape()))?;
        let buffers = [values, shape];
        let filled = call(
            sites,
            "store/pass",
            side.fragment_fill(&FragmentFill {
                library,
                vertex: RENDER_VERTEX,
                fragment: lanes.fragment(),
                format: FORMAT,
                size: [STORE_WIDTH, STORE_HEIGHT],
                row_pitch: STORE_WIDTH * WORD_BYTES,
                buffers: &buffers,
                vertices: RENDER_VERTICES,
            }),
            |filled| filled.status.clone(),
        )?;
        if filled.status != Value::unit_variant("Completed") {
            return Err(Abort {
                at: "store/pass".to_owned(),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&filled.status)),
            });
        }
        let digest = Digest::sha256(&filled.bytes);
        Ok(Node::branch("output", Vec::new()).with(Node::leaf(
            "words",
            Observed::bytes(filled.bytes, texture_layout(STORE_WIDTH, STORE_HEIGHT))
                .digested(digest),
        )))
    }
}

impl Case for Rgb10A2Srgb {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let shader = self.shader();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", shader.name),
                    entry: shader.entry.to_owned(),
                    source_sha256: crate::sha256_hex(shader.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    /// What a store case's pass announced on MoltenVK, measured: a pass that loads nothing into
    /// the tile, and the `half` fragment's output conversion. Nothing else announced anything, so
    /// no other case here names one.
    fn allowances(&self) -> Vec<Allowance> {
        use metal2vulkan_runtime::Degradation as D;
        // Every question reads its answer back through a `Shared` buffer, and a device with no
        // unified memory keeps a `Shared` allocation in system memory because that is the only
        // memory both sides can reach: the venue's memory model rather than this format's answer.
        let mut named = vec![crate::replay::shared_memory_notice()];
        match &self.question {
            Question::Stores {
                lanes: Lanes::Float,
            } => named.push(notice_allowance(D::DontCareLoadTiles.code())),
            Question::Stores { lanes: Lanes::Half } => named.extend([
                notice_allowance(D::DontCareLoadTiles.code()),
                notice_allowance(D::HalfOutputFastMathBlend.code()),
            ]),
            _ => {}
        }
        named
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let shader = self.shader();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(shader.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(shader.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("format", &FORMAT.as_raw().to_le_bytes()),
            Input::bytes("buffer:blank", &[BLANK]),
        ];
        match &self.question {
            Question::Reads => inputs.push(Input::bytes("texture:words", &read_words())),
            Question::Writes => inputs.push(Input::bytes("buffer:values", &write_values())),
            Question::Samples { filter, .. } => {
                let (texels, coordinates) = sampled_inputs();
                inputs.push(Input::bytes("texture:words", &texels));
                inputs.push(Input::bytes("coordinates", &coordinates));
                inputs.push(Input::bytes(
                    "sampler",
                    &SamplerSpec::filtered(*filter).words(),
                ));
            }
            Question::Stores { lanes } => {
                inputs.push(Input::bytes("buffer:values", &store_values(*lanes)));
                inputs.push(Input::bytes("buffer:shape", &store_shape()));
                // Both functions of the pipeline, and the draw, so a case that drew another
                // triangle or another fragment function is not this one.
                inputs.push(Input::bytes(
                    "pipeline:functions",
                    format!(
                        "{RENDER_VERTEX}/{} {RENDER_VERTICES} vertices",
                        lanes.fragment()
                    )
                    .as_bytes(),
                ));
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let shader = self.shader();
        let loaded = shader.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        // A store case's pipeline is the render one `fragment_fill` makes of the same library; the
        // three compute questions make theirs here.
        let pipeline = |side: &mut dyn Side, sites: &mut Sites| {
            made(
                sites,
                "pipeline",
                side.compute_pipeline(library, shader.entry),
            )
        };
        let output = match &self.question {
            Question::Reads => {
                let made = pipeline(side, sites)?;
                self.run_reads(side, sites, made)?
            }
            Question::Writes => {
                let made = pipeline(side, sites)?;
                self.run_writes(side, sites, made)?
            }
            Question::Samples { filter, lanes } => {
                let made = pipeline(side, sites)?;
                self.run_samples(side, sites, made, *filter, *lanes)?
            }
            Question::Stores { lanes } => self.run_stores(side, sites, library, *lanes)?,
        };
        sites.push(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every colour code occurs once in each channel of the read case's texture, and the two
    /// padding bits take each of their four values, so the sweep is exhaustive over the codes.
    #[test]
    fn the_read_texture_sweeps_every_code_and_padding() {
        let bytes = read_words();
        assert_eq!(bytes.len(), READ_WIDTH as usize * 4);
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        for shift in [0, 10, 20] {
            let channel: BTreeSet<u32> = words.iter().map(|word| (word >> shift) & 1023).collect();
            assert_eq!(channel.len(), 1024, "the channel at bit {shift}");
        }
        let padding: BTreeSet<u32> = words.iter().map(|word| word >> 30).collect();
        assert_eq!(padding.len(), 4);
    }

    /// The write case's grid is larger than its sweep, which is the defect the port closed: the
    /// kernel writes 20,480 texels and the lane's formatter looked at 16,401 of them.
    #[test]
    fn the_write_grid_covers_more_texels_than_the_sweep() {
        let texels = (WRITE_WIDTH * WRITE_HEIGHT) as usize;
        assert_eq!(WRITE_HEIGHT, 5);
        assert_eq!(texels, 20_480);
        assert_eq!(WRITE_VALUES, 16_401);
        assert_eq!(texels - WRITE_VALUES, 4_079);
        // The input buffer covers the whole grid, so every one of those texels is written.
        assert_eq!(write_values().len(), texels * 16);
        // The site's layout covers all of them, as the lane's digest now does.
        assert_eq!(
            texture_layout(WRITE_WIDTH, WRITE_HEIGHT).extent(),
            texels * WORD_BYTES as usize
        );
    }

    /// The sampled texels and the positions come off one stream in the probe's order, so the
    /// texture and the coordinates cannot drift apart from the probe's.
    #[test]
    fn the_sampled_inputs_come_off_one_stream_in_order() {
        let (texels, coordinates) = sampled_inputs();
        assert_eq!(texels.len(), (SAMPLE_SIDE * SAMPLE_SIDE) as usize * 4);
        assert_eq!(coordinates.len(), SAMPLES * 8);
        // Drawing the texels alone leaves the stream where the positions begin.
        let mut state = SAMPLE_SEED;
        for _ in 0..(SAMPLE_SIDE * SAMPLE_SIDE) {
            xorshift(&mut state);
        }
        let drawn = xorshift(&mut state);
        let first = f32::from_le_bytes([
            coordinates[0],
            coordinates[1],
            coordinates[2],
            coordinates[3],
        ]);
        assert_eq!(first, ((drawn % 4608) as i64 - 256) as f32 / 4096.0);
    }

    /// Some of the probe's positions lie outside the texture on each side, which is what makes the
    /// address mode part of the answer.
    #[test]
    fn the_positions_reach_outside_the_texture() {
        let (_, coordinates) = sampled_inputs();
        let axes: Vec<f32> = coordinates
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        assert!(axes.iter().any(|axis| *axis < 0.0), "below the texture");
        assert!(axes.iter().any(|axis| *axis > 1.0), "past the texture");
    }

    /// Eight cases, each a different question.
    #[test]
    fn the_cases_are_distinct() {
        let names: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        assert_eq!(names.len(), 8);
    }

    /// The store cases' target is larger than their sweep, which is the second hole the port
    /// closes: the draw stores 20,480 texels and the lane's formatter looked at 16,405 of them.
    #[test]
    fn the_store_target_covers_more_texels_than_the_sweep() {
        let texels = (STORE_WIDTH * STORE_HEIGHT) as usize;
        assert_eq!(STORE_HEIGHT, 5);
        assert_eq!(texels, 20_480);
        assert_eq!(STORE_VALUES, 16_405);
        assert_eq!(texels - STORE_VALUES, 4_075);
        // The values buffer covers every texel of the target, so every one of them is drawn.
        assert_eq!(store_values(Lanes::Float).len(), texels * 16);
        assert_eq!(store_values(Lanes::Half).len(), texels * 8);
        // The site's layout covers all of them, where the lane's digest covered 16,405.
        assert_eq!(
            texture_layout(STORE_WIDTH, STORE_HEIGHT).extent(),
            texels * WORD_BYTES as usize
        );
    }

    /// The store sweep's specials are the probe `r.m`'s twenty, not the compute write's sixteen:
    /// the two questions sweep different values, and a case that took the other list would record
    /// another measurement under this one's name.
    #[test]
    fn the_store_specials_are_the_render_probe_s() {
        assert_eq!(STORE_SPECIALS.len(), 20);
        assert_eq!(WRITE_SPECIALS.len(), 16);
        // `half`'s largest finite value and the first that rounds to an infinity are the store
        // sweep's own, and neither is in the write sweep.
        assert!(STORE_SPECIALS.contains(&65504.0));
        assert!(STORE_SPECIALS.contains(&65520.0));
        assert!(!WRITE_SPECIALS.contains(&65504.0));
        // The write sweep's second NaN is not the store sweep's: `r.m` drew one.
        assert_eq!(STORE_SPECIALS.iter().filter(|v| v.is_nan()).count(), 1);
        assert_eq!(WRITE_SPECIALS.iter().filter(|v| v.is_nan()).count(), 2);
    }

    /// `half_bits` is the lane's, pinned by the values `runtime_lane::rgb10a2_srgb`'s own test
    /// states: the `half` store case's inputs are that lane's inputs and not merely similar ones.
    #[test]
    fn the_half_bits_are_the_lane_s() {
        assert_eq!(half_bits(1.0), 0x3c00);
        assert_eq!(half_bits(-2.0), 0xc000);
        assert_eq!(half_bits(65504.0), 0x7bff);
        assert_eq!(half_bits(65520.0), 0x7c00);
        assert_eq!(half_bits(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(half_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(half_bits(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(half_bits(2f32.powi(-24)), 0x0001);
        assert_eq!(half_bits(2f32.powi(-25)), 0x0000);
        assert_eq!(half_bits(3.0 * 2f32.powi(-25)), 0x0002);
        assert_eq!(half_bits(0.333_333_34), 0x3555);
        assert_eq!(half_bits(1e-30), 0);
        assert_eq!(half_bits(f32::NAN), 0x7e00);
        assert_eq!(half_bits(-0.0), 0x8000);
    }

    /// The tail past the sweep takes the values buffer's zeros in both lane widths, so what the
    /// 4,075 words hold is a draw of `(0, 0, 0, 0)` and not of whatever was left in the buffer.
    #[test]
    fn the_store_tail_draws_zeros() {
        for lanes in [Lanes::Float, Lanes::Half] {
            let bytes = store_values(lanes);
            let stride = 4 * lanes.element().size() as usize;
            assert!(
                bytes[STORE_VALUES * stride..].iter().all(|byte| *byte == 0),
                "the tail of the {lanes:?} values"
            );
            // And the last value of the sweep itself is not zero, so the test is not vacuous.
            assert!(bytes[(STORE_VALUES - 1) * stride..STORE_VALUES * stride]
                .iter()
                .any(|byte| *byte != 0));
        }
    }
}
