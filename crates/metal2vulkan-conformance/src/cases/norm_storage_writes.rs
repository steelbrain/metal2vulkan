//! What a kernel stores when it writes a `float` to a **normalized** storage texture, ported from
//! `runtime_lane::norm_storage_writes`.
//!
//! Metal stores such a write as a fragment function's write to a color attachment of the format: a
//! unorm channel of `b` bits takes `x` clamped to `[0, 1]` (a NaN 0) times `2^b - 1` **exactly**,
//! rounded to nearest with a tie **up**; a snorm channel the same on `[-1, 1]` with a tie **away
//! from zero**, stored two's complement. A 2-bit channel is a unorm channel like any other. The
//! lane's one row for this is an Apple M4 under macOS 27.0.1, and that machine is gone, so the port
//! re-measures it here: record mode runs the case and the recordings hold Metal's own bytes, with
//! the rule above as a committed cross-check over them
//! (`every_recorded_code_is_metals_rule`).
//!
//! The 10:10:10:2 formats also say **where** each channel lands. Metal orders `RGB10A2Unorm`'s
//! fields red first and `BGR10A2Unorm`'s blue first, which is a fact about the format rather than
//! about any device, and a texel whose four channels are four different codes is what tells the two
//! apart (`every_recorded_field_holds_its_channels_code`).
//!
//! Each case writes through all four of the owned fixture `hsw_storage_writes`' routes to a
//! texture, because a format's conversion is the same fact however the kernel reaches it and a
//! route that converted differently would be the interesting answer: `hsw_write`
//! (`access::write`), `hsw_read_write` (`access::read_write`), `hsw_buffer` (a `texture_buffer`)
//! and `hsw_held` (a `texture2d` inside an argument buffer, bound by its resource ID). Every
//! kernel takes the values at buffer 0; the two-dimensional three take the row's width at buffer 1,
//! and `hsw_held` the argument buffer at buffer 2.
//!
//! **The sweep is this port's, not the lane's.** The lane runs every code and every tie of each
//! format it touches, which for a 16-bit channel is 65,536 codes and 65,535 ties — hundreds of
//! thousands of values, four routes each. The lane had no recording to keep and this family has a
//! 512 KiB ceiling a case, so the sweep is stated here and pinned by
//! `the_sweeps_cover_what_they_claim`:
//!
//! - a channel of **at most 10 bits** is swept **whole**: every code's exact value, and every tie
//!   between two codes with two `f32` steps either side of it, which is where a tie-up rule and a
//!   tie-down rule differ;
//! - a **16-bit** channel takes the lowest 128 codes, the highest 128 and every 512th code
//!   between them, each with the tie above it and the same two steps either side. Both ends are
//!   whole because that is where the product's spacing is furthest from `f32`'s, and the stride
//!   covers the middle at one code in 512;
//! - every case also writes [`SPECIALS`], which is what the rule's clamp and NaN are about.
//!
//! Nothing here can fault a GPU: one compute pass a route, no loop, no sampling, and every
//! dispatch a whole number of threadgroups.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, ArgumentSlot, Bindings, Case, Failure, Handle, Input, MetalSource, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::TextureRequest;
use crate::transcript::{ByteLayout, Element, Node, Observed, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The fixture every kernel lives in.
const FIXTURE: &str = "hsw_storage_writes";

/// The texels of a row of the two-dimensional routes' texture, and the threads of a threadgroup
/// along each axis: 64 by 8 threads a group over a 64-wide row, so every dispatch of this family
/// is a whole number of threadgroups and no route announces a split.
const WIDTH: u32 = 64;

/// The threads of a threadgroup, `WIDTH` by this many rows.
const ROWS_PER_GROUP: u32 = 8;

/// The texels of a texture-buffer dispatch, and its threadgroup's threads.
const BUFFER_GROUP: u32 = 64;

/// The values every case writes beside its codes: what the rule's clamp, its NaN and its signed
/// zero are about.
pub const SPECIALS: [f32; 20] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.0,
    -2.0,
    1e-45,
    -1e-45,
    1.175_494_2e-38,
    -1.175_494_2e-38,
    1e-10,
    -1e-10,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    -f32::NAN,
];

/// A channel of a normalized format: the kind, and its width in bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// `x` clamped to `[0, 1]` times `2^bits - 1`, a tie up.
    Unorm(u32),
    /// `x` clamped to `[-1, 1]` times `2^(bits-1) - 1`, a tie away from zero, two's complement.
    Snorm(u32),
}

impl Channel {
    pub const fn bits(self) -> u32 {
        match self {
            Channel::Unorm(bits) | Channel::Snorm(bits) => bits,
        }
    }

    /// Metal's stored code of `x` in this channel: the module docs' rule.
    ///
    /// The product is taken in `f64`, where it is exact for every width here, so the tie is the
    /// rule's tie and not a rounding of one.
    pub fn code(self, x: f32) -> u64 {
        let mask = (1u64 << self.bits()) - 1;
        match self {
            Channel::Unorm(bits) => {
                let unit = match x.is_nan() {
                    true => 0.0,
                    false => f64::from(x).clamp(0.0, 1.0),
                };
                let max = ((1u64 << bits) - 1) as f64;
                ((unit * max) + 0.5).floor() as u64 & mask
            }
            Channel::Snorm(bits) => {
                let unit = match x.is_nan() {
                    true => 0.0,
                    false => f64::from(x).clamp(-1.0, 1.0),
                };
                let max = ((1i64 << (bits - 1)) - 1) as f64;
                let scaled = unit * max;
                let rounded = match scaled >= 0.0 {
                    true => (scaled + 0.5).floor(),
                    false => (scaled - 0.5).ceil(),
                };
                (rounded as i64) as u64 & mask
            }
        }
    }
}

/// The `bits` of `texel` starting at `offset`, little-endian over the texel's bytes.
pub fn field(texel: &[u8], offset: u32, bits: u32) -> u64 {
    let mut word = 0u64;
    for (at, byte) in texel.iter().enumerate().take(8) {
        word |= u64::from(*byte) << (8 * at);
    }
    (word >> offset) & ((1u64 << bits) - 1)
}

/// `x` and the two `f32` steps either side of it.
fn around(x: f32) -> [f32; 5] {
    let up = |x: f32| f32::from_bits(x.to_bits() + 1);
    let down = |x: f32| f32::from_bits(x.to_bits() - 1);
    [down(down(x)), down(x), x, up(x), up(up(x))]
}

/// Every code's exact value of a unorm channel of `bits`, then every tie with [`around`].
fn unorm_sweep(bits: u32) -> Vec<f32> {
    let max = ((1u64 << bits) - 1) as f64;
    let codes = 1u64 << bits;
    let mut out: Vec<f32> = (0..codes).map(|k| (k as f64 / max) as f32).collect();
    for k in 0..codes - 1 {
        out.extend(around(((k as f64 + 0.5) / max) as f32));
    }
    out
}

/// Every code's exact value of a snorm channel of `bits`, then every tie with [`around`].
fn snorm_sweep(bits: u32) -> Vec<f32> {
    let max = (1i64 << (bits - 1)) - 1;
    let mut out: Vec<f32> = (-max..=max)
        .map(|k| (k as f64 / max as f64) as f32)
        .collect();
    for k in -max..max {
        out.extend(around(((k as f64 + 0.5) / max as f64) as f32));
    }
    out
}

/// The codes a 16-bit channel is swept at: the lowest and highest [`ENDS`], and every
/// [`STRIDE`]th code between them.
pub const ENDS: u64 = 128;

/// One code in this many across a 16-bit channel's middle.
pub const STRIDE: u64 = 512;

/// The codes `sixteen_bit` sweeps, in order and without repeats.
pub fn sixteen_bit_codes() -> Vec<u64> {
    let last = (1u64 << 16) - 1;
    let mut codes: Vec<u64> = (0..ENDS).collect();
    codes.extend((ENDS..=last - ENDS).filter(|code| code.is_multiple_of(STRIDE)));
    codes.extend(last - ENDS + 1..=last);
    codes
}

/// A 16-bit channel's stated selection: [`sixteen_bit_codes`]' exact values, and the tie above
/// each with [`around`].
fn sixteen_bit(signed: bool) -> Vec<f32> {
    let max = match signed {
        true => ((1u64 << 15) - 1) as f64,
        false => ((1u64 << 16) - 1) as f64,
    };
    let mut out = Vec::new();
    for code in sixteen_bit_codes() {
        let value = match signed {
            // A snorm code's value is read off the signed range, so the selection's codes stand
            // for magnitudes and both signs are swept.
            true => (code.min(1 << 15) as f64 / max) as f32,
            false => (code as f64 / max) as f32,
        };
        out.push(value);
        if signed {
            out.push(-value);
        }
        out.extend(around(match signed {
            true => ((code.min(1 << 15) as f64 + 0.5) / max) as f32,
            false => ((code as f64 + 0.5) / max) as f32,
        }));
    }
    out
}

/// How a kernel reaches the texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// `texture2d<float, access::write>` at `[[texture(0)]]`.
    Write,
    /// `texture2d<float, access::read_write>` at `[[texture(0)]]`.
    ReadWrite,
    /// `texture_buffer<float, access::write>` at `[[texture(0)]]`.
    Buffer,
    /// A `texture2d<float, access::write>` inside an argument buffer at `[[buffer(2)]]`.
    Held,
}

impl Route {
    pub const ALL: [Route; 4] = [Route::Write, Route::ReadWrite, Route::Buffer, Route::Held];

    pub const fn kernel(self) -> &'static str {
        match self {
            Route::Write => "hsw_write",
            Route::ReadWrite => "hsw_read_write",
            Route::Buffer => "hsw_buffer",
            Route::Held => "hsw_held",
        }
    }
}

/// What a case writes: one value in every channel of a texel, or four codes' values in the four.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Writes {
    /// The sweep, each value in all four lanes of its texel.
    Codes,
    /// 1,024 texels whose red, green, blue and alpha are four different codes' values.
    Fields,
}

/// One case: a format, where its channels live, and what the case writes into it.
pub struct Format {
    pub case: &'static str,
    pub pf: PixelFormat,
    /// The bytes of a texel.
    pub bytes: usize,
    /// Each channel's first bit and kind, in Metal's channel order (red, green, blue, alpha).
    pub channels: &'static [(u32, Channel)],
    pub writes: Writes,
    values: fn() -> Vec<[f32; 4]>,
}

/// The sweep's values, each in all four lanes of a texel.
fn same(values: Vec<f32>) -> Vec<[f32; 4]> {
    values.into_iter().map(|x| [x; 4]).collect()
}

/// 1,024 texels whose four channels are four different codes of a 10-bit (and, for alpha, 2-bit)
/// channel, so a field holding another channel's code is a difference and not a coincidence.
pub fn distinct_texels() -> Vec<[f32; 4]> {
    (0..1024u32)
        .map(|i| {
            let code = |k: u32| ((i * k + k) % 1024) as f64 / 1023.0;
            [
                code(7) as f32,
                code(13) as f32,
                code(29) as f32,
                (i % 4) as f32 / 3.0,
            ]
        })
        .collect()
}

const U2: Channel = Channel::Unorm(2);
const U8: Channel = Channel::Unorm(8);
const U10: Channel = Channel::Unorm(10);
const U16: Channel = Channel::Unorm(16);
const S8: Channel = Channel::Snorm(8);
const S16: Channel = Channel::Snorm(16);

/// Metal's field order of the two 10:10:10:2 formats: `RGB10A2Unorm` red first, `BGR10A2Unorm`
/// blue first, alpha at bit 30 in both.
const RGB10A2: &[(u32, Channel)] = &[(0, U10), (10, U10), (20, U10), (30, U2)];
const BGR10A2: &[(u32, Channel)] = &[(20, U10), (10, U10), (0, U10), (30, U2)];

pub const FORMATS: [Format; 7] = [
    Format {
        case: "norm-storage-codes-rgba8unorm",
        pf: PixelFormat::RGBA8Unorm,
        bytes: 4,
        channels: &[(0, U8), (8, U8), (16, U8), (24, U8)],
        writes: Writes::Codes,
        values: || same([unorm_sweep(8), SPECIALS.to_vec()].concat()),
    },
    Format {
        case: "norm-storage-codes-rgba8snorm",
        pf: PixelFormat::RGBA8Snorm,
        bytes: 4,
        channels: &[(0, S8), (8, S8), (16, S8), (24, S8)],
        writes: Writes::Codes,
        values: || same([snorm_sweep(8), SPECIALS.to_vec()].concat()),
    },
    Format {
        case: "norm-storage-codes-rgb10a2unorm",
        pf: PixelFormat::RGB10A2Unorm,
        bytes: 4,
        channels: RGB10A2,
        writes: Writes::Codes,
        values: || same([unorm_sweep(10), SPECIALS.to_vec()].concat()),
    },
    Format {
        case: "norm-storage-codes-rgba16unorm",
        pf: PixelFormat::RGBA16Unorm,
        bytes: 8,
        channels: &[(0, U16), (16, U16), (32, U16), (48, U16)],
        writes: Writes::Codes,
        values: || same([sixteen_bit(false), SPECIALS.to_vec()].concat()),
    },
    Format {
        case: "norm-storage-codes-rgba16snorm",
        pf: PixelFormat::RGBA16Snorm,
        bytes: 8,
        channels: &[(0, S16), (16, S16), (32, S16), (48, S16)],
        writes: Writes::Codes,
        values: || same([sixteen_bit(true), SPECIALS.to_vec()].concat()),
    },
    Format {
        case: "norm-storage-fields-rgb10a2unorm",
        pf: PixelFormat::RGB10A2Unorm,
        bytes: 4,
        channels: RGB10A2,
        writes: Writes::Fields,
        values: distinct_texels,
    },
    Format {
        case: "norm-storage-fields-bgr10a2unorm",
        pf: PixelFormat::BGR10A2Unorm,
        bytes: 4,
        channels: BGR10A2,
        writes: Writes::Fields,
        values: distinct_texels,
    },
];

impl Format {
    /// The texels the case writes.
    pub fn texels(&self) -> Vec<[f32; 4]> {
        (self.values)()
    }

    /// The routes this format states.
    ///
    /// Every format but one states all four. Metal's own API validation refuses a texture buffer
    /// of `BGR10A2Unorm` -- "pixel format (MTLPixelFormatBGR10A2Unorm) cannot be used for a
    /// texture buffer" -- while plain Metal makes one and stores the right bytes through it. A
    /// case of this family is evidence about a conversion, so it asks Metal for nothing Metal
    /// says is not a texture buffer; the other three routes carry the format's field order, and
    /// what Metal refuses is recorded here rather than built on.
    pub fn routes(&self) -> Vec<Route> {
        Route::ALL
            .into_iter()
            .filter(|route| *route != Route::Buffer || self.pf != PixelFormat::BGR10A2Unorm)
            .collect()
    }

    /// The rows of the two-dimensional routes' texture: enough for every texel, in whole
    /// threadgroups.
    pub fn rows(&self) -> u32 {
        let texels = self.texels().len() as u32;
        texels
            .div_ceil(WIDTH)
            .div_ceil(ROWS_PER_GROUP)
            .max(1)
            .saturating_mul(ROWS_PER_GROUP)
    }

    /// The texels of the texture-buffer route, in whole threadgroups.
    pub fn buffer_texels(&self) -> u32 {
        (self.texels().len() as u32)
            .div_ceil(BUFFER_GROUP)
            .max(1)
            .saturating_mul(BUFFER_GROUP)
    }

    pub fn row_pitch(&self) -> u32 {
        WIDTH * self.bytes as u32
    }

    /// The bytes of the texels the case compares: the sweep's, not the padding's.
    pub fn stored_bytes(&self) -> usize {
        self.texels().len() * self.bytes
    }

    /// The layout the recording states for the stored texels.
    ///
    /// A texel of this family is a packed field word, so the element is the word's width and the
    /// comparison names the texel that moved rather than a byte of one. Every route's read-back
    /// is the same run of texels, whichever resource it came out of, so the layout is the bytes
    /// as read rather than the texture's rows.
    pub fn stored_layout(&self) -> ByteLayout {
        let element = match self.bytes {
            4 => Element::U32,
            _ => Element::U64,
        };
        ByteLayout::buffer(0, self.stored_bytes() as u64, element)
    }

    /// The `float4`s the kernels read, padded to `texels` with zeros.
    fn values_bytes(&self, texels: usize) -> Vec<u8> {
        let written = self.texels();
        (0..texels)
            .flat_map(|at| written.get(at).copied().unwrap_or([0.0; 4]))
            .flat_map(f32::to_le_bytes)
            .collect()
    }
}

pub struct NormStorageWrites {
    at: usize,
}

impl NormStorageWrites {
    pub fn all() -> Vec<Box<dyn Case>> {
        (0..FORMATS.len())
            .map(|at| Box::new(Self { at }) as Box<dyn Case>)
            .collect()
    }

    pub fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }
}

impl Case for NormStorageWrites {
    fn name(&self) -> &'static str {
        self.format().case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration and no tolerance: the buffer each dispatch reads its words back through is `Shared`, which is
        // what [`crate::replay::shared_memory_notice`] is for. It loosens the notices site alone,
        // so every value this case reads still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for route in Route::ALL {
            // Every case's shaders are the fixture's four, whichever routes it runs: the library
            // a case loads is one text, and a route it does not run is still in it.
            let fixture = Fixture {
                name: FIXTURE,
                entry: route.kernel(),
            };
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: route.kernel().to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let format = self.format();
        let fixture = Fixture::new(FIXTURE);
        let mut inputs = vec![Input::bytes(
            "library",
            fixture.text(RUNTIME_TEXT)?.as_bytes(),
        )];
        // The values every route writes, and the geometry each route writes them through: either
        // moving makes the recording's answer another program's.
        inputs.push(Input::bytes(
            "values",
            &format.values_bytes(format.texels().len()),
        ));
        let mut geometry = Vec::new();
        geometry.extend(format.pf.as_raw().to_le_bytes());
        geometry.extend((format.bytes as u32).to_le_bytes());
        geometry.extend(WIDTH.to_le_bytes());
        geometry.extend(format.rows().to_le_bytes());
        geometry.extend(format.buffer_texels().to_le_bytes());
        for route in format.routes() {
            geometry.extend(route.kernel().as_bytes());
        }
        inputs.push(Input::bytes("geometry", &geometry));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let format = self.format();
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let width = made(sites, "buffer/width", side.buffer(&WIDTH.to_le_bytes()))?;
        for route in format.routes() {
            let at = |site: &str| format!("{}/{site}", route.kernel());
            let pipeline = made(
                sites,
                &at("pipeline"),
                side.compute_pipeline(library, route.kernel()),
            )?;
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let (texture, backing, texels, grid, group) = match route {
                Route::Buffer => {
                    let texels = format.buffer_texels();
                    let row = u64::from(texels) * format.bytes as u64;
                    let backing = made(
                        sites,
                        &at("buffer/backing"),
                        side.zeroed_buffer(row as usize),
                    )?;
                    let request =
                        TextureRequest::texel_buffer(format.pf.as_raw(), texels, backing, 0, row)
                            .used(usage);
                    let texture = made(sites, &at("texture"), side.texture(&request))?;
                    (
                        texture,
                        Some(backing),
                        texels as usize,
                        [texels, 1, 1],
                        [BUFFER_GROUP, 1, 1],
                    )
                }
                _ => {
                    let rows = format.rows();
                    let request =
                        TextureRequest::read_2d(format.pf.as_raw(), WIDTH, rows).used(usage);
                    let texture = made(sites, &at("texture"), side.texture(&request))?;
                    (
                        texture,
                        None,
                        (WIDTH * rows) as usize,
                        [WIDTH, rows, 1],
                        [WIDTH, ROWS_PER_GROUP, 1],
                    )
                }
            };
            let values = made(
                sites,
                &at("buffer/values"),
                side.buffer(&format.values_bytes(texels)),
            )?;
            let held = match route {
                Route::Held => Some(made(
                    sites,
                    &at("argument-buffer"),
                    side.argument_buffer(&[ArgumentSlot::Texture(texture)]),
                )?),
                _ => None,
            };
            // The held route reaches the texture through the argument buffer alone, so it binds
            // no texture at `[[texture(0)]]`: a texture bound there as well is a binding the
            // pipeline names nowhere, which Metal's own validation reports.
            let bound: Vec<Handle> = match route {
                Route::Held => Vec::new(),
                _ => vec![texture],
            };
            let buffers = match (route, held) {
                (Route::Buffer, _) => vec![values],
                (_, Some(held)) => vec![values, width, held],
                _ => vec![values, width],
            };
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(pipeline, &Bindings::new(&bound, &buffers), &[], grid, group),
            )?;
            call(
                sites,
                &at("status"),
                side.wait(command_buffer),
                Clone::clone,
            )?;
            let stored = match backing {
                Some(backing) => call(
                    sites,
                    &at("read-back"),
                    side.read_buffer(backing, 0, format.stored_bytes()),
                    |_| Value::Absent,
                )?,
                None => {
                    let whole = call(
                        sites,
                        &at("read-back"),
                        side.read_texture(texture, [WIDTH, format.rows()], format.row_pitch()),
                        |_| Value::Absent,
                    )?;
                    whole[..format.stored_bytes()].to_vec()
                }
            };
            sites.push(Node::leaf(
                at("texels"),
                Observed::bytes(stored, format.stored_layout()),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::read;

    /// The stored texels a recording holds for `case`'s `route`.
    fn recorded(case: &str, route: Route) -> Vec<u8> {
        let recording = read(case).unwrap_or_else(|why| panic!("{case}: {why}"));
        let answer = recording
            .answers
            .first()
            .unwrap_or_else(|| panic!("{case}: no answer"));
        let outcome = answer
            .outcomes
            .first()
            .unwrap_or_else(|| panic!("{case}: no outcome"));
        let name = format!("{}/texels", route.kernel());
        let node = outcome
            .transcript
            .root
            .child(&name)
            .unwrap_or_else(|| panic!("{case}: no site {name}"));
        match &node.observed.value {
            Value::Bytes(bytes) => bytes.clone(),
            other => panic!("{case}: {name} holds {other:?}"),
        }
    }

    /// Every code Metal stored for a swept value is the rule's, in every channel and through every
    /// route.
    #[test]
    fn every_recorded_code_is_metals_rule() {
        let mut checked = 0usize;
        for format in &FORMATS {
            if format.writes != Writes::Codes {
                continue;
            }
            let texels = format.texels();
            for route in format.routes() {
                let stored = recorded(format.case, route);
                assert_eq!(
                    stored.len(),
                    format.stored_bytes(),
                    "{}: {}",
                    format.case,
                    route.kernel()
                );
                for (at, written) in texels.iter().enumerate() {
                    let texel = &stored[at * format.bytes..(at + 1) * format.bytes];
                    for (channel_at, &(offset, channel)) in format.channels.iter().enumerate() {
                        let want = channel.code(written[channel_at]);
                        let got = field(texel, offset, channel.bits());
                        assert_eq!(
                            got,
                            want,
                            "{} by {}: texel {at} channel {channel_at} of {:e} ({:#010x}) at bit \
                             {offset}",
                            format.case,
                            route.kernel(),
                            written[channel_at],
                            written[channel_at].to_bits()
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100_000, "{checked} codes checked");
    }

    /// Each 10:10:10:2 format's fields hold their own channel's code, which is what tells the two
    /// field orders apart.
    #[test]
    fn every_recorded_field_holds_its_channels_code() {
        let mut checked = 0usize;
        let mut exchanged = 0usize;
        for format in &FORMATS {
            if format.writes != Writes::Fields {
                continue;
            }
            let texels = format.texels();
            for route in format.routes() {
                let stored = recorded(format.case, route);
                for (at, written) in texels.iter().enumerate() {
                    let texel = &stored[at * format.bytes..(at + 1) * format.bytes];
                    for (channel_at, &(offset, channel)) in format.channels.iter().enumerate() {
                        let want = channel.code(written[channel_at]);
                        assert_eq!(
                            field(texel, offset, channel.bits()),
                            want,
                            "{} by {}: texel {at} channel {channel_at} at bit {offset}",
                            format.case,
                            route.kernel()
                        );
                        checked += 1;
                    }
                    // Red and blue are the two the orders exchange, so a texel whose red and blue
                    // codes differ is one this case can tell apart at all.
                    let red = format.channels[0];
                    let blue = format.channels[2];
                    if red.1.code(written[0]) != blue.1.code(written[2]) {
                        exchanged += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "no field was checked");
        assert!(
            exchanged > 4000,
            "{exchanged} texels tell the two field orders apart"
        );
    }

    /// The routes a format states are all four but the one Metal's validation refuses.
    #[test]
    fn every_format_states_every_route_metal_allows() {
        for format in &FORMATS {
            let routes = format.routes();
            match format.pf {
                PixelFormat::BGR10A2Unorm => {
                    assert_eq!(routes, vec![Route::Write, Route::ReadWrite, Route::Held]);
                    assert!(!routes.contains(&Route::Buffer), "{}", format.case);
                }
                _ => assert_eq!(routes, Route::ALL.to_vec(), "{}", format.case),
            }
        }
        // And the one that is short is short of exactly one.
        let short: Vec<&str> = FORMATS
            .iter()
            .filter(|format| format.routes().len() != Route::ALL.len())
            .map(|format| format.case)
            .collect();
        assert_eq!(short, vec!["norm-storage-fields-bgr10a2unorm"]);
    }

    /// The two orders really are different, so the field test is two facts and not one repeated.
    #[test]
    fn the_two_ten_bit_orders_name_different_bits() {
        let rgb: Vec<u32> = RGB10A2.iter().map(|&(offset, _)| offset).collect();
        let bgr: Vec<u32> = BGR10A2.iter().map(|&(offset, _)| offset).collect();
        assert_eq!(rgb, vec![0, 10, 20, 30]);
        assert_eq!(bgr, vec![20, 10, 0, 30]);
        assert_ne!(rgb, bgr);
    }

    /// The sweeps are the ones the module docs state: whole where a channel is at most ten bits,
    /// and the stated selection at sixteen, every tie carrying two `f32` steps either side.
    #[test]
    fn the_sweeps_cover_what_they_claim() {
        for format in &FORMATS {
            if format.writes != Writes::Codes {
                continue;
            }
            let bits = format.channels[0].1.bits();
            let values: Vec<f32> = format.texels().iter().map(|texel| texel[0]).collect();
            let held: std::collections::BTreeSet<u64> = values
                .iter()
                .map(|&x| format.channels[0].1.code(x))
                .collect();
            if bits <= 10 {
                // Whole: every code of the channel is a code the sweep stores.
                let codes = match format.channels[0].1 {
                    Channel::Unorm(bits) => 1u64 << bits,
                    Channel::Snorm(bits) => (1u64 << bits) - 1,
                };
                assert_eq!(
                    held.len() as u64,
                    codes,
                    "{}: {} of {codes} codes",
                    format.case,
                    held.len()
                );
            } else {
                // The selection's ends are whole and its middle is strided.
                let selected = sixteen_bit_codes();
                assert_eq!(selected.len() as u64, ENDS * 2 + 127, "{}", format.case);
                assert!(
                    (0..ENDS).all(|code| selected.contains(&code)),
                    "{}: the lowest {ENDS} codes",
                    format.case
                );
                let last = (1u64 << 16) - 1;
                assert!(
                    (last - ENDS + 1..=last).all(|code| selected.contains(&code)),
                    "{}: the highest {ENDS} codes",
                    format.case
                );
                assert!(
                    held.len() > 300,
                    "{}: {} codes stored",
                    format.case,
                    held.len()
                );
            }
            // Every tie carries the two steps either side, which is the point of the sweep.
            let steps = around(0.5);
            assert_eq!(steps.len(), 5);
            assert!(steps[0] < steps[1] && steps[1] < steps[2]);
            assert!(steps[2] < steps[3] && steps[3] < steps[4]);
            // And the specials are in it.
            assert!(values.iter().any(|x| x.is_nan()), "{}: a NaN", format.case);
            assert!(
                values.contains(&f32::INFINITY),
                "{}: an infinity",
                format.case
            );
        }
    }

    /// Every case's stated layout covers exactly the bytes its read-back compares, and each case's
    /// dispatches are whole threadgroups, so no route announces a split.
    #[test]
    fn the_layouts_and_the_dispatches_are_whole() {
        let mut names = std::collections::BTreeSet::new();
        for format in &FORMATS {
            assert!(names.insert(format.case), "{}", format.case);
            let layout = format.stored_layout();
            assert_eq!(
                layout.region,
                crate::transcript::Region::Buffer {
                    offset: 0,
                    length: format.stored_bytes() as u64
                }
            );
            assert_eq!(layout.element.size() as usize, format.bytes);
            assert_eq!(layout.row_elements as usize, format.texels().len());
            assert!(
                format.stored_bytes() <= 220_000,
                "{}: {} bytes a route",
                format.case,
                format.stored_bytes()
            );
            assert_eq!(format.rows() % ROWS_PER_GROUP, 0, "{}", format.case);
            assert_eq!(format.buffer_texels() % BUFFER_GROUP, 0, "{}", format.case);
            assert!(format.rows() * WIDTH >= format.texels().len() as u32);
            assert!(format.buffer_texels() >= format.texels().len() as u32);
        }
        assert_eq!(names.len(), FORMATS.len());
    }

    /// The rule itself, at the ties where a tie rule is the whole difference.
    ///
    /// Only a tie that is **exactly** an `f32` is a test of the tie: `0.5 / 255` is not one, and
    /// its nearest `f32` times 255 lands under the half, which is why the sweeps carry two steps
    /// either side of every tie rather than the tie alone. The ties used here are exact: 127.5 of
    /// 255 and 63.5 of 127 are both 0.5.
    #[test]
    fn the_rules_ties_go_up_and_away_from_zero() {
        // A unorm tie goes up.
        assert_eq!(Channel::Unorm(8).code(0.5), 128);
        assert_eq!(Channel::Unorm(1).code(0.5), 1);
        assert_eq!(Channel::Unorm(2).code(0.5), 2);
        // Just under the tie goes down, so the rule is a tie rule and not a ceiling.
        assert_eq!(
            Channel::Unorm(8).code(f32::from_bits(0.5f32.to_bits() - 1)),
            127
        );
        // A snorm tie goes away from zero on both sides.
        assert_eq!(Channel::Snorm(8).code(0.5), 64);
        assert_eq!(Channel::Snorm(8).code(-0.5), 0xc0);
        assert_eq!(
            Channel::Snorm(8).code(f32::from_bits(0.5f32.to_bits() - 1)),
            63
        );
        // The clamp and the NaN.
        assert_eq!(Channel::Unorm(8).code(2.0), 255);
        assert_eq!(Channel::Unorm(8).code(-1.0), 0);
        assert_eq!(Channel::Unorm(8).code(f32::NAN), 0);
        assert_eq!(Channel::Snorm(8).code(f32::NAN), 0);
        assert_eq!(Channel::Snorm(8).code(-2.0), 0x81);
        assert_eq!(Channel::Snorm(8).code(f32::INFINITY), 127);
        assert_eq!(Channel::Snorm(8).code(f32::NEG_INFINITY), 0x81);
    }
}
