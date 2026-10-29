//! A tile function's implicit imageblock over a normalized color attachment (`lower::access::tile`)
//! on the device the cfg selects, under the validation layer. The owned fixture `cfb_tile_norm`'s
//! functions read each cell into a buffer, or write the buffer into the cells, over one attachment
//! 16 pixels wide in 16x16 tiles, one thread a pixel, and one writes its cell and reads it back. As
//! Metal probes measured on the M3 Max and an M4, plain and under the API validation layer (the
//! codes' reads and the stores' boundaries and specials over every code; the channel order; a read
//! after the function's own write):
//!
//! - a read gives each channel's value: an unorm code `c` of `b` bits is `c / (2^b - 1)`
//!   correctly rounded, an snorm one `max(c / (2^(b-1) - 1), -1)`, an sRGB colour code its 12-bit
//!   linear value over 4095 (`bind::SRGB_RESOLVE_LINEAR`); a member of four floats over fewer
//!   channels reads green and blue 0 and alpha 1 past them;
//! - a store gives each channel's code: `x` clamped to `[0, 1]` (`[-1, 1]` for snorm), a NaN 0,
//!   times the scale exactly, rounded to nearest even; an sRGB colour channel's 12-bit index so,
//!   encoded by Metal's thresholds (`bind::SRGB_RESOLVE_THRESHOLDS`); a member past the channels
//!   drops its lanes;
//! - `BGRA8Unorm`, its sRGB form and `BGR10A2Unorm` hold blue in the texel's low byte or field;
//! - a read after the function's own write reads the codes it stored.
//!
//! The reads cover every code of each 8-bit channel; the stores every tie between two 8-bit
//! codes with its neighbours, every seventh of the sRGB grid's, and the specials.
//!
//! Through a member of `half`s (a probe on the M3 Max and an M4, plain and under the API validation
//! layer, every code of every format and every `half` pattern stored): a read gives each channel's
//! exact value correctly rounded to `half` once, not its `float` value rounded again (which differs
//! at unorm16 codes 65455 and 65519 and snorm16 ±32759), and green and blue 0 and alpha 1 past the
//! channels; a store gives each channel's code of the `float` of the same value. These cover every
//! code of every channel and every `half` pattern in every lane.
//!
//! The function converts by the device's conversion (`arith::pack`, `device-pack-conversion`), so
//! each read and store is checked within that note's bound of Metal's ([`near`], [`code_near`]):
//! a value within 3 `f32` units in the last place (an sRGB colour within half a twelve-bit step
//! and the transfer's `Pow`, `1/8190 + 2^-14`), a `half` one within one more `half` step, a code
//! within one (a NaN channel's the device's).
//!
//! A format the device does not render to, or whose staging view it cannot store to, is refused
//! typed (`Unsupported`) and said.

use super::tile_states::on_device;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{SRGB_RESOLVE_LINEAR, SRGB_RESOLVE_THRESHOLDS};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{StageSource, TileRenderPipelineDescriptor};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, Device, Error, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, TextureDescriptor,
};

/// The attachment's width and the tile's side.
const SIDE: u32 = 16;

/// A channel of a normalized format: an unsigned or signed normalized one of its width, or an
/// sRGB colour channel of 8 bits.
#[derive(Clone, Copy, Debug)]
enum Channel {
    Unorm(u32),
    Snorm(u32),
    Srgb,
}

impl Channel {
    fn bits(self) -> u32 {
        match self {
            Channel::Unorm(bits) | Channel::Snorm(bits) => bits,
            Channel::Srgb => 8,
        }
    }

    /// The value Metal reads of code `c` (the module docs): the quotient is exact in `f64` and
    /// rounded once, no such quotient lying near a tie of `f32`.
    fn value(self, c: u32) -> f32 {
        match self {
            Channel::Unorm(bits) => (f64::from(c) / f64::from((1u32 << bits) - 1)) as f32,
            Channel::Snorm(bits) => {
                let signed = ((c << (32 - bits)) as i32) >> (32 - bits);
                let scale = f64::from((1u32 << (bits - 1)) - 1);
                (f64::from(signed) / scale).max(-1.0) as f32
            }
            Channel::Srgb => (f64::from(SRGB_RESOLVE_LINEAR[c as usize]) / 4095.0) as f32,
        }
    }

    /// The code Metal stores of `x` (the module docs); the product is exact in `f64`.
    fn code(self, x: f32) -> u32 {
        let x = if x.is_nan() { 0.0 } else { f64::from(x) };
        match self {
            Channel::Unorm(bits) => {
                (x.clamp(0.0, 1.0) * f64::from((1u32 << bits) - 1)).round_ties_even() as u32
            }
            Channel::Snorm(bits) => {
                let scale = f64::from((1u32 << (bits - 1)) - 1);
                let c = (x.clamp(-1.0, 1.0) * scale).round_ties_even() as i32;
                (c as u32) & ((1u32 << bits) - 1)
            }
            Channel::Srgb => {
                let index = (x.clamp(0.0, 1.0) * 4095.0).round_ties_even() as u16;
                SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= index) as u32
            }
        }
    }

    /// The exact value of code `c` ([`Channel::value`]'s before its rounding to `f32`), correctly
    /// rounded to `f64`: no `half` tie lies within an `f64` step of it (the grids are odd, so no
    /// quotient but 0 and 1 is dyadic, and each is at least `2^-51` from a `half` tie).
    fn exact(self, c: u32) -> f64 {
        match self {
            Channel::Unorm(bits) => f64::from(c) / f64::from((1u32 << bits) - 1),
            Channel::Snorm(bits) => {
                let signed = ((c << (32 - bits)) as i32) >> (32 - bits);
                let scale = f64::from((1u32 << (bits - 1)) - 1);
                (f64::from(signed) / scale).max(-1.0)
            }
            Channel::Srgb => f64::from(SRGB_RESOLVE_LINEAR[c as usize]) / 4095.0,
        }
    }

    /// The `half` Metal reads of code `c` through a `half` member, as its `f32` (the module
    /// docs): the exact value rounded to nearest even once.
    fn half(self, c: u32) -> f32 {
        half_value(f64_to_half_bits(self.exact(c)))
    }
}

/// The IEEE binary16 bits of `x` (within `[-1, 1]`) rounded to nearest even once.
fn f64_to_half_bits(x: f64) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    let bits = if a < f64::powi(2.0, -14) {
        // The denormal grid `2^-24`, whose top step is the least normal's bits.
        (a * f64::powi(2.0, 24)).round_ties_even() as u16
    } else {
        let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
        let n = (a * f64::powi(2.0, 10 - e)).round_ties_even() as i32;
        let (e, n) = if n == 2048 { (e + 1, 1024) } else { (e, n) };
        (((e + 15) << 10) | (n - 1024)) as u16
    };
    sign | bits
}

/// The `f32` of `half` bits `h`: a denormal is a normal `f32`, an infinity an infinity, and a
/// NaN a NaN.
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1F);
    let man = f32::from(h & 0x3FF);
    sign * match exp {
        0 => man * f32::powi(2.0, -24),
        31 if man == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024.0 + man) * f32::powi(2.0, exp - 25),
    }
}

/// A normalized attachment format: its channels in Metal's order, each with its bit offset in
/// the texel, and the texel's bytes.
struct Format {
    pf: PixelFormat,
    channels: &'static [(Channel, u32)],
    bytes: u32,
}

/// Whether `have`, the device's conversion's value of a `channel` read (through a `half` member
/// where `half`), lies within `PrecisionNote::DevicePackConversion`'s bound of Metal's `want` (the
/// module docs). A lane past the channels holds its constant exactly.
fn near(channel: Option<Channel>, have: f32, want: f32, half: bool) -> bool {
    let Some(channel) = channel else {
        return have.to_bits() == want.to_bits();
    };
    let ulp = |x: f32| f32::from_bits(x.abs().to_bits() + 1) - x.abs();
    let half_step = |x: f32| match x.abs() {
        a if a < f32::powi(2.0, -14) => f32::powi(2.0, -24),
        a => f32::powi(2.0, (a.to_bits() >> 23) as i32 - 127 - 10),
    };
    let mut bound = match channel {
        Channel::Srgb => 1.0 / 8190.0 + f32::powi(2.0, -14),
        Channel::Unorm(_) | Channel::Snorm(_) => 3.0 * ulp(want),
    };
    if half {
        bound += half_step(want);
    }
    (have - want).abs() <= bound
}

/// Whether code `a` of `channel` lies within one of code `b` (signed codes as their values).
fn code_near(channel: Channel, a: u32, b: u32) -> bool {
    match channel {
        Channel::Snorm(bits) => {
            let signed = |c: u32| i64::from(((c << (32 - bits)) as i32) >> (32 - bits));
            (signed(a) - signed(b)).abs() <= 1
        }
        Channel::Unorm(_) | Channel::Srgb => a.abs_diff(b) <= 1,
    }
}

impl Format {
    /// The code of channel `k` of `texel`.
    fn code(&self, texel: u64, k: usize) -> u32 {
        let (channel, at) = self.channels[k];
        ((texel >> at) & ((1u64 << channel.bits()) - 1)) as u32
    }
}

const FORMATS: [Format; 15] = {
    use Channel::{Snorm, Srgb, Unorm};
    [
        Format {
            pf: PixelFormat::RGBA8Unorm,
            channels: &[(Unorm(8), 0), (Unorm(8), 8), (Unorm(8), 16), (Unorm(8), 24)],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::BGRA8Unorm,
            channels: &[(Unorm(8), 16), (Unorm(8), 8), (Unorm(8), 0), (Unorm(8), 24)],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::RGBA8UnormSRGB,
            channels: &[(Srgb, 0), (Srgb, 8), (Srgb, 16), (Unorm(8), 24)],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::BGRA8UnormSRGB,
            channels: &[(Srgb, 16), (Srgb, 8), (Srgb, 0), (Unorm(8), 24)],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::RGBA8Snorm,
            channels: &[(Snorm(8), 0), (Snorm(8), 8), (Snorm(8), 16), (Snorm(8), 24)],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::RGB10A2Unorm,
            channels: &[
                (Unorm(10), 0),
                (Unorm(10), 10),
                (Unorm(10), 20),
                (Unorm(2), 30),
            ],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::BGR10A2Unorm,
            channels: &[
                (Unorm(10), 20),
                (Unorm(10), 10),
                (Unorm(10), 0),
                (Unorm(2), 30),
            ],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::RG16Unorm,
            channels: &[(Unorm(16), 0), (Unorm(16), 16)],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::RG16Snorm,
            channels: &[(Snorm(16), 0), (Snorm(16), 16)],
            bytes: 4,
        },
        Format {
            pf: PixelFormat::RGBA16Unorm,
            channels: &[
                (Unorm(16), 0),
                (Unorm(16), 16),
                (Unorm(16), 32),
                (Unorm(16), 48),
            ],
            bytes: 8,
        },
        Format {
            pf: PixelFormat::RGBA16Snorm,
            channels: &[
                (Snorm(16), 0),
                (Snorm(16), 16),
                (Snorm(16), 32),
                (Snorm(16), 48),
            ],
            bytes: 8,
        },
        Format {
            pf: PixelFormat::RG8Unorm,
            channels: &[(Unorm(8), 0), (Unorm(8), 8)],
            bytes: 2,
        },
        Format {
            pf: PixelFormat::RG8Snorm,
            channels: &[(Snorm(8), 0), (Snorm(8), 8)],
            bytes: 2,
        },
        Format {
            pf: PixelFormat::R16Unorm,
            channels: &[(Unorm(16), 0)],
            bytes: 2,
        },
        Format {
            pf: PixelFormat::R16Snorm,
            channels: &[(Snorm(16), 0)],
            bytes: 2,
        },
    ]
};

/// The floats a store writes: specials, the bounds, every tie between two 8-bit codes with its
/// neighbours, and every seventh of the 12-bit sRGB grid's with its neighbours.
fn inputs() -> Vec<f32> {
    let mut xs = vec![
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        -1.0,
        1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        1.0f32.next_up(),
        (-1.0f32).next_down(),
        1.0f32.next_down(),
    ];
    for k in 0..255 {
        let tie = ((f64::from(k) + 0.5) / 255.0) as f32;
        xs.extend([tie.next_down(), tie, tie.next_up(), -tie]);
    }
    for k in (0..4095).step_by(7) {
        let tie = ((f64::from(k) + 0.5) / 4095.0) as f32;
        xs.extend([tie.next_down(), tie, tie.next_up()]);
    }
    xs
}

/// The functions of `cfb_tile_norm` on one device.
struct Fixture {
    device: Device,
    library: Library,
}

impl Fixture {
    fn new(device: &Device) -> Fixture {
        let bitcode = llvm_assemble(&engine_dir().join("cfb_tile_norm.ll")).unwrap();
        Fixture {
            library: device.new_library_with_air(&bitcode).unwrap(),
            device: device.clone(),
        }
    }

    /// Dispatch `entry` once per tile in a pass over a new attachment of `format` holding
    /// `texels` (`SIDE` wide, loaded and stored), with `buffer` at tile buffer 0: the buffer and
    /// the texels after, or the refusal of the state, the attachment or the dispatch.
    fn run(
        &self,
        entry: &str,
        format: &Format,
        texels: &[u64],
        buffer: &[u8],
    ) -> Result<(Vec<u8>, Vec<u64>), Error> {
        let function = self.library.new_function_with_name(entry).unwrap();
        let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Function(
            function.function_descriptor(),
        ));
        descriptor.color_attachment_formats[0] = Some(format.pf);
        descriptor.threadgroup_size_matches_tile_size = true;
        let state = self
            .device
            .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())?;
        let height = (texels.len() as u64).div_ceil(u64::from(SIDE));
        let bytes = u64::from(format.bytes);
        let row = u64::from(SIDE) * bytes;
        let mut texture = TextureDescriptor::texture_2d(format.pf, SIDE.into(), height, false);
        texture.usage = TextureUsage::RENDER_TARGET;
        texture.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&texture)?;
        let upload: Vec<u8> = texels
            .iter()
            .flat_map(|t| t.to_le_bytes()[..format.bytes as usize].to_vec())
            .collect();
        let region = Region::new_2d(0, 0, SIDE.into(), height);
        target
            .replace_region(region, 0, 0, &upload, row, row * height)
            .unwrap();
        let buffer = buffer_with(&self.device, buffer).unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.color_attachments[0].texture = Some(target.clone());
        pass.color_attachments[0].load_action = LoadAction::Load;
        pass.color_attachments[0].store_action = StoreAction::Store;
        pass.tile_width = SIDE.into();
        pass.tile_height = SIDE.into();
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(&state)?;
        encoder.set_tile_buffer(Some(&buffer), 0, 0)?;
        let dispatched = encoder.dispatch_threads_per_tile(Size::new(SIDE.into(), SIDE.into(), 1));
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        dispatched?;
        let mut after = vec![0u8; upload.len()];
        target
            .get_bytes(&mut after, row, row * height, region, 0, 0)
            .unwrap();
        let after = after
            .chunks_exact(format.bytes as usize)
            .map(|t| {
                let mut word = [0u8; 8];
                word[..t.len()].copy_from_slice(t);
                u64::from_le_bytes(word)
            })
            .collect();
        Ok((read_buffer(&buffer, 0, buffer.length()).unwrap(), after))
    }
}

/// `run`'s answer, or `None` where the device refuses the format, said.
fn ran(
    name: &str,
    entry: &str,
    format: &Format,
    run: Result<(Vec<u8>, Vec<u64>), Error>,
) -> Option<(Vec<u8>, Vec<u64>)> {
    match run {
        Ok(answer) => Some(answer),
        Err(Error::Unsupported(why)) => {
            eprintln!("refused: {entry} over {:?} on {name}: {why:?}", format.pf);
            None
        }
        Err(error) => panic!("{entry} over {:?} on {name}: {error}", format.pf),
    }
}

/// The floats of `bytes`.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|f| f32::from_le_bytes(*f))
        .collect()
}

/// Each format's reads, through a member of its channels and of four floats: every texel's
/// value, and the texels kept.
#[test]
fn a_normalized_attachment_reads_as_metal_does() {
    on_device(|device| {
        let f = Fixture::new(device);
        let name = device.name().to_owned();
        let cells = (SIDE * SIDE) as usize;
        for format in &FORMATS {
            let n = format.channels.len();
            // Texel `i`'s channel `k` holds code `(i (2k + 1) + 37k) mod 2^bits`, so each 8-bit
            // channel takes every code; the last two every bit and each channel's top bit.
            let mut texels: Vec<u64> = (0..cells as u64)
                .map(|i| {
                    format
                        .channels
                        .iter()
                        .enumerate()
                        .fold(0u64, |t, (k, &(c, at))| {
                            let k = k as u64;
                            t | (((i * (2 * k + 1) + 37 * k) & ((1u64 << c.bits()) - 1)) << at)
                        })
                })
                .collect();
            texels[cells - 1] = u64::MAX >> (64 - 8 * format.bytes);
            texels[cells - 2] = format
                .channels
                .iter()
                .fold(0, |t, &(c, at)| t | ((1u64 << (c.bits() - 1)) << at));
            let mut reads = vec![(n, format!("cfb_tn_read{n}"))];
            if n < 4 {
                reads.push((4, "cfb_tn_read4".to_owned()));
            }
            for (lanes, entry) in reads {
                let run = f.run(&entry, format, &texels, &vec![0xAB; cells * 16]);
                let Some((buffer, after)) = ran(&name, &entry, format, run) else {
                    continue;
                };
                let got = floats(&buffer);
                for (i, &t) in texels.iter().enumerate() {
                    let want: Vec<f32> = (0..4)
                        .map(|k| match k {
                            k if k < n => format.channels[k].0.value(format.code(t, k)),
                            k if k < lanes => [0.0, 0.0, 0.0, 1.0][k],
                            _ => -7.0,
                        })
                        .collect();
                    let have = &got[4 * i..4 * i + 4];
                    let channel = |k: usize| format.channels.get(k).map(|c| c.0);
                    assert!(
                        (0..4).all(|k| near(channel(k), have[k], want[k], false)),
                        "{name}: {entry} over {:?}, texel {t:#x}: {have:?}, Metal's {want:?}",
                        format.pf
                    );
                }
                assert_eq!(after, texels, "{name}: {entry} over {:?} keeps", format.pf);
            }
        }
    });
}

/// Each format's stores, through a member of its channels and of four floats: every texel's
/// codes; and a read after the function's own write.
#[test]
fn a_normalized_attachment_stores_as_metal_does() {
    on_device(|device| {
        let f = Fixture::new(device);
        let name = device.name().to_owned();
        let xs = inputs();
        let tile = (SIDE * SIDE) as usize;
        let cells = xs.len().div_ceil(3).div_ceil(tile) * tile;
        let lane = |i: usize, k: usize| xs[(3 * i + k) % xs.len()];
        let values: Vec<u8> = (0..cells)
            .flat_map(|i| (0..4).flat_map(move |k| lane(i, k).to_le_bytes()))
            .collect();
        let cleared = vec![0xCDCD_CDCD_CDCD_CDCD; cells];
        for format in &FORMATS {
            let n = format.channels.len();
            let mut writes = vec![format!("cfb_tn_write{n}")];
            if n < 4 {
                writes.push("cfb_tn_write4".to_owned());
            }
            for entry in writes {
                let run = f.run(&entry, format, &cleared, &values);
                let Some((_, after)) = ran(&name, &entry, format, run) else {
                    continue;
                };
                for (i, &t) in after.iter().enumerate() {
                    let want = format
                        .channels
                        .iter()
                        .enumerate()
                        .fold(0u64, |w, (k, &(c, at))| {
                            w | (u64::from(c.code(lane(i, k))) << at)
                        });
                    let unwritten = !format
                        .channels
                        .iter()
                        .fold(0u64, |m, &(c, at)| m | (((1u64 << c.bits()) - 1) << at));
                    // A NaN channel's code is the device's (the note's bound).
                    let codes_near = (format.channels.iter().enumerate()).all(|(k, &(c, _))| {
                        lane(i, k).is_nan() || code_near(c, format.code(t, k), format.code(want, k))
                    });
                    assert!(
                        codes_near && (t & unwritten) == (want & unwritten),
                        "{name}: {entry} over {:?}, cell {i}: {t:#x}, Metal's {want:#x}: {:?}",
                        format.pf,
                        (0..4).map(|k| lane(i, k)).collect::<Vec<_>>()
                    );
                }
            }
        }
        // A read after the function's own write reads the codes it stored.
        let format = &FORMATS[9];
        assert_eq!(format.pf, PixelFormat::RGBA16Unorm);
        let run = f.run("cfb_tn_round4", format, &cleared, &values);
        let Some((buffer, after)) = ran(&name, "cfb_tn_round4", format, run) else {
            return;
        };
        let got = floats(&buffer);
        for i in 0..cells {
            for k in 0..4 {
                let (channel, _) = format.channels[k];
                let stored = format.code(after[i], k);
                assert!(
                    lane(i, k).is_nan() || code_near(channel, stored, channel.code(lane(i, k))),
                    "{name}: round cell {i} lane {k}: code {stored}"
                );
                let (have, want) = (got[4 * i + k], channel.value(stored));
                assert!(
                    near(Some(channel), have, want, false),
                    "{name}: round cell {i} lane {k}: {have}, Metal's {want}"
                );
            }
        }
    });
}

/// Each format's reads through a member of its channels and of four `half`s: every code of every
/// channel (`2^b` texels of a `b`-bit channel at most, channel `k` of texel `i` code
/// `(i (2k + 1) + 37k) mod 2^b`, a bijection), each texel's value, and the texels kept.
#[test]
fn a_normalized_attachment_reads_through_halves_as_metal_does() {
    on_device(|device| {
        let f = Fixture::new(device);
        let name = device.name().to_owned();
        let tile = (SIDE * SIDE) as usize;
        let mut twice = 0;
        for format in &FORMATS {
            let n = format.channels.len();
            let widest = format.channels.iter().map(|(c, _)| c.bits()).max().unwrap();
            let cells = (1usize << widest).max(tile);
            let texels: Vec<u64> = (0..cells as u64)
                .map(|i| {
                    format
                        .channels
                        .iter()
                        .enumerate()
                        .fold(0u64, |t, (k, &(c, at))| {
                            let k = k as u64;
                            t | (((i * (2 * k + 1) + 37 * k) & ((1u64 << c.bits()) - 1)) << at)
                        })
                })
                .collect();
            let mut reads = vec![(n, format!("cfb_tn_hread{n}"))];
            if n < 4 {
                reads.push((4, "cfb_tn_hread4".to_owned()));
            }
            for (lanes, entry) in reads {
                let run = f.run(&entry, format, &texels, &vec![0xAB; cells * 16]);
                let Some((buffer, after)) = ran(&name, &entry, format, run) else {
                    continue;
                };
                let got = floats(&buffer);
                for (i, &t) in texels.iter().enumerate() {
                    let want: Vec<f32> = (0..4)
                        .map(|k| match k {
                            k if k < n => format.channels[k].0.half(format.code(t, k)),
                            k if k < lanes => [0.0, 0.0, 0.0, 1.0][k],
                            _ => -7.0,
                        })
                        .collect();
                    let have = &got[4 * i..4 * i + 4];
                    let channel = |k: usize| format.channels.get(k).map(|c| c.0);
                    assert!(
                        (0..4).all(|k| near(channel(k), have[k], want[k], true)),
                        "{name}: {entry} over {:?}, texel {t:#x}: {have:?}, Metal's {want:?}",
                        format.pf
                    );
                    for (k, &(c, _)) in format.channels.iter().enumerate() {
                        let code = format.code(t, k);
                        let again = f64_to_half_bits(f64::from(c.value(code)));
                        twice += usize::from(again != f64_to_half_bits(c.exact(code)));
                    }
                }
                assert_eq!(after, texels, "{name}: {entry} over {:?} keeps", format.pf);
            }
        }
        eprintln!("{name}: {twice} channel reads where a float rounded again would differ");
    });
}

/// Each format's stores through a member of its channels and of four `half`s: every `half`
/// pattern in every lane (cell `i`'s lane `k` holds pattern `(i + 21845 k) mod 2^16`), and every
/// texel's codes.
#[test]
fn a_normalized_attachment_stores_halves_as_metal_does() {
    on_device(|device| {
        let f = Fixture::new(device);
        let name = device.name().to_owned();
        let cells = 1usize << 16;
        let lane = |i: usize, k: usize| ((i + 21845 * k) & 0xFFFF) as u16;
        let values: Vec<u8> = (0..cells)
            .flat_map(|i| (0..4).flat_map(move |k| lane(i, k).to_le_bytes()))
            .collect();
        let cleared = vec![0xCDCD_CDCD_CDCD_CDCD; cells];
        for format in &FORMATS {
            let n = format.channels.len();
            let mut writes = vec![format!("cfb_tn_hwrite{n}")];
            if n < 4 {
                writes.push("cfb_tn_hwrite4".to_owned());
            }
            for entry in writes {
                let run = f.run(&entry, format, &cleared, &values);
                let Some((_, after)) = ran(&name, &entry, format, run) else {
                    continue;
                };
                for (i, &t) in after.iter().enumerate() {
                    let want = format
                        .channels
                        .iter()
                        .enumerate()
                        .fold(0u64, |w, (k, &(c, at))| {
                            w | (u64::from(c.code(half_value(lane(i, k)))) << at)
                        });
                    // A NaN channel's code is the device's (the note's bound).
                    let codes_near = (format.channels.iter().enumerate()).all(|(k, &(c, _))| {
                        half_value(lane(i, k)).is_nan()
                            || code_near(c, format.code(t, k), format.code(want, k))
                    });
                    assert!(
                        codes_near,
                        "{name}: {entry} over {:?}, cell {i}: {t:#x}, Metal's {want:#x}: {:04x?}",
                        format.pf,
                        (0..4).map(|k| lane(i, k)).collect::<Vec<_>>()
                    );
                }
            }
        }
    });
}
