//! `srgb-storage-writes`: what a kernel's `access::write` stores into an 8-bit sRGB texture.
//!
//! Metal promises a write to an `RGBA8Unorm_sRGB` or `BGRA8Unorm_sRGB` texture on every Metal 3
//! device (the feature set tables: both formats are "All" in every family, "Write" among them),
//! and the Metal Shading Language specification (8.7.7) says how: "Conversion from linear to sRGB
//! space is automatically done when writing to an sRGB texture. If the texture has an alpha
//! channel, the alpha data is stored in linear color space." `read_write` access is not promised
//! for either format (`MTLReadWriteTextureTier` lists no sRGB format), so this case writes only.
//!
//! The kernels are the owned fixture `fpv_srgb_storage`'s `fpv_srgb_write` and
//! `fpv_srgb_write_half`, which write buffer 0's `float4` or `half4` at the thread's texel of a
//! row [`WIDTH`] texels wide (buffer 1). Each writes a [`WIDTH`] by [`HEIGHT`] texture of each
//! format, and the case records the texture's bytes by `getBytes:`, one code a channel.
//!
//! The values are the linear values the sRGB transfer decodes each code to ([`center`]), so the
//! subject is the encode at the write and the channel order, not where a device's encode rounds
//! a value between two codes (`pass-bytes-srgb-encode-sweep` reads that): row 0 takes every code
//! in red, green and blue, each channel in its own order, and alpha `i / 255`; row 1 starts with
//! values past either end, each clamped, then every code again in other orders.
//!
//! One compute pass a sweep, one thread a texel, no loop and no sampling: nothing here can fault
//! a GPU.

use super::extended_range_targets::half_bits;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::TextureRequest;
use crate::transcript::{Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The fixture both kernels live in.
const FIXTURE: &str = "fpv_srgb_storage";

/// The kernels the case runs: a `float4` write and a `half4` write.
const KERNELS: [&str; 2] = ["fpv_srgb_write", "fpv_srgb_write_half"];

/// The formats the case writes.
const FORMATS: [PixelFormat; 2] = [PixelFormat::RGBA8UnormSRGB, PixelFormat::BGRA8UnormSRGB];

/// The texture's width, which buffer 1 holds.
pub const WIDTH: u32 = 256;

/// The texture's height.
pub const HEIGHT: u32 = 2;

/// The values past either end that start row 1, a texel each: Metal clamps each channel to
/// `[0, 1]` before it encodes.
const CLAMPED: [[f32; 4]; 4] = [
    [-1.0, 2.0, 0.0, 1.0],
    [-0.0, 1.0, f32::INFINITY, -0.5],
    [f32::NEG_INFINITY, 1.5, 1.0, 2.0],
    [0.0, -0.25, 1e6, 0.0],
];

/// The linear value the sRGB transfer decodes code `code` to: `c / 12.92` up to `c = 0.04045`,
/// else `((c + 0.055) / 1.055)^2.4`, for `c = code / 255`, in `f64`, rounded to `f32`. Its encode
/// lands on the code itself, far from the half-code where an encode rounds.
pub fn center(code: u32) -> f32 {
    let c = f64::from(code) / 255.0;
    let linear = match c <= 0.04045 {
        true => c / 12.92,
        false => ((c + 0.055) / 1.055).powf(2.4),
    };
    linear as f32
}

/// Every texel's `float4`, row by row.
pub fn texels() -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    for i in 0..WIDTH {
        let alpha = i as f32 / 255.0;
        out.push([center(i), center(255 - i), center((i * 37) & 255), alpha]);
    }
    for i in 0..WIDTH {
        match CLAMPED.get(i as usize) {
            Some(&texel) => out.push(texel),
            None => out.push([
                center(255 - i),
                center((i * 101) & 255),
                center(i ^ 0x55),
                1.0 - i as f32 / 255.0,
            ]),
        }
    }
    out
}

/// Buffer 0's bytes for `kernel`: each texel's `float4`, or its `half4`.
fn values(kernel: &str) -> Vec<u8> {
    let half = kernel.ends_with("_half");
    let mut bytes = Vec::new();
    for value in texels().into_iter().flatten() {
        match half {
            true => bytes.extend(half_bits(value).to_le_bytes()),
            false => bytes.extend(value.to_bits().to_le_bytes()),
        }
    }
    bytes
}

/// Buffer 1's bytes: the row's width, as the kernels' `constant uint &width`.
fn width_bytes() -> Vec<u8> {
    WIDTH.to_le_bytes().to_vec()
}

/// The texture a sweep writes: shared storage, written by shaders alone.
fn request(format: PixelFormat) -> TextureRequest<'static> {
    TextureRequest::read_2d(format.as_raw(), WIDTH, HEIGHT).used(TextureUsage::SHADER_WRITE)
}

/// The layout the stored bytes are read under: one row a texel row, one code a channel.
fn stored_layout() -> ByteLayout {
    let pitch = WIDTH * 4;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [WIDTH, HEIGHT, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: pitch,
        row_pitch: pitch,
        rows: HEIGHT,
        image_pitch: pitch * HEIGHT,
        images: 1,
        // Four codes a texel cover every byte of a row, so there is no gap.
        gaps: Gaps::Ignored,
    }
}

/// The case.
pub struct SrgbStorageWrites;

impl Case for SrgbStorageWrites {
    fn name(&self) -> &'static str {
        "srgb-storage-writes"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for kernel in KERNELS {
            let fixture = Fixture {
                name: FIXTURE,
                entry: kernel,
            };
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: kernel.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture::new(FIXTURE);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:1", &width_bytes()),
        ];
        for kernel in KERNELS {
            inputs.push(Input::bytes(format!("{kernel}/buffer:0"), &values(kernel)));
        }
        for format in FORMATS {
            inputs.push(Input::bytes(format.name(), &request(format).words()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let shape = made(sites, "buffer/width", side.buffer(&width_bytes()))?;
        for kernel in KERNELS {
            let pipeline = made(
                sites,
                &format!("{kernel}/pipeline"),
                side.compute_pipeline(library, kernel),
            )?;
            let values = made(
                sites,
                &format!("{kernel}/buffer/0"),
                side.buffer(&values(kernel)),
            )?;
            for format in FORMATS {
                let at = |site: &str| format!("{}/{kernel}/{site}", format.name());
                let texture = made(sites, &at("texture"), side.texture(&request(format)))?;
                let command_buffer = made(
                    sites,
                    &at("dispatch"),
                    side.dispatch(
                        pipeline,
                        &Bindings::new(&[texture], &[values, shape]),
                        [WIDTH / 16, HEIGHT, 1],
                        [16, 1, 1],
                    ),
                )?;
                call(
                    sites,
                    &at("status"),
                    side.wait(command_buffer),
                    Clone::clone,
                )?;
                let stored = call(
                    sites,
                    &at("texels"),
                    side.read_texture(texture, [WIDTH, HEIGHT], WIDTH * 4),
                    |_| crate::transcript::Value::Absent,
                )?;
                sites.push(Node::leaf(
                    at("stored"),
                    Observed::bytes(stored, stored_layout()),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MSL transfer's encode of `value`, rounded to nearest: `12.92 v` below 0.0031308, else
    /// `1.055 v^(1/2.4) - 0.055`, times 255, of `value` clamped to `[0, 1]`.
    fn encoded(value: f32) -> f64 {
        let v = f64::from(value).clamp(0.0, 1.0);
        let c = match v < 0.0031308 {
            true => 12.92 * v,
            false => 1.055 * v.powf(1.0 / 2.4) - 0.055,
        };
        c * 255.0
    }

    /// Every code's value encodes within a hundredth of a code of the code itself in `float`, and
    /// within a twentieth through its `half`, so neither side's encode meets a value between two
    /// codes.
    #[test]
    fn every_value_encodes_at_its_code() {
        for code in 0..256 {
            let value = center(code);
            let float = encoded(value);
            assert!(
                (float - f64::from(code)).abs() < 0.01,
                "float {code}: {float}"
            );
            let half = half_encoded(value);
            assert!((half - f64::from(code)).abs() < 0.05, "half {code}: {half}");
        }
    }

    /// The encode of `value`'s `half`.
    fn half_encoded(value: f32) -> f64 {
        let bits = half_bits(value);
        let exponent = i32::from((bits >> 10) & 0x1f);
        let significand = f64::from(bits & 0x3ff);
        let size = match exponent {
            0 => significand * 2f64.powi(-24),
            e => (1.0 + significand / 1024.0) * 2f64.powi(e - 15),
        };
        encoded(size as f32)
    }
}
