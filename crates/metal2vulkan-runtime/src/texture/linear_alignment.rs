//! `minimumLinearTextureAlignmentForPixelFormat:` on the device: per pixel format, the alignment of
//! a texture's offset and rows in a buffer at which the device's images hold the texture in place.
//!
//! - **Metal's rule stays the rule.** The runtime makes a texture over a buffer at any offset and
//!   `bytesPerRow` that are multiples of 16 (`linear_plan.rs`'s `LINEAR_ALIGNMENT`), as Metal
//!   does, and answers no alignment below it. A texture under the answer but at 16 is made as
//!   before: an image in place where the device takes its layout, else a shadow, announced
//!   (`shadowed-texture`). Nothing is refused for lying under the answer.
//! - **The answer is where the device's images sit.** For a format, the smallest power of two `P`
//!   from 16 to [`MOST`] at which a 2D texture of every width of [`WIDTHS`], [`ROWS`] rows tall
//!   (no multiple of 4, as a driver may render an explicit plane's rows in blocks of 4:
//!   `RuntimeDriverDefect::ExplicitPlaneRowsPastHeight`), of `ShaderRead` usage, its rows
//!   `width × texel` bytes rounded up to `P` apart, gets the image a texture over host pages takes
//!   in place (`texture/in_place_image.rs`'s `in_place_unbound`: the driver's linear image made for
//!   host allocations where it lays the rows out at that pitch, else the `DRM_FORMAT_MOD_LINEAR`
//!   image at it), and the image's memory alignment divides `P`, so that an offset that is a
//!   multiple of `P` meets it. The runtime asks the driver and takes its answer: each image is made
//!   and released, never bound.
//! - **Metal's 16 elsewhere**: on a device that imports no host pages, where no image is ever in
//!   place over them; for a format no candidate brings every width in place at (the device makes
//!   no linear image of it, or lays its rows at no power of two's pitch); and where a probe failed
//!   (a Vulkan error).
//! - **Fixed for the device's life.** A program lays its buffers out by the first answer it gets,
//!   so a format's first answer is the device's for good: kept by format
//!   (`TextureState::linear_alignments`), the first kept when two threads ask at once.
//! - **Kept per driver.** A probe's answer is written as a record in the shader cache's content
//!   directory, named by the driver's identity, the runtime's version, the probe's ([`VERSION`])
//!   and the format ([`record_name`]): a device of the same driver and runtime reads it at its first
//!   query of the format and probes nothing. A failed probe's 16 is kept nowhere. Counted
//!   (`WorkCounts::linear_alignments_probed`, `linear_alignments_recalled`).
//! - **The device's own** (`deviceLinearTextureAlignmentBytes` and `linearTextureAlignmentBytes`,
//!   `private_properties.rs`) is the largest of the answers of the formats a buffer backs.

use crate::buffer_texture::texel_bytes;
use crate::capabilities::linear_alignment_rule;
use crate::device::lock;
use crate::error::Error;
use crate::image_plan::{plan_image, ImageFacts};
use crate::in_place::DriverIdentity;
use crate::linear_plan::{linear_alignment, BufferLayout, LINEAR_ALIGNMENT};
use crate::shader_cache::store::{hex, Lookup};
use crate::texture_descriptor::{texture_usage, TextureLayout, TextureSwizzleChannels};
use crate::{release, Device};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use sha2::{Digest, Sha256};

/// The largest alignment the probe tries.
pub(crate) const MOST: u64 = 4096;

/// The widths, in texels, each candidate alignment is tried at: rows of one texel, of widths no
/// power of two divides, of powers of two and of a screen's width.
pub(crate) const WIDTHS: [u64; 9] = [1, 3, 17, 64, 65, 100, 129, 1000, 1920];

/// The rows of each texture the probe tries: no multiple of 4.
pub(crate) const ROWS: u64 = 3;

/// The version of what the probe tries and of how its record is named and framed: a change to
/// either bumps it, and a record of another version is never read.
pub(crate) const VERSION: u32 = 1;

/// A record's first bytes.
const MAGIC: [u8; 8] = *b"M2VLINA\0";

/// The domain tag of a record's name.
const NAME_TAG: &[u8] = b"metal2vulkan linear alignment record\0";

/// A record's length: magic, version, runtime, identity, answer, checksum.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 + 8 + 32;

/// The digest of `identity` and `pf` that names the format's record and that the record holds
/// again.
fn digest(identity: &DriverIdentity, pf: PixelFormat) -> [u8; 32] {
    let kind = format!("linear alignment of pixel format {}\0", pf.as_raw());
    identity.digest_for(NAME_TAG, kind.as_bytes(), VERSION)
}

/// The name of the record of `identity`'s answer for `pf` in the shader cache's content
/// directory.
pub(crate) fn record_name(identity: &DriverIdentity, pf: PixelFormat) -> String {
    hex(&digest(identity, pf))
}

/// The bytes of a record of `answer` for `pf` on `identity`:
///
/// ```text
/// magic          8 bytes   "M2VLINA\0"
/// version        u32       VERSION
/// runtime        u32       RUNTIME_VERSION
/// identity       32 bytes  the digest of the identity and the format
/// answer         u64       the alignment, in bytes
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian.
pub(crate) fn encode(identity: &DriverIdentity, pf: PixelFormat, answer: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&metal2vulkan::cache::RUNTIME_VERSION.to_le_bytes());
    bytes.extend_from_slice(&digest(identity, pf));
    bytes.extend_from_slice(&answer.to_le_bytes());
    let checksum: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The answer `bytes` record for `pf` on `identity`: `None` for anything but a whole record of
/// this version and runtime, for this identity and format, whose checksum holds and whose answer
/// is a power of two from 16 to [`MOST`].
pub(crate) fn decode(identity: &DriverIdentity, pf: PixelFormat, bytes: &[u8]) -> Option<u64> {
    if bytes.len() != RECORD_BYTES {
        return None;
    }
    let (body, checksum) = bytes.split_at(RECORD_BYTES - 32);
    let computed: [u8; 32] = Sha256::digest(body).into();
    if computed[..] != *checksum {
        return None;
    }
    let (magic, rest) = body.split_at(8);
    let (version, rest) = rest.split_at(4);
    let (runtime, rest) = rest.split_at(4);
    let (identity_digest, rest) = rest.split_at(32);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC
        || word(version) != VERSION
        || word(runtime) != metal2vulkan::cache::RUNTIME_VERSION
        || identity_digest != digest(identity, pf)
    {
        return None;
    }
    let answer = u64::from_le_bytes(rest.try_into().ok()?);
    (answer.is_power_of_two() && (LINEAR_ALIGNMENT..=MOST).contains(&answer)).then_some(answer)
}

impl Device {
    /// `minimumLinearTextureAlignmentForPixelFormat:`: the alignment of a texture's offset and
    /// `bytesPerRow` in a buffer at which this device's images hold a texture of `format` in place,
    /// at least the 16 bytes Metal answers on an Apple M3 Max and an M4 for every format a buffer
    /// can back.
    ///
    /// The runtime asks the device's driver at the format's first query: it makes the images a
    /// 2D texture over host pages takes in place, at each power of two from 16 bytes to 4096 and
    /// at widths of 1 to 1920 texels, and answers the first at which every width's image lies at
    /// the texture's pitch and its memory alignment divides the answer. None is bound. That first
    /// answer stays the device's for its life, and a device of the same driver and runtime
    /// release reads it again from the shader cache's content directory. It differs from Metal's
    /// where the device's images need more than 16 bytes: Mesa's RADV lays a linear image's rows
    /// at multiples of 256 bytes and aligns its memory to 256, and lavapipe aligns its memory to
    /// 64. It is 16 on a device that imports no host pages, for a format whose images lie at no
    /// such pitch, and where asking the driver failed.
    ///
    /// A texture at any other multiple of 16 is made all the same, as Metal makes it: in place
    /// where the device's images take its layout, else as a shadow of the buffer's bytes,
    /// announced ([`Degradation::ShadowedTexture`](crate::Degradation::ShadowedTexture)).
    ///
    /// Refused, `Invalid` with
    /// [`BufferTextureProblem::NotLinear`](crate::BufferTextureProblem::NotLinear), for a
    /// compressed, depth or stencil format, on which Metal's query asserts, and for a format Metal
    /// describes none of ([`Device::pixel_format_info`](crate::Device::pixel_format_info) answers
    /// [`PixelFormatInfo::INVALID`](crate::PixelFormatInfo::INVALID)), on which it asserts too.
    pub fn minimum_linear_texture_alignment_for_pixel_format(
        &self,
        format: PixelFormat,
    ) -> Result<u64, Error> {
        linear_alignment_rule(format)?;
        Ok(self.linear_texture_alignment(format))
    }

    /// The device's alignment of a texture of `pf`'s offset and rows in a buffer (the module
    /// docs), for a format a buffer backs: its first answer, kept for the device's life, else
    /// the record's, else the probe's.
    pub(crate) fn linear_texture_alignment(&self, pf: PixelFormat) -> u64 {
        let shared = self.shared();
        if shared.host_import.is_err() {
            return LINEAR_ALIGNMENT;
        }
        let kept = &shared.textures.linear_alignments;
        if let Some(&answer) = lock(kept).get(&pf) {
            return answer;
        }
        let answer = self.linear_texture_alignment_found(pf);
        *lock(kept).entry(pf).or_insert(answer)
    }

    /// The largest of [`linear_texture_alignment`](Self::linear_texture_alignment)'s answers
    /// over the pixel formats a buffer backs: the device's own linear texture alignment.
    pub(crate) fn device_linear_texture_alignment(&self) -> u64 {
        (PixelFormat::ALL.iter())
            .filter(|&&pf| linear_alignment(pf).is_ok())
            .map(|&pf| self.linear_texture_alignment(pf))
            .fold(LINEAR_ALIGNMENT, u64::max)
    }

    /// The answer for `pf` from its record, else from the probe, which is recorded; Metal's 16
    /// where the probe failed, recorded nowhere.
    fn linear_texture_alignment_found(&self, pf: PixelFormat) -> u64 {
        let shared = self.shared();
        let identity = self.driver_identity();
        let name = record_name(&identity, pf);
        let store = shared.shader_cache.content_store();
        if let Lookup::Hit(answer) =
            store.load_content(&name, |bytes| decode(&identity, pf, &bytes))
        {
            shared.counts.linear_alignment_recalled();
            return answer;
        }
        shared.counts.linear_alignment_probed();
        match self.linear_texture_alignment_probed(pf) {
            Ok(answer) => {
                // A failed write leaves the answer this device's alone.
                let _ = store.write_record_soon(&name, encode(&identity, pf, answer));
                answer
            }
            Err(_) => LINEAR_ALIGNMENT,
        }
    }

    /// The smallest power of two from 16 to [`MOST`] at which every width of [`WIDTHS`] takes an
    /// image in place (the module docs), else 16.
    fn linear_texture_alignment_probed(&self, pf: PixelFormat) -> Result<u64, Error> {
        let Some(texel) = texel_bytes(pf) else {
            return Ok(LINEAR_ALIGNMENT);
        };
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let mut alignment = LINEAR_ALIGNMENT;
        while alignment <= MOST {
            let mut every = true;
            for width in WIDTHS {
                if !self.in_place_at(pf, (width, texel), alignment, &facts)? {
                    every = false;
                    break;
                }
            }
            if every {
                return Ok(alignment);
            }
            alignment *= 2;
        }
        Ok(LINEAR_ALIGNMENT)
    }

    /// Whether a 2D `ShaderRead` texture of `pf`, `width` texels of `texel` bytes wide and
    /// [`ROWS`] tall, its rows its row's bytes rounded up to `alignment` apart, takes an image in
    /// place over host pages whose memory alignment divides `alignment`: the image is made and
    /// released.
    fn in_place_at(
        &self,
        pf: PixelFormat,
        (width, texel): (u64, u64),
        alignment: u64,
        facts: &ImageFacts<'_>,
    ) -> Result<bool, Error> {
        let layout = TextureLayout {
            texture_type: TextureType::Type2D,
            pixel_format: pf,
            extent: [width as u32, ROWS as u32, 1],
            levels: 1,
            layers: 1,
            samples: 1,
        };
        let usage = texture_usage(TextureUsage::SHADER_READ, TextureSwizzleChannels::DEFAULT);
        let Ok(plan) = plan_image(&layout, usage, false, facts) else {
            return Ok(false);
        };
        let row_bytes = width * texel;
        let row_pitch = row_bytes.next_multiple_of(alignment);
        let texels = BufferLayout {
            offset: 0,
            row_pitch,
            slice_pitch: row_pitch * ROWS,
            row_bytes,
            rows: ROWS,
            slices: 1,
        };
        let (made, _) = match self.in_place_unbound(&plan, &texels, facts)? {
            Ok(made) => made,
            Err(_) => return Ok(false),
        };
        // SAFETY: the image is live and unbound, and no command uses it.
        unsafe { release::object(self.shared().vk(), made.raw) };
        Ok(alignment.is_multiple_of(made.requirements.alignment.max(1)))
    }
}
