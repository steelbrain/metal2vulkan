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
//! - **Every format at once, fixed for the device's life.** The first query of any format works
//!   out every format a buffer backs (a probe of all 175 took 17 to 23 ms on an M3 Max, a Radeon
//!   780M and lavapipe, and reading them again from the record 0.3 ms), and those answers are the device's for good
//!   (`TextureState::linear_alignments`): a program lays its buffers out by the first answer it
//!   gets, and two threads asking at once take the one set.
//! - **Kept per driver.** The answers are written as one record in the shader cache's content
//!   directory, named by the driver's identity, the runtime's version and the probe's
//!   ([`VERSION`], [`record_name`]): a device of the same driver and runtime reads it at its first
//!   query and probes nothing. Where a probe failed, the format's 16 is the device's and no record
//!   is written. Counted (`WorkCounts::linear_alignments_probed`, `linear_alignments_recalled`).
//! - **The device's own** (`deviceLinearTextureAlignmentBytes` and `linearTextureAlignmentBytes`,
//!   `private_properties.rs`) is the largest of the answers of the formats a buffer backs.

use crate::buffer_texture::texel_bytes;
use crate::capabilities::linear_alignment_rule;
use crate::error::Error;
use crate::image_plan::{plan_image, ImageFacts};
use crate::in_place::DriverIdentity;
use crate::linear_plan::{linear_alignment, BufferLayout, LINEAR_ALIGNMENT};
use crate::shader_cache::store::{hex, Lookup};
use crate::texture_descriptor::{texture_usage, TextureLayout, TextureSwizzleChannels};
use crate::{release, Device};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

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

/// What the record holds, in its name.
const KIND: &[u8] = b"linear alignment of every pixel format a buffer backs\0";

/// The bytes of a record before its answers: magic, version, runtime, identity, count.
const HEAD_BYTES: usize = 8 + 4 + 4 + 32 + 4;

/// The bytes of one answer in a record: the format's raw value and the alignment.
const ANSWER_BYTES: usize = 4 + 8;

/// The pixel formats a buffer backs, whose answers a record holds: every one
/// `linear_plan::linear_alignment` answers.
pub(crate) fn backed_formats() -> impl Iterator<Item = PixelFormat> {
    (PixelFormat::ALL.iter().copied()).filter(|&pf| linear_alignment(pf).is_ok())
}

/// The digest of `identity` that names its record and that the record holds again.
fn digest(identity: &DriverIdentity) -> [u8; 32] {
    identity.digest_for(NAME_TAG, KIND, VERSION)
}

/// The name of the record of `identity`'s answers in the shader cache's content directory.
pub(crate) fn record_name(identity: &DriverIdentity) -> String {
    hex(&digest(identity))
}

/// The bytes of a record of `answers` on `identity`:
///
/// ```text
/// magic          8 bytes   "M2VLINA\0"
/// version        u32       VERSION
/// runtime        u32       RUNTIME_VERSION
/// identity       32 bytes  the identity's digest for this record
/// count          u32       the answers that follow
/// answers        12 bytes each, by raw format: the format (u32) and its alignment (u64)
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian.
pub(crate) fn encode(identity: &DriverIdentity, answers: &HashMap<PixelFormat, u64>) -> Vec<u8> {
    let mut sorted: Vec<(u32, u64)> = (answers.iter())
        .map(|(pf, &answer)| (pf.as_raw(), answer))
        .collect();
    sorted.sort_unstable();
    let mut bytes = Vec::with_capacity(HEAD_BYTES + ANSWER_BYTES * sorted.len() + 32);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&metal2vulkan::cache::RUNTIME_VERSION.to_le_bytes());
    bytes.extend_from_slice(&digest(identity));
    bytes.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
    for (raw, answer) in sorted {
        bytes.extend_from_slice(&raw.to_le_bytes());
        bytes.extend_from_slice(&answer.to_le_bytes());
    }
    let checksum: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The answers `bytes` record on `identity`: `None` for anything but a whole record of this
/// version and runtime, for this identity, whose checksum holds, that answers every format a
/// buffer backs once and nothing else, each a power of two from 16 to [`MOST`].
pub(crate) fn decode(identity: &DriverIdentity, bytes: &[u8]) -> Option<HashMap<PixelFormat, u64>> {
    let body_len = bytes.len().checked_sub(32)?;
    if body_len < HEAD_BYTES || !(body_len - HEAD_BYTES).is_multiple_of(ANSWER_BYTES) {
        return None;
    }
    let (body, checksum) = bytes.split_at(body_len);
    let computed: [u8; 32] = Sha256::digest(body).into();
    if computed[..] != *checksum {
        return None;
    }
    let (magic, rest) = body.split_at(8);
    let (version, rest) = rest.split_at(4);
    let (runtime, rest) = rest.split_at(4);
    let (identity_digest, rest) = rest.split_at(32);
    let (count, rest) = rest.split_at(4);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC
        || word(version) != VERSION
        || word(runtime) != metal2vulkan::cache::RUNTIME_VERSION
        || identity_digest != digest(identity)
        || word(count) as usize != rest.len() / ANSWER_BYTES
    {
        return None;
    }
    let backed: HashMap<u32, PixelFormat> = backed_formats().map(|pf| (pf.as_raw(), pf)).collect();
    let mut answers = HashMap::new();
    for entry in rest.as_chunks::<ANSWER_BYTES>().0 {
        let (raw, answer) = entry.split_at(4);
        let pf = *backed.get(&word(raw))?;
        let answer = u64::from_le_bytes(answer.try_into().ok()?);
        let valid = answer.is_power_of_two() && (LINEAR_ALIGNMENT..=MOST).contains(&answer);
        if !valid || answers.insert(pf, answer).is_some() {
            return None;
        }
    }
    (answers.len() == backed.len()).then_some(answers)
}

impl Device {
    /// `minimumLinearTextureAlignmentForPixelFormat:`: the alignment of a texture's offset and
    /// `bytesPerRow` in a buffer at which this device's images hold a texture of `format` in place,
    /// at least the 16 bytes Metal answers on an Apple M3 Max and an M4 for every format a buffer
    /// can back.
    ///
    /// The runtime asks the device's driver at the first query of any format, for every format a
    /// buffer can back: it makes the images a 2D texture over host pages takes in place, at each
    /// power of two from 16 bytes to 4096 and at widths of 1 to 1920 texels, and answers the first
    /// at which every width's image lies at the texture's pitch and its memory alignment divides
    /// the answer. None is bound; every format took 17 to 23 ms on an Apple M3 Max, a Radeon 780M
    /// and lavapipe. Those answers stay the device's for its life, and a device of the same driver
    /// and runtime release reads them again from one record in the shader cache's content
    /// directory (0.3 ms there).
    ///
    /// It differs from Metal's where the device's images need more than 16 bytes: Mesa's RADV
    /// lays a linear image's rows at multiples of 256 bytes and aligns its memory to 256, and
    /// lavapipe aligns its memory to 64. It is 16 on a device that imports no host pages, for a
    /// format whose images lie at no such pitch, and where asking the driver failed.
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
    /// docs), for a format a buffer backs: every format's answers, worked out at the first query
    /// of any (the record's, else the probe's) and kept for the device's life.
    pub(crate) fn linear_texture_alignment(&self, pf: PixelFormat) -> u64 {
        let shared = self.shared();
        if shared.host_import.is_err() {
            return LINEAR_ALIGNMENT;
        }
        let answers = (shared.textures.linear_alignments).get_or_init(|| self.linear_alignments());
        answers.get(&pf).copied().unwrap_or(LINEAR_ALIGNMENT)
    }

    /// The largest of [`linear_texture_alignment`](Self::linear_texture_alignment)'s answers
    /// over the pixel formats a buffer backs: the device's own linear texture alignment.
    pub(crate) fn device_linear_texture_alignment(&self) -> u64 {
        backed_formats()
            .map(|pf| self.linear_texture_alignment(pf))
            .fold(LINEAR_ALIGNMENT, u64::max)
    }

    /// Every format's answer from the driver's record, else from the probe, which is recorded
    /// where every format's probe answered; Metal's 16 for a format whose probe failed.
    fn linear_alignments(&self) -> HashMap<PixelFormat, u64> {
        let shared = self.shared();
        let identity = self.driver_identity();
        let name = record_name(&identity);
        let store = shared.shader_cache.content_store();
        if let Lookup::Hit(answers) = store.load_content(&name, |bytes| decode(&identity, &bytes)) {
            shared
                .counts
                .linear_alignments_recalled(answers.len() as u64);
            return answers;
        }
        let mut answers = HashMap::new();
        let mut whole = true;
        for pf in backed_formats() {
            shared.counts.linear_alignment_probed();
            let answer = self
                .linear_texture_alignment_probed(pf)
                .unwrap_or_else(|_| {
                    whole = false;
                    LINEAR_ALIGNMENT
                });
            answers.insert(pf, answer);
        }
        if whole {
            // A failed write leaves the answers this device's alone.
            let _ = store.write_record_soon(&name, encode(&identity, &answers));
        }
        answers
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
