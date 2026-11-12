//! `minimumLinearTextureAlignmentForPixelFormat:` on the device: per pixel format, the alignment of
//! a texture's offset and rows in a buffer at which the device's images hold the texture in place.
//!
//! - **Metal's rule stays the rule.** The runtime makes a texture over a buffer at any offset and
//!   `bytesPerRow` that are multiples of 16 (`linear_plan.rs`'s `LINEAR_ALIGNMENT`), as Metal
//!   does, and answers no alignment below it. A texture under the answer but at 16 is made as
//!   before: an image in place where the device takes its layout, else a shadow, announced
//!   (`shadowed-texture`). Nothing is refused for lying under the answer.
//! - **The answer is where the device's images sit.** For a format and each usage of
//!   [`USAGES`] (reading; reading and rendering; reading and a shader's writes), the smallest
//!   power of two `P` from 16 to [`MOST`] at which a 2D texture of every width of [`WIDTHS`],
//!   [`ROWS`] rows tall (no multiple of 4, as a driver may render an explicit plane's rows in
//!   blocks of 4: `RuntimeDriverDefect::ExplicitPlaneRowsPastHeight`), of that usage, its rows
//!   `width × texel` bytes rounded up to `P` apart, gets the image a texture over host pages takes
//!   in place (`texture/in_place_image.rs`'s `in_place_unbound`: the driver's linear image made for
//!   host allocations where it lays the rows out at that pitch, as `vkGetImageSubresourceLayout`
//!   reports them, else the `DRM_FORMAT_MOD_LINEAR` image at it), and the image's memory
//!   alignment divides `P`, so that an offset that is a multiple of `P` meets it. The format's
//!   answer is the largest of its usages' (a usage no candidate brings in place, as one the
//!   format's linear images cannot take, has none). The runtime asks the driver and takes its
//!   answer: each image is made and released, never bound.
//! - **At least the device's line of memory.** The answer is at least `nonCoherentAtomSize`, "the
//!   size and alignment in bytes that bounds concurrent access to host-mapped device memory"
//!   (vkspec, Limits), the line the device's check of images in place measures
//!   (`in_place::RowEnds`): a texture at an offset and `bytesPerRow` of the answer whose rows'
//!   bytes are a multiple of the line shares no line with bytes beside its texels, so it is in
//!   place for its renders and a shader's writes too on a device whose stores write back a row's
//!   last line whole (the Iris Xe: its check fails those of a texture whose texels share a line
//!   with bytes beside them). Its rows' padding lies in their last line whatever the answer where
//!   their bytes are no multiple of the line (an `RGBA8Unorm` row of 100 texels, 400 bytes, at
//!   64-byte lines), so on such a device that texture is a shadow for its renders and writes at
//!   every alignment, Metal's 256 on an AMD GPU included: the answer promises no texture in place
//!   for every width.
//! - **Metal's 16 elsewhere**: on a device that imports no host pages, where no image is ever in
//!   place over them; for a format no candidate brings every width in place at (the device makes
//!   no linear image of it, or lays its rows at no power of two's pitch); and where a probe failed
//!   (a Vulkan error).
//! - **Every format at once, fixed for the device's life.** The first query of any format works
//!   out every format a buffer backs (a probe of all 175 at the three usages took 43 ms on
//!   lavapipe, 119 ms on a Radeon RX 550 and 124 ms on an Iris Xe in a debug build, and reading
//!   them again from the record under 1 ms), and those answers are the device's for good
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
use crate::linear_plan::{linear_alignment, linear_lack, BufferLayout, LINEAR_ALIGNMENT};
use crate::shader_cache::store::{hex, Lookup};
use crate::texture_descriptor::{
    texture_usage, TextureDescriptor, TextureLayout, TextureSwizzleChannels,
};
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

/// The usages the probe tries each format's textures at: read, rendered and written by a shader,
/// each beside reading, as a program's textures over a buffer are made.
pub(crate) const USAGES: [TextureUsage; 3] = [
    TextureUsage::SHADER_READ,
    TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
    TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
];

/// The version of what the probe tries and of how its record is named and framed: a change to
/// either bumps it, and a record of another version is never read.
pub(crate) const VERSION: u32 = 2;

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

/// A format's answer from its usages' smallest in-place alignments `usages` (`None` for one no
/// candidate brings in place) on a device whose line of memory is `line` bytes
/// (`nonCoherentAtomSize`): the largest of them, at least the line; Metal's 16 where no usage has
/// one, as a texture of the format is then never in place.
pub(crate) fn placed_answer(usages: &[Option<u64>], line: u64) -> u64 {
    match usages.iter().flatten().max() {
        Some(&placed) => placed.max(line).clamp(LINEAR_ALIGNMENT, MOST),
        None => LINEAR_ALIGNMENT,
    }
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
    /// buffer can back: it makes the images a 2D texture over host pages takes in place, read,
    /// rendered and written by a shader where the format's linear images take those usages, at
    /// each power of two from 16 bytes to 4096 and at widths of 1 to 1920 texels, and answers the
    /// first at which every width's image lies at the texture's pitch and its memory alignment
    /// divides the answer, for every usage that has one. None is bound. The answer is at least the
    /// device's `nonCoherentAtomSize`, "the size and alignment in bytes that bounds concurrent
    /// access to host-mapped device memory" (vkspec, Limits): at it, a texture whose rows' bytes
    /// fill whole lines of that size shares no line with bytes beside its texels, which a device
    /// whose stores write back a row's last line whole (the Iris Xe) needs of a texture it renders
    /// or writes in place. Those answers stay the device's for its life, and a device of the same
    /// driver and runtime release reads them again from one record in the shader cache's content
    /// directory.
    ///
    /// It differs from Metal's where the device's images or its lines need more than 16 bytes:
    /// Mesa's RADV lays a linear image's rows at multiples of 256 bytes and aligns its memory to
    /// 256, lavapipe aligns its memory to 64, and ANV's lines are 64 bytes. It is 16 on a device
    /// that imports no host pages, for a format whose images lie at no such pitch, and where
    /// asking the driver failed.
    ///
    /// The answer does not make every texture in place: a texture whose rows' bytes are no
    /// multiple of the device's line (an `RGBA8Unorm` row of 100 texels, 400 bytes, at 64-byte
    /// lines) keeps its padding in its rows' last line at every alignment, so on a device whose
    /// check of images in place fails such textures' renders and writes (the Iris Xe) it is a
    /// shadow for them, announced, whatever the alignment.
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
        Ok(self.current().linear_texture_alignment(format))
    }

    /// No Metal counterpart (a runtime extension): the `bytesPerRow` at which a texture of
    /// `descriptor` over a buffer of the runtime's own memory (`Buffer::new_texture`, not over the
    /// caller's pages) can be the device's own linear image over the buffer, made
    /// `PREINITIALIZED`, which takes no check of the device's: the row pitch the driver lays that
    /// image's rows at (`vkGetImageSubresourceLayout`'s `rowPitch`), which the texture's rows must
    /// meet exactly. The texture's offset must meet the image's memory alignment too
    /// (`BufferDescriptor::with_aligned_byte`), and rows with bytes between them hold those bytes
    /// in the image only over a buffer that declares them unobserved
    /// (`BufferDescriptor::with_padding_unobserved`) or, for a texture that renders nothing, one
    /// that only the device's commands write (a `Private` buffer, a `Managed` one on a device
    /// without unified memory): docs/API.md, `shadowed-texture`. `None` for a texture of more than
    /// one level, sample or slice, or of another type than 2D, and where the device makes no
    /// linear image of it that does all its optimal image does.
    ///
    /// It is the driver's pitch for that width, no alignment: a pitch rounded up to
    /// [`minimum_linear_texture_alignment_for_pixel_format`](Self::minimum_linear_texture_alignment_for_pixel_format)'s
    /// answer may miss it, as that answer is where images over host pages lie (where the driver
    /// lays no linear one at a pitch, a `DRM_FORMAT_MOD_LINEAR` image at it), and Metal's 16 on a
    /// device that imports no host pages. The image is made and released, never bound. Refused as
    /// `Buffer::new_texture` refuses the descriptor's checks and its image's plan.
    pub fn linear_texture_bytes_per_row(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<Option<u64>, Error> {
        const CALL: &str = "linear_texture_bytes_per_row";
        let layout = descriptor
            .check(self.properties().texture_rules())
            .map_err(|refused| refused.error(CALL))?;
        let one = layout.texture_type == TextureType::Type2D
            && (layout.levels, layout.layers, layout.samples) == (1, 1, 1);
        if !one {
            return Ok(None);
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let (_, plan) = self.declared_plan(CALL, descriptor, &layout, &facts)?;
        if linear_lack(&plan, &facts).is_some() {
            return Ok(None);
        }
        Ok(match self.linear_image(&plan)? {
            Ok(made) => {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(self.shared().vk(), made.raw) };
                Some(made.report.row_pitch)
            }
            Err(_) => None,
        })
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

    /// The format's answer (the module docs): the largest of its usages' smallest alignments at
    /// which every width of [`WIDTHS`] takes an image in place, at least the device's line of
    /// memory ([`placed_answer`]); 16 where no usage's does.
    fn linear_texture_alignment_probed(&self, pf: PixelFormat) -> Result<u64, Error> {
        let Some(texel) = texel_bytes(pf) else {
            return Ok(LINEAR_ALIGNMENT);
        };
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let mut usages = Vec::with_capacity(USAGES.len());
        for usage in USAGES {
            usages.push(self.smallest_in_place(pf, texel, usage, &facts)?);
        }
        Ok(placed_answer(&usages, self.shared().in_place.line()))
    }

    /// The smallest power of two from 16 to [`MOST`] at which every width of [`WIDTHS`] takes an
    /// image in place at `usage`; `None` where none does.
    fn smallest_in_place(
        &self,
        pf: PixelFormat,
        texel: u64,
        usage: TextureUsage,
        facts: &ImageFacts<'_>,
    ) -> Result<Option<u64>, Error> {
        let mut alignment = LINEAR_ALIGNMENT;
        while alignment <= MOST {
            let mut every = true;
            for width in WIDTHS {
                if !self.in_place_at(pf, (width, texel), usage, alignment, facts)? {
                    every = false;
                    break;
                }
            }
            if every {
                return Ok(Some(alignment));
            }
            alignment *= 2;
        }
        Ok(None)
    }

    /// Whether a 2D texture of `pf` and `usage`, `width` texels of `texel` bytes wide and
    /// [`ROWS`] tall, its rows its row's bytes rounded up to `alignment` apart, takes an image in
    /// place over host pages whose memory alignment divides `alignment`: the image is made and
    /// released.
    fn in_place_at(
        &self,
        pf: PixelFormat,
        (width, texel): (u64, u64),
        usage: TextureUsage,
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
        let usage = texture_usage(usage, TextureSwizzleChannels::DEFAULT);
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
