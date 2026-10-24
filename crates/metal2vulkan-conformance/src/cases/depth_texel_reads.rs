//! What a kernel gets back from a `Depth16Unorm` texel, for every one of the 65,536 codes, read
//! and sampled.
//!
//! The owned fixture `format_scout`'s `scout_read_d` reads texel `g` of a `depth2d<float>` into
//! `out[16 g.y + g.x]`, and its `scout_sample_d` takes a nearest sample at the texel's
//! `(g + (0.25, 0.75)) / 16`. The texture is 16 by 16, so the case walks the codes 256 at a time
//! over 256 pages, each page a texture of its own holding its 256 codes, and both accesses run
//! over every page.
//!
//! **Why this case exists, and what it adds to the lane it comes from.**
//! `runtime_lane::depth_texel_reads` holds one 64-digit constant: the SHA-256 of 65,536 `float`s.
//! Nothing in the tree holds the values. The lane's own test
//! (`the_rule_is_metals_answer`) recomputes `c / 65535` for every code and asserts the digest, so
//! the digest pins a **rule**, and the probe that tied the rule to Metal is gone. Two things
//! follow, and both are why the port was worth doing rather than transcribing the digest:
//!
//! - The values **are** recoverable, and this case recovers them: record mode runs the two
//!   accesses on native Metal and the recording holds all 131,072 `float`s. Where the lane could
//!   only say "the digest differs", the case names the code — `element [row 0, 44731]` is code
//!   44,731 — and says which access moved and by how many units in the last place.
//! - A rule a digest pins is only as good as the digest. The recording replaces the rule with the
//!   measurement, so a release that changes Metal's conversion shows as the codes that moved
//!   rather than as one hex string that no longer matches anything.
//!
//! Every code compares exactly, and the two accesses are separate sites because they are separate
//! questions: a device may fetch a depth texel one way and filter it another.
//!
//! Nothing here can fault a GPU: one compute pass an access a page, a 16 by 16 grid over a 16 by
//! 16 texture, one thread a texel, no loop and no sampling outside the texture.

use crate::allowance::Allowance;
use crate::case::{
    made, Abort, Answer, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
    TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The fixture both kernels live in.
const FIXTURE: &str = "format_scout";

/// The kernel that reads texel `g`.
const READ: &str = "scout_read_d";

/// The kernel that takes a nearest sample inside texel `g`.
const SAMPLE: &str = "scout_sample_d";

/// The texture's side, which the kernels' `16 g.y + g.x` fixes.
const SIDE: u32 = 16;

/// The codes one page holds.
const PAGE: usize = (SIDE * SIDE) as usize;

/// The codes a 16-bit format has.
const CODES: usize = 65_536;

/// The pages the codes take.
const PAGES: usize = CODES / PAGE;

/// The byte every output word holds before the pass, so a word the kernel did not write is a
/// recorded value rather than whatever was there.
const BLANK: u8 = 0xab;

/// The accesses the case compares, by the site each writes.
const ACCESSES: [(&str, &str); 2] = [("read", READ), ("nearest-sample", SAMPLE)];

/// The codes page `page` holds, as the texture's bytes: 256 little-endian 16-bit codes, row major,
/// so texel `i` of the page holds code `256 page + i` and the kernel writes it to `out[i]`.
fn page_bytes(page: usize) -> Vec<u8> {
    (0..PAGE)
        .flat_map(|index| ((page * PAGE + index) as u16).to_le_bytes())
        .collect()
}

/// Every page's bytes, in page order: the case's whole input over the texture.
fn all_page_bytes() -> Vec<u8> {
    (0..PAGES).flat_map(page_bytes).collect()
}

/// One page's output buffer before the pass.
fn blank_output() -> Vec<u8> {
    vec![BLANK; PAGE * 4]
}

/// The layout the codes' answers are read under: one row of 65,536 `float`s, so the element a diff
/// names **is** the code.
fn layout() -> ByteLayout {
    let length = (CODES * 4) as u32;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(length),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: CODES as u32,
        row_pitch: length,
        rows: 1,
        image_pitch: length,
        images: 1,
        // The row is the whole value, so there is no padding to compare.
        gaps: Gaps::Ignored,
    }
}

/// A call every page makes, whose answer the transcript holds no site of its own for.
///
/// 256 pages times three calls an access is more than 1,500 sites that all say `Ok`, which would
/// bury the answer the case is about. A failure still emits its site, names the page and the
/// access, and stops the run, so nothing goes unrecorded: what is left out is the successful
/// answer of a call whose success the read-back already proves.
fn per_page<T>(sites: &mut Sites, at: &str, answer: Answer<T>) -> Result<T, Abort> {
    match answer {
        Ok(value) => Ok(value),
        Err(failure) => {
            sites.push(Node::leaf(at, Observed::value(failure.recorded())));
            Err(Abort {
                at: at.to_owned(),
                failure,
            })
        }
    }
}

/// The case: both accesses over every code.
pub struct DepthTexelReads;

/// The one case of the family.
pub fn all() -> Vec<DepthTexelReads> {
    vec![DepthTexelReads]
}

impl DepthTexelReads {
    /// The fixture under the kernel `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }

    /// The texture one page's codes go in: 16 by 16 `Depth16Unorm`, shader-read, two bytes a texel.
    ///
    /// Shader-read and nothing else, because Metal holds no depth texture of shader-write usage:
    /// the role is part of what the case means, which is why a case states it rather than letting
    /// either side pick.
    fn spec() -> TextureSpec {
        TextureSpec::new(
            PixelFormat::Depth16Unorm,
            [SIDE, SIDE],
            SIDE * 2,
            TextureRole::Read,
        )
    }

    /// Run one access over every page and answer its 65,536 words.
    fn sweep(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        access: &str,
        pipeline: Handle,
        sampler: Option<Handle>,
    ) -> Result<Vec<u8>, Abort> {
        let samplers: Vec<Handle> = sampler.into_iter().collect();
        let mut words = Vec::with_capacity(CODES * 4);
        for page in 0..PAGES {
            let at = |call: &str| format!("{access}/page/{page}/{call}");
            let texture = per_page(
                sites,
                &at("texture"),
                side.texture_2d_in_role(&Self::spec(), &page_bytes(page)),
            )?;
            let output = per_page(sites, &at("buffer"), side.buffer(&blank_output()))?;
            let command_buffer = per_page(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::new(&[texture], &[output]).sampled(&samplers),
                    &[],
                    [SIDE, SIDE, 1],
                    [SIDE, SIDE, 1],
                ),
            )?;
            let status = per_page(sites, &at("status"), side.wait(command_buffer))?;
            if status != Value::unit_variant("Completed") {
                sites.push(Node::leaf(at("status"), Observed::value(status.clone())));
                return Err(Abort {
                    at: at("status"),
                    failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
                });
            }
            let read = per_page(
                sites,
                &at("read-back"),
                side.read_buffer(output, 0, PAGE * 4),
            )?;
            words.extend(read);
        }
        Ok(words)
    }
}

impl Case for DepthTexelReads {
    fn name(&self) -> &'static str {
        "depth16-texel-reads"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The fixture's AIR asks for fast math, and a device whose compiler rewrites the
        // arithmetic it allows announces that at the pipeline's making. Only the notice set is
        // loosened: every one of the 131,072 words still compares exactly, so a device that
        // rewrites nothing observable reads as announced with nothing differing, and one that
        // changes a code shows the code. The allowance fails the case if the notice stops firing,
        // so it cannot outlive the reason for it.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // this case, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::FastMathRewrites.code()),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (_, entry) in ACCESSES {
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
        let fixture = Self::fixture(READ);
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // Every page's codes, in page order: the whole sweep, so a generator that changed one
            // page makes the case unrecorded rather than comparing against other codes.
            Input::bytes("textures:codes", &all_page_bytes()),
            Input::bytes("buffer:blank", &blank_output()),
            // The format and the sampler state the case names: a different format or a different
            // filter is a different question.
            Input::bytes("format", &PixelFormat::Depth16Unorm.as_raw().to_le_bytes()),
            Input::bytes("sampler", &SamplerSpec::new().words()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = Self::fixture(READ)
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let read = made(sites, "pipeline/read", side.compute_pipeline(library, READ))?;
        let sample = made(
            sites,
            "pipeline/nearest-sample",
            side.compute_pipeline(library, SAMPLE),
        )?;
        // `MTLSamplerDescriptor`'s defaults, which are the nearest, clamp-to-edge, normalized
        // state the lane's probe sampled through.
        let sampler = made(sites, "sampler", side.sampler(&SamplerSpec::new()))?;

        let mut output = Node::branch("output", Vec::new());
        for (access, entry) in ACCESSES {
            let pipeline = if entry == READ { read } else { sample };
            let sampler = (entry == SAMPLE).then_some(sampler);
            let words = self.sweep(side, sites, access, pipeline, sampler)?;
            output = output.with(Node::leaf(access, Observed::bytes(words, layout())));
        }
        sites.push(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every code occurs exactly once over the pages, in code order: the sweep is exhaustive over
    /// the format's whole input space, which is what makes an exact comparison worth taking.
    #[test]
    fn the_pages_hold_every_code_once_in_order() {
        let bytes = all_page_bytes();
        assert_eq!(bytes.len(), CODES * 2);
        let codes: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        assert_eq!(codes.len(), CODES);
        for (at, code) in codes.iter().enumerate() {
            assert_eq!(usize::from(*code), at, "code at {at}");
        }
        let distinct: BTreeSet<u16> = codes.iter().copied().collect();
        assert_eq!(distinct.len(), CODES);
    }

    /// The layout covers every byte of the read-back and names the code as its element index, so a
    /// diff over the answers says which code moved.
    #[test]
    fn the_layout_names_the_code() {
        let layout = layout();
        assert_eq!(layout.extent(), CODES * 4);
        assert_eq!(layout.element, Element::F32);
        for code in [0usize, 1, 512, 44_731, CODES - 1] {
            assert_eq!(layout.at(0, 0, code as u32), code * 4, "code {code}");
        }
    }

    /// The rule the lane holds in place of Metal's values — every code as `c / 65535` correctly
    /// rounded — hashes to the lane's own constant. The case is recorded from Metal rather than
    /// from this, and this test is what lets the report say whether the two agree.
    #[test]
    fn the_lanes_rule_hashes_to_the_lanes_constant() {
        let bytes: Vec<u8> = (0..CODES)
            .flat_map(|code| (code as f32 / 65535.0).to_le_bytes())
            .collect();
        assert_eq!(
            crate::sha256_hex(&bytes),
            "a940e05b402805a0f114a2009566daa556ac9cc732c04127d1cfaf7d98c13b0d"
        );
    }
}
