//! Kernel writes and reads of a texture of Apple's two-plane extended-range format.
//!
//! The owned fixture `xr_two_planes` writes a buffer's `half4` or `float4` values to a
//! `texture2d<half>`/`<float>` of `RGB10A8_2P_XR10` (or its sRGB twin) through `access::write`, and
//! reads the texture back through `access::read`, one thread per texel of a grid whose rows are `w`
//! texels apart, under 32 by 32 threadgroups. One case per format, each holding six outputs:
//!
//! - `half-written/*`: a 256 by 256 texture written with every `half` pattern in every channel
//!   (lane `c` of texel `n` holds `(n + 16384 c) mod 65536`), read back as `half4` and as `float4`;
//! - `float-written/*`: the same texture written with a dense `float` grid
//!   (`((n + 16384 c) mod 65536 - 32768) / 16384`), read back both ways;
//! - `consumer-shape/*`: the consumer's shape, a 64 by 4 texture under 2 by 1 threadgroups of
//!   32 by 32 — a 64 by 32 grid whose rows 4 to 31 are outside the texture — written with
//!   `0x3000 + n + 2048 c` and read back both ways.
//!
//! **What the port settles, and where it stops.** `runtime_lane::extended_range_two_planes` states
//! Metal's answer for these six outputs as six SHA-256 constants and nothing else, from a probe
//! whose numbers are gone. Two different things were in the way of recovering them, and only one of
//! them turned out to be real:
//!
//! - *Not* the format. The two-plane extended-range formats are not in Apple's public
//!   `MTLPixelFormat` list, and an earlier batch took that to mean the Metal side has no public
//!   call for them. It does: `MTLTextureDescriptor`'s pixel format is a raw value, and native Metal
//!   makes, writes and reads back a texture of 550 and 551 through entirely public calls. Record
//!   mode runs all six outputs on Metal, so every digest here is the oracle chip's own, taken by
//!   running the case.
//! - The size. The four 256 by 256 outputs are 65,536 threads of a `half4` or a `float4`, which is
//!   6 MiB of Metal's bytes across the two formats — more than belongs in a committed recording.
//!   Those four keep the digest, with a **window** of the first and the last 256 threads recorded
//!   whole beside it, so a failure at either end of the grid names the thread and the lane instead
//!   of only saying that the digest differs. The consumer's shape is 2,048 threads, so it is
//!   recorded whole.
//!
//! So the honest accounting is: four outputs are a digest plus a window and remain less informative
//! than raw values would be, and two are raw. All six are now Metal's measured answer on the chip
//! the suite names, rather than a digest from a probe on another machine.
//!
//! Every output compares exactly, or by its digest where the digest is all that is recorded, and
//! the case carries **no allowance**: on both venues this round covers, MoltenVK and lavapipe, the
//! runtime announces nothing at all here. A venue that does announce something will read as
//! divergent until whoever adds it names the notice with a run behind it, which is the way round
//! the suite's default should fail.
//!
//! Nothing here can fault a GPU: compute passes of one kernel each, one texture and two buffers, no
//! loop and no sampling. The consumer's grid does reach past the texture, which is what the case is
//! about and what Metal defines: a write outside is dropped and a read outside answers zero.

use crate::case::{
    made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, Side, TextureRole,
    TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;

/// The fixture all four kernels live in.
const FIXTURE: &str = "xr_two_planes";

/// The kernel that writes `half4`s.
const WRITE_HALF: &str = "xr2p_write_half";
/// The kernel that writes `float4`s.
const WRITE_FLOAT: &str = "xr2p_write_float";
/// The kernel that reads `half4`s.
const READ_HALF: &str = "xr2p_read_half";
/// The kernel that reads `float4`s.
const READ_FLOAT: &str = "xr2p_read_float";

/// Every entry point the case makes a pipeline of.
const ENTRIES: [&str; 4] = [WRITE_HALF, WRITE_FLOAT, READ_HALF, READ_FLOAT];

/// The square texture's side.
const SIDE: u32 = 256;

/// The square texture's texels, one thread each.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The consumer's texture: 64 by 4 texels.
const CONSUMER_WIDTH: u32 = 64;
const CONSUMER_HEIGHT: u32 = 4;

/// The consumer's grid: 2 by 1 threadgroups of 32 by 32, so 64 by 32 threads, of which rows 4 to
/// 31 are outside the texture.
const CONSUMER_GRID_WIDTH: u32 = 64;
const CONSUMER_THREADS: usize = 64 * 32;

/// The threads in each dimension of a threadgroup, as the consumer dispatches them.
const GROUP: u32 = 32;

/// A texel of a two-plane format: plane 0's word and plane 1's byte, which Metal lays out as the
/// format's own bytes. Four bytes a texel of each plane in the images the kernels write.
const TEXEL_BYTES: u32 = 4;

/// The threads a window records whole at each end of a large output.
const WINDOW: usize = 256;

/// The byte every output word holds before a read, so a word the kernel did not write is a
/// recorded value rather than whatever was there.
const BLANK: u8 = 0xab;

/// `n` threads' `half4`s: lane `c` of thread `t` holds `pattern(t, c)`'s low sixteen bits.
fn halves(n: usize, pattern: impl Fn(usize, usize) -> usize) -> Vec<u8> {
    (0..n)
        .flat_map(|thread| (0..4).map(move |lane| (thread, lane)))
        .flat_map(|(thread, lane)| ((pattern(thread, lane) & 0xffff) as u16).to_le_bytes())
        .collect()
}

/// `n` threads' `float4`s: lane `c` of thread `t` holds `(k mod 65536 - 32768) / 16384` of
/// `k = t + 16384 c`, a dense grid across the format's range.
fn floats(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|thread| (0..4).map(move |lane| (thread, lane)))
        .flat_map(|(thread, lane)| {
            let k = (thread + 16384 * lane) & 0xffff;
            ((k as f32 - 32768.0) / 16384.0).to_le_bytes()
        })
        .collect()
}

/// What one read-back's results are: the lanes' width, which is what the diff reads them as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lanes {
    /// `half4`: four 16-bit floats a thread.
    Half,
    /// `float4`: four 32-bit floats a thread.
    Float,
}

impl Lanes {
    /// The kernel that reads this way.
    fn entry(self) -> &'static str {
        match self {
            Lanes::Half => READ_HALF,
            Lanes::Float => READ_FLOAT,
        }
    }

    /// The site's name.
    fn name(self) -> &'static str {
        match self {
            Lanes::Half => "half-lanes",
            Lanes::Float => "float-lanes",
        }
    }

    /// One element of a thread's four.
    fn element(self) -> Element {
        match self {
            Lanes::Half => Element::F16,
            Lanes::Float => Element::F32,
        }
    }

    /// Bytes one thread's four lanes take.
    fn stride(self) -> u32 {
        4 * self.element().size()
    }
}

/// The layout `threads` threads' lanes are read under, from `offset` bytes into the output buffer
/// whose whole length is `length`: one row a thread of four lanes, so a diff names the thread and
/// the lane rather than a byte offset.
fn layout(lanes: Lanes, threads: u32, offset: u64, length: u64) -> ByteLayout {
    let stride = lanes.stride();
    ByteLayout {
        region: Region::Buffer { offset, length },
        element: lanes.element(),
        row_offset: 0,
        row_elements: 4,
        row_pitch: stride,
        rows: threads,
        image_pitch: stride * threads,
        images: 1,
        // Each thread's four lanes fill its row, so there is no padding between them.
        gaps: Gaps::Ignored,
    }
}

/// How a dispatch covers its texture: the grid's row width, which the kernels read at buffer 1,
/// the grid, and the threadgroups where the case dispatches threadgroups rather than threads.
///
/// One value rather than three arguments, so a write and the reads that follow it cannot drift
/// apart on the shape they cover.
#[derive(Clone, Copy)]
struct Shape {
    grid_width: u32,
    grid: [u32; 3],
    /// `Some` where the case dispatches threadgroups, so the grid reaches past the texture, as the
    /// consumer's shape does; `None` where it dispatches one thread a texel.
    threadgroups: Option<[u32; 3]>,
}

/// What one read-back reads: the texture, the lanes it reads them as, the grid's threads and the
/// shape the grid covers.
#[derive(Clone, Copy)]
struct Read {
    texture: Handle,
    lanes: Lanes,
    threads: usize,
    shape: Shape,
}

/// What one dispatch runs: the kernel, the texture, the buffer 0 values, and the shape.
struct Over<'a> {
    pipeline: Handle,
    texture: Handle,
    values: &'a [u8],
    shape: Shape,
}

/// A case: one two-plane format, its six outputs.
pub struct ExtendedRangeTwoPlanes {
    name: &'static str,
    format: PixelFormat,
}

/// Both formats, as the lane sweeps them.
pub fn all() -> Vec<ExtendedRangeTwoPlanes> {
    vec![
        ExtendedRangeTwoPlanes {
            name: "xr-two-planes-rgb10a8-2p-xr10",
            format: PixelFormat::RGB10A82PXR10,
        },
        ExtendedRangeTwoPlanes {
            name: "xr-two-planes-rgb10a8-2p-xr10-srgb",
            format: PixelFormat::RGB10A82PXR10SRGB,
        },
    ]
}

/// The kernels' pipelines, by entry point, in the order [`ENTRIES`] names them.
struct Pipelines([Handle; 4]);

impl Pipelines {
    /// The pipeline of `entry`.
    fn of(&self, entry: &str) -> Handle {
        let at = ENTRIES
            .iter()
            .position(|named| *named == entry)
            .expect("an entry point the case made a pipeline of");
        self.0[at]
    }
}

impl ExtendedRangeTwoPlanes {
    /// The fixture under the kernel `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }

    /// A texture of the case's format, `size` texels, that the kernels read and write.
    ///
    /// Made with no bytes: every texel the case reads is one a kernel wrote first, so a
    /// `replaceRegion` of the planes would be a second question and not this one.
    fn spec(&self, size: [u32; 2]) -> TextureSpec {
        TextureSpec::new(
            self.format,
            size,
            size[0] * TEXEL_BYTES,
            TextureRole::ReadWrite,
        )
    }

    /// Run one kernel over `texture`, with `values` at buffer 0 and the grid's row width at
    /// buffer 1, and wait for it.
    fn run(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        over: Over<'_>,
    ) -> Result<Handle, Abort> {
        let Over {
            pipeline,
            texture,
            values,
            shape:
                Shape {
                    grid_width,
                    grid,
                    threadgroups,
                },
        } = over;
        let buffer = made(sites, &format!("{at}/buffer"), side.buffer(values))?;
        let width = made(
            sites,
            &format!("{at}/width"),
            side.buffer(&grid_width.to_le_bytes()),
        )?;
        let textures = [texture];
        let buffers = [buffer, width];
        let bindings = Bindings::new(&textures, &buffers);
        let command_buffer = match threadgroups {
            // The consumer dispatches threadgroups, so the grid reaches past the texture; the
            // square cases dispatch threads, one a texel.
            Some(groups) => made(
                sites,
                &format!("{at}/dispatch"),
                side.dispatch(pipeline, &bindings, groups, [GROUP, GROUP, 1]),
            )?,
            None => made(
                sites,
                &format!("{at}/dispatch"),
                side.dispatch_threads(pipeline, &bindings, &[], grid, [GROUP, GROUP, 1]),
            )?,
        };
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
        Ok(buffer)
    }

    /// Read `texture` back `lanes`-wise over `threads` threads, and answer the bytes.
    fn read_back(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        pipelines: &Pipelines,
        read: Read,
    ) -> Result<Vec<u8>, Abort> {
        let Read {
            texture,
            lanes,
            threads,
            shape,
        } = read;
        let length = threads * lanes.stride() as usize;
        let blank = vec![BLANK; length];
        let output = self.run(
            side,
            sites,
            at,
            Over {
                pipeline: pipelines.of(lanes.entry()),
                texture,
                values: &blank,
                shape,
            },
        )?;
        made(
            sites,
            &format!("{at}/read-back"),
            side.read_buffer(output, 0, length),
        )
        .and_then(|bytes| {
            if bytes.len() == length {
                Ok(bytes)
            } else {
                Err(Abort {
                    at: format!("{at}/read-back"),
                    failure: Failure::named(
                        "ShortReadBack",
                        format!("{} bytes of {length}", bytes.len()),
                    ),
                })
            }
        })
    }

    /// The node a large output emits: the digest of every byte Metal wrote, and a window of the
    /// first and the last [`WINDOW`] threads recorded whole beside it.
    fn windowed_node(&self, name: &str, lanes: Lanes, threads: usize, bytes: Vec<u8>) -> Node {
        let stride = lanes.stride() as usize;
        let length = bytes.len() as u64;
        let tail = (threads - WINDOW) * stride;
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "digest",
                Observed::absent().digested(Digest::sha256(&bytes)),
            ))
            .with(Node::leaf(
                "first-threads",
                Observed::bytes(
                    bytes[..WINDOW * stride].to_vec(),
                    layout(lanes, WINDOW as u32, 0, length),
                ),
            ))
            .with(Node::leaf(
                "last-threads",
                Observed::bytes(
                    bytes[tail..].to_vec(),
                    layout(lanes, WINDOW as u32, tail as u64, length),
                ),
            ))
    }

    /// The node a whole output emits: every thread's lanes raw, and the digest beside them as a
    /// projection of the value.
    fn whole_node(&self, name: &str, lanes: Lanes, threads: usize, bytes: Vec<u8>) -> Node {
        let length = bytes.len() as u64;
        Node::leaf(
            name,
            Observed::bytes(bytes.clone(), layout(lanes, threads as u32, 0, length))
                .digested(Digest::sha256(&bytes)),
        )
    }
}

impl Case for ExtendedRangeTwoPlanes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone: every value of this family still compares against Metal's exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09), where it fired for all 2 rows of the family
        // and each row was divergent on that field and nothing else (replay-nvidia.log).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in ENTRIES {
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
        let fixture = Self::fixture(WRITE_HALF);
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The format the case is of: the sRGB twin is a different question.
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("values:half", &halves(TEXELS, |n, c| n + 16384 * c)),
            Input::bytes("values:float", &floats(TEXELS)),
            Input::bytes(
                "values:consumer",
                &halves(CONSUMER_THREADS, |n, c| 0x3000 + n + 2048 * c),
            ),
            // What an output buffer holds before a read.
            Input::bytes("buffer:blank", &[BLANK]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = Self::fixture(WRITE_HALF)
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let mut made_pipelines = Vec::new();
        for entry in ENTRIES {
            made_pipelines.push(made(
                sites,
                &format!("pipeline/{entry}"),
                side.compute_pipeline(library, entry),
            )?);
        }
        let pipelines = Pipelines([
            made_pipelines[0],
            made_pipelines[1],
            made_pipelines[2],
            made_pipelines[3],
        ]);

        let mut output = Node::branch("output", Vec::new());

        // The square texture, written each way and read back both ways after each write.
        let square = made(
            sites,
            "texture/square",
            side.texture_2d_in_role(&self.spec([SIDE, SIDE]), &[]),
        )?;
        let square_shape = Shape {
            grid_width: SIDE,
            grid: [SIDE, SIDE, 1],
            threadgroups: None,
        };
        for (step, entry, values) in [
            (
                "half-written",
                WRITE_HALF,
                halves(TEXELS, |n, c| n + 16384 * c),
            ),
            ("float-written", WRITE_FLOAT, floats(TEXELS)),
        ] {
            self.run(
                side,
                sites,
                &format!("{step}/write"),
                Over {
                    pipeline: pipelines.of(entry),
                    texture: square,
                    values: &values,
                    shape: square_shape,
                },
            )?;
            let mut node = Node::branch(step, Vec::new());
            for lanes in [Lanes::Half, Lanes::Float] {
                let bytes = self.read_back(
                    side,
                    sites,
                    &format!("{step}/{}", lanes.name()),
                    &pipelines,
                    Read {
                        texture: square,
                        lanes,
                        threads: TEXELS,
                        shape: square_shape,
                    },
                )?;
                node = node.with(self.windowed_node(lanes.name(), lanes, TEXELS, bytes));
            }
            output = output.with(node);
        }

        // The consumer's shape: a 64 by 4 texture under a 64 by 32 grid.
        let consumer = made(
            sites,
            "texture/consumer",
            side.texture_2d_in_role(&self.spec([CONSUMER_WIDTH, CONSUMER_HEIGHT]), &[]),
        )?;
        let consumer_shape = Shape {
            grid_width: CONSUMER_GRID_WIDTH,
            grid: [CONSUMER_GRID_WIDTH, GROUP, 1],
            threadgroups: Some([2, 1, 1]),
        };
        let consumer_values = halves(CONSUMER_THREADS, |n, c| 0x3000 + n + 2048 * c);
        self.run(
            side,
            sites,
            "consumer-shape/write",
            Over {
                pipeline: pipelines.of(WRITE_HALF),
                texture: consumer,
                values: &consumer_values,
                shape: consumer_shape,
            },
        )?;
        let mut node = Node::branch("consumer-shape", Vec::new());
        for lanes in [Lanes::Half, Lanes::Float] {
            let bytes = self.read_back(
                side,
                sites,
                &format!("consumer-shape/{}", lanes.name()),
                &pipelines,
                Read {
                    texture: consumer,
                    lanes,
                    threads: CONSUMER_THREADS,
                    shape: consumer_shape,
                },
            )?;
            node = node.with(self.whole_node(lanes.name(), lanes, CONSUMER_THREADS, bytes));
        }
        output = output.with(node);

        sites.push(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The `half` sweep writes every one of the 65,536 patterns in every channel, which is what
    /// makes it exhaustive over the format's `half` input space.
    #[test]
    fn the_half_sweep_writes_every_pattern_in_every_channel() {
        let bytes = halves(TEXELS, |n, c| n + 16384 * c);
        assert_eq!(bytes.len(), TEXELS * 8);
        for lane in 0..4usize {
            let seen: BTreeSet<u16> = (0..TEXELS)
                .map(|thread| {
                    let at = (thread * 4 + lane) * 2;
                    u16::from_le_bytes([bytes[at], bytes[at + 1]])
                })
                .collect();
            assert_eq!(seen.len(), 65_536, "lane {lane}");
        }
    }

    /// Each window lies inside its output and the two do not overlap, so between them they record
    /// 512 distinct threads of the 65,536.
    #[test]
    fn the_windows_lie_inside_the_output_and_apart() {
        for lanes in [Lanes::Half, Lanes::Float] {
            let stride = lanes.stride() as usize;
            let length = TEXELS * stride;
            let tail = (TEXELS - WINDOW) * stride;
            assert!(WINDOW * stride <= tail, "{lanes:?}");
            assert_eq!(tail + WINDOW * stride, length, "{lanes:?}");
            let head = layout(lanes, WINDOW as u32, 0, length as u64);
            assert_eq!(head.extent(), WINDOW * stride, "{lanes:?}");
        }
    }

    /// The two cases name distinct formats, so neither can stand for the other's recording.
    #[test]
    fn the_cases_name_the_two_formats_apart() {
        let formats: BTreeSet<u32> = all().iter().map(|case| case.format.as_raw()).collect();
        assert_eq!(formats.len(), 2);
        assert!(formats.contains(&550) && formats.contains(&551));
    }

    /// The consumer's grid reaches past the texture, which is the fact that shape is here for.
    #[test]
    fn the_consumers_grid_reaches_past_its_texture() {
        assert_eq!(CONSUMER_THREADS, (CONSUMER_GRID_WIDTH * GROUP) as usize);
        let rows_past = GROUP.saturating_sub(CONSUMER_HEIGHT);
        assert_eq!(rows_past, 28, "the grid's rows past the texture");
        assert_eq!(
            CONSUMER_THREADS,
            (CONSUMER_WIDTH * CONSUMER_HEIGHT) as usize * 8
        );
    }
}
