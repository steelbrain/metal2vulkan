//! A swizzled texture read and written through a **storage** slot, ported from
//! `runtime_lane::swizzled_storage`.
//!
//! What plain Metal does (the lane: an Apple M4 on macOS 27.0.1, plain twice alike) with a 2 by 2
//! texture whose every texel is `(10, 20, 30, 40)` in bytes — `RG8Unorm` `(10, 20)`, `R32Float`
//! 7.5, `R32Uint` 1000 — either **viewed** with a swizzle or **made** with one:
//!
//! - an `access::read` or `read_write` read answers the texel swizzled: each channel the source its
//!   swizzle names, `Zero` 0 and `One` 1, a format's missing green and blue 0 and its missing alpha
//!   1;
//! - an `access::write` or `read_write` write of `v` stores channel `c` from the **first** lane
//!   whose source is `c`, and lane `c` where no lane names it — the swizzle undone;
//! - `atomic_fetch_add` adds to red and answers the texel **unswizzled**.
//!
//! **The port re-measures it on this M3 Max**, because the lane's one row is the M4's and that
//! machine is gone. Record mode runs the case and the recordings hold Metal's own bytes; the three
//! rules above are committed cross-checks over them (`every_recorded_read_is_the_lanes_rule`,
//! `every_recorded_store_is_the_lanes_rule`, `every_recorded_atomic_ignores_the_swizzle`).
//!
//! The kernels are the owned fixture `rrp_swizzled_storage`'s, all binding the texture at
//! `[[texture(0)]]`: `rrp_sw_read_f` (`access::read`), `rrp_sw_rwread_f` and `rrp_sw_rwread_u`
//! (`read_write`) store texel (0, 0) to buffer 0; `rrp_sw_write_f`, `rrp_sw_rwwrite_f` and
//! `rrp_sw_write_u` write buffer 1's value to each thread's texel of a 2 by 2 grid;
//! `rrp_sw_atomic_u` adds 5 to texel (0, 0) and stores what it held.
//!
//! **A write kernel declares buffer 1 and no buffer 0**, and [`Bindings`] binds buffers from index 0
//! up, so such a dispatch binds the value buffer at both 0 and 1 and the kernel reads index 1. A
//! bound buffer no slot of the pipeline names is ordinary Metal, and this keeps the shared
//! [`Bindings`] vocabulary as it is rather than growing an index per buffer for one family.
//!
//! Every sub-run writes the probe's texel into the texture again before it, because a write kernel
//! leaves the texture holding its own value and the next kernel's answer is of the probe's texel.
//! That ordering is part of what the case means, so it is a stated [`Side::replace_region`] rather
//! than something a side arranges.
//!
//! One case per format, with the two made-with-a-swizzle rows and the atomics as cases of their own,
//! so a venue that holds no storage texture of a format answers its typed refusal at that row.
//!
//! Nothing here can fault a GPU: one compute pass a dispatch, at most four threads, no loop and no
//! sampling.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureUsage};

/// The fixture every kernel lives in.
const FIXTURE: &str = "rrp_swizzled_storage";

/// The texture's side: 2 by 2 texels, as the probe made it.
pub const SIDE: u32 = 2;

/// The byte the output buffer holds before a dispatch, so a word the kernel did not write is a
/// recorded value.
const BLANK: u8 = 0xab;

/// A format the probe measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Rgba8,
    Rg8,
    R32Float,
    R32Uint,
}

impl Format {
    /// The pixel format.
    pub fn pixel_format(self) -> PixelFormat {
        match self {
            Format::Rgba8 => PixelFormat::RGBA8Unorm,
            Format::Rg8 => PixelFormat::RG8Unorm,
            Format::R32Float => PixelFormat::R32Float,
            Format::R32Uint => PixelFormat::R32Uint,
        }
    }

    /// Bytes a texel.
    pub fn bytes(self) -> usize {
        match self {
            Format::Rg8 => 2,
            _ => 4,
        }
    }

    /// The channels the format stores, which are the ones a read-back holds.
    pub fn channels(self) -> usize {
        match self {
            Format::Rgba8 => 4,
            Format::Rg8 => 2,
            _ => 1,
        }
    }

    /// Whether a channel is an 8-bit normalized code rather than a whole word.
    pub fn normalized(self) -> bool {
        matches!(self, Format::Rgba8 | Format::Rg8)
    }

    /// The probe's texel, as its bytes.
    pub fn texel_bytes(self) -> Vec<u8> {
        match self {
            Format::Rgba8 => vec![10, 20, 30, 40],
            Format::Rg8 => vec![10, 20],
            Format::R32Float => 7.5f32.to_le_bytes().to_vec(),
            Format::R32Uint => 1000u32.to_le_bytes().to_vec(),
        }
    }

    /// The probe's texel as Metal reads it unswizzled, and `One`'s value: an 8-bit normalized
    /// format's codes (`One` 255), a float's bits (`One` 1.0), an integer's word (`One` 1), a
    /// missing green or blue 0 and a missing alpha the constant.
    pub fn texel(self) -> ([u32; 4], u32) {
        match self {
            Format::Rgba8 => ([10, 20, 30, 40], 255),
            Format::Rg8 => ([10, 20, 0, 255], 255),
            Format::R32Float => {
                let one = 1.0f32.to_bits();
                ([7.5f32.to_bits(), 0, 0, one], one)
            }
            Format::R32Uint => ([1000, 0, 0, 1], 1),
        }
    }

    /// Every texel of the texture, which is the bytes it is loaded with and reset to.
    pub fn level(self) -> Vec<u8> {
        self.texel_bytes().repeat((SIDE * SIDE) as usize)
    }

    /// Bytes a row of the level.
    pub fn row_pitch(self) -> u32 {
        SIDE * self.bytes() as u32
    }

    /// The usage the parent texture is made with: shader-read, shader-write and viewable, and
    /// atomic where the format's atomics are what a case asks about.
    fn parent_usage(self) -> TextureUsage {
        let usage = TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::PIXEL_FORMAT_VIEW;
        match self {
            Format::R32Uint => usage | TextureUsage::SHADER_ATOMIC,
            _ => usage,
        }
    }

    /// The kernels whose read of texel (0, 0) a case records: both float accesses, or the one
    /// `read_write` integer access the fixture holds.
    pub fn read_kernels(self) -> &'static [&'static str] {
        match self {
            Format::R32Uint => &["rrp_sw_rwread_u"],
            _ => &["rrp_sw_read_f", "rrp_sw_rwread_f"],
        }
    }

    /// The kernels whose write a case records, each with the four lanes it writes: an 8-bit
    /// normalized format's lanes are codes, a float's and an integer's are words.
    pub fn write_kernels(self) -> &'static [(&'static str, [u32; 4])] {
        match self {
            Format::Rgba8 | Format::Rg8 => &[
                ("rrp_sw_write_f", [0x10, 0x20, 0x30, 0x40]),
                ("rrp_sw_rwwrite_f", [0x50, 0x60, 0x70, 0x80]),
            ],
            Format::R32Float => &[
                (
                    "rrp_sw_write_f",
                    [0x3fa0_0000, 0x4020_0000, 0x4070_0000, 0x40a0_0000],
                ),
                (
                    "rrp_sw_rwwrite_f",
                    [0x40c8_0000, 0x40f0_0000, 0x410c_0000, 0x4120_0000],
                ),
            ],
            Format::R32Uint => &[("rrp_sw_write_u", [11, 22, 33, 44])],
        }
    }

    /// The bytes buffer 1 holds for a write of `lanes`: an 8-bit normalized lane is its code over
    /// 255, which every device stores back as the code, and every other lane is its own word.
    pub fn value_bytes(self, lanes: [u32; 4]) -> Vec<u8> {
        lanes
            .iter()
            .flat_map(|&lane| match self.normalized() {
                true => (lane as f32 / 255.0).to_le_bytes(),
                false => lane.to_le_bytes(),
            })
            .collect()
    }
}

/// The swizzle `name` spells: each letter a channel's source, `0` and `1` the constants.
pub fn swizzle(name: &str) -> TextureSwizzleChannels {
    let sources: Vec<TextureSwizzle> = name
        .chars()
        .map(|letter| match letter {
            'R' => TextureSwizzle::Red,
            'G' => TextureSwizzle::Green,
            'B' => TextureSwizzle::Blue,
            'A' => TextureSwizzle::Alpha,
            '0' => TextureSwizzle::Zero,
            '1' => TextureSwizzle::One,
            other => panic!("a swizzle letter the family does not spell: {other}"),
        })
        .collect();
    assert_eq!(sources.len(), 4, "a swizzle names four channels: {name}");
    TextureSwizzleChannels::new(sources[0], sources[1], sources[2], sources[3])
}

/// Metal's read of `texel` through `s`, `one` the constant's value.
pub fn read_rule(s: TextureSwizzleChannels, texel: [u32; 4], one: u32) -> [u32; 4] {
    [s.red, s.green, s.blue, s.alpha].map(|source| match source.as_raw() {
        0 => 0,
        1 => one,
        raw => texel[(raw - 2) as usize],
    })
}

/// The channels Metal stores for a write of `v` through `s`: channel `c` from the first lane whose
/// source is `c`, else lane `c`.
pub fn write_rule(s: TextureSwizzleChannels, v: [u32; 4]) -> [u32; 4] {
    let sources = [s.red, s.green, s.blue, s.alpha].map(|source| source.as_raw());
    std::array::from_fn(
        |c| match sources.iter().position(|&raw| raw == c as u32 + 2) {
            Some(lane) => v[lane],
            None => v[c],
        },
    )
}

/// What a case asks of its swizzles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Asks {
    /// The reads and the writes of each swizzle.
    ReadsAndWrites,
    /// The atomic of each swizzle: what it answered, and the red channel every texel then holds.
    Atomics,
}

/// One format's sweep over its swizzles.
pub struct SwizzledStorage {
    name: &'static str,
    format: Format,
    swizzles: &'static [&'static str],
    /// Whether the texture is **made** with the swizzle, of `Unknown` usage, rather than viewed
    /// through it.
    with: bool,
    asks: Asks,
}

/// Every case of the family, in the order the lane sweeps them.
pub fn all() -> Vec<SwizzledStorage> {
    vec![
        SwizzledStorage {
            name: "swizzled-storage-rgba8unorm",
            format: Format::Rgba8,
            swizzles: &[
                "RGBA", "GBAR", "BGRA", "ABGR", "GRBA", "RRRR", "R01A", "0000", "1111", "RGB1",
                "RGBR", "0GBA",
            ],
            with: false,
            asks: Asks::ReadsAndWrites,
        },
        SwizzledStorage {
            name: "swizzled-storage-rg8unorm",
            format: Format::Rg8,
            swizzles: &["GR01", "GRBA", "RRRR", "0R01"],
            with: false,
            asks: Asks::ReadsAndWrites,
        },
        SwizzledStorage {
            name: "swizzled-storage-r32float",
            format: Format::R32Float,
            swizzles: &["GRBA", "RRRR", "ARGB"],
            with: false,
            asks: Asks::ReadsAndWrites,
        },
        SwizzledStorage {
            name: "swizzled-storage-r32uint",
            format: Format::R32Uint,
            swizzles: &["GRBA", "RRRR", "ARGB", "1RG0"],
            with: false,
            asks: Asks::ReadsAndWrites,
        },
        SwizzledStorage {
            name: "swizzled-storage-made-rgba8unorm",
            format: Format::Rgba8,
            swizzles: &["GBAR", "RRRR"],
            with: true,
            asks: Asks::ReadsAndWrites,
        },
        SwizzledStorage {
            name: "swizzled-storage-atomics",
            format: Format::R32Uint,
            swizzles: &["RRRR", "GRBA", "ARGB", "1RG0"],
            with: false,
            asks: Asks::Atomics,
        },
    ]
}

/// The atomic kernel, whose answer is the texel unswizzled.
const ATOMIC: &str = "rrp_sw_atomic_u";

/// What `rrp_sw_atomic_u` adds to red.
pub const ATOMIC_ADDS: u32 = 5;

impl SwizzledStorage {
    /// The format the case sweeps.
    pub fn format(&self) -> Format {
        self.format
    }

    /// The swizzles the case sweeps, by name.
    pub fn swizzles(&self) -> &'static [&'static str] {
        self.swizzles
    }

    /// Every kernel the case runs, in the order it runs them.
    fn kernels(&self) -> Vec<&'static str> {
        match self.asks {
            Asks::Atomics => vec![ATOMIC],
            Asks::ReadsAndWrites => self
                .format
                .read_kernels()
                .iter()
                .copied()
                .chain(
                    self.format
                        .write_kernels()
                        .iter()
                        .map(|(kernel, _)| *kernel),
                )
                .collect(),
        }
    }

    /// The texture the case's kernels bind, as its request: the parent made with the swizzle where
    /// the case makes it so, else a view of the parent through it.
    fn requests(&self, name: &str) -> (TextureRequest<'static>, Option<TextureRequest<'static>>) {
        let format = self.format.pixel_format().as_raw();
        let parent = TextureRequest::read_2d(format, SIDE, SIDE);
        if self.with {
            // Made with the swizzle, of `Unknown` usage, as the probe made it: the access the
            // kernels use is not in the texture's declared usage, which is the misuse Metal's own
            // API validation stops and plain Metal runs.
            return (
                parent.used(TextureUsage::empty()).swizzled(swizzle(name)),
                None,
            );
        }
        let view = TextureRequest::read_2d(format, SIDE, SIDE)
            .used(TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW)
            .swizzled(swizzle(name))
            .viewing(Handle(0));
        (parent.used(self.format.parent_usage()), Some(view))
    }

    /// The layout the output buffer's four words are read under, as unsigned words: a `float`
    /// channel's bits compare exactly that way, and a diff prints what Metal stored.
    fn words_layout() -> ByteLayout {
        ByteLayout::buffer(0, 16, Element::U32)
    }

    /// The layout the texture's stored texels are read under: one row a texel row, the format's
    /// own channels each, so a diff names the texel and the channel.
    fn stored_layout(&self) -> ByteLayout {
        let pitch = self.format.row_pitch();
        let element = match self.format.normalized() {
            true => Element::U8,
            false => Element::U32,
        };
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [SIDE, SIDE, 1],
            },
            element,
            row_offset: 0,
            row_elements: SIDE * self.format.channels() as u32,
            row_pitch: pitch,
            rows: SIDE,
            image_pitch: pitch * SIDE,
            images: 1,
            // The channels cover every byte of a row, so there is no gap.
            gaps: Gaps::Ignored,
        }
    }

    /// Make the parent, and the view where the case views one, answering the texture the kernels
    /// bind. The parent is loaded with the probe's texel as it is made.
    fn textures(
        &self,
        name: &str,
        side: &mut dyn Side,
        sites: &mut Sites,
    ) -> Result<(Handle, Handle), Abort> {
        let level = self.format.level();
        let uploads = [Upload::level_0(
            [SIDE, SIDE],
            self.format.row_pitch(),
            SIDE,
            &level,
        )];
        let (parent, view) = self.requests(name);
        let parent = made(
            sites,
            &format!("{name}/texture/parent"),
            side.texture(&parent.loaded(&uploads)),
        )?;
        let Some(view) = view else {
            return Ok((parent, parent));
        };
        let bound = made(
            sites,
            &format!("{name}/texture/view"),
            side.texture(&view.viewing(parent)),
        )?;
        Ok((parent, bound))
    }
}

impl Case for SwizzledStorage {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration and no tolerance: the buffer each dispatch reads its texels back through is `Shared`, which is
        // what [`crate::replay::shared_memory_notice`] is for. It loosens the notices site alone,
        // so every value this case reads still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for kernel in self.kernels() {
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
            // The texels every sub-run starts from, and the blank the output buffer holds.
            Input::bytes("texture:level", &self.format.level()),
            Input::bytes("buffer:0", &[BLANK; 16]),
        ];
        // Every texture the sweep makes whole, parent and view, so a descriptor that moves is a
        // change to the case's inputs.
        let level = self.format.level();
        let uploads = [Upload::level_0(
            [SIDE, SIDE],
            self.format.row_pitch(),
            SIDE,
            &level,
        )];
        for name in self.swizzles {
            let (parent, view) = self.requests(name);
            inputs.push(Input::bytes(
                format!("texture:parent/{name}"),
                &parent.loaded(&uploads).words(),
            ));
            if let Some(view) = view {
                inputs.push(Input::bytes(format!("texture:view/{name}"), &view.words()));
            }
        }
        // Every value a write kernel is handed, so a lane that moves is a change to the inputs.
        if self.asks == Asks::ReadsAndWrites {
            for (kernel, lanes) in self.format.write_kernels() {
                inputs.push(Input::bytes(
                    format!("buffer:1/{kernel}"),
                    &self.format.value_bytes(*lanes),
                ));
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let mut pipelines = Vec::new();
        for kernel in self.kernels() {
            pipelines.push((
                kernel,
                made(
                    sites,
                    &format!("pipeline/{kernel}"),
                    side.compute_pipeline(library, kernel),
                )?,
            ));
        }
        let level = self.format.level();
        for name in self.swizzles {
            let (parent, bound) = self.textures(name, side, sites)?;
            let pipeline = |kernel: &str| {
                pipelines
                    .iter()
                    .find(|(held, _)| *held == kernel)
                    .map(|(_, pipeline)| *pipeline)
                    .expect("a pipeline the case made")
            };
            match self.asks {
                Asks::Atomics => {
                    let at = |site: &str| format!("{name}/{site}");
                    let out = made(sites, &at("buffer/out"), side.buffer(&[BLANK; 16]))?;
                    let command_buffer = made(
                        sites,
                        &at("dispatch"),
                        side.dispatch_threads(
                            pipeline(ATOMIC),
                            &Bindings::new(&[bound], &[out]),
                            &[],
                            [1, 1, 1],
                            [1, 1, 1],
                        ),
                    )?;
                    call(
                        sites,
                        &at("status"),
                        side.wait(command_buffer),
                        Clone::clone,
                    )?;
                    let answered = call(
                        sites,
                        &at("read-back"),
                        side.read_buffer(out, 0, 16),
                        |_| Value::Absent,
                    )?;
                    let stored = call(
                        sites,
                        &at("texels"),
                        side.read_texture(parent, [SIDE, SIDE], self.format.row_pitch()),
                        |_| Value::Absent,
                    )?;
                    sites.push(
                        Node::branch(format!("{name}/atomic"), Vec::new())
                            .with(Node::leaf(
                                "answered",
                                Observed::bytes(answered, Self::words_layout()),
                            ))
                            .with(Node::leaf(
                                "texels",
                                Observed::bytes(stored, self.stored_layout()),
                            )),
                    );
                }
                Asks::ReadsAndWrites => {
                    for kernel in self.format.read_kernels() {
                        let at = |site: &str| format!("{name}/{kernel}/{site}");
                        // The texels the read answers are the probe's, whatever an earlier
                        // kernel of this sweep left behind.
                        call(
                            sites,
                            &at("reset"),
                            side.replace_region(
                                parent,
                                [SIDE, SIDE],
                                self.format.row_pitch(),
                                &level,
                            ),
                            |_| Value::Absent,
                        )?;
                        let out = made(sites, &at("buffer/out"), side.buffer(&[BLANK; 16]))?;
                        let command_buffer = made(
                            sites,
                            &at("dispatch"),
                            side.dispatch_threads(
                                pipeline(kernel),
                                &Bindings::new(&[bound], &[out]),
                                &[],
                                [1, 1, 1],
                                [1, 1, 1],
                            ),
                        )?;
                        call(
                            sites,
                            &at("status"),
                            side.wait(command_buffer),
                            Clone::clone,
                        )?;
                        let got = call(
                            sites,
                            &at("read-back"),
                            side.read_buffer(out, 0, 16),
                            |_| Value::Absent,
                        )?;
                        sites.push(Node::leaf(
                            format!("{name}/{kernel}/read"),
                            Observed::bytes(got, Self::words_layout()),
                        ));
                    }
                    for (kernel, lanes) in self.format.write_kernels() {
                        let at = |site: &str| format!("{name}/{kernel}/{site}");
                        call(
                            sites,
                            &at("reset"),
                            side.replace_region(
                                parent,
                                [SIDE, SIDE],
                                self.format.row_pitch(),
                                &level,
                            ),
                            |_| Value::Absent,
                        )?;
                        let value = made(
                            sites,
                            &at("buffer/value"),
                            side.buffer(&self.format.value_bytes(*lanes)),
                        )?;
                        let command_buffer = made(
                            sites,
                            &at("dispatch"),
                            // The kernel declares buffer 1 alone, and bindings bind from 0 up, so
                            // the value buffer sits at both indices and the kernel reads 1.
                            side.dispatch_threads(
                                pipeline(kernel),
                                &Bindings::new(&[bound], &[value, value]),
                                &[],
                                [SIDE, SIDE, 1],
                                [SIDE, SIDE, 1],
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
                            side.read_texture(parent, [SIDE, SIDE], self.format.row_pitch()),
                            |_| Value::Absent,
                        )?;
                        sites.push(Node::leaf(
                            format!("{name}/{kernel}/stored"),
                            Observed::bytes(stored, self.stored_layout()),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every outcome's bytes at the site `site` names, over every answer of `case`'s recording;
    /// `None` where the case has no recording yet.
    ///
    /// A case emits its sites as **flat** names that happen to hold slashes, so the site is looked
    /// up as one child of the root; `leaf` names a child under it, which is how the atomic's two
    /// facts sit under one branch.
    fn recorded(case: &str, site: &str, leaf: Option<&str>) -> Option<Vec<Vec<u8>>> {
        let recording = crate::recording::read(case).ok()?;
        let mut found = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let node = outcome.transcript.root.child(site);
                let node = match leaf {
                    Some(leaf) => node.and_then(|node| node.child(leaf)),
                    None => node,
                };
                if let Some(node) = node {
                    if let Value::Bytes(bytes) = &node.observed.value {
                        found.push(bytes.clone());
                    }
                }
            }
        }
        Some(found)
    }

    /// The four words at the start of `bytes`.
    fn words(bytes: &[u8]) -> [u32; 4] {
        std::array::from_fn(|at| {
            u32::from_le_bytes(bytes[4 * at..4 * at + 4].try_into().expect("four bytes"))
        })
    }

    /// Every recorded read is the texel swizzled, as the lane's rule gives it — an 8-bit
    /// normalized channel's code `c` as `c / 255` correctly rounded, so no channel may sit an ulp
    /// from it.
    #[test]
    fn every_recorded_read_is_the_lanes_rule() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for case in all() {
            if case.asks != Asks::ReadsAndWrites {
                continue;
            }
            for name in case.swizzles {
                for kernel in case.format.read_kernels() {
                    let site = format!("{name}/{kernel}/read");
                    let Some(outcomes) = recorded(case.name, &site, None) else {
                        unrecorded.push(case.name);
                        continue;
                    };
                    let (texel, one) = case.format.texel();
                    let read = read_rule(swizzle(name), texel, one);
                    let want: [u32; 4] = match case.format.normalized() {
                        true => read.map(|code| (code as f32 / 255.0).to_bits()),
                        false => read,
                    };
                    for bytes in outcomes {
                        checked += 1;
                        if words(&bytes) != want {
                            differ.push(format!(
                                "{}/{site}: Metal {:08x?}, the rule's {want:08x?}",
                                case.name,
                                words(&bytes)
                            ));
                        }
                    }
                }
            }
        }
        unrecorded.sort_unstable();
        unrecorded.dedup();
        assert!(
            differ.is_empty(),
            "{checked} reads checked; Metal's recorded reads differ from the lane's rule:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
        assert!(checked > 0, "the recordings held no read to check");
    }

    /// Every recorded store is the value with the swizzle undone, in every texel.
    #[test]
    fn every_recorded_store_is_the_lanes_rule() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for case in all() {
            if case.asks != Asks::ReadsAndWrites {
                continue;
            }
            for name in case.swizzles {
                for (kernel, lanes) in case.format.write_kernels() {
                    let site = format!("{name}/{kernel}/stored");
                    let Some(outcomes) = recorded(case.name, &site, None) else {
                        unrecorded.push(case.name);
                        continue;
                    };
                    let rule = write_rule(swizzle(name), *lanes);
                    let want = &rule[..case.format.channels()];
                    for bytes in outcomes {
                        for texel in 0..(SIDE * SIDE) as usize {
                            let size = case.format.bytes();
                            let held = &bytes[texel * size..(texel + 1) * size];
                            let got: Vec<u32> = match case.format.normalized() {
                                true => held.iter().map(|&code| u32::from(code)).collect(),
                                false => {
                                    vec![u32::from_le_bytes(held.try_into().expect("four bytes"))]
                                }
                            };
                            checked += 1;
                            if got != want {
                                differ.push(format!(
                                    "{}/{site} texel {texel}: Metal {got:08x?}, the rule's \
                                     {want:08x?}",
                                    case.name
                                ));
                            }
                        }
                    }
                }
            }
        }
        unrecorded.sort_unstable();
        unrecorded.dedup();
        assert!(
            differ.is_empty(),
            "{checked} texels checked; Metal's recorded stores differ from the lane's rule:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
        assert!(checked > 0, "the recordings held no store to check");
    }

    /// Every recorded atomic answers the texel **unswizzled** and adds to red: the one access of
    /// the family the swizzle does not reach, whatever swizzle the view carries.
    #[test]
    fn every_recorded_atomic_ignores_the_swizzle() {
        let case = all()
            .into_iter()
            .find(|case| case.asks == Asks::Atomics)
            .expect("the atomics case");
        let (texel, _) = case.format.texel();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        let mut unrecorded = false;
        for name in case.swizzles {
            let Some(answers) = recorded(case.name, &format!("{name}/atomic"), Some("answered"))
            else {
                unrecorded = true;
                continue;
            };
            for bytes in answers {
                checked += 1;
                if words(&bytes) != texel {
                    differ.push(format!(
                        "{name}: the atomic answered {:?}, not the unswizzled texel {texel:?}",
                        words(&bytes)
                    ));
                }
            }
            let Some(texels) = recorded(case.name, &format!("{name}/atomic"), Some("texels"))
            else {
                unrecorded = true;
                continue;
            };
            // Texel (0, 0) took the addend and the other three stayed as they were.
            let want: Vec<u32> = (0..(SIDE * SIDE))
                .map(|at| texel[0] + if at == 0 { ATOMIC_ADDS } else { 0 })
                .collect();
            for bytes in texels {
                let got: Vec<u32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word))
                    .collect();
                checked += 1;
                if got != want {
                    differ.push(format!("{name}: the texels' red is {got:?}, not {want:?}"));
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} checked:\n{}",
            differ.join("\n")
        );
        assert!(
            !unrecorded,
            "no recording yet for {}: run `conformance record` on a Mac",
            case.name
        );
        assert!(checked > 0, "the recording held no atomic to check");
    }

    /// The cases name distinct recordings, every swizzle spells four channels, and no case repeats
    /// a swizzle, so no two sub-runs share a site.
    #[test]
    fn the_cases_and_their_swizzles_are_distinct() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        assert_eq!(named.len(), all().len());
        for case in all() {
            let swizzled: BTreeSet<&str> = case.swizzles.iter().copied().collect();
            assert_eq!(swizzled.len(), case.swizzles.len(), "{}", case.name);
            for name in case.swizzles {
                assert_eq!(name.chars().count(), 4, "{name}");
                let _ = swizzle(name);
            }
        }
    }

    /// Each case's stored layout covers exactly the level its read-back holds.
    #[test]
    fn the_layouts_cover_their_read_backs() {
        for case in all() {
            assert_eq!(
                case.stored_layout().extent(),
                case.format.row_pitch() as usize * SIDE as usize,
                "{}",
                case.name
            );
        }
        assert_eq!(SwizzledStorage::words_layout().extent(), 16);
    }

    /// The write rule's two halves are both exercised: a swizzle that names a channel moves its
    /// lane, and one that names none leaves the channel to its own lane.
    ///
    /// Without both, a sweep could pass with the rule read as "store the lanes in order".
    #[test]
    fn the_sweeps_hold_a_moved_lane_and_an_unnamed_channel() {
        // `GBAR` names every channel, each from another lane: every channel moves.
        let moved = write_rule(swizzle("GBAR"), [1, 2, 3, 4]);
        assert_eq!(
            moved,
            [4, 1, 2, 3],
            "every channel takes the lane that names it"
        );
        // `0000` names no channel at all: every channel keeps its own lane.
        assert_eq!(write_rule(swizzle("0000"), [1, 2, 3, 4]), [1, 2, 3, 4]);
        // `RRRR` names red four times: red takes the first lane that names it, and green, blue and
        // alpha keep their own.
        assert_eq!(write_rule(swizzle("RRRR"), [1, 2, 3, 4]), [1, 2, 3, 4]);
        // `ARGB` moves every channel one the other way.
        assert_eq!(write_rule(swizzle("ARGB"), [1, 2, 3, 4]), [2, 3, 4, 1]);
        // And the sweeps really hold those swizzles.
        let swept: BTreeSet<&str> = all()
            .iter()
            .flat_map(|case| case.swizzles.iter().copied())
            .collect();
        for name in ["GBAR", "0000", "RRRR", "ARGB"] {
            assert!(swept.contains(name), "{name}");
        }
    }

    /// A made-with-a-swizzle case states `Unknown` usage and a viewed one does not, and only the
    /// viewed ones make a view: the two shapes the lane measured, kept apart in the request.
    #[test]
    fn a_made_case_states_unknown_usage_and_makes_no_view() {
        for case in all() {
            let (parent, view) = case.requests(case.swizzles[0]);
            if case.with {
                assert_eq!(parent.usage, TextureUsage::empty(), "{}", case.name);
                assert_eq!(parent.swizzle, swizzle(case.swizzles[0]), "{}", case.name);
                assert!(view.is_none(), "{}", case.name);
            } else {
                assert_eq!(
                    parent.swizzle,
                    TextureSwizzleChannels::DEFAULT,
                    "{}: the parent carries no swizzle; the view does",
                    case.name
                );
                let view = view.unwrap_or_else(|| panic!("{}", case.name));
                assert_eq!(view.swizzle, swizzle(case.swizzles[0]), "{}", case.name);
                assert!(parent.usage.contains(TextureUsage::PIXEL_FORMAT_VIEW));
            }
        }
    }
}
