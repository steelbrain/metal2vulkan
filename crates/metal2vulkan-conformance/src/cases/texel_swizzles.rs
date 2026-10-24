//! A texture **buffer** made or viewed with a swizzle, ported from `runtime_lane::texel_swizzles`.
//!
//! Metal applies a texture buffer's swizzle to every read, out of bounds included (the lane: an M3
//! Max on macOS 26.7 and an Apple M4 on 27.0.1, plain and under Metal's API validation alike). Each
//! channel reads the texel's channel its swizzle names, `Zero` 0 and `One` 1 — `1.0` for a float
//! format, `1` for an integer one — and a format's missing green and blue read 0 while its missing
//! alpha reads 1. A read past the texture buffer reads the texel `(0, 0, 0, 0)`, with alpha 1 where
//! the format has no alpha, swizzled alike. A **view** made with a swizzle reads its own, replacing
//! the texture's.
//!
//! **The port re-measures all of it on this M3 Max under 27.0.1**, which is a release the lane's M3
//! Max row predates: its 26.7 numbers were taken before the oracle moved, so the one thing the lane
//! cannot tell us is whether this chip on this build still agrees. Record mode runs the case, the
//! recording holds Metal's own words, and the rule is a committed cross-check
//! (`every_recorded_word_is_the_lanes_rule`).
//!
//! The kernels are two owned fixtures':
//!
//! - `texture_ops`'s `texel_buffers` writes twice the `float` texel of its `[[texture(0)]]` texture
//!   buffer, at each thread's index, into its `[[texture(1)]]` one. The source holds [`TEXELS`]
//!   texels and the grid [`GRID`] threads, so half the reads lie past it.
//! - `ob1_texel_buffer` writes the red channel of the `uint` texel its `[[texture(0)]]` texture
//!   buffer holds at the index buffer 0 gives it, past the texture buffer for half the threads, into
//!   buffer 1.
//!
//! One case per source format, each sweeping that format's swizzles, and one case for the views, so
//! a venue that holds no texture buffer of a format answers its typed refusal at that row rather
//! than leaving the rest of the sweep uncompared.
//!
//! The destination of each dispatch holds a fill byte beforehand (`BLANK`), so a texel the kernel
//! did not write is a recorded value rather than whatever the allocator left there.
//!
//! A swizzle beside `ShaderWrite` usage is refused when the texture is made, by Metal and by the
//! runtime alike, so only the sources carry one here and every destination is unswizzled.
//!
//! Nothing here can fault a GPU: one compute pass a dispatch, eight threads, no loop and no
//! sampling. The reads past the texture buffer are the texel-buffer containment form, whose results
//! MSL 4.1 defines (a read out of bounds reads zero, a write out of bounds is ignored), so they are
//! not an out-of-bounds access the device may fault on.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::TextureRequest;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureUsage};

/// The texels of each source texture buffer.
pub const TEXELS: u32 = 4;

/// The threads of each grid, one threadgroup: the indices from [`TEXELS`] on read past the source.
pub const GRID: u32 = 8;

/// The byte every destination holds before a dispatch, so a texel the kernel did not write is a
/// recorded value.
const BLANK: u8 = 0x5a;

/// The fixture whose `texel_buffers` kernel doubles a `float` texel.
const FLOAT_FIXTURE: Fixture = Fixture {
    name: "texture_ops",
    entry: "texel_buffers",
};

/// The fixture whose kernel reads a `uint` texel at an index a buffer gives it.
const UINT_FIXTURE: Fixture = Fixture::new("ob1_texel_buffer");

/// The four swizzles the lane read, and the identity: `bgr1`, `01ar`, `gggg` and `abgr`.
pub fn swizzles() -> [(&'static str, TextureSwizzleChannels); 5] {
    use TextureSwizzle::{Alpha, Blue, Green, One, Red, Zero};
    let s = |red, green, blue, alpha| TextureSwizzleChannels {
        red,
        green,
        blue,
        alpha,
    };
    [
        ("rgba", TextureSwizzleChannels::DEFAULT),
        ("bgr1", s(Blue, Green, Red, One)),
        ("01ar", s(Zero, One, Alpha, Red)),
        ("gggg", s(Green, Green, Green, Green)),
        ("abgr", s(Alpha, Blue, Green, Red)),
    ]
}

/// The one-channel swizzles the integer sweep reads: red from each of the six sources, every other
/// channel the format's own, so one dispatch answers what one source gives red.
pub fn red_sources() -> [(&'static str, TextureSwizzle); 6] {
    use TextureSwizzle::{Alpha, Blue, Green, One, Red, Zero};
    [
        ("red-zero", Zero),
        ("red-one", One),
        ("red-red", Red),
        ("red-green", Green),
        ("red-blue", Blue),
        ("red-alpha", Alpha),
    ]
}

/// The swizzle whose red is `source` and whose other channels are their own.
fn red_swizzle(source: TextureSwizzle) -> TextureSwizzleChannels {
    TextureSwizzleChannels {
        red: source,
        ..TextureSwizzleChannels::DEFAULT
    }
}

/// A source texture buffer's format, its texels' bytes, and each texel's four channels as Metal
/// reads them unswizzled (a float format's as the bits of its `float`).
///
/// The table is public because it is the family's description: what reads `texels`, `alpha` and
/// `one` is the cross-check that holds the recordings to the lane's rule.
pub struct Source {
    pub name: &'static str,
    pub format: PixelFormat,
    /// The source texture buffer's bytes, which are the buffer's own.
    pub bytes: Vec<u8>,
    /// Each texel's four channels unswizzled, a float format's as the bits of its `float`.
    pub texels: Vec<[u32; 4]>,
    /// Whether the format has an alpha channel: a read past the texture buffer reads alpha 0 where
    /// it has one, and `one` where it does not.
    pub alpha: bool,
    /// `One`'s value, and a missing alpha's: `1.0`'s bits for a float format, else 1.
    pub one: u32,
}

/// The float sources: 8-bit normalized RGBA, and one and two 32-bit float channels.
pub fn float_sources() -> Vec<Source> {
    let one = 1.0f32.to_bits();
    let codes: Vec<u8> = (0..4 * TEXELS)
        .map(|i| ((7 * i + 3) & 0xff) as u8)
        .collect();
    let unorm = codes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| [0, 1, 2, 3].map(|i| (f32::from(c[i]) / 255.0).to_bits()))
        .collect();
    let floats = [1.5f32, -2.25, 3.0e5, 0.125, 7.0, -0.5, 1024.0, 3.75];
    let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let two = TEXELS as usize;
    vec![
        Source {
            name: "texel-swizzles-rgba8unorm",
            format: PixelFormat::RGBA8Unorm,
            bytes: codes.clone(),
            texels: unorm,
            alpha: true,
            one,
        },
        Source {
            name: "texel-swizzles-r32float",
            format: PixelFormat::R32Float,
            bytes: bytes(&floats[..two]),
            texels: floats[..two]
                .iter()
                .map(|f| [f.to_bits(), 0, 0, one])
                .collect(),
            alpha: false,
            one,
        },
        Source {
            name: "texel-swizzles-rg32float",
            format: PixelFormat::RG32Float,
            bytes: bytes(&floats),
            texels: floats
                .as_chunks::<2>()
                .0
                .iter()
                .map(|f| [f[0].to_bits(), f[1].to_bits(), 0, one])
                .collect(),
            alpha: false,
            one,
        },
    ]
}

/// The unsigned sources: one 32-bit channel, and four 16-bit ones.
pub fn uint_sources() -> Vec<Source> {
    let words = [0x1122_3344u32, 7, 0xffff_ffff, 0x8000_0000];
    let halves: Vec<u16> = (0..4 * TEXELS as u16)
        .map(|i| i.wrapping_mul(0x1357).wrapping_add(0x0203))
        .collect();
    vec![
        Source {
            name: "texel-swizzles-r32uint",
            format: PixelFormat::R32Uint,
            bytes: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            texels: words.iter().map(|&w| [w, 0, 0, 1]).collect(),
            alpha: false,
            one: 1,
        },
        Source {
            name: "texel-swizzles-rgba16uint",
            format: PixelFormat::RGBA16Uint,
            bytes: halves.iter().flat_map(|h| h.to_le_bytes()).collect(),
            texels: halves
                .as_chunks::<4>()
                .0
                .iter()
                .map(|h| [0, 1, 2, 3].map(|i| u32::from(h[i])))
                .collect(),
            alpha: true,
            one: 1,
        },
    ]
}

impl Source {
    /// The channels Metal reads at `index` through `swizzle`, past the texels too.
    pub fn read(&self, index: usize, swizzle: TextureSwizzleChannels) -> [u32; 4] {
        let texel = match self.texels.get(index) {
            Some(&texel) => texel,
            None => [0, 0, 0, if self.alpha { 0 } else { self.one }],
        };
        [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha].map(|source| {
            match source.as_raw() {
                0 => 0,
                1 => self.one,
                raw => texel[(raw - 2) as usize],
            }
        })
    }

    /// The source as a read-only texture buffer over `buffer`, made with `swizzle` and `usage`
    /// beside `ShaderRead`.
    fn request(
        &self,
        buffer: Handle,
        swizzle: TextureSwizzleChannels,
        usage: TextureUsage,
    ) -> TextureRequest<'static> {
        TextureRequest::texel_buffer(
            self.format.as_raw(),
            TEXELS,
            buffer,
            0,
            self.bytes.len() as u64,
        )
        .used(TextureUsage::SHADER_READ | usage)
        .swizzled(swizzle)
    }
}

/// `texel`'s float channels doubled, as `texel_buffers` writes them.
pub fn twice(texel: [u32; 4]) -> [u32; 4] {
    texel.map(|channel| (2.0 * f32::from_bits(channel)).to_bits())
}

/// The indices the integer sweep's threads read: four inside the texture buffer and four past it.
pub const AT: [u32; GRID as usize] = [0, 1, 2, 3, 4, 7, 64, 1 << 20];

/// Buffer 0 of the integer sweep: the indices its threads read, then the indices they write, which
/// are the eight texels of the destination.
fn indices() -> Vec<u8> {
    AT.iter()
        .copied()
        .chain(0..GRID)
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// The `RGBA32Float` destination the float sweep writes: [`GRID`] texels of four words.
const DOUBLED_BYTES: usize = GRID as usize * 16;

/// The `R32Uint` output the integer sweep writes: one word a thread.
const OUT_BYTES: usize = GRID as usize * 4;

/// The layout the float sweep's answer is read under: one row a texel, four unsigned words each, so
/// a diff names the texel and the channel and prints the bits Metal produced rather than a float.
fn doubled_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: DOUBLED_BYTES as u64,
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: 16,
        rows: GRID,
        image_pitch: DOUBLED_BYTES as u32,
        images: 1,
        // The rows cover every byte of the buffer, so there is no gap between them.
        gaps: Gaps::Ignored,
    }
}

/// The layout the integer sweep's answer is read under: one unsigned word a thread.
fn out_layout() -> ByteLayout {
    ByteLayout::buffer(0, OUT_BYTES as u64, Element::U32)
}

/// Which sweep a case runs.
enum Sweep {
    /// Every swizzle's `float` reads of one source, doubled into an `RGBA32Float` texture buffer.
    Float,
    /// Red from each of the six sources, of one `uint` source, at the indices [`AT`] names.
    Uint,
}

/// One source format's sweep.
pub struct TexelSwizzles {
    at: usize,
    sweep: Sweep,
}

/// A view of a texture buffer reading its own swizzle, over the lane's two parents and two views.
pub struct ViewedTexelSwizzles;

/// Every source format's case, and the views.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for at in 0..float_sources().len() {
        cases.push(Box::new(TexelSwizzles {
            at,
            sweep: Sweep::Float,
        }));
    }
    for at in 0..uint_sources().len() {
        cases.push(Box::new(TexelSwizzles {
            at,
            sweep: Sweep::Uint,
        }));
    }
    cases.push(Box::new(ViewedTexelSwizzles));
    cases
}

/// The two shader texts of `fixture`, as a case's shader references.
fn shaders_of(fixture: Fixture) -> Result<Vec<ShaderRef>, String> {
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{}{extension}", fixture.name),
                entry: fixture.entry.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// The two shader texts of `fixture`, as a case's first two inputs.
fn shader_inputs(fixture: Fixture) -> Result<Vec<Input>, String> {
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// The library and the compute pipeline of `fixture`, made on `side`.
fn pipeline_of(fixture: Fixture, side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
    let loaded = fixture.loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    let library = made(sites, "library", side.library(&loaded.source()))?;
    made(
        sites,
        "pipeline",
        side.compute_pipeline(library, loaded.entry),
    )
}

/// The `RGBA32Float` texture buffer the float sweep writes, over `buffer`.
fn doubled_request(buffer: Handle) -> TextureRequest<'static> {
    TextureRequest::texel_buffer(
        PixelFormat::RGBA32Float.as_raw(),
        GRID,
        buffer,
        0,
        DOUBLED_BYTES as u64,
    )
    .used(TextureUsage::SHADER_WRITE)
}

/// The `R32Uint` texture buffer the integer sweep writes, over `buffer`.
fn written_request(buffer: Handle) -> TextureRequest<'static> {
    TextureRequest::texel_buffer(
        PixelFormat::R32Uint.as_raw(),
        GRID,
        buffer,
        0,
        OUT_BYTES as u64,
    )
    .used(TextureUsage::SHADER_WRITE)
}

impl TexelSwizzles {
    /// The source the case sweeps.
    fn source(&self) -> Source {
        match self.sweep {
            Sweep::Float => float_sources().swap_remove(self.at),
            Sweep::Uint => uint_sources().swap_remove(self.at),
        }
    }

    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        match self.sweep {
            Sweep::Float => FLOAT_FIXTURE,
            Sweep::Uint => UINT_FIXTURE,
        }
    }

    /// The float sweep: every swizzle's reads of the source, doubled.
    fn run_float(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let pipeline = pipeline_of(FLOAT_FIXTURE, side, sites)?;
        let source = self.source();
        for (name, swizzle) in swizzles() {
            let at = |site: &str| format!("{name}/{site}");
            let bytes = made(sites, &at("buffer/source"), side.buffer(&source.bytes))?;
            let read = made(
                sites,
                &at("texture/source"),
                side.texture(&source.request(bytes, swizzle, TextureUsage::empty())),
            )?;
            let back = made(
                sites,
                &at("buffer/doubled"),
                side.buffer(&[BLANK; DOUBLED_BYTES]),
            )?;
            let written = made(
                sites,
                &at("texture/doubled"),
                side.texture(&doubled_request(back)),
            )?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::textures(&[read, written]),
                    &[],
                    [GRID, 1, 1],
                    [GRID, 1, 1],
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
                side.read_buffer(back, 0, DOUBLED_BYTES),
                |_| Value::Absent,
            )?;
            sites.push(Node::leaf(
                format!("{name}/doubled"),
                Observed::bytes(got, doubled_layout()),
            ));
        }
        Ok(())
    }

    /// The integer sweep: red from each of the six sources, at the indices [`AT`] names.
    fn run_uint(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let pipeline = pipeline_of(UINT_FIXTURE, side, sites)?;
        let source = self.source();
        let index = made(sites, "buffer/indices", side.buffer(&indices()))?;
        for (name, red) in red_sources() {
            let at = |site: &str| format!("{name}/{site}");
            let swizzle = red_swizzle(red);
            let bytes = made(sites, &at("buffer/source"), side.buffer(&source.bytes))?;
            let read = made(
                sites,
                &at("texture/source"),
                side.texture(&source.request(bytes, swizzle, TextureUsage::empty())),
            )?;
            let target = made(
                sites,
                &at("buffer/written"),
                side.buffer(&[BLANK; OUT_BYTES]),
            )?;
            let written = made(
                sites,
                &at("texture/written"),
                side.texture(&written_request(target)),
            )?;
            let out = made(sites, &at("buffer/out"), side.buffer(&[BLANK; OUT_BYTES]))?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::new(&[read, written], &[index, out]),
                    &[],
                    [GRID, 1, 1],
                    [GRID, 1, 1],
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
                side.read_buffer(out, 0, OUT_BYTES),
                |_| Value::Absent,
            )?;
            sites.push(Node::leaf(
                format!("{name}/red"),
                Observed::bytes(got, out_layout()),
            ));
        }
        Ok(())
    }
}

impl Case for TexelSwizzles {
    fn name(&self) -> &'static str {
        match self.sweep {
            Sweep::Float => float_sources()[self.at].name,
            Sweep::Uint => uint_sources()[self.at].name,
        }
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration and no tolerance: the buffer each dispatch writes its swizzled texels into is `Shared`, which is
        // what [`crate::replay::shared_memory_notice`] is for. It loosens the notices site alone,
        // so every value this case reads still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(self.fixture())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = shader_inputs(self.fixture())?;
        let source = self.source();
        // The source buffer's bytes, and every texture the sweep makes whole: a texture whose
        // format, extent, usage, swizzle or backing bytes move is a change to the case's inputs
        // rather than a quiet pass against Metal's answer for another texture.
        inputs.push(Input::bytes("buffer:source", &source.bytes));
        match self.sweep {
            Sweep::Float => {
                inputs.push(Input::bytes("buffer:doubled", &[BLANK; DOUBLED_BYTES]));
                for (name, swizzle) in swizzles() {
                    let request = source.request(Handle(0), swizzle, TextureUsage::empty());
                    inputs.push(Input::bytes(
                        format!("texture:source/{name}"),
                        &request.words(),
                    ));
                }
                inputs.push(Input::bytes(
                    "texture:doubled",
                    &doubled_request(Handle(0)).words(),
                ));
            }
            Sweep::Uint => {
                inputs.push(Input::bytes("buffer:indices", &indices()));
                inputs.push(Input::bytes("buffer:out", &[BLANK; OUT_BYTES]));
                for (name, red) in red_sources() {
                    let request =
                        source.request(Handle(0), red_swizzle(red), TextureUsage::empty());
                    inputs.push(Input::bytes(
                        format!("texture:source/{name}"),
                        &request.words(),
                    ));
                }
                inputs.push(Input::bytes(
                    "texture:written",
                    &written_request(Handle(0)).words(),
                ));
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match self.sweep {
            Sweep::Float => self.run_float(side, sites),
            Sweep::Uint => self.run_uint(side, sites),
        }
    }
}

/// The parents the view case makes the texture buffer with, by their swizzle's name.
pub const PARENTS: [&str; 2] = ["rgba", "bgr1"];

/// The swizzles the view case views it through.
pub const VIEWS: [&str; 2] = ["01ar", "abgr"];

/// The swizzle `swizzles()` names `name`.
fn named(name: &str) -> TextureSwizzleChannels {
    swizzles()
        .into_iter()
        .find(|(held, _)| *held == name)
        .map(|(_, swizzle)| swizzle)
        .expect("a swizzle the sweep names")
}

impl ViewedTexelSwizzles {
    /// The source the views are of: the lane's, the 8-bit normalized one.
    fn source() -> Source {
        float_sources().swap_remove(0)
    }

    /// The parent texture buffer, made with `parent`'s swizzle and viewable in another format.
    fn parent_request(source: &Source, buffer: Handle, parent: &str) -> TextureRequest<'static> {
        source.request(buffer, named(parent), TextureUsage::PIXEL_FORMAT_VIEW)
    }

    /// The view: the source's own format and a texture buffer again, through `view`'s swizzle.
    fn view_request(source: &Source, parent: Handle, view: &str) -> TextureRequest<'static> {
        TextureRequest::texel_buffer(source.format.as_raw(), TEXELS, Handle(0), 0, 0)
            .used(TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW)
            .swizzled(named(view))
            .viewing(parent)
    }
}

impl Case for ViewedTexelSwizzles {
    fn name(&self) -> &'static str {
        "texel-swizzles-views"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration and no tolerance: the buffer each dispatch writes its swizzled texels into is `Shared`, which is
        // what [`crate::replay::shared_memory_notice`] is for. It loosens the notices site alone,
        // so every value this case reads still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(FLOAT_FIXTURE)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = shader_inputs(FLOAT_FIXTURE)?;
        let source = Self::source();
        inputs.push(Input::bytes("buffer:source", &source.bytes));
        inputs.push(Input::bytes("buffer:doubled", &[BLANK; DOUBLED_BYTES]));
        for parent in PARENTS {
            inputs.push(Input::bytes(
                format!("texture:parent/{parent}"),
                &Self::parent_request(&source, Handle(0), parent).words(),
            ));
        }
        for view in VIEWS {
            inputs.push(Input::bytes(
                format!("texture:view/{view}"),
                &Self::view_request(&source, Handle(0), view).words(),
            ));
        }
        inputs.push(Input::bytes(
            "texture:doubled",
            &doubled_request(Handle(0)).words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let pipeline = pipeline_of(FLOAT_FIXTURE, side, sites)?;
        let source = Self::source();
        for parent_name in PARENTS {
            let bytes = made(
                sites,
                &format!("{parent_name}/buffer/source"),
                side.buffer(&source.bytes),
            )?;
            let parent = made(
                sites,
                &format!("{parent_name}/texture/parent"),
                side.texture(&Self::parent_request(&source, bytes, parent_name)),
            )?;
            for view_name in VIEWS {
                let at = |site: &str| format!("{parent_name}/{view_name}/{site}");
                let view = made(
                    sites,
                    &at("texture/view"),
                    side.texture(&Self::view_request(&source, parent, view_name)),
                )?;
                let back = made(
                    sites,
                    &at("buffer/doubled"),
                    side.buffer(&[BLANK; DOUBLED_BYTES]),
                )?;
                let written = made(
                    sites,
                    &at("texture/doubled"),
                    side.texture(&doubled_request(back)),
                )?;
                let command_buffer = made(
                    sites,
                    &at("dispatch"),
                    side.dispatch_threads(
                        pipeline,
                        &Bindings::textures(&[view, written]),
                        &[],
                        [GRID, 1, 1],
                        [GRID, 1, 1],
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
                    side.read_buffer(back, 0, DOUBLED_BYTES),
                    |_| Value::Absent,
                )?;
                sites.push(Node::leaf(
                    format!("{parent_name}/{view_name}/doubled"),
                    Observed::bytes(got, doubled_layout()),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every word of every committed recording is the lane's rule's, bit for bit — an 8-bit
    /// normalized code's `code / 255` correctly rounded included, so no channel may sit an ulp
    /// from it.
    ///
    /// This is the cross-check the port exists for. The lane's M3 Max row was taken on macOS 26.7,
    /// before this round's oracle, and its 27.0.1 row was an Apple M4 that is gone; the recordings
    /// hold what this M3 Max on 27.0.1 answered. A difference means the release or the chip moved
    /// the rule, which is a finding rather than something to edit away.
    #[test]
    fn every_recorded_word_is_the_lanes_rule() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for source in float_sources() {
            let recording = match crate::recording::read(source.name) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(source.name);
                    continue;
                }
            };
            for (name, swizzle) in swizzles() {
                let site = format!("{name}/doubled");
                let case = source.name;
                for bytes in recorded(&recording, &site) {
                    for index in 0..GRID as usize {
                        let read = twice(source.read(index, swizzle));
                        for (channel, wanted) in read.into_iter().enumerate() {
                            let at = index * 16 + channel * 4;
                            let got = u32::from_le_bytes(
                                bytes[at..at + 4].try_into().expect("four bytes"),
                            );
                            checked += 1;
                            if got != wanted && differ.len() < 24 {
                                differ.push(format!(
                                    "{case}/{site} texel {index} channel {channel}: Metal \
                                     {got:#010x}, the rule's {wanted:#010x}"
                                ));
                            }
                        }
                    }
                }
            }
        }
        for source in uint_sources() {
            let recording = match crate::recording::read(source.name) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(source.name);
                    continue;
                }
            };
            for (name, red) in red_sources() {
                let site = format!("{name}/red");
                for bytes in recorded(&recording, &site) {
                    for (thread, &index) in AT.iter().enumerate() {
                        let wanted = source.read(index as usize, red_swizzle(red))[0];
                        let at = thread * 4;
                        let got =
                            u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
                        checked += 1;
                        if got != wanted && differ.len() < 24 {
                            differ.push(format!(
                                "{}/{site} at {index}: Metal {got:#010x}, the rule's {wanted:#010x}",
                                source.name
                            ));
                        }
                    }
                }
            }
        }
        // A view reads its own swizzle, replacing its parent's, so the rule takes the view's and
        // the parent's swizzle never enters it. A leg that read the parent's instead would differ
        // here, which is the whole question the view case asks.
        let source = ViewedTexelSwizzles::source();
        match crate::recording::read(ViewedTexelSwizzles.name()) {
            Err(_) => unrecorded.push(ViewedTexelSwizzles.name()),
            Ok(recording) => {
                for parent in PARENTS {
                    for view in VIEWS {
                        let site = format!("{parent}/{view}/doubled");
                        for bytes in recorded(&recording, &site) {
                            for index in 0..GRID as usize {
                                let read = twice(source.read(index, named(view)));
                                for (channel, wanted) in read.into_iter().enumerate() {
                                    let at = index * 16 + channel * 4;
                                    let got = u32::from_le_bytes(
                                        bytes[at..at + 4].try_into().expect("four bytes"),
                                    );
                                    checked += 1;
                                    if got != wanted && differ.len() < 24 {
                                        differ.push(format!(
                                            "views/{site} texel {index} channel {channel}: Metal \
                                             {got:#010x}, the rule's {wanted:#010x}"
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} words checked; Metal's recorded words differ from the lane's rule:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
        assert!(checked > 0, "the recordings held no word to check");
    }

    /// Every outcome's bytes at `site`, over every answer of `recording`.
    fn recorded<'a>(recording: &'a crate::recording::Recording, site: &str) -> Vec<&'a Vec<u8>> {
        let mut found = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                if let Some(node) = outcome.transcript.root.child(site) {
                    if let Value::Bytes(bytes) = &node.observed.value {
                        found.push(bytes);
                    }
                }
            }
        }
        found
    }

    /// The cases name distinct recordings, and the swizzles and red sources distinct sites.
    #[test]
    fn the_cases_and_the_sites_are_distinct() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name()).collect();
        assert_eq!(named.len(), all().len());
        let swizzled: BTreeSet<&str> = swizzles().iter().map(|(name, _)| *name).collect();
        assert_eq!(swizzled.len(), swizzles().len());
        let reds: BTreeSet<&str> = red_sources().iter().map(|(name, _)| *name).collect();
        assert_eq!(reds.len(), red_sources().len());
        // The parents and the views are drawn from the sweep's own swizzles, and no name is both.
        for name in PARENTS.iter().chain(&VIEWS) {
            assert!(swizzles().iter().any(|(held, _)| held == name), "{name}");
        }
        assert!(PARENTS.iter().all(|parent| !VIEWS.contains(parent)));
    }

    /// Each layout covers exactly the bytes its read-back holds.
    #[test]
    fn the_layouts_cover_their_read_backs() {
        assert_eq!(doubled_layout().extent(), DOUBLED_BYTES);
        assert_eq!(out_layout().extent(), OUT_BYTES);
    }

    /// Half the indices each sweep reads lie past the texture buffer, and half inside it: the
    /// question the family asks is what a read out of bounds gives, so a sweep that lost those
    /// would compare only the easy half.
    #[test]
    fn half_of_every_sweeps_indices_lie_past_the_texture_buffer() {
        let past = AT.iter().filter(|&&index| index >= TEXELS).count();
        assert_eq!(past, AT.len() / 2, "{AT:?}");
        assert!(AT.iter().any(|&index| index < TEXELS));
        // The float sweep reads index 0 up to `GRID`, so its own second half is past the source.
        let beyond = (0..GRID).filter(|index| *index >= TEXELS).count();
        assert_eq!(
            beyond,
            (GRID / 2) as usize,
            "the float sweep reads as many indices past the source as inside it"
        );
        for source in float_sources().iter().chain(&uint_sources()) {
            assert_eq!(
                source.texels.len(),
                TEXELS as usize,
                "{}: its texels are the texture buffer's",
                source.name
            );
        }
    }

    /// A missing alpha reads 1 inside the texture buffer and past it, and a present one reads 0
    /// past it: the two halves of the rule that a sweep of one format alone could not separate.
    #[test]
    fn a_missing_alpha_reads_one_past_the_texture_buffer() {
        let rgba = &float_sources()[0];
        assert!(rgba.alpha, "{} has an alpha channel", rgba.name);
        assert_eq!(
            rgba.read(TEXELS as usize, TextureSwizzleChannels::DEFAULT)[3],
            0
        );
        let r32 = &float_sources()[1];
        assert!(!r32.alpha, "{} has no alpha channel", r32.name);
        assert_eq!(
            r32.read(TEXELS as usize, TextureSwizzleChannels::DEFAULT)[3],
            1.0f32.to_bits()
        );
        assert_eq!(
            r32.read(0, TextureSwizzleChannels::DEFAULT)[3],
            1.0f32.to_bits()
        );
    }
}
