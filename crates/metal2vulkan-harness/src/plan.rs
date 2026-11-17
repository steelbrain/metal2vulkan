//! A workload as data: what a [`Session`](crate::run::Session) makes, the calls of one run, what
//! the first run must leave in the buffers, and which buffers runtimes compare.

use crate::{Result, Size, Storage};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::DataType;
use std::path::PathBuf;

/// One workload.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Its name in reports.
    pub name: String,
    /// The library its kernels are in.
    pub library: Source,
    /// The buffers it makes, in order; dispatches, expectations and outputs name them by index.
    pub buffers: Vec<BufferPlan>,
    /// The textures it makes, in order; dispatches and expectations name them by index.
    pub textures: Vec<TexturePlan>,
    /// The dispatches of one run, in order, in one compute encoder of each command buffer.
    pub dispatches: Vec<Dispatch>,
    /// How many dispatches each command buffer of a run holds, its command buffers committed in
    /// order; `None`: one command buffer holds them all. A binding stays set for the dispatches
    /// after it in its command buffer, not in the next.
    pub per_command_buffer: Option<usize>,
    /// What the first run must leave in its buffers: a CPU reference.
    pub expect: Vec<Expect>,
    /// What the first run must leave in its textures: a CPU reference.
    pub expect_textures: Vec<TextureExpect>,
    /// The buffers whose contents after the first run each runtime's must match the first
    /// runtime's, for a workload with no CPU reference.
    pub outputs: Vec<Output>,
}

/// Where a plan's kernels are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// The authored workloads' library (`crate::authored`).
    Authored,
    /// A metallib file.
    Metallib(PathBuf),
}

/// One buffer of a plan.
#[derive(Clone, Debug)]
pub struct BufferPlan {
    /// Its length in bytes: a multiple of 16, the harness's transfer unit.
    pub length: u64,
    /// Its storage mode.
    pub storage: Storage,
    /// The bytes it starts with, `length` of them: for `Shared` storage written through
    /// `contents`, for `Private` copied in from a `Shared` buffer by the harness's own kernel.
    /// `None`: as `newBufferWithLength:` made it.
    pub contents: Option<Vec<u8>>,
}

/// One texture of a plan: 2D, `RGBA16Float`, one level, `Private`, of `ShaderRead | ShaderWrite`
/// usage. Its width and height are multiples of 16, the harness's transfer kernels' threadgroups.
#[derive(Clone, Debug)]
pub struct TexturePlan {
    /// Its width in texels.
    pub width: u32,
    /// Its height in texels.
    pub height: u32,
    /// The texels it starts with, row by row, four `float`s each, written in by the harness's own
    /// kernel (`texels_in`) from a `Shared` buffer. `None`: as `newTextureWithDescriptor:` made it.
    pub texels: Option<Vec<f32>>,
}

/// A sampler state's filter: its `minFilter` and `magFilter`. Its coordinates are normalized and
/// each address mode clamps to the edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// `MTLSamplerMinMagFilterNearest`.
    Nearest,
    /// `MTLSamplerMinMagFilterLinear`.
    Linear,
}

/// One dispatch: the calls an encoder makes for it. The pipeline state is set where it differs
/// from the dispatch before's; bindings stay set for the dispatches after, as Metal's do.
#[derive(Clone, Debug)]
pub struct Dispatch {
    /// The kernel's name in the plan's library.
    pub function: String,
    /// The values of the kernel's function constants, by index; none where it declares none.
    pub constants: Vec<Constant>,
    /// The binding calls before the dispatch, in order.
    pub bindings: Vec<Binding>,
    /// The dispatch call.
    pub grid: Grid,
}

/// A function constant's value, as `setConstantValue:type:atIndex:` takes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constant {
    /// The constant's index.
    pub index: u32,
    /// Its `MTLDataType`, raw.
    pub data_type: u32,
    /// Its bytes.
    pub value: Vec<u8>,
}

/// `constants` as the translator's `MTLFunctionConstantValues`, which both runtimes take.
pub(crate) fn values(constants: &[Constant]) -> Result<FunctionConstantValues> {
    let mut values = FunctionConstantValues::defaults();
    for constant in constants {
        let data_type = DataType::try_from(constant.data_type)?;
        let value = FunctionConstantValue::new(data_type, constant.value.as_slice());
        values = values.with_index(constant.index, value);
    }
    Ok(values)
}

/// A binding call.
#[derive(Clone, Debug)]
pub enum Binding {
    /// `setBuffer:offset:atIndex:` of the plan's buffer `buffer`.
    Buffer {
        /// The argument table index.
        index: u32,
        /// The plan's buffer.
        buffer: usize,
        /// The offset in it.
        offset: u64,
    },
    /// `setBytes:length:atIndex:`.
    Bytes {
        /// The argument table index.
        index: u32,
        /// The bytes.
        bytes: Vec<u8>,
    },
    /// `setTexture:atIndex:` of the plan's texture `texture`.
    Texture {
        /// The texture table index.
        index: u32,
        /// The plan's texture.
        texture: usize,
    },
    /// `setSamplerState:atIndex:` of a sampler state of `filter`, made once per plan.
    Sampler {
        /// The sampler table index.
        index: u32,
        /// Its filter.
        filter: Filter,
    },
}

/// A dispatch call.
#[derive(Clone, Copy, Debug)]
pub enum Grid {
    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    Threadgroups {
        /// Threadgroups per grid.
        groups: Size,
        /// Threads per threadgroup.
        threads: Size,
    },
    /// `dispatchThreads:threadsPerThreadgroup:`.
    Threads {
        /// Threads per grid.
        threads: Size,
        /// Threads per threadgroup.
        per_group: Size,
    },
}

impl Grid {
    /// Threadgroups of 256 threads in one dimension.
    pub fn groups_of_256(groups: u64) -> Grid {
        Grid::Threadgroups {
            groups: [groups, 1, 1],
            threads: [256, 1, 1],
        }
    }
}

/// What a buffer must hold after the first run.
#[derive(Clone, Debug)]
pub struct Expect {
    /// The plan's buffer.
    pub buffer: usize,
    /// Its contents from its start.
    pub values: Values,
}

/// What a texture must hold after the first run.
#[derive(Clone, Debug)]
pub struct TextureExpect {
    /// The plan's texture.
    pub texture: usize,
    /// Its texels from its first, row by row, four `float`s each, as the harness's own kernel
    /// reads them out (`texels_out`).
    pub values: Values,
}

/// A buffer whose contents after the first run runtimes compare.
#[derive(Clone, Copy, Debug)]
pub struct Output {
    /// The plan's buffer.
    pub buffer: usize,
    /// How its contents compare.
    pub comparison: Comparison,
}

/// How two runtimes' contents of a buffer compare.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Comparison {
    /// Each 32-bit word exactly.
    Exact,
    /// Each 32-bit word as a `float`, within `tolerance`; NaN matches NaN.
    Floats {
        /// The largest difference allowed.
        tolerance: f32,
    },
}

impl Comparison {
    /// The values `bytes` (little-endian) hold, as this comparison reads them.
    pub fn values(self, bytes: &[u8]) -> Values {
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w));
        match self {
            Comparison::Exact => Values::Words(words.collect()),
            Comparison::Floats { tolerance } => Values::Floats {
                values: words.map(f32::from_bits).collect(),
                tolerance,
            },
        }
    }
}

/// Expected contents.
#[derive(Clone, Debug)]
pub enum Values {
    /// 32-bit words, each exactly.
    Words(Vec<u32>),
    /// `float`s, each within `tolerance` of its value, a NaN matching a NaN: the device's own
    /// arithmetic may differ from the CPU's, or another runtime's code, in its last bits.
    Floats {
        /// The values.
        values: Vec<f32>,
        /// The largest difference allowed.
        tolerance: f32,
    },
}

impl Values {
    /// The bytes the values cover.
    pub fn length(&self) -> u64 {
        let count = match self {
            Values::Words(words) => words.len(),
            Values::Floats { values, .. } => values.len(),
        };
        4 * count as u64
    }

    /// `Ok` where `bytes` (little-endian) holds the values; else how many differ and the first.
    pub fn compare(&self, bytes: &[u8]) -> Result<()> {
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w));
        let (mut differ, mut first) = (0usize, None);
        let mut note = |index: usize, got: String, want: String| {
            differ += 1;
            first.get_or_insert((index, got, want));
        };
        match self {
            Values::Words(want) => {
                for (index, (got, &want)) in words.zip(want).enumerate() {
                    if got != want {
                        note(index, format!("{got:#x}"), format!("{want:#x}"));
                    }
                }
            }
            Values::Floats { values, tolerance } => {
                for (index, (got, &want)) in words.map(f32::from_bits).zip(values).enumerate() {
                    // The difference is false for a NaN on either side.
                    let within =
                        (got - want).abs() <= *tolerance || (got.is_nan() && want.is_nan());
                    if !within {
                        note(index, format!("{got}"), format!("{want} +- {tolerance}"));
                    }
                }
            }
        }
        match first {
            None => Ok(()),
            Some((index, got, want)) => Err(format!(
                "{differ} values differ; the first, [{index}], is {got}, not {want}"
            )
            .into()),
        }
    }
}

impl Plan {
    /// `Ok` where every index it names exists and every length is whole and agrees.
    pub fn validate(&self) -> Result<()> {
        let name = &self.name;
        for (i, buffer) in self.buffers.iter().enumerate() {
            if buffer.length == 0 || buffer.length % 16 != 0 {
                return Err(format!(
                    "{name}: buffer {i}'s length {} is not a positive multiple of 16",
                    buffer.length
                )
                .into());
            }
            if let Some(contents) = &buffer.contents {
                if contents.len() as u64 != buffer.length {
                    return Err(format!(
                        "{name}: buffer {i} has {} bytes of contents for {}",
                        contents.len(),
                        buffer.length
                    )
                    .into());
                }
            }
        }
        for (i, texture) in self.textures.iter().enumerate() {
            let (width, height) = (texture.width, texture.height);
            if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
                return Err(format!(
                    "{name}: texture {i}'s size {width}x{height} is not in whole 16x16 tiles"
                )
                .into());
            }
            if let Some(texels) = &texture.texels {
                if texels.len() as u64 != 4 * u64::from(width) * u64::from(height) {
                    return Err(format!(
                        "{name}: texture {i} has {} floats of texels for {width}x{height}",
                        texels.len()
                    )
                    .into());
                }
            }
        }
        let known = |buffer: usize| buffer < self.buffers.len();
        for (i, dispatch) in self.dispatches.iter().enumerate() {
            for binding in &dispatch.bindings {
                let (kind, at, count) = match binding {
                    Binding::Buffer { buffer, .. } => ("buffer", *buffer, self.buffers.len()),
                    Binding::Texture { texture, .. } => ("texture", *texture, self.textures.len()),
                    Binding::Bytes { .. } | Binding::Sampler { .. } => continue,
                };
                if at >= count {
                    return Err(format!(
                        "{name}: dispatch {i} binds {kind} {at}, which the plan lacks"
                    )
                    .into());
                }
            }
        }
        for expect in &self.expect {
            if !known(expect.buffer) {
                return Err(format!(
                    "{name}: an expectation names buffer {}, which the plan lacks",
                    expect.buffer
                )
                .into());
            }
            if expect.values.length() > self.buffers[expect.buffer].length {
                return Err(format!(
                    "{name}: buffer {} is shorter than its expected values",
                    expect.buffer
                )
                .into());
            }
        }
        for expect in &self.expect_textures {
            let Some(texture) = self.textures.get(expect.texture) else {
                return Err(format!(
                    "{name}: an expectation names texture {}, which the plan lacks",
                    expect.texture
                )
                .into());
            };
            if expect.values.length() > 16 * u64::from(texture.width) * u64::from(texture.height) {
                return Err(format!(
                    "{name}: texture {} is smaller than its expected values",
                    expect.texture
                )
                .into());
            }
        }
        for output in &self.outputs {
            if !known(output.buffer) {
                return Err(format!(
                    "{name}: an output names buffer {}, which the plan lacks",
                    output.buffer
                )
                .into());
            }
        }
        if self.dispatches.is_empty() || self.per_command_buffer == Some(0) {
            return Err(format!("{name}: its command buffers hold no dispatch").into());
        }
        Ok(())
    }
}

/// `count` words of SplitMix64 from `seed`, the low half of each output: the harness's inputs.
pub fn seeded_words(seed: u64, count: usize) -> Vec<u32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) as u32
        })
        .collect()
}
