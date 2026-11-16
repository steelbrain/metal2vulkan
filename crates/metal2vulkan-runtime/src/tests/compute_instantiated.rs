//! A compute encoder's dispatch takes the commands its kept bind state kept of what it planned
//! (`compute::Instantiated`, kept with the state, which goes on to the device's next compute
//! encoder), and instantiates them again once a kind of value they may read moved, with the
//! answer a fresh instantiation gives (builds with debug assertions instantiate again at each
//! dispatch that takes them, and compare):
//!
//! - **Texture bindings a plan does not read**: two textures of one shape in turn, an encoder for
//!   each dispatch, as a guest compute benchmark's workloads bind them: the second dispatch on takes the
//!   commands (`WorkCounts::dispatches_instantiated_as_before`), and each samples its own texture.
//! - **A buffer's binding**: the output bound at another offset instantiates again
//!   (`dispatches_instantiated`) and writes there.
//! - **A field**: a buffer made resident (`use_resource`), which the state lists among its used
//!   resources, instantiates again.
//! - **Texture bindings a plan reads**: texture buffers over one buffer at two offsets, read
//!   through a texel-buffer view whose range the commands state, instantiate again at each change,
//!   and each dispatch reads its own texels.
//!
//! Every dispatch is the only one of its compute encoder, in a command buffer of its own, waited
//! for before the next.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernels are owned fixtures under
//! `validation/fixtures/public/engine/` (`llvm-as`): `fpv_swz_sample` of `fpv_swizzled_reads.ll`
//! (`o[i] = t.sample(s, p[i])`, its texture and sampler at 0, `o` at buffer 0 and `p` at 1) and
//! `ob1_texel_buffer` of `ob1_texel_buffer.ll` (thread `t` reads texel `idx[t]` of the texture
//! buffer at 0 into `out[t]` and writes `0xA0000000 | t` to texel `idx[8 + t]` of the one at 1;
//! `idx` at buffer 0, `out` at 1).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, ComputeCommandEncoder, ComputePipelineState, Device,
    PipelineOptions, Region, ResourceOptions, SamplerState, Size, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};

/// The state of `function` of the fixture at `path`.
fn state(device: &Device, path: &str, function: &str) -> ComputePipelineState {
    let function = library(device, path)
        .new_function_with_name(function)
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// The dispatches whose commands were instantiated, and those that took kept ones.
fn counts(device: &Device) -> [u64; 2] {
    let counts = device.work_counts();
    [
        counts.dispatches_instantiated,
        counts.dispatches_instantiated_as_before,
    ]
}

/// A `Shared` buffer of `words`.
fn words(device: &Device, words: &[u32]) -> Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let (from, length) = (bytes.as_ptr(), bytes.len());
    // SAFETY: the new buffer's mapped bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(from, buffer.contents().unwrap().as_ptr(), length) };
    buffer
}

/// The `count` words of `buffer` from byte `offset`.
fn read(buffer: &Buffer, offset: u64, count: usize) -> Vec<u32> {
    // SAFETY: inside the buffer, which no command uses any more.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            buffer.contents().unwrap().as_ptr().add(offset as usize),
            4 * count,
        )
    };
    let words = bytes.chunks(4);
    words
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

/// Encode, in a compute encoder of its own in a command buffer of its own of `queue`, the
/// dispatch `encode` binds and makes, and wait for it: the counts its encoding moved.
fn dispatch(queue: &CommandQueue, encode: impl FnOnce(&mut ComputeCommandEncoder)) -> [u64; 2] {
    let device = queue.device();
    let before = counts(device);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    let after = counts(device);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    [0, 1].map(|at| after[at] - before[at])
}

/// A 2 by 1 `R32Float` texture of `texels`.
fn texels(device: &Device, texels: [f32; 2]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, 2, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = texels.into_iter().flat_map(f32::to_le_bytes).collect();
    (texture.replace_region(Region::new_2d(0, 0, 2, 1), 0, 0, &bytes, 8, 0)).unwrap();
    texture
}

/// What a sampling dispatch binds: its texture, its output at an offset, and a buffer it makes
/// resident, where it makes one.
struct Sampling<'a> {
    texture: &'a Texture,
    out: (&'a Buffer, u64),
    resident: Option<&'a Buffer>,
}

/// The sampling kernel's state, its nearest sampler state and the coordinates `(0.75, 0.5)`, of
/// texel 1.
struct Sampler {
    state: ComputePipelineState,
    nearest: SamplerState,
    coords: Buffer,
}

impl Sampler {
    fn new(device: &Device) -> Sampler {
        let coords = [0.75f32.to_bits(), 0.5f32.to_bits()];
        Sampler {
            state: state(device, "engine/fpv_swizzled_reads.ll", "fpv_swz_sample"),
            nearest: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
            coords: words(device, &coords),
        }
    }

    /// One thread sampling `bound.texture` into `bound.out`.
    fn dispatch(&self, queue: &CommandQueue, bound: &Sampling<'_>) -> [u64; 2] {
        dispatch(queue, |encoder| {
            encoder.set_compute_pipeline_state(&self.state).unwrap();
            encoder.set_texture(Some(bound.texture), 0).unwrap();
            encoder.set_sampler_state(Some(&self.nearest), 0).unwrap();
            let (out, offset) = bound.out;
            encoder.set_buffer(Some(out), offset, 0).unwrap();
            encoder.set_buffer(Some(&self.coords), 0, 1).unwrap();
            if let Some(resident) = bound.resident {
                let resource = resident.as_resource();
                encoder.use_resource(resource, ResourceUsage::READ).unwrap();
            }
            let one = Size::new(1, 1, 1);
            encoder.dispatch_threads(one, one).unwrap();
        })
    }
}

/// The red `out` holds at `offset`.
fn red(out: &Buffer, offset: u64) -> f32 {
    f32::from_bits(read(out, offset, 1)[0])
}

/// Two textures of one shape in turn, an encoder each: the first dispatch instantiates, each
/// later one takes its commands, and each samples its own texture.
#[test]
fn textures_bound_in_turn_take_the_kept_commands() {
    let scratch = Scratch::new("instantiated-textures");
    let device = device(&scratch.0);
    let sampler = Sampler::new(&device);
    let (a, b) = (texels(&device, [0.0, 1.0]), texels(&device, [2.0, 3.0]));
    let out = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    for (at, (texture, texel)) in [(&a, 1.0), (&b, 3.0), (&a, 1.0), (&b, 3.0)]
        .into_iter()
        .enumerate()
    {
        let bound = Sampling {
            texture,
            out: (&out, 0),
            resident: None,
        };
        let moved = sampler.dispatch(&queue, &bound);
        let want = if at == 0 { [1, 0] } else { [0, 1] };
        assert_eq!(moved, want, "dispatch {at}");
        assert_eq!(
            red(&out, 0),
            texel,
            "dispatch {at}: texel 1 of its own texture"
        );
    }
}

/// The output bound at another offset instantiates again and is written there; at that offset
/// again, the commands are kept.
#[test]
fn a_buffer_bound_at_another_offset_instantiates_again() {
    let scratch = Scratch::new("instantiated-offset");
    let device = device(&scratch.0);
    let sampler = Sampler::new(&device);
    let texture = texels(&device, [0.0, 1.0]);
    let out = device.new_buffer(512, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    for (at, (offset, want)) in [(0, [1, 0]), (256, [1, 0]), (256, [0, 1])]
        .into_iter()
        .enumerate()
    {
        let bound = Sampling {
            texture: &texture,
            out: (&out, offset),
            resident: None,
        };
        assert_eq!(sampler.dispatch(&queue, &bound), want, "dispatch {at}");
        assert_eq!(red(&out, offset), 1.0, "dispatch {at}: texel 1 at {offset}");
    }
}

/// A buffer made resident, which the state lists among its used resources, instantiates again;
/// the same buffer made resident again keeps the commands.
#[test]
fn a_resource_made_resident_instantiates_again() {
    let scratch = Scratch::new("instantiated-resident");
    let device = device(&scratch.0);
    let sampler = Sampler::new(&device);
    let texture = texels(&device, [0.0, 1.0]);
    let out = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let resident = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    for (at, (made, want)) in [
        (None, [1, 0]),
        (Some(&resident), [1, 0]),
        (Some(&resident), [0, 1]),
    ]
    .into_iter()
    .enumerate()
    {
        let bound = Sampling {
            texture: &texture,
            out: (&out, 0),
            resident: made,
        };
        assert_eq!(sampler.dispatch(&queue, &bound), want, "dispatch {at}");
        assert_eq!(red(&out, 0), 1.0, "dispatch {at}: texel 1");
    }
}

/// A plan that reads its texture slots' values (a texel-buffer view's range): texture buffers of
/// one shape over one buffer at two offsets, in turn, an encoder each, instantiate again at each
/// dispatch, and each reads its own texels.
#[test]
fn texture_buffers_bound_in_turn_instantiate_again() {
    let scratch = Scratch::new("instantiated-texel-buffers");
    let device = device(&scratch.0);
    let state = state(&device, "engine/ob1_texel_buffer.ll", "ob1_texel_buffer");
    let texels: Vec<u32> = (0..128).map(|at| 0x5000_0000 | at).collect();
    let texels = words(&device, &texels);
    let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let descriptor = TextureDescriptor::texture_buffer(
        PixelFormat::R32Uint,
        8,
        texels.resource_options(),
        usage,
    );
    let over = |buffer: &Buffer, offset| buffer.new_texture(&descriptor, offset, 32).unwrap();
    // Texels 0 to 7 and 64 to 71 of the words; the written one apart.
    let (a, b) = (over(&texels, 0), over(&texels, 256));
    let written_words = device.new_buffer(32, ResourceOptions::shared()).unwrap();
    let written = over(&written_words, 0);
    // Each thread reads its own texel and writes its own.
    let idx: Vec<u32> = (0..8).chain(0..8).collect();
    let idx = words(&device, &idx);
    let out = device.new_buffer(32, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    for (at, (texture, first)) in [(&a, 0u32), (&b, 64), (&a, 0)].into_iter().enumerate() {
        let moved = dispatch(&queue, |encoder| {
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_buffer(Some(&idx), 0, 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            encoder.set_texture(Some(texture), 0).unwrap();
            encoder.set_texture(Some(&written), 1).unwrap();
            let threads = Size::new(8, 1, 1);
            encoder.dispatch_threads(threads, threads).unwrap();
        });
        assert_eq!(moved, [1, 0], "dispatch {at}");
        let want: Vec<u32> = (first..first + 8).map(|at| 0x5000_0000 | at).collect();
        assert_eq!(
            read(&out, 0, 8),
            want,
            "dispatch {at}: its own texture's texels"
        );
    }
}
