//! Textures of an address-mode map and of an alpha that reads 1 (runtime extensions,
//! `TextureDescriptor::address_mode_map` and `alpha_reads_one`) on the device the cfg selects,
//! under the validation layer.
//!
//! Metal has neither property: a consumer states with them the measured rules of formats it
//! presents through other textures. Its measurements (an M3 Max, nearest and linear samples and
//! gathers of red and blue, inside and outside `[0, 1]`) gave two such rules: one format samples
//! mirror clamp to edge, repeat and mirrored repeat exactly as clamp to edge and keeps the border
//! modes; another samples all five non-default modes as clamp to edge. The runtime's part is the
//! mechanism, measured here on an owned format against itself:
//!
//! - a texture of a map samples, bit for bit, as the same texels unmapped through the sampler of
//!   the mapped modes, through the same pipeline variant (its SPIR-V) and the same bind shape
//!   but for the map the shape states, so every emulation that reads the mode follows; a map that
//!   leaves the bound modes as they are changes neither. The same holds for a fragment function's
//!   texture sampled through a runtime sampler slot, the consumer's path, and for one encoder's
//!   draws that rebind that slot between a mapped texture and its texels unmapped;
//! - a constexpr sampler whose modes a mapped texture's map changes samples it through its slot's
//!   pipeline variant, its immutable sampler holding the mapped modes; one whose modes the map
//!   keeps samples the texture as an unmapped one, through the unmapped texture's pipeline; a map
//!   sending a mode where no constexpr sampler addresses is refused, typed, naming the map;
//! - a texture whose alpha reads 1 samples and gathers, inside and at every border, as the
//!   consumer's present form, the same texels through a `(R, G, B, One)` swizzle, and its writes
//!   and render target keep the alpha written. Its reads answer alpha 1 for every one of the
//!   65,536 `half` codes its alpha can hold, each of which stays the texture's own code.
//!
//! The kernels are owned fixtures' (`rrp_mirror_clamp`'s `rrp_mc_sample` and `rrp_mc_gather`:
//! five results a lane, with the texel offsets 0, -3, 2, -7 and 5 on `s`; `a2_clamp_to_zero`;
//! `rrp_swizzled_storage`'s reads and write; `tob_renders`'s `tob_rect` and `tob_sample`).

use super::degrade_notices::on_device;
use super::degrade_notices::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, SamplerAddressMode as Mode, SamplerBorderColor,
    SamplerMinMagFilter, TextureSwizzle, TextureSwizzleChannels, TextureUsage,
};
use metal2vulkan::pipeline::{
    BindingSource, ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor,
    StageSource,
};
use metal2vulkan::vk;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    AddressModeMap, ClearColor, CommandBufferStatus, ComputePipelineState, Device, Error,
    ExecutedPipeline, LoadAction, MappedSampling, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, RuntimeGap, SamplerState, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The texture's extent: wide enough that the offsets move a tap (a width of 3, 5, 6 or 7 texels
/// drops them), and more than one row, so `t` wraps and clamps apart too.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 4;

/// The lanes of a threadgroup.
const GROUP: u64 = 32;

/// The consumer's first rule: mirror clamp, repeat and mirrored repeat as clamp to edge.
fn edges() -> AddressModeMap {
    AddressModeMap::IDENTITY
        .with(Mode::MirrorClampToEdge, Mode::ClampToEdge)
        .with(Mode::Repeat, Mode::ClampToEdge)
        .with(Mode::MirrorRepeat, Mode::ClampToEdge)
}

/// The consumer's second: every mode as clamp to edge.
fn all_edges() -> AddressModeMap {
    edges()
        .with(Mode::ClampToZero, Mode::ClampToEdge)
        .with(Mode::ClampToBorderColor, Mode::ClampToEdge)
}

/// The `half` bits of `x`, exact for the values here.
fn half(x: f32) -> u16 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if x == 0.0 {
        return 0;
    }
    assert!(
        (1..31).contains(&exp) && bits & 0x1fff == 0,
        "{x} is no exact half"
    );
    (((bits >> 16) & 0x8000) | ((exp as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16
}

/// Texel `(j, i)`'s value: red its column, green its row, blue both, alpha one eighth, every one
/// exact in `half`.
fn texel(j: u32, i: u32) -> [f32; 4] {
    [
        j as f32 + 1.0,
        i as f32 + 1.0,
        (j * HEIGHT + i) as f32 / 32.0,
        0.125,
    ]
}

/// An `RGBA16Float` texture of [`texel`]s, of `usage`, made with `edit` applied.
fn texture(
    device: &Device,
    usage: TextureUsage,
    edit: impl FnOnce(&mut TextureDescriptor),
) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, WIDTH.into(), HEIGHT.into(), false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::shared();
    edit(&mut descriptor);
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..HEIGHT)
        .flat_map(|i| (0..WIDTH).flat_map(move |j| texel(j, i)))
        .flat_map(|x| half(x).to_le_bytes())
        .collect();
    texture
        .replace_region(
            Region::new_2d(0, 0, WIDTH.into(), HEIGHT.into()),
            0,
            0,
            &bytes,
            u64::from(WIDTH) * 8,
            0,
        )
        .unwrap();
    texture
}

fn state(device: &Device, fixture: &str, kernel: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join(fixture)).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// The lanes' normalized coordinates: `s` from two widths left of the texture to two right, at
/// a quarter texel past each centre, and `t` from below it to above it.
fn coordinates() -> Vec<[f32; 2]> {
    let n = i64::from(WIDTH);
    let lanes: Vec<[f32; 2]> = (-2 * n..2 * n)
        .flat_map(|c| (0..4).map(move |k| (c, k)))
        .enumerate()
        .map(|(lane, (c, k))| {
            let s = (c as f32 + 0.5 + k as f32 / 4.0) / WIDTH as f32;
            let t = ((lane % 11) as f32 - 2.75) / HEIGHT as f32;
            [s, t]
        })
        .collect();
    assert_eq!(lanes.len() as u64 % GROUP, 0, "whole threadgroups");
    lanes
}

/// A sampler of `filter` addressing every axis by `mode`, its border `border`.
fn sampler(
    mode: Mode,
    filter: SamplerMinMagFilter,
    border: SamplerBorderColor,
) -> SamplerDescriptor {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.s_address_mode = mode;
    descriptor.t_address_mode = mode;
    descriptor.r_address_mode = mode;
    descriptor.border_color = border;
    descriptor
}

/// The states of the sampler of `mode` and of the sampler of the mode `map` maps it to, made
/// together so that both live through both commands that compare them: the device's sampler
/// states are part of every command's bind shape (the argument samplers' shapes), so a state made
/// for one command alone (a mirror-clamp one) would plan the other apart.
fn states(
    device: &Device,
    mode: Mode,
    map: AddressModeMap,
    filter: SamplerMinMagFilter,
    border: SamplerBorderColor,
) -> Result<(SamplerState, SamplerState), Error> {
    let through = device.new_sampler_state(&sampler(mode, filter, border))?;
    let direct = device.new_sampler_state(&sampler(map.mode(mode), filter, border))?;
    Ok((through, direct))
}

/// The sampling kernels of `rrp_mirror_clamp`, by name.
fn kernels(device: &Device) -> [(&'static str, ComputePipelineState); 2] {
    ["rrp_mc_sample", "rrp_mc_gather"].map(|k| (k, state(device, "rrp_mirror_clamp.ll", k)))
}

/// What one dispatch read: the five results of each lane as bits, and the executed variant's
/// SPIR-V and bind shape (but for the texture's map, `unmapped_shape`).
#[derive(PartialEq)]
struct Sampled {
    texels: Vec<[u32; 4]>,
    spirv: Vec<u32>,
    shape: String,
}

/// What one dispatch of `state` (`rrp_mc_sample`'s or `rrp_mc_gather`'s) read through `sampler`
/// on `texture`. `Err` for the dispatch's refusal.
fn sampled(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Result<Sampled, Error> {
    let lanes = coordinates();
    let coords: Vec<u8> = lanes
        .iter()
        .flat_map(|&[s, t]| [s, t, 0.0, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let coords = buffer_with(device, &coords).unwrap();
    let count = lanes.len() as u64;
    let out = buffer_with(device, &vec![0xab; (count * 5 * 16) as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(sampler), 0).unwrap();
    encoder.set_buffer(Some(&coords), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    let dispatched = encoder.dispatch_threads(Size::new(count, 1, 1), Size::new(GROUP, 1, 1));
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    dispatched?;
    let bytes = read_buffer(&out, 0, count * 5 * 16).unwrap();
    let texels = bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| {
            let w = |i: usize| u32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]);
            [w(0), w(4), w(8), w(12)]
        })
        .collect();
    let executions = command_buffer.executions();
    assert_eq!(executions.len(), 1);
    let ExecutedPipeline::Compute(pipeline) = executions[0].pipeline() else {
        panic!("a dispatch executed no compute pipeline");
    };
    let shape = unmapped_shape(executions[0].shape(), texture);
    Ok(Sampled {
        texels,
        spirv: pipeline.stage.spirv.to_vec(),
        shape,
    })
}

/// `shape`'s `Debug` with `texture`'s address-mode map read as the identity: the shape states
/// the bound texture's map (`TextureShape::Texture::address_modes`, which a slot a constexpr
/// sampler samples compiles under), so a mapped texture's shape is the unmapped one's but for it.
fn unmapped_shape(shape: &impl std::fmt::Debug, texture: &Texture) -> String {
    let named = |map: AddressModeMap| format!("address_modes: {map:?}");
    format!("{shape:?}").replace(
        &named(texture.address_mode_map()),
        &named(AddressModeMap::IDENTITY),
    )
}

/// Whether `error` is a typed refusal, a device's or the runtime's.
fn refused(error: &Error) -> bool {
    matches!(
        error,
        Error::Unsupported(_) | Error::NotImplemented(_) | Error::Bind(_)
    )
}

/// A mapped texture samples and gathers, every lane bit for bit, as its texels unmapped through
/// the sampler of the mapped modes, on the same variant and shape: each of the consumer's two
/// rules, every mode, nearest and linear, two border colours. Prints each refusal and each case
/// that differs; the lanes reach past the texture, so repeating and clamping read apart.
#[test]
fn a_mapped_texture_samples_as_its_mapped_sampler() {
    on_device_with(at_plan, |device, _| {
        let kernels = kernels(device);
        let plain = texture(device, TextureUsage::SHADER_READ, |_| {});
        let mut wrong = Vec::new();
        let (mut checked, mut refusals) = (0usize, 0usize);
        for (rule, map) in [("edges", edges()), ("all edges", all_edges())] {
            let mapped = texture(device, TextureUsage::SHADER_READ, |d| {
                d.address_mode_map = map;
            });
            assert_eq!(mapped.address_mode_map(), map);
            assert!(plain.address_mode_map().is_identity());
            for (kernel, state) in &kernels {
                for filter in [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear] {
                    for border in [
                        SamplerBorderColor::TransparentBlack,
                        SamplerBorderColor::OpaqueWhite,
                    ] {
                        for &mode in Mode::ALL {
                            let at = format!("{rule} {kernel} {filter:?} {border:?} {mode:?}");
                            let (through, direct) = match states(device, mode, map, filter, border)
                            {
                                Ok(states) => states,
                                // A sampler state the device cannot make is no case of the map's.
                                Err(error) => {
                                    assert!(refused(&error), "{at}: {error:?}");
                                    println!(
                                        "MAP-REFUSED {}: {at}: the sampler: {error:?}",
                                        device.name()
                                    );
                                    refusals += 1;
                                    continue;
                                }
                            };
                            match (
                                sampled(device, state, &mapped, &through),
                                sampled(device, state, &plain, &direct),
                            ) {
                                (Ok(got), Ok(want)) => {
                                    checked += 1;
                                    if got.texels != want.texels {
                                        wrong.push(format!("{at}: the texels differ"));
                                    }
                                    if got.spirv != want.spirv || got.shape != want.shape {
                                        wrong.push(format!("{at}: another variant or shape"));
                                    }
                                }
                                (Err(a), Err(b)) if refused(&a) && refused(&b) => {
                                    println!("MAP-REFUSED {}: {at}: {a:?}", device.name());
                                    refusals += 1;
                                }
                                (got, want) => wrong.push(format!(
                                    "{at}: mapped {:?}, direct {:?}",
                                    got.map(|_| ()),
                                    want.map(|_| ())
                                )),
                            }
                        }
                    }
                }
            }
        }
        // The lanes decide: unmapped, repeating reads other texels than clamping.
        let filter = SamplerMinMagFilter::Nearest;
        let border = SamplerBorderColor::TransparentBlack;
        let sample = &kernels[0].1;
        let made = |mode| {
            device
                .new_sampler_state(&sampler(mode, filter, border))
                .unwrap()
        };
        let (repeat, clamp) = (made(Mode::Repeat), made(Mode::ClampToEdge));
        let repeat = sampled(device, sample, &plain, &repeat);
        let clamp = sampled(device, sample, &plain, &clamp);
        assert!(
            repeat.unwrap().texels != clamp.unwrap().texels,
            "the lanes stay inside"
        );
        for line in &wrong {
            println!("MAP-DIFFERS {}: {line}", device.name());
        }
        println!(
            "MAP {}: {checked} cases checked, {} differ, {refusals} refused alike",
            device.name(),
            wrong.len()
        );
        assert!(checked > 0);
        assert!(wrong.is_empty(), "{} cases differ", wrong.len());
    });
}

/// The consumer's path: a fragment function's texture sampled through a runtime sampler slot
/// (`tob_renders`'s `tob_sample`, at the pixel's centre over the texture's size, into a target
/// four textures wide and three high, so it reads past the right and top edges). A mapped
/// texture draws, bit for bit, what the same texels unmapped draw through the sampler of the
/// mapped modes, through the same variant and bind shape; unmapped, repeating draws other texels
/// than clamping.
#[test]
fn a_fragments_runtime_sampler_takes_its_textures_map() {
    on_device_with(at_plan, |device, _| {
        let pipeline = sampling_pipeline(device);
        let plain = texture(device, TextureUsage::SHADER_READ, |_| {});
        let mut wrong = Vec::new();
        let (mut checked, mut refusals) = (0usize, 0usize);
        for (rule, map) in [("edges", edges()), ("all edges", all_edges())] {
            let mapped = texture(device, TextureUsage::SHADER_READ, |d| {
                d.address_mode_map = map;
            });
            for filter in [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear] {
                for border in [
                    SamplerBorderColor::TransparentBlack,
                    SamplerBorderColor::OpaqueWhite,
                ] {
                    for &mode in Mode::ALL {
                        let at = format!("fragment {rule} {filter:?} {border:?} {mode:?}");
                        let (through, direct) = match states(device, mode, map, filter, border) {
                            Ok(states) => states,
                            // A sampler state the device cannot make is no case of the map's.
                            Err(error) => {
                                assert!(refused(&error), "{at}: {error:?}");
                                println!(
                                    "MAP-REFUSED {}: {at}: the sampler: {error:?}",
                                    device.name()
                                );
                                refusals += 1;
                                continue;
                            }
                        };
                        match (
                            drawn(device, &pipeline, &mapped, &through),
                            drawn(device, &pipeline, &plain, &direct),
                        ) {
                            (Ok(got), Ok(want)) => {
                                checked += 1;
                                if got.texels != want.texels {
                                    wrong.push(format!("{at}: the texels differ"));
                                }
                                if got.spirv != want.spirv || got.shape != want.shape {
                                    wrong.push(format!("{at}: another variant or shape"));
                                }
                            }
                            (Err(a), Err(b)) if refused(&a) && refused(&b) => {
                                println!("MAP-REFUSED {}: {at}: {a:?}", device.name());
                                refusals += 1;
                            }
                            (got, want) => wrong.push(format!(
                                "{at}: mapped {:?}, direct {:?}",
                                got.map(|_| ()),
                                want.map(|_| ())
                            )),
                        }
                    }
                }
            }
        }
        let filter = SamplerMinMagFilter::Nearest;
        let border = SamplerBorderColor::TransparentBlack;
        let made = |mode| {
            device
                .new_sampler_state(&sampler(mode, filter, border))
                .unwrap()
        };
        let (repeat, clamp) = (made(Mode::Repeat), made(Mode::ClampToEdge));
        let repeat = drawn(device, &pipeline, &plain, &repeat);
        let clamp = drawn(device, &pipeline, &plain, &clamp);
        assert!(
            repeat.unwrap().texels != clamp.unwrap().texels,
            "the pixels stay inside"
        );
        for line in &wrong {
            println!("MAP-DIFFERS {}: {line}", device.name());
        }
        println!(
            "MAP {}: fragment: {checked} cases checked, {} differ, {refusals} refused alike",
            device.name(),
            wrong.len()
        );
        assert!(checked > 0);
        assert!(wrong.is_empty(), "{} cases differ", wrong.len());
    });
}

/// A pipeline state of `tob_renders`'s `tob_rect` (a rectangle of the corners at vertex buffer 0)
/// and `tob_sample` into an `RGBA32Float` color attachment.
fn sampling_pipeline(device: &Device) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("tob_renders.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_sample"));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// One encoder's draws that rebind the fragment texture slot in turn to a mapped texture and to
/// the same texels unmapped, alike but for the map, through one repeating sampler state (the
/// consumer's second rule maps repeat to clamp to edge): each texture draws twice in a row over
/// its half of the target, so a bind takes the view and the sampler its write took last, and
/// after each rebind those an earlier bind took, the sampler's modes mapped or not. Each half is,
/// bit for bit, what a draw of the unmapped texels over the whole target draws there through the
/// mode its texture is sampled with ([`drawn`]): clamp to edge for the mapped texture, repeat for
/// the other; and the two modes draw other texels in each half.
#[test]
fn draws_rebinding_a_slot_between_a_texture_and_its_mapped_twin_keep_each_view_and_sampler() {
    on_device_with(at_plan, |device, _| {
        let pipeline = sampling_pipeline(device);
        let plain = texture(device, TextureUsage::SHADER_READ, |_| {});
        let mapped = texture(device, TextureUsage::SHADER_READ, |d| {
            d.address_mode_map = all_edges();
        });
        assert_eq!(all_edges().mode(Mode::Repeat), Mode::ClampToEdge);
        let filter = SamplerMinMagFilter::Nearest;
        let border = SamplerBorderColor::TransparentBlack;
        let (repeat, clamp) = states(device, Mode::Repeat, all_edges(), filter, border).unwrap();
        let (w, h) = (u64::from(4 * WIDTH), u64::from(3 * HEIGHT));
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, w, h, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target = device.new_texture(&target).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&repeat), 0)
            .unwrap();
        let halves = [
            (&mapped, [-1.0f32, -1.0, 0.0, 1.0]),
            (&plain, [0.0, -1.0, 1.0, 1.0]),
        ];
        for _ in 0..3 {
            for (texture, corners) in &halves {
                let corners: Vec<u8> = corners.iter().flat_map(|v| v.to_le_bytes()).collect();
                encoder.set_vertex_bytes(&corners, 0).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
                for _ in 0..2 {
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                }
            }
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        let mut bytes = vec![0u8; (16 * w * h) as usize];
        target
            .get_bytes(&mut bytes, 16 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
            .unwrap();
        let got: Vec<[u32; 4]> = bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|v| {
                let w = |i: usize| u32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]);
                [w(0), w(4), w(8), w(12)]
            })
            .collect();
        let clamped = drawn(device, &pipeline, &plain, &clamp).unwrap().texels;
        let repeated = drawn(device, &pipeline, &plain, &repeat).unwrap().texels;
        // The left half's pixels, or the right half's, by row.
        let half = |texels: &[[u32; 4]], right: bool| -> Vec<[u32; 4]> {
            (texels.iter().enumerate())
                .filter(|&(at, _)| (at as u64 % w >= w / 2) == right)
                .map(|(_, &texel)| texel)
                .collect()
        };
        for right in [false, true] {
            assert_ne!(
                half(&clamped, right),
                half(&repeated, right),
                "right {right}"
            );
        }
        assert_eq!(
            half(&got, false),
            half(&clamped, false),
            "the mapped texture's half"
        );
        assert_eq!(
            half(&got, true),
            half(&repeated, true),
            "the unmapped texture's half"
        );
    });
}

/// What one draw of `pipeline` (`tob_rect` over the whole target, `tob_sample`) into an
/// `RGBA32Float` target four textures wide and three high read from `texture` through `sampler`:
/// each pixel's bits, and the executed variant's SPIR-V and bind shape. `Err` for the draw's
/// refusal.
fn drawn(
    device: &Device,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Result<Sampled, Error> {
    let (w, h) = (u64::from(4 * WIDTH), u64::from(3 * HEIGHT));
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, w, h, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let corners: Vec<u8> = [-1.0f32, -1.0, 1.0, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let corners = buffer_with(device, &corners).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(sampler), 0)
        .unwrap();
    let drew = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    drew?;
    let mut bytes = vec![0u8; (16 * w * h) as usize];
    target
        .get_bytes(&mut bytes, 16 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
        .unwrap();
    let texels = bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| {
            let w = |i: usize| u32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]);
            [w(0), w(4), w(8), w(12)]
        })
        .collect();
    let executions = command_buffer.executions();
    assert_eq!(executions.len(), 1);
    let ExecutedPipeline::Render(pipeline) = executions[0].pipeline() else {
        panic!("a draw executed no render pipeline");
    };
    let shape = unmapped_shape(executions[0].shape(), texture);
    Ok(Sampled {
        texels,
        spirv: pipeline
            .stages
            .iter()
            .flat_map(|stage| stage.spirv.iter().copied())
            .collect(),
        shape,
    })
}

/// A constexpr sampler samples a mapped texture through its slot's pipeline variant, its modes
/// mapped (`a2_clamp_to_zero`'s clamp-to-zero samplers, the float texture at slot 0 mapped):
/// under the consumer's second rule the variant's immutable samplers of slot 0 clamp to the
/// edge, under the first (which keeps clamp-to-zero) and the identity they clamp to the border,
/// and the first reads, word for word, what the unmapped texture reads; the integer textures'
/// samplers, of unmapped slots, are never mapped. Only the second rule runs a variant of its
/// own: the first keeps the slot's modes, so it keeps the unmapped texture's pipeline. A map
/// sending clamp-to-zero to mirror-clamp-to-edge, which no constexpr sampler states, is refused,
/// naming the map, before anything is recorded.
#[test]
fn a_constexpr_sampler_samples_a_mapped_texture_through_its_variant() {
    on_device_with(at_plan, |device, _| {
        let state = state(device, "a2_clamp_to_zero.ll", "a2_clamp_to_zero");
        let sampler = device
            .new_sampler_state(&sampler(
                Mode::ClampToEdge,
                SamplerMinMagFilter::Nearest,
                SamplerBorderColor::TransparentBlack,
            ))
            .unwrap();
        let integer = |format: PixelFormat| {
            let mut d = TextureDescriptor::texture_2d(format, 4, 4, false);
            d.usage = TextureUsage::SHADER_READ;
            device.new_texture(&d).unwrap()
        };
        let (uint, int) = (integer(PixelFormat::R32Uint), integer(PixelFormat::R32Sint));
        let coords = buffer_with(device, &[0u8; 32]).unwrap();
        let unstated = AddressModeMap::IDENTITY.with(Mode::ClampToZero, Mode::MirrorClampToEdge);
        let (mut read, mut keys) = (Vec::new(), Vec::new());
        for map in [all_edges(), edges(), AddressModeMap::IDENTITY, unstated] {
            let out = buffer_with(device, &[0xab; 4 * 5 * 16]).unwrap();
            let float = texture(device, TextureUsage::SHADER_READ, |d| {
                d.address_mode_map = map
            });
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_texture(Some(&float), 0).unwrap();
            encoder.set_texture(Some(&uint), 1).unwrap();
            encoder.set_texture(Some(&int), 2).unwrap();
            encoder.set_sampler_state(Some(&sampler), 0).unwrap();
            encoder.set_buffer(Some(&coords), 0, 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            let dispatched = encoder.dispatch_threads(Size::new(4, 1, 1), Size::new(4, 1, 1));
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            if map == unstated {
                assert_eq!(
                    dispatched,
                    Err(Error::NotImplemented(RuntimeGap::MappedTextureSampling {
                        map,
                        through: MappedSampling::ConstexprSampler,
                    }))
                );
                assert!(command_buffer.executions().is_empty());
                continue;
            }
            dispatched.unwrap();
            let executions = command_buffer.executions();
            assert_eq!(executions.len(), 1);
            let ExecutedPipeline::Compute(pipeline) = executions[0].pipeline() else {
                panic!("a dispatch executed no compute pipeline");
            };
            // The address modes of the immutable samplers of the texture slot `index`.
            let modes = |index: u32| -> Vec<_> {
                pipeline
                    .layout
                    .sets
                    .iter()
                    .flat_map(|set| &set.bindings)
                    .filter(|b| {
                        matches!(b.source, BindingSource::CombinedTextureSampler { texture, .. }
                            if texture.index == index)
                    })
                    .filter_map(|b| b.immutable_sampler)
                    .map(|d| [d.address_mode_u, d.address_mode_v, d.address_mode_w])
                    .collect()
            };
            let float_modes = modes(0);
            assert!(
                !float_modes.is_empty(),
                "{map:?}: no immutable sampler at slot 0"
            );
            let want = match map == all_edges() {
                true => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                false => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            };
            assert!(
                float_modes.iter().flatten().all(|&m| m == want),
                "{map:?}: {float_modes:?}"
            );
            for index in [1, 2] {
                assert!(
                    modes(index)
                        .iter()
                        .flatten()
                        .all(|&m| m == vk::SamplerAddressMode::CLAMP_TO_BORDER),
                    "{map:?}: slot {index}"
                );
            }
            keys.push(pipeline.key);
            if map != all_edges() {
                read.push(read_buffer(&out, 0, 4 * 5 * 16).unwrap());
            }
        }
        assert_eq!(
            read[0], read[1],
            "the map that keeps clamp-to-zero reads as the unmapped texture"
        );
        // The map that sends clamp-to-zero to the edge runs its own variant; the one that keeps
        // it states no map for the slot, so its draw keeps the unmapped texture's pipeline.
        assert_ne!(keys[0], keys[2]);
        assert_eq!(keys[1], keys[2]);
    });
}

/// `swizzle` from its four letters: `R`, `G`, `B`, `A`, `0` and `1`.
fn swizzle(s: &str) -> TextureSwizzleChannels {
    let c = |ch: char| match ch {
        'R' => TextureSwizzle::Red,
        'G' => TextureSwizzle::Green,
        'B' => TextureSwizzle::Blue,
        'A' => TextureSwizzle::Alpha,
        '0' => TextureSwizzle::Zero,
        _ => TextureSwizzle::One,
    };
    let v: Vec<TextureSwizzle> = s.chars().map(c).collect();
    TextureSwizzleChannels::new(v[0], v[1], v[2], v[3])
}

/// A texture whose alpha reads 1 samples and gathers, every lane bit for bit, as its texels
/// through a `(R, G, B, One)` swizzle, inside and at every border (each border colour, clamp to
/// zero, clamp to edge), nearest and linear, and so does such a texture made with a swizzle of
/// its own, against the swizzle that reads its alpha as 1. Its usage may hold `ShaderWrite` and
/// `RenderTarget`, which a swizzle's may not.
#[test]
fn a_texture_whose_alpha_reads_one_samples_as_the_swizzled_view() {
    on_device_with(at_plan, |device, _| {
        let kernels = kernels(device);
        let written =
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
        let mut wrong = Vec::new();
        let (mut checked, mut refusals) = (0usize, 0usize);
        for (own, read) in [("RGBA", "RGB1"), ("BGRA", "BGR1"), ("ARGB", "1RGB")] {
            let usage = if own == "RGBA" {
                written
            } else {
                TextureUsage::SHADER_READ
            };
            let alpha = texture(device, usage, |d| {
                d.alpha_reads_one = true;
                d.swizzle = swizzle(own);
            });
            assert!(alpha.alpha_reads_one());
            assert_eq!(alpha.swizzle(), swizzle(own));
            assert_eq!(alpha.usage(), usage);
            let view = texture(device, TextureUsage::SHADER_READ, |d| {
                d.swizzle = swizzle(read)
            });
            for (kernel, state) in &kernels {
                for filter in [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear] {
                    for (mode, border) in [
                        (Mode::ClampToEdge, SamplerBorderColor::TransparentBlack),
                        (Mode::ClampToZero, SamplerBorderColor::TransparentBlack),
                        (
                            Mode::ClampToBorderColor,
                            SamplerBorderColor::TransparentBlack,
                        ),
                        (Mode::ClampToBorderColor, SamplerBorderColor::OpaqueBlack),
                        (Mode::ClampToBorderColor, SamplerBorderColor::OpaqueWhite),
                    ] {
                        let at = format!("{own} {kernel} {filter:?} {mode:?} {border:?}");
                        let through = match device.new_sampler_state(&sampler(mode, filter, border))
                        {
                            Ok(through) => through,
                            Err(error) => {
                                assert!(refused(&error), "{at}: {error:?}");
                                println!(
                                    "ALPHA-REFUSED {}: {at}: the sampler: {error:?}",
                                    device.name()
                                );
                                refusals += 1;
                                continue;
                            }
                        };
                        match (
                            sampled(device, state, &alpha, &through),
                            sampled(device, state, &view, &through),
                        ) {
                            (Ok(got), Ok(want)) => {
                                checked += 1;
                                if got.texels != want.texels {
                                    wrong.push(format!("{at}: the texels differ"));
                                }
                            }
                            (Err(a), Err(b)) if refused(&a) && refused(&b) => {
                                println!("ALPHA-REFUSED {}: {at}: {a:?}", device.name());
                                refusals += 1;
                            }
                            (got, want) => wrong.push(format!(
                                "{at}: alpha {:?}, swizzled {:?}",
                                got.map(|_| ()),
                                want.map(|_| ())
                            )),
                        }
                    }
                }
            }
        }
        for line in &wrong {
            println!("ALPHA-DIFFERS {}: {line}", device.name());
        }
        println!(
            "ALPHA {}: {checked} cases checked, {} differ, {refusals} refused alike",
            device.name(),
            wrong.len()
        );
        assert!(checked > 0);
        assert!(wrong.is_empty(), "{} cases differ", wrong.len());
    });
}

/// The textures [`every_alpha_code_of_a_texture_whose_alpha_reads_one_reads_one`] reads in one
/// round: one dispatch each in one command buffer, so the 65,536 codes take 256 of them.
const CODES_AT_ONCE: usize = 256;

/// The bytes a round's read of one texture writes into the out buffer.
const CODE_STRIDE: usize = 256;

/// Every one of the 65,536 `half` codes an `RGBA16Float` texture's alpha can hold reads 1 through
/// a texture whose alpha reads 1, and stays the texture's own code: the code is written into texel
/// `(0, 0)` by `replace_region`, read back by `get_bytes`, and read by `rrp_sw_read_f` through the
/// slot the swizzle reaches ([`CODES_AT_ONCE`] textures a round, one pipeline). The swizzle's
/// `One` is the value 1 whatever code the texel holds, a NaN's included.
#[test]
fn every_alpha_code_of_a_texture_whose_alpha_reads_one_reads_one() {
    on_device(|device, _| {
        let textures: Vec<Texture> = (0..CODES_AT_ONCE)
            .map(|_| {
                texture(device, TextureUsage::SHADER_READ, |d| {
                    d.alpha_reads_one = true
                })
            })
            .collect();
        let state = state(device, "rrp_swizzled_storage.ll", "rrp_sw_read_f");
        let bytes = vec![0xab; CODES_AT_ONCE * CODE_STRIDE];
        let out = buffer_with(device, &bytes).unwrap();
        let queue = device.new_command_queue().unwrap();
        // The colour lanes, which the same swizzle takes from themselves, beside the alpha code.
        let colour = [1.5, 2.5, 0.5].map(half);
        let mut checked = 0u32;
        for round in 0..(0x1_0000 / CODES_AT_ONCE) {
            let code = |i: usize| (round * CODES_AT_ONCE + i) as u16;
            for (i, texture) in textures.iter().enumerate() {
                let mut texel = [0u8; 8];
                let lanes = [colour[0], colour[1], colour[2], code(i)];
                for (lane, bits) in texel.chunks_mut(2).zip(lanes) {
                    lane.copy_from_slice(&bits.to_le_bytes());
                }
                texture
                    .replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &texel, 8, 0)
                    .unwrap();
                assert_eq!(first_texel(texture), lanes, "code {:#06x}: stored", code(i));
            }
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            for (i, texture) in textures.iter().enumerate() {
                encoder.set_texture(Some(texture), 0).unwrap();
                encoder
                    .set_buffer(Some(&out), (i * CODE_STRIDE) as u64, 0)
                    .unwrap();
                encoder
                    .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(command_buffer.error(), None);
            let read = read_buffer(&out, 0, (CODES_AT_ONCE * CODE_STRIDE) as u64).unwrap();
            for i in 0..CODES_AT_ONCE {
                let at = i * CODE_STRIDE;
                let lanes = [0, 4, 8, 12].map(|k| {
                    let word = [
                        read[at + k],
                        read[at + k + 1],
                        read[at + k + 2],
                        read[at + k + 3],
                    ];
                    f32::from_le_bytes(word)
                });
                assert_eq!(lanes, [1.5, 2.5, 0.5, 1.0], "code {:#06x}: read", code(i));
                checked += 1;
            }
        }
        println!("ALPHA-CODES {}: {checked} codes read 1", device.name());
        assert_eq!(checked, 0x1_0000);
    });
}

/// Texel `(0, 0)` of `texture`, as its `half` bits.
fn first_texel(texture: &Texture) -> [u16; 4] {
    let mut bytes = [0u8; 8];
    texture
        .get_bytes(&mut bytes, 8, 0, Region::new_2d(0, 0, 1, 1), 0, 0)
        .unwrap();
    [0, 2, 4, 6].map(|i| u16::from_le_bytes([bytes[i], bytes[i + 1]]))
}

/// What `kernel` of `rrp_swizzled_storage` reads at texel `(0, 0)` of `texture`, or, for a write
/// kernel, writes `value` to its 2 x 2 texels.
fn storage(device: &Device, kernel: &str, texture: &Texture, value: [f32; 4]) -> [f32; 4] {
    let state = state(device, "rrp_swizzled_storage.ll", kernel);
    let out = buffer_with(device, &[0xab; 16]).unwrap();
    let value: Vec<u8> = value.iter().flat_map(|x| x.to_le_bytes()).collect();
    let value = buffer_with(device, &value).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&value), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(2, 2, 1), Size::new(2, 2, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let bytes = read_buffer(&out, 0, 16).unwrap();
    [0, 4, 8, 12].map(|i| f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]))
}

/// A texture whose alpha reads 1 keeps the alpha written to it: a function's write, through an
/// `access::write` and an `access::read_write` slot, and a render pass's cleared store; its
/// shaders' reads, those slots' included, read alpha 1.
#[test]
fn writes_keep_the_alpha_of_a_texture_whose_alpha_reads_one() {
    on_device_with(at_plan, |device, _| {
        let usage =
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
        let alpha = texture(device, usage, |d| d.alpha_reads_one = true);
        assert_eq!(first_texel(&alpha), texel(0, 0).map(half));
        for kernel in ["rrp_sw_write_f", "rrp_sw_rwwrite_f"] {
            let value = [0.25, 0.5, 0.75, 0.375];
            storage(device, kernel, &alpha, value);
            assert_eq!(first_texel(&alpha), value.map(half), "{kernel}");
            for read in ["rrp_sw_read_f", "rrp_sw_rwread_f"] {
                assert_eq!(
                    storage(device, read, &alpha, [0.0; 4]),
                    [0.25, 0.5, 0.75, 1.0],
                    "{kernel} then {read}"
                );
            }
        }
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(alpha.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.5, 0.25, 0.125, 0.625);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(first_texel(&alpha), [0.5, 0.25, 0.125, 0.625].map(half));
        assert_eq!(
            storage(device, "rrp_sw_read_f", &alpha, [0.0; 4]),
            [0.5, 0.25, 0.125, 1.0]
        );
    });
}
