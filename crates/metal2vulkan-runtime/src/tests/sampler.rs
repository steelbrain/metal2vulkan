//! Sampler states: the device's samplers shared by create info, and on devices the states.
//!
//! - The table makes one sampler per create info and shares it among its holders; past its cap a
//!   new create info is `Limit::DeviceSamplers`, and once a sampler's last holder gives it up the
//!   sampler is the reaper's and its room is free again.
//! - Under a device cfg, on every device of the cfg's kind: a state's ID is its index and the
//!   translator's sampler word. A nearest state clamped to the edge takes one device sampler for
//!   its plain and integer-texture uses, and an equal state, or one differing only in a border
//!   colour no address mode reads, shares it; a linear state with a compare function takes three;
//!   an anisotropic state is made. A cubic filter is `Gap::CubicSampler`. A state made at an index
//!   another holds takes it, and the earlier state's going leaves it; the descriptor-slot gauge
//!   counts the index until its owner is gone. The device's samplers come back to zero with the
//!   device. A state's constexpr samplers numbered up to one the translator refuses give back
//!   what the earlier ones took.
//! - Under a device cfg, a state bound with LOD clamps, on every device of the cfg's kind: clamps
//!   that read as its own bind the state, others its variant of them on the compute encoder and
//!   the render encoder's vertex and fragment functions, and a compute kernel samples with them as
//!   Metal does; a state keeps its 16 most recently bound variants, filled past that, and set 1's
//!   4,000 variant elements are filled past their capacity; the live variants hold flat across a
//!   loop that binds, retires and binds again, and come back to zero with the device.

use crate::error::{Error, Limit};
use crate::sampler::SamplerTable;
use ash::vk::{self, Handle};
use std::cell::Cell;

#[test]
fn a_device_sampler_is_shared_by_create_info_up_to_the_cap() {
    let mut table = SamplerTable::<u32>::new(2);
    let made = Cell::new(0_u64);
    let make = || {
        made.set(made.get() + 1);
        Ok(vk::Sampler::from_raw(made.get()))
    };
    let raw = |result: Result<vk::Sampler, Error>| match result {
        Ok(sampler) => sampler.as_raw(),
        Err(error) => panic!("{error}"),
    };
    assert_eq!(raw(table.take(&1, make)), 1);
    assert_eq!(raw(table.take(&1, make)), 1, "an equal create info shares");
    assert_eq!(raw(table.take(&2, make)), 2);
    match table.take(&3, make) {
        Err(Error::Limit(limit)) => assert_eq!(limit, Limit::DeviceSamplers { max: 2 }),
        other => panic!("past the cap: {:?}", other.map(|s| s.as_raw())),
    }
    assert_eq!(made.get(), 2, "nothing is made past the cap");
    assert_eq!(table.give(&1), None, "a holder is left");
    assert_eq!(table.give(&1).map(|s| s.as_raw()), Some(1));
    assert_eq!(raw(table.take(&3, make)), 3, "the room is free again");
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, within, Arena};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        CommandBuffer, CompareFunction, ComputeCommandEncoder, ComputePipelineDescriptor, Counters,
        Device, DeviceOptions, DispatchType, Error, FunctionDescriptor, Gap, HostBytes, IndexClass,
        KernelInputKind, Library, Limit, PixelFormat, Region, RenderPassDescriptor,
        SamplerBorderColor, SamplerDescriptor, SamplerMinMagFilter, SamplerMipFilter, SamplerState,
        StageSource, StorageMode, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan::metal::sampler::SamplerWord;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    #[test]
    fn every_device_shares_sampler_states_device_samplers() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        for (name, physical) in devices {
            let device = Device::new(&physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let _grant = device
                .grant_indices(IndexClass::Sampler, 1, 16)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = |descriptor: &SamplerDescriptor, index| -> SamplerState {
                device
                    .new_sampler_state(descriptor, index)
                    .unwrap_or_else(|error| panic!("{name}: {descriptor:?}: {error}"))
            };
            let defaults = counters.gauges().device_samplers;
            let nearest = SamplerDescriptor::new();
            let a = state(&nearest, 1);
            assert_eq!(a.gpu_resource_id().index, 1, "{name}");
            assert_eq!(
                a.gpu_resource_id().word,
                SamplerWord::of(&nearest).0,
                "{name}"
            );
            assert_eq!(counters.gauges().device_samplers, defaults + 1, "{name}");
            let b = state(&nearest, 2);
            let mut unread_border = nearest;
            unread_border.border_color = SamplerBorderColor::OpaqueWhite;
            let c = state(&unread_border, 3);
            assert_eq!(counters.gauges().device_samplers, defaults + 1, "{name}");
            let mut linear = nearest;
            linear.min_filter = SamplerMinMagFilter::Linear;
            linear.mag_filter = SamplerMinMagFilter::Linear;
            linear.compare_function = CompareFunction::Less;
            // Its plain and comparison samplers are new; its integer-texture use, nearest, reads
            // as the nearest state's plain sampler and shares it.
            let d = state(&linear, 4);
            assert_eq!(counters.gauges().device_samplers, defaults + 3, "{name}");
            let mut anisotropic = linear;
            anisotropic.compare_function = CompareFunction::Never;
            anisotropic.max_anisotropy = 8;
            let e = state(&anisotropic, 5);
            let mut cubic = nearest;
            cubic.min_filter = SamplerMinMagFilter::Cubic;
            assert!(
                matches!(
                    device.new_sampler_state(&cubic, 6),
                    Err(Error::Unsupported(Gap::CubicSampler))
                ),
                "{name}"
            );
            // A state at the index `a` holds takes it; `a`'s going leaves it.
            let later = state(&nearest, 1);
            assert_eq!(counters.gauges().samplers, 6, "{name}");
            assert_eq!(counters.gauges().descriptor_slots, 5, "{name}");
            drop(a);
            assert_eq!(counters.gauges().descriptor_slots, 5, "{name}");
            drop(later);
            assert_eq!(counters.gauges().descriptor_slots, 4, "{name}");
            state(&linear, 1);
            drop((b, c, d, e));
            assert_eq!(counters.gauges().descriptor_slots, 0, "{name}");
            drop(device);
            let gauges = counters.gauges();
            assert_eq!(
                (
                    gauges.samplers,
                    gauges.device_samplers,
                    gauges.descriptor_slots
                ),
                (0, 0, 0),
                "{name}"
            );
        }
    }

    /// The state the LOD-clamp tests bind: nearest filters and a nearest mip filter, clamped to
    /// the edge, its own clamps [`OWN`].
    fn clamped_state(name: &str, device: &Device) -> SamplerState {
        let mut d = SamplerDescriptor::new();
        d.mip_filter = SamplerMipFilter::Nearest;
        (d.lod_min_clamp, d.lod_max_clamp) = OWN;
        (device.new_sampler_state(&d, 1)).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// The clamp tests' state's own LOD clamps.
    const OWN: (f32, f32) = (1.0, 3.0);

    /// The `k`-th of the pairs the variant tests bind: a lower bound below 0, `-1 - k`, and an
    /// upper bound of 2. Each reads otherwise than [`OWN`], every one alike with the others, and
    /// each is a pair of its own: their variants share their device samplers.
    fn pair(k: usize) -> (f32, f32) {
        (-1.0 - k as f32, 2.0)
    }

    /// Binds `state` at slot 0 with [`pair`] `k`.
    fn bind(name: &str, encoder: &mut ComputeCommandEncoder, state: &SamplerState, k: usize) {
        let (lo, hi) = pair(k);
        (encoder.set_sampler_state_with_lod_clamps(Some(state), lo, hi, 0))
            .unwrap_or_else(|error| panic!("{name}: pair {k}: {error}"));
    }

    /// Commits `command_buffer` and waits for it, then for it to retire: until the live variants
    /// are `live`.
    fn retired(name: &str, counters: &Counters, command_buffer: CommandBuffer, live: u64) {
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        drop(command_buffer);
        within("the command buffer's retirement", || {
            counters.gauges().lod_clamp_variants == live
        });
    }

    /// A state bound with LOD clamps on the compute encoder and on the render encoder's vertex
    /// and fragment functions. Clamps that read as its own, `(0, 1000)` over its `(0, FLT_MAX)` as
    /// a replay stream binds it and `(-1, infinity)`, and a NaN bound bind the state itself;
    /// clamps that read otherwise, a raised lower bound or an upper bound below the last level a
    /// sample can read, bind its variant of them, one for both encoders, which the state keeps
    /// past its command buffer and which goes with the state. No state binds with any clamps.
    #[test]
    fn every_device_binds_a_state_with_lod_clamps() {
        each_device("lod_clamp_binds", |name, _, device| {
            let counters = device.counters();
            let variants = || counters.gauges().lod_clamp_variants;
            let _grant = device
                .grant_indices(IndexClass::Sampler, 1, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = device
                .new_sampler_state(&SamplerDescriptor::new(), 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let bound = |result: Result<(), Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut compute = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            bound(compute.set_sampler_state_with_lod_clamps(Some(&state), 0.0, 1000.0, 0));
            bound(compute.set_sampler_state_with_lod_clamps(Some(&state), -1.0, f32::INFINITY, 1));
            bound(compute.set_sampler_state_with_lod_clamps(Some(&state), f32::NAN, 4.0, 3));
            assert_eq!(variants(), 0, "{name}: clamps that read as the state's own");
            bound(compute.set_sampler_state_with_lod_clamps(Some(&state), 0.5, f32::MAX, 2));
            assert_eq!(variants(), 1, "{name}: a raised lower bound");
            bound(compute.set_sampler_state_with_lod_clamps(None, 3.0, 2.0, 2));
            compute.end_encoding();
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            d.usage = TextureUsage::RENDER_TARGET;
            let color = (device.new_texture_for_views(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut pass = RenderPassDescriptor::default();
            pass.color_attachments[0].texture = Some(color);
            let mut render = (command_buffer.render_command_encoder(&pass))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            bound(render.set_vertex_sampler_state_with_lod_clamps(Some(&state), 0.0, 1000.0, 0));
            bound(render.set_fragment_sampler_state_with_lod_clamps(Some(&state), 0.0, 1000.0, 0));
            bound(render.set_fragment_sampler_state_with_lod_clamps(
                Some(&state),
                0.0,
                f32::NAN,
                1,
            ));
            assert_eq!(variants(), 1, "{name}");
            bound(render.set_vertex_sampler_state_with_lod_clamps(Some(&state), 0.0, 14.0, 1));
            assert_eq!(variants(), 2, "{name}: an upper bound below the last level");
            bound(render.set_fragment_sampler_state_with_lod_clamps(
                Some(&state),
                0.5,
                f32::MAX,
                2,
            ));
            assert_eq!(variants(), 2, "{name}: the compute bind's variant");
            render.end_encoding();
            retired(name, &counters, command_buffer, 2);
            drop(state);
            within("the state's variants", || variants() == 0);
            drop(device);
            let gauges = counters.gauges();
            println!(
                "M2V-MATRIX gauge lod_clamp_binds/lod_clamp_variants@drop {}",
                gauges.lod_clamp_variants
            );
        });
    }

    /// The levels a state of [`OWN`] samples with the LOD clamps a compute bind gives, through the
    /// owned fixture `unv_runtime_levels`' `unv_rpos2d` at levels 0 to 7 of a 16 by 16 `R32Float`
    /// texture of five levels, every texel of level `l` holding `l + 1`, as Metal answers on the
    /// M3 Max (conformance case `sampler-bind-lod-clamps-kernel`): the pair replaces the state's
    /// clamps, a lower bound above the upper reads at the upper's level, a plain bind after a
    /// clamped one reads the state's own clamps, and a NaN bound binds the state as it is. A pair
    /// bound again binds its kept variant and writes no element of set 1.
    #[test]
    fn every_device_samples_with_the_lod_clamps_a_bind_gives() {
        const SIDE: u32 = 16;
        const LEVELS: u32 = 5;
        const LANES: usize = 8;
        let library = match llvm_assemble(&engine_dir().join("unv_runtime_levels.ll")) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("unv_runtime_levels.ll: {error:?}"),
        };
        // Each row's binds in order, `None` a plain one, and the clamps it samples with.
        type Row = (&'static [Option<(f32, f32)>], (f32, f32));
        let rows: [Row; 8] = [
            (&[None], OWN),
            (&[Some(OWN)], OWN),
            (&[Some((0.0, 4.0))], (0.0, 4.0)),
            (&[Some((3.0, 1.0))], (1.0, 1.0)),
            (&[Some((6.0, 9.0))], (6.0, 9.0)),
            (&[Some((0.0, 4.0)), None], OWN),
            (&[Some((0.0, 4.0))], (0.0, 4.0)),
            (&[Some((f32::NAN, 4.0))], OWN),
        ];
        const AGAIN: usize = 6;
        each_device("lod_clamp_samples", |name, _, device| {
            let counters = device.counters();
            let ok = |result: Result<(), Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let _grants = [IndexClass::Texture, IndexClass::Sampler].map(|class| {
                (device.grant_indices(class, 1, 1))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let mut td =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::R32Float, SIDE, SIDE);
            td.mipmap_level_count = LEVELS;
            td.usage = TextureUsage::SHADER_READ;
            td.storage_mode = StorageMode::Shared;
            let texture = (device.new_texture(&td, Some(1)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for level in 0..LEVELS {
                let side = SIDE >> level;
                let texel = (level + 1) as f32;
                let bytes: Vec<u8> = (0..side * side).flat_map(|_| texel.to_le_bytes()).collect();
                let whole = Region {
                    origin: [0; 3],
                    size: [side, side, 1],
                };
                let row = u64::from(side) * 4;
                (texture.replace_region(
                    whole,
                    level,
                    0,
                    HostBytes::from(&bytes[..]),
                    row,
                    row * u64::from(side),
                ))
                .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
            }
            let state = clamped_state(name, &device);
            let function = library.function("unv_rpos2d").expect("the kernel");
            let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(function),
            ));
            let pipeline = (device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: unv_rpos2d: {error}"));
            let mut arena = Arena::new(&device);
            let points = arena.buffer();
            let out = arena.buffer();
            // Lane `l` samples at `(0.3, 0.6)` at level `l`.
            let at: Vec<u8> = (0..LANES)
                .flat_map(|lane| [0.3_f32, 0.6, lane as f32, 0.0])
                .flat_map(f32::to_le_bytes)
                .collect();
            let contents = points.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds 256 bytes, more than the 128 written, and no work is
            // pending.
            unsafe { (at.as_ptr()).copy_to_nonoverlapping(contents.as_ptr(), at.len()) };
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(encoder.set_compute_pipeline_state(&pipeline));
            ok(encoder.set_texture(Some(&texture), 0));
            ok(encoder.set_buffer(Some(&points), 0, 0));
            let mut again = None;
            for (row, (binds, _)) in rows.iter().enumerate() {
                let before = counters.metrics().sampler_element_writes;
                for bind in *binds {
                    ok(match bind {
                        None => encoder.set_sampler_state(Some(&state), 0),
                        Some((lo, hi)) => {
                            encoder.set_sampler_state_with_lod_clamps(Some(&state), *lo, *hi, 0)
                        }
                    });
                }
                if row == AGAIN {
                    again = Some(counters.metrics().sampler_element_writes - before);
                }
                ok(encoder.set_buffer(Some(&out), (row * 4 * LANES) as u64, 1));
                (encoder.dispatch_threads([LANES as u32, 1, 1], [LANES as u32, 1, 1]))
                    .unwrap_or_else(|error| panic!("{name}: row {row}: {error}"));
            }
            assert_eq!(
                again,
                Some(0),
                "{name}: (0, 4) bound again binds its kept variant"
            );
            assert_eq!(
                counters.gauges().lod_clamp_variants,
                3,
                "{name}: the variants of (0, 4), (1, 1) and (6, 9)"
            );
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the dispatches that wrote the buffer finished, and their writes reach the
            // host; the buffer holds 256 bytes, the eight rows of eight words.
            let words = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), 64) };
            for (row, (binds, (lo, hi))) in rows.iter().enumerate() {
                let last = (LEVELS - 1) as f32;
                let metal: Vec<f32> = (0..LANES)
                    .map(|lane| (lane as f32).max(*lo).min(*hi).clamp(0.0, last) + 1.0)
                    .collect();
                assert_eq!(
                    &words[row * LANES..][..LANES],
                    &metal[..],
                    "{name}: row {row}, binds {binds:?}"
                );
            }
            println!(
                "lod_clamp_samples: {name}: {} rows read Metal's levels through 3 variants",
                rows.len()
            );
        });
    }

    /// A state's variants past the 16 it keeps, filled past that capacity. One command buffer
    /// binds 17 pairs: it holds all 16 kept variants, so the 17th is made beside them, held by
    /// the command buffer alone, and goes as it retires. A later one binds a kept pair, which
    /// writes no element of set 1, and new pairs, each of which takes the place of the least
    /// recently bound variant no command buffer holds: a pair given up so is made again. The
    /// state's variants go with it.
    #[test]
    fn every_device_keeps_the_16_variants_a_state_bound_last() {
        each_device("lod_clamp_variants_kept", |name, _, device| {
            let counters = device.counters();
            let variants = || counters.gauges().lod_clamp_variants;
            let writes = || counters.metrics().sampler_element_writes;
            let _grant = (device.grant_indices(IndexClass::Sampler, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = clamped_state(name, &device);
            let queue = device.new_command_queue(4).expect("a queue");
            let first = queue.command_buffer().expect("a command buffer");
            let mut encoder = (first.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for k in 0..16 {
                bind(name, &mut encoder, &state, k);
            }
            assert_eq!(variants(), 16, "{name}");
            bind(name, &mut encoder, &state, 16);
            assert_eq!(
                variants(),
                17,
                "{name}: work holds all 16, so the 17th is made beside"
            );
            encoder.end_encoding();
            retired(name, &counters, first, 16);
            let second = queue.command_buffer().expect("a command buffer");
            let mut encoder = (second.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let kept = |k: usize, encoder: &mut ComputeCommandEncoder| {
                let before = writes();
                bind(name, encoder, &state, k);
                assert_eq!(writes(), before, "{name}: pair {k} binds its kept variant");
            };
            let made = |k: usize, encoder: &mut ComputeCommandEncoder| {
                let before = writes();
                bind(name, encoder, &state, k);
                assert!(writes() > before, "{name}: pair {k}'s variant is made");
                assert_eq!(
                    variants(),
                    16,
                    "{name}: pair {k} takes an idle variant's place"
                );
            };
            // Kept: 0 to 15, 0 the least recently bound. Pair 0 bound again is the most recent,
            // so pair 17 takes pair 1's place, pair 1 bound again pair 2's and pair 2 pair 3's.
            kept(0, &mut encoder);
            made(17, &mut encoder);
            kept(0, &mut encoder);
            made(1, &mut encoder);
            kept(17, &mut encoder);
            made(2, &mut encoder);
            kept(4, &mut encoder);
            encoder.end_encoding();
            retired(name, &counters, second, 16);
            drop(state);
            within("the state's variants", || variants() == 0);
            drop(device);
            println!(
                "M2V-MATRIX gauge lod_clamp_variants_kept/lod_clamp_variants@drop {}",
                counters.gauges().lod_clamp_variants
            );
        });
    }

    /// Set 1's 4,000 variant elements filled past their capacity: one command buffer binds 4,000
    /// pairs of one state, holding each variant, and the 4,001st is
    /// `Limit::LodClampVariants { max: 4000 }`, binding nothing and holding no device sampler;
    /// the variants share their device samplers. Once the command buffer retires the state keeps
    /// 16, and a new pair binds again in a freed element.
    #[test]
    fn every_device_fills_the_variant_elements_past_their_capacity() {
        each_device("lod_clamp_variants_full", |name, _, device| {
            let counters = device.counters();
            let variants = || counters.gauges().lod_clamp_variants;
            let _grant = (device.grant_indices(IndexClass::Sampler, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = clamped_state(name, &device);
            let samplers = counters.gauges().device_samplers;
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for k in 0..4000 {
                bind(name, &mut encoder, &state, k);
            }
            assert_eq!(variants(), 4000, "{name}");
            let full = counters.gauges().device_samplers;
            assert!(
                full - samplers <= 2,
                "{name}: the variants' device samplers, one per use at most: {}",
                full - samplers
            );
            let (lo, hi) = pair(4000);
            match encoder.set_sampler_state_with_lod_clamps(Some(&state), lo, hi, 0) {
                Err(Error::Limit(limit)) => {
                    assert_eq!(limit, Limit::LodClampVariants { max: 4000 }, "{name}")
                }
                other => panic!("{name}: the 4,001st variant: {other:?}"),
            }
            assert_eq!(variants(), 4000, "{name}");
            assert_eq!(counters.gauges().device_samplers, full, "{name}");
            encoder.end_encoding();
            retired(name, &counters, command_buffer, 16);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            bind(name, &mut encoder, &state, 4000);
            assert_eq!(
                variants(),
                16,
                "{name}: the new pair takes an idle variant's place"
            );
            encoder.end_encoding();
            retired(name, &counters, command_buffer, 16);
            drop(state);
            within("the state's variants", || variants() == 0);
            drop(device);
            println!(
                "M2V-MATRIX gauge lod_clamp_variants_full/lod_clamp_variants@drop {}",
                counters.gauges().lod_clamp_variants
            );
        });
    }

    /// A loop that binds, retires and binds again: 200 iterations, each one command buffer that
    /// binds four of 32 pairs of one state, each iteration one pair further, so every iteration
    /// binds a pair the state gave up and takes an idle variant's place. The live variants, the
    /// device samplers and the descriptor slots read the same after iteration 100 and after
    /// iteration 200, 16 variants, and zero once the device is gone.
    #[test]
    fn every_device_s_variants_hold_flat_across_a_bind_and_retire_loop() {
        each_device("lod_clamp_variants_loop", |name, _, device| {
            let counters = device.counters();
            let _grant = (device.grant_indices(IndexClass::Sampler, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = clamped_state(name, &device);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut read = Vec::new();
            for iteration in 1..=200 {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                for k in 0..4 {
                    bind(name, &mut encoder, &state, (iteration + k) % 32);
                }
                encoder.end_encoding();
                // Pairs 1 to `iteration + 3` bound so far, the state keeping 16 of them.
                retired(
                    name,
                    &counters,
                    command_buffer,
                    (iteration as u64 + 3).min(16),
                );
                if iteration % 100 == 0 {
                    let gauges = counters.gauges();
                    read.push((
                        gauges.lod_clamp_variants,
                        gauges.device_samplers,
                        gauges.descriptor_slots,
                    ));
                }
            }
            println!(
                "lod_clamp_variants_loop: {name}: variants, device samplers and slots after \
                 iterations 100 and 200: {read:?}"
            );
            assert_eq!(read[0], read[1], "{name}");
            assert_eq!(read[0].0, 16, "{name}");
            drop(state);
            drop(device);
            println!(
                "M2V-MATRIX gauge lod_clamp_variants_loop/lod_clamp_variants@drop {}",
                counters.gauges().lod_clamp_variants
            );
        });
    }

    /// A state's constexpr samplers, numbered up to one whose device sampler the translator does
    /// not state: the refusal gives back what the earlier one took, its arrays' elements at once
    /// and its device sampler through the reaper, and leaves its index free.
    #[test]
    fn a_refused_constexpr_sampler_gives_back_the_ones_numbered_before_it() {
        use metal2vulkan::pipeline::{CombinedSampler, NumberedSampler, SamplerRef};
        // Linear on both filters, clamped to the edge on every axis, levels 0 to 65504: no
        // default sampler shares its create info.
        let linear = [
            1 | (1 << 3) | (1 << 6) | (1 << 9) | (1 << 11) | (0x7bff << 40),
            0,
        ];
        // The same state with a NaN level bias, of which the translator makes no device sampler.
        let nan_bias = [linear[0], 0x7e00];
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        for (name, physical) in devices {
            let device = Device::new(&physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let _grant = device
                .grant_indices(IndexClass::Sampler, 1, 2)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let before = counters.gauges();
            let samplers = vec![
                NumberedSampler::Constexpr(linear),
                NumberedSampler::Constexpr(nan_bias),
            ];
            let uses = [linear, nan_bias]
                .map(|words| CombinedSampler::new(SamplerRef::Constexpr(words), false, false));
            let numbered = crate::pipeline::number_samplers(
                device.core(),
                samplers,
                &[(0, uses[0]), (1, uses[1])],
                |_| Ok(vec![1, 2]),
                || {
                    Error::Unsupported(Gap::KernelInput {
                        kind: KernelInputKind::ConstexprSampler,
                    })
                },
            );
            assert!(
                matches!(numbered, Err(Error::Translation { .. })),
                "{name}: the NaN bias is the translator's refusal"
            );
            assert_eq!(
                counters.gauges().descriptor_slots,
                before.descriptor_slots,
                "{name}: the linear sampler's elements are given back"
            );
            crate::tests::command::on_device::within("the reaper frees the device sampler", || {
                counters.gauges().device_samplers == before.device_samplers
            });
        }
    }
}
